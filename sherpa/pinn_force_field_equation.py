# Imports
import numpy as np
import tensorflow as tf
import sherpa
import pickle as pkl
tf.config.list_physical_devices(device_type=None)

###################################################################################
'''
Description: Defines the class for a PINN model implementing train_step, fit, and predict functions. Note, it is necessary 
to design each PINN seperately for each system of PDEs since the train_step is customized for a specific system. 
This PINN in particular solves the force-field equation for solar modulation of cosmic rays. Once trained, the PINN can predict the solution space given 
domain bounds and the input space. 
'''
class PINN(tf.keras.Model):
    def __init__(self, inputs, outputs, lower_bound, upper_bound, p, r, f_boundary, size, n_samples=20000, n_boundary=50):
        super(PINN, self).__init__(inputs=inputs, outputs=outputs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p = p
        self.r = r
        self.f_boundary = f_boundary
        self.n_samples = n_samples
        self.n_boundary = n_boundary
        self.size = size
        
    '''
    Description: A system of PDEs are determined by 2 types of equations: the main partial differential equations 
    and the boundary value equations. These two equations will serve as loss functions which 
    we train the PINN to satisfy. If a PINN can satisfy BOTH equations, the system is solved. Since there are 2 types of 
    equations (PDE, Boundary Value), we will need 2 types of inputs. Each input is composed of a spatial 
    variable 'r' and a momentum variable 'p'. The different types of (p, r) pairs are described below.
    
    Inputs: 
        p, r: (batchsize, 1) shaped arrays : These inputs are used to derive the main partial differential equation loss.
        Train step first feeds (p, r) through the PINN for the forward propagation. This expression is PINN(p, r) = f. 
        Next, the partials f_p and f_r are obtained. We utilize TF2s GradientTape data structure to obtain all partials. 
        Once we obtain these partials, we can compute the main PDE loss and optimize weights w.r.t. to the loss. 
        
        p_boundary, r_boundary : (boundary_batchsize, 1) shaped arrays : These inputs are used to derive the boundary value
        equations. The boundary value loss relies on target data (**not an equation**), so we can just measure the MAE of 
        PINN(p_boundary, r_boundary) = f_pred_boundary and boundary_f.
        
        f_boundary: (boundary_batchsize, 1) shaped arrays : This is the target data for the boundary value inputs
        
        alpha = weight on pinn_loss
        
    Outputs: sum_loss, pinn_loss, boundary_loss
    '''
    def train_step(self, p, r, p_boundary, r_boundary, f_boundary, loss, alpha):
        with tf.GradientTape(persistent=True) as t2: 
            with tf.GradientTape(persistent=True) as t1: 
                # Forward pass P (PINN data)
                P = tf.concat((p, r), axis=1)
                f = self.tf_call(P)

                # Forward pass P_boundary (boundary condition data)
                P_boundary = tf.concat((p_boundary, r_boundary), axis=1)
                f_pred_boundary = self.tf_call(P_boundary)

                # Calculate boundary loss
                if loss=='mse': boundary_loss = tf.math.reduce_mean(tf.math.square(f_pred_boundary - f_boundary))
                elif loss=='mae': boundary_loss = tf.math.reduce_mean(tf.math.abs(f_pred_boundary - f_boundary))

            # Calculate first-order PINN gradients
            f_p = t1.gradient(f, p)
            f_r = t1.gradient(f, r)
            
            pinn_loss = self.pinn_loss(p, r, f_p, f_r, loss)
            total_loss = alpha*pinn_loss + (1-alpha)*boundary_loss
        
        # Backpropagation
        gradients = t2.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Return losses
        return pinn_loss.numpy(), boundary_loss.numpy()
    
    '''
    Description: The fit function used to iterate through epoch * steps_per_epoch steps of train_step. 
    
    Inputs: 
        client, trial: Sherpa client and trial
        
        P_predict: (N, 2) array: Input data for entire spatial and temporal domain. Used for vizualization for
        predictions at the end of each epoch. Michael created a very pretty video file with it. 
                
        loss: what type of loss function to use, MSE or MAE
        
        alpha: weight on pinn_loss
        
        batchsize: batchsize for (p, r) in train step
        
        boundary_batchsize: batchsize for (x_lower, t_boundary) and (x_upper, t_boundary) in train step
        
        epochs: epochs
        
        lr: learning rate
        
        size: size of the prediction data (i.e. len(p) and len(r))
        
        save: Whether or not to save the model to a checkpoint every 10 epochs
        
        load_epoch: If -1, a saved model will not be loaded. Otherwise, the model will be 
        loaded from the provided epoch
        
        lr_decay: If -1, learning rate will not be decayed. Otherwise, lr = lr_decay*lr if loss doesn't
        decrease for 3 epochs
        
        weight_change: If -1, alpha will not be changed. Otherwise, alpha = weight_change*alpha if loss 
        doesn't decrease for 3 epochs
        
        patience: Number of epochs to check whether loss has decreased before updating lr or alpha
        
        filename: Name for the checkpoint file
    
    Outputs: Losses for each equation (Total, PDE, Boundary Value), and predictions for each epoch.
    '''
    def fit(self, client, trial, P_predict, loss='mse', alpha=1, batchsize=64, boundary_batchsize=16, epochs=20, lr=3e-3, size=256, 
            save=False, load_epoch=-1, lr_decay=-1, weight_change=-1, patience=3, filename=''):
        # If load == True, load the weights
        if load_epoch != -1:
            name = './ckpts/pinn_' + filename + '_trial_' + str(trial.id) + '_epoch_' + str(load_epoch)
            self.load_weights(name)
        
        # Initialize losses as zeros
        steps_per_epoch = np.ceil(self.n_samples / batchsize).astype(int)
        total_pinn_loss = np.zeros((epochs, ))
        total_boundary_loss = np.zeros((epochs, ))
        predictions = np.zeros((size**2, 1, epochs))
        
        # For each epoch, sample new values in the PINN and boundary areas and pass them to train_step
        for epoch in range(epochs):
            # Compile
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            self.compile(optimizer=opt)

            # Reset loss variables
            sum_loss = np.zeros((steps_per_epoch,))
            pinn_loss = np.zeros((steps_per_epoch,))
            boundary_loss = np.zeros((steps_per_epoch,))
            
            # For each step, get PINN and boundary variables and pass them to train_step
            for step in range(steps_per_epoch):
                # Get PINN p and r variables via uniform distribution sampling between lower and upper bounds
                p = tf.Variable(tf.random.uniform((batchsize, 1), minval=self.lower_bound[0], maxval=self.upper_bound[0]))
                r = tf.Variable(tf.random.uniform((batchsize, 1), minval=self.lower_bound[1], maxval=self.upper_bound[1]))
                
                # Randomly sample boundary_batchsize from p_boundary and f_boundary
                p_idx = np.expand_dims(np.random.choice(self.f_boundary.shape[0], boundary_batchsize, replace=False), axis=1)
                p_boundary = self.p[p_idx]
                f_boundary = self.f_boundary[p_idx]
                
                # Create r_boundary array = r_HP
                upper_bound = np.zeros((boundary_batchsize, 1))
                upper_bound[:] = self.upper_bound[1]
                r_boundary = tf.Variable(upper_bound, dtype=tf.float32)
                
                # Pass variables through the model via train_step and get losses
                losses = self.train_step(p, r, p_boundary, r_boundary, f_boundary, loss, alpha)
                pinn_loss[step] = losses[0]
                boundary_loss[step] = losses[1]
            
            # Calculate and print total losses for the epoch
            total_pinn_loss[epoch] = np.sum(pinn_loss)
            total_boundary_loss[epoch] = np.sum(boundary_loss)
            # print(f'Training loss for epoch {epoch}: pinn: {total_pinn_loss[epoch]:.4f}, boundary: {total_boundary_loss[epoch]:.4f}, total: {(total_boundary_loss[epoch]+total_pinn_loss[epoch]):.4f}')
            
            # Predict
            predictions[:, :, epoch] = self.predict(P_predict, size)
            
            # Determine if loss has decreased for the past 2 or 5 epochs
            if (epoch > patience):
                hasDecreased = False
                if (total_pinn_loss[epoch] + total_boundary_loss[epoch]) < (total_pinn_loss[epoch-patience] + total_boundary_loss[epoch-patience]):
                    hasDecreased = True
                        
                # If loss hasn't decreased for the past 2 epochs, decrease lr by lr_decay
                if (lr_decay != -1) & (not hasDecreased):
                    lr = lr_decay*lr

                # If pinn loss hasn't decreased for the past 2 epochs, increase alpha by weight_change
                if (weight_change != -1) & (not hasDecreased):
                    alpha = np.tanh(weight_change*alpha)
                    # print(f'New learning rate {lr} and alpha {alpha}') 
            
            # If the epoch is a multiple of 10, save to a checkpoint
            if (epoch%10 == 0) & (save == True):
                name = './ckpts/pinn_' + filename + '_trial_' + str(trial.id) + '_epoch_' + str(epoch)
                self.save_weights(name, overwrite=True, save_format=None, options=None)
        
            # Send metrics
            if client:
                obj = total_pinn_loss[epoch] + total_boundary_loss[epoch]
                sherpa = client.send_metrics(
                         trial,
                         iteration=epoch,
                         objective=obj
                )
        
        # Return epoch losses
        return total_pinn_loss, total_boundary_loss, predictions
    
    # Predict for some P's the value of the neural network f(r, p)
    def predict(self, P, size, batchsize=2048):
        steps_per_epoch = np.ceil(P.shape[0] / batchsize).astype(int)
        preds = np.zeros((size**2, 1))
        
        # For each step calculate start and end index values for prediction data
        for step in range(steps_per_epoch):
            start_idx = step * 64
            
            # If last step of the epoch, end_idx is shape-1. Else, end_idx is start_idx + 64 
            if step == steps_per_epoch - 1:
                end_idx = P.shape[0] - 1
            else:
                end_idx = start_idx + 64
                
            # Pass prediction data through the model
            preds[start_idx:end_idx, :] = self.tf_call(P[start_idx:end_idx, :]).numpy()
        
        # Return f
        return preds
    
    def evaluate(self, ): 
        pass
    
    # pinn_loss calculates the PINN loss by calculating the MAE of the pinn function
    @tf.function
    def pinn_loss(self, p, r, f_p, f_r, loss): # To-do: add loss pass-through!
        # Note: p and r are taken out of logspace for the PINN calculation
        p = tf.math.exp(p) # GeV/c
        r = tf.math.exp(r) # km
        V = 400 # 400 km/s
        m = 0.938 # GeV/c^2
        k_0 = 1e11 # km^2/s
        k_1 = k_0 * tf.math.divide(r, 150e6) # km^2/s
        k_2 = p # unitless, k_2 = p/p0 and p0 = 1 GeV/c
        R = p # GV
        beta = tf.math.divide(p, tf.math.sqrt(tf.math.square(p) + tf.math.square(m))) 
        k = beta*k_1*k_2
        
        # Calculate physics loss
        if loss=='mae': l_f = tf.math.reduce_mean(tf.math.abs(f_r + (tf.math.divide(R*V, 3*k) * f_p)))
        elif loss=='mse': l_f = tf.math.reduce_mean(tf.math.square(f_r + (tf.math.divide(R*V, 3*k) * f_p)))
        
        return l_f
    
    # tf_call passes inputs through the neural network
    @tf.function
    def tf_call(self, inputs): 
        return self.call(inputs, training=True)

###########################################################################################

def main(client, trial):
    # Constants  
    m = 0.938 # GeV/c^2
    gamma = -3 # Between -2 and -3
    size = 512 # size of r, T, p, and f_boundary
    r_lims = [119, 120]
    T_lims = [0.001, 1000]

    # Create intial r, p, and T predict data
    T = np.logspace(np.log10(T_lims[0]), np.log10(T_lims[1]), size).flatten()[:,None] # GeV
    p = (np.sqrt((T+m)**2-m**2)).flatten()[:,None] # GeV/c
    r = np.logspace(np.log10(r_lims[0]*150e6), np.log10(r_lims[1]*150e6), size).flatten()[:,None] # km

    # Create boundary f data (f at r_HP) for boundary loss
    f_boundary = ((T + m)**gamma)/(p**2) # particles/(m^3 (GeV/c)^3)

    # Take the log of all input data
    r = np.log(r)
    T = np.log(T)
    p = np.log(p)
    f_boundary = np.log(f_boundary)

    # Domain bounds
    lb = np.array([p[0], r[0]]) # (p, r) in (GeV, AU)
    ub = np.array([p[-1], r[-1]]) # (p, r) in (GeV, AU)

    # Flatten and transpose data for ML
    P, R = np.meshgrid(p, r)
    P_star = np.hstack((P.flatten()[:,None], R.flatten()[:,None]))
    
    # Sherpa
    hps = trial.parameters

    # Hyperparameters
    epochs = 5
    save = False
    load_epoch = -1
    lr = 3e-2
    filename = ''
    alpha = hps.get('alpha')
    lr_decay = hps.get('lr_decay')
    patience = hps.get('patience')
    batchsize = hps.get('batchsize')
    boundary_batchsize = hps.get('boundary_batchsize')
    weight_change = hps.get('weight_change')
    num_layers = hps.get('num_layers')
    num_hidden_units = hps.get('num_hidden_units')
    activation = hps.get('activation')

    # Create model
    inputs = tf.keras.Input((2))
    x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(inputs)
    for i in range(num_layers):
        x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(x_)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x_) 

    # Train
    pinn = PINN(inputs=inputs, outputs=outputs, lower_bound=lb, upper_bound=ub, p=p[:, 0], r=r[:, 0], 
                f_boundary=f_boundary[:, 0], size=size)
    pinn_loss, boundary_loss, predictions = pinn.fit(client, trial, P_predict=P_star, alpha=alpha, batchsize=batchsize, boundary_batchsize=boundary_batchsize,
                                                     epochs=epochs, lr=lr, size=size, save=save, load_epoch=load_epoch, lr_decay=lr_decay,
                                                     weight_change=weight_change, patience=patience, filename=filename)


if __name__ == '__main__':
    client = sherpa.Client()
    trial = client.get_trial()
    main(client, trial)
    