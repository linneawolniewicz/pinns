# Imports
import numpy as np
import scipy.io
import tensorflow as tf
from pyDOE import lhs
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
tf.config.list_physical_devices(device_type=None)

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
        
        beta = weight on boundary_loss
        
    Outputs: sum_loss, pinn_loss, boundary_loss
    '''
    def train_step(self, p, r, p_boundary, r_boundary, f_boundary, alpha=1, beta=1):
        with tf.GradientTape(persistent=True) as t2: 
            with tf.GradientTape(persistent=True) as t1: 
                # Forward pass P (PINN data)
                P = tf.concat((p, r), axis=1)
                f = self.tf_call(P)

                # Forward pass P_boundary (boundary condition data)
                P_boundary = tf.concat((p_boundary, r_boundary), axis=1)
                f_pred_boundary = self.tf_call(P_boundary)

                # Calculate boundary loss
                boundary_loss = tf.math.reduce_mean(tf.math.abs(f_pred_boundary - f_boundary))

            # Calculate first-order PINN gradients
            f_p = t1.gradient(f, p)
            f_r = t1.gradient(f, r)

            # Calculate resulting loss = PINN loss + boundary loss
            pinn_loss = self.pinn_loss(p, r, f_p, f_r)
            sum_loss = alpha*pinn_loss + beta*boundary_loss
        
        # Backpropagation
        gradients = t2.gradient(sum_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Return losses
        return sum_loss.numpy(), pinn_loss.numpy(), boundary_loss.numpy()
    
    '''
    Description: The fit function used to iterate through epoch * steps_per_epoch steps of train_step. 
    
    Inputs: 
        P_predict: (N, 2) array: Input data for entire spatial and temporal domain. Used for vizualization for
        predictions at the end of each epoch. Michael created a very pretty video file with it. 
        
        size: size of the prediction data (i.e. len(p) and len(r))
        
        alpha = weight on pinn_loss
        
        beta = weight on boundary_loss
        
        batchsize: batchsize for (p, r) in train step
        
        boundary_batchsize: batchsize for (x_lower, t_boundary) and (x_upper, t_boundary) in train step
        
        epochs: epochs
        
        save: Whether or not to save the model to a checkpoint every 10 epochs
        
        load_epoch: If -1, a saved model will not be loaded. Otherwise, the model will be loaded from the provided epoch
    
    Outputs: Losses for each equation (Total, PDE, Boundary Value), and predictions for each epoch.
    '''
    def fit(self, P_predict, size, alpha=1, beta=1, batchsize=64, boundary_batchsize=16, epochs=20, save=True, load_epoch=-1):
        # If load == True, load the weights
        if load_epoch != -1:
            self.load_weights('./ckpts/pinn_epoch_' + str(load_epoch))
        
        # Initialize losses as zeros
        steps_per_epoch = np.ceil(self.n_samples / batchsize).astype(int)
        total_pinn_loss = np.zeros((epochs, ))
        total_boundary_loss = np.zeros((epochs, ))
        total_loss = np.zeros((epochs, ))
        total_predictions = np.zeros((size**2, 1, epochs))
        
        # For each epoch, sample new values in the PINN and boundary areas and pass them to train_step
        for epoch in range(epochs):
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
                losses = self.train_step(p, r, p_boundary, r_boundary, f_boundary, alpha, beta)
                sum_loss[step] = losses[0]
                pinn_loss[step] = losses[1]
                boundary_loss[step] = losses[2]
            
            # Calculate and print total losses for the epoch
            total_loss[epoch] = np.sum(sum_loss)
            total_pinn_loss[epoch] = np.sum(pinn_loss)
            total_boundary_loss[epoch] = np.sum(boundary_loss)
            print(f'Training loss for epoch {epoch}: pinn: {total_pinn_loss[epoch]:.4f}, boundary: {total_boundary_loss[epoch]:.4f}, total: {total_loss[epoch]:.4f}')
            
            # Get prediction variable loss by the predict function (below)
            total_predictions[:, :, epoch] = self.predict_epoch(P_predict, size)
            
            # If the epoch is a multiple of 10, save to a checkpoint
            if (epoch%10 == 0) & (save == True):
                self.save_weights('./ckpts/pinn_epoch_' + str(epoch), overwrite=True, save_format=None, options=None)
        
        # Return epoch losses
        return total_loss, total_pinn_loss, total_boundary_loss, total_predictions
    
    # Predict for some P's the value of the neural network f(r, p)
    def predict_epoch(self, P, size, batchsize=2048):
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
    
    def predict(self, p, r):
        P = tf.concat((p, r), axis=1)
        f = self.tf_call(P)
        
        return f
    
    def evaluate(self, ): 
        pass
    
    # pinn_loss calculates the PINN loss by calculating the MAE of the pinn function
    @tf.function
    def pinn_loss(self, p, r, f_p, f_r): 
        # Note: p and r are taken out of logspace for the PINN calculation
        p = tf.math.exp(p)
        r = tf.math.exp(r)
        R = p # GeV * km/s / coulomb Note: R = p * c/q but skipping c/q here 
        V = 400 # 400 km/s
        M = 0.938 # GeV
        k_0 = 1 # km^2/s
        
        # Calculate k and l_f
        k = tf.math.divide(p, tf.math.sqrt(tf.math.square(p) + tf.math.square(M))) * k_0 * r * p
        l_f = tf.math.reduce_mean(tf.math.abs(f_r + (tf.math.divide(R * V, 3 * k) * f_p)))
        
        return l_f
    
    # tf_call passes inputs through the neural network
    @tf.function
    def tf_call(self, inputs): 
        return self.call(inputs, training=True)
        
###########################################################################################

def main():
    # Constants  
    M = 0.938 # GeV
    gamma = -2.5 # Between -2 and -3
    size = 256 # size of r, T, p, and f_boundary

    # Create intial r, p, and T predict data
    T = np.linspace(0.001, 1000, size).flatten()[:,None]
    p = np.sqrt(T**2 + 2*T*M).flatten()[:,None] # p values
    r = np.linspace(1, 120, size).flatten()[:,None] # r values

    # Create boundary f data (f at r_HP) for boundary loss
    f_boundary = ((T + M)**gamma)/(p**2)

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

    # Define neural network. Note: 2 inputs- (p, r), 1 output- f(r, p)
    inputs = tf.keras.Input((2))
    x_ = tf.keras.layers.Dense(100, activation='tanh')(inputs)
    x_ = tf.keras.layers.Dense(500, activation='tanh')(x_)
    x_ = tf.keras.layers.Dense(500, activation='tanh')(x_)
    x_ = tf.keras.layers.Dense(100, activation='tanh')(x_)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x_) 

    # Define hyperparameters
    alpha = 1 # pinn_loss weight
    beta = 15 # boundary_loss weight
    lr = 3e-4
    batchsize = 1032
    boundary_batchsize = 256
    epochs = 600
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr)

    # Initialize, compile, and fit the PINN
    pinn = PINN(inputs=inputs, outputs=outputs, lower_bound=lb, upper_bound=ub, p=p[:, 0], r=r[:, 0], 
                f_boundary=f_boundary[:, 0], size=size)
    pinn.compile(optimizer=optimizer)
    total_loss, pinn_loss, boundary_loss, predictions = pinn.fit(P_predict=P_star, alpha=alpha, beta=beta, batchsize=batchsize, 
                                                     boundary_batchsize=boundary_batchsize, epochs=epochs, size=size)

    # Predict f at the boundary r_HP
    r_boundary = np.zeros((p.shape[0], 1))
    r_boundary[:] = ub[1]
    f_pred_boundary = pinn.predict(p, r_boundary).numpy()

    # Predict f at the final epoch for all (p, r)
    f_predict = pinn.tf_call(P_star).numpy()

    # Save PINN outputs
    with open('./figures/total_loss.pkl', 'wb') as file:
        pkl.dump(total_loss, file)

    with open('./figures/pinn_loss.pkl', 'wb') as file:
        pkl.dump(pinn_loss, file)

    with open('./figures/boundary_loss.pkl', 'wb') as file:
        pkl.dump(boundary_loss, file)

    with open('./figures/predictions.pkl', 'wb') as file:
        pkl.dump(predictions, file)

    with open('./figures/f_boundary.pkl', 'wb') as file:
        pkl.dump(f_boundary, file)

    with open('./figures/p.pkl', 'wb') as file:
        pkl.dump(p, file)

    with open('./figures/f_pred_boundary.pkl', 'wb') as file:
        pkl.dump(f_pred_boundary, file)

    with open('./figures/f_predict.pkl', 'wb') as file:
        pkl.dump(f_predict, file)

if __name__=="__main__":
    main()
    