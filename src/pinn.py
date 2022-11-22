# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sherpa
import pickle as pkl

tfd = tfp.distributions
tfm = tf.math
tf.config.list_physical_devices(device_type=None)

###################################################################################
'''
Description: Defines the class for a PINN model implementing train_step, fit, and predict functions. Note, it is necessary 
to design each PINN seperately for each system of PDEs since the train_step is customized for a specific system. 
This PINN in particular solves the force-field equation for solar modulation of cosmic rays. Once trained, the PINN can predict the solution space given 
domain bounds and the input space. 
'''
class PINN(tf.keras.Model):
    def __init__(self, inputs, outputs, lower_bound, upper_bound, p, f_boundary, f_bound, size, n_samples=20000):
        super(PINN, self).__init__(inputs=inputs, outputs=outputs)
        self.lower_bound = lower_bound # In log space
        self.upper_bound = upper_bound # In log space
        self.p = p # In real space
        self.f_boundary = f_boundary # In log and scaled space
        self.n_samples = n_samples
        self.size = size
        self.f_bound = f_bound # In log space
        
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
        PINN(p_boundary, r_boundary) = f_pred_boundary and f_boundary.
        
        g_boundary: (boundary_batchsize, 1) shaped arrays : This is the target data for the boundary value inputs. G_boundary is the
                    scaled version of f_boundary, and relates to f_boundary via g = (log(f) - min(log(f)))/(max(log(f)) - min(log(f)))
        
        alpha: weight on boundary_loss, 1-alpha weight on pinn_loss
        
        beta: weight on pinn_loss to scale it to the same order of magnitude as boundary_loss
        
    Outputs: sum_loss, pinn_loss, boundary_loss
    '''
    def train_step(self, p, r, p_boundary, r_boundary, f_boundary, alpha, beta):
        with tf.GradientTape(persistent=True) as t2: 
            with tf.GradientTape(persistent=True) as t1: 
                t1.watch(p)
                t1.watch(r)
                t1.watch(p_boundary)
                t1.watch(r_boundary)
                
                # PINN loss data
                p_scaled = (tfm.log(p) - self.lower_bound[0])/tfm.abs(self.upper_bound[0] - self.lower_bound[0])
                r_scaled = (tfm.log(r) - self.lower_bound[1])/tfm.abs(self.upper_bound[1] - self.lower_bound[1])
                P = tf.concat((p_scaled, r_scaled), axis=1)
                g = self.tf_call(P)

                # Boundary loss data
                p_boundary_scaled = (tfm.log(p_boundary) - self.lower_bound[0])/tfm.abs(self.upper_bound[0] - self.lower_bound[0])
                r_boundary_scaled = (tfm.log(r_boundary) - self.lower_bound[1])/tfm.abs(self.upper_bound[1] - self.lower_bound[1])
                P_boundary = tf.concat((p_boundary_scaled, r_boundary_scaled), axis=1)
                g_pred_boundary = self.tf_call(P_boundary)
                
                # Calculate boundary loss
                boundary_loss = tfm.reduce_mean(tfm.square(g_pred_boundary - f_boundary))

            # Calculate first-order gradients
            g_p = t1.gradient(g, p)
            g_r = t1.gradient(g, r)
            
            g_p_boundary = t1.gradient(g_pred_boundary, p_boundary)
            g_r_boundary = t1.gradient(g_pred_boundary, r_boundary)
            
            # Calculate f (real space) from g (scaled sapce) and get df/dg
            with tf.GradientTape(persistent=True) as t3: 
                t3.watch(g)
                t3.watch(g_pred_boundary)
                
                diff = tfm.abs(self.f_bound[1] - self.f_bound[0])

                f = tfm.exp(g*diff + self.f_bound[0])
                f_pred_boundary = tfm.exp(g_pred_boundary*diff + self.f_bound[0])

                f_g = t3.gradient(f, g)
                f_g_boundary = t3.gradient(f_pred_boundary, g_pred_boundary)
            
            # Use df/dg in the chain rule to calculate df/dp and df/dr
            f_p = f_g*g_p
            f_r = f_g*g_r
            
            f_p_boundary = f_g_boundary*g_p_boundary
            f_r_boundary = f_g_boundary*g_r_boundary
            
            # Calculate PINN loss and total loss
            pinn_loss = beta*self.pinn_loss(p, r, f_p, f_r) + beta*self.pinn_loss(p_boundary, r_boundary, f_p_boundary, f_r_boundary)
            
            total_loss = (1-alpha)*pinn_loss + alpha*boundary_loss

        # Backpropagation
        gradients = t2.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return pinn_loss.numpy(), boundary_loss.numpy()
    
    '''
    Description: The fit function used to iterate through epoch * steps_per_epoch steps of train_step. 
    
    Inputs: 
        P_predict: (N, 2) array: Input data for entire spatial and temporal domain. Used for vizualization for
        predictions at the end of each epoch. Michael created a very pretty video file with it. 
        
        client, trial: Sherpa client and trial
        
        alpha: weight on boundary_loss, 1-alpha weight on pinn_loss
        
        beta: weight on pinn_loss to scale it to the same order of magnitude as boundary_loss
        
        batchsize: batchsize for (p, r) in train step
        
        boundary_batchsize: batchsize for (x_lower, t_boundary) and (x_upper, t_boundary) in train step
        
        epochs: epochs
        
        lr: learning rate
        
        size: size of the prediction data (i.e. len(p) and len(r))
        
        save: Whether or not to save the model to a checkpoint every 10 epochs
        
        load_epoch: If -1, a saved model will not be loaded. Otherwise, the model will be 
        loaded from the provided epoch
        
        lr_decay: If -1, learning rate will not be decayed. Otherwise, lr = lr_decay*lr if loss hasn't 
        decreased
        
        alpha_decay: If -1, alpha will not be changed. Otherwise, alpha = alpha_decay*alpha if loss 
        hasn't decreased
        
        r_lower_change: If -1, r_lower will not be changed. Otherwise r_lower will decrease from the self.upper_bound[1] to 
        self.lower_bound[1].
        
        alpha_limit = Minimum alpha value to decay to
        
        patience: Number of epochs to check whether loss has decreased before updating lr or alpha
        
        filename: Name for the checkpoint file
    
    Outputs: Losses for each equation (Total, PDE, Boundary Value), and predictions for each epoch.
    '''
    def fit(self, P_predict, client=None, trial=None, alpha=0.5, beta=1, batchsize=64, boundary_batchsize=16, epochs=20, lr=3e-3, size=256, 
            save=False, load_epoch=-1, lr_decay=-1, alpha_decay=-1, r_lower_change=-1, alpha_limit = 0, patience=3, filename=''):
        
        # If load == True, load the weights
        if load_epoch != -1:
            name = './outputs/ckpts/pinn_' + filename + '_epoch_' + str(load_epoch)
            self.load_weights(name)
        
        # Initialize
        steps_per_epoch = np.ceil(self.n_samples / batchsize).astype(int)
        total_pinn_loss = np.zeros((epochs,))
        total_boundary_loss = np.zeros((epochs,))
        predictions = np.zeros((size**2, 1, epochs))
        
        # Lower r boundary
        r_lower = self.upper_bound[1]
        
        # For each epoch, sample new values in the PINN and boundary areas and pass them to train_step
        for epoch in range(epochs):
            # Compile
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            self.compile(optimizer=opt)

            sum_loss = np.zeros((steps_per_epoch,))
            pinn_loss = np.zeros((steps_per_epoch,))
            boundary_loss = np.zeros((steps_per_epoch,))
            
            # For each step, sample data and pass to train_step
            for step in range(steps_per_epoch):
                # Sample p and r
                beta_dist = tfd.Beta(3, 1)
                uniform_dist = tfd.Uniform(0, 1)

                p = (uniform_dist.sample((batchsize, 1))*tfm.abs(self.upper_bound[0] - self.lower_bound[0])) + self.lower_bound[0]
                p = tfm.exp(p)
                r = (beta_dist.sample((batchsize, 1))*tfm.abs(tfm.exp(self.upper_bound[1]) - tfm.exp(r_lower))) + tfm.exp(r_lower)
                
                # Randomly sample boundary_batchsize from p_boundary and f_boundary
                p_idx = np.expand_dims(np.random.choice(self.f_boundary.shape[0], boundary_batchsize, replace=False), axis=1)
                p_boundary = tf.Variable(self.p[p_idx], dtype=tf.float32)
                f_boundary = self.f_boundary[p_idx]
                
                # Create r_boundary array = r_HP
                upper_boundary = np.zeros((boundary_batchsize, 1))
                upper_boundary[:] = tfm.exp(self.upper_bound[1])
                r_boundary = tf.Variable(upper_boundary, dtype=tf.float32)
                
                # Train and get loss
                losses = self.train_step(p, r, p_boundary, r_boundary, f_boundary, alpha, beta)
                pinn_loss[step] = losses[0]
                boundary_loss[step] = losses[1]
            
            # Sum losses
            total_pinn_loss[epoch] = np.sum(pinn_loss)
            total_boundary_loss[epoch] = np.sum(boundary_loss)
            print(f'Epoch {epoch}. Current alpha: {alpha:.6f}, lr: {lr:.10f}, r_lower: {(np.exp(r_lower[0])/150e6):.1f}. ' + 
                  f'Training losses: pinn: {total_pinn_loss[epoch]:.10f}, boundary: {total_boundary_loss[epoch]:.10f}, ' +
                  f'weighted total: {((alpha*total_boundary_loss[epoch])+((1-alpha)*total_pinn_loss[epoch])):.10f}')
            
            predictions[:, :, epoch] = self.predict(P_predict, batchsize)
            
            # Decay lr if loss hasn't decreased since current epoch - patience
            hasntDecreased = False
            if (epoch > patience):
                if (total_pinn_loss[epoch] + total_boundary_loss[epoch]) > (total_pinn_loss[epoch-patience] + total_boundary_loss[epoch-patience]):
                    hasntDecreased = True
                        
            if (lr_decay != -1) & hasntDecreased:
                lr = lr_decay*lr

            # Decrease alpha each epoch
            if (alpha_decay != -1) & (alpha >= alpha_limit) & (epoch%10 == 0):
                alpha = alpha_decay*alpha
            
            # Decrease lower r boundary from 120au to 0.4au every 10 epochs
            if (r_lower_change != -1) & (r_lower >= self.lower_bound[1]) & (epoch%10 == 0): #23.6) & (epoch%10 == 0):
                r_lower = r_lower_change*r_lower

            # If the epoch is a multiple of 100, save to a checkpoint
            if (epoch%100 == 0) & (save == True):
                name = './outputs/ckpts/pinn_' + filename + '_epoch_' + str(epoch)
                self.save_weights(name, overwrite=True, save_format=None, options=None)
                
            # Send metrics
            if client:
                if (np.isnan(total_pinn_loss[epoch]) and np.isnan(total_boundary_loss[epoch])):
                    obj = np.inf
                else:
                    obj = total_pinn_loss[epoch] + total_boundary_loss[epoch]
                client.send_metrics(
                         trial=trial,
                         iteration=epoch,
                         objective=obj)
        
        return total_pinn_loss, total_boundary_loss, predictions
    
    # Predict for some P's the value of the neural network f(r, p)
    def predict(self, P, batchsize):
        P_size = P.shape[0]
        steps_per_epoch = np.ceil(P_size / batchsize).astype(int)
        predictions = np.zeros((P_size, 1))
        
        # For each step predict on data between start and end indices
        for step in range(steps_per_epoch):
            start_idx = step * 64
            
            # Calculate end_idx
            if step == steps_per_epoch - 1:
                end_idx = P_size - 1
            else:
                end_idx = start_idx + 64
                
            # Predict
            predictions[start_idx:end_idx, :] = self.tf_call(P[start_idx:end_idx, :]).numpy()
        
        return predictions
    
    def evaluate(self, ): 
        pass
    
    # pinn_loss calculates the PINN loss by calculating the MAE of the pinn function
    @tf.function
    def pinn_loss(self, p, r, f_p, f_r):
        V = 400 # 400 km/s
        m = 0.938 # GeV/c^2
        k_0 = 1e11 # km^2/s
        k_1 = k_0 * tfm.divide(r, 150e6) # km^2/s
        k_2 = p # unitless, k_2 = p/p0 and p0 = 1 GeV/c
        R = p # GV
        beta = tfm.divide(p, tfm.sqrt(tfm.square(p) + tfm.square(m))) 
        k = beta*k_1*k_2
        
        # Calculate physics loss
        l_f = tfm.reduce_mean(tfm.square(f_r + (tfm.divide(R*V, 3*k) * f_p)))
        
        return l_f
    
    # tf_call passes inputs through the neural network
    @tf.function
    def tf_call(self, inputs): 
        return self.call(inputs, training=True)