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
    def __init__(self, inputs, outputs, lower_bound, upper_bound, p, f_boundary, f_bound, size, num_samples=20_000):
        super(PINN, self).__init__(inputs=inputs, outputs=outputs)
        self.lower_bound = lower_bound # In log space
        self.upper_bound = upper_bound # In log space
        self.p = p # In real space
        self.f_boundary = f_boundary # In scaled space (0 to 1)
        self.num_samples = num_samples
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
                p_scaled = self.scale(p, self.upper_bound[0], self.lower_bound[0], should_take_log=True)
                r_scaled = self.scale(r, self.upper_bound[1], self.lower_bound[1], should_take_log=True)
                P = tf.concat((p_scaled, r_scaled), axis=1)
                g = self.tf_call(P)

                # Boundary loss data
                p_boundary_scaled = self.scale(p_boundary, self.upper_bound[0], self.lower_bound[0], should_take_log=True)
                r_boundary_scaled = self.scale(r_boundary, self.upper_bound[1], self.lower_bound[1], should_take_log=True)
                P_boundary = tf.concat((p_boundary_scaled, r_boundary_scaled), axis=1)
                g_pred_boundary = self.tf_call(P_boundary)
                
                # Calculate boundary loss
                boundary_loss = tfm.reduce_mean(tfm.square(g_pred_boundary - f_boundary))

            # Calculate first-order gradients
            dg_dp = t1.gradient(g, p)
            dg_dr = t1.gradient(g, r)
            
            dg_dp_boundary = t1.gradient(g_pred_boundary, p_boundary)
            dg_dr_boundary = t1.gradient(g_pred_boundary, r_boundary)
            
            # Calculate f (real space) from g (scaled sapce) and get df/dg
            with tf.GradientTape(persistent=True) as t3: 
                t3.watch(g)
                t3.watch(g_pred_boundary)
                
                diff = tfm.abs(self.f_bound[1] - self.f_bound[0])

                f = tfm.exp(g*diff + self.f_bound[0])
                f_pred_boundary = tfm.exp(g_pred_boundary*diff + self.f_bound[0])

                df_dg = t3.gradient(f, g)
                df_dg_boundary = t3.gradient(f_pred_boundary, g_pred_boundary)
            
            # Use chain rule to calculate df/dp and df/dr
            df_dp = df_dg*dg_dp
            df_dr = df_dg*dg_dr
            
            df_dp_boundary = df_dg_boundary*dg_dp_boundary
            df_dr_boundary = df_dg_boundary*dg_dr_boundary
            
            # Calculate PINN loss and total loss
            pinn_loss = beta*(self.pinn_loss(p, r, df_dp, df_dr) + self.pinn_loss(p_boundary, r_boundary, df_dp_boundary, df_dr_boundary))
            
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
        
        beta: weight on pinn_loss to scale it to the same order of magnitude as boundary_loss
        
        batchsize: batchsize for (p, r) in train step
        
        boundary_batchsize: batchsize for (x_lower, t_boundary) and (x_upper, t_boundary) in train step
        
        epochs: epochs
        
        lr: learning rate
        
        size: size of the prediction data (i.e. len(p) and len(r))
        
        save: Whether or not to save the model to a checkpoint every 10 epochs
        
        load_epoch: If -1, a saved model will not be loaded. Otherwise, the model will be 
        loaded from the provided epoch
        
        lr_schedule: Determines the schedule lr will be on. Options include 'decay' and 'oscillate', else lr will remain constant
        
        alpha_schedule: Determines the schedule alpha will be on. Choices include 'decay', 'grow', and 'oscillate'
        
        r_lower: Changes the r sampling. R will be sampled between r_lower and self.upper_bound[1] according to
        the sampling_method. Default r_lower is self.lower_bound[1], or np.log(0.4 * 150e6)
        
        patience: Number of epochs to check whether loss has decreased before decaying lr, if lr_schedule='decay'
        
        num_cycles = Number of cycles to oscillate lr for, if lr_schedule='oscillate'
        
        filename: Name for the checkpoint file
        
        sampling_method = Method for sampling r. Choices include 'beta_3_1' or 'beta_1_3', otherwise will sample uniformally in real space
        
        adam_beta1 = Value of beta1 for the Adam optimizer. Changes emphasis on momentum during training
        
        should_r_lower_change = Toggle for whether to decrease r_lower from self.upper_bound[1] to self.lower_bound[1] or not.
    
    Outputs: Losses for each equation (Total, PDE, Boundary Value), and predictions for each epoch.
    '''
    def fit(self, P_predict, client=None, trial=None, beta=1, batchsize=64, boundary_batchsize=16, epochs=20, lr=3e-3, 
            size=256, save=False, load_epoch=-1, lr_schedule='', alpha_schedule='', r_lower=17.909855, patience=3, num_cycles=10,
            filename='', sampling_method='uniform', adam_beta1=0.9, should_r_lower_change=False):
        
        # If load == True, load the weights
        if load_epoch != -1:
            name = './outputs/ckpts/pinn_' + filename + '_epoch_' + str(load_epoch)
            self.load_weights(name)
        
        # Initialize alpha based on alpha_schedule
        if (alpha_schedule == 'decay'): alpha = 1.0
        elif (alpha_schedule == 'grow'): alpha = 0.001
        else: alpha = 0.5
        
        # Initialize variables for oscillating lr schedule
        just_decreased = False
        max_lr = lr
        min_lr = max_lr/1000
        stepsize = (max_lr-min_lr)/(epochs/(num_cycles/2))
        
        # Initialize
        steps_per_epoch = np.ceil(self.num_samples / batchsize).astype(int)
        total_pinn_loss = np.zeros((epochs,))
        total_boundary_loss = np.zeros((epochs,))
        predictions = np.zeros((size**2, 1, epochs))
        
        # For each epoch, sample new values in the PINN and boundary areas and pass them to train_step
        for epoch in range(epochs):
            # Compile
            opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=adam_beta1)
            self.compile(optimizer=opt)

            sum_loss = np.zeros((steps_per_epoch,))
            pinn_loss = np.zeros((steps_per_epoch,))
            boundary_loss = np.zeros((steps_per_epoch,))
            
            # For each step, sample data and pass to train_step
            for step in range(steps_per_epoch):
                # Sample p according to a uniform distribution in log space
                uniform_dist = tfd.Uniform(0, 1)
                p = (uniform_dist.sample((batchsize, 1))*tfm.abs(self.upper_bound[0] - self.lower_bound[0])) + self.lower_bound[0]
                p = tfm.exp(p)
                
                # Sample r according to sampling_method variable
                if(sampling_method=='beta_1_3'): dist = tfd.Beta(1, 3)
                elif(sampling_method=='beta_3_1'): dist = tfd.Beta(3, 1)
                else: dist = tfd.Uniform(0, 1)
                
                r = (dist.sample((batchsize, 1))*tfm.abs(tfm.exp(self.upper_bound[1]) - tfm.exp(r_lower))) + tfm.exp(r_lower)
                
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
            print(f'Epoch {epoch}. Current alpha: {alpha:.6f}, lr: {lr:.10f}, ' + 
                  f'Training losses: pinn: {total_pinn_loss[epoch]:.10f}, boundary: {total_boundary_loss[epoch]:.10f}, ' +
                  f'weighted total: {((alpha*total_boundary_loss[epoch])+((1-alpha)*total_pinn_loss[epoch])):.10f}')
            
            predictions[:, :, epoch] = self.predict(P_predict, batchsize)
            
            # Adjust alpha based on the schedule, only every 10 epochs
            if (epoch%10 == 0):
                if alpha_schedule=='decay': alpha = alpha*0.995
                elif (alpha_schedule=='grow') & (alpha <= 1): alpha = alpha*1.015

            # Check if loss has decreased
            hasnt_decreased = False
            if (epoch > patience):
                if (total_pinn_loss[epoch] + total_boundary_loss[epoch]) > (total_pinn_loss[epoch-patience] + total_boundary_loss[epoch-patience]):
                    hasnt_decreased = True
                        
            # If loss hasn't decreased, adjust lr based on the assigned schedule
            if ((lr_schedule == 'decay') & hasnt_decreased): lr = lr*0.95
            elif (lr_schedule == 'oscillate'): 
                lr, just_decreased = self.oscillate_lr(just_decreased, lr, min_lr, max_lr, stepsize)
                
            
            # Decrease lower r if told to
            if should_r_lower_change & (epoch%10==0): r_lower = r_lower*0.95

            # Save the model to a checkpoint
            should_save = (epoch%100 == 0) & (save == True)
            if should_save:
                name = './outputs/ckpts/pinn_' + filename + '_epoch_' + str(epoch)
                self.save_weights(name, overwrite=True, save_format=None, options=None)
                
            # Send metrics if running Sherpa optimization
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
    def pinn_loss(self, p, r, df_dp, df_dr):
        V = 400 # 400 km/s
        M = 0.938 # GeV/c^2
        k_0 = 1e11 # km^2/s
        k_1 = k_0 * tfm.divide(r, 150e6) # km^2/s
        k_2 = p # unitless, k_2 = p/p0 and p0 = 1 GeV/c
        R = p # GV
        beta = tfm.divide(p, tfm.sqrt(tfm.square(p) + tfm.square(M))) 
        k = beta*k_1*k_2
        
        # Calculate physics loss
        l_f = tfm.reduce_mean(tfm.square(df_dr + (tfm.divide(R*V, 3*k) * df_dp)))
        
        return l_f
    
    # tf_call passes inputs through the neural network
    @tf.function
    def tf_call(self, inputs): 
        return self.call(inputs, training=True)
    
    # Scales input data and returns scaled version
    def scale(self, data, upper_bound, lower_bound, should_take_log=True):
        if should_take_log: scaled_data = (tfm.log(data) - lower_bound)/tfm.abs(upper_bound - lower_bound)   
        else: scaled_data = (data - lower_bound)/tfm.abs(upper_bound - lower_bound)
            
        return scaled_data
    
    f'''
    Description: Implements an oscillating lr according to the triangular CLR schedule: decrease lr linearly to min_lr, then increase linearly to max_lr, and repeat
    
    Inputs: 
        just_decreased: Boolean, True if lr decreased last epoch and False if not
        
        current_lt: float, Current learning rate
        
        min_lr: float, minimum learning rate to decrease to. In literature defined as 1/R smaller than max_lr
        
        max_lr: float, maximum learning rate to increase to
        
        stepsize: float, value to increase/decrease lr by
        
    Outputs: 
        just_decreased and new lr
    '''
    def oscillate_lr(self, just_decreased, lr, min_lr, max_lr, stepsize):
        decrease_lr = (just_decreased and (lr - stepsize >= min_lr)) or ((not just_decreased) and (lr + stepsize > max_lr))
        increase_lr = ((not just_decreased) and (lr + stepsize <= max_lr)) or (just_decreased and (lr - stepsize < min_lr))

        if decrease_lr: 
            lr = lr - stepsize
            just_decreased = True
        elif increase_lr:
            lr = lr + stepsize
            just_decreased = False

        return lr, just_decreased