# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sherpa
import pickle as pkl
import os
import sys
import time
import pandas as pd
import sherpa.schedulers
import sherpa.algorithms

tfd = tfp.distributions
tfm = tf.math
tf.config.list_physical_devices(device_type=None)

CURRENT_PATH = os.getcwd()
PINN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "src"))
DATA_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "data"))
OUTPUTS_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "sherpa", "study"))
sys.path.insert(1, PINN_PATH)

from pinn import PINN


def main():
    # Load data
    with open(DATA_PATH + '/f_boundary_2048.pkl', 'rb') as file:
        f_boundary = pkl.load(file)

    with open(DATA_PATH + '/p_2048.pkl', 'rb') as file:
        p = pkl.load(file)

    with open(DATA_PATH + '/T_2048.pkl', 'rb') as file:
        T = pkl.load(file)

    with open(DATA_PATH + '/r_2048.pkl', 'rb') as file:
        r = pkl.load(file)

    with open(DATA_PATH + '/P_predict_2048.pkl', 'rb') as file:
        P_predict = pkl.load(file)

    # Get upper and lower bounds
    lb = np.log(np.array([p[0], r[0]], dtype='float32'))
    ub = np.log(np.array([p[-1], r[-1]], dtype='float32'))
    min_f_log_space = -34.54346331847909
    max_f_log_space = 6.466899920699378
    f_bound = np.array([min_f_log_space, max_f_log_space], dtype='float32')
    size = len(f_boundary[:, 0])

    # Sherpa
    parameters = [
        sherpa.Discrete(name='num_hidden_units', range=[100, 1_000]),
        sherpa.Choice(name='batchsize', range=[512, 1024, 2048, 4096]),
        sherpa.Choice(name='boundary_batchsize', range=[512, 1024, 2048]),
        sherpa.Choice(name='num_layers', range=[2, 3]),
        sherpa.Choice(name='sampling_method', range=['uniform', 'beta_3_1', 'beta_1_3']),
        sherpa.Choice(name='lr_schedule', range=['decay', 'oscillate']),
        sherpa.Choice(name='alpha_schedule', range=['static', 'grow', 'decay']),
        sherpa.Choice(name='final_activation', range=['linear', 'sigmoid'])
    ]
    
    n_run = 750
    study = sherpa.Study(
        parameters=parameters,
        algorithm=sherpa.algorithms.RandomSearch(max_num_trials=n_run),
        lower_is_better=True
    )

    # Hyperparameters
    lr = 3e-3
    epochs = 100
    beta = 1e13
    lr_decay = 0.95
    patience = 20
    num_cycles = 3
    activation = 'selu'
    r_lower = np.log(0.4*150e6).astype('float32')
    num_samples = 20_000
    save = False
    load_epoch = -1
    should_r_lower_change = False
    filename = '_lrSchedulesMorePatienceFullR'

    # run Sherpa experiment
    dfs = []
    for i, trial in enumerate(study):
        print("-----------------------------------------------------------")
        print(f'Trial id: {i}')
        start = time.time()
        
        # Get hyperparameters
        num_hidden_units = trial.parameters['num_hidden_units']
        batchsize = trial.parameters['batchsize']
        boundary_batchsize = trial.parameters['boundary_batchsize']
        sampling_method = trial.parameters['sampling_method']
        lr_schedule = trial.parameters['lr_schedule']
        alpha_schedule = trial.parameters['alpha_schedule']
        final_activation = trial.parameters['final_activation']
        num_layers = trial.parameters['num_layers']

        # Create model
        inputs = tf.keras.Input((2))
        x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(inputs)
        for _ in range(num_layers-1):
            x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(x_)
        outputs = tf.keras.layers.Dense(1, activation=final_activation)(x_)
                                  
        # Train the PINN
        pinn = PINN(inputs=inputs, outputs=outputs, lower_bound=lb, upper_bound=ub, p=p[:, 0], f_boundary=f_boundary[:, 0], f_bound=f_bound, size=size, num_samples=num_samples)
        pinn_loss, boundary_loss, predictions = pinn.fit(P_predict=P_predict, client=None, trial=None, beta=beta, batchsize=batchsize, 
                                                         boundary_batchsize=boundary_batchsize, epochs=epochs, lr=lr, size=size, save=save, load_epoch=load_epoch, 
                                                         lr_schedule=lr_schedule, alpha_schedule=alpha_schedule, r_lower=r_lower, patience=patience, num_cycles=num_cycles,
                                                         filename=filename, sampling_method=sampling_method, should_r_lower_change=should_r_lower_change)
        
        # Save model output dataframe
        df = pd.DataFrame()
        df['trial_id'] = i,
        df['num_layers'] = num_layers
        df['num_hidden_units'] = num_hidden_units
        df['lr'] = lr
        df['alpha_schedule'] = alpha_schedule
        df['r_lower'] = r_lower
        df['batchsize'] = batchsize
        df['boundary_batchsize'] = boundary_batchsize
        df['lr_schedule'] = lr_schedule
        df['patience'] = patience
        df['num_cycles'] = num_cycles
        df['epochs'] = epochs
        df['activation'] = activation
        df['final_activation'] = final_activation
        df['sampling_method'] = sampling_method
        df['final_boundary_loss'] = boundary_loss[-1]
        df['final_pinn_loss'] = pinn_loss[-1]
        dfs.append(df)
        
        # Every 5 trials save dataframe
        pd.concat(dfs).to_csv(OUTPUTS_PATH + '/outputs/sherpa_' + str(n_run) + filename + '.csv')
        
        # Pickle predictions and losses
        with open(OUTPUTS_PATH + '/pickles/pinn_loss_sherpa_' + str(n_run) + '_trial_id_' + str(i) + filename + '.pkl', 'wb') as file:
            pkl.dump(pinn_loss, file)
        with open(OUTPUTS_PATH + '/pickles/boundary_loss_sherpa_' + str(n_run) + '_trial_id_' + str(i) + filename + '.pkl', 'wb') as file:
            pkl.dump(boundary_loss, file)
        with open(OUTPUTS_PATH + '/pickles/predictions_sherpa_' + str(n_run) + '_trial_id_' + str(i) + filename + '.pkl', 'wb') as file:
            pkl.dump(predictions, file)
        
        # Write progress.txt
        end = time.time()
        line = '===============================================\n'
        line += "Trial id: {}, elapsed time: {:.3f}".format(i, end - start) + '\n'
        with open(OUTPUTS_PATH + '/progress.txt', 'a') as f:
            f.write(line)
            
    # Save final dataframe
    pd.concat(dfs).to_csv(OUTPUTS_PATH + '/outputs/sherpa_' + str(n_run) + filename + '.csv')

if __name__ == '__main__':
    main()
    