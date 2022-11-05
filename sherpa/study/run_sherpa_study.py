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
    with open(DATA_PATH + '/f_boundary.pkl', 'rb') as file:
        f_boundary = pkl.load(file)

    with open(DATA_PATH + '/p.pkl', 'rb') as file:
        p = pkl.load(file)

    with open(DATA_PATH + '/T.pkl', 'rb') as file:
        T = pkl.load(file)

    with open(DATA_PATH + '/r.pkl', 'rb') as file:
        r = pkl.load(file)

    with open(DATA_PATH + '/P_predict.pkl', 'rb') as file:
        P_predict = pkl.load(file)
        
    # Get upper and lower bounds
    lb = np.log(np.array([p[0], r[0]], dtype='float32'))
    ub = np.log(np.array([p[-1], r[-1]], dtype='float32'))
    f_bound = np.array([-34.54346331847909, 6.466899920699378], dtype='float32')
    size = len(f_boundary[:, 0])

    # Sherpa
    parameters = [
        sherpa.Continuous(name='alpha_decay', range=[0.99, 1]),
        sherpa.Continuous(name='lr', range=[3e-5, 3e-8]),
        sherpa.Discrete(name='num_hidden_units', range=[10, 500]),
        sherpa.Discrete(name='num_layers', range=[2, 10])
    ]
    
    n_run = 200
    study = sherpa.Study(
        parameters=parameters,
        algorithm=sherpa.algorithms.RandomSearch(max_num_trials=n_run),
        lower_is_better=True
    )

    # Hyperparameters
    epochs = 50
    alpha = 1
    alpha_limit = 0
    lr_decay = 0.95
    patience = 10
    batchsize = 1032
    boundary_batchsize = 512
    activation = 'selu'
    save = False
    load_epoch = -1
    filename = ''
    n_samples = 20000
    
    # run Sherpa experiment
    dfs = []
    for i, trial in enumerate(study):
        print("-----------------------------------------------------------")
        print(f'Trial id: {i}')
        start = time.time()
        
        # Get hyperparameters
        alpha_decay = trial.parameters['alpha_decay']
        lr = trial.parameters['lr']
        num_layers = trial.parameters['num_layers']
        num_hidden_units = trial.parameters['num_hidden_units']

        # Create model
        inputs = tf.keras.Input((2))
        x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(inputs)
        for _ in range(num_layers-1):
            x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(x_)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x_)
                                  
        # Train the PINN
        pinn = PINN(inputs=inputs, outputs=outputs, lower_bound=lb, upper_bound=ub, p=p[:, 0], f_boundary=f_boundary[:, 0], f_bound=f_bound, size=size, n_samples=n_samples)
        pinn_loss, boundary_loss, predictions = pinn.fit(P_predict=P_predict, client=None, trial=None, alpha=alpha, batchsize=batchsize, 
                                                         boundary_batchsize=boundary_batchsize, epochs=epochs, lr=lr, size=size, save=save, load_epoch=load_epoch, 
                                                         lr_decay=lr_decay, alpha_decay=alpha_decay, alpha_limit=alpha_limit, patience=patience, filename=filename)
        
        # Save model output dataframe
        df = pd.DataFrame()
        df['trial_id'] = i,
        df['num_layers'] = num_layers
        df['num_hidden_units'] = num_hidden_units
        df['lr'] = lr
        df['alpha_decay'] = alpha_decay
        df['batchsize'] = batchsize
        df['boundary_batchsize'] = boundary_batchsize
        df['alpha'] = alpha
        df['lr_decay'] = lr_decay
        df['alpha_limit'] = alpha_limit
        df['patience'] = patience
        df['epochs'] = epochs
        df['activation'] = activation
        df['final_boundary_loss'] = boundary_loss[-1]
        df['final_pinn_loss'] = pinn_loss[-1]
        dfs.append(df)
        
        # Every 5 trials save dataframe
        pd.concat(dfs).to_csv(OUTPUTS_PATH + '/outputs/sherpa_' + str(n_run) + '.csv')
        
        # Pickle predictions and losses
        with open(OUTPUTS_PATH + '/pickles/pinn_loss_sherpa_' + str(n_run) + '_trial_id_' + str(i) + '.pkl', 'wb') as file:
            pkl.dump(pinn_loss, file)
        with open(OUTPUTS_PATH + '/pickles/boundary_loss_sherpa_' + str(n_run) + '_trial_id_' + str(i) + '.pkl', 'wb') as file:
            pkl.dump(boundary_loss, file)
        with open(OUTPUTS_PATH + '/pickles/predictions_sherpa_' + str(n_run) + '_trial_id_' + str(i) + '.pkl', 'wb') as file:
            pkl.dump(predictions, file)
        
        # Write progress.txt
        end = time.time()
        line = '===============================================\n'
        line += "Trial id: {}, elapsed time: {:.3f}".format(i, end - start) + '\n'
        with open(OUTPUTS_PATH + '/progress.txt', 'a') as f:
            f.write(line)
            
    # Save final dataframe
    pd.concat(dfs).to_csv(OUTPUTS_PATH + '/outputs/sherpa_' + str(n_run) + '.csv')

if __name__ == '__main__':
    main()
    