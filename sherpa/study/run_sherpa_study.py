# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sherpa
import pickle as pkl
import os
import sys

tfd = tfp.distributions
tfm = tf.math
tf.config.list_physical_devices(device_type=None)

CURRENT_PATH = os.getcwd()
PINN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", "..", "src"))
DATA_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", "..", "data"))
sys.path.insert(1, PINN_PATH)

from pinn import PINN


def main(client, trial):
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

    # Sherpa
    parameters = [
        sherpa.Ordinal(name='lr', range=[3e-3, 3e-4, 3e-5, 3e-6]),
        sherpa.Continuous(name='alpha', range=[0.95, 1.0]),
        sherpa.Discrete(name='num_hidden_units', range=[100, 700]),
        sherpa.Discrete(name='num_layers', range=[5, 10]),
        sherpa.Continuous(name='sample_value', range=[1, 3])
    ]
    
    n_run = 70
    study = sherpa.Study(
        parameters=parameters,
        algorithm=sherpa.algorithms.RandomSearch(max_num_trials=n_run),
        lower_is_better=True
    )

    # Hyperparameters
    alpha_decay = 0.998
    alpha_limit = 0.1
    beta = 1e9
    lr_decay = 0.95
    patience = 10
    batchsize = 1032
    boundary_batchsize = 512
    epochs = 100
    activation = 'selu'
    save = False
    load_epoch = -1
    n_samples = 20000
    filename = ''
    
    # run Sherpa experiment
    dfs = []
    for i, trial in enumerate(study):
        print("-----------------------------------------------------------")
        print(f'Trial id: {i}')
        start = time.time()
        
        # Get hyperparameters
        lr = trial.parameters['lr']
        alpha = trial.parameters['alpha']
        num_layers = trial.parameters['num_layers']
        num_hidden_units = trial.parameters['num_hidden_units']
        sample_value = trial.parameters['sample_value']

        # Create model
        inputs = tf.keras.Input((2))
        x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(inputs)
        for _ in range(num_layers-1):
            x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(x_)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x_)
                                  
        # Train the PINN
        pinn = PINN(inputs=inputs, outputs=outputs, lower_bound=lb, upper_bound=ub, p=p[:, 0], f_boundary=f_boundary[:, 0], size=size, n_samples=n_samples)
        pinn_loss, boundary_loss, predictions = pinn.fit(P_predict=P_predict, alpha=alpha, beta=beta, batchsize=batchsize, boundary_batchsize=boundary_batchsize,
                                                                 epochs=epochs, lr=lr, size=size, save=save, load_epoch=load_epoch, lr_decay=lr_decay,
                                                                 alpha_decay=alpha_decay, alpha_limit=alpha_limit, sample_value=sample_value, 
                                                                 patience=patience, filename=filename)
        
        # Save model output dataframe
        df = pd.DataFrame()
        df['trial_id'] = i,
        df['num_layers'] = num_layers
        df['num_hidden_units'] = num_hidden_units
        df['lr'] = lr
        df['batchsize'] = batchsize
        df['boundary_batchsize'] = boundary_batchsize
        df['sample_value'] = sample_value
        df['beta'] = beta
        df['alpha'] = alpha
        df['alpha_decay'] = alpha_decay
        df['lr_decay'] = lr_decay
        df['alpha_limit'] = alpha_limit
        df['patience'] = patience
        df['epochs'] = epochs
        df['activation'] = activation
        df['final_boundary_loss'] = boundary_loss[-1]
        df['final_pinn_loss'] = pinn_loss[-1]
        dfs.append(df)
        
        # Every 5 trials save dataframe
        if i%5==0: pd.concat(dfs).to_csv('./outputs/sherpa_' + str(n_run) + '.csv')
        
        # Pickle predictions and losses
        with open('./pickles/pinn_loss_sherpa_' + str(n_run) + '_trial_id_' + str(i) + '.pkl', 'wb') as file:
            pkl.dump(pinn_loss, file)
        with open('./pickles/boundary_loss_sherpa_' + str(n_run) + '_trial_id_' + str(i) + '.pkl', 'wb') as file:
            pkl.dump(boundary_loss, file)
        with open('./pickles/predictions_sherpa_' + str(n_run) + '_trial_id_' + str(i) + '.pkl', 'wb') as file:
            pkl.dump(predictions, file)
        
        # Write progress.txt
        end = time.time()
        line = '===============================================\n'
        line += "Trial id: {}, elapsed time: {:.3f}".format(i, end - start) + '\n'
        with open('progress.txt', 'a') as f:
            f.write(line)
            
    # Save final dataframe
    pd.concat(dfs).to_csv('./outputs/sherpa_' + str(n_run) + '.csv')

if __name__ == '__main__':
    main()
    