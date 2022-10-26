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

CURRENT_PATH = os.path.dirname(__file__)
PINN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", "..", "src"))
sys.path.insert(1, PINN_PATH)

from pinn import PINN


def main(client, trial):
    # Load data
    with open('./../data/f_boundary.pkl', 'rb') as file:
        f_boundary = pkl.load(file)

    with open('./../data/p.pkl', 'rb') as file:
        p = pkl.load(file)

    with open('./../data/T.pkl', 'rb') as file:
        T = pkl.load(file)

    with open('./../data/r.pkl', 'rb') as file:
        r = pkl.load(file)

    with open('./../data/P_predict.pkl', 'rb') as file:
        P_predict = pkl.load(file)
    
    # Sherpa
    hyperparameters = trial.parameters

    # Hyperparameters
    alpha = 0.99
    alpha_decay = 0.998
    alpha_limit = 0.1
    beta = 1e9
    lr_decay = 0.95
    patience = 10
    batchsize = 1032
    boundary_batchsize = 256
    epochs = 100
    activation = 'selu'
    save = False
    load_epoch = -1
    filename = ''
    n_samples = 20000
    lr = hyperparameters.get['lr']
    num_layers = hyperparameters.get['num_layers']
    num_hidden_units = hyperparameters.get['num_hidden_units']

    # Create model
    inputs = tf.keras.Input((2))
    x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(inputs)
    for _ in range(num_layers-1):
        x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(x_)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x_)

    # Train the PINN
    pinn = PINN(inputs=inputs, outputs=outputs, lower_bound=lb, upper_bound=ub, p=p[:, 0], f_boundary=f_boundary[:, 0], size=size, n_samples=n_samples)
    pinn_loss, boundary_loss, predictions = pinn.fit(client, trial, P_predict=P_predict, alpha=alpha, beta=beta, batchsize=batchsize, 
                                                     boundary_batchsize=boundary_batchsize, epochs=epochs, lr=lr, size=size, save=save, load_epoch=load_epoch, 
                                                     lr_decay=lr_decay, alpha_decay=alpha_decay, patience=patience, filename=filename)

if __name__ == '__main__':
    client = sherpa.Client()
    trial = client.get_trial()
    
    print('Made it to the main method!')
    
    main(client, trial)
    