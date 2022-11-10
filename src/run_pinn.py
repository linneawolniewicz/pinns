# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle as pkl
import os
import sys

tfd = tfp.distributions
tfm = tf.math
tf.config.list_physical_devices(device_type=None)

# Configure paths
CURRENT_PATH = os.getcwd()
PINN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "src"))
DATA_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "data"))
OUTPUTS_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "outputs"))

sys.path.insert(1, PINN_PATH)
from pinn import PINN

# Load data
with open(DATA_PATH + '/f_boundary.pkl', 'rb') as file:
    f_boundary = pkl.load(file)
    
with open(DATA_PATH + '/p.pkl', 'rb') as file:
    p = pkl.load(file)
    
with open(DATA_PATH + '/T.pkl', 'rb') as file:
    T = pkl.load(file)
    
with open(DATA_PATH + '/r_119au.pkl', 'rb') as file:
    r = pkl.load(file)
    
with open(DATA_PATH + '/P_predict.pkl', 'rb') as file:
    P_predict = pkl.load(file)
    
# Get upper and lower bounds
lb = np.log(np.array([p[0], r[0]], dtype='float32'))
ub = np.log(np.array([p[-1], r[-1]], dtype='float32'))
f_bound = np.array([-34.54346331847909, 6.466899920699378], dtype='float32')
size = len(f_boundary[:, 0])

# Hyperparameters
epochs = 1000
alpha = 1
alpha_decay = 0.9945024823413915
alpha_limit = 0
lr_decay = 0.95
patience = 10
batchsize = 1032
boundary_batchsize = 512
activation = 'selu'
save = True
load_epoch = -1
filename = 'newSampling'
n_samples = 20000
lr = 7.931296428571565e-07
num_layers = 6
num_hidden_units = 162

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

# Save PINN outputs
with open(OUTPUTS_PATH + '/pinn_loss_' + filename + '.pkl', 'wb') as file:
    pkl.dump(pinn_loss, file)
    
with open(OUTPUTS_PATH + '/boundary_loss_' + filename + '.pkl', 'wb') as file:
    pkl.dump(boundary_loss, file)
     
with open(OUTPUTS_PATH + '/predictions_' + filename + '.pkl', 'wb') as file:
    pkl.dump(predictions, file)