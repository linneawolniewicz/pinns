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
size = 1024

with open(DATA_PATH + '/f_boundary_' + str(size) + '.pkl', 'rb') as file:
    f_boundary = pkl.load(file)

with open(DATA_PATH + '/p_' + str(size) + '.pkl', 'rb') as file:
    p = pkl.load(file)

with open(DATA_PATH + '/T_' + str(size) + '.pkl', 'rb') as file:
    T = pkl.load(file)

with open(DATA_PATH + '/r_' + str(size) + '.pkl', 'rb') as file:
    r = pkl.load(file)

with open(DATA_PATH + '/P_predict_' + str(size) + '.pkl', 'rb') as file:
    P_predict = pkl.load(file)

# Get upper and lower bounds
lb = np.log(np.array([p[0], r[0]], dtype='float32'))
ub = np.log(np.array([p[-1], r[-1]], dtype='float32'))
min_f_log_space = -34.54346331847909
max_f_log_space = 6.466899920699378
f_bound = np.array([min_f_log_space, max_f_log_space], dtype='float32')

# Hyperparameters
epochs = 500
beta = 1e13
adam_beta1 = 0.9
lr_schedule = 'decay'
patience = 30
num_cycles = 2
batchsize = 2048
boundary_batchsize = 512
activation = 'selu'
save = True
load_epoch = -1
num_samples = 20_000
lr = 3e-3
num_layers = 3
num_hidden_units = 224
final_activation = 'sigmoid'
filename = '3layers_224units_evolutionarySampling_and_adaptiveLoss'

# Create model
inputs = tf.keras.Input((2))
x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(inputs)
for _ in range(num_layers-1):
    x_ = tf.keras.layers.Dense(num_hidden_units, activation=activation)(x_)
outputs = tf.keras.layers.Dense(1, activation=final_activation)(x_)

# Train the PINN
pinn = PINN(inputs=inputs, outputs=outputs, lower_bound=lb, upper_bound=ub, p=p[:, 0], f_boundary=f_boundary[:, 0], f_bound=f_bound, size=size, num_samples=num_samples)
pinn_loss, boundary_loss, predictions = pinn.fit(P_predict=P_predict, client=None, trial=None, beta=beta, batchsize=batchsize, boundary_batchsize=boundary_batchsize, epochs=epochs, 
                                                 lr=lr, size=size, save=save, load_epoch=load_epoch, lr_schedule=lr_schedule, patience=patience, num_cycles=num_cycles, 
                                                 adam_beta1=adam_beta1, filename=filename)

# Save PINN outputs
with open(OUTPUTS_PATH + '/pinn_loss_' + filename + '.pkl', 'wb') as file:
    pkl.dump(pinn_loss, file)
    
with open(OUTPUTS_PATH + '/boundary_loss_' + filename + '.pkl', 'wb') as file:
    pkl.dump(boundary_loss, file)
     
with open(OUTPUTS_PATH + '/predictions_' + filename + '.pkl', 'wb') as file:
    pkl.dump(predictions[:, :, -100:], file)