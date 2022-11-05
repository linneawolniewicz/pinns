# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

tfd = tfp.distributions
tfm = tf.math
tf.config.list_physical_devices(device_type=None)

CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "data"))

# Constants  
m = 0.938 # GeV/c^2
gamma = -3 # Between -2 and -3
size = 512 # size of r, T, p, and f_boundary
au = 150e6 # 150e6 m/AU
r_limits = [0.4, 120] 
T_limits = [0.001, 1000]

# Create boundary data
T = np.logspace(np.log10(T_limits[0]), np.log10(T_limits[1]), size).flatten()[:, None]
p = (np.sqrt((T+m)**2-m**2)).flatten()[:,None] # GeV/c
r = np.logspace(np.log10(r_limits[0]*au), np.log10(r_limits[1]*au), size).flatten()[:, None] # km
f_boundary = ((T + m)**gamma)/(p**2) # particles/(m^3 (GeV/c)^3)

# Create J
J = np.zeros((size, size))
J_boundary = (f_boundary*(p**2))[:, 0]

# Define phi
def phi_func(r):
    k_0 = 1e11
    func = np.log((120*150e6)/r)*((150e6*400)/(3*k_0))
    return func

# For each r, get phi
for i in range(size):
    phi = phi_func(r[i])
    
    # for each T, calculate J[r, T]
    for j in range(size):
        where_phi = np.argmin(np.abs(T[:, 0] - (T[j, 0] + phi)))
        J[i, j] = ((T[j, 0]*(T[j, 0]+2*m))/((T[j, 0]+phi)*(T[j, 0]+phi+2*m))) * J_boundary[where_phi]

# Scale f_boundary
f_boundary = np.log(f_boundary)
f_boundary = (f_boundary - np.min(f_boundary))/np.abs(np.max(f_boundary) - np.min(f_boundary))

# Create test data
p_predict = np.log(p)
r_predict = np.log(r)

# Get upper and lower bounds
lb = np.array([p_predict[0], r_predict[0]], dtype='float32')
ub = np.array([p_predict[-1], r_predict[-1]], dtype='float32')

p_predict = (p_predict - lb[0])/np.abs(ub[0] - lb[0])
r_predict = (r_predict - lb[1])/np.abs(ub[1] - lb[1])
    
P, R = np.meshgrid(p_predict, r_predict)
P_predict = np.hstack((P.flatten()[:,None], R.flatten()[:,None]))

# Check inputs
print(f'r: {r.shape}, p: {p.shape}, T: {T.shape}, f_boundary: {f_boundary.shape}, P_predict: {P_predict.shape}')
print(f'lb: {lb}, ub:{ub}')

# Visualize
fig, axs = plt.subplots(4, figsize=(5, 15))
for ax, data, name in zip(axs, [T, f_boundary, r, p], ['T', 'f_boundary', 'r', 'p']):
    ax.set_title(name)
    pd.Series(data[:, 0]).hist(ax=ax)
plt.show()
    
# Save data
with open(DATA_PATH + '/f_boundary.pkl', 'wb') as file:
    pkl.dump(f_boundary, file)

with open(DATA_PATH + '/p.pkl', 'wb') as file:
    pkl.dump(p, file)

with open(DATA_PATH + '/T.pkl', 'wb') as file:
    pkl.dump(T, file)

with open(DATA_PATH + '/r_119au.pkl', 'wb') as file:
    pkl.dump(r, file)
    
with open(DATA_PATH + '/J.pkl', 'wb') as file:
    pkl.dump(J, file)
    
with open(DATA_PATH + '/P_predict.pkl', 'wb') as file:
    pkl.dump(P_predict, file)