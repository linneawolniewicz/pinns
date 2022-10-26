# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfm = tf.math
tf.config.list_physical_devices(device_type=None)

# Constants  
m = 0.938 # GeV/c^2
gamma = -3 # Between -2 and -3
size = 512 # size of r, T, p, and f_boundary
au = 150e6 # 150e6 m/AU
r_limits = [119, 120]
T_limits = [0.001, 1000]

# Create boundary data
T = np.logspace(np.log10(T_limits[0]), np.log10(T_limits[1]), size).flatten()[:, None]
p = (np.sqrt((T+m)**2-m**2)).flatten()[:,None] # GeV/c
r = np.logspace(np.log10(r_limits[0]*au), np.log10(r_limits[1]*au), size).flatten()[:, None] # km
f_boundary = ((T + m)**gamma)/(p**2) # particles/(m^3 (GeV/c)^3)

# Get upper and lower bounds
lb = np.array([p[0], r[0]], dtype='float32')
ub = np.array([p[-1], r[-1]], dtype='float32')

# Create test data
p_predict = np.log(p)
r_predict = np.log(r)

p_predict = (p_predict - np.log(lb[0]))/np.abs(np.log(ub[0]) - np.log(lb[0]))
r_predict = (r_predict - np.log(lb[1]))/np.abs(np.log(ub[1]) - np.log(lb[1]))
    
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
    
# Save data
with open('f_boundary.pkl', 'wb') as file:
    pkl.dump(f_boundary, file)

with open('p.pkl', 'wb') as file:
    pkl.dump(p, file)

with open('T.pkl', 'wb') as file:
    pkl.dump(T, file)

with open('r.pkl', 'wb') as file:
    pkl.dump(r, file)
    
with open('P_predict.pkl', 'wb') as file:
    pkl.dump(P_predict, file)