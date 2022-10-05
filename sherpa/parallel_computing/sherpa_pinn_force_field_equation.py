import os
import sys
import pickle as pkl
import sherpa
import sherpa.schedulers
import sherpa.algorithms
import tensorflow as tf

# Set-up output directory
output_dir = '/home/linneamw/sadow_lts/personal/linneamw/pinns/sherpa/output'
if os.path.isdir(output_dir):
    print('Warning: Overwriting directory {}'.format(output_dir))
    import shutil
    shutil.rmtree(output_dir)
else:
    print(f'Writing to directory {output_dir}')

# Sherpa parameters
parameters = [
    sherpa.Continuous(name='lr_decay', range=[0.8, 1.0]),
    sherpa.Discrete(name='patience', range=[2, 5]),
    sherpa.Continuous(name='alpha', range=[0.1, 0.5]),
    sherpa.Continuous(name='weight_change', range=[1.0, 1.5]),
    sherpa.Ordinal(name='batchsize', range=[256, 512, 1032, 2048]),
    sherpa.Ordinal(name='boundary_batchsize', range=[64, 128, 256, 512]),
    sherpa.Discrete(name='num_hidden_units', range=[10, 500]),
    sherpa.Discrete(name='num_layers', range=[2, 10]),
]

n_trials = 2
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=n_trials)
env = '/home/linneamw/sadow_lts/personal/linneamw/anaconda3/envs/pinns'
opt = '-N 1 -J sherpa_pinns -p gpu --gres=gpu:1 --constraint="volta" --mem=32gb -c 8 -t 2-23:00:00'

scheduler = sherpa.schedulers.SLURMScheduler(environment=env,
                                             submit_options=opt,
                                             output_dir=output_dir)

filename = '/home/linneamw/sadow_lts/personal/linneamw/pinns/sherpa/pinn_force_field_equation.py'

results = sherpa.optimize(parameters=parameters,
                          algorithm=algorithm,
                          lower_is_better=True,
                          filename=filename,
                          output_dir=output_dir,
                          scheduler=scheduler,
                          max_concurrent=15,
                          verbose=1,
                          db_port=8905,
                          mongodb_args={'bind_ip_all': ''})

pkl.dump(results, open(output_dir + '/results.pkl', 'wb'))
