import os
import sys
import pickle as pkl
import sherpa
import sherpa.schedulers
import sherpa.algorithms
import tensorflow as tf

# Set-up output directory
output_dir = '/home/linneamw/sadow_lts/personal/linneamw/pinns/sherpa/parallel_output'
if os.path.isdir(output_dir):
    print('Warning: Overwriting directory {}'.format(output_dir))
    import shutil
    shutil.rmtree(output_dir)
else:
    print(f'Writing to directory {output_dir}')

# Sherpa
parameters = [
    sherpa.Ordinal(name='lr', range=[0.03, 0.003, 0.0003]),
    sherpa.Continuous(name='alpha', range=[0.1, 1.0]),
    sherpa.Discrete(name='num_hidden_units', range=[100, 1000]),
    sherpa.Discrete(name='num_layers', range=[2, 10])
]

n_trials = 2
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=n_trials)
env = '/home/linneamw/sadow_lts/personal/linneamw/anaconda3/envs/pinns'
opt = '-N 1 -J sherpa_pinns -p gpu --gres=gpu:1 --constraint="volta" --mem=32gb -c 8 -t 3-00:00:00'

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
