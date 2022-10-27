import os
import sys
import pickle as pkl
import sherpa
import sherpa.schedulers
import sherpa.algorithms
import tensorflow as tf

# Set-up output directory
output_dir = '/home/linneamw/sadow_lts/personal/linneamw/research/pinns/sherpa/parallel/output'
if os.path.isdir(output_dir):
    print('Warning: Overwriting directory {}'.format(output_dir))
    import shutil
    shutil.rmtree(output_dir)
else:
    print(f'Writing to directory {output_dir}')

# Sherpa
parameters = [
    sherpa.Ordinal(name='lr', range=[3e-3, 3e-4, 3e-5, 3e-6, 3e-7]),
    sherpa.Discrete(name='num_hidden_units', range=[200, 1000]),
    sherpa.Discrete(name='num_layers', range=[3, 20]),
]

n_trials = 5
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=n_trials)
environment = '/home/linneamw/sadow_lts/personal/linneamw/anaconda3/envs/pinns'
options = '-N 1 -J sherpa -p sadow --account sadow --gres=gpu:NV-RTX2070:1 --mem=32gb -c 3 -t 3-00:00:00'

scheduler = sherpa.schedulers.SLURMScheduler(environment=environment,
                                             submit_options=options,
                                             output_dir=output_dir)

db_port = sherpa.core._port_finder(8895, 8910)

filename = '/home/linneamw/sadow_lts/personal/linneamw/research/pinns/sherpa/parallel/run_pinn.py'

print(f'Running mongodb on port: {db_port}')

results = sherpa.optimize(parameters=parameters,
                          algorithm=algorithm,
                          lower_is_better=True,
                          filename=filename,
                          output_dir=output_dir,
                          scheduler=scheduler,
                          max_concurrent=5,
                          verbose=1,
                          db_port=db_port,
                          mongodb_args={'bind_ip_all':''}
                         )

pkl.dump(results, open(output_dir + '/results.pkl', 'wb'))
