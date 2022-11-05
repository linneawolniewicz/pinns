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
    sherpa.Continuous(name='alpha_decay', range=[0.999, 1]),
    sherpa.Continuous(name='lr', range=[3e-5, 3e-8]),
    sherpa.Discrete(name='num_hidden_units', range=[10, 500]),
    sherpa.Discrete(name='num_layers', range=[2, 10])
]

n_trials = 200
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=n_trials)
environment = '/home/linneamw/sadow_lts/personal/linneamw/anaconda3/envs/pinns'
options = '-N 1 -J sherpa -p sadow --account sadow --gres=gpu:NV-RTX2070:1 --mem=12gb -c 1 -t 3-00:00:00'

scheduler = sherpa.schedulers.SLURMScheduler(environment=environment,
                                             submit_options=options,
                                             output_dir=output_dir)

# db_port = sherpa.core._port_finder(8895, 8910)
db_port = 8910

filename = '/home/linneamw/sadow_lts/personal/linneamw/research/pinns/sherpa/parallel/run_pinn.py'

print(f'Running mongodb on port: {db_port}')

results = sherpa.optimize(parameters=parameters,
                          algorithm=algorithm,
                          lower_is_better=True,
                          filename=filename,
                          output_dir=output_dir,
                          scheduler=scheduler,
                          max_concurrent=15,
                          verbose=1,
                          db_port=db_port,
                          mongodb_args={'bind_ip_all':''}
                         )

pkl.dump(results, open(output_dir + '/results.pkl', 'wb'))
