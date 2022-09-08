import os
import sys
import pickle as pkl
import sherpa
import sherpa.schedulers
import sherpa.algorithms
import tensorflow as tf

# Set-up output directory
output_dir = '/home/linneamw/sadow_lts/personal/linneamw/pinns/sherpa'
if os.path.isdir(output_dir):
    print('Warning: Overwriting directory {}'.format(output_dir))
    import shutil
    shutil.rmtree(output_dir)
else:
    print(f'Writing to directory {output_dir}')

# Sherpa parameters
parameters = [
    sherpa.Ordinal(name='beta', range=[5, 10, 15, 20, 30]),
    sherpa.Continuous(name='lr', range=[3e-5, 3e-1], scale='log'),
    sherpa.Ordinal(name='batchsize', range=[256, 512, 1032, 2048]),
    sherpa.Ordinal(name='boundary_batchsize', range=[64, 128, 256]),
    sherpa.Ordinal(name='num_hidden_units', range=[100, 500, 1000]),
    sherpa.Choice(name='activation', range=['relu', 'tanh'])
]

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=50)
env = '/home/linneamw/sadow_lts/personal/linneamw/anaconda3/envs/pinns' # to-do: check this is right env path
opt = '-N 1 -J s_tbdxa -p gpu --gres=gpu:1 --constraint="volta" --mem=32gb -c 8 -t 2-23:00:00' # to-do: figure out what value to make this

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
                          mongodb_args={'bind_ip_all': ''}) # to-do: what are verbose, db_port, mongodb_args, and max_concurrent??

pkl.dump(results, open(output_dir + '/results.pkl', 'wb'))
