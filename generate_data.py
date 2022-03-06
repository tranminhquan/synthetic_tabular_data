# QuanTran
# TO-DO: Implement CLI for generating sample

from syntabtf.processing.transform import TabTransform
from syntabtf.nn.vae import TVAE
from syntabtf.utils.configs import load_generating_config
import pickle
import os
import torch

# LOAD CONFIG
cfg = load_generating_config('configs/local/generating.yaml')

save_dir = cfg['save_dir']
model_prefix = cfg['model_prefix']
n_samples = cfg['n_samples']
batch_size = cfg['batch_size']
use_sigmoid = cfg['use_sigmoid']

# LOAD MODEL AND TRANFORM
config_path = os.path.join(save_dir, model_prefix + '_config.pkl')
transform_path = os.path.join(save_dir, model_prefix + '_transform.pkl')
weights_path = os.path.join(save_dir, model_prefix + '_weights.py')

# load model
config = pickle.load(open(config_path, 'rb'))
model = TVAE(data_dim=config['data_dim'], 
            encoder_hiddens=config['encoder_hiddens'],
            decoder_hiddens=config['decoder_hiddens'],
            emb_dim=config['emb_dim'])

model.load_state_dict(torch.load(weights_path))

# load transform
tabtransform = pickle.load(open(transform_path, 'rb'))

# GENERATE
inv_data, sigmas = model.cpu().sample(n_samples=n_samples, batch_size=batch_size)
inv_df = tabtransform.inverse_transform(inv_data, sigmas, sigmoid=use_sigmoid)


categorical_columns = ['gender', 'high_spec', 'degree_type', 'work_experience', 'mba_spec', 'placed']

# print synthetic data
print(inv_df)

# show the distribution of categorical values
for cate_col in categorical_columns:
    print(inv_df[cate_col].value_counts())
    print('---')