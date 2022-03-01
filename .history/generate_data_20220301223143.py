# QuanTran
# TO-DO: Implement CLI for generating sample

from processing.transform import TabTransform
from nn.vae import TVAE
import pickle
import os
import torch

# CONFIG =============================
save_dir = 'results/demo_v2'
model_prefix = 'model'

n_samples = 200
batch_size = 64
use_sigmoid = False # in testing, seting sigmoid to True in generation gives the incorrect results!
# END CONFIG =============================

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

for cate_col in categorical_columns:
    print(inv_df, inv_df[cate_col].value_counts())
    print('---')