# QuanTran
# TO-DO: Implement CLI for training

from pandas.core.arrays import categorical
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import pickle
import pandas as pd
import os

from syntabtf.processing.cleaning import TabCleaning
from syntabtf.processing.transform import TabTransform
from syntabtf.utils.losses import vae_loss
from syntabtf.utils.training import train
from syntabtf.nn.vae import TVAE
from syntabtf.utils.configs import load_training_config

# LOAD CONFIG
data_dir, preprocessing_cfg, transform_cfg, model_cfg, training_cfg = load_training_config('configs/local/training.yaml')

# LOAD DATA
data_dir = pickle.load(open('data/demo/demo_tabular_data.pkl', 'rb'))
assert type(data_dir) is str or type(data_dir) is pd.DataFrame
df = pd.read_csv(data_dir) if type(data_dir) is str else data_dir
print(df)

# CLEAN DATA
if preprocessing_cfg['clean_data']:
    df = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(df, preprocessing_cfg['min_freq_threshold'], preprocessing_cfg['percentage_to_remove'])
    print(df)

# TRANSFORM DATA
tabtransform = TabTransform(categorical_cols=transform_cfg['categorical_columns'], 
                            max_cols=transform_cfg['max_cols'], 
                            max_gaussian_components=transform_cfg['numerical_settings']['max_gaussian_components'], 
                            gaussian_weight_threshold=transform_cfg['numerical_settings']['gaussian_weight_threshold'],
                            col_name_embedding=transform_cfg['signature_settings']['column_name_embedding'])
tabtransform.fit(df, categorical_norm=transform_cfg['categorical_settings']['normalize'])
data = tabtransform.transform(df)

# DATASET AND DATA LOADER
dataset = TensorDataset(torch.from_numpy(data.astype('float32')))
train_loader = DataLoader(dataset, batch_size=training_cfg['batch_size'], shuffle=training_cfg['shuffle_training'])

# TRAINING
device = torch.device('cuda') if training_cfg['use_gpu'] and torch.cuda.is_available() else torch.device('cpu')
model = TVAE(data_dim = transform_cfg['max_cols'], 
             encoder_hiddens=model_cfg['encoder_hiddens'], 
             decoder_hiddens=model_cfg['decoder_hiddens'], 
             emb_dim=model_cfg['embedding_dim'])
optimizer = Adam(model.parameters(), lr=training_cfg['lr'], weight_decay=training_cfg['l2norm'])

model, hist = train(model, train_loader, training_cfg['epochs'], optimizer, 
                    criterion=vae_loss, 
                    device=device, 
                    val_loader=None, 
                    hist=[], 
                    output_info_list=tabtransform.output_info_list, 
                    recloss_factor=training_cfg['recloss_factor'],
                    optimizer_signature=training_cfg['optimize_signature_features'])

save_dir = training_cfg['save_dir']
model_prefix = training_cfg['model_prefix']
if save_dir is not None:
    model_prefix = 'model' if model_prefix is None else model_prefix
    weights_name = model_prefix + '_weights.py'
    hist_name = model_prefix + '_hist.pkl'
    transform_name = model_prefix + '_transform.pkl'
    config_name = model_prefix + '_config.pkl'

    torch.save(model.state_dict(), os.path.join(save_dir, weights_name))
    pickle.dump(hist, open(os.path.join(save_dir, hist_name), 'wb'))
    pickle.dump(tabtransform, open(os.path.join(save_dir, transform_name), 'wb'))
    pickle.dump(vars(model), open(os.path.join(save_dir, config_name), 'wb'))
