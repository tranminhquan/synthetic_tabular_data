# QuanTran
# TO-DO: Implement CLI for training

from pandas.core.arrays import categorical
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import pickle
import pandas as pd
import os

from processing.cleaning import TabCleaning
from processing.transform import TabTransform
from utils.losses import vae_loss
from utils.training import train
from nn.vae import TVAE

# CONFIG ================================
# data config
clean_data = True
# data_dir = 'data/demo/demo_tabular_data.csv'
data_dir = pickle.load(open('data/demo/demo_tabular_data.pkl', 'rb'))

# transform config
max_cols = 50
categorical_columns = ['gender', 'high_spec', 'degree_type', 'work_experience', 'mba_spec', 'placed']
max_guassian_components = 10
gaussian_weight_threshold = 5e-3

# model config
encoder_hiddens = (128, 128)
decoder_hiddens = (128, 128)
embedding_dim = 128

# training config
batch_size = 64
epochs = 300
shuffle_training = True
lr = 1e-3
l2norm = 1e-5
gpu=True
recloss_factor = 1.
save_dir='results/demo' # folder to store weights of model
model_prefix=None # name of weights
# END CONFIG ================================


# LOAD DATA
assert type(data_dir) is str or type(data_dir) is pd.DataFrame
df = pd.read_csv(data_dir) if type(data_dir) is str else data_dir
print(df)

# CLEAN DATA
if clean_data:
    df = TabCleaning().clean(df)
    print(df)

# TRANSFORM DATA
tabtransform = TabTransform(categorical_cols=categorical_columns, 
                            max_cols=max_cols, 
                            max_guassian_components=max_guassian_components, 
                            gaussian_weight_threshold=gaussian_weight_threshold)
tabtransform.fit(df)
data = tabtransform.transform(df)

# DATASET AND DATA LOADER
dataset = TensorDataset(torch.from_numpy(data.astype('float32')))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_training)

# TRAINING
device = torch.device('cuda') if gpu and torch.cuda.is_available() else torch.device('cpu')
model = TVAE(data_dim = max_cols, encoder_hiddens=encoder_hiddens, decoder_hiddens=decoder_hiddens, emb_dim=embedding_dim)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2norm)

model, hist = train(model, train_loader, epochs, optimizer, criterion=vae_loss, device=device, val_loader=None, hist=[], 
        output_info_list=tabtransform.output_info_list, recloss_factor=recloss_factor)

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
