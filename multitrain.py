# Multi-training on source dataset

from pandas.core.arrays import categorical
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import pickle
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import KFold

from syntabtf.processing.cleaning import TabCleaning
from syntabtf.processing.transform import TabTransform
from syntabtf.utils.losses import vae_loss
from syntabtf.utils.training import train
from syntabtf.nn.vae import TVAE
from syntabtf.utils.configs import load_multi_training_config

import gc
import random
import yaml

SEED = 12
random.seed(SEED)

def train_source_domains(src_data_dirs, tar_data_dirs, model, optimizer, device, transform_cfg, **kwargs):
    """ Train source domain dataset given by data_dirs

    Args:
        src_data_dirs ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        device ([type]): [description]
        transform_cfg ([type]): [description]

    Returns:
        Pytorch trained model
    """
    
    fold = kwargs['fold'] if 'fold' in kwargs else None
    
    
    for iter, data_dir in enumerate(src_data_dirs):
        # LOAD DATAFRAME
        print('Process data ', os.path.basename(data_dir))
        df = pd.read_csv(data_dir)
        
        # TEMP: limit the number of columns to test the pipeline
        if len(df.columns) >= 10:
            print('Temporary limit # columns, get first 10 of ', len(df.columns))
            df = df.iloc[:, :10]
        
        # CLEAN DATA
        if preprocessing_cfg['clean_data']:
            df = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(df, 
                                                                                    min_freq_threshold=preprocessing_cfg['min_freq_threshold'], 
                                                                                    pct_to_remove=preprocessing_cfg['percentage_to_remove'])

        # TRANSFORM DATA
        # print(transform_cfg['categorical_columns'], type(transform_cfg['categorical_columns']))
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

        model, hist = train(model, train_loader, training_cfg['epochs'], optimizer, 
                            criterion=vae_loss, 
                            device=device, 
                            val_loader=None, 
                            hist=[], 
                            output_info_list=tabtransform.output_info_list, 
                            recloss_factor=training_cfg['recloss_factor'],
                            optimizer_signature=training_cfg['optimize_signature_features'])
        
        # SAVE MODEL ARTIFACTS
        # save checkpoint
        if training_cfg['save_checkpoint']:
            print('\t Save checkpoint')
            model_prefix = 'model_' + str(iter)
        
            if training_cfg['save_dir'] is not None:
                # Create dest dir to store artifacts
                save_dir = training_cfg['save_dir']
                if fold is not None:
                    save_dir = os.path.join(save_dir, fold)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    
                model_prefix = 'model' if model_prefix is None else model_prefix
                weights_name = model_prefix + '_weights.pt'
                hist_name = model_prefix + '_hist.pkl'
                config_name = model_prefix + '_config.pkl'
                
                torch.save(model.state_dict(), os.path.join(save_dir, weights_name))
                pickle.dump(hist, open(os.path.join(save_dir, hist_name), 'wb'))
                pickle.dump(vars(model), open(os.path.join(save_dir, config_name), 'wb'))
                
        transform_name = model_prefix + '_transform.pkl'
        pickle.dump(tabtransform, open(os.path.join(save_dir, transform_name), 'wb'))
                
        del(df)
        del(tabtransform)
        del(data)
        del(dataset)
        del(train_loader)
        gc.collect()
                
        print('***')

    # save final model
    print('Save final model artifacts')

    # Create dest dir to store artifacts
    save_dir = training_cfg['save_dir']
    if fold is not None:
        save_dir = os.path.join(save_dir, fold)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    # save training info
    training_info = dict(source_data_directory=[str(k) for k in src_data_dirs],
                         target_data_directory=str(tar_data_dirs))
    
    # print('tar_data_dirs: ', type(tar_data_dirs), len(tar_data_dirs))
    # print(np.fromstring(tar_data_dirs))
    
    yaml.dump(training_info, open(os.path.join(save_dir, 'training_info.yaml'), 'w'), default_flow_style=False)

    # save model
    model_prefix = 'model_final' if training_cfg['model_prefix'] is None else training_cfg['model_prefix']
    weights_name = model_prefix + '_weights.py'
    # hist_name = model_prefix + '_hist.pkl'
    # transform_name = model_prefix + '_transform.pkl'
    config_name = model_prefix + '_config.pkl'

    torch.save(model.state_dict(), os.path.join(save_dir, weights_name))
    # pickle.dump(hist, open(os.path.join(save_dir, hist_name), 'wb'))
    # pickle.dump(tabtransform, open(os.path.join(save_dir, transform_name), 'wb'))
    pickle.dump(vars(model), open(os.path.join(save_dir, config_name), 'wb'))
    
    return model


def train_finetune_and_score(src_data_dirs, tar_data_dirs, **kwargs):
    # INIT MODEL AND OPTIMIZER
    device = torch.device('cuda') if training_cfg['use_gpu'] and torch.cuda.is_available() else torch.device('cpu')
    model = TVAE(data_dim = transform_cfg['max_cols'], 
                    encoder_hiddens=model_cfg['encoder_hiddens'], 
                    decoder_hiddens=model_cfg['decoder_hiddens'], 
                    emb_dim=model_cfg['embedding_dim'])
    optimizer = Adam(model.parameters(), lr=training_cfg['lr'], weight_decay=training_cfg['l2norm'])
    
    # train source domains
    src_model = train_source_domains(src_data_dirs=src_data_dirs, 
                                     tar_data_dirs = tar_data_dirs,
                                    model=model, 
                                    optimizer=optimizer, 
                                    device=device, 
                                    transform_cfg=transform_cfg,
                                    **kwargs)
    
    # TODO: fine-tune on small amount of data of target domain
        
    # TODO: calculate score based on SDMetrics
    
    del(src_data_dirs)
    del(tar_data_dirs)
    del(src_model)
    del(model)
    del(optimizer)
    del(device)
    
    gc.collect()
    
    
# LOAD CONFIG
source_domain_dir, target_domain_dir, preprocessing_cfg, transform_cfg, model_cfg, training_cfg = load_multi_training_config('configs/local/multi_training.yaml')

# LOAD DATA
# get all sub dir csv file
sub_dirs = os.listdir(source_domain_dir)

# get all csv file and meta data in sub dir
data_dirs = [[os.path.join(source_domain_dir, _dir, k) for k in os.listdir(os.path.join(source_domain_dir, _dir)) if '.csv' in k] for _dir in sub_dirs]
data_dirs = [item for sublist in data_dirs for item in sublist]
print('All csv file in source domain: ', data_dirs)
# meta_dirs = [[os.path.join(source_domain_dir, _dir, k) for k in os.listdir(os.path.join(source_domain_dir, _dir)) if 'metadata.json' in k] for _dir in sub_dirs]
# meta_dirs = [item for sublist in meta_dirs for item in sublist]

# TRAIN, FINE AND SCORE
if target_domain_dir is not None: # auto split 1-vs-rest as target domain
    print('Target domain dir: ', target_domain_dir)
    target_domain_dir = [[os.path.join(target_domain_dir, _dir, k) for k in os.listdir(os.path.join(target_domain_dir, _dir)) if '.csv' in k] for _dir in sub_dirs]
    target_domain_dir = [item for sublist in target_domain_dir for item in sublist]
    
    train_finetune_and_score(data_dirs, target_domain_dir)
    
else:
    print('Target domain dir is not found, automatically apply 1-vs-rest')
    
    # SPLIT 1-VS-REST (1 DST DOMAIN, REST SRC DOMAINS)
    k_fold = KFold(n_splits=len(data_dirs), shuffle=True, random_state=SEED) # 1-vs-rest

    # Iter through each fold
    for iter_step, (src_indices, tar_index) in enumerate(k_fold.split(data_dirs)):
        print('=====================Iter {}========================'.format(iter_step))
        
        src_data_dirs = np.array(data_dirs)[src_indices]
        tar_data_dirs = np.array(data_dirs)[tar_index][0]
        
        print('* source dirs: ', src_data_dirs)
        print('* destination ', tar_data_dirs)
        
        train_finetune_and_score(src_data_dirs, tar_data_dirs, fold='fold_' + str(iter_step))
        
        print('===================================================')
    
    