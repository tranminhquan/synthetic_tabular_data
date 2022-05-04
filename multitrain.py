# Multi-training on source dataset

from pandas.core.arrays import categorical
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import pickle
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import KFold, train_test_split

from syntabtf.processing.cleaning import TabCleaning
from syntabtf.processing.transform import TabTransform
from syntabtf.utils.losses import vae_loss
from syntabtf.utils.training import train
from syntabtf.nn.vae import TVAE
from syntabtf.utils.configs import load_multi_training_config

import sdmetrics

import gc
import random
import yaml

SEED = 12
random.seed(SEED)

def temp_filter_data(df):
     # TEMP: limit the number of columns to test the pipeline
    if len(df.columns) >= 10:
        print('Temporary limit # columns, get first 10 of ', len(df.columns))
        df = df.iloc[:, :10]
        
    return df

def train_through_pipeline(df, preprocessing_cfg, transform_cfg, model, optimizer, device):
    """Training with conventional pipeline: Cleaning -> Transform - Train

    Args:
        df ([type]): [description]
        preprocessing_cfg ([type]): [description]
        transform_cfg ([type]): [description]
        model ([type]): [description]
    """
    
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
    
    del(data)
    del(dataset)
    del(train_loader)
    gc.collect()
    
    return model, hist, tabtransform

def train_source_domains(src_data_dirs, tar_data_dirs, model, optimizer, device, transform_cfg, training_cfg, **kwargs):
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
    
    # loop through each source domain dataset
    for iter, data_dir in enumerate(src_data_dirs):
        # LOAD DATAFRAME
        print('Process data ', os.path.basename(data_dir))
        df = pd.read_csv(data_dir)
        
        # TEMP: limit the number of columns to test the pipeline
        df = temp_filter_data(df)
            
        model, hist, tabtransform = train_through_pipeline(df, preprocessing_cfg, transform_cfg, model, optimizer, device)
        
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

def finetune_target_domains(tar_data_dirs, src_model, finetuning_cfg, training_cfg, preprocessing_cfg, transform_cfg, optimizer, **kwargs):
    
    fold = kwargs['fold'] if 'fold' in kwargs else None
    device = kwargs['device'] if 'device' in kwargs else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # load data
    df = pd.read_csv(tar_data_dirs)
    
    # split to get the small data for finetuning
    val_df, finetune_df = train_test_split(df, test_size=finetuning_cfg['finetune_size'], random_state=SEED)
    val_indices, finetune_indices = val_df.index.tolist(), finetune_df.index.tolist() # store and save indices for later filtering
    
    finetuned_model, finetune_hist, finetune_tabstransform = train_through_pipeline(df, preprocessing_cfg, transform_cfg, src_model, optimizer, device)
    
    # generate and save synthetic data
    # GENERATE
    n_samples = len(df)
    inv_data, sigmas = finetuned_model.cpu().sample(n_samples=n_samples, batch_size=training_cfg['batch_size'])
    inv_df = finetune_tabstransform.inverse_transform(inv_data, sigmas, sigmoid=False)

    # save training info
    finetuning_info = dict(target_data_directory=str(tar_data_dirs),
                           finetuning_size=finetuning_cfg['finetune_size'],
                           validation_indices=val_indices,
                           finetuning_indices=finetune_indices)
    # Create dest dir to store artifacts
    save_dir = training_cfg['save_dir']
    if fold is not None:
        save_dir = os.path.join(save_dir, fold)
    yaml.dump(finetuning_info, open(os.path.join(save_dir, 'finetuning_info.yaml'), 'w'), default_flow_style=False)
    
    # save model
    model_prefix = 'model_final_finetune' if training_cfg['model_prefix'] is None else training_cfg['model_prefix'] + '_finetune'
    weights_name = model_prefix + '_weights.py'
    hist_name = model_prefix + '_hist.pkl'
    transform_name = model_prefix + '_transform.pkl'
    config_name = model_prefix + '_config.pkl'
    synthetic_path = model_prefix + '_synthetic_data.csv'

    torch.save(finetuned_model.state_dict(), os.path.join(save_dir, weights_name))
    pickle.dump(finetune_hist, open(os.path.join(save_dir, hist_name), 'wb'))
    pickle.dump(finetune_tabstransform, open(os.path.join(save_dir, transform_name), 'wb'))
    pickle.dump(vars(finetuned_model), open(os.path.join(save_dir, config_name), 'wb'))
    inv_df.to_csv(os.path.join(save_dir, synthetic_path))
    
    gc.collect()
    
    return finetuned_model, inv_df

def scoring(real_df, synthetic_df):
    assert type(real_df) is str or type(real_df) is pd.DataFrame
    assert type(synthetic_df) is str or type(synthetic_df) is pd.DataFrame
    
    # real_df = pd.read_csv(real_df) if type(real_df) is str else real_df
    # synthetic_df = pd.read_csv(synthetic_df) if type(synthetic_df) is str else synthetic_df
    
    # scoring single table
    metrics = sdmetrics.single_table.SingleTableMetric.get_subclasses()
    vsreal_scoring_df = sdmetrics.compute_metrics(metrics, real_df, synthetic_df, metadata=None)
    
    
    print('Real vs Finetune synthetic data: ', vsreal_scoring_df)
    
    # TODO: compare between real data and finetuned synthetic data
    # vssynt_scoring_df = 
    
    # compare between original synthetic data vs finetuned synthetic data
    
    return vsreal_scoring_df

def train_finetune_and_score(src_data_dirs, tar_data_dirs, transform_cfg, training_cfg, finetuning_cfg, **kwargs):
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
                                    training_cfg=training_cfg,
                                    **kwargs)
    
    # TODO: fine-tune on small amount of data of target domain
    finetuned_model, synthetic_data = finetune_target_domains(tar_data_dirs,
                            src_model,
                            finetuning_cfg,
                            training_cfg,
                            preprocessing_cfg,
                            transform_cfg,
                            optimizer,
                            **kwargs)
    
    # TODO: calculate score based on SDMetrics
    real_tar_data = pd.read_csv(tar_data_dirs)
    
    # filter due to limited memory
    real_tar_data = temp_filter_data(real_tar_data)
    synthetic_data = temp_filter_data(synthetic_data)
    
    # cleaning real and synthetic data
    print('- Cleaning real and synthetic data')
    real_tar_data = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(real_tar_data, 
                                                                                min_freq_threshold=preprocessing_cfg['min_freq_threshold'], 
                                                                                pct_to_remove=preprocessing_cfg['percentage_to_remove'])
    
    synthetic_data = synthetic_data[real_tar_data.columns]
    
    print('- Correct dtype of synthetic data')    
    # correct dtype
    for col in synthetic_data.columns:
        if real_tar_data[col].dtypes == int: # first convert corresponding column in synthetic data into float
            synthetic_data[col] = synthetic_data[col].astype(float).astype(real_tar_data[col].dtypes)
        else:
            synthetic_data[col] = synthetic_data[col].astype(real_tar_data[col].dtypes)
        
    # check
    print('real data columns: ', real_tar_data.columns)
    print('synthetic data columns: ', synthetic_data.columns)
    
    # print('Check columns and dtypes: ')
    # for col in synthetic_data.columns:
    #     print(real_tar_data[col].name, ': ' , real_tar_data[col].dtypes, ' | ', synthetic_data[col].name, ': ', synthetic_data[col].dtypes)
    
    # compare real vs fine-tune data
    print('- Scoring ...')
    score_df = scoring(real_tar_data, synthetic_data)
    if training_cfg['save_dir'] is not None:
        # Create dest dir to store artifacts
        save_dir = training_cfg['save_dir']
        if 'fold' in kwargs:
            save_dir = os.path.join(save_dir, kwargs['fold'])
            
    score_df.to_csv(os.path.join(save_dir, 'finetuning_scoring.csv'))
    
    # TODO: compare between real and original synthetic data
    
    del(src_data_dirs)
    del(tar_data_dirs)
    del(src_model)
    del(model)
    del(optimizer)
    del(device)
    
    gc.collect()
    
    
# LOAD CONFIG
source_domain_dir, target_domain_dir, preprocessing_cfg, transform_cfg, model_cfg, training_cfg, finetuning_cfg = load_multi_training_config('configs/local/multi_training.yaml')

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
if target_domain_dir is not None:
    print('Target domain dir: ', target_domain_dir)
    target_domain_dir = [[os.path.join(target_domain_dir, _dir, k) for k in os.listdir(os.path.join(target_domain_dir, _dir)) if '.csv' in k] for _dir in sub_dirs]
    target_domain_dir = [item for sublist in target_domain_dir for item in sublist]
    
    train_finetune_and_score(data_dirs, target_domain_dir, transform_cfg, training_cfg, finetuning_cfg)
    
else: # auto split 1-vs-rest as target domain
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
        
        train_finetune_and_score(src_data_dirs, tar_data_dirs, transform_cfg, training_cfg, finetuning_cfg, fold='fold_' + str(iter_step))
        
        print('===================================================')
    
    