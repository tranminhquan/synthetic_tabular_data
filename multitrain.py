# Multi-training on source dataset

# from curses import meta
# from importlib_metadata import metadata
# from pandas.core.arrays import categorical
import os
import numpy as np

from sklearn.model_selection import KFold

from syntabtf.utils.configs import load_multi_training_config
from syntabtf.utils.tf import train_finetune_and_score
from syntabtf.utils.preprocessing import preprocess_clean, preprocess_transform

import random


global SEED
SEED = 12
random.seed(SEED)
    
# LOAD CONFIG
source_domain_dir, target_domain_dir, n_folds,\
    preprocessing_cfg, transform_cfg, \
        model_cfg, training_cfg, finetuning_cfg = load_multi_training_config('configs/local/multi_training.yaml')

# LOAD DATA
# get all sub dir csv file
sub_dirs = os.listdir(source_domain_dir)

# get all csv file and meta data in sub dir
data_dirs = [[os.path.join(source_domain_dir, _dir, k) for k in os.listdir(os.path.join(source_domain_dir, _dir)) if '.csv' in k and 'clean_' not in k] for _dir in sub_dirs]
data_dirs = [item for sublist in data_dirs for item in sublist]
print('All csv file in source domain: ', data_dirs)

meta_dirs = [[os.path.join(source_domain_dir, _dir, k) for k in os.listdir(os.path.join(source_domain_dir, _dir)) if 'metadata.json' in k] for _dir in sub_dirs]
meta_dirs = [item for sublist in meta_dirs for item in sublist]
print('All json file in source domain: ', meta_dirs)

# TRAIN, FINE AND SCORE
if target_domain_dir is not None:
    print('Target domain dir: ', target_domain_dir)
    # csv file
    target_domain_dir = [[os.path.join(target_domain_dir, _dir, k) \
        for k in os.listdir(os.path.join(target_domain_dir, _dir)) if '.csv' in k and 'clean_' not in k] for _dir in sub_dirs]
    target_domain_dir = [item for sublist in target_domain_dir \
        for item in sublist]
    
    # metadata file
    target_meta_dir = [[os.path.join(target_domain_dir, _dir, k) \
        for k in os.listdir(os.path.join(target_domain_dir, _dir)) if '.csv' in k and 'clean_' not in k] for _dir in sub_dirs]
    target_meta_dir = [item for sublist in target_domain_dir \
        for item in sublist]
    
    train_finetune_and_score(data_dirs, meta_dirs, target_domain_dir, target_meta_dir, preprocessing_cfg, transform_cfg, model_cfg, training_cfg, finetuning_cfg)
    
else: # auto split 1-vs-rest as target domain
    print('Target domain dir is not found, automatically apply 1-vs-rest')
    
    # # Prepare TabCleaning and TabTransform
    # print('Pre-process cleaning ...')
    # cleaning_prefix = preprocess_clean(data_dirs, preprocessing_cfg)
    # print('Pre-process transforming ...')
    # instance_name, data_name = preprocess_transform(data_dirs, transform_cfg, cleaning_prefix='clean_')
    
    # SPLIT 1-VS-REST (1 DST DOMAIN, REST SRC DOMAINS)
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED) # 1-vs-rest

    # Iter through each fold
    for iter_step, (src_indices, tar_index) in enumerate(k_fold.split(data_dirs)):
        print('=====================Iter {}========================'.format(iter_step))
        
        src_data_dirs = np.array(data_dirs)[src_indices]
        tar_data_dirs = np.array(data_dirs)[tar_index]
        
        src_meta_dirs = np.array(meta_dirs)[src_indices]
        tar_meta_dirs = np.array(meta_dirs)[tar_index]
        
        print('* source dirs: ', src_data_dirs)
        print('* destination ', tar_data_dirs)
        
        train_finetune_and_score(src_data_dirs, 
                                 src_meta_dirs, 
                                 tar_data_dirs, 
                                 tar_meta_dirs,
                                 preprocessing_cfg, 
                                 transform_cfg, 
                                 model_cfg,
                                 training_cfg, 
                                 finetuning_cfg, 
                                 fold='fold_' + str(iter_step))
        
        print('===================================================')
    
    