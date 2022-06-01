
import gc
import pickle
import yaml
import json
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from syntabtf.processing.cleaning import TabCleaning
from syntabtf.processing.transform import TabTransform
from syntabtf.utils.losses import vae_loss
from syntabtf.nn.vae import TVAE
from syntabtf.utils.training import train
from syntabtf.utils.preprocessing import temp_filter_data, load_preprocessing_file
from syntabtf.utils.eval import scoring
from torch.optim import Adam

from sklearn.model_selection import train_test_split

def train_finetune_and_score(src_data_dirs, src_meta_dirs, tar_data_dirs, tar_meta_dirs, preprocessing_cfg, transform_cfg, model_cfg, training_cfg, finetuning_cfg, **kwargs):
    # INIT MODEL AND OPTIMIZER
    device = torch.device('cuda') if training_cfg['use_gpu'] and torch.cuda.is_available() else torch.device('cpu')
    model = TVAE(data_dim = transform_cfg['max_cols'], 
                    encoder_hiddens=model_cfg['encoder_hiddens'], 
                    decoder_hiddens=model_cfg['decoder_hiddens'], 
                    emb_dim=model_cfg['embedding_dim'])
    optimizer = Adam(model.parameters(), lr=training_cfg['lr'], weight_decay=training_cfg['l2norm'])
    
    # train source domains
    print('==TRAIN SOURCE DOMAINS==')
    src_model = train_source_domains(src_data_dirs=src_data_dirs, 
                                     tar_data_dirs = tar_data_dirs,
                                    model=model, 
                                    optimizer=optimizer, 
                                    device=device,
                                    preprocessing_cfg=preprocessing_cfg, 
                                    transform_cfg=transform_cfg,
                                    training_cfg=training_cfg,
                                    model_cfg=model_cfg,
                                    **kwargs)
    
    # fine-tune on small amount of data of target domains (loop through single target domains)
    
    print('==FINE-TUNE TARGET DATASET==')
    for i, tar_data_dir in enumerate(tar_data_dirs):
        print('- {}: {} --------'.format(i, tar_data_dir))
        
        finetuned_model, synthetic_data, scratch_model, scratch_synthetic_data = finetune_target_domains(tar_data_dir,
                                src_model,
                                finetuning_cfg,
                                training_cfg,
                                preprocessing_cfg,
                                transform_cfg,
                                model_cfg,
                                optimizer,
                                **kwargs)
    
        # real_tar_data = pd.read_csv(tar_data_dir)
        print('==SCORING WITH SDMETRICS==')
        real_tar_data, real_clean_df, _, _= load_preprocessing_file(tar_data_dir)

        meta_data = json.load(open(tar_meta_dirs[i]))
        df_name = str(tar_data_dir.split('/')[-2])
        print('df_name', df_name)
        print('metadata: ', meta_data['tables'].keys())
        meta_data = meta_data['tables'][df_name]
    
        # filter due to limited memory
        real_tar_data = temp_filter_data(real_tar_data)
        synthetic_data = temp_filter_data(synthetic_data)
        scratch_synthetic_data = temp_filter_data(scratch_synthetic_data)
    
        # cleaning real and synthetic data
        print('- Cleaning real and synthetic data')
        if real_clean_df is False:
            real_tar_data = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(real_tar_data, 
                                                                                    min_freq_threshold=preprocessing_cfg['min_freq_threshold'], 
                                                                                    pct_to_remove=preprocessing_cfg['percentage_to_remove'])
        
        synthetic_data = synthetic_data[real_tar_data.columns]
        scratch_synthetic_data = scratch_synthetic_data[real_tar_data.columns]
    
        print('- Correct dtype of synthetic data')    
        # correct dtype
        for col in synthetic_data.columns:
            if real_tar_data[col].dtypes == int: # first convert corresponding column in synthetic data into float
                synthetic_data[col] = scratch_synthetic_data[col] = synthetic_data[col].astype(float).astype(real_tar_data[col].dtypes)
            else:
                synthetic_data[col] = scratch_synthetic_data[col] = synthetic_data[col].astype(real_tar_data[col].dtypes)
        
        # check
        print('real data columns: ', real_tar_data.columns)
        print('synthetic data columns: ', synthetic_data.columns)
        print('scratch synthetic data columns: ', scratch_synthetic_data.columns)
    
        # print('Check columns and dtypes: ')
        # for col in synthetic_data.columns:
        #     print(real_tar_data[col].name, ': ' , real_tar_data[col].dtypes, ' | ', synthetic_data[col].name, ': ', synthetic_data[col].dtypes)
        
        # compare real vs fine-tune data
        print('- Scoring ...')
        
        #   scoring between real and fine-tune model
        score_finetune_df = scoring(real_tar_data, synthetic_data, meta_data)
        #   scoring between real and training-from-scratch model
        score_scratch_df = scoring(real_tar_data, scratch_synthetic_data, meta_data)
        
        # Save
        if training_cfg['save_dir'] is not None:
            # Create dest dir to store artifacts
            save_dir = training_cfg['save_dir']
            if 'fold' in kwargs:
                save_dir = os.path.join(save_dir, kwargs['fold'])
        
        file_name = os.path.basename(os.path.splitext(tar_data_dir)[0])
        file_path = os.path.join(save_dir, file_name)
        
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        
        score_finetune_df.to_csv(os.path.join(file_path, 'scoring_real_vs_finetune.csv'))
        score_scratch_df.to_csv(os.path.join(file_path, 'scoring_real_vs_scratch.csv'))
        
        
        del(tar_data_dirs)
        del(model)
        del(synthetic_data)
        del(scratch_synthetic_data)
        
        del(score_finetune_df)
        del(score_scratch_df)
        gc.collect()
        
    del(src_data_dirs)
    del(src_model)    
    del(optimizer)
    del(device)
    
    gc.collect()
    
    return None

def train_source_domains(src_data_dirs, tar_data_dirs, model, optimizer, device, preprocessing_cfg, transform_cfg, training_cfg, model_cfg, **kwargs):
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
    print('FOLD: ', fold)
    
    # loop through each source domain dataset
    for iter, data_dir in enumerate(src_data_dirs):
        # LOAD DATAFRAME
        print('Process data ', os.path.basename(data_dir))
        df, clean_df, transformed_data, transform_instance = load_preprocessing_file(data_dir)
        
        # TEMP: limit the number of columns to test the pipeline
        df = temp_filter_data(df)
        
        model, hist, tabtransform = train_through_pipeline(df, preprocessing_cfg, transform_cfg, 
                                                           model_cfg, training_cfg, model, optimizer, device, 
                                                           clean_df, transform_instance, transformed_data)
        
        # SAVE MODEL ARTIFACTS
        # save checkpoint
        model_prefix = 'model_' + str(iter)
        save_dir = training_cfg['save_dir']
        
        # Create dest dir to store artifacts
        if fold is not None:
            save_dir = os.path.join(save_dir, fold)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        if training_cfg['save_checkpoint']:
            print('\t Save checkpoint')
            
            if training_cfg['save_dir'] is not None:
                model_prefix = 'model' if model_prefix is None else model_prefix
                weights_name = model_prefix + '_weights.pt'
                hist_name = model_prefix + '_hist.pkl'
                config_name = model_prefix + '_config.pkl'
                
                torch.save(model.state_dict(), os.path.join(save_dir, weights_name))
                pickle.dump(hist, open(os.path.join(save_dir, hist_name), 'wb'))
                pickle.dump(vars(model), open(os.path.join(save_dir, config_name), 'wb'))
                
            transform_name = model_prefix + '_transform.pkl'
            # print('SAVE DIR: ', save_dir)
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

def finetune_target_domains(tar_data_dirs, src_model, finetuning_cfg, training_cfg, preprocessing_cfg, transform_cfg, model_cfg, optimizer, **kwargs):
    """Finetune and train from scratch with the target data

    Args:
        tar_data_dirs ([type]): [description]
        src_model ([type]): [description]
        finetuning_cfg ([type]): [description]
        training_cfg ([type]): [description]
        preprocessing_cfg ([type]): [description]
        transform_cfg ([type]): [description]
        optimizer ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    fold = kwargs['fold'] if 'fold' in kwargs else None
    device = kwargs['device'] if 'device' in kwargs else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # load data
    # clean_path = os.path.join(os.path.basename(tar_data_dirs), 'clean_' + os.path.basename(tar_data_dirs))
    # df = pd.read_csv(tar_data_dirs)
    df, clean_df, transformed_data, transform_instance = load_preprocessing_file(tar_data_dirs)
    
    # split to get the small data for finetuning
    val_df, finetune_df = train_test_split(df, test_size=finetuning_cfg['finetune_size'], random_state=12)
    val_indices, finetune_indices = val_df.index.tolist(), finetune_df.index.tolist() # store and save indices for later filtering
    
    # fine-tune with finetune_df
    finetuned_model, finetune_hist, finetune_tabstransform = train_through_pipeline(finetune_df, 
                                                                                    preprocessing_cfg, 
                                                                                    transform_cfg, 
                                                                                    model_cfg,
                                                                                    training_cfg, 
                                                                                    src_model, 
                                                                                    optimizer, 
                                                                                    device,
                                                                                    clean_df, 
                                                                                    None, # subset of df, re-calculate the TabTransform
                                                                                    None) # subset of df, re-calculate the TabTransform
    
    # training from scratch with whole target data
    scratch_model, scratch_hist, scratch_tabstransform = train_through_pipeline(df, 
                                                                                preprocessing_cfg, 
                                                                                transform_cfg, 
                                                                                model_cfg, 
                                                                                training_cfg,
                                                                                None, # train from scratch, set model to None
                                                                                optimizer, 
                                                                                device,
                                                                                clean_df,
                                                                                transform_instance,
                                                                                transformed_data)
    
    # generate and save synthetic data
    n_samples = len(df)
    
    #   generate from fine-tune model
    inv_finetune_data, finetune_sigmas = finetuned_model.cpu().sample(n_samples=n_samples, batch_size=training_cfg['batch_size'])
    inv_finetune_df = finetune_tabstransform.inverse_transform(inv_finetune_data, finetune_sigmas, sigmoid=False)
    
    #   generate from training-from-scratch model
    inv_scratch_data, scratch_sigmas = scratch_model.cpu().sample(n_samples=n_samples, batch_size=training_cfg['batch_size'])
    inv_scratch_df = scratch_tabstransform.inverse_transform(inv_scratch_data, scratch_sigmas, sigmoid=False)

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
    
    # save fine-tune model
    model_finetune_prefix = 'model_final_finetune' if training_cfg['model_prefix'] is None else training_cfg['model_prefix'] + '_finetune'
    weights_name = model_finetune_prefix + '_weights.py'
    hist_name = model_finetune_prefix + '_hist.pkl'
    transform_name = model_finetune_prefix + '_transform.pkl'
    config_name = model_finetune_prefix + '_config.pkl'
    synthetic_path = model_finetune_prefix + '_synthetic_data.csv'

    torch.save(finetuned_model.state_dict(), os.path.join(save_dir, weights_name))
    pickle.dump(finetune_hist, open(os.path.join(save_dir, hist_name), 'wb'))
    pickle.dump(finetune_tabstransform, open(os.path.join(save_dir, transform_name), 'wb'))
    pickle.dump(vars(finetuned_model), open(os.path.join(save_dir, config_name), 'wb'))
    inv_finetune_df.to_csv(os.path.join(save_dir, synthetic_path))
    
    # save training-from-scratch model
    model_scratch_prefix = 'model_final_scratch' if training_cfg['model_prefix'] is None else training_cfg['model_prefix'] + '_scratch'
    weights_name = model_scratch_prefix + '_weights.py'
    hist_name = model_scratch_prefix + '_hist.pkl'
    transform_name = model_scratch_prefix + '_transform.pkl'
    config_name = model_scratch_prefix + '_config.pkl'
    synthetic_path = model_scratch_prefix + '_synthetic_data.csv'

    torch.save(scratch_model.state_dict(), os.path.join(save_dir, weights_name))
    pickle.dump(scratch_hist, open(os.path.join(save_dir, hist_name), 'wb'))
    pickle.dump(scratch_tabstransform, open(os.path.join(save_dir, transform_name), 'wb'))
    pickle.dump(vars(scratch_model), open(os.path.join(save_dir, config_name), 'wb'))
    inv_scratch_df.to_csv(os.path.join(save_dir, synthetic_path))
    
    gc.collect()
    
    return finetuned_model, inv_finetune_df, scratch_model, inv_scratch_df

def train_through_pipeline(df, preprocessing_cfg, transform_cfg, model_cfg, training_cfg, model, optimizer, device, clean_df=False, transform_object=None, transformed_data_object=None):
    """Training with conventional pipeline: Cleaning -> Transform - Train

    Args:
        df ([type]): [description]
        preprocessing_cfg ([type]): [description]
        transform_cfg ([type]): [description]
        model ([type]): [description]
    """
    
    
     # CLEAN DATA
    if preprocessing_cfg['clean_data'] and clean_df is False:
        print('*No existing clean_df found, apply TabCleaning')
        df = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(df, 
                                                                                min_freq_threshold=preprocessing_cfg['min_freq_threshold'], 
                                                                                pct_to_remove=preprocessing_cfg['percentage_to_remove'])
   
        
    print('Dataframe dtypes: ', df.dtypes)

    # TRANSFORM DATA
    # print(transform_cfg['categorical_columns'], type(transform_cfg['categorical_columns']))
    if transformed_data_object is None and transform_object is None:
        print('*No existing tabtransform found, auto apply TabTransform')
        tabtransform = TabTransform(categorical_cols=transform_cfg['categorical_columns'], 
                                    max_cols=transform_cfg['max_cols'], 
                                    max_gaussian_components=transform_cfg['numerical_settings']['max_gaussian_components'], 
                                    gaussian_weight_threshold=transform_cfg['numerical_settings']['gaussian_weight_threshold'],
                                    col_name_embedding=transform_cfg['signature_settings']['column_name_embedding'])
        tabtransform.fit(df, categorical_norm=transform_cfg['categorical_settings']['normalize'])
        data = tabtransform.transform(df)
        
    elif transformed_data_object is not None:
        print('*Use pre-transformed data object')
        tabtransform = transform_object
        data = transformed_data_object
        
    else:
        print('*Apply pre-transformed data instance')
        tabtransform = transformed_data_object
        data = transform_object.transform(df)
    
    # DATASET AND DATA LOADER
    dataset = TensorDataset(torch.from_numpy(data.astype('float32')))
    train_loader = DataLoader(dataset, batch_size=training_cfg['batch_size'], shuffle=training_cfg['shuffle_training'])
    
    if model is None: # if model is not given, init and train from scratch
        model = TVAE(data_dim = transform_cfg['max_cols'], 
                    encoder_hiddens=model_cfg['encoder_hiddens'], 
                    decoder_hiddens=model_cfg['decoder_hiddens'], 
                    emb_dim=model_cfg['embedding_dim'])

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

