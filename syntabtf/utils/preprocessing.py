import os
import pickle
import pandas as pd
from syntabtf.processing.cleaning import TabCleaning
from syntabtf.processing.transform import TabTransform

def temp_filter_data(df, max_cols=None):
     # TEMP: limit the number of columns to test the pipeline
    if max_cols is None:
        return df
    
    df = df.iloc[:, :max_cols]
        
    return df

def load_preprocessing_file(data_dir, cleaning_prefix='clean_', transformed_data_prefix='transformed_data', transform_instance_prefix='tabtransform'):
    
    # check if clean version of data exist
    clean_path = os.path.join(os.path.dirname(data_dir), cleaning_prefix + os.path.basename(data_dir))
    transformed_data_path = os.path.join(os.path.dirname(data_dir), transformed_data_prefix + '.pkl')
    transform_instance_path = os.path.join(os.path.dirname(data_dir), transform_instance_prefix + '.pkl')
    
    
    # clean df
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        clean_df = True
    else:
        df = pd.read_csv(data_dir)
        clean_df = False
        
    # transformed data
    if os.path.exists(transformed_data_path):
        transformed_data = pickle.load(open(transformed_data_path, 'rb'))
    else:
        transformed_data = None
        
    # transform instance (TabTransform)
    if os.path.exists(transform_instance_path):
        transform_instance = pickle.load(open(transform_instance_path, 'rb'))
    else:
        transform_instance = None
        
    return df, clean_df, transformed_data, transform_instance
        
def preprocess_clean(data_dirs, preprocessing_cfg):
    """Apply TabCleaning to all data, save clean_df in the same directory with prefix 'clean_'

    Args:
        data_dirs ([type]): [description]
        preprocessing_cfg ([type]): [description]
    """
    
    for _dir in data_dirs:
        df = pd.read_csv(_dir)
        
        if preprocessing_cfg['clean_data']:
            df = TabCleaning(exclude=preprocessing_cfg['clean_exclude_columns']).clean(df, 
                                                                                    min_freq_threshold=preprocessing_cfg['min_freq_threshold'], 
                                                                                    pct_to_remove=preprocessing_cfg['percentage_to_remove'])
            
            # save
            prefix = 'clean_'
            save_path = os.path.dirname(_dir)
            save_path = os.path.join(save_path, prefix + os.path.basename(_dir))
            df.to_csv(save_path, index=False)
            
    return prefix

def preprocess_transform(data_dirs, transform_cfg, cleaning_prefix):
    # find the path of clean data if any
    
    for _dir in data_dirs:
        try:
            clean_path = os.path.join(os.path.dirname(_dir), cleaning_prefix + os.path.basename(_dir))
            df = pd.read_csv(clean_path)
        except:
            df = pd.read_csv(_dir)
        
        # apply TabTransform
        tabtransform = TabTransform(categorical_cols=transform_cfg['categorical_columns'], 
                                max_cols=transform_cfg['max_cols'], 
                                max_gaussian_components=transform_cfg['numerical_settings']['max_gaussian_components'], 
                                gaussian_weight_threshold=transform_cfg['numerical_settings']['gaussian_weight_threshold'],
                                col_name_embedding=transform_cfg['signature_settings']['column_name_embedding'])
        tabtransform.fit(df, categorical_norm=transform_cfg['categorical_settings']['normalize'])
        data = tabtransform.transform(df)
        
        # save both fitted tabtransform instance, and data after transforming
        instance_path = os.path.join(os.path.dirname(_dir), 'tabtransform.pkl')
        data_path = os.path.join(os.path.dirname(_dir), 'transformed_data.pkl')
        
        pickle.dump(tabtransform, open(instance_path, 'wb'))
        pickle.dump(data, open(data_path, 'wb'))
        
    return 'tabtransform.pkl', 'transformed_data.pkl'