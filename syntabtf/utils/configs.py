import yaml

def load_training_config(config):
    assert type(config) is str or type(config) is dict
    
    config = yaml.safe_load(open(config)) if type(config) is str else config
    clean_data = config['clean_data']
    df_path = config['df_path']
    transform = config['transform']
    model = config['model']
    training = config['training']
    
    return df_path, clean_data, transform, model, training
    
def load_generating_config(config):
    assert type(config) is str or type(config) is dict
    
    config = yaml.safe_load(open(config)) if type(config) is str else config

    return config