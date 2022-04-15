import yaml

def load_training_config(config):
    assert type(config) is str or type(config) is dict
    
    config = yaml.safe_load(open(config)) if type(config) is str else config
    preprocessing = config['preprocessing']
    df_path = config['df_path']
    transform = config['transform']
    model = config['model']
    training = config['training']
    
    return df_path, preprocessing, transform, model, training
    
def load_multi_training_config(config):
    assert type(config) is str or type(config) is dict
    
    config = yaml.safe_load(open(config)) if type(config) is str else config
    preprocessing = config['preprocessing']
    source_domain_dir = config['source_domain_dir']
    transform = config['transform']
    model = config['model']
    training = config['training']
    
    return source_domain_dir, preprocessing, transform, model, training

def load_generating_config(config):
    assert type(config) is str or type(config) is dict
    
    config = yaml.safe_load(open(config)) if type(config) is str else config

    return config