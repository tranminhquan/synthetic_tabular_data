


class FrequencyEncoder():
    """
    Encode data by its frequency
    """
    
    def __init__(self):
        self.mapping_dict = None
    
    def fit(self, data):
        assert isinstance(data, pd.Series)
        
        self.mapping_dict = data.value_counts().to_dict()
    
    def transform(self, data):
        assert self.mapping_dict is not None
        
        transformed_data = data.apply(lambda k: self.mapping_dict[k])
        
        return transformed_data
    
    def inverse_transform(self, data):
        assert self.mapping_dict is not None
        
        inv_mapping_dict = {v:k for k,v in self.mapping_dict.items()}
        inv_transformed_data = data.apply(lambda k: inv_mapping_dict[k])
        
        return inv_transformed_data
        