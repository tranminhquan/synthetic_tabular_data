import pandas as pd
import numpy as np

class FrequencyEncoder():
    """
    Encode data by its frequency or counting
    """
    
    def __init__(self):
        self.mapping_dict = None
    
    def fit(self, data, by_counting=True):
        # TO-DO: encode by frequency
        
        assert by_counting is True
        
        if type(data) is pd.core.series.Series:
            self.mapping_dict = data.value_counts().to_dict()
        elif type(data) is np.ndarray:
            labels, counts = np.unique(data, return_counts=True)
            self.mapping_dict = dict(zip(labels, counts))
    
    def transform(self, data):
        assert self.mapping_dict is not None
        
        if type(data) is pd.core.series.Series:
            transformed_data = data.apply(lambda k: self.mapping_dict[k]).astype(int)
        elif type(data) is np.ndarray:
            transformed_data = np.array([self.mapping_dict[k] for k in data])
        
        return transformed_data
    
    def inverse_transform(self, data, cal_from_dist=True):
        """
        Inverse transform
        
            cal_from_dist: calculate the distance of column values to each labels and chose the minimum one
        """
        assert self.mapping_dict is not None
        inv_mapping_dict = {v:k for k,v in self.mapping_dict.items()}
        
        
        if cal_from_dist:            
            pred_indices = np.array([[abs(k - lb) for lb in list(inv_mapping_dict.keys())] for k in data])
            pred_indices = np.argmin(pred_indices, axis=1)
#             print(pred_indices.shape)
#             print(list(inv_mapping_dict.keys()))
            pred_labels = [list(inv_mapping_dict.keys())[k] for k in pred_indices]
            inv_transformed_data = [inv_mapping_dict[k] for k in pred_labels]
        
        return np.array(inv_transformed_data)
        
        