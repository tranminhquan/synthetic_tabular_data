import pandas as pd
import numpy as np

class FrequencyEncoder():
    """
    Encode data by its frequency or counting
    """
    
    def __init__(self, normalized=False):
        self.mapping_dict = None
        self.normalized = normalized
    
    def fit(self, data, by_counting=True):
        # add epsilon
        
        assert by_counting is True
        
        if type(data) is pd.core.series.Series:
            self.mapping_dict = data.value_counts().to_dict() if not self.normalized else (data.value_counts() / len(data)).to_dict()
        elif type(data) is np.ndarray:
            labels, counts = np.unique(data, return_counts=True)
            if self.normalized:
                self.mapping_dict = dict(zip(labels, counts / counts.sum()))
            else:
                self.mapping_dict = dict(zip(labels, counts))
    
    def transform(self, data):
        assert self.mapping_dict is not None
        
        if type(data) is pd.core.series.Series:
            transformed_data = data.apply(lambda k: self.mapping_dict[k])
        elif type(data) is np.ndarray:
            transformed_data = np.array([self.mapping_dict[k] for k in data])
        
        return transformed_data
    
    def inverse_transform(self, data, cal_from_dist=True):
        """
        Inverse transform
        
            cal_from_dist: calculate the distance of column values to each labels and chose the minimum one
        """
        assert self.mapping_dict is not None
        
        if cal_from_dist:            
            pred_indices = np.array([[abs(k - lb) for lb in list(self.mapping_dict.values())] for k in data])
            pred_indices = np.argmin(pred_indices, axis=1)
            
            pred_labels = [list(self.mapping_dict.keys())[k] for k in pred_indices]
        
        return np.array(pred_labels)
        
        