
from collections import namedtuple
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from utils.encoding import FrequencyEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type',
        'transform', 'transform_aux',
        'output_info', 'output_dimensions'
    ]
)


class TabTransform():
    
    # TODO: inverse_transform
    
    def __init__(self, categorical_cols, max_cols, max_guassian_components=10, gaussian_weight_threshold=5e-3):
        self.categorical_cols = categorical_cols
        self.max_cols = max_cols
        self.max_guassian_components = max_guassian_components
        self.gaussian_weight_threshold = gaussian_weight_threshold
        
    def fit(self, df):
        self.output_info_list = []
        self.col_transform_info_list = []
        self.dimensions = 0
        
        for col in df.columns:
            if col in self.categorical_cols:
                col_transform_info = self.fit_categorical(col, df[col])
            else:
                col_transform_info = self.fit_numerical(col, df[col])
                
            self.output_info_list.append(col_transform_info.output_info)
            self.col_transform_info_list.append(col_transform_info)
            self.dimensions += col_transform_info.output_dimensions
        
    
    def fit_categorical(self, col_name, data):
        
        frequency_encoder = FrequencyEncoder()
        frequency_encoder.fit(data)
        
        return ColumnTransformInfo(column_name=col_name,
                                   column_type='categorical',
                                   transform=frequency_encoder,
                                   transform_aux=frequency_encoder.mapping_dict,
                                   output_info=[SpanInfo(1, 'softmax')],
                                   output_dimensions=1)
    
    def fit_numerical(self, col_name, data):
        
        gm = BayesianGaussianMixture(n_components=self.max_guassian_components,
                                     weight_concentration_prior_type='dirichlet_process',
                                     weight_concentration_prior=0.001,
                                     n_init=1)
        
        gm.fit(data.to_numpy().reshape(-1,1))
        valid_component_indicator = gm.weights_ > self.gaussian_weight_threshold
        num_components = valid_component_indicator.sum()
        
        return ColumnTransformInfo(column_name=col_name,
                                   column_type='numerical',
                                   transform=gm,
                                   transform_aux=valid_component_indicator,
                                   output_info=[SpanInfo(1, None), SpanInfo(1, 'softmax')],
                                   output_dimensions=2)
    
    def transform(self, df):
        col_data_list = []
        
        # transform
        for col_transform_info in self.col_transform_info_list:
            
            if col_transform_info.column_type == 'categorical':
                col_data_list += self.transform_categorical(col_transform_info, df[col_transform_info.column_name])
                
            elif col_transform_info.column_type == 'numerical':
                col_data_list += self.transform_numerical(col_transform_info, df[col_transform_info.column_name])
                
        col_data_list = np.concatenate(col_data_list, axis=1)
        
        # TODO: cropping
        
        # add padding 
        if col_data_list.shape[1] <= self.max_cols:
            col_data_list = np.concatenate([col_data_list, np.zeros((col_data_list.shape[0], self.max_cols - col_data_list.shape[1]))], axis=1)
        
        return col_data_list
    
    def transform_categorical(self, col_trans_info, data):
        encoder = col_trans_info.transform
        data = encoder.transform(data)
        
        return [np.expand_dims(data.to_numpy(), axis=-1)]
    
    def transform_numerical(self, col_trans_info, data, eps=1e-6):
        data = np.expand_dims(data.to_numpy(), axis=-1)
        gm = col_trans_info.transform
        valid_component_indicator = col_trans_info.transform_aux
        n_valid_components = valid_component_indicator.sum()
        
        means = gm.means_.reshape((1, self.max_guassian_components))
        stds = np.sqrt(gm.covariances_).reshape((1, self.max_guassian_components))
        normalized_values = ((data - means) / (4 * stds))[:, valid_component_indicator] # shape: (n_data, n_components)
        component_probs = gm.predict_proba(data)[:, valid_component_indicator]
        
        # select component
        selected_components = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            component_porb_t = component_probs[i] + eps
            component_porb_t = component_porb_t / component_porb_t.sum()
            selected_components[i] = np.random.choice(np.arange(n_valid_components), p=component_porb_t)
            
        
        aranged = np.arange(len(data))
        selected_normalied_values = normalized_values[aranged, selected_components].reshape([-1,1])
        selected_normalied_values = np.clip(selected_normalied_values, -.99, .99)
        
        return [selected_normalied_values, selected_components.reshape(-1,1)] # keep the selected components indices as labels