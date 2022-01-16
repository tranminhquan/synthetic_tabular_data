
from collections import namedtuple
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import pandas as pd
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
    
    def __init__(self, categorical_cols, max_cols, max_guassian_components=10, gaussian_weight_threshold=5e-3):
        self.categorical_cols = categorical_cols # list or numpy array of categorical column names
        self.max_cols = max_cols # fixed maximum cols (len of inputs) to fit to model
        self.max_guassian_components = max_guassian_components # fixed maximum gaussian components for numerical col
        self.gaussian_weight_threshold = gaussian_weight_threshold # threshold to consider a gaussian component
        
    def fit(self, df):
        """
        fit the column values (numerical or categorical)
        
        for categorical col: 1 col <-> 1 output, output is label encoded by counting (or frequency) encoding 
        for numerical col: 1 col <-> 2 outputs
            1st output: numerical value sampling from GMM fited to the column
            2nd output: label of component in GMM (label encoded by counting (or frequency) encoding)
            
        each column as a `ColumnTransformInfo` storing information about the fitting process, including
            column_name
            column_type: 'categorical' or 'numerical'
            transform: list of applied transformation
            transform_aux: list of aux transformation (consistent order with transform param)
            output_info: list of `SpanInfo`, each one consists of
                dim: number of dimension
                activation_fn: applied activation for the output. (For the current version: `tanh` for numerical, `None` for categorical, but is denoted as `softmax`)
            output_dimension: sum of outputs' dimensions
        """
        
        self.output_info_list = [] # additionally store `SpanInfo`
        self.col_transform_info_list = [] # store `ColumnTransformInfo`
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
        """
        Categorical column values are encoded by counting (or calculate frequency)
        """
        
        frequency_encoder = FrequencyEncoder()
        frequency_encoder.fit(data)
        
        return ColumnTransformInfo(column_name=col_name,
                                   column_type='categorical',
                                   transform=[frequency_encoder],
                                   transform_aux=[frequency_encoder.mapping_dict],
                                   output_info=[SpanInfo(1, 'softmax')],
                                   output_dimensions=1)
    
    def fit_numerical(self, col_name, data):
        """
        Fit Bayesian Gaussian Mixture to numerical column
        """
        
        # fit BayesianGaussianMixture model
        gm = BayesianGaussianMixture(n_components=self.max_guassian_components,
                                     weight_concentration_prior_type='dirichlet_process',
                                     weight_concentration_prior=0.001,
                                     n_init=1)
        
        gm.fit(data.to_numpy().reshape(-1,1))
        
        # choose component by thresholding
        valid_component_indicator = gm.weights_ > self.gaussian_weight_threshold
        num_components = valid_component_indicator.sum()
        
        # placeholder to encode GMM components labels
        component_encoder = FrequencyEncoder()
        
        return ColumnTransformInfo(column_name=col_name,
                                   column_type='numerical',
                                   transform=[gm, component_encoder],
                                   transform_aux=[valid_component_indicator, component_encoder.mapping_dict],
                                   output_info=[SpanInfo(1, 'tanh'), SpanInfo(1, 'softmax')],
                                   output_dimensions=2)
    
    def transform(self, df):
        col_data_list = []
        
        self.raw_dtypes = df.infer_objects().dtypes # holder for later inverse transformation
        
        # transform
        for col_transform_info in self.col_transform_info_list:
            
            if col_transform_info.column_type == 'categorical':
                col_data_list += self.transform_categorical(col_transform_info, df[col_transform_info.column_name])
                
            elif col_transform_info.column_type == 'numerical':
                col_data_list += self.transform_numerical(col_transform_info, df[col_transform_info.column_name])
                
        col_data_list = np.concatenate(col_data_list, axis=1)
        
        # TODO: cropping if number of cols of tabular > max_cols
        
        # add padding 
        if col_data_list.shape[1] <= self.max_cols:
            col_data_list = np.concatenate([col_data_list, np.zeros((col_data_list.shape[0], self.max_cols - col_data_list.shape[1]))], axis=1)
        
        return col_data_list
    
    def transform_categorical(self, col_trans_info, data):
        """
        Transform using stored encoder from `ColumnTransformInfo`
        """
        encoder = col_trans_info.transform[0]
        data = encoder.transform(data)
        
        return [np.expand_dims(data.to_numpy(), axis=-1).astype(int)]
    
    def transform_numerical(self, col_trans_info, data, eps=1e-6):
        """
        Transform using stored Bayesian GMM
        """
        
        # get model and chosen component from fitting step
        data = np.expand_dims(data.to_numpy(), axis=-1)
        gm = col_trans_info.transform[0]
        valid_component_indicator = col_trans_info.transform_aux[0]
        n_valid_components = valid_component_indicator.sum()
        
        # get means and stds
        means = gm.means_.reshape((1, self.max_guassian_components))
        stds = np.sqrt(gm.covariances_).reshape((1, self.max_guassian_components))
        
        # normalize data by means and stds
        normalized_values = ((data - means) / (4 * stds))[:, valid_component_indicator] # shape: (n_data, n_components)
        
        # use fitted Bayesian GMM to calculate prob for data along chosen components
        component_probs = gm.predict_proba(data)[:, valid_component_indicator]
        
        # chose component: for each data sample, randomly selected components with probability calculate from Bayesian GMM
        selected_components = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            component_porb_t = component_probs[i] + eps
            component_porb_t = component_porb_t / component_porb_t.sum()
            selected_components[i] = np.random.choice(np.arange(n_valid_components), p=component_porb_t)
        
        # select normalied values: get normalized values by its corresponding selected component (above steps)
        aranged = np.arange(len(data))
        selected_normalized_values = normalized_values[aranged, selected_components].reshape([-1,1])
        selected_normalized_values = np.clip(selected_normalized_values, -.99, .99)
        
        # for selected components, encode those labels by the same method used for categorical columns
        component_encoder = col_trans_info.transform[1]
        component_encoder.fit(selected_components)
        selected_components = component_encoder.transform(selected_components)
        
        return [selected_normalized_values, selected_components.reshape(-1,1)] # selected_components' labels are encoded by FrequencyEncoder
    
    def inverse_transform(self, data, sigmas=None):
        start = 0
        
        rs_data = []
        column_names = []
        for col_transform_info in self.col_transform_info_list:
            dim = col_transform_info.output_dimensions
            col_data = data[:, start:start+dim]
            
            if col_transform_info.column_type == 'categorical':
                inverse_data = self.inverse_transform_categorical(col_transform_info, col_data)
            elif col_transform_info.column_type == 'numerical':
                inverse_data = self.inverse_transform_numerical(col_transform_info, col_data, sigmas, start)
                
            rs_data.append(inverse_data)
            column_names.append(col_transform_info.column_name)

            start = start + dim
        
        rs_data = np.column_stack(rs_data)
        
        # BUG: problem with `int` datatype
#         return pd.DataFrame(rs_data, columns=column_names).astype(self.raw_dtypes)
        return pd.DataFrame(rs_data, columns=column_names)
                
    def inverse_transform_categorical(self, col_trans_info, data):
        encoder = col_trans_info.transform[0]
        inverse_trans_data = encoder.inverse_transform(data[:,0])
        
        return np.array(inverse_trans_data)
    
    def inverse_transform_numerical(self, col_trans_info, data, sigmas, start):
        gm = col_trans_info.transform[0]
        valid_component_indicator = col_trans_info.transform_aux[0]
        
        selected_normalized_values = data[:, 0] # 1 dim
        selected_component_probs = data[:, 1] # 1 dim
        
        # apply activation function
        act_fn = np.tanh if col_trans_info.output_info[0].activation_fn == 'tanh' else np.identity
        selected_normalized_values = act_fn(selected_normalized_values)
        
        if sigmas is not None:
            sig = sigmas[start]
            selected_normalized_values = np.random.normal(selected_normalized_values, sig)
            
        selected_normalized_values = np.clip(selected_normalized_values, -1, 1)
        component_probs = np.ones((len(data), self.max_guassian_components)) * -100

        component_encoder = col_trans_info.transform[1]
        selected_components = component_encoder.inverse_transform(selected_component_probs, cal_from_dist=True)
        
        means = gm.means_.reshape([1, -1])
        stds = np.sqrt(gm.covariances_).reshape([1, -1])
        means_t = means[:, selected_components]
        stds_t = stds[:, selected_components]
        
        inverse_trans_data = selected_normalized_values * 4 * stds_t + means_t
        inverse_trans_data = inverse_trans_data.reshape([-1])
        
        return inverse_trans_data