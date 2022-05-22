
from collections import namedtuple
from matplotlib.transforms import Transform
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import pandas as pd
from syntabtf.utils.encoding import FrequencyEncoder
from syntabtf.processing.signatures import TransformerEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type',
        'transform', 'transform_aux',
        'output_info', 'output_dimensions'
    ]
)


from collections import namedtuple
from sklearn.mixture import BayesianGaussianMixture
import numpy as np

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type',
        'transform', 'transform_aux',
        'output_info', 'output_dimensions'
    ]
)


class TabTransform():
    
    def __init__(self, categorical_cols=None, 
                 max_cols=50, 
                 max_guassian_components=10, 
                 gaussian_weight_threshold=5e-3,
                 col_name_embedding=False,
                 **kwargs):
        """AI is creating summary for __init__

        Args:
            categorical_cols ([type], optional): [description]. Defaults to None.
            max_cols (int, optional): [description]. Defaults to 50.
            max_guassian_components (int, optional): [description]. Defaults to 10.
            gaussian_weight_threshold ([type], optional): [description]. Defaults to 5e-3.
            col_name_embedding (bool, optional): [description]. Defaults to False.
        """
        
        self.categorical_cols = categorical_cols # list or numpy array of categorical column names
        self.max_cols = max_cols # fixed maximum cols (len of inputs) to fit to model
        self.max_guassian_components = max_guassian_components # fixed maximum gaussian components for numerical col
        self.gaussian_weight_threshold = gaussian_weight_threshold # threshold to consider a gaussian component
        
        # column names embedding
        if col_name_embedding:
            max_seq_length = kwargs['max_seq_length'] if 'max_seq_length' in kwargs else 10
            decomposited_size = kwargs['decomposited_size'] if 'decomposited_size' in kwargs else 32
            self.colname_transformer = TransformerEncoder("bert-base-multilingual-uncased", max_seq_length=max_seq_length, decomposited_size=decomposited_size)
        else:
            self.colname_transformer = None
                
    def fit(self, df, categorical_norm=True):
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
        
        # if categorical cols is not provided, automatically detect by consider dtype of column from pandas
        if self.categorical_cols is None:
            print('Automatically detect categorical columns')
            self.categorical_cols = [col for col in df.columns if df[col].dtypes == object]
            print('Found categorical columns: ', self.categorical_cols)
        
        
        print('Fitting columns ...')
        for col in df.columns:
            print(' - {}'.format(col))
            if col in self.categorical_cols:
                print(' -> categorical')
                col_transform_info = self.fit_categorical(col, df[col], categorical_norm)
            else:
                print(' -> numerical')
                col_transform_info = self.fit_numerical(col, df[col], categorical_norm)
                
            self.output_info_list.append(col_transform_info.output_info)
            self.col_transform_info_list.append(col_transform_info)
            self.dimensions += col_transform_info.output_dimensions
            print(' --- ')
        
    
    def fit_categorical(self, col_name, data, categorical_norm):
        """
        Categorical column values are encoded by counting (or calculate frequency)
        """
        
        frequency_encoder = FrequencyEncoder(categorical_norm)
        frequency_encoder.fit(data)
        
        output_info = [SpanInfo(1, 'softmax')]
        output_dimensions = 1
        if self.colname_transformer is not None:
            output_info.append(SpanInfo(self.colname_transformer.embedding_dim, 'colname_embedding'))
            output_dimensions += self.colname_transformer.embedding_dim
        
        return ColumnTransformInfo(column_name=col_name,
                                   column_type='categorical',
                                   transform=[frequency_encoder],
                                   transform_aux=[frequency_encoder.mapping_dict],
                                   output_info=output_info,
                                   output_dimensions=output_dimensions)
    
    def fit_numerical(self, col_name, data, categorical_norm):
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
        component_encoder = FrequencyEncoder(categorical_norm)
        
        
        output_info = [SpanInfo(1, 'tanh'), SpanInfo(1, 'softmax')]        
        output_dimensions = 2
        if self.colname_transformer is not None:
            output_info.append(SpanInfo(self.colname_transformer.embedding_dim, 'colname_embedding'))
            output_dimensions += self.colname_transformer.embedding_dim
        
        return ColumnTransformInfo(column_name=col_name,
                                   column_type='numerical',
                                   transform=[gm, component_encoder],
                                   transform_aux=[valid_component_indicator, component_encoder.mapping_dict],
                                   output_info=output_info,
                                   output_dimensions=output_dimensions)
    
    def transform(self, df):
        col_data_list = []
        
        self.raw_dtypes = df.infer_objects().dtypes # holder for later inverse transformation
        
        # transform
        print('Transforming columns ...')
        for col_transform_info in self.col_transform_info_list:
            print('- {}'.format(col_transform_info.column_name))
            if col_transform_info.column_type == 'categorical':
                print('-> categorical')
                col_data_list += self.transform_categorical(col_transform_info, df[col_transform_info.column_name])
                
            elif col_transform_info.column_type == 'numerical':
                print('-> numerical')
                col_data_list += self.transform_numerical(col_transform_info, df[col_transform_info.column_name])
            print(' --- ')
            
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
        
        # column name embedding
        if self.colname_transformer is not None:
            colname_embedding = self.colname_transformer.encode_col_name(col_trans_info.column_name)
            return [np.expand_dims(data.to_numpy(), axis=-1).astype(float), np.tile(colname_embedding, (len(data),1))]
            
        return [np.expand_dims(data.to_numpy(), axis=-1).astype(float)]
        
    
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
        
        # column name embedding
        if self.colname_transformer is not None:
            colname_embedding = self.colname_transformer.encode_col_name(col_trans_info.column_name)
            return [selected_normalized_values, selected_components.reshape(-1,1), np.tile(colname_embedding, (len(data),1))] # selected_components' labels are encoded by FrequencyEncoder
        # 
        return [selected_normalized_values, selected_components.reshape(-1,1)] # selected_components' labels are encoded by FrequencyEncoder
    
    def inverse_transform(self, data, sigmas=None, sigmoid=False):
        start = 0
        
        rs_data = []
        column_names = []
        for col_transform_info in self.col_transform_info_list:
            dim = col_transform_info.output_dimensions
            col_data = data[:, start:start+dim]
            
            # categorical column
            if col_transform_info.column_type == 'categorical':
                inverse_data = self.inverse_transform_categorical(col_transform_info, col_data, sigmoid)
                
            # numerical column
            elif col_transform_info.column_type == 'numerical':
#                 print(col_transform_info.column_name, '=================')
#                 print('col_data shape: ', col_data.shape)
#                 print('col_data [:, 0]: ', col_data[:,0])
#                 print('col_data [:, 1]: ', col_data[:,1])
                inverse_data = self.inverse_transform_numerical(col_transform_info, col_data, sigmas, start, sigmoid)
                
            rs_data.append(inverse_data)
            column_names.append(col_transform_info.column_name)
            start = start + dim
        
        rs_data = np.column_stack(rs_data)
        
        # TO-DO: problem with `int` datatype
#         return pd.DataFrame(rs_data, columns=column_names).astype(self.raw_dtypes)
        return pd.DataFrame(rs_data, columns=column_names)
                
    def inverse_transform_categorical(self, col_trans_info, data, sigmoid=False):
        encoder = col_trans_info.transform[0]
        
        if sigmoid:
            data[:,0] = 1 / (1 + np.exp(-data[:,0]))
        inverse_trans_data = encoder.inverse_transform(data[:,0])
        
        return np.array(inverse_trans_data)
    
    def inverse_transform_numerical(self, col_trans_info, data, sigmas, start, sigmoid=False):
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
#         component_probs = np.ones((len(data), self.max_guassian_components)) * -100 #redundant code

        component_encoder = col_trans_info.transform[1]
    
        if sigmoid:
            selected_component_probs = 1 / (1 + np.exp(-selected_component_probs))
        selected_components = component_encoder.inverse_transform(selected_component_probs, cal_from_dist=True)
        
        means = gm.means_.reshape([1, -1])
        stds = np.sqrt(gm.covariances_).reshape([1, -1])
        means_t = means[:, selected_components]
        stds_t = stds[:, selected_components]
        
#         print('selected components probs: ', selected_component_probs)
#         print('selected components: ', selected_components)
#         print('means_t: ', means_t)
#         print('stds_t: ', stds_t)
        
        inverse_trans_data = selected_normalized_values * 4 * stds_t + means_t
        inverse_trans_data = inverse_trans_data.reshape([-1])
        
        return inverse_trans_data
    