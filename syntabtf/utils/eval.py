
import sdmetrics
import json
import pandas as pd

# def scoring(real_df, synthetic_df, metadata, metric_names = ['SVCDetection', 'GMLogLikelihood', 'CSTest', 'KSTest', 'KSTestExtended', 'ContinuousKLDivergence', 'DiscreteKLDivergence']):
# def scoring(real_df, synthetic_df, metadata, metric_names = ['GMLogLikelihood', 'CSTest', 'KSTest', 'KSTestExtended', 'ContinuousKLDivergence', 'DiscreteKLDivergence']):
def scoring(real_df, synthetic_df, metadata, metric_names = ['BinaryDecisionTreeClassifier', 'BinaryAdaBoostClassifier', 'BinaryMLPClassifier', 'BinaryLogisticRegression', 'BinaryMLPClassifier', \
                                                             'MulticlassDecisionTreeClassifier', 'MulticlassMLPClassifier', 'LinearRegression', 'MLPRegressor', \
                                                             'GMLogLikelihood', 'CSTest', 'KSTest', 'KSTestExtended', 'ContinuousKLDivergence', 'DiscreteKLDivergence']):
    
    assert type(real_df) is str or type(real_df) is pd.DataFrame
    assert type(synthetic_df) is str or type(synthetic_df) is pd.DataFrame
    
    metadata = json.load(open(metadata)) if type(metadata) is str else metadata
    metadata = None if type(metadata) is not dict else metadata
    
    print('metadata: ', metadata)
    
    # real_df = pd.read_csv(real_df) if type(real_df) is str else real_df
    # synthetic_df = pd.read_csv(synthetic_df) if type(synthetic_df) is str else synthetic_df
    
    # modify 'fields' in metadata to be consistent with column of real df
    norm_fields = {k:v for k,v in metadata['fields'].items() if k in real_df.columns.to_list()}
    metadata['fields'] = norm_fields
    
    print('normalized metadata: ', metadata)
    
    # scoring single table
    
    metrics = sdmetrics.single_table.SingleTableMetric.get_subclasses()
    if metric_names is not None:
        print('Apply specific metrics: ', metric_names)
        metrics = {k:v for k,v in metrics.items() if k in metric_names}
    
    scoring_df = sdmetrics.compute_metrics(metrics, real_df, synthetic_df, metadata=metadata)
    
    print('resulted scoring df: ', scoring_df)
    
    # TODO: compare between real data and finetuned synthetic data
    # vssynt_scoring_df = 
    
    # compare between original synthetic data vs finetuned synthetic data
    
    return scoring_df
    