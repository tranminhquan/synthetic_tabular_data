import pandas as pd

class TabCleaning():
#     def __init__(self):
#         pass
    
    def is_timestamp(self, series):
        if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(series):
            return True
        return False
            
    def is_low_frequency_categorical_col(self, series):
        if pd.core.dtypes.common.is_string_dtype(series) or pd.core.dtypes.common.is_integer_dtype(series):
            frequency = series.value_counts().to_dict()
            
            if max(list(frequency.values())) < 5:
                return True
            
            if len(frequency) >= 0.8 * len(series):
                return True
            
        return False
    
    
    def fill_na(self, series):
        if not series.isna().any():
            return series
        mean = series[series.isna() == False].mean()
        series = series.fillna(mean)
        
        return series
            
    def clean(self, df, remove_timestamp=True, remove_low_frequency=True, remove_id=True, verbose=1):
        columns = df.columns
        for col in columns:
            if verbose: print('col ', col)
            
            series = df[col]
            
            # REMOVE
            if remove_timestamp:
                if self.is_timestamp(series):
                    if verbose: print('\t del: timestamp datatype')
                    del df[col]
                    continue
            
            if remove_low_frequency:
                if self.is_low_frequency_categorical_col(series):
                    if verbose: print('\t del: low frequency values or id column')
                    del df[col]
                    continue
                    
            # FILL NA
            series = self.fill_na(series)
            
            # UPDATE SERIES
            df[col] = series
            
            if verbose: print('\t pass')
                    
        return df