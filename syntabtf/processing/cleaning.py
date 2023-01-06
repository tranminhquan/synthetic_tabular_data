import pandas as pd

class TabCleaning():
    def __init__(self, exclude=[], remove=[]):
        """AI is creating summary for __init__

        Args:
            exclude: exclude these columns in automated cleaning
            remove: remove these columns when cleaning 
        """
        self.exclude_cols = exclude
        self.remove_cols = remove
    
    def is_timestamp(self, series):
        if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(series):
            return True
        return False
            
    def is_low_frequency_categorical_col(self, series, min_freq_threshold=0.08, pct_to_remove=0.8):
        """
        Detect low frequency of categories (e.g. columns with ids, addresses, etc.)
        Args:
            series: `Series` pandas, data input
            min_freq_threshold: `float`, default is 0.08. Unless maximum category frequency is equal or greater than the threhold, the column will be removed
            pct_to_remove: `float`, percentage compare to number of values to remove. If the number of categories is equal or greater than pct*number of data, it will be removed
        """
        if pd.core.dtypes.common.is_string_dtype(series) or pd.core.dtypes.common.is_integer_dtype(series):
            frequency = series.value_counts().to_dict()
            frequency = {k : v / len(series.values) for k,v in frequency.items()}
            
            if max(list(frequency.values())) < min_freq_threshold:
                return True
            
            if len(frequency) >= pct_to_remove * len(series):
                return True
            
        return False
    
    def is_nan(self, series):
        if series.isna().any():
            return True
        
        return False
    
    
    def fill_na(self, series):
        mean = series[series.isna() == False].mean()
        series = series.fillna(mean)
        
        return series
            
    def clean(self, df, remove_timestamp=True, remove_low_frequency=True, remove_id=True, verbose=1, **kwargs):
        columns = df.columns
        for col in columns:
            if verbose: print('col ', col)

            if col in self.remove_cols:
                del df[col]
                print('\t Removed')
                continue
            
            if col in self.exclude_cols:
                print('\t Excluded ')
                continue
            
            series = df[col]
            
            # REMOVE
            if remove_timestamp:
                if self.is_timestamp(series):
                    if verbose: print('\t del: timestamp datatype')
                    del df[col]
                    continue
            
            if remove_low_frequency:
                min_freq_threshold = kwargs['min_freq_threshold'] if 'min_freq_threshold' in kwargs else 0.08
                pct_to_remove = kwargs['pct_to_remove'] if 'pct_to_remove' in kwargs else 0.8
                if self.is_low_frequency_categorical_col(series, min_freq_threshold=min_freq_threshold, pct_to_remove=pct_to_remove):
                    if verbose: print('\t del: low frequency values or id column')
                    del df[col]
                    continue
                
            # UPDATE SERIES
            # FILL NA
            if self.is_nan(series):
                if series.dtype == object: # remove nan
                    df.dropna(subset=[series.name], inplace=True)
                else:
                    series = self.fill_na(series)
                    df[col] = series
            else:
                df[col] = series
            
            if verbose: print('\t pass')
                    
        return df