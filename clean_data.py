# TO-DO: Implement CLI for data cleaning

from processing.cleaning import TabCleaning 
import pandas as pd

data_dir = None
dst_dir = None

# load dataset
assert type(data_dir) is str or type(data_dir) is pd.DataFrame
df = pd.read_csv(data_dir) if type(data_dir) is str else data_dir

# clean
tabclean = TabCleaning()
clean_df = tabclean.clean(df)

# save
if dst_dir is not None:
    df.to_csv(dst_dir, index=False)