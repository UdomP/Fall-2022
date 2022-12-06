# Pandas example: deal with missing data
# modified from https://github.com/MorvanZhou/tutorials/blob/master/numpy%26pandas/14_nan.py

import pandas as pd
import numpy as np

dates = pd.date_range('20221201', periods=8)
df = pd.DataFrame(np.arange(32).reshape((8,4)), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)
print('----------------------------------------')
print(df.dropna(axis=0, how='any'))   # how={'any', 'all'}, drop horizontally
print('++++++++++++++++++++++++++++++++++++++++')
print(df.dropna(axis=1, how='any'))   # how={'any', 'all'}, drop vertically
print('++++++++++++++++++++++++++++++++++++++++')
print(df.fillna(value=0))
print('++++++++++++++++++++++++++++++++++++++++')
print(pd.isnull(df))