# Pandas example: data selection by index or condition
# modified from https://github.com/MorvanZhou/tutorials/blob/master/numpy%26pandas/12_selection.py

import pandas as pd
import numpy as np

dates = pd.date_range('20221201', periods=10)
df = pd.DataFrame(np.random.randn(10,5), index=dates, columns=['c1','c2','c3','c4','c5'])

print(df['c1'])     # col-index   # same as print(df.c1)
print('-------------------')
print(df[0:3])      # row-index
print('-------------------')
print(df['20221203':'20221205'])
print('-------------------')
# select by label: loc
print(df.loc['20221203'])
print(df.loc[:,['c1','c2']])
print(df.loc['20221202', ['c1','c2']])
print('++++++++++++++++++++++++++++')
# select by position: iloc, like slicing, row-index and col-index
print(df.iloc[3])
print('++++++++++++++++++++++++++++')
print(df.iloc[3, 1])
print('++++++++++++++++++++++++++++')
print(df.iloc[3:5,0:2])
print('++++++++++++++++++++++++++++')
print(df.iloc[[1,2,4],[0,2]])
print('++++++++++++++++++++++++++++')
# Boolean indexing
print(df[df.c1 > 0])
