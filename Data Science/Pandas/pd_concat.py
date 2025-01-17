# Pandas example: concatenate data
# modified from: https://github.com/MorvanZhou/tutorials/blob/master/numpy%26pandas/16_concat.py

import pandas as pd
import numpy as np

# concatenating
# ignore index
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)  # axis=0: for rows; axis=1: for columns
# What happen if ignore_index=False?
print(res)
print('----------------------------')
# join, ('inner' for overlap index only, 'outer' for all index)
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])
res = pd.concat([df1, df2], axis=1, join='outer')
print(res)
print('-------------++++++++++++++++')
res = pd.concat([df1, df2], axis=1, join='inner')  # axis=1 for column
print(res)
print('++++++++++++++++++++++++++++')

# append
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])
res = df1.append(df2, ignore_index=True)
print(res)
print('============================')
res = df1.append([df2, df3])
print(res)
print('============================')

s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res = df1.append(s1, ignore_index=True)
print(res)
print('++++++++++++++++++++++++++++')