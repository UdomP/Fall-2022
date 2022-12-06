# Pandas example: introduction to pd
# modified from https://github.com/MorvanZhou/tutorials/blob/master/numpy%26pandas/11_pandas_intro.py
import pandas as pd
import numpy as np

s = pd.Series([11,5,1,5,9,10,8])
print(s)
print('-------------------')
dates = pd.date_range('20221201', periods=10)
df = pd.DataFrame(np.random.randn(10,5), index=dates, columns=['c1','c2','c3','c4','c5'])
print(df)
print(df['c4'])
print('-------------------')
print(df.index)
print(df.columns)
print(df.values)
print(df.describe())
print(df.T)
print(df.sort_index(axis=0, ascending=False))   # 0: sort by row, 1: sort by column
print(df.sort_values(by='c5'))
print('-------------------')
df2 = pd.DataFrame({'A' : 22, 'B' : pd.Timestamp('20221201'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'demo'})
print(df2)
