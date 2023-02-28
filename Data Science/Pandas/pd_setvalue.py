# Pandas example: set value by condition 
# modified from https://github.com/MorvanZhou/tutorials/blob/master/numpy%26pandas/13_set_value.py

import pandas as pd
import numpy as np

dates = pd.date_range('20221201', periods=7)
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[2,2] = 1111
df.loc['2022-12-05', 'D'] = 2222
df.A[df.A>0] = 0.123456
df['F'] = np.nan
df['added'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20221204', periods=6))
print(df)