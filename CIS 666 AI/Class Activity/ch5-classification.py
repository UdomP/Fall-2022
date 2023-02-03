import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd

lv0 = np.array([[1,4,7],[2,5,8],[-3,6,-9]]).T
print(lv0)

lv1 = normalize(lv0, norm='l1')
print(lv1)

df = pd.read_csv('loan_status_new.csv')
print(df)