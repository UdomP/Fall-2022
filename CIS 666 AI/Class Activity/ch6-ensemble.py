import pandas as pd
import numpy as np

df = pd.read_csv('created_dataset.txt', header = None)
X = df[:][:]
print(X)
class_2 = np.array(df[y==2])