import numpy as np
import pandas as pd
import sklearn
import matplotlib

df = pd.read_csv("car_price_new.csv")
df = df.fillna(df.mean(skipna=True, axis=0), axis=0)

X = df
Y = df["price"]
del X["price"]

len = df.shape[0]

xTraining = X.iloc[:len*.8,:]
xTesting = X.iloc[len*.2:,:]
print(xTraining)
print(xTesting)

yTraining = Y.iloc[:len*.8,:]
yTesting = Y.iloc[len*.2:,:]
print(yTraining)
print(yTesting)