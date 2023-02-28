# Pandas example: linear regression to predict housing price:
# training data: house_training.csv, testing data: house_testing.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# 1. read csv with pandas
training_data = pd.read_csv('house_training.csv')
# print(training_data)
testing_data = pd.read_csv('house_testing.csv')
# print(testing_data)
ax = training_data.plot.scatter(x='area-sqft', y='price-k', color='Blue', label="training")
plt.show()
print('--------------------------------')

# 2. learn a linear regression model from the training data
regr = linear_model.LinearRegression()

# Train the model using the training sets
X = np.array(training_data.loc[:, ['area-sqft','bedroom','bathroom']])
Y = np.array(training_data['price-k'])
print(X)
print(Y)
regr.fit(X,Y)
# Apply the trained model to predict the testing data
tmp=np.array(testing_data)
pred = regr.predict(np.array(testing_data))
print(pred)
fig, bx = plt.subplots()
bx.scatter(np.array(testing_data['area-sqft']), y=pred, color='Red', label="predicted")
bx.set_xlabel("area-sqft")
bx.set_ylabel("predicted price")
plt.show()
print('--------------------------------')
# Apply the trained model to predict the price of interested house
interested_house = regr.predict(np.array([1050, 2, 1]).reshape(1, -1))
print(interested_house)