import pandas as pd
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as mlp

df = pd.read_csv('student_math.csv', sep=';')

# Remove row where value in column G3 is zero
# df = df[df['G3'] !=0]

# Randomize rows
# df = df.sample(frac=1)

toBeCleanList = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
replaceDic = {'school': {'GP': 0, 'MS': 1}, 
            'sex': {'F': 0, 'M': 1}, 
            'address': {'U': 0, 'R': 1}, 
            'famsize': {'GT3': 0, 'LE3': 1}, 
            'Pstatus': {'A': 0, 'T': 1}, 
            'Mjob': {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4}, 
            'Fjob': {'teacher': 0, 'other': 1, 'services': 2, 'health': 3, 'at_home': 4}, 
            'reason': {'course': 0, 'other': 1, 'home': 2, 'reputation': 3}, 
            'guardian': {'mother': 0, 'father': 1, 'other': 2}, 
            'schoolsup': {'yes': 1, 'no': 0}, 
            'famsup': {'no': 0, 'yes': 1}, 
            'paid': {'no': 0, 'yes': 1}, 
            'activities': {'no': 0, 'yes': 1}, 
            'nursery': {'yes': 1, 'no': 0}, 
            'higher': {'yes': 1, 'no': 0}, 
            'internet': {'no': 0, 'yes': 1}, 
            'romantic': {'no': 0, 'yes': 1}
            }
lis = ['G3']
for h in df.head():
    try:
        float(df[h][1])
    except:
        lis.append(h)

newDF = df.copy()
for l in toBeCleanList:
    for ll in replaceDic[l]:
        newDF[l] = newDF[l].replace(ll, (replaceDic[l])[ll])
print('Old dataframe')
print(df)
print('New dataframe')
print(newDF)

toBeRemovedList = ['G3']
y = newDF['G3'].to_numpy()
x = newDF.drop(toBeRemovedList, axis=1).to_numpy()
print(x.shape)
n = (len(np.transpose(x)))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=40, shuffle=True)
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)
print(np.average(y))
MLR = MLPRegressor(hidden_layer_sizes=(n, n), activation='relu', solver='sgd', max_iter=100, alpha=.001, verbose=10)
MLR.fit(X_train, y_train)
loss = MLR.loss_curve_
predict_train = MLR.predict(X_train)
predict_test = MLR.predict(X_test)

def MSE(y, yh, n):
    yy = (y - yh)**2
    return np.sum(yy) * (1/n)

for a, b in zip(y_test, predict_test):
    print(a, b)
# print(predict_test)

print('MSE: ', MSE(y_test, predict_test, len(y_test)))

mlp.plot(loss)
mlp.show()