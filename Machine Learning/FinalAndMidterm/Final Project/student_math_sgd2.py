import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('problems/regression_math/student_math.csv', sep=';')

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

def fx(z):
    return np.exp(z)/sum(np.exp(z))

def relu(z):
    return np.maximum(0, z)

def prime_relu(z):
    return np.heaviside(z, 0)

def identity(z):
    return z

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z) * (1-sigmoid(z))

def loss_function(yh, y):
    data_size = y.shape[1]
    cost = (-1 / data_size) * (np.dot(y, np.log(yh).T) + np.dot(1 - y, np.log(1 - yh).T))
    return cost

def cost(y, yh):
    return np.sum(.5 * (np.abs(y - yh)**2))

def costPrime(yh):
    return yh * (1 - yh)

def nn2(iter, featureN, _x, _y, alpha):
    weightMatrix1 = np.ones((featureN, featureN)) * .1
    weightMatrix2 = np.ones((featureN, featureN)) * .1
    weightMatrixLast = np.ones((1, featureN)) * .1
    b1 = np.ones((featureN, 1))
    b2 = np.ones((1, 1))
    lossList = []
    for k in range(iter):
        for i in range(len(_x)):
            curX = x[i].reshape(featureN, 1)
            curY = _y[0][i].reshape(1,1)
            z1 = (weightMatrix1 @ curX)
            a1 = relu(z1)

            yh = (weightMatrixLast @ a1) 
            yh = (yh) + b2
            loss = cost(curY, yh)
            lossList.append(loss)

            z2prime = yh - curY
            s = 0
            for z in range(curX.shape[0]):
                s = np.sum(weightMatrixLast[0] * prime_relu(z1[z]) * weightMatrix1.T[z])
                weightMatrix1.T[z] -= alpha * (z2prime * s * curX[z])[0]
                b1[z] -= alpha * (z2prime * s * curX[z])[0]
            for z in range(curX.shape[0]):
                s = np.sum(weightMatrixLast[0])
                weightMatrixLast.T[z] -= alpha * (z2prime * s)[0]
            b2 -=  alpha * (z2prime * s)[0]
    return weightMatrix1, weightMatrixLast, b1, b2, lossList

def nn2Predict(w1, w2, b1, b2, _x):
    weightMatrix1 = w1
    weightMatrix2 = w2
    yhList = []
    for i in range(len(_x)):
        curX = x[i].reshape(featureN, 1)
        z1 = (weightMatrix1 @ curX) + b1
        a1 = relu(z1)
        yh = (weightMatrix2 @ a1)
        yh = (yh) + b2
        yhList.append(yh[0][0])
    return yhList

def nn3(iter, featureN, _x, _y, alpha):
    weightMatrix1 = np.ones((featureN, featureN)) * .1
    weightMatrix2 = np.ones((featureN, featureN)) * .1
    weightMatrixLast = np.ones((1, featureN)) * .1
    b1 = np.ones((featureN, 1))
    b2 = np.ones((1, 1))
    lossList = []
    for k in range(iter):
        for i in range(len(_x)):
            curX = x[i].reshape(featureN, 1)
            curY = _y[0][i].reshape(1,1)
            z1 = (weightMatrix1 @ curX)
            a1 = relu(z1)

            z2 = (weightMatrix2 @ a1)
            a2 = relu(z2)

            yh = (weightMatrixLast @ a2)
            yh = (yh) 
            z2 = cost(curY, yh)

            z2prime = (yh - curY)
            for z in range(curX.shape[0]):
                s = np.sum(weightMatrixLast * prime_relu(z2) * weightMatrix2.T[z])
                weightMatrix1.T[z] -= alpha * (z2prime * s * curX[z] * prime_relu(z1[z]))[0]
            
            for z in range(curX.shape[0]):
                s = np.sum(weightMatrixLast * prime_relu(z2) * weightMatrix2.T[z])
                weightMatrix2.T[z] -= alpha * (z2prime * s)[0]

            for z in range(curX.shape[0]):
                s = np.sum(weightMatrixLast)
                weightMatrixLast.T[z] -= alpha * (z2prime * s)[0]
    return weightMatrix1, weightMatrix2, weightMatrixLast

def nn3Predict(w1, w2, wn, _x):
    weightMatrix1 = w1
    weightMatrix2 = w2
    weightMatrixLast = wn
    yhList = []
    for i in range(len(_x)):
        curX = x[i].reshape(featureN, 1)
        z1 = (weightMatrix1 @ curX)
        a1 = relu(z1)

        z2 = (weightMatrix2 @ a1)
        a2 = relu(z2)
        yh = (weightMatrixLast @ a2)
        yhList.append(yh[0][0])
    return yhList

def MSE(y, yh, n):
    yy = (y - yh)**2
    return np.sum(yy) * (1/n)


toBeRemovedList = ['G3', 'school', 'sex', 'address', 'guardian', 'paid', 'nursery']
y = newDF['G3'].to_numpy()
x = newDF.drop(toBeRemovedList, axis=1).to_numpy()
n = (len(np.transpose(x)))
featureN = len(x.T)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
y_train = y_train.reshape(1, len(y_train))
y_test = y_test.reshape(1, len(y_test))
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)
print('start')
w1, w2, b1, b2, loss = nn2(2, len(X_train.T), X_train, y_train, .001)
# w1, w2, wn = nn3(3, len(X_train.T), X_train, y_train, .0005)
print(w1)
print(w1.shape)
print(w2)
print(w2.shape)

# predict_train = nn3Predict(w1, w2, w3, X_train)
predict_test = nn2Predict(w1, w2, b1, b2, X_test)
# predict_test = nn3Predict(w1, w2, wn, X_test)
print(predict_test)

# # for a, b in zip(y_train, predict_train):
# #     print(a, b)

# # print('MSE: ', MSE(y_train, predict_train, len(y_train)))

for a, b in zip(y_test[0], predict_test):
    print(a, b)
# print(predict_test)

print('MSE: ', MSE(y_test, predict_test, y_test.shape[1]))
print(y_test.shape[1])
# print(loss)
# print(np.sort(loss))
# print(len(loss))
# print(loss[-1])