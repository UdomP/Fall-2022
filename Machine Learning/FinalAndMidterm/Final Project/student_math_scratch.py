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

def h(theta, x):
    # print(x.shape)
    # print(theta[1:].shape)
    return theta[0] + ((theta[1:].T @ x))

def nn(iter, featureN, _x, _y, alpha):
    weightMatrix1 = np.ones((featureN, featureN)) * .1
    weightMatrix2 = np.ones((featureN, featureN)) * .1
    weightMatrixLast = np.ones((featureN, featureN)) * .1
    theta = np.ones((featureN + 1, 1))
    lossList = []
    loop = 0
    for k in range(iter):
        if loop >= (len(_x)):
            loop -= len(_x)
        # Forward
        curX = x[loop].reshape(featureN, 1)
        curY = _y[0][loop].reshape(1,1)
        z1 = (weightMatrix1 @ curX)
        a1 = relu(z1)

        z2 = (weightMatrix2 @ a1)
        a2 = relu(z2)

        zn = (weightMatrixLast @ a2)
        yh = h(theta, zn)

        znPrime = yh - curY
        # print(znPrime.shape)
        # print(a1.shape)
        # exit(0)
        wnPrime = a1 @ znPrime
        
        # bnPrime = np.sum(znPrime, axis=1)
        #
        # print(wnPrime.shape)
        # print(znPrime.shape)
        # print(prime_relu(z2).shape)
        # exit(0)
        z2Prime = (wnPrime @ znPrime) * prime_relu(z2)
        
        w2Prime = z2Prime @ a2.T
        
        # b2Prime = np.sum(z2Prime, axis=1)
        #
        z1Prime = (w2Prime @ z2Prime) * prime_relu(z1)
        
        w1Prime = z1Prime @ curX.T
        
        # b1Prime = np.sum(z1Prime, axis=1)

        weightMatrix1 -= alpha * w1Prime
        weightMatrix2 -= alpha * w2Prime
        weightMatrixLast -= alpha * wnPrime

        hPrime = (yh - curY) * curX
        theta[0] -= alpha * ((yh - curY) * 1)[0]
        theta[1:] -= alpha * hPrime

        # print('curX: ', curX.shape)
        # print('curY: ', curY.shape)
        # print('_x: ', _x.shape)
        # print('_y: ', _y.shape)
        # print('z1: ', z1.shape)
        # print('a1: ', a1.shape)
        # print('z2: ', z2.shape)
        # print('a2: ', a2.shape)
        # print('zn: ', zn.shape)
        # print('yh: ', yh.shape)
        # exit(0)

        loop += 1
        if k % 1000 == 0:
            lossList.append(cost(curY, yh))

    return weightMatrix1, weightMatrix2, weightMatrixLast, theta, lossList

# def nnPredict(w1, w2, wn, _x, theta):
#     z1 = (w1 @ _x)
#     a1 = relu(z1)

#     z2 = (w2 @ a1)
#     a2 = relu(z2)

#     zn = (wn @ a2)
#     yh = h(theta, zn)
#     return yh

def nnPredict(w1, w2, wn, _x, theta):
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
        zn = (weightMatrixLast @ a2)
        yh = h(theta, zn)
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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
y_train = y_train.reshape(1, len(y_train))
y_test = y_test.reshape(1, len(y_test))
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)
print('start')
w1, w2, wn, theta, l = nn(19000, len(X_train.T), X_train, y_train, .0005)
predicted = nnPredict(w1, w2, wn, X_test, theta)

for a, b in zip(y_test[0], predicted):
    print(a, b)
# print(predict_test)

print('MSE: ', MSE(y_test, predicted, y_test.shape[1]))
print(l)