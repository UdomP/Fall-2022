import pandas as pd
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('problems/regression_math/student_math.csv', sep=';')
df = df.sample(frac=1)

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

def J(y, yh):
    return np.average(.5*(np.abs(y - yh)**2))

def h(theta, x):
    return 1 + (np.transpose(theta) @ x)

def nn3(iter, _x, _y, alpha, inputSize, featureSize):
    w1 = np.ones((featureSize, featureSize)) * .1
    w2 = np.ones((featureSize, featureSize)) * .1
    wn = np.ones((1, featureSize)) * .1
    b1 = np.ones((featureSize, 1))
    b2 = np.ones((featureSize, 1))
    bn = np.ones((1, 1))
    loss_list = []
    for k in range(iter):
        # Forward
        z1 = (w1 @ _x)
        a1 = relu(z1)

        z2 = (w2 @ a1) + b2
        a2 = relu(z2)

        zn = (wn @ a2)
        yh = (zn)

        rowLen = _y.shape[1]

        # Backward
        znPrime = (yh - _y)

        wnPrime = znPrime @ a1.T / rowLen
        
        bnPrime = np.sum(znPrime, axis=1) / rowLen
        #
        z2Prime = (wnPrime.T @ znPrime) * prime_relu(z2)
        
        w2Prime = z2Prime @ a2.T / rowLen
        
        b2Prime = np.sum(z2Prime, axis=1) / rowLen
        #
        z1Prime = (w2Prime.T @ z2Prime) * prime_relu(z1)
        
        w1Prime = z1Prime @ _x.T / rowLen
        
        b1Prime = np.sum(z1Prime, axis=1) / rowLen
        
        b1Prime = b1Prime.reshape(b1.shape)
        b2Prime = b2Prime.reshape(b2.shape)
        w1 -= alpha * w1Prime
        b1 -= alpha * b1Prime
        w2 -= alpha * w2Prime
        b2 -= alpha * b2Prime
        wn -= alpha * wnPrime
        bn -= alpha * bnPrime

        if k % 10 == 0:
            loss_list.append(J(yh, _y))
        # print('z1: ', z1.shape)
        # print('a1: ', a1.shape)
        # print('z2: ', z2.shape)
        # print('a2: ', a2.shape)
        # print('zn: ', zn.shape)
        # print('yh: ', yh.shape)
        # print('znPrime: ', znPrime.shape)
        # print('wnPrime: ', wnPrime.shape)
        # print('bnPrime: ', bnPrime.shape)
        # print('z1Prime: ', z1Prime.shape) 
        # print('w1Prime: ', w1Prime.shape)
        # print('b1Prime: ', b1Prime.shape)
        # print('z2Prime: ', z2Prime.shape) 
        # print('w2Prime: ', w2Prime.shape)
        # print('b2Prime: ', b2Prime.shape)

    return w1, b1, w2, b2, wn, bn, loss_list

def nn3Predict(_x, w1, b1, w2, b2, wn, bn):
    z1 = (w1 @ _x)
    a1 = relu(z1)

    z2 = (w2 @ a1) + b2
    a2 = relu(z2)

    zn = (wn @ a2)
    yh = (zn)

    # print('z1: ', z1.shape)
    # print('a1: ', a1.shape)
    # print('zn: ', zn.shape)
    # print('yh: ', yh.shape)
    return yh

def nn(iter, _x, _y, alpha, inputSize, featureSize):
    w1 = np.ones((featureSize, featureSize)) * .1
    wn = np.ones((1, featureSize)) * .1
    b1 = np.ones((featureSize, 1))
    bn = np.ones((1, 1))
    loss_list = []
    for k in range(iter):
        # Forward
        z1 = (w1 @ _x) + b1
        a1 = relu(z1)

        zn = (wn @ a1) + bn
        yh = (zn)

        rowLen = _y.shape[1]

        # Backward
        znPrime = yh - _y

        wnPrime = znPrime @ a1.T / rowLen
        
        bnPrime = np.sum(znPrime, axis=1) / rowLen
        
        z1Prime = (wnPrime.T @ znPrime) * prime_relu(z1)
        
        w1Prime = z1Prime @ _x.T / rowLen
        
        b1Prime = np.sum(z1Prime, axis=1) / rowLen
        
        b1Prime = b1Prime.reshape(b1.shape)
        w1 -= alpha * w1Prime
        b1 -= alpha * b1Prime
        wn -= alpha * wnPrime
        bn -= alpha * bnPrime

        if k % 10 == 0:
            loss_list.append(J(yh, _y))
        # print('z1: ', z1.shape)
        # print('a1: ', a1.shape)
        # print('zn: ', zn.shape)
        # print('yh: ', yh.shape)
        # print('znPrime: ', znPrime.shape)
        # print('wnPrime: ', wnPrime.shape)
        # print('bnPrime: ', bnPrime.shape)
        # print('z1Prime: ', znPrime.shape) 
        # print('w1Prime: ', w1Prime.shape)
        # print('b1Prime: ', b1Prime.shape)

    return w1, b1, wn, bn, loss_list

def nnPredict(_x, w1, b1, wn, bn):
    z1 = (w1 @ _x) + b1
    a1 = relu(z1)
    # print('z1: ', z1.shape)
    # print('a1: ', a1.shape)
    zn = (wn @ a1) + bn
    yh = (zn)
    # print('zn: ', zn.shape)
    # print('yh: ', yh.shape)
    return yh


y = df['G3'].to_numpy()
x = df.drop(lis, axis=1).to_numpy()
n = (len(np.transpose(x)))
featureN = len(x.T)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=40)
y_train = y_train.reshape(1, len(y_train))
y_test = y_test.reshape(1, len(y_test))
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)
print('start')

w1, b1, wn, bn, loss = nn(1000, X_train.T, y_train, .05, X_train.shape[0], X_train.shape[1])
result = nnPredict(X_test.T, w1, b1, wn, bn)
print(loss)
for a, b in zip(y_test[0], result[0]):
    print(a,b)