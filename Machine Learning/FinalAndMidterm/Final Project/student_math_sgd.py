import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('problems/regression_math/student_math.csv', sep=';')
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
            'schoolsup': {'yes': 0, 'no': 1}, 
            'famsup': {'no': 0, 'yes': 1}, 
            'paid': {'no': 0, 'yes': 1}, 
            'activities': {'no': 0, 'yes': 1}, 
            'nursery': {'yes': 0, 'no': 1}, 
            'higher': {'yes': 0, 'no': 1}, 
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
    return np.maximum(0, z)
    # return np.exp(z)/sum(np.exp(z))

# , w2, w3, x, y, yhat, z1
def layer1Update(w1, w2, w3, z1, z2, x, y, yhat):
    tempW1 = w1.copy()
    for rowIndex, row in enumerate(w1):
        jz = yhat - y
        fz1 = fx(z1[rowIndex])
        xi = x[rowIndex]
        const = jz * fz1 * xi
        for colIndex, col in enumerate(row):
            s = 0
            for i in range(len(row)):
                s += w3[0][i] * (fx(z2[i]) * (1 - fx(z2[i]))) * w2[i][colIndex]
            tempW1[rowIndex][colIndex] = const * s
    return tempW1

    # , w2, w3, x, y, yhat, z1
def layer2Update(w2, w3, z2, y, yhat):
    tempW2 = w2.copy()
    for rowIndex, row in enumerate(w2):
        jz = yhat - y
        const = jz
        for colIndex, col in enumerate(row):
            s = 0
            for i in range(len(row)):
                s += w3[0][i] * (fx(z2[i]) * (1 - fx(z2[i]))) * w2[i][colIndex]
            tempW2[rowIndex][colIndex] = const * s
    return tempW2

def layer3Update(w3, y, yhat):
    tempW3 = w3.copy()
    for rowIndex, row in enumerate(w3):
        jz = yhat - y
        const = jz
        for colIndex, col in enumerate(row):
            tempW3[rowIndex][colIndex] = jz * col
    return tempW3

def nn3(iter, featureN, _x, _y, alpha):
    weightMatrix1 = np.ones((featureN, featureN))
    weightMatrix2 = np.ones((featureN, featureN))
    weightMatrix3 = np.ones((1, featureN))
    for k in range(iter):
        for i in range(len(_x)):
            curX = x[i].reshape(featureN, 1)
            curY = _y[0][i].reshape(1,1)
            z1 = weightMatrix1 @ curX
            a1 = fx(z1)
            z2 = weightMatrix2 @ a1
            a2 = fx(z2)
            yh = weightMatrix3 @ a2
            tempWeight1 = layer1Update(weightMatrix1, weightMatrix2, weightMatrix3, z1, z2, curX, curY, yh)
            weightMatrix1 = weightMatrix1 - (alpha * tempWeight1)
            tempWeight2 = layer2Update(weightMatrix2, weightMatrix3, z2, curY, yh)
            weightMatrix2 = weightMatrix2 - (alpha * tempWeight2)
            tempWeight3 = layer3Update(weightMatrix3, curY, yh)
            weightMatrix3 = weightMatrix3 - (alpha * tempWeight3)
    return weightMatrix1, weightMatrix2, weightMatrix3

def nn3Predict(w1, w2, w3, _x):
    yhList = []
    for i in range(len(_x)):
        curX = x[i].reshape(featureN, 1)
        z1 = w1 @ curX
        a1 = fx(z1)
        z2 = w2 @ a1
        a2 = fx(z2)
        yh = w3 @ a2
        yhList.append(yh[0][0])

    return yhList


def nn2(iter, featureN, _x, _y):
    weightMatrix1 = np.ones((featureN, featureN))
    weightMatrix2 = np.ones((1, featureN))
    for k in range(iter):
        for i in range(len(_x)):
            curX = x[i].reshape(featureN, 1)
            curY = _y[0][i].reshape(1,1)
            print('curX: ', curX.shape)
            print('curX: ', curX)
            print('curY: ', curY.shape)
            z1 = weightMatrix1 @ curX
            a1 = fx(z1)
            print('z1: ', z1.shape)
            print('a1: ', a1.shape)
            print('w2: ', weightMatrix2.shape)
            yh = weightMatrix2 @ a1
            print('yh: ', yh.shape)
            print('y: ', curY.shape)
            
def MSE(y, yh, n):
    yy = (y - yh)**2
    return np.sum(yy) * (1/n)


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

w1, w2, w3 = nn3(1, len(X_train.T), X_train, y_train, .05)
# predict_train = nn3Predict(w1, w2, w3, X_train)
predict_test = nn3Predict(w1, w2, w3, X_test)
print(predict_test)

# for a, b in zip(y_train, predict_train):
#     print(a, b)

# print('MSE: ', MSE(y_train, predict_train, len(y_train)))

for a, b in zip(y_test, predict_test):
    print(a, b)
# print(predict_test)

print('MSE: ', MSE(y_test, predict_test, len(y_test)))