from matplotlib import testing
import pandas as pd
import numpy as np

infile=open('17701310.txt','r+')

theta = np.asarray([.5,.5,.5,.5])
alpha = .005
iteration = 400
# alpha = .00255
# iteration = 500
# alpha = .00952
# iteration = 900

data = []
output = []
for w in infile.read().split():
    n = w.split(',')
    data.append([1, float(n[0]), float(n[1]), float(n[2])])
    if n[3] == '2':
        output.append(1.0)
    else:
        output.append(0.0)

infile.close()

trainingData = np.asarray([i for i in data[1: int(len(data) * .85)]])
testingData = np.asarray([i for i in data[int(len(data) * .85):]])

trainingOutput = np.asarray([i for i in output[1:int(len(output) * .85)]])
testingOutput = np.asarray([i for i in output[int(len(output) * .85):]])

def normalization(x):
    newX = x
    for j in range(1, len(x[0])):
        mean = np.mean(x[:,j])
        std = np.std(x[:,j])
        for i in range(len(x)):
            newX[i][j] = (x[i][j] - mean)/std
    return newX

def h(x, t, index):
    thetaX = np.dot(np.transpose(t),x[index])
    return 1/(1 + np.exp(-thetaX))

def gradientAsscent(x, y, t, alpha):
    newTheta = t
    for j in range(len(t)):
        s = 0
        for i in range(len(y)):
            s += (y[i] - h(x, t, i)) * x[i][j]
        newTheta[j] = newTheta[j] + alpha * (s/len(y))
        # newTheta[j] = newTheta[j] + alpha * (s)
    return newTheta

def train(iteration, x, y, alpha, t):
    th = t
    for i in range(iteration):
        th = gradientAsscent(x, y, th, alpha)
    return th

def predict(x, y, t):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        actual = y[i]
        predict = np.round(h(x, t, i))
        # predict = h(x, t, i)
        # print("a = " + str(actual))
        # print("p = " + str(predict))
        if(actual == 1 and predict == 1):
            TP += 1
        elif(actual == 0 and predict == 0):
            TN += 1
        elif(actual == 0 and predict == 1):
            FP += 1
        elif(actual == 1 and predict == 0):
            FN += 1
    return TP,TN,FP,FN

normalizedData = normalization(np.asarray(data))
nTrainingData = np.asarray(normalizedData[0: int(len(normalizedData) * .85)])
nTestingData = np.asarray(normalizedData[int(len(normalizedData) * .85):])

print('Maximum Likelihood Estimation with normalized Data')
print("Alpha = " + str(alpha))
print("Iteration = " + str(iteration))
theta = train(iteration, nTrainingData, trainingOutput, alpha, theta)
print('Theta = ', end = '')
print(theta)

TP,TN,FP,FN = predict(nTestingData, testingOutput, theta)
print("True Positive = " + str(TP))
print("True Negative = " + str(TN))
print("False Positive = " + str(FP))
print("False Negative = " + str(FN))
print("Total = " + str(TP+TN+FP+FN))

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print("precision = " + str(precision))
print("recall = " + str(recall))
f1Score = 2 * (precision * recall)/(precision + recall)
print("F1-Score = " + str(f1Score))