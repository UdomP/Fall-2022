from matplotlib import testing
import pandas as pd
import numpy as np

infile=open('17701310.txt','r+')

theta = np.random.uniform(0.0, 1.0, size=4)
alpha = .01

data = []
output = []
for w in infile.read().split():
    n = w.split(',')
    data.append([1, int(n[0]), int(n[1]), int(n[2])])
    output.append([int(n[3])])

# trainingData = np.asarray(data[1:int(len(data) * .85)])
# testingData = np.asarray(data[int(len(data) * .85):])

# trainingOutput = np.asarray(output[1:int(len(output) * .85)])
# testingOutput = np.asarray(output[int(len(output) * .85):])

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
        # newTheta[j] = newTheta[j] + alpha * (s/len(y))
        newTheta[j] = newTheta[j] + alpha * (s)
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
        predict = h(x, t, i)
        if(actual == 1 and predict == 1):
            TP += 1
        elif(actual == 2 and predict == 2):
            TN += 1
        elif(actual == 2 and predict == 1):
            FP += 1
        elif(actual == 1 and predict == 2):
            FN += 1
    return TP,TN,FP,FN

nTrainingData = normalization(trainingData)
nTestingData = normalization(testingData)

theta = train(100, nTrainingData, trainingOutput, alpha, theta)
print(theta)

TP,TN,FP,FN = predict(nTestingData, testingOutput, theta)
print("True Positive = " + str(TP))
print("True Negative = " + str(TN))
print("False Positive = " + str(FP))
print("False Negative = " + str(FN))
print("Total = " + str(TP+TN+FP+FN))

precision = TP/(TP + FP)
recall = TP/(TP + FN)

f1Score = 2 * (precision * recall)/(precision + recall)
print("F1-Score = " + str(f1Score))