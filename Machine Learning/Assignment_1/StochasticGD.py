from operator import length_hint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crabData = pd.read_csv('CrabAgePrediction.csv')

def IdentifyCrabSex(str):
    if(str == 'F'):
        return 0
    elif(str == 'M'):
        return 1
    elif(str == 'I'):
        return 2

def CleanSexData():
    for i in crabData['Sex']:
        crabData['Sex'] = crabData['Sex'].replace([i], IdentifyCrabSex(i))
CleanSexData()

def h(x, theta):
    return np.dot(np.transpose(theta),x)

def SGD(x, y, theta, m, alpha, N):
    for N in range(N):
        for i in range(m):
            curH  = h(x[i], theta)
            for j in range(len(theta)):
                tj = (h(x[i], theta) - y[i]) * x[i][j]
                tj *= 1/m
                theta[j] = theta[j] -  (alpha * tj)
    return theta

def testOutput(num, x, y):
    a = 0
    for i in range(len(theta)):
        a += x[num][i] * theta[i]
    print(str(num) + ' = ' + str(a))
    print(str(num) + ' = ' + str(y[num]))

age = np.asarray(crabData['Age'])
sex = np.asarray(crabData['Sex'])
length = np.asarray(crabData['Length'])
diameter = np.asarray(crabData['Diameter'])
height = np.asarray(crabData['Height'])
weight = np.asarray(crabData['Weight'])
shuckedWeight = np.asarray(crabData['Shucked Weight'])
visceraWeight = np.asarray(crabData['Viscera Weight'])
shellWeight = np.asarray(crabData['Shell Weight'])

temp = np.asarray([[a,b,c,d,e,f,g,h] for a,b,c,d,e,f,g,h in zip(sex, length, diameter, height, weight, shuckedWeight, visceraWeight, shellWeight)]) # Gets our features
training_features, test_features = temp[:int(len(temp) * 0.9)], temp[int(len(temp) * 0.1):] # Splits our features in half between training and testing
temp = np.asarray([a for a in age]) # Gets our output
training_output, test_output = temp[:int(len(temp) * 0.9)], temp[int(len(temp) * 0.1):]  # Splits our outputs in half between training and testing

theta = np.random.uniform(0.0, 1.0, size=8)
alpha = .001

t = SGD(training_features, training_output, theta, len(training_output), alpha, 150)
print(theta)
testOutput(5, training_features, training_output)
testOutput(4, training_features, training_output)
testOutput(3, training_features, training_output)
testOutput(10, training_features, training_output)
testOutput(10, test_features, test_output)
testOutput(100, test_features, test_output)
testOutput(82, test_features, test_output)
