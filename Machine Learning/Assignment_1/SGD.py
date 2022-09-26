import this
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.pyplot import cm

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

crabDataTraining = crabData[:2603]
crabDataTesting = crabData[2603:]

def h(theta, x, row):
    s = 0
    for i, j in enumerate(featureList):
        s += theta[i] * x[i][row]
    return s

def SGD(n, x, y, alpha):
    for i in range(1, n):
        thetaList = []
        for index, j in enumerate(featureList):
            # curTheta = ThetaJ[-1][index] - alpha * (((ThetaJ[-1][index] * x[index][i]) - y[i]) * x[index][i])
            curTheta = ThetaJ[-1][index] - alpha * ((h(ThetaJ[-1], ) - y[i]) * x[index][i])
            thetaList.append(curTheta)
        ThetaJ.append(thetaList)

N = 10
alpha = .5
featureList = ["Sex", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]
x = []
y = crabDataTraining['Age']
ThetaJ = [[]]
nTraining = 1

# Initialize theta at i = 0 for each element/feature
for i, j in enumerate(featureList):
    ThetaJ[0].append(0 - alpha * (((0 * crabDataTraining[j][0]) - y[0]) * crabDataTraining[j][0]))
    x.append(crabDataTraining[j])

SGD(N, x, y, alpha)
for a in ThetaJ:   
    print('===========================')
    print(a)

cc = 0
for i, j in enumerate(featureList):
    cc += x[i][5] * ThetaJ[-1][i]
    print(j + ' : ' +  str(x[i][5] * ThetaJ[-1][i]) + ' = ' + str(y[5]))
print(cc)