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

def h(theta, x):
    s = theta[0]
    for i in range(1, len(theta)):
        s += theta[i] + x[i]
    return s

def costFunc(theta, x, y, alpha):
    newTheta = []
    for i in range(len(theta)):
        newTheta.append(theta[i] - (alpha * sum((h(theta, x) - y) * x[i])))
    return newTheta

N = 5
alpha = .5
featureList = ["Sex", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]
x = []
y = crabDataTraining['Age']
l = [[]]
ThetaJ = [[]]
nTraining = 1

# Initialize theta at i = 0 for each element/feature
for i, j in enumerate(featureList):
    l[0].append(0)
    x.append(crabDataTraining[j])

ThetaJ.append(costFunc(l, x, y, alpha))
print(ThetaJ)
print(h(ThetaJ, x))
print(sum(h(ThetaJ, x)))
print(costFunc(ThetaJ, x, y, alpha))
