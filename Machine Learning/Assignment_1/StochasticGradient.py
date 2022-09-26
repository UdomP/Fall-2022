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

print(crabData)
print(crabDataTraining)
print(crabDataTesting)
print(crabDataTraining['Length'][1])

def sumOfSGD(n, thetaIndex):
    z = crabData
    sum = 0
    for index, i in enumerate(featureList):
        print(z[i][n])
        print(ThetaJ[index][thetaIndex])
        sum += z[i][n] * ThetaJ[index][thetaIndex]
    print('Y: ' + str(y[n]))
    print('X: ' + str(sum))

def lossFunc(theta, x, y):
    return (alpha * ((theta * x) - y) * x)

def SGD():
    for i in range(1, N):
        for index, j in enumerate(featureList):
            lastTheta = ThetaJ[index][-1]
            curTheta = lastTheta - (lossFunc(lastTheta, x[j][i], y[i]))
            ThetaJ[index].append(curTheta)
N = 50
alpha = 1
featureList = ["Sex", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]
x = crabDataTraining
y = crabDataTraining['Age']
ThetaJ = []
lol = 0
# Initialize theta at i = 0 for each element/feature
for index, feature in enumerate(featureList):
    ThetaJ.append([-lossFunc(0, x[feature][0], y[0])])
SGD()

print(ThetaJ)

def ScatterPlotRegression(dataName, thetaIndex, range):
    x = crabDataTesting[dataName]
    y = crabDataTesting['Age']

    fig, ax = plt.subplots(figsize = (9, 9))

    # Add scatterplot
    ax.scatter(x, y, s=100, alpha=1, edgecolors="k")

    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(x, y, 1)

    #add linear regression line to scatterplot 
    plt.plot(x, m*x+b)

    # for i, j in enumerate(featureList):
    #     f = lambda x: tData[0]*x
    #     plt.plot(x,f(x),lw=2.5, c='b',label='Line ')

    color = cm.rainbow(np.linspace(0, 1, len(range)))
    for i, c in zip(range, color):
        f = lambda x: ThetaJ[thetaIndex][i]*x
        plt.plot(x,f(x),lw=2.5, c=c,label=str(i))

    ax.set_title(dataName)
    ax.legend(loc='upper left', fontsize='small')
    plt.show()

# ScatterPlotRegression('Length', 1, range(1,4))
# ScatterPlotRegression('Length', 1, range(4,15))

# ScatterPlotRegression('Diameter', 2, range(1,4))

ScatterPlotRegression('Weight', 4, range(4,5))