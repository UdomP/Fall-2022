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

def cost_func(theta1, x, y):
    """The cost function, J(theta1) describing the goodness of fit."""
    theta1 = np.atleast_2d(np.asarray(theta1))
    return np.average((hypothesis(x, theta1) - y)**2, axis=1)/2

def hypothesis(x, theta1):
    """Our "hypothesis function", a straight line through the origin."""
    return theta1*x

# def SGD(N, alpha, thetaJ):
#     for i in range(N):
#         for j in range(len(thetaJ)):
#             thetaJ[j] = thetaJ[j] - alpha * ((thetaJ[j] * crabDataTraining['Length'][i]) - crabDataTraining['Age'][i]) * crabDataTraining['Length'][i]

#     print(thetaJ)

# def SGD(N, alpha, featureList):
#     newThetaJ = [0, 0, 0, 0, 0, 0, 0, 0]
#     for i in range(N):
#         for j in featureList:
#             print(featureList[j])
#             newThetaJ[j] = newThetaJ[j] - alpha * ((newThetaJ[j] * crabDataTraining[featureList[j]][i]) - crabDataTraining['Age'][i]) * crabDataTraining[featureList[j]][i]
#     print(newThetaJ)

# J = []
# ThetaJ = []
# def SGD(N, alpha, featureList):
#     newThetaJ = [0, 0, 0, 0, 0, 0, 0, 0]
#     for i in range(N):
#         for index, feature in enumerate(featureList):
#             newThetaJ[index] = newThetaJ[index] - alpha * ((newThetaJ[index] * crabDataTraining[feature][i]) - crabDataTraining['Age'][i]) * crabDataTraining[feature][i]
#         for j in newThetaJ:
#             cost_func(j, )
#     print(newThetaJ)

# featureList = ["Sex", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]
# SGD(8, 1, featureList)

def JFunction(theta, xList, yList):
    sum = 0
    for x, y in zip(xList, yList):
        sum += ((x * theta) - y)**2
    return (.5 * sum)

ThetaList = []
JList = []
def SGD(N, alpha, featureList):
    newThetaJ = [0] * len(featureList)
    newJ = [0] * len(featureList)
    for i in range(N):
        for index, feature in enumerate(featureList):
            newThetaJ[index] = newThetaJ[index] - alpha * ((newThetaJ[index] * crabDataTraining[feature][i]) - crabDataTraining['Age'][i]) * crabDataTraining[feature][i]  
            newJ[index] = JFunction(newThetaJ[index], crabDataTraining[feature][:N], crabDataTraining['Age'][:N])
        ThetaList.extend(newThetaJ)
        JList.extend(newJ)
        
N = 20
alpha = 1
featureList = ["Sex", "Length", "Diameter", "Height", "Weight", "Shucked Weight", "Viscera Weight", "Shell Weight"]
SGD(N, alpha, featureList)

ThetaNList = [ThetaList[x:x+8] for x in range(0, len(ThetaList), 8)]
JNList = [JList[x:x+8] for x in range(0, len(JList), 8)]

print('lllllllll \n')
print(ThetaNList)
print(JNList)

def splitRow(list, rowNum):
    newList = []
    for i in list:
        newList.append(i[rowNum])
    return newList

def GrapthData(tData, jData):
    # The data to fit
    m = 20
    theta1_true = 0.5
    x = np.linspace(-1,1,m)
    y = theta1_true * x

    # The plot: LHS is the data, RHS will be the cost function.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
    ax[0].scatter(x, y, marker='x', s=40, color='k')

    # First construct a grid of theta1 parameter pairs and their corresponding
    # cost function values.
    theta1_grid = np.linspace(-0.2,1,50)
    J_grid = jData

    # The cost function as a function of its single parameter, theta1.
    ax[1].plot(theta1_grid, J_grid, 'k')

    # Annotate the cost function plot with coloured points indicating the
    # parameters chosen and red arrows indicating the steps down the gradient.
    # Also plot the fit function on the LHS data plot in a matching colour.
    colors = ['b', 'g', 'm', 'c', 'orange']
    ax[0].plot(x, hypothesis(x, tData[0]), color=colors[0], lw=2,
            label=r'$\theta_1 = {:.3f}$'.format(tData[0]))
    for j in range(1,N):
        ax[1].annotate('', xy=(tData[j], jData[j]), xytext=(tData[j-1], jData[j-1]),
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center')
        ax[0].plot(x, hypothesis(x, tData[j]), color=colors[j], lw=2,
                label=r'$\theta_1 = {:.3f}$'.format(tData[j]))

    # Labels, titles and a legend.
    ax[1].scatter(tData, J, c=colors, s=40, lw=0)
    ax[1].set_xlim(-0.2,1)
    ax[1].set_xlabel(r'$\theta_1$')
    ax[1].set_ylabel(r'$J(\theta_1)$')
    ax[1].set_title('Cost function')
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_title('Data and fit')
    ax[0].legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

# GrapthData(splitRow(ThetaNList, 1), splitRow(JNList, 1))


def ScatterPlotRegression(tData, jData):
    # Generate data
    x = crabData['Length']
    y = crabData['Age']

    # Initialize layout
    fig, ax = plt.subplots(figsize = (9, 9))

    # Add scatterplot
    ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")

    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(x, y, 1)

    #add linear regression line to scatterplot 
    plt.plot(x, m*x+b)

    f = lambda x: tData[0]*x
    plt.plot(x,f(x),lw=2.5, c='b',label='Line ')

    f = lambda x: tData[1]*x
    plt.plot(x,f(x),lw=2.5, c='g',label='Line ')

    f = lambda x: tData[2]*x
    plt.plot(x,f(x),lw=2.5, c='r',label='Line ')

    f = lambda x: tData[3]*x
    plt.plot(x,f(x),lw=2.5, c='c',label='Line ')

    f = lambda x: tData[4]*x
    plt.plot(x,f(x),lw=2.5, c='m',label='Line ')

    f = lambda x: tData[5]*x
    plt.plot(x,f(x),lw=2.5, c='y',label='Line ')

    f = lambda x: tData[6]*x
    plt.plot(x,f(x),lw=2.5, c='k',label='Line ')

    f = lambda x: tData[7]*x
    plt.plot(x,f(x),lw=2.5, c='w',label='Line ')

    plt.show()

# ScatterPlotRegression(splitRow(ThetaNList, 0), splitRow(JNList, 0))
ScatterPlotRegression(splitRow(ThetaNList, 1), splitRow(JNList, 1))
# for i in range(5):
#     sum = 0
#     for j in splitRow(ThetaNList, 1):
#         sum += crabData['Length'][i] * j
#     print(sum)


sum = crabData[featureList[0]][1] * ThetaNList[0][0] + crabData[featureList[1]][1] * ThetaNList[0][1]  + crabData[featureList[2]][1] * ThetaNList[0][2]  + crabData[featureList[1]][1] * ThetaNList[0][1]  + crabData[featureList[1]][1] * ThetaNList[0][1]  + crabData[featureList[1]][1] * ThetaNList[0][1]  + crabData[featureList[1]][1] * ThetaNList[0][1]  + crabData[featureList[1]][1] * ThetaNList[0][1]  + crabData[featureList[1]][1] * ThetaNList[0][1] 

# ScatterPlotRegression(splitRow(ThetaNList, 2), splitRow(JNList, 2))
# ScatterPlotRegression(splitRow(ThetaNList, 3), splitRow(JNList, 3))
# ScatterPlotRegression(splitRow(ThetaNList, 4), splitRow(JNList, 4))
# ScatterPlotRegression(splitRow(ThetaNList, 5), splitRow(JNList, 5))
# ScatterPlotRegression(splitRow(ThetaNList, 6), splitRow(JNList, 6))
# ScatterPlotRegression(splitRow(ThetaNList, 7), splitRow(JNList, 7))