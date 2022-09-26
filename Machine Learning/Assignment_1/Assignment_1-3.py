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

def h(theta, x):
    s = theta[0]
    for i in range(1, len(theta)):
        s += theta[i] * x[i]
    return s

def J(theta, x, y):
    s = 0
    for i in range(1, len(y)):
        s += ((h(theta, x[i]) - y[i])**2)
    return s/(2* len(y))

def costFunc(theta, x, y, jIndex, alpha, n):
    sum = 0
    for i in range(n):
        sum += (h(theta, x[i]) - y[i]) * x[i][jIndex]
    return alpha * (sum/n)

def SGD(iteration, theta, x, y, alpha):
    for i in range(iteration):
        for n in range(1, len(y)):
            curH = h(theta, x[n])
            for j in range(len(theta)):
                tj = (curH - y[n]) * x[n][j]
                theta[j] = theta[j] - (alpha * tj)
        stochasticThetaList.append(theta.tolist())
    return theta

def W(x, datum, bandwidth):
    wList = []
    for i in range(len(x)):
        a = -(((x[i] - datum)**2) / (2*(bandwidth**2)))
        wList.append(np.exp(a))
    return wList

def getTrainingData(x, y, theta):
    s = 0
    thetaXList = []
    for i in range(len(y)):
        # s += ((h(theta, x[i]) - y[i])**2)
        thetaXList.append(((y[i] - h(theta, x[i]))**2))
    # return s * .5
    return thetaXList

def LWRS(iteration, wList, theta, x, y):
    for i in range(iteration):
        for n in range(1, len(y)):
            curH = h(theta, x[n])
            for j in range(len(theta)):
                tj = wList[n] * (curH - y[n]) * x[n][j]
                theta[j] = theta[j] - (alpha * tj)
        stochasticThetaList.append(theta.tolist())
    return theta
# def BGD(iteration, theta, x, y, alpha):
#     for i in range(1, iteration):
#         curTheta = theta
#         for j in range(len(theta)):
#             curTheta[j] = theta[j] - costFunc(theta, x, y, j, alpha, len(y))
#         theta = curTheta
#         batchThetaList.append(theta.tolist())
#     return theta

def testOutput(num, x, y):
    a = theta[0]
    for i in range(1, len(theta)):
        a += x[num][i] * theta[i]
    print(str(num) + ' = ' + str(a))
    print(str(num) + ' = ' + str(y[num]))
    return a

def MSE(x, y, theta):
    s = 0
    for i in range(len(y)):
        s += ((h(theta, x[i]) - y[i])**2) * (1/len(y))
    return s/len(y)

def graphJ(thetaB, thetaS, x, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
    for j in range(len(thetaB)):
        ax[0].scatter(j, J(thetaB[j], x, y), s=100, alpha=1, edgecolors="k")
        ax[1].scatter(j, J(thetaS[j], x, y), s=100, alpha=1, edgecolors="k")
    ax[0].set_xlabel(r'$Number of iteration$')
    ax[0].set_ylabel(r'$J(θ)$')
    ax[0].set_title('Batch Gradiant For Every Iterative')
    ax[0].legend(loc='upper left', fontsize='small')
    ax[1].set_xlabel(r'$Number of iteration$')
    ax[1].set_ylabel(r'$J(θ)$')
    ax[1].set_title('Stochastic Gradiant For Every Iterative')
    ax[1].legend(loc='upper left', fontsize='small')

    plt.show()

crabDataTraining = crabData[:2603]
crabDataTesting = crabData[2603:]

age = np.asarray(crabDataTraining['Age'])
sex = np.asarray(crabDataTraining['Sex'])
length = np.asarray(crabDataTraining['Length'])
diameter = np.asarray(crabDataTraining['Diameter'])
height = np.asarray(crabDataTraining['Height'])
weight = np.asarray(crabDataTraining['Weight'])
shuckedWeight = np.asarray(crabDataTraining['Shucked Weight'])
visceraWeight = np.asarray(crabDataTraining['Viscera Weight'])
shellWeight = np.asarray(crabDataTraining['Shell Weight'])

t_age = np.asarray(crabDataTesting['Age'])
t_sex = np.asarray(crabDataTesting['Sex'])
t_length = np.asarray(crabDataTesting['Length'])
t_diameter = np.asarray(crabDataTesting['Diameter'])
t_height = np.asarray(crabDataTesting['Height'])
t_weight = np.asarray(crabDataTesting['Weight'])
t_shuckedWeight = np.asarray(crabDataTesting['Shucked Weight'])
t_visceraWeight = np.asarray(crabDataTesting['Viscera Weight'])
t_shellWeight = np.asarray(crabDataTesting['Shell Weight'])

trainingFeatures = np.asarray([[a,b,c,d,e,f,g,h,i] for a,b,c,d,e,f,g,h,i in zip(age, sex, length, diameter, height, weight, shuckedWeight, visceraWeight, shellWeight)])
testingFeatures = np.asarray([[a,b,c,d,e,f,g,h,i] for a,b,c,d,e,f,g,h,i in zip(t_age, t_sex, t_length, t_diameter, t_height, t_weight, t_shuckedWeight, t_visceraWeight, t_shellWeight)])
trainingOutput = np.asarray([a for a in age])
testingOutput = np.asarray([a for a in t_age])

# theta = np.random.uniform(0.0, 1.0, size=9)
# #theta = np.ones([9,1])
# alpha = .001
# batchThetaList = [theta.tolist()]
# iteration = 100

# BGD(iteration, theta, trainingFeatures, trainingOutput, alpha)
# print(MSE(testingFeatures, testingOutput, theta))

theta = np.random.uniform(0.0, 1.0, size=9)
#theta = np.ones([9,1])
alpha = .001
stochasticThetaList = [theta.tolist()]
iteration = 10

# SGD(iteration, theta, trainingFeatures, trainingOutput, alpha)
# print(MSE(testingFeatures, testingOutput, theta))

predictedList = getTrainingData(testingFeatures, testingOutput, theta)

wList = W(predictedList, np.average(testingOutput), .856)

LWRS(iteration, wList, theta, testingFeatures, testingOutput)
print('MSE LWR Sochastic Gradient: ')
print(MSE(testingFeatures, testingOutput, theta))