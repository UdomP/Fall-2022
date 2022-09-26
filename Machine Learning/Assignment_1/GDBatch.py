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

def costFunc(theta, x, y, jIndex, alpha, n):
    sum = 0
    for i in range(n):
        sum += (h(theta, x[i]) - y[i]) * x[i][jIndex]
    return alpha * (sum/n)

def J(theta, x, y):
    s = 0
    for i in range(1, len(y)):
        s += (h(theta, x[i]) - y[i])**2
    return s/(2* len(y))

def BGD(iteration, theta, x, y, alpha):
    for i in range(1, iteration):
        curTheta = theta
        for j in range(len(theta)):
            curTheta[j] = theta[j] - costFunc(theta, x, y, j, alpha, len(y))
        theta = curTheta
        thetaList.append(theta.tolist())
    return theta

def testOutput(num, x, y):
    a = 0
    for i in range(len(theta)):
        a += x[num][i] * theta[i]
    print(str(num) + ' = ' + str(a))
    print(str(num) + ' = ' + str(y[num]))
    return a

def graphJ(theta, x, y):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6.15))
    for j in range(len(theta)):
        ax.scatter(j, J(theta[j], x, y), s=100, alpha=1, edgecolors="k")
    ax.set_xlabel(r'$Number of iteration$')
    ax.set_ylabel(r'$J(θ)$')
    ax.set_title('Batch Gradiant For Every Iterative')
    ax.legend(loc='upper left', fontsize='small')

    plt.show()
    
def MSE(x, y, theta):
    s = 0
    for i in range(len(y)):
        s += (h(theta, x[i]) - y[i])**2
    return s/len(y)

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

theta = np.random.uniform(0.0, 1.0, size=9)
#theta = np.ones([9,1])
alpha = .0001
thetaList = [theta.tolist()]
iteration = 100
# p = a = np.zeros((1, 9))
# thetaList = np.concatenate((thetaList, p))
# thetaList = np.concatenate((thetaList, p))
# print

BGD(iteration, theta, trainingFeatures, trainingOutput, alpha)
print(MSE(testingFeatures, testingOutput, theta))
graphJ(thetaList, testingFeatures, testingOutput)

testOutput(5, trainingFeatures, trainingOutput)
testOutput(4, trainingFeatures, trainingOutput)
testOutput(3, trainingFeatures, trainingOutput)
testOutput(10, trainingFeatures, trainingOutput)