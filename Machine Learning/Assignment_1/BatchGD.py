from operator import length_hint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def h(x, theta):
    return np.dot(np.transpose(theta),x)

def BGD(x, y, theta, n, alpha):
    for j in range(len(theta)):
        gradient = 0
        for i in range(n):
            gradient += (h(x[i], theta) - y[i]) * x[i][j]
        gradient *= 1/n
        theta[j] = theta[j] -  (alpha * gradient)
    return theta

def GraphGradient(x, y, theta1):
    # The plot: LHS is the data, RHS will be the cost function.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
    ax[0].scatter(x, y, marker='x', s=100, color='k')
    # Annotate the cost function plot with coloured points indicating the
    # parameters chosen and red arrows indicating the steps down the gradient.
    # Also plot the fit function on the LHS data plot in a matching colour.
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    ax[0].plot(x, h(x, theta1[0]), color=colors[0], lw=2,
            label=r'$\theta_1 = {:.3f}$'.format(theta1[0]))
    for j in range(1, len(theta1)):
        ax[0].plot(x, h(x, theta1[j]), color=colors[j], lw=2,
                label=r'$\theta_1 = {:.3f}$'.format(theta1[j]))

    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_title('Data and fit')
    ax[0].legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

def testOutput(num, x, y):
    a = 0
    for i in range(len(theta)):
        a += x[num][i] * theta[i]
    print(str(num) + ' = ' + str(a))
    print(str(num) + ' = ' + str(y[num]))
    return a

def hypothesis(theta, x):
    sum = []
    for i in range(len(x)):
        sum.append(theta[1] * x[i][1] + theta[2] * x[i][2] + theta[3] * x[i][3] + theta[4] * x[i][4] + theta[5] * x[i][5] + theta[6] * x[i][6] + theta[7] * x[i][7])
    return sum

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

trainingFeatures = np.asarray([[a,b,c,d,e,f,g,h] for a,b,c,d,e,f,g,h in zip(sex, length, diameter, height, weight, shuckedWeight, visceraWeight, shellWeight)])
testingFeatures = np.asarray([[a,b,c,d,e,f,g,h] for a,b,c,d,e,f,g,h in zip(t_sex, t_length, t_diameter, t_height, t_weight, t_shuckedWeight, t_visceraWeight, t_shellWeight)])
trainingOutput = np.asarray([a for a in age])
testingOutput = np.asarray([a for a in t_age])

theta = np.random.uniform(0.0, 1.0, size=8)
alpha = .0001
N = 20
theta1 = [np.array((0,0,0,0,0,0,0,0))]

for n in range(N):
    BGD(trainingFeatures, trainingOutput, theta, len(trainingOutput), alpha)
    theta1.append(theta)
    
GraphGradient(hypothesis([1,1,1,1,1,1,1,1], testingFeatures), testingOutput, theta1[-1])

testOutput(5, trainingFeatures, trainingOutput)
testOutput(4, trainingFeatures, trainingOutput)
testOutput(3, trainingFeatures, trainingOutput)
testOutput(10, trainingFeatures, trainingOutput)

