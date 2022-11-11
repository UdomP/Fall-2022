import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crabData = pd.read_csv('CrabAgePrediction.csv')
crabDataSize = len(crabData)

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

crabDicTraining = {}
crabDicTesting = {}
l = []
for cd in crabData:
    l.append(cd)
    crabDicTraining[cd] = crabData[cd][:int(crabDataSize*.9)]
    crabDicTesting[cd] = crabData[cd][int(crabDataSize*.9):]
# ['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Age']
# featureList = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
featureList = ['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
# featureList = ['Length', 'Diameter', 'Height', 'Weight']
# featureList = ['Length', 'Height']

trainingLen = int(crabDataSize*.9)
testingLen = int(crabDataSize*.1)
featureDataTraining = [crabDicTraining[f] for f in featureList]
featureDataTesting = [crabDicTesting[f] for f in featureList]
OutputDataTraining = [crabDicTraining['Age']]
OutputDataTesting = [crabDicTesting['Age']]

training = np.array(featureDataTraining)
output = np.array(OutputDataTraining)
trainingTest = np.array(featureDataTesting)
outputTest = np.array(OutputDataTesting)

nnLayerLen = 3
nnFeatureLen = len(featureList)

thetaMatrix = np.ones((nnFeatureLen, 1)) * .5
weightMatrix = np.ones((nnFeatureLen, nnFeatureLen)) * .5
weightMatrix2 = np.ones((nnFeatureLen, nnFeatureLen)) * .5
# weightMatrix3 = np.ones((nnFeatureLen, 1))
weightMatrix3 = np.ones((1, nnFeatureLen)) * .5

# weightMatrix = np.random.uniform(0,1,(nnFeatureLen, nnFeatureLen))
# print(weightMatrix)
zMatrix = np.zeros((nnFeatureLen, 1))

JList = []
alpha = 0.01

######################################## Data and Configurations #########################################################

def getZ(wMatrix, xMatrix):
    tempX = xMatrix.reshape(len(training),1)
    return wMatrix @ tempX

def sigmoidtheta(z, x):
    return (1/(1 + (np.exp(-(np.transpose(z) @ x)))))

def sigmoidtheta1(z, x):
    return (np.transpose(z) @ x)

def sigmoid(z):
    return (1/(1 + (np.exp(-(z)))))

def fx(zM):
    temp = np.exp(-zM)
    return 1/(1+temp)

def yHat(w, a):
    return w @ a

# , w2, w3, x, y, yhat, z1
def layer1Update(w1, w2, w3, z1, z2, x, y, yhat):
    tempW1 = w1.copy()
    for rowIndex, row in enumerate(w1):
        jz = yhat - y
        a1z1 = sigmoid(z1[rowIndex])
        z1w11 = x[rowIndex]
        const = jz * a1z1 * z1w11
        for colIndex, col in enumerate(row):
            s = 0
            for i in range(nnFeatureLen):
                s += w3[0][i] * (sigmoid(z2[i]) * (1 - sigmoid(z2[i]))) * w2[rowIndex][i]
            tempW1[rowIndex][colIndex] = const * s
    return tempW1

def layer2Update(w2, w3, z2, y, yhat):
    tempW2 = w2.copy()
    for rowIndex, row in enumerate(w2):
        jz = yhat - y
        const = jz
        for colIndex, col in enumerate(row):
            s = 0
            for i in range(nnFeatureLen):
                s += w3[0][i] * (sigmoid(z2[i]) * (1 - sigmoid(z2[i]))) * w2[rowIndex][i]
            tempW2[rowIndex][colIndex] = const * s
    return tempW2

def layer3Update(w3, y, yhat):
    tempW3 = w3.copy()
    for rowIndex, row in enumerate(w3):
        jz = yhat - y
        const = jz
        for colIndex, col in enumerate(row):
            tempW3[rowIndex][colIndex] = jz * w3[0][colIndex]
    return tempW3

def MSE(y, yh, n):
    yy = (y - yh)**2
    return np.sum(yy) * (1/n)

def J(y, yHat):
    temp = np.abs(yHat - y)
    return  0.5 * (temp**2)

def JTheta(y, yhat):
    temp = (np.abs(yhat - y))**2
    return np.sum(temp[0]) * (1/len(temp[0]))

def graphJ(j):
    plt.plot([j for j in range(len(j))], j.reshape(len(j)))
    plt.xlabel('Number of iterations')
    plt.ylabel('J(θ)')
    plt.title('Neural Network For Every Iterations')
    plt.legend(loc='upper left', fontsize='small')

    plt.show()
    plt.close()

def neuralNetwork():
    y = []
    yhat = []
    j = 0
    for i in range(trainingLen):
        tempTraining = training[:, i].reshape(len(training),1)
        tempOutput = output[:, i].reshape(len(output),1)
        global weightMatrix
        global weightMatrix2
        global weightMatrix3

        z = getZ(weightMatrix, tempTraining)
        a = fx(z)

        z2 = getZ(weightMatrix2, a)
        a2 = fx(z2)
        
        yh = yHat(weightMatrix3, a2)

        y.append(tempOutput[0][0])
        yhat.append(yh[0][0])

        j += J(tempOutput[0][0], yh[0][0])

        tempWeight1 = layer1Update(weightMatrix, weightMatrix2, weightMatrix3, z, z2, tempTraining, tempOutput, yh)
        weightMatrix = weightMatrix - (alpha * tempWeight1)
        tempWeight2 = layer2Update(weightMatrix2, weightMatrix3, z2, tempOutput, yh)
        weightMatrix2 = weightMatrix2 - (alpha * tempWeight2)
        tempWeight3 = layer3Update(weightMatrix3, tempOutput, yh)
        weightMatrix3 = weightMatrix3 - (alpha * tempWeight3)
    return y, yh, j

def neuralNetworkTest():
    yhList = []
    yList = []
    jList = []
    for i in range(testingLen):
        tempTraining = trainingTest[:, i].reshape(len(trainingTest),1)
        tempOutput = outputTest[:, i].reshape(len(outputTest),1)
        global weightMatrix
        global weightMatrix2
        global weightMatrix3

        z = getZ(weightMatrix, tempTraining)
        a = fx(z)

        z2 = getZ(weightMatrix2, a)
        a2 = fx(z2)
        
        yh = yHat(weightMatrix3, a2)

        # yh = J(tempOutput, z3)
        # print(yh)
        # print(j)
        # tempWeight1 = layer1Update(weightMatrix, weightMatrix2, weightMatrix3, z, z2, tempTraining, tempOutput, yh)
        # weightMatrix = weightMatrix - (alpha * tempWeight1)
        # tempWeight2 = layer2Update(weightMatrix2, weightMatrix3, z2, tempOutput, yh)
        # weightMatrix2 = weightMatrix2 - (alpha * tempWeight2)
        # tempWeight3 = layer3Update(weightMatrix3, tempOutput, yh)
        # weightMatrix3 = weightMatrix3 - (alpha * tempWeight3)

        yhList.append(yh[0][0])
        yList.append(tempOutput[0][0])
        # jList.append(z3[0])
    return yhList, yList, jList

JJ = []
JJJ = []
for i in range(3):
    print('iteration = ', i)
    y1, yh2, jjj = neuralNetwork()
    JJJ.append(jjj)
    JJ.append(JTheta(y1, yh2))
print('Final weightMatrix 1 : ', weightMatrix)
print('Final weightMatrix 2 : ', weightMatrix2)
print('Final weightMatrix 3 : ', weightMatrix3)

YhatList, YList, JList = neuralNetworkTest()

print(YhatList)
print(YList)
print('MSE = ', MSE(np.array(YList), np.array(YhatList), testingLen))
# print(MSE(np.array(YList), np.array(JList), testingLen))
# graphJ(np.array(JJ))
graphJ(np.array(JJJ))