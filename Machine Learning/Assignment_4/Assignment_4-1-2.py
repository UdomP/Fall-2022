import pandas as pd
import numpy as np

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
featureList = ['Length', 'Diameter', 'Height', 'Weight']
# featureList = ['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
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
weightMatrix3 = np.ones((1, nnFeatureLen))

# weightMatrix = np.random.uniform(0,1,(nnFeatureLen, nnFeatureLen))
# print(weightMatrix)
zMatrix = np.zeros((nnFeatureLen, 1))

JList = []
alpha = 0.001

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
                # s += w3[0][i]
                # s += sigmoid(z2[i])
                # s += (1 - sigmoid(z2[i]))
                # s += (w2[rowIndex][i])
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
                s += w3[0][i] * (sigmoid(z2[i]) * (1 - sigmoid(z2[i]))) * col # w2[rowIndex][i]
            tempW2[rowIndex][colIndex] = const * s
    return tempW2

def layer3Update(w3, y, yhat, a2):
    tempW3 = w3.copy()
    for rowIndex, row in enumerate(w3):
        jz = yhat - y
        const = jz
        for colIndex, col in enumerate(row):
            tempW3[rowIndex][colIndex] = jz * a2[colIndex]
    return tempW3

def MSE(y, yh, n):
    yy = (y - yh)**2
    return np.sum(yy) * (1/n)

def J(y, yHat):
    temp = np.abs(y - yHat)
    return  0.5 * (temp**2)

print(weightMatrix)
print(weightMatrix2)
print(weightMatrix3)

def neuralNetwork():
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
        
        z3 = yHat(weightMatrix3, a2)

        yh = J(tempOutput, z3)
        # print(yh)
        # print(j)
        tempWeight1 = layer1Update(weightMatrix, weightMatrix2, weightMatrix3, z, z2, tempTraining, tempOutput, yh)
        weightMatrix = weightMatrix - (alpha * tempWeight1)
        tempWeight2 = layer2Update(weightMatrix2, weightMatrix3, z2, tempOutput, yh)
        weightMatrix2 = weightMatrix2 - (alpha * tempWeight2)
        tempWeight3 = layer3Update(weightMatrix3, tempOutput, yh, a2)
        weightMatrix3 = weightMatrix3 - (alpha * tempWeight3)
    return j

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
        
        z3 = yHat(weightMatrix3, a2)

        yh = J(tempOutput, z3)
        # print(yh)
        # print(j)
        tempWeight1 = layer1Update(weightMatrix, weightMatrix2, weightMatrix3, z, z2, tempTraining, tempOutput, yh)
        weightMatrix = weightMatrix - (alpha * tempWeight1)
        tempWeight2 = layer2Update(weightMatrix2, weightMatrix3, z2, tempOutput, yh)
        weightMatrix2 = weightMatrix2 - (alpha * tempWeight2)
        tempWeight3 = layer3Update(weightMatrix3, tempOutput, yh, a2)
        weightMatrix3 = weightMatrix3 - (alpha * tempWeight3)
        yhList.append(yh[0][0])
        yList.append(tempOutput[0][0])
        # jList.append(z3[0])
    return yhList, yList, jList

for i in range(1):
    print(i)
    neuralNetwork()
print(weightMatrix)
print(weightMatrix2)
print(weightMatrix3)

YhatList, YList, JList = neuralNetworkTest()

print(YhatList)
print(YList)
print(JList)
print(MSE(np.array(YList), np.array(YhatList), testingLen))
# print(MSE(np.array(YList), np.array(JList), testingLen))

