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
featureList = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']

trainingLen = int(crabDataSize*.9)
testingLen = int(crabDataSize*.1)
featureDataTraining = [crabDicTraining[f] for f in featureList]
featureDataTesting = [crabDicTesting[f] for f in featureList]
OutputDataTraining = [crabDicTraining['Age']]
OutputDataTesting = [crabDicTesting['Age']]

training = np.array(featureDataTraining)
print(featureList)

nnLayerLen = 3
nnFeatureLen = len(featureList)

thetaMatrix = np.ones((nnFeatureLen, 1)) * .5
weightMatrix = np.ones((nnFeatureLen, nnFeatureLen)) * .5
# weightMatrix = np.random.uniform(0,1,(nnFeatureLen, nnFeatureLen))
zMatrix = np.zeros((nnFeatureLen, 1))

JList = []
alpha = 0.5

######################################## Data and Configurations #########################################################

def getZ(wMatrix, xMatrix):
    tempX = xMatrix.reshape(len(training),1)
    return wMatrix @ tempX

def sigmoid(z, x):
    return (1/(1 + (np.exp(-(np.transpose(z) @ x)))))

def fx(zM):
    temp = np.exp(-zM)
    return 1/(1+temp)

# def J(y, w, x):
#     tempY = y.copy()
#     tempW = w.copy()
#     tempX = x.copy()
#     for i in range(len(y)):
#         jy = y[i] - 1
#         yz = y[i] * (1 - y[i])
#         jz = jy * yz
#         tempW[i,:] *= 
#         print(tempY[i])

def J(x , a):
    temp = np.abs(x - a)
    return  0.5 * (temp**2)

for i in range(trainingLen):
    tempTraining = training[:, i].reshape(len(training),1)
    z = getZ(weightMatrix, training[:, i])
    a = fx(z)
    j = J(tempTraining, a)
    thetaMatrix -= alpha * j

print(thetaMatrix)
print(thetaMatrix.shape)