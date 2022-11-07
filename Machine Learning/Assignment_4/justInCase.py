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
featureData = [crabDicTraining[f] for f in featureList]

arr = np.array(featureData)
print(featureList)
print(arr.shape)