from operator import length_hint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

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

def normalEquation(x, y):
    return np.dot(inv(np.dot(np.transpose(x),x)), np.dot(np.transpose(x),y))

print(sum(normalEquation(trainingFeatures, trainingOutput)))