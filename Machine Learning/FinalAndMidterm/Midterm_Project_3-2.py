import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################# Read and store data #########################
dataDic = {}
dataDic['test'] = os.listdir("dataset/test/")
dataDic['train'] = os.listdir("dataset/train/")
dataDic['valid'] = os.listdir("dataset/valid/")

# Sort file name
tempLT = []
for d in dataDic['test']:
    res = re.split('(\d+)', d)
    for dd in res:
        try:
            tempLT.append((int(dd)))
        except:
            continue
tempLT.sort()
dataDic['test'] = [str(d) + '.csv' for d in tempLT]

tempLV = []
for d in dataDic['valid']:
    res = re.split('(\d+)', d)
    for dd in res:
        try:
            tempLV.append((int(dd)))
        except:
            continue
tempLV.sort()
dataDic['valid'] = [str(d) + '.csv' for d in tempLV]
# end of sort by name

dataListDic = {}

for i in dataDic:
    tempDic = {}
    tempList = []
    for j in dataDic[i]:
        if ".csv" in j:
            df = pd.read_csv("dataset/" + i + "/" + j)
            tempDic[j] = df
    dataListDic [i] = tempDic

########################### Read and store data stop #######################

def MVGA(data, sig, meanDic):
    p = 1
    d = len(featureList)
    for f in featureList:
        a = np.transpose(data[f] - meanDic[f]) @ np.linalg.inv(sig)
        b = a @ (data - meanDic[f])
        c = b * (-0.5)
        p *= (1/((np.power(2*np.pi, d/2)) * np.power(np.linalg.det(sig), 0.5))) * np.exp(c)
    return p
    
def normalize(dic):
    normDic = dic.copy()
    for dir in normDic:
        for fname in normDic[dir]:
            for feature in normDic[dir][fname]:
                normDic[dir][fname][feature] = normDic[dir][fname][feature]/np.linalg.norm(normDic[dir][fname][feature])
    return normDic

# return mean value in dictionary form
def getMeanDic(featureList, trainDic):
    dic = {}
    temp = trainDic.copy()
    for feature in featureList:
        dic[feature] = np.mean(temp[feature])
    return dic

# return std value in dictionary form
def getStdDic(featureList, trainDic):
    dic = {}
    temp = trainDic.copy()
    for feature in featureList:
        dic[feature] = np.std(temp[feature])**2
    return dic

def predict(predicted, sub):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in predicted:
        if(predicted[i] == 1 and sub[i] == 1):
            TP += 1
        elif(predicted[i] == 0 and sub[i] == 0):
            TN += 1
        elif(predicted[i] == 0 and sub[i] == 1):
            FP += 1
        elif(predicted[i] == 1 and sub[i] == 0):
            FN += 1
    return TP,TN,FP,FN

def pmv(e, f):
    temp = {}
    for key in (dataDic[f]):
        try:
            sMatrix, xu = getSigmaMatrix(dataListDic[f][key], meanDic)
            px = MVGA(featureList, sMatrix, xu)
            if(px < e):
                temp[key] = 1
            else:
                temp[key] = 0
        except:
            continue
    return temp

def pmvnorm(e, f):
    temp = {}
    for key in (dataDic[f]):
        try:
            sMatrix, xu = getSigmaMatrix(normDataListDic[f][key], normMeanDic)
            px = MVGA(featureList, sMatrix, xu)
            if(px < e):
                temp[key] = 1
            else:
                temp[key] = 0
        except:
            continue
    return temp

def getSigmaMatrix(dic, meanDic):
    n = len(dic['time'])
    fn = len(featureList)
    sigmaMatrixNP = np.zeros([fn,fn])
    sigmaList = []
    for feature in featureList:
        xu = (dic[feature]) - meanDic[feature]
        sigmaList.append((xu))

    sigmaNPArray = np.array(sigmaList)

    a = sigmaNPArray @ np.transpose(sigmaNPArray)

    sm = a * (1/n)
    return sm, sigmaNPArray
    
# featureList = ['f1_c', 'f1_a', 'f1_s', 'f1_d', 'f2_c', 'f2_a', 'f2_s', 'f2_d',
#        'prg_c', 'prg_a', 'prg_s', 'prg_d', 'prd_c', 'prd_a', 'prd_s', 'prd_d',
#        'pr_s', 'pr_d', 'lq_s', 'lq_d', 'cmp_a_s', 'cmp_a_d', 'cmp_b_s',
#        'cmp_b_d', 'cmp_c_s', 'cmp_c_d']
featureList = ['f2_c', 'f2_a', 'f2_s', 'prd_c', 'prd_a', 'prd_s', 'lq_s', 'cmp_a_s', 'cmp_b_s', 'cmp_c_s']
# featureList = ['f1_c', 'f1_a', 'f1_s', 'f1_d','f2_d', 'prg_c', 'prg_a', 'prg_s', 'prg_d', 'prd_d',
#        'pr_s', 'pr_d', 'lq_d', 'cmp_a_d', 'cmp_b_d', 'cmp_c_d']
# featureList = ['f2_c', 'f2_a', 'f2_s', 'prd_c', 'prd_a', 'prd_s', 'lq_s', 'cmp_a_s']
# featureList = ['cmp_b_s', 'cmp_c_s']

df = pd.read_csv('dataset/train/training-data.csv')
dataFeature = (np.array([(dataListDic['train']['training-data.csv'])[f] for f in featureList]))
meanDic = getMeanDic(featureList, dataListDic['train']['training-data.csv'])
stdDic = getStdDic(featureList, dataListDic['train']['training-data.csv'])
sMatrix, xu = getSigmaMatrix(dataListDic['train']['training-data.csv'], meanDic)
dataCov = np.cov((dataFeature))

ddd = dataListDic.copy()
normDataListDic = normalize(ddd)
normMeanDic = getMeanDic(featureList, normDataListDic['train']['training-data.csv'])
normStdDic = getStdDic(featureList, normDataListDic['train']['training-data.csv'])
NsMatrix, Nxu = getSigmaMatrix(normDataListDic['train']['training-data.csv'], normMeanDic)
dataFeatureNorm = (np.array([normDataListDic['train']['training-data.csv'][f] for f in featureList]))
dataCovNorm = np.cov((dataFeatureNorm))
print(np.linalg.det(dataCovNorm))

pxm = (MVGA(dataListDic['train']['training-data.csv'], dataCov, meanDic))
print(pxm.shape)
# hf = plt.figure()
# ha = hf.add_subplot(111, projection='3d')

# X, Y = np.meshgrid(range(len(pxm)), range(len(pxm[0])))  # `plot_surface` expects `x` and `y` data to be 2D
# ha.plot_surface(X, Y, pxm)

# plt.show()
pxmN = (MVGA(dataFeatureNorm, dataCovNorm, meanDic))
print(pxmN.shape)

e = np.sum(np.sum(pxm, axis = 0))
print('e = ', e)

pTest = pmv(e, 'test')
pVal = pmv(e, 'valid')

eN = pxmN
print('enorm = ', eN)

pTestN = pmvnorm(eN, 'test')
pValN = pmvnorm(eN, 'valid')

print("=======Predict dataset/test======")
print(pTest)
print("=======Predict dataset/valid======")
print(pVal)

subDic = {}
infile = open('dataset/submission.txt', 'r')
for w in infile.readlines():
    ww = w.split()
    subDic[str(ww[0]) + '.csv'] = int(ww[1])
print("submission.txt")
print(subDic)

subDicVal = {}
infile = open('dataset/valid/valid-key.txt', 'r')
for w in infile.readlines():
    ww = w.split()
    subDicVal[str(ww[0]) + '.csv'] = int(ww[1])
print('dataset/valid/valid-key.txt')
print(subDicVal)

print('e = ', e)
TP,TN,FP,FN = predict(pTest, subDic)
print('Data in dataset/test')
print("True Positive = " + str(TP))
print("True Negative = " + str(TN))
print("False Positive = " + str(FP))
print("False Negative = " + str(FN))
print("Total = " + str(TP+TN+FP+FN))

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print("precision = " + str(precision))
print("recall = " + str(recall))
f1Score = 2 * (precision * recall)/(precision + recall)
print("F1-Score = " + str(f1Score))

TP,TN,FP,FN = predict(pVal, subDicVal)
print('Data in dataset/valid')
print("True Positive = " + str(TP))
print("True Negative = " + str(TN))
print("False Positive = " + str(FP))
print("False Negative = " + str(FN))
print("Total = " + str(TP+TN+FP+FN))

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print("precision = " + str(precision))
print("recall = " + str(recall))
f1Score = 2 * (precision * recall)/(precision + recall)
print("F1-Score = " + str(f1Score))

print("=======Normalized Predict dataset/test======")
print(pTestN)
print("=======Normalized Predict dataset/valid======")
print(pValN)
print('e = ', eN)
TP,TN,FP,FN = predict(pTestN, subDic)
print('Data in dataset/test')
print("True Positive = " + str(TP))
print("True Negative = " + str(TN))
print("False Positive = " + str(FP))
print("False Negative = " + str(FN))
print("Total = " + str(TP+TN+FP+FN))

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print("precision = " + str(precision))
print("recall = " + str(recall))
f1Score = 2 * (precision * recall)/(precision + recall)
print("F1-Score = " + str(f1Score))

TP,TN,FP,FN = predict(pValN, subDicVal)
print('Data in dataset/valid')
print("True Positive = " + str(TP))
print("True Negative = " + str(TN))
print("False Positive = " + str(FP))
print("False Negative = " + str(FN))
print("Total = " + str(TP+TN+FP+FN))

precision = TP/(TP + FP)
recall = TP/(TP + FN)
print("precision = " + str(precision))
print("recall = " + str(recall))
f1Score = 2 * (precision * recall)/(precision + recall)
print("F1-Score = " + str(f1Score))
