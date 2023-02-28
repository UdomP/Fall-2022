import pandas as pd
import os
import numpy as np
import re

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

def MVGA(featureList, sig, xu):
    d = len(featureList)
    xusum = xu
    xusum = np.asarray(xusum.sum(axis=1)).reshape((len(xusum),1))
    # print(xusum)
    # print(xusum.shape)
    # print(len(xusum))
    # print(len(xusum[0]))
    a = np.transpose(xusum) @ np.linalg.inv(sig)
    b = a @ xusum
    c = b * (-0.5)
    p = (1/((np.power(2*np.pi, d/2)) * np.power(np.linalg.det(sig), 0.5))) * np.exp(c)
    return np.sum(p)
    
def normalize(dic):
    normDic = dic
    for dir in dic:
        for fname in dic[dir]:
            for feature in dic[dir][fname]:
                normDic[dir][fname][feature] = dic[dir][fname][feature]/np.linalg.norm(dic[dir][fname][feature])
    return normDic

# return mean value in dictionary form
def getMeanDic(featureList, trainDic):
    dic = {}
    for feature in featureList:
        dic[feature] = np.mean(trainDic[feature])
    return dic

# return std value in dictionary form
def getStdDic(featureList, trainDic):
    dic = {}
    for feature in featureList:
        dic[feature] = np.std(trainDic[feature])**2
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
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1Score = 2 * (precision * recall)/(precision + recall)
    return TP,TN,FP,FN,f1Score

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

def tuneE(eList, f, subDic):
    bestE = 0
    bestf1 = 0
    bestTP = 0
    bestTN = 0
    bestFP = 0
    bestFN = 0
    bestPx = {}
    for e in eList:
        px = pmv(eN, f)
        TP,TN,FP,FN,f1 = predict(px, subDic)
        if f1 > bestf1:
            bestPx = px
            bestE = e
            bestf1 = f1
            bestTP = TP
            bestTN = TN
            bestFP = FP
            bestFN = FN
    return bestE,bestPx,bestf1,bestTP,bestTN,bestFP,bestFN

def tuneEN(eList, f, subDic):
    bestE = 0
    bestf1 = 0
    bestTP = 0
    bestTN = 0
    bestFP = 0
    bestFN = 0
    bestPx = {}
    for e in eList:
        px = pmvnorm(eN, f)
        TP,TN,FP,FN,f1 = predict(px, subDic)
        if f1 > bestf1:
            bestPx = px
            bestE = e
            bestf1 = f1
            bestTP = TP
            bestTN = TN
            bestFP = FP
            bestFN = FN
    return bestE,bestPx,bestf1,bestTP,bestTN,bestFP,bestFN

# featureList = ['f1_c', 'f1_a', 'f1_s', 'f1_d', 'f2_c', 'f2_a', 'f2_s', 'f2_d',
#        'prg_c', 'prg_a', 'prg_s', 'prg_d', 'prd_c', 'prd_a', 'prd_s', 'prd_d',
#        'pr_s', 'pr_d', 'lq_s', 'lq_d', 'cmp_a_s', 'cmp_a_d', 'cmp_b_s',
#        'cmp_b_d', 'cmp_c_s', 'cmp_c_d']
featureList = ['f2_c', 'f2_a', 'f2_s', 'prd_c', 'prd_a', 'prd_s', 'lq_s', 'cmp_a_s', 'cmp_b_s', 'cmp_c_s']
# featureList = ['f1_c', 'f1_a', 'f1_s', 'f1_d','f2_d', 'prg_c', 'prg_a', 'prg_s', 'prg_d', 'prd_d',
#        'pr_s', 'pr_d', 'lq_d', 'cmp_a_d', 'cmp_b_d', 'cmp_c_d']
# featureList = ['f2_c', 'f2_a', 'f2_s', 'prd_c', 'prd_a', 'prd_s', 'lq_s', 'cmp_a_s']
# featureList = ['cmp_b_s', 'cmp_c_s']

meanDic = getMeanDic(featureList, dataListDic['train']['training-data.csv'])
stdDic = getStdDic(featureList, dataListDic['train']['training-data.csv'])
sMatrix, xu = getSigmaMatrix(dataListDic['train']['training-data.csv'], meanDic)

normDataListDic = normalize(dataListDic)
normMeanDic = getMeanDic(featureList, normDataListDic['train']['training-data.csv'])
normStdDic = getStdDic(featureList, normDataListDic['train']['training-data.csv'])
NsMatrix, Nxu = getSigmaMatrix(normDataListDic['train']['training-data.csv'], normMeanDic)
# sMatrix, xu = getSigmaMatrix(normDataListDic['train']['training-data.csv'], normMeanDic)
# print(sMatrix)
# print(xu)
# (np.linalg.det(sMatrix))
# print(sMatrix.shape)
# print(xu.shape)

pxm = (MVGA(featureList, sMatrix, xu))
pxmN = (MVGA(featureList, NsMatrix, Nxu))

pxm = np.array(pxm)

e = pxm
print('e = ',e)
pTest = pmv(e, 'test')
pVal = pmv(e, 'valid')

eN = pxmN
print('normalized e = ',eN)
pTestN = pmvnorm(eN, 'test')
pValN = pmvnorm(eN, 'valid')

subDic = {}
infile = open('dataset/submission.txt', 'r')
for w in infile.readlines():
    ww = w.split()
    subDic[str(ww[0]) + '.csv'] = int(ww[1])
# print("submission.txt")
# print(subDic)

subDicVal = {}
infile = open('dataset/valid/valid-key.txt', 'r')
for w in infile.readlines():
    ww = w.split()
    subDicVal[str(ww[0]) + '.csv'] = int(ww[1])
# print('dataset/valid/valid-key.txt')
# print(subDicVal)

##########################################################
print('##########################################################')
TP,TN,FP,FN,f1 = predict(pTestN, subDic)
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

TP,TN,FP,FN,f1 = predict(pValN, subDicVal)
print('Data in dataset/validate')
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
##########################################################
# 1e-64
print('##########################################################')
eList = list(np.linspace(0,1e-10,10))
print(eList)
e,p1,f1,TP,TN,FP,FN = tuneEN(eList, 'test', subDic)
print('e = ' + str(e))
print(p1)
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


# eList = list(np.linspace(1e-1,9e-10,100))
# print(eList)
# e,p1,f1,TP,TN,FP,FN = tuneE(eList, 'test', subDic)
# print('e = ' + str(e))
# print(p1)
# print('Data in dataset/test')
# print("True Positive = " + str(TP))
# print("True Negative = " + str(TN))
# print("False Positive = " + str(FP))
# print("False Negative = " + str(FN))
# print("Total = " + str(TP+TN+FP+FN))

# precision = TP/(TP + FP)
# recall = TP/(TP + FN)
# print("precision = " + str(precision))
# print("recall = " + str(recall))
# f1Score = 2 * (precision * recall)/(precision + recall)
# print("F1-Score = " + str(f1Score))

eList = list(np.linspace(1e-1,1e-10,10))
print(eList)
e,p2,f1,TP,TN,FP,FN = tuneEN(eList, 'valid', subDicVal)
print('e = ' + str(e))
print('Data in dataset/valid')
print(p2)
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