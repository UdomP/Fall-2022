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

def IGA(featureList, trainDic, meanDic, stdDic):
    p = 1
    n = len(trainDic['time'])
    for feature in featureList:
        p *= (1/((np.power(2*np.pi, .5)) * stdDic[feature])) * np.exp(-((trainDic[feature] - meanDic[feature])**2)/(2 * np.power(stdDic[feature],2)))
    return np.sum(p)

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
    return TP,TN,FP,FN

def p(e, f):
    temp = {}
    for key in (dataDic[f]):
        try:
            meanDic = getMeanDic(featureList, dataListDic[f][key])
            stdDic = getStdDic(featureList, dataListDic[f][key])
            px = (IGA(featureList, dataListDic[f][key], meanDic, stdDic))
            if(px < e):
                temp[key] = 1
            else:
                temp[key] = 0
        except:
            continue
    return temp

def writeFile(fileName, dicToWrite):
    file = open(fileName, 'w')
    for d in dicToWrite:
        dd = d.split('.')
        file.write(dd[0] + ' ' + str(dicToWrite[d]) + '\n')
    file.close()

# featureList = ['f1_c', 'f1_a', 'f1_s', 'f1_d', 'f2_c', 'f2_a', 'f2_s', 'f2_d',
#        'prg_c', 'prg_a', 'prg_s', 'prg_d', 'prd_c', 'prd_a', 'prd_s', 'prd_d',
#        'pr_s', 'pr_d', 'lq_s', 'lq_d', 'cmp_a_s', 'cmp_a_d', 'cmp_b_s',
#        'cmp_b_d', 'cmp_c_s', 'cmp_c_d']
featureList = ['f2_c', 'f2_a', 'f2_s', 'prd_c', 'prd_a', 'prd_s', 'lq_s', 'cmp_a_s', 'cmp_b_s', 'cmp_c_s']
# featureList = ['f1_c', 'f1_a', 'f1_s', 'f1_d','f2_d', 'prg_c', 'prg_a', 'prg_s', 'prg_d', 'prd_d',
#        'pr_s', 'pr_d', 'lq_d', 'cmp_a_d', 'cmp_b_d', 'cmp_c_d']
# featureList = ['f2_c', 'f2_a', 'f2_s', 'prd_c', 'prd_a', 'prd_s', 'lq_s', 'cmp_a_s']

meanDic = getMeanDic(featureList, dataListDic['train']['training-data.csv'])
stdDic = getStdDic(featureList, dataListDic['train']['training-data.csv'])

px = (IGA(featureList, dataListDic['train']['training-data.csv'], meanDic, stdDic))
print('Selected features: ', end = '')
print(featureList)

e = px
print('e = ', e)

pTest = p(e, 'test')
pVal = p(e, 'valid')

subDic = {}
infile = open('dataset/submission.txt', 'r')
for w in infile.readlines():
    ww = w.split()
    subDic[str(ww[0]) + '.csv'] = int(ww[1])

subDicVal = {}
infile = open('dataset/valid/valid-key.txt', 'r')
for w in infile.readlines():
    ww = w.split()
    subDicVal[str(ww[0]) + '.csv'] = int(ww[1])

TP,TN,FP,FN = predict(pVal, subDicVal)
print('Data in dataset/cross-valid')
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

writeFile('task2_solution_valid.txt', pVal)