import pandas as pd
import os
import matplotlib.pyplot as plt

def graphScatter(fileName, feature):
    dataValid = dataListDic['valid'][fileName]
    dataTest = dataListDic['test'][fileName]
    xValid = list(range(len(dataValid[feature])))
    yValid = list(dataValid[feature])
    xTest = list(range(len(dataTest[feature])))
    yTest = list(dataTest[feature])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
    ax[0].plot(xValid, yValid)
    ax[1].plot(xTest, yTest)
    ax[0].set_xlabel('Each measures')
    ax[0].set_ylabel(feature)
    ax[0].set_title('Valid Graph')
    ax[0].legend(loc='upper left', fontsize='small')
    ax[1].set_xlabel('Each measures')
    ax[1].set_ylabel(feature)
    ax[1].set_title('Test Graph')
    ax[1].legend(loc='upper left', fontsize='small')

    plt.show()

def graphScatterList(fileName, featureList):
    dataValid = dataListDic['valid'][fileName]
    dataTest = dataListDic['test'][fileName]
    fig, ax = plt.subplots(nrows=2, ncols=len(featureList))
    for i,feature in enumerate(featureList):

        xValid = list(range(len(dataValid[feature])))
        yValid = list(dataValid[feature])
        xTest = list(range(len(dataTest[feature])))
        yTest = list(dataTest[feature])
        ax[0, i].plot(xValid, yValid)
        ax[1, i].plot(xTest, yTest)
        ax[0, i].set_xlabel('Each measures')
        ax[0, i].set_ylabel(feature)
        ax[0, i].set_title('Valid Graph')
        ax[0, i].legend(loc='upper left', fontsize='small')
        ax[1, i].set_xlabel('Each measures')
        ax[1, i].set_ylabel(feature)
        ax[1, i].set_title('Test Graph')
        ax[1, i].legend(loc='upper left', fontsize='small')

    plt.show()

def graphScatterTrain(fileName, featureList, dic):
    data = dic[fileName]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for f in featureList:
        x = list(range(len(data[f])))
        y = list(data[f])
        ax.scatter(x,y)
        ax.set_xlabel('Each measures')
        ax.set_ylabel(f)
        ax.set_title('All Feature Graph')
    plt.show()

def graphScatterTrainS(fileName, featureList, dic):
    data = dic[fileName]
    
    for f in featureList:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        x = list(range(len(data[f])))
        y = list(data[f])
        ax.scatter(x,y)
        ax.set_xlabel('Each measures')
        ax.set_ylabel(f)
        ax.set_title('All Feature Graph')
        plt.show()

############################# Read and store data #########################
dataDic = {}
dataDic['test'] = os.listdir("dataset/test/")
dataDic['train'] = os.listdir("dataset/train/")
dataDic['valid'] = os.listdir("dataset/valid/")

dataListDic = {}

for i in dataDic:
    tempDic = {}
    for j in dataDic[i]:
        if ".csv" in j:
            df = pd.read_csv("dataset/" + i + "/" + j)
            tempDic[j] = df
    dataListDic [i] = tempDic

########################### Read and store data stop #######################

featureList = ['time', 'f1_c', 'f1_a', 'f1_s', 'f1_d', 'f2_c', 'f2_a', 'f2_s', 'f2_d',
       'prg_c', 'prg_a', 'prg_s', 'prg_d', 'prd_c', 'prd_a', 'prd_s', 'prd_d',
       'pr_s', 'pr_d', 'lq_s', 'lq_d', 'cmp_a_s', 'cmp_a_d', 'cmp_b_s',
       'cmp_b_d', 'cmp_c_s', 'cmp_c_d']

graphScatterTrainS('training-data.csv', featureList, dataListDic['train'])
# graphScatter('0.csv', featureList[1])
# graphScatterList('0.csv', featureList[1:8])