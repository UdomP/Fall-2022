import os
import pandas as pd
import random

vocabTraining = {}
vocabTesting = {}
vocabNumTraining = 0
vocabNumTesting = 0
trainingFileN = []
trainingFileS = []
testingFileN = []
testingFileS = []

def readFileTraining(path):
    infile=open(path,'r')
    global vocabTraining
    global vocabNumTraining
    for w in infile.read().split():
        if w not in vocabTraining:
            vocabTraining[w] = vocabNumTraining
            vocabNumTraining += 1
        # w.lower()
        # if w.lower() not in vocab:
        #     vocab[w.lower()] = vocabNum
        #     vocabNum += 1

def readFileTesting(path):
    infile=open(path,'r')
    global vocabNumTesting
    global vocabTesting
    for w in infile.read().split():
        if w not in vocabTesting:
            vocabTesting[w] = vocabNumTesting
            vocabNumTesting += 1

def readEachFile():
    normalEmailList =  os.listdir("email/normal/")
    spamEmailList =  os.listdir("email/spam/")

    # Random number
    testingIndex = [5,2,11,4,17]
    testingIndex2 = [3,2,8,10,9]
    # Remove 5 elements from each list according to the number in the list above.
    # Then use those removed list to create another testing vocab dictionary
    for x in (testingIndex):
        readFileTesting("email/normal/" + normalEmailList[x])
        testingFileN.append("email/normal/" + normalEmailList[x])
        del normalEmailList[x]
    for y in (testingIndex2):
        readFileTesting("email/spam/" + spamEmailList[y])
        testingFileS.append("email/spam/" + spamEmailList[y])
        del spamEmailList[y]
    
    # Read each files and add them to vocab dictionary
    for i in normalEmailList:
        readFileTraining("email/normal/" + i)
        trainingFileN.append("email/normal/" + i)
    for j in spamEmailList:
        readFileTraining("email/spam/" + j)
        trainingFileS.append("email/normal/" + i)

def readFile(path):
    sList = []
    infile=open(path,'r')
    for w in infile.read().split():
        sList.append(w)
    return sList

def wordOfVector(dic, pathList):
    sentenceWoV = []
    for p in pathList:
        pList = readFile(p)
        print(pList)
        sen = []
        for k in dic:
            if k in pList:
                sen.append(1)
            else:
                sen.append(0)
        sentenceWoV.append(sen)
    return sentenceWoV


readEachFile()
dfTraining = pd.DataFrame(list(vocabTraining.items()), columns = ['Key','Value'])
dfTesting = pd.DataFrame(list(vocabTesting.items()), columns = ['Key','Value'])

print(trainingFileN)
print(trainingFileS)
print(testingFileN)
print(testingFileS)

print(vocabTraining)
print(trainingFileN[:2])
print(wordOfVector(vocabTraining, trainingFileN[:2]))
# sen = []
# for a in vocabTraining:
#     if a in sentenceTraining[1]:
#         sen.append(1)
#     else:
#         sen.append(0)
# print(sen)