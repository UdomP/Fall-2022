import os
import pandas as pd
import random

vocabTraining = {}
vocabTesting = {}
vocabNumTraining = 0
vocabNumTesting = 0

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
        del normalEmailList[x]
    for y in (testingIndex2):
        readFileTesting("email/normal/" + spamEmailList[y])
        del spamEmailList[y]
    
    # Read each files and add them to vocab dictionary
    for i in normalEmailList:
        readFileTraining("email/normal/" + i)
    for j in spamEmailList:
        readFileTraining("email/spam/" + j)

readEachFile()
dfTraining = pd.DataFrame(list(vocabTraining.items()), columns = ['Key','Value'])
dfTesting = pd.DataFrame(list(vocabTesting.items()), columns = ['Key','Value'])
print(vocabTraining)
print(vocabTesting)
print(dfTraining)
print(dfTesting)