import os
import pandas as pd
import numpy as np

vocabTraining = {}
vocabTesting = {}
vocabNumTraining = 0
vocabNumTesting = 0
trainingFileN = []
trainingFileS = []
testingFileN = []
testingFileS = []
trainingOutput = []
testingOutput = []

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
        trainingFileS.append("email/spam/" + i)

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
        sen = []
        for k in dic:
            if k in pList:
                sen.append(1)
            else:
                sen.append(0)
        sentenceWoV.append(sen)
    return sentenceWoV

def h(x, t, index):
    thetaX = np.dot(np.transpose(t),x[index])
    return 1/(1 + np.exp(-thetaX))

def gradientAsscent(x, y, t, alpha):
    newTheta = t
    for j in range(len(t)):
        s = 0
        for i in range(len(y)):
            s += (y[i] - h(x, t, i)) * x[i][j]
        newTheta[j] = newTheta[j] + alpha * (s/len(y))
        # newTheta[j] = newTheta[j] + alpha * (s)
    return newTheta

def train(iteration, x, y, alpha, t):
    th = t
    for i in range(iteration):
        th = gradientAsscent(x, y, th, alpha)
    return th

def predict(x, y, t):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        actual = y[i]
        predict = np.round(h(x, t, i))
        if(actual == 1 and predict == 1):
            TP += 1
        elif(actual == 0 and predict == 0):
            TN += 1
        elif(actual == 0 and predict == 1):
            FP += 1
        elif(actual == 1 and predict == 0):
            FN += 1
    return TP,TN,FP,FN

readEachFile()
dfTraining = pd.DataFrame(list(vocabTraining.items()), columns = ['Key','Value'])
dfTesting = pd.DataFrame(list(vocabTesting.items()), columns = ['Key','Value'])

classifiedSentences = (wordOfVector(vocabTraining, (trainingFileN + trainingFileS)))
newClassifiedSentences = []
for i in classifiedSentences:
    x = i
    x.insert(0, 1)
    newClassifiedSentences.append(x)

sentencesNpArray = np.asarray(newClassifiedSentences)

classifiedSentencesTR = (wordOfVector(vocabTesting, (testingFileN + testingFileS)))
newClassifiedSentencesTR = []
for i in classifiedSentencesTR:
    x = i
    x.insert(0, 1)
    newClassifiedSentencesTR.append(x)

sentencesNpArray = np.asarray(newClassifiedSentences)
sentencesNpArrayTR = np.asarray(newClassifiedSentencesTR)

print(sentencesNpArray)

# theta = np.asarray([.5,.5,.5,.5])
# alpha = .005
# iteration = 400

# print('Maximum Likelihood Estimation with normal Data')
# print("Alpha = " + str(alpha))
# print("Iteration = " + str(iteration))
# theta = train(iteration, trainingData, trainingOutput, alpha, theta)
# print('Theta = ', end = '')
# print(theta)

# TP,TN,FP,FN = predict(testingData, testingOutput, theta)
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