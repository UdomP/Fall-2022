import os
import pandas as pd
import numpy as np

vocab = {}
vocabNum = 0

normalFileName = []
spamFileName = []
normalOutput = []
spamOutput = []

normalFileNameTest = []
spamFileNameTest = []
normalOutputTest = []
spamOutputTest = []

trainingFile = []
trainingOutputFile = []
testingFile = []
testingOutputFile = []

def readFile(path):
    infile=open(path,'r')
    global vocab
    global vocabNum
    for w in infile.read().split():
        if w not in vocab:
            vocab[w] = vocabNum
            vocabNum += 1

def readFileForWord(path):
    sList = []
    infile=open(path,'r')
    for w in infile.read().split():
        sList.append(w)
    return sList

def readEachFile():
    normalEmailList =  os.listdir("email/normal/")
    spamEmailList =  os.listdir("email/spam/")
    # Remove 5 elements from each list according to the number in the list above.
    # Then use those removed list to create another testing vocab dictionary
    for name in normalEmailList:
        normalFileName.append("email/normal/" + name)
        normalOutput.append(0)
        readFile("email/normal/" + name)
    for name in spamEmailList:
        spamFileName.append("email/spam/" + name)
        spamOutput.append(1)
        readFile("email/spam/" + name)

def wordOfVector(dic, pathList):
    sentenceWoV = []
    for p in pathList:
        pList = readFileForWord(p)
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
df = pd.DataFrame(list(vocab.items()), columns = ['Key','Value'])

# Random number index
normalIndex = [5,2,11,4,17]
spamIndex = [3,2,8,10,9]
# Select random file at random index, then put them in testing set, and finally take it out of the original list.
for i, j in zip(normalIndex, spamIndex):
    normalFileNameTest.append(normalFileName[i])
    spamFileNameTest.append(spamFileName[j])
    normalOutputTest.append(normalOutput[i])
    spamOutputTest.append(spamOutput[j])
    del normalFileName[i]
    del spamFileName[j]
    del normalOutput[i]
    del spamOutput[j]

classifiedSentences = (wordOfVector(vocab, (normalFileName + spamFileName)))
newClassifiedSentences = []
for i in classifiedSentences:
    x = i
    x.insert(0, 1)
    newClassifiedSentences.append(x)

classifiedSentencesTR = (wordOfVector(vocab, (normalFileNameTest + spamFileNameTest)))
newClassifiedSentencesTR = []
for i in classifiedSentencesTR:
    x = i
    x.insert(0, 1)
    newClassifiedSentencesTR.append(x)

sentencesNpArray = np.asarray(newClassifiedSentences)
sentencesNpArrayOutput = np.asarray(normalOutput + spamOutput)
sentencesNpArrayTR = np.asarray(newClassifiedSentencesTR)
sentencesNpArrayOutputTR = np.asarray(normalOutputTest + spamOutputTest)
print(normalOutputTest + spamOutputTest)
# print(sentencesNpArray)
# print(sentencesNpArrayOutput)
# print(sentencesNpArrayTR)
# print(sentencesNpArrayOutputTR)
# print(len(sentencesNpArray))
# print(len(sentencesNpArrayOutput))
# print(len(sentencesNpArrayTR))
# print(len(sentencesNpArrayOutputTR))

ranTheta = [.5] * (len(sentencesNpArray[0]))
theta = np.asarray(ranTheta)
alpha = .9
iteration = 50

print(classifiedSentences)


print('Maximum Likelihood Estimation with normal Data')
print("Alpha = " + str(alpha))
print("Iteration = " + str(iteration))
theta = train(iteration, classifiedSentences, sentencesNpArrayOutput, alpha, theta)
print('Theta = ', end = '')
print(theta)

TP,TN,FP,FN = predict(classifiedSentencesTR, sentencesNpArrayOutputTR, theta)
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