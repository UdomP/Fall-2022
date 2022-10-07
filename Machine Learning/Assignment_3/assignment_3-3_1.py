import os
import pandas as pd
import numpy as np
import re
# import nltk
# from nltk.corpus import stopwords
# nltk.download ()

vocab = {}
vocabTest = {}
vocabNum = 0
allWordCount = 0

normalFileName = []
spamFileName = []
normalOutput = []
spamOutput = []

normalFileNameTest = []
spamFileNameTest = []
normalOutputTest = []
spamOutputTest = []

wordCount = 0
wordCountNormal = 0
wordCountSpam = 0

stopwords = ['ourselves', 'an', 'or', 'nor', 'yourself', 'own', 'their', 'any', "should've", 'to', 'there', 'more', 'mustn', 'shan', 'she', 're', "haven't", 'why', 'few', 'were', 'but', "don't", 'doing', "didn't", 'am', 'won', 'ours', 'down', 'most', 'only', 'how', 'did', 'when', 'couldn', 'as', 'ain', "mustn't", 'aren', 'myself', 'below', 'at', 'me', 'further', 'other', "you're", 'just', 'y', 'same', 'than', 'will', "couldn't", 'has', 'm', "won't", 'll', 'that', 'being', "weren't", 'with', 'until', 'above', 'we', 'such', 'not', 'can', 'during', 'haven', 'where', 'd', "you'll", "hasn't", 'weren', 'from', 'having', 'off', "it's", 'don', 'wouldn', 'theirs', "aren't", 'by', 'they', 'yours', 'your', 'now', 'hadn', 'shouldn', 'for', 'are', 'because', 'what', 'after', 'be', 'in', 'then', "shouldn't", 'is', 'the', 'you', 'which', 'through', 'hasn', "wouldn't", 'out', 'and', 'this', 'should', 'those', 'does', 'wasn', 'whom', 'before', 'o', 'him', 't', 'our', "isn't", "wasn't", 'had', 'herself', 'once', 'i', 've', 'under', 'ma', 'of', 'do', "mightn't", 'himself', 'my', 'yourselves', 'themselves', 'he', 'a', 'on', 'too', 'again', 'was', "hadn't", 'her', "that'll", "doesn't", 'these', 'isn', "you'd", 'both', 'didn', 'here', 'who', 'each', 'them', "needn't", 'while', 'very', 's', 'doesn', "shan't", 'mightn', 'against', 'no', 'about', 'all', 'needn', 'hers', 'its', 'itself', 'into', 'it', 'over', 'his', 'between', 'some', 'have', 'if', "you've", 'been', "she's", 'so', 'up']
regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')

def readFile(path):
    infile=open(path,'r')
    global vocab
    global vocabNum
    global allWordCount
    global wordCount
    for w in infile.read().split():
        if (w not in vocab) and (regex.search(w) == None) and (w not in stopwords):
        # if (w not in vocab):
            vocab[w] = vocabNum
            vocabNum += 1
            allWordCount += 1
        wordCount += 1

def readFileForWord(path):
    sList = []
    infile=open(path,'r')
    for w in infile.read().split():
        sList.append(w)
    return sList

def readEachFile():
    normalEmailList =  os.listdir("email/normal/")
    spamEmailList =  os.listdir("email/spam/")

    # Random list for testing
    normalIndex = ['email/normal/2.txt', 'email/normal/12.txt', 'email/normal/7.txt', 'email/normal/16.txt', 'email/normal/21.txt']
    spamIndex = ['email/spam/2.txt', 'email/spam/14.txt', 'email/spam/10.txt', 'email/spam/5.txt', 'email/spam/22.txt']

    global wordCount
    global wordCountNormal
    global wordCountSpam
    # Remove 5 elements from each list according to the number in the list above.
    # Then use those removed list to create another testing vocab dictionary
    for name in normalEmailList:
        if ("email/normal/" + name) not in normalIndex:
            normalFileName.append("email/normal/" + name)
            normalOutput.append(0)
            readFile("email/normal/" + name)
        else:
            normalFileNameTest.append("email/normal/" + name)
            normalOutputTest.append(0)
    wordCountNormal = wordCount
    wordCount = 0
    for name in spamEmailList:
        if ("email/spam/" + name) not in spamIndex:
            spamFileName.append("email/spam/" + name)
            spamOutput.append(1)
            readFile("email/spam/" + name)
        else:
            spamFileNameTest.append("email/spam/" + name)
            spamOutputTest.append(1)
    wordCountSpam = wordCount
    wordCount = 0

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

def bagOfWord(dic, pathList):
    sentenceBOW = {}
    for p in pathList:
        pList = readFileForWord(p)
        sen = dict.fromkeys(dic, 0)
        for word in pList:
            if word in dic:
                sen[word] += 1
        sentenceBOW[p] = sen
    return sentenceBOW

readEachFile()

def predict(x0, x1, y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        actual = y[i]
        predict = 0
        if x0[i] >= x1[i]:
            predict = 0
        else:
            predict = 1
        if(actual == 1 and predict == 1):
            TP += 1
        elif(actual == 0 and predict == 0):
            TN += 1
        elif(actual == 0 and predict == 1):
            FP += 1
        elif(actual == 1 and predict == 0):
            FN += 1
    return TP,TN,FP,FN

df = pd.DataFrame(list(vocab.items()), columns = ['Key','Value'])

classifiedSentences = (bagOfWord(vocab, (normalFileName + spamFileName)))
classifiedSentencesTR = (bagOfWord(vocab, (normalFileNameTest + spamFileNameTest)))

vocabLen = len(vocab)
totalSize = len(normalFileName + spamFileName + normalFileNameTest + spamFileNameTest)
pclass1 = len(spamFileName + spamFileNameTest)
pclass0 = len(normalFileName+ normalFileNameTest)
# print(vocab)
print(vocabLen)
print(totalSize)
print(pclass0)
print(pclass1)

def p(total, vocabS, voc, wc):
    t = 0
    for w in voc:
        t += np.log((voc[w] + 1)/(wc + vocabS))
        # t += np.log10((voc[w] + 1)/(wc + vocabS))
    return t

def NaiveBayes(classList, probY, wc):
    probList = []
    for cls in classList:
        # prob = np.log10(probY/42) + (p(allWordCount, vocabLen, classList[cls], wc))
        prob = np.log(1/2) + p(1,vocabLen ,classList[cls], wc)
        # prob = np.log(wc/allWordCount) + p(1,vocabLen ,classList[cls], wc)
        probList.append(prob)
    return probList

# print(classifiedSentencesTR)

lmda = 1

print(wordCountNormal)
print(wordCountSpam)
y1 = NaiveBayes(classifiedSentencesTR, pclass1, wordCountSpam)
y0 = NaiveBayes(classifiedSentencesTR, pclass0, wordCountNormal)

print(y0)
print(y1)
print("testing", end = " ")
print(normalFileNameTest + spamFileNameTest)

TP,TN,FP,FN = predict(y0, y1, (normalOutputTest + spamOutputTest))
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

print(allWordCount)
print(wordCountNormal)
print(wordCountSpam)
print(vocab)