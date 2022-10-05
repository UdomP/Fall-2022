import os
import pandas as pd
import numpy as np

vocab = {}
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

def readFile(path):
    infile=open(path,'r')
    global vocab
    global vocabNum
    global allWordCount
    for w in infile.read().split():
        if w not in vocab:
            vocab[w] = vocabNum
            vocabNum += 1
        allWordCount += 1

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

classifiedSentences = (bagOfWord(vocab, (normalFileName + spamFileName)))

classifiedSentencesTR = (bagOfWord(vocab, (normalFileNameTest + spamFileNameTest)))

vocabLen = len(vocab)
totalSize = len(normalFileName + spamFileName + normalFileNameTest + spamFileNameTest)
pclass0 = len(spamFileName + spamFileNameTest)
pclass1 = len(normalFileName+ normalFileNameTest)
print(vocabLen)
print(totalSize)
print(pclass0)
print(pclass1)

def p(total, vocabS, voc):
    t = 0
    for w in voc:
        t += np.log((voc[w] + 1)/(len(voc) + vocabS))
    return t


def NaiveBayes(classList, probY):
    probList = []
    for cls in classList:
        prob = np.log(probY/totalSize) + p(allWordCount, vocabLen, classList[cls])
        probList.append(prob)
    return probList


y0 = NaiveBayes(classifiedSentencesTR, pclass0)
y1 = NaiveBayes(classifiedSentencesTR, pclass1)

print(y0)
print(y1)
# print(classifiedSentences)
# print(classifiedSentences['email/normal/1.txt'])
# print(classifiedSentencesTR)
# print(classifiedSentencesTR['email/normal/6.txt'])