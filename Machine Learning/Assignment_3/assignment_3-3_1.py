import os
import pandas as pd
import numpy as np
import re
# import nltk
# from nltk.corpus import stopwords
# nltk.download ()

vocab = {}
vocab1 = {}
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
        if (regex.search(w) == None) and (w not in stopwords):
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

def countWord(pathList):
    l = 0
    for p in pathList:
        pList = readFileForWord(p)
        l += len(pList)
    return l

# def p(wc, bow, lmda, sj):
#     pp = 1
#     for b in bow.values():
#         pp *= ((b + lmda)/(wc + (sj * lmda)))
#     return pp


# def NaiveBayes(BOW, lmda, k, sj, wc, tot):
#     probList = []
#     for b in BOW:
#         prob = p(wc, BOW[b], lmda, sj) * ((wc + lmda)/(tot + (k * lmda)))
#         probList.append(prob)
#     return probList

def p(wc, bow, lmda, sj):
    pp = 0
    for b in bow.values():
        pp += np.log((b + lmda)/(wc + (sj * lmda)))
    return pp


def NaiveBayes(BOW, lmda, k, sj, wc, tot):
    probList = []
    for b in BOW:
        prob = p(wc, BOW[b], lmda, sj) + np.log((wc + lmda)/(tot + (k * lmda)))
        probList.append(prob)
    return probList

readEachFile()

df = pd.DataFrame(list(vocab.items()), columns = ['Key','Value'])

classifiedSentences = (bagOfWord(vocab, (normalFileName + spamFileName)))
classifiedSentencesTR = (bagOfWord(vocab, (normalFileNameTest + spamFileNameTest)))
tw = countWord(normalFileName + normalFileNameTest + spamFileName + spamFileNameTest)
ln = countWord(normalFileName + normalFileNameTest)
ls = countWord(spamFileName + spamFileNameTest)

print((tw))
print((ln))
print((ls))
print("Vocab length/unique word = " + str(len(vocab)))
print("All word counted = " + str(allWordCount))
print("All word count in normal email = " + str(wordCountNormal))
print("All word count in spam email = " + str(wordCountSpam))

lmda = 1
k = 2
sj = len(vocab)
y1 = NaiveBayes(classifiedSentencesTR, lmda, k, sj, ls, tw)
y0 = NaiveBayes(classifiedSentencesTR, lmda, k, sj, ln, tw)

print(y0)
print(y1)