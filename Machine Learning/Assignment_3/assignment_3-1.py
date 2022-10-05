import os
import pandas as pd
import random

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

readEachFile()
df = pd.DataFrame(list(vocab.items()), columns = ['Key','Value'])

# Random number
normalIndex = [5,2,11,4,17]
spamIndex = [3,2,8,10,9]

for i, j in zip(normalIndex, spamIndex):
    normalFileNameTest.append(normalFileName[i])
    spamFileNameTest.append(spamFileName[j])
    normalOutputTest.append(normalOutput[i])
    spamOutputTest.append(spamOutput[j])
    del normalFileName[i]
    del spamFileName[j]
    del normalOutput[i]
    del spamOutput[j]

print('Training data')
print(normalFileName + spamFileName)
print('Trainint output')
print(normalOutput + spamOutput)
print('Testing data')
print(normalFileNameTest + spamFileNameTest)
print('Testing output')
print(normalOutputTest+ spamOutputTest)

# print(normalFileName)
# print(normalFileNameTest)
# print(normalOutput)
# print(normalOutputTest)
# print(spamFileName)
# print(spamFileNameTest)
# print(spamOutput)
# print(spamOutputTest)