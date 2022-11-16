import numpy as np
import math
from numpy import linalg

infile = open('secom.txt', 'r')

d = []
for w in infile.read().split('\n'):
    ww = w.split(' ')
    if len(d) == 0:
        for i, www in enumerate(ww):
            try:
                d.append([float(www)])
            except:
                continue
    for i, www in enumerate(ww):
        try:
            d[i].append(float(www))
        except:
            continue


data = np.array(d)
normData = data.copy()
N = len(data[0])
D = len(data)
M = 10
for i in range(len(normData)):
    s = 0
    for n in normData[i]:
        if math.isnan(n):
            continue
        else:
            s += n
    mean = s / len(normData[i])
    for j in range(len(normData[i])):
        if math.isnan(normData[i][j]):
            data[i][j] = mean
            normData[i][j] = mean
    normData[i] -= mean

normDataT = np.transpose(normData)

principle = np.ones((N, M))
print('YYYYYYYYYYYYY')
Y = normData @ principle
YT = np.transpose(Y)
print(Y)
print(Y.shape)
print(YT)
print(YT.shape)
covNormData = np.cov(YT)
print(covNormData)
print(covNormData.shape)
det = linalg.det(covNormData)
print('Det')
print(det)
print(det.shape)
eigValue, eigVector = linalg.eig(covNormData)
print('eigen')
print(eigValue)
print(eigValue.shape)
print(eigVector)
print(eigVector.shape)
print(normDataT.shape)
PC = eigVector @ normData
PCT = normDataT @ eigVector
print(PC)
print(normData.shape)
print(PC.shape)

PCCov = np.cov(PC)
print(PCCov)
print(PCCov.shape)

PCList = []
for i in range(len(PCCov)):
    PCList.append(PCCov[i][i])

PCArray = np.array(PCList)
print(PCArray)
print(PCArray.shape)
# PCSum = np.sum(PCArray)
# nn = 0
# s = 0
# for a in PCArray:
#     print(nn)
#     s += a
#     print((s/PCSum) * 100)
#     nn += 1