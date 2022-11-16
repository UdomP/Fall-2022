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
print(data[0])
print(normData[0])
exit(0)
principle = np.ones((N, M))

Y = normData @ principle

print(Y)
print(Y.shape)

covNormData = np.cov(Y)
print(covNormData)
print(covNormData.shape)
exit(0)

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
PCSum = np.sum(PCArray)
nn = 0
s = 0
for a in PCArray:
    s += a
    p = (s/PCSum) * 100
    if p >= 99:
        print(nn)
        print(p)
    nn += 1