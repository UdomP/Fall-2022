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
M = 19
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

print('normalized data shape')
print(normData.shape)
print(normData[0].shape)

# principle = np.ones((N, M))
# print('PC shape')
# print(principle.shape)

# Y = normData @ principle

# # print(Y)
# print('Y shape')
# print(Y.shape)

# covNormData = np.cov(Y)
covNormData = np.cov(normData)
# print(covNormData)
print('covariance shape')
print(covNormData.shape)

eigValue, eigVector = linalg.eig(covNormData)
print('eigen value shape')
# print(eigValue)
print(eigValue.shape)
# print(eigVector)
print('eigen vector shape')
print(eigVector.shape)

print('Normalized data transpose')
print(normDataT.shape)
newX = normDataT @ eigVector
print('new X')
# print(newX)
print(normData.shape)
print(newX.shape)

PCCov = np.cov(np.transpose(newX))
print('PC Cov')
# print(PCCov)
print(PCCov.shape)

PCList = []
for i in range(len(PCCov)):
    PCList.append(PCCov[i][i])

PCArray = np.array(PCList)
# print(PCArray)
print(PCArray.shape)
PCSum = np.sum(PCArray)
nn = 0
s = 0
for a in range(M):
    nn += 1
    s += PCList[a]
    # print((PCList[a]/PCSum) * 100, '\t', (s/PCSum) * 100)
    # print('variance: ', (PCList[a]/PCSum) * 100, '\tComulative variance: ', (s/PCSum) * 100)
    print('variance: ', (PCList[a]/PCSum) * 100, ' %', '\tComulative variance: ', (s/PCSum) * 100, ' %', '\tk = ', nn)

