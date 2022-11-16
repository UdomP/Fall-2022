import numpy as np
import math

infile = open('secom.txt', 'r')
d = []

for w in infile.read().split('\n'):
    temp = []
    for ww in w.split(' '):
        try:
            temp.append(float(ww))
        except:
            continue
    if len(temp) > 0:
        d.append(temp)

D = len(d[0])
N = len(d)
M = 10
data = np.array(d)
normData = data.copy()
tempList = []
meanList = []
for i in range(len(data[0])):
    col = data[:,i]
    s = 0
    for c in col:
        if math.isnan(c):
            continue
        else:
            s += c   
    mean = s/len(col)
    newCol = col.copy()
    for c in range(len(newCol)):
        if math.isnan(newCol[c]):
            newCol[c] = mean
            normData[:,i] = mean
    normData[:,i] -= mean
    meanList.append(mean)
    tempList.append(newCol.tolist())

print(data.shape)
print(data[0].shape)
print(data[0])
# print(normData[0])
exit(0)

pMatrix = np.ones((D, M))
print(pMatrix.shape)
PC = normData @ pMatrix
print(PC)
print(PC.shape)

covNormData = np.cov(PC)
print(covNormData)
print(covNormData.shape)