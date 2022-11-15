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
M = 1
data = np.array(d)
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
    meanList.append(mean)
    tempList.append(newCol.tolist())

newData = np.array(tempList)
for row in newData



print('data')
print(data[:,1])
print(len(data[:,1]))
print(data.shape)
print(normData[:,1])
print(len(normData[:,1]))
print(normData.shape)

# meanData = np.mean(data, axis=0)
# meanData.reshape(len(meanData), 1)
# print(meanData.shape)
# normData = np.subtract(data, meanData)
# print(normData)
# print(normData.shape)