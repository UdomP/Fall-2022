import pandas as pd
import numpy as np
from numpy import linalg

df = pd.read_csv('secom.txt', sep = ' ', header=None)
print(df)

data = df.to_numpy()

meanArray = np.nanmean(data, axis=0)

betterData = np.nan_to_num(data, nan=meanArray)

normData = betterData - meanArray

normDataT = normData
normData = np.transpose(normData)


print('normalized data shape')
print(normData.shape)

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
for a in range(len(PCList)):
    nn += 1
    s += PCList[a]
    # print((PCList[a]/PCSum) * 100, '\t', (s/PCSum) * 100)
    # print('variance: ', (PCList[a]/PCSum) * 100, '\tComulative variance: ', (s/PCSum) * 100)
    print('variance: ', (PCList[a]/PCSum) * 100, ' %  \tComulative variance: ', (s/PCSum) * 100, ' %\tk = ', nn)
    if ((s/PCSum) * 100) >= 99:
        break