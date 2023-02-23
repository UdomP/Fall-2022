import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

df = pd.read_csv('NN_Data.txt', header = None)
plt.scatter(df[0], df[1])
# plt.show()
k = 10
target = np.array([[2.0] , [3.0]]).T
# knn = NearestNeighbors(n_neighbors=1)
# knn.fit(df)
# result = knn.kneighbors(target, return_distance=False)

# for rank, index in enumerate(target[0][:1], start=1):
#     print(str(rank) + " ==>", df[index])

knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(df)
distances, indices = knn_model.kneighbors(target)

print(distances)
print(indices[0][k-1])

for rank, index in enumerate(indices[0][:k-1], start=1):
    print(str(rank) + " ==>", ((df.T)[index]))


plt.scatter(df[:, 0], df[:, 1], marker='o', s=75, color='k')
plt.scatter(df[indices][0][:][:, 0], df[indices][0][:][:, 1], marker='o', s=250, color='k',
facecolors='none')
plt.scatter(target[0], target[1],marker='x', s=75, color='k')

# print(df)
