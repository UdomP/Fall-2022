import numpy as np

X = np.random.uniform(20, 100, (20,30))
Y = np.random.uniform(20, 100, (20,30))

l2Dist1 = np.sum(np.abs(X - Y))
l2Dist2 = np.sum(np.abs(Y - X))

print('L1 Distance(X - Y) = ', l2Dist1)
print('L2 Distance(Y - X) = ', l2Dist2)