import numpy as np

x = np.array([[-3, -1, -1, 1, 1, 3],[-4, -2, 0, 0, 2, 4]])
x2 = np.array([[-3, -4], [-1,-2], [-1,0], [1,0], [1,2], [3,4]])
print(x)
print(x.shape)
xcov = np.cov(x)
print(xcov)
print(xcov.shape)

print(x2)
print(x2.shape)
xcov = np.cov(x2)
print(xcov)
print(xcov.shape)