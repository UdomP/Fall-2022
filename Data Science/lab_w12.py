import numpy as np
from regex import B
t = np.array([[1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15],
            [16,17,18,19,20],
            [21,22,23,24,25]])

print(t[1:-1:2,2:])
print(t[1:-1:1,3:])
print(t[1:-1:2])

a = np.full((5,5),8)
print(a)

b = np.eye(3)
print(b)
print('***********************')
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b = a[:2, 1:3]
print(b)
print(a[0,1])
b[0,0] = 77
print(b)
print(a[0,1])
print(a)