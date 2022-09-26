import math
import numpy as np

i = 0
N = 100
sum = 0
while i < N + 1:
    sum += i
    i += 1
print(sum)

print(2**10)
print(10**4)

print(math.exp(1.0))

print(math.log(1024, 2))
print(math.log(10000, 10))
print(math.log(2.71828182846))


def quadFun(a, b, c):
    x1 = (-b + ((b**2) - (4 * a * c))**.5)/(2*a)
    x2 = (-b - ((b**2) - (4 * a * c))**.5)/(2*a)
    print(x1)
    print(x2)

x1 = 0
x2 = 0
quadFun(1,-4,-8)

l = []
for i in np.arange(1.0, 2.1, 0.1):
    l.append(i)
print(type(l))

A = [1.0 + i*0.1 for i in range(10+1)]
print(A)

A = [[[1,2,3], [4,5,6]] , 99, 100]

print(A[0][1][0])
print(A[1])
print(A[2])

A = [2,2,2,2,2,2,2,2,2,2]
B = [4,1,1,1,1,1,1,1,1,3]

c = []
for a,b in zip(A,B):
    c.append((a**2) - b)

print(c)

A = [[1,2,3], [4,5,6], [7,8,9]]
C = []
for row in A:
    a,b,c = row
    C.append([2*a+3, 2*b+3, 2*c+3])
print(C)

A = [[1,2,3,4,], [5,6,7,8], [9,10,11,12]]
B = [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]

for a in range(len(A)):
    for b in range(len(A[i])):
        B[j][i] = A[i][j]