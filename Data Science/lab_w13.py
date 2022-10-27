import numpy as np

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1,0,1])
y = np.zeros(x.shape)

print(y)

for i in range(4):
    y[i,:] = x[i, :] + v

print(y)

vv = np.tile(v, (4,1))
print(vv)

v = 2
v = np.array([2])
v = np.array([2,3])
v = np.array([2,3,1])
# v = np.array([2,3,1,2])

print(x)
print(x + v)

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

print(x)
print(y)
print(x*y)
print(y*x)
print(x@y)
print(np.dot(x,y))

print('1111111111111111111111111111')

x = np.array([[1,2]])
y = np.array([[5,6]])

print(x)
print(y)
print(x*y)
print(y*x)
# print(np.dot(x,y))

print('1111111111111111111111111111')

f0 = np.array([1,2,3,4,5])
f1 = np.array([1,2,3,4,5])
f2 = np.array([11,12,13,14,15])

print(f0-f1)
print(np.square(f0 - f1))
print(np.sum(np.square(f0 - f1)))

l2Dist1 = np.sqrt(np.sum(np.square(f0 - f1)))
l2Dist2 = np.sqrt(np.sum(np.square(f0 - f2)))
print(l2Dist1)
print(l2Dist2)

print('1111111111111111111111111111')

f0 = np.array([1,2,3,4,5])
f1 = np.array([1,2,3,4,5])
f2 = np.array([11,12,13,14,15])

print(f0-f1)
print(np.abs(f0 - f1))
print(np.sum(np.abs(f0 - f1)))

l2Dist1 = np.sum(np.abs(f0 - f1))
l2Dist1 = np.sum(np.abs(f0 - f2))

print(l2Dist1)
print(l2Dist2)

print('124333333333333333')

z = [4.0, 2.0, 3.0, 6.0, 1.0, 2.0, 3.0]
softmax = np.exp(z)/np.sum(np.exp(z))
print(softmax)
print(np.sum(softmax))

linear_prob = np.array(z)/ np.sum(np.array(z))
print(linear_prob)
print(np.sum(linear_prob))
