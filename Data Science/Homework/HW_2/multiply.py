import numpy as np

X = np.random.uniform(0, 1, (5,5))
Y = np.random.uniform(0, 1, (5,5))

print('multiplication = ', X * Y)
print('dot product = ', np.dot(X, Y))