import numpy as np

X = np.random.uniform(-200.0, 1000.0, (200,200))
print('matrix X= ', X)
print('minimum value: ', np.amin(X))
print('maximum value: ', np.amax(X))
print('average: ', np.average(X))
print('standard deviation: ', np.std(X))