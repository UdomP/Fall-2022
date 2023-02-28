import numpy as np

X = np.random.uniform(-20, 100, (20,30))
softmax = np.exp(X)/np.sum(np.exp(X))

print('Normalized X = ', softmax)
print('Sum = ', np.sum(softmax))