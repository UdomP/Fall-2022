import numpy as np

def findMin(func, x):
    min = func(x[0])
    minX = x[0]
    for i in x:
        if(min > func(i)):
            min = func(i)
            minX = i
    print('X = ' + str(minX))
    print('f(X) = ' + str(min))


xList = np.arange(0.0, 2.01, 0.01)
# xList = [i/100 for i in range(0,100)]
print(xList)
func = lambda x : (x**3) - (2 * np.cos((x))) + 9
findMin(func, xList)