def twoSum(A, x):
    y = 0
    for nI, i in enumerate(A):
        for nJ, j in enumerate(A):
            if (nI != nJ):
                if ((i + j) == x):
                    print('yes, it is')
                    return
    print('no, it is not')

A=[1,2,3,4,5,6,7]
X = 8
print(A)
print('X = ' + str(X))
twoSum(A, X)
print(A)
print('X = ' + str(X))
X = 21
twoSum(A, X)