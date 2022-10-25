from numpy import *
# A = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

# B = []
# for i in range(len(A[0])):
#     temp = []
#     for j in range(len(A)):
#         temp.append(A[j][i])
#     B.append(temp)

# print(B)

n = 100
x = linspace(0, 4*pi, n+1)
y = 2.5 + x**2 * exp(-0.5*x) * sin(x - pi/3)

print(type(x))
print(y)

def f(x):
    if type(x) == type(zeros(1)):
        return ones(len(x)) * 2
    else:
        return 2

print(f(x))
print(f(n)

def f(x):
    
