import math

def f(x, m, s):
    return (1/((2*math.pi)**.5) * s) * (math.exp((-0.5) * ((x - m)/s)**2))


m = 0
s = 2
x = 1
print(f(x, m, s))