from my_modules.my_gps import *

p1 = point(1, 1)
p2 = point(0, 0)
g = gps(p1, p2, 1)
print(g.distance_l2())
print(g.distance_l1())
print(g.speed(2))
print(g.speed(1))
L=[p1, g]
print(L[0].x)
print(L[1].speed(1))