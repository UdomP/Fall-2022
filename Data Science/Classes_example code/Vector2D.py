import math


class Vector2D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def add(self, other):
        return Vector2D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y, self.z + other.z)

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other):
        return not self.__eq__(other)  # reuse __eq__

    def __str__(self):
        return '(%g, %g, %g)' % (self.x, self.y, self.z)

    def __len__(self):
        return 3

    def __float__(self):
        return float(0.5*(self.x+self.y+self.z))

    def dist_L2(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __xxx__(self):
        return 3


if __name__ == '__main__':
    u = Vector2D(2, 1, 3)
    v = Vector2D(1, 0, 2)
    print(u)
    print(v)
    print(u + v)
    print(u.add(v))
    print(u*v)
    a = u + v
    w = Vector2D(1, 1, 2)
    print(a == w)
    print(u - v)
    print(u * v)
    print(len(u))
    print(u.dist_L2())
    print(float(u))
    print(u.__xxx__())