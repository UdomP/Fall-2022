import math


class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return self.x * other.x + self.y * other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)  # reuse __eq__

    def __str__(self):
        return '(%g, %g)' % (self.x, self.y)

    def __len__(self):
        return 2

    def __float__(self):
        return float(0.5*(self.x+self.y))

    def dist_L2(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __xxx__(self):
        return 3


if __name__ == '__main__':
    u = Vector2D(2, 1)
    v = Vector2D(1, 0)
    print(u)
    print(v)
    print(u + v)
    print(u.add(v))
    print(u*v)
    a = u + v
    w = Vector2D(1, 1)
    print(a == w)
    print(u - v)
    print(u * v)
    print(len(u))
    print(u.dist_L2())
    print(float(u))
    print(u.__xxx__())