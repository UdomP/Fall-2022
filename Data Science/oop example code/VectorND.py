class Vector2D:     # an example for polymorphism
    def __init__(self, x, y):
        self.x = x; self.y = y

    def __str__(self):
        return 'x: %g, y: %g' % (self.x, self.y)

    def square(self):
        return self.x**2 + self.y**2

    def cube(self):
        return self.x**3 + self.y**3

    def pow(self,n):
        return self.x**n + self.y**n

    def test(self):
        return 'this is a test'


class Vector3D(Vector2D):
    def __init__(self, x, y, z):
        Vector2D.__init__(self, x, y)
        self.z = z

    def __str__(self):
        return 'x: %g, y: %g, z: %g' % (self.x, self.y, self.z)

    def square(self):
        return Vector2D.square(self) + self.z**2

    def cube(self):
        return Vector2D.cube(self) + self.z**3

    def pow(self,n):
        return Vector2D.pow(self, n) + self.z**n


def utility(s):
    print(s)
    print('square is %g: ' % s.square())
    print('cube is %g: ' % s.cube())


if __name__ == '__main__':
    v2 = Vector2D(1, 2)
    print(v2.square())
    print(v2.cube())
    print(v2.pow(3))
    print(v2.test())
    v3 = Vector3D(1, 2, 3)
    print(v3.square())
    print(v3.cube())
    print(v3.pow(3))
    print(v3.test())
    utility(v2)
    utility(v3)