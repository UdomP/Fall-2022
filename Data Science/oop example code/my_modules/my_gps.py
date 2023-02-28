import numpy as np


class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class gps:
    def __init__(self, p1, p2, time):
        self.origin = p1
        self.destination = p2
        if time == 0:
            self.time = 0.000001
        else:
            self.time = time

    def distance_l2(self):
        return np.sqrt((self.origin.x-self.destination.x)**2 + (self.origin.y-self.destination.y)**2)

    def distance_l1(self):
        return np.abs(self.origin.x - self.destination.x) + np.abs(self.origin.y - self.destination.y)

    def speed(self, flag):
        if flag == 1:
            return self.distance_l1() / self.time
        else:
            return self.distance_l2() / self.time


if __name__ == '__main__':
    p1 = point(1, 1)
    p2 = point(0, 0)
    g = gps(p1, p2, 1)
    # print(g.distance_l2())
    # print(g.distance_l1())
    print(g.speed(2))
    print(g.speed(1))