from my_modules.my_gps import *


class taxi:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.record = []    # record is a list, each element is an instance of the class gps

    def add(self, gps):
        self.record.append(gps)

    def dump(self):
        for i in range(len(self.record)):
            print('# %d, O: [%g, %g], D: [%g, %g], time: %g ' % (i, self.record[i].origin.x, \
                                                       self.record[i].origin.y, self.record[i].destination.x, \
                                                       self.record[i].destination.y, self.record[i].time))


if __name__ == '__main__':
    p1 = point(1, 1)
    p2 = point(0, 0)
    p3 = point(2, 2)
    g1 = gps(p1, p2, 1)
    g2 = gps(p1, p3, 2)
    g3 = gps(p2, p3, 1.5)
    t = taxi('Hongkai', 'TX0001')
    t.add(g1)
    t.add(g2)
    t.add(g3)
    t.dump()