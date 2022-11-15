import matplotlib.pyplot as plt
import numpy as np
class Y:

    def __init__(self, v0):
        self.v0 = v0
        self.g = 9.81

    def value(self, t):
        return self.v0*t - 0.5*self.g*t**2

v0 = 5
y = Y(5)
# print(y.value(0.1))
# print(y.v0)

class Gauss:
    def __init__(self, mean, std, min_x, max_x):
        self.mean = mean
        self.std = std
        self.x = np.linspace(min_x, max_x, 1000)

    def my_gauss(self):
        return (1.0/np.sqrt(2*np.pi*self.std**2))*np.exp(-(self.x-self.mean)**2/(2*self.std**2))


    def draw(self):
        plt.plot(self.x, self.my_gauss())
        plt.show()

if __name__ == '__main__':
    g = Gauss(10, 2, -100, 100)
    g.draw()
