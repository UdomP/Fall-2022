import matplotlib.pyplot as plt
import numpy as np


class Gauss:
    def __init__(self, mean, std, min_x, max_x):
        self.mean = mean
        self.std = std
        self.min_x = min_x
        self.max_x = max_x

    def my_gauss(self, x, mean, std):
        return (1.0 / np.sqrt(2 * np.pi * std ** 2)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))

    def draw(self):
        x = np.linspace(self.min_x, self.max_x, 1000)
        y = self.my_gauss(x, self.mean, self.std)
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Plotting Gaussian Distribution')
        plt.show()


















if __name__ == '__main__':
    g = Gauss(20, 5, -100, 100)     # mean, std, min_x, max_x
    g.draw()