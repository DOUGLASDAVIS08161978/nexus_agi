import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import random

class FractalGeometryInvestigator:
    def __init__(self):
        self.max_depth = 10
        self.min_depth = 1
        self.max_points = 1000
        self.min_points = 10

    def mandelbrot(self, c, max_iter):
        z = c
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter

    def generate_mandelbrot(self, x_min, x_max, y_min, y_max, width, height, max_iter):
        r1 = np.linspace(x_min, x_max, width)
        r2 = np.linspace(y_min, y_max, height)
        return (r1,r2,np.array([[self.mandelbrot(complex(r, i),max_iter) for r in r1] for i in r2]))

    def julia_set(self, c, z, max_iter):
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter

    def generate_julia(self, x_min, x_max, y_min, y_max, width, height, max_iter, c):
        r1 = np.linspace(x_min, x_max, width)
        r2 = np.linspace(y_min, y_max, height)
        return (r1,r2,np.array([[self.julia_set(c, complex(r, i),max_iter) for r in r1] for i in r2]))

    def plot_mandelbrot(self, x_min, x_max, y_min, y_max, width, height, max_iter):
        d = self.generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(d[0], d[1], d[2], cmap='hot', linewidth=0)
        plt.show()

    def plot_julia(self, x_min, x_max, y_min, y_max, width, height, max_iter, c):
        d = self.generate_julia(x_min, x_max, y_min, y_max, width, height, max_iter, c)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(d[0], d[1], d[2], cmap='hot', linewidth=0)
        plt.show()

    def plot_fractal(self, fractal_type, x_min, x_max, y_min, y_max, width, height, max_iter, c=None):
        if fractal_type == 'mandelbrot':
            self.plot_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter)
        elif fractal_type == 'julia':
            self.plot_julia(x_min, x_max, y_min, y_max, width, height, max_iter, c)
        else:
            print('Invalid fractal type')

    def generate_fractal(self, fractal_type, x_min, x_max, y_min, y_max, width, height, max_iter, c=None):
        if fractal_type == 'mandelbrot':
            return self.generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter)
        elif fractal_type == 'julia':
            return self.generate_julia(x_min, x_max, y_min, y_max, width, height, max_iter, c)
        else:
            print('Invalid fractal type')

investigator = FractalGeometryInvestigator()
investigator.plot_fractal('mandelbrot', -2.0, 1.0, -1.5, 1.5, 1000, 1000, 256)
investigator.plot_fractal('julia', -1.5, 1.5, -1.5, 1.5, 1000, 1000, 256, -0.8 + 0.156j)
