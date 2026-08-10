import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class FractalGeometryRefiner:
    def __init__(self, width=800, height=800, iterations=10, scale=2.0):
        self.width = width
        self.height = height
        self.iterations = iterations
        self.scale = scale

    def mandelbrot(self, c, max_iter):
        z = c
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter

    def draw_mandelbrot(self):
        x_min, x_max, y_min, y_max = -2.5, 1.5, -1.5, 1.5
        x_range = x_max - x_min
        y_range = y_max - y_min
        pixels = np.zeros((self.height, self.width))

        for x in range(self.width):
            for y in range(self.height):
                c_x = x_min + x_range * x / self.width
                c_y = y_min + y_range * y / self.height
                c = complex(c_x, c_y)
                m = self.mandelbrot(c, self.iterations)
                pixels[y, x] = m

        return pixels

    def refine_fractal(self, pixels):
        refined_pixels = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                if pixels[y, x] > 0:
                    refined_pixels[y, x] = pixels[y, x]
                else:
                    refined_pixels[y, x] = self.refine_pixel(pixels, x, y)
        return refined_pixels

    def refine_pixel(self, pixels, x, y):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx = x + i
                ny = y + j
                if nx >= 0 and nx < self.width and ny >= 0 and ny < self.height:
                    neighbors.append(pixels[ny, nx])
        if len(neighbors) > 0:
            return np.mean(neighbors)
        else:
            return 0

    def draw_refined_fractal(self, pixels):
        refined_pixels = self.refine_fractal(pixels)
        refined_pixels = gaussian_filter(refined_pixels, sigma=1.0)
        plt.imshow(refined_pixels, cmap='hot', interpolation='none')
        plt.show()

    def save_fractal(self, pixels, filename):
        refined_pixels = self.refine_fractal(pixels)
        refined_pixels = gaussian_filter(refined_pixels, sigma=1.0)
        img = Image.fromarray((refined_pixels * 255).astype(np.uint8))
        img.save(filename)

    def generate_fractal(self, filename):
        pixels = self.draw_mandelbrot()
        self.save_fractal(pixels, filename)

if __name__ == "__main__":
    refiner = FractalGeometryRefiner()
    refiner.generate_fractal('refined_fractal.png')
