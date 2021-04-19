import numpy as np


def gaussian(x1, x2, sigma):
    return np.exp(-((x1 - x2) @ (x1 - x2)) / (2 * sigma ** 2))


def linear(x1, x2):
    return x1.flatten() @ x2.flatten()


if __name__ == '__main__':
    x1, x2 = np.array([1, 2, 1]), np.array([0, 4, - 1])
    sigma = 2
    print(gaussian(x1, x2, sigma))
