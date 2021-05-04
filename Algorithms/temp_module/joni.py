import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# n+1 splines
# x,y,z -> n+1
# h,b -> n
# v,u -> n-1


# x = y = range(10)
# z = np.zeros(n)


def init(x, y):
    n = x.shape[0] - 1

    # init
    h = x[1:] - x[:-1]
    b = (6 / h) * (y[1:] - y[:-1])
    u, v = 2 * (h[1:] + h[:-1]), b[1:] - b[:-1]

    # rank
    for i in range(1, n - 1):
        u[i] -= ((h[i] ** 2) / u[i - 1])
        v[i] -= (h[i] / u[i - 1]) * v[i - 1]

    # solve
    z = np.zeros((n + 1,))
    for i in range(n - 1, 0, - 1):
        z[i] = (v[i - 1] - (h[i] * z[i + 1])) / u[i - 1]

    # c[i] = (y[i+1] / h[i]) - (z[i+1]*h[i]) / 6
    # d[i] = (y[i]/h[i]) - (z[i] * h[i]) / 6
    c = y[1:] / h - z[1:] * h / 6
    d = y[:-1] / h - z[:-1] * h / 6

    x_ = sp.symbols('x')
    S = z[:-1] / (6 * h) * (x[1:] - x_) ** 3 + z[1:] / (6 * h) * (x_ - x[:-1]) ** 3 + c * (x_ - x[:-1]) + d * (
                x[1:] - x_)




if __name__ == '__main__':
    # example from juda1
    x = np.array([1, 4, 9], dtype=np.float64)
    y = np.array([1, 2, 3], dtype=np.float64)
    # example from juda1
    # x = np.array([0, 1, 2, 3], dtype=np.float64)
    # y = np.array([0, 1, 0, -1], dtype=np.float64)
    # exmaple from some website
    # x = np.array([0.9, 1.3, 1.9, 2.1], dtype=np.float64)
    # y = np.array([1.3, 1.5, 1.85, 2.1], dtype=np.float64)

    init(x, y)
