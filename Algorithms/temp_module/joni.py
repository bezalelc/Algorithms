"""
authors:
    bezalel cohen , 308571207
    Yehonatan Deri, 209173988
"""
import numpy as np
import tkinter
import sympy as sp
import matplotlib.pyplot as plt

# n+1 splines
# x,y,z -> n+1
# h,b -> n
# v,u -> n-1

def f(x):
    y = (np.sin(x) + np.cos(x))
    return np.sign(y) * np.abs(y) ** (1 / 3)


def init(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    # sort x,y
    idx_sort = x.argsort()
    x[:], y[:] = x[idx_sort], y[idx_sort]

    # init
    h = x[1:] - x[:-1]
    b = (6 / h) * (y[1:] - y[:-1])
    u, v = 2 * (h[1:] + h[:-1]), b[1:] - b[:-1]
    return h, b, u, v


def rowReduction(u, v, h):
    # rank
    for i in range(1, len(u)):
        u[i] -= ((h[i] ** 2) / u[i - 1])
        v[i] -= (h[i] / u[i - 1]) * v[i - 1]
    return u, v


def calcCD(y, h, z):
    # c,d
    c = y[1:] / h - z[1:] * h / 6
    d = y[:-1] / h - z[:-1] * h / 6
    return c, d


def calcZ(h, u, v):
    # solve
    n = len(h)
    z = np.zeros((n + 1,))
    for i in range(n - 1, 0, - 1):
        z[i] = (v[i - 1] - (h[i] * z[i + 1])) / u[i - 1]

    return z


def calcS(x, h, z, c, d):
    # S
    var = sp.symbols('x')
    S = z[:-1] / (6 * h) * (x[1:] - var) ** 3 + z[1:] / (6 * h) * (var - x[:-1]) ** 3 + c * (var - x[:-1]) + d * \
        (x[1:] - var)
    return S, var


def pointMapping(S, x, var):
    # map function
    func = [sp.lambdify(var, s, 'numpy') for s in S]
    map_points = lambda x0: np.searchsorted(x, x0) - 1 if x0 != x[0] else 0
    splines = np.vectorize(lambda x0: func[map_points(x0)](x0))
    return splines


def cubicSplain(x, y):
    h, b, u, v = init(x, y)
    u, v = rowReduction(u, v, h)
    z = calcZ(h, u, v)
    c, d = calcCD(y, h, z)
    S, var = calcS(x, h, z, c, d)
    sMap = pointMapping(S, x, var)
    return sMap


if __name__ == '__main__':
    # example from juda1
    # x = np.array([1, 9, 4], dtype=np.float64)
    # y = np.array([1, 3, 2], dtype=np.float64)
    # example from juda1
    # x = np.array([0, 1, 2, 3], dtype=np.float64)
    # y = np.array([0, 1, 0, -1], dtype=np.float64)
    # exmaple from some website
    # x = np.array([0.9, 1.3, 1.9, 2.1], dtype=np.float64)
    # y = np.array([1.3, 1.5, 1.85, 2.1], dtype=np.float64)

    sizes = np.array([2, 4, 6, 12])
    X = [np.linspace(-6, 6, num=size + 1) for size in sizes]
    Y = [f(x) for x in X]
    splines = [cubicSplain(x, y) for x, y in zip(X, Y)]

    # plot
    plt.style.use('seaborn-bright')
    z = np.linspace(-6, 6, 1000)
    fig = plt.figure(figsize=(14, 10))
    for x, y, spline, i in zip(X, Y, splines, range(221, 225)):
        fig_i = fig.add_subplot(i)
        fig_i.plot(z, f(z), c='k', alpha=1, label='f(x)')
        fig_i.plot(z, spline(z), c='b', alpha=0.7, label='spline')
        fig_i.scatter(x, spline(x), marker='o', c='r', label='spline points')
        fig_i.legend(loc='lower left')
        fig_i.set_title(f'{len(x) - 1} splines')
        fig_i.set_xlabel('x')
        fig_i.set_ylabel('f(x)')

    plt.show()
    # dark_background,seaborn-bright,grayscale,ggplot,fivethirtyeight,bmh,seaborn-poster
    # darkorange