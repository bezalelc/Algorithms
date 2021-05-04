"""
Numerical derivative
"""
import math
import numpy as np
import sympy as sp
from Algorithms.polynomial import interpolation as inter


def dimple_diff(f, x, h=1e-2):
    """
    simple derivative
        can be numeric error if h vary small

    :param f:f(x) function to calculate derivative
    :param x: point to calculate f at
    :param h:

    :return: derivative, f'(x) in given point x

    :efficiency: O()

    :error: O(h)
    """
    return (f(x + h) - f(x)) / h


def bi_directional_diff(f, x, h=1e-2):
    """
    calculate derivative in two direction
        can be numeric error if h vary small

    :param f:f(x) function to calculate derivative
    :param x: point to calculate f at
    :param h:

    :return: derivative, f'(x) in given point x

    :efficiency: O()

    :error: O(h^2)
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def interpolation_diff(f, x0, h=1, n=2, interpolation_method=inter.newton):
    """
    interpolation derivative: interpolation on the f around the given x and derivative on the interpolation polynomial
        note: can be a numeric error if h vary small

    :param
        :f(x) function to calculate derivative
        x: point to calculate f at
        h:
        range_:
        interpolation_method:

    :return: derivative, f'(x) in given point x

    :efficiency: O(interpolation_method*n) where O(interpolation_method) = O(n^2) in newton/lagrange
                 or O(n^3) in vandermoda

    :error: O(interpolation_method)=O((f^(n+1)/(n+1)!)*pi(x-x_i))
    """
    points = np.arange(n + 1) * h / n + x0 - h / 2
    np.vectorize(f)
    points = np.concatenate((points[:, None], f(points)[:, None]), axis=1)
    poly = interpolation_method(points)
    # poly = np.around(np.array(inter_method(points), dtype=np.float64), decimals=1)
    return np.sum(poly[1:] * np.arange(1, n + 1) * x0 ** np.arange(n))


def richardson(f, x0, h=1e-2, diff=bi_directional_diff, m=3):  # , data=None
    """
    simple derivative
        can be numeric error if h vary small

    :param m:
    :param diff:
    :param f:f(x) function to calculate derivative
    :param x0: point to calculate f at
    :param h:

    :return: derivative, f'(x) in given point x

    :efficiency: O()

    :error: O(h^(2m+2))
    """
    D = np.zeros((m, m))

    # if diff is interpolation_diff:
    #     for i in range(m):
    #         D[i, 0] = interpolation_diff(f, x0, h=h / 2 ** i, n=2 ** i, interpolation_method=data['interpolation'])
    # else:
    for i in range(m):
        D[i, 0] = diff(f, x0, h / 2 ** i)

    for i in range(1, m):
        D[i:, i] = 4 ** i / (4 ** i - 1) * D[i:, i - 1] - (1 / (4 ** i - 1)) * D[i - 1:-1, i - 1]

    return D
