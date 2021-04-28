"""
Numerical derivative
"""
import numpy as np
import sympy as sp


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


def richardson(f, x, h=1e-2, diff=bi_directional_diff, m=3):
    """
    simple derivative
        can be numeric error if h vary small

    :param m:
    :param diff:
    :param f:f(x) function to calculate derivative
    :param x: point to calculate f at
    :param h:

    :return: derivative, f'(x) in given point x

    :efficiency: O()

    :error: O(h^(2m+2))
    """
    D = np.zeros((m, m + 90))

    for i in range(m):
        D[i, 0] = diff(f, x, h / 2 ** i)

    for i in range(1, m + 90):
        D[i:, i] = 4 ** i / (4 ** i - 1) * D[i:, i - 1] - (1 / (4 ** i - 1)) * D[i - 1:-1, i - 1]

    return D


if __name__ == '__main__':
    x = sp.symbols('x')
    f, df, x_, h = sp.lambdify(x, 1 / x, 'numpy'), sp.lambdify(x, sp.diff(1 / x, x), 'numpy'), 2, 0.5
    print(dimple_diff(f, x_, h=h), df(x_), np.abs(dimple_diff(f, x_, h=h) - np.abs(df(x_))))
    print(bi_directional_diff(f, x_, h=h), df(x_), np.abs(bi_directional_diff(f, x_, h=h)) - np.abs(df(x_)))
    richardson(f, x_, h=1, m=3)
