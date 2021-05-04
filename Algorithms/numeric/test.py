import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy as sc
from Algorithms.polynomial import interpolation as inter, fft, multiply as mul
from Algorithms.numeric import derivative as diff, integrate as integ


def test_derivative():
    x = sp.symbols('x')
    f, df, x0, h = sp.lambdify(x, sp.sqrt(x), 'numpy'), sp.lambdify(x, sp.diff(sp.sqrt(x), x), 'numpy'), 4, 1
    # print(dimple_diff(f, x_, h=h), df(x_), np.abs(dimple_diff(f, x_, h=h) - np.abs(df(x_))))
    # print(bi_directional_diff(f, x_, h=h), df(x_), np.abs(bi_directional_diff(f, x_, h=h)) - np.abs(df(x_)))
    print(diff.richardson(f, x0, h=h, m=3))
    print('-------------------------  interpolation test  ----------------------------')
    x = sp.symbols('x')
    f, df, x0, h = sp.lambdify(x, sp.sqrt(x), 'numpy'), sp.lambdify(x, sp.diff(sp.sqrt(x), x), 'numpy'), 4, 1
    diff.interpolation_diff(f, x0, 1, n=4, interpolation_method=inter.newton)
    # df_x0 = diff.richardson(f, x0, 1, diff.interpolation_diff, m=3, data={'interpolation': inter.newton})
    # print(df_x0)
    print(df(x0))


if __name__ == '__main__':
    test_derivative()
