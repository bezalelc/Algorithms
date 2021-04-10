"""
Author: Bezalel Cohen
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
from numpy.lib import math
from sympy import Symbol, diff, init_printing, init_session, Matrix, lambdify


# ****************************************  derivative  ***************************************
def derivative_simple(f, x0, x1):
    return (f(x1) - f(x0)) / (x1 - x0)


def grad(f, x):
    pass


# ****************************************  functions  ****************************************
def f1(x):
    return np.sin(x)


def f2(x):
    return x ** 4 - 4 * x ** 3 + x ** 2


def f3(x):
    return np.exp(x ** 2) - 2 * np.cos(x ** 2) ** 3 + (-x) ** x


def f4(x, y):
    return x + y


'''''
# return np.exp(x ** 2) - 2 * np.cos(x ** 2) ** 3 + (-x) ** x
# return np.sin(x)
# return x ** 2 + 1
# return x ** (3 - x ** 3)
# return x ** 3  # + 7 - 8 * x ** 4 + (x / 5) ** 5
# return x ** 2 - 17
# return x ** 4 - 4 * x ** 3
# return x**2-x**x


# init_session()
# init_printing()  # ,use_latex=True use_unicode=True
# x, y, z = sp.symbols('x, y, z')
# eq1 = sp.Eq(x + y + z, 1)  # x + y + z  = 1
# eq2 = sp.Eq(x + y + 2 * z, 3)  # x + y + 2z = 3
# eq3 = sp.Eq(x + 6 * y + 2 * 9 * z, 89)  # x + y + 2z = 3
# ans = sp.solve(np.array([eq1, eq2, eq3]), (x, y, z))
# print(ans)
'''''


# ****************************************  derivative  ***************************************
def newton_raphson_sympy(f, x, guess=None, landa=1e-15, epsilon=1e-15, max_iter=10000):
    """
    newton raphson method for find roots fo multiple variable using sympy function

    :param f: array of sympy function
           x: array of sympy variable
           guess: first guess
           landa: error for |f{x(n+1)}-f{x(n)}|
           epsilon: error for |x(n+1)-x(n)|
           max_iter: max iteration if the minimization not success

    :return: x_n:
             x_history:
             f_history:
             err_history:

    :efficiency: O(iter*n^3) when iter=number of iteration and n=length of x
    """
    # -------------  init  ---------------------------
    f, x = np.array(f), np.array(x),
    x_n = np.array(guess, dtype=np.float64) if guess is not None else np.random.rand(len(x))
    F = Matrix(f)
    M = F.jacobian(x)
    F, M = lambdify(x, F, 'numpy'), lambdify(x, M, 'numpy')
    x_history, f_history, err_history = [x_n.copy()], [F(*x_n).copy()], [0]

    # --------------  calc  ------------------------
    for i in range(max_iter):
        J, f_x = M(*x_n), F(*x_n)
        x_n += np.linalg.solve(J, -f_x).reshape(x_n.shape)
        x_history.append(x_n.copy()), f_history.append(f_x.copy()), err_history.append(0)

        if np.abs(f_x).all() < landa or np.abs(x_n - x_history[-2]).all() < epsilon:
            print('np.abs(f_x).all() < landa:', np.abs(f_x).all() < landa)
            print('np.abs(x_n - x_history[-2]).all() < epsilon:', np.abs(x_n - x_history[-2]).all() < epsilon)

    return x_n, x_history, f_history, err_history


def newton_raphson(f, jacobian_, guess=None, landa=1e-15, epsilon=1e-15, max_iter=10000):
    """
    newton raphson method for find roots fo multiple variable

    :param f: array of function
           x: array of variable
           guess: first guess
           landa: error for |f{x(n+1)}-f{x(n)}|
           epsilon: error for |x(n+1)-x(n)|
           max_iter: max iteration if the minimization not success

    :return: x_n:
             x_history:
             f_history:
             err_history:

    :efficiency: O(iter*n^3) when iter=number of iteration and n=length of x
    """
    # # -------------  init  ---------------------------
    # f = np.array(f)
    # x_n = np.array(guess, dtype=np.float64) if guess is not None else np.random.rand(len(x))
    #
    # x_history, f_history, err_history = [x_n.copy()], [F(*x_n).copy()], [0]
    #
    # # --------------  calc  ------------------------
    # for i in range(max_iter):
    #     J, f_x = M(*x_n), F(*x_n)
    #     x_n += np.linalg.solve(J, -f_x).reshape(x_n.shape)
    #     x_history.append(x_n.copy()), f_history.append(f_x.copy()), err_history.append(0)
    #
    #     if np.abs(f_x).all() < landa or np.abs(x_n - x_history[-2]).all() < epsilon:
    #         print('np.abs(f_x).all() < landa:', np.abs(f_x).all() < landa)
    #         print('np.abs(x_n - x_history[-2]).all() < epsilon:', np.abs(x_n - x_history[-2]).all() < epsilon)
    #
    # return x_n, x_history, f_history, err_history
    pass


def newton_raphson_one_var(f, x0, grad, landa=1e-15, epsilon=1e-15, max_iter=100):
    """
    find root for function using newton's method


    :param func: function to find root
    :param x0:
    :param x1:
    :param landa:
    :param epsilon:
    :param max_iter:

    :return:
    """
    x1 = x0 + 1
    x_history, f_history, err_history = [x1], [f(x1)], []
    for i in range(max_iter):
        df = (f(x1) - f(x0)) / (x1 - x0)

        try:
            x0, x1 = x1, x0 - f(x0) / df  # df
            x_history.append(x1)
        except ZeroDivisionError:
            print('choose other points')
            return

        if np.abs(f(x1)) < landa or np.abs(x1 - x0) < epsilon:
            return x1, x_history


def cross(f, guess=(-1, 1), landa=1e-15, epsilon=1e-15, max_iter=10000):
    """
    cross method for find root fo one variable using sympy function

    :param f: sympy function
           x: sympy variable
           guess: first guess
           landa: error for |f{x(n+1)}-f{x(n)}|
           epsilon: error for |x(n+1)-x(n)|
           max_iter: max iteration if the minimization not success

    :return: x_history:
             f_history:
             err_history:

    :efficiency: O(iter*n) when iter=number of iteration and n=time to eval f(x)
    """
    # -------------  init  ---------------------------
    x_l, x_r = guess[0], guess[1]
    f_xl, f_xr = f(x_l), f(x_r)
    x_history, f_history, err_history = [(x_l, x_r)], [f(x_l), f(x_r)], [0]
    sign = lambda x: bool(x > 0) - bool(x < 0)

    # --------------  calc  ------------------------
    for i in range(max_iter):
        x_cross = x_l + (x_r - x_l) / 2
        f_cross = f(x_cross)
        if sign(f_xl) == sign(f_cross):
            f_xl, x_l = f_cross, x_cross
        else:  # elif sign(f_xr) == sign(f_cross):
            f_xr, x_r = f_cross, x_cross

        x_history.append((x_l, x_r)), f_history.append((f_xl, f_xr)), err_history.append(0)
        if np.abs(f_xr - f_xl) < landa and np.abs(x_r - x_l) < epsilon:
            # print('iter:', i)
            # print('np.abs(f_xr - f_xl) < landa:', np.abs(f_xr - f_xl) < landa)
            # print('np.abs(x_r - x_l) < epsilon:', np.abs(x_r - x_l) < epsilon)
            break

    return x_history, f_history, err_history


if __name__ == '__main__':
    np.vectorize(f1)
    np.vectorize(f2)
    np.vectorize(f3)

    x = sp.symarray('x', 3)
    f_1 = x[0] ** 2
    f_2 = x[1] ** 2
    f_3 = x[2] ** 2
    funcs = [f_1, f_2, f_3]
    guess = [-2, 1, 3]

    # newton_raphson_sympy(funcs, x, guess)
    cross(sp.lambdify(x[0], x[0] ** 2 - 900, 'numpy'), [-0.2, 45645], max_iter=350866)
