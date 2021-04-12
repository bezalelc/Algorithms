"""
Author: Bezalel Cohen
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
from numpy.lib import math
from sympy import Symbol, diff, init_printing, init_session, Matrix, lambdify

"""
#np.seterr('raise')
#import math
"""


# ****************************************  derivative  ***************************************
def derivative_simple(f, x0, x1):
    return (f(x1) - f(x0)) / (x1 - x0)


def grad(f, x):
    pass


# ****************************************  functions  ****************************************
def f1(x):
    return np.sin(x)


def df1(x):
    return np.cos(x)


def f2(x):
    return x ** 4 - 4 * x ** 3 + x ** 2


def df2(x):
    return 4 * x ** 3 - 12 * x ** 2 + 2 * x


def f3(x):
    return np.exp(x ** 2) - 300


def df3(x):
    return 2 * x * np.exp(x ** 2)


def f4(x, y):
    return x + y


def f5(x):
    return x ** 2


def df5(x):
    return 2 * x


def f6(x):
    return x ** 7 - 10 * x ** 5 + x ** 2 - 900


def df6(x):
    return 7 * x ** 6 - 50 * x ** 4 + 2 * x


def f7(x):
    return np.exp(-x) + 3 * x ** 2.4 - x ** 5


def df7(x):
    return -np.exp(-x) + 3 * 2.4 * x ** 1.4 - 5 * x ** 4


'''
# init_session()
# init_printing()  # ,use_latex=True use_unicode=True
# x, y, z = sp.symbols('x, y, z')
# eq1 = sp.Eq(x + y + z, 1)  # x + y + z  = 1
# eq2 = sp.Eq(x + y + 2 * z, 3)  # x + y + 2z = 3
# eq3 = sp.Eq(x + 6 * y + 2 * 9 * z, 89)  # x + y + 2z = 3
# ans = sp.solve(np.array([eq1, eq2, eq3]), (x, y, z))
# print(ans)
'''


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

    :return: x_history:

    :efficiency: O(iter*n^3) when iter=number of iteration and n=length of x
    """
    # -------------  init  ---------------------------
    f, x = np.array(f), np.array(x)
    guess = np.array(guess, dtype=np.float64) if guess is not None else np.random.rand(len(x))
    F = Matrix(f)
    jacobi = F.jacobian(x)
    F, jacobi = lambdify(x, F, 'numpy'), lambdify(x, jacobi, 'numpy')
    return newton_raphson(F, jacobi, guess, landa, epsilon, max_iter)


def newton_raphson(f, jacobi, guess, landa=1e-10, epsilon=1e-10, max_iter=10000):
    """
    newton raphson method for find roots fo multiple variable

    :param f: function that get x=[x0,x1,x2,...,xn] array and return [f1([x0,x1,x2,...,xn]),...,fn([x0,x1,x2,...,xn])]
           jacobi: derivative function for one variable or Jacobean for multiple variables
           guess: first guess
           landa: error for |f{x(n+1)}-f{x(n)}|
           epsilon: error for |x(n+1)-x(n)|
           max_iter: max iteration if the minimization not success

    :return: x_history:
             f_history:
             err_history:

    :efficiency: O(iter*n^3) when iter=number of iteration and n=length of x
    """
    # -------------  init  ---------------------------
    x_n = np.array(guess, dtype=np.float64)  # if guess is not None else np.random.rand(len(guess))
    x_history = [x_n.copy()]  # , f_history, x_err_history, p, c = [x_n.copy()], [], [-f(*x_n) / jacobi(*x_n)], [], []

    # --------------  calc  ------------------------
    for i in range(max_iter):
        # calculate
        J, f_x = jacobi(*x_n), f(*x_n)
        x_n += np.linalg.solve(np.array(J).reshape((-1, 1)), np.array(-f_x).reshape((-1, 1))).reshape(x_n.shape)

        # append data
        x_history.append(x_n.copy())  # , x_err_history.append(-f_x / (jacobi(*x_n) + epsilon))
        # , f_history.append(f_x.copy())
        # if i >= 1:
        #     p.append(np.log(x_err_history[-1] / x_err_history[-2]) / np.log(x_err_history[-2] / x_err_history[-3]))
        #     c.append(
        #         x_err_history[-1] / (np.sign(x_err_history[-2]) * np.float_power(np.abs(x_err_history[-2]), p[-1])))

        # stop condition
        if np.abs(f_x).all() < landa or np.abs(x_n - x_history[-2]).all() < epsilon:
            break

    return x_history


def newton_raphson_simple_dvision(f, x0, landa=1e-10, epsilon=1e-10, max_iter=100):
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
    x_history = [x1]
    for i in range(max_iter):
        df = (f(x1) - f(x0)) / (x1 - x0)

        try:
            x0, x1 = x1, x0 - f(x0) / df  # df
            x_history.append(x1)
        except ZeroDivisionError:
            print('choose other points')
            break
        if np.abs(f(x1)) < landa or np.abs(x1 - x0) < epsilon:
            break
    return x_history


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
    x_history, f_history, x_err_history, p, c = [(x_l, x_r)], [f(x_l), f(x_r)], [(x_r - x_l) / 2], [], []
    sign = lambda x: bool(x > 0) - bool(x < 0)

    # --------------  calc  ------------------------
    for i in range(max_iter):
        x_cross = x_l + (x_r - x_l) / 2
        f_cross = f(x_cross)
        if sign(f_xl) == sign(f_cross):
            f_xl, x_l = f_cross, x_cross
        else:  # elif sign(f_xr) == sign(f_cross):
            f_xr, x_r = f_cross, x_cross

        x_history.append((x_l, x_r)), f_history.append((f_xl, f_xr)), x_err_history.append((x_r - x_l) / 2)
        c.append(x_err_history[-1] / x_err_history[-2])
        if np.abs(f_xr - f_xl) < landa and np.abs(x_r - x_l) < epsilon:
            print('iter:', i)
            # print('np.abs(f_xr - f_xl) < landa:', np.abs(f_xr - f_xl) < landa)
            # print('np.abs(x_r - x_l) < epsilon:', np.abs(x_r - x_l) < epsilon)
            break

    return x_history, f_history, x_err_history, p, c


# -----------------------------  The order/constant of convergence  -----------------------------------
def convergence_order(x_history, f):
    """
    calculate the order of convergence:
        error = |x_i-x_n|
        p ~ ln(en/en_1)/ln(en_1/en_2)
        c = en/(en_1^p)

    :param x_history: all x that wew found for f(x)
    :param f: function that we found roots for

    :return: error: the error in each step
             p: The order of convergence
             c: The constant of convergence

    :efficiency: O(n) when n is len(x_history)
    """
    x_history = np.array(x_history)
    x = x_history[-1]
    error = np.abs(x_history - x)
    p = np.log(error[2:-1] / error[1:-2]) / np.log(error[1:-2] / error[:-3])
    c = error[2:-1] / (error[1:-2] ** p)
    return error, p, c


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

    # x_history, f_history, x_err_history, p, c = newton_raphson_sympy(funcs, x, guess=guess)
    # x_history, f_history, x_err_history = cross(sp.lambdify(x[0], x[0] ** 2 - 900, 'numpy'), [-0.2, 45645],
    #                                                   max_iter=350866)
    # newton_raphson_one_var(sp.lambdify(x[0], x[0] ** 2 - 900, 'numpy'), 2, max_iter=350866)

    # -----------------------  ido  --------------------------------
    f, df = f2, df2
    points = np.linspace(-10, 10, num=1000)
    np.vectorize(f)

    it = 50
    x_history = newton_raphson(f, df, guess=[2, ], max_iter=it)
    x_history_ = newton_raphson_simple_dvision(f, 2, max_iter=it)
    x = x_history[-1]
    x_ = x_history_[-1]
    print('--------------------------  e  -----------------------------')
    e = (x_history - x)
    e_ = np.abs(np.array(x_history_, dtype=np.float128) - x_)  # if xi - x != 0
    print(e.tolist())
    print(e_.tolist())
    # print(np.log(e[-2] / e[-3]) / np.log(e[-3] / e[-4]))
    # print(e[:-2].shape, e[1:-1].shape)
    print('--------------------------  p  -----------------------------')
    p = np.log(e[2:-1] / e[1:-2]) / np.log(e[1:-2] / e[:-3])
    p_ = np.log(e_[2:-1] / e_[1:-2]) / np.log(e_[1:-2] / e_[:-3])
    print(p.tolist())
    print(p_.tolist())
    print('--------------------------  c  -----------------------------')
    c = e[2:-1] / (e[1:-2] ** p)
    c_ = e_[2:-1] / (e_[1:-2] ** p_)
    print(c.tolist())
    print(c_.tolist())
    print('--------------------------  c*e  -----------------------------')
    print(c * e[3:])

    # print(e[1:] / e[:-1] ** 2)

    # plot the function
    plt.figure(figsize=(12, 8))
    plt.title('ido')
    plt.subplot(1, 2, 1)
    plt.plot(points, f(points))
    plt.legend('f(x) graph')
    plt.scatter(x_history[:-1], f(np.array(x_history[:-1])), marker='.', linewidths=2, color='black')
    plt.scatter(x_history[-1], f(np.array(x_history[-1])), marker='*', linewidths=3, color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    # plot the error
    plt.subplot(1, 2, 2)
    error = np.array([np.abs(x_ - x) for x_ in x_history])
    plt.plot(range(len(error)), error)
    plt.scatter(range(len(error)), error, marker='+', linewidths=2, color='black')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.show()

    # plot the order of the
    # plt.plot(len())
    print(f'x={x}, f({x})={f(x)}')
    print(f'x={x_}, f({x_})={f(x_)}')
    # plt.show()
