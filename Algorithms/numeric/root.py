"""
Author: Bezalel Cohen
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
from numpy.lib import math
from sympy import Symbol, diff, init_printing, init_session, Matrix, lambdify, vector


# ****************************************  derivative  ***************************************
def secant(f, x0, x1):
    return (f(x1) - f(x0)) / (x1 - x0)


# ****************************************  functions  ****************************************
def f1(x):
    return np.sin(x)


def f2(x):
    return x ** 4 - 4 * x ** 3 + x ** 2


def f3(x):
    return np.exp(x ** 2) - 2 * np.cos(x ** 2) ** 3 + (-x) ** x


def f4(x, y):
    return x + y


def f5(x):
    return x ** 4 - 4 * x ** 3 + x ** 2


# derivative function of f
def df5(x):
    return 4 * x ** 3 - 12 * x ** 2 + 2 * x


# ****************************************  derivative  ***************************************
def newton_raphson_sympy(f, x, guess=None, landa=1e-10, epsilon=1e-10, max_iter=100):
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


def newton_raphson(f, jacobi, guess, landa=1e-10, epsilon=1e-10, max_iter=100):
    """
    newton raphson method for find roots fo multiple variable

    :param f: function that get x=[x0,x1,x2,...,xn] array and return [f1([x0,x1,x2,...,xn]),...,fn([x0,x1,x2,...,xn])]
           jacobi: derivative function for one variable or Jacobean for multiple variables
           guess: first guess
           landa: error for |f{x(n+1)}-f{x(n)}|
           epsilon: error for |x(n+1)-x(n)|
           max_iter: max iteration if the minimization not success

    :return: x_history:

    :efficiency: O(iter*n^3) when iter=number of iteration and n=length of x
    """
    # -------------  init  ---------------------------
    x_n = np.array(guess, dtype=np.float64)  # if guess is not None else np.random.rand(len(guess))
    try:
        _ = iter(x_n)
    except TypeError:
        x_n = np.array([guess, ], dtype=np.float64)
    x_history = [x_n.copy()]  # , f_history, x_err_history, p, c = [x_n.copy()], [], [-f(*x_n) / jacobi(*x_n)], [], []

    # --------------  calc  ------------------------
    for i in range(max_iter):
        # calculate
        J, f_x = np.array(jacobi(*x_n)), f(*x_n)
        if len(J.shape) == 0:
            J = np.array([J, ]).reshape((-1, 1))
        # r = np.array()
        # w1, w2 = np.array(J).reshape(-1, 1), np.array(-f_x).reshape((-1, 1))
        x_n += np.linalg.solve(J, np.array(-f_x).reshape((-1, 1))).reshape(x_n.shape)

        # append data
        x_history.append(x_n.copy())

        # stop condition
        if np.abs(f_x).all() < landa or np.abs(x_n - x_history[-2]).all() < epsilon:
            break

    return x_history


def newton_raphson_secant(f, x0, landa=1e-10, epsilon=1e-10, max_iter=100):
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
            break

        if np.abs(f(x1)) < landa or np.abs(x1 - x0) < epsilon:
            break
    return x_history


def cross(f, guess=(-1, 1), landa=1e-10, epsilon=1e-10, max_iter=100):
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
    x_history, f_xl, f_xr = [(x_l, x_r)], f(x_l), f(x_r)
    sign = lambda x: bool(x > 0) - bool(x < 0)

    # --------------  calc  ------------------------
    for i in range(max_iter):
        x_cross = x_l + (x_r - x_l) / 2
        f_cross = f(x_cross)
        if sign(f_xl) == sign(f_cross):
            f_xl, x_l = f_cross, x_cross
        else:  # elif sign(f_xr) == sign(f_cross):
            f_xr, x_r = f_cross, x_cross

        x_history.append((x_l, x_r))
        if np.abs(f_xr - f_xl) < landa and np.abs(x_r - x_l) < epsilon:
            break

    return x_history


def jacoby(A, b, w=1, max_iter=50, eps=1e-7, landa=1e-7, norm_ord=np.inf):
    A, b, L, D, U, DL_, C, x = __gauss_seidel_jacoby_init(A, b, gauss_seidel=False)

    x_next = None
    for i in range(max_iter):
        x_next = -DL_ @ U @ x + C
        # print(x, x_next)
        if np.linalg.norm(x_next - x, ord=norm_ord) < eps or np.linalg.norm(A @ x - A @ x_next, ord=norm_ord) < landa:
            print('stop in iter: ', i)
            break
        x = x_next

    return x_next


def gauss_seidel(A, b, w=1, max_iter=50, eps=1e-7, landa=1e-7, norm_ord=np.inf):
    """
    compute x for A @ x = b

    :param A: matrix
    :param b: vector
    :param w:
    :param max_iter:
    :param eps:
    :param landa:
    :param norm_ord: norm for vector option:
        norm_ord=1, norm 1: ||x||=sum(abs(x))
        norm_ord=2, norm 2: ||x||=sum(x^2)^0.5
        norm_ord=np.inf, norm inf: ||x||=max(abs(x))

    :return: x: vector of result that A @ x = b

    :complexity: O(max_iter * n^3) where n size of b is n
    """
    A, b, L, D, U, D_, C, x = __gauss_seidel_jacoby_init(A, b, gauss_seidel=True)

    x_next = None
    for i in range(max_iter):
        x_next = -D_ @ (L + U) @ x + C
        # print(x, x_next)
        if np.linalg.norm(x_next - x, ord=norm_ord) < eps or np.linalg.norm(A @ x - A @ x_next, ord=norm_ord) < landa:
            print('stop in iter: ', i)
            break
        x = x_next

    return x_next


def __gauss_seidel_jacoby_init(A, b, gauss_seidel=True):
    """
    init parameters for jacoby or gauss_seidel

    :param A: matrix
    :param b: vector
    :param gauss_seidel: if gauss_seidel: D_=D^-1
                         if jacoby: D_=(D+L)^-1

    :return: x that A @ x = b

    :complexity: O(n^3) where n size of b is n
    """
    A, b = np.array(A, dtype=np.float64), np.array(b, dtype=np.float64)
    L, D, U = np.tril(A, k=-1), np.diag(np.diag(A)), np.triu(A, k=1)
    D_ = np.linalg.pinv(D) if gauss_seidel else np.linalg.pinv(D + L)
    C, x = D_ @ b, np.random.rand(A.shape[0])
    return A, b, L, D, U, D_, C, x


# ***************************************  The order/constant of convergence  ***************************************
def convergence_order(x_history):
    """
    calculate the order of convergence:
        error = |x_i-x_n|
        p ~ ln(en/en_1)/ln(en_1/en_2)
        c = en/(en_1^p)

    :param x_history: all x that we found for f(x)
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


# ***************************************  analyze result  ***************************************
def analyze_result(x_history, f):  # analyze result
    error, p, c = convergence_order(x_history)
    x, x_history = x_history[-1], np.array(x_history)

    # plot graph of f(x)
    points = np.linspace(x - 5, x + 5, num=1000)
    np.vectorize(f)
    plt.plot(points, f(points))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = x^4 - 4x^3 + x^2')
    # plt.show()

    # plot newton error
    plt.plot(range(len(error)), error)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.scatter(len(error) - 1, 0, linewidths=3)
    print('root =', x, '. Iterations =', len(error) - 1, '\nn\t\terror\t\t\t\t\t\t\tXn\t\t\t\t\t\t\tp')
    for i in range(len(error)):
        print(i, '\t', error[i], '\t\t\t', x_history[i], '\t\t\t',
              f"{p[i - 2] if i >= 2 and i < len(error) - 1 else ''}")
    print()

    # plot secant error
    plt.plot(range(len(error)), error)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.scatter(len(error) - 1, 0, linewidths=3)
    plt.grid(ls='--')
    plt.show()

    # plot p
    plt.plot(range(2, len(p) + 2), p)
    plt.plot(range(2, len(p) + 2), p)
    plt.xlabel('Iterations')
    plt.ylabel('p')
    plt.title('The order of convergence')
    plt.legend(['p'])
    plt.grid(ls='--')
    plt.show()

    print('x=', x_history.tolist())
    print('e=', error.tolist())
    print('p=', p.tolist())
    print('c=', c.tolist())


# ***************************************  main  ***************************************
if __name__ == '__main__':
    # np.vectorize(f1)
    # np.vectorize(f2)
    # np.vectorize(f3)
    #
    # x = sp.symarray('x', 3)
    # f_1 = x[0] ** 2
    # f_2 = x[1] ** 2
    # f_3 = x[2] ** 2
    # funcs = [f_1, f_2, f_3]
    # guess = [-2, 1, 3]
    # x_history = newton_raphson_sympy(funcs, x, guess)
    # print(x_history)
    # funcs_ = sp.lambdify(x, Matrix(funcs), 'numpy')
    # print(funcs_(*x_history[-1]).tolist())
    #
    # # f, df, it, lamda, epsilon = f5, df5, 5, 1e-7, 1e-7
    # cross(sp.lambdify(x[0], x[0] ** 2 - 900, 'numpy'), [-0.2, 45645], max_iter=350866)
    #
    # # ******************************  analyze  *********************************
    # f, df, x, itr, lamda, epsilon = f5, df5, sp.symbols('x'), 20, 1e-7, 1e-7
    # f_ = x ** 4 - 4 * x ** 3 + x ** 2
    #
    # # plot graph of f(x)
    # points = np.linspace(-1, 3.8, num=10000)
    # np.vectorize(f)
    # plt.plot(points, f(points))
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.title('f(x) = x^4 - 4x^3 + x^2')
    # plt.show()
    #
    # x_history_sympy = newton_raphson_sympy([f_, ], [x, ], [1.0, ], landa=lamda, epsilon=epsilon, max_iter=itr)
    # error_sympy, p_sympy, c_sympy = convergence_order(x_history_sympy)
    # x_sympy = x_history_sympy[-1]
    # x_history_newton = newton_raphson(f, df, [1.0, ], landa=lamda, epsilon=epsilon, max_iter=itr)
    # error_newton, p_newton, c_newton = convergence_order(x_history_newton)
    # x_newton = x_history_newton[-1]
    # x_history_secant = newton_raphson_secant(f, 1., landa=lamda, epsilon=epsilon, max_iter=itr)
    # error_secant, p_secant, c_secant = convergence_order(x_history_secant)
    # x_secant = x_history_secant[-1]
    # x_history_cross = cross(f, [-20., 20.], landa=lamda, epsilon=epsilon, max_iter=itr)
    # # x_history_cross = np.array(x_history_cross)
    # # idx = np.argmin([f(x_history_cross[:, 0]), f(x_history_cross[:, 1])], axis=0)
    # # x_history_cross = np.array(x_history_cross)[:, idx]
    # error_cross, p_cross, c_cross = convergence_order(x_history_cross)
    # x_cross = x_history_cross[-1]
    #
    # # plot error
    # # plt.plot(range(len(error_sympy)), error_sympy)
    # plt.plot(range(len(error_newton)), error_newton)
    # plt.plot(range(len(error_secant)), error_secant)
    # plt.plot(range(len(error_cross)), error_cross)
    # plt.grid(ls='--')
    # plt.xlabel('Iteration')
    # plt.ylabel('Error')
    # plt.legend(['e Newton', 'e Secant', 'e cross l', 'e cross r'])  # 'e sympy',
    # plt.show()
    #
    # # plot p
    # # plt.plot(range(2, len(p_sympy) + 2), p_sympy)
    # plt.plot(range(2, len(p_newton) + 2), p_newton)
    # plt.plot(range(2, len(p_secant) + 2), p_secant)
    # plt.plot(range(2, len(p_cross) + 2), p_cross)
    # plt.xlabel('Iterations')
    # plt.ylabel('p')
    # plt.title('The order of convergence')
    # plt.legend(['p Newton', 'p Secant', 'p cross l', 'p cross r'])  # 'p sympy',
    # plt.grid(ls='--')
    # plt.show()
    print('-----------  new  -------------------')
    A, b = np.array([[2, -1], [1, 2]]), np.array([1, 3])
    # print(np.linalg.solve(A, b))
    gauss_seidel(A, b)
    jacoby(A, b)
    # print(np.sum(b ** 2) ** 0.5)
    # print(np.linalg.norm(b, ), '\n')
    # print(np.abs(np.max(b)))
    # print(np.linalg.norm(b, ord=np.inf), '\n')
    # print(np.sum(np.abs(b)))
    # print(np.linalg.norm(b, ord=1), '\n')
