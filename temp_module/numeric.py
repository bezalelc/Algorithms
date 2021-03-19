import numpy as np


# import scipy


def f1(f, start=-float('inf'), end=float('inf'), max_iter=10):
    pass


def f2(f, start=-float('inf'), end=float('inf'), max_iter=10):
    pass


def f3(f, start=-float('inf'), end=float('inf'), max_iter=10):
    X = []
    pass


def f4(f, start=-float('inf'), end=float('inf'), max_iter=10):
    J = []
    X = []
    pass


import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from sympy import Symbol, diff, init_printing, init_session, Matrix, lambdify


# init_session()
# init_printing()  # ,use_latex=True use_unicode=True
# x, y, z = sp.symbols('x, y, z')
# eq1 = sp.Eq(x + y + z, 1)  # x + y + z  = 1
# eq2 = sp.Eq(x + y + 2 * z, 3)  # x + y + 2z = 3
# eq3 = sp.Eq(x + 6 * y + 2 * 9 * z, 89)  # x + y + 2z = 3
# ans = sp.solve(np.array([eq1, eq2, eq3]), (x, y, z))
# print(ans)


def newton_raphson(funcs, vars, delta_x, m=2, verbose=False):
    # -------------  init  ---------------------------
    funcs = np.array(funcs)
    vars = np.array(vars)
    delta_x = np.array(delta_x, dtype=np.float64)
    # ----------------------  F  -------------
    F = Matrix(funcs)
    F = lambdify(vars, F, 'numpy')
    # --------------------  M   -----------------
    M = Matrix(funcs).jacobian(vars)
    M = lambdify(vars, M, 'numpy')  # numpy
    # --------------  calc  ------------------------
    for i in range(m):
        J = M(*delta_x)
        f = -F(*delta_x)
        delta_x += np.linalg.solve(J, f).reshape((delta_x.shape[0]))
        if verbose:
            print(f'i={i}', delta_x)
            res = F(*delta_x)
            print(f'delta_x={res.T}')
            print(f'error={np.mean(np.abs(res))}')
            print()

    return delta_x


def cross(funcs, vars, delta_x, range_=(-10, 10), m=2, verbose=False):
    # -------------  init  ---------------------------
    funcs = np.array(funcs)
    vars = np.array(vars)
    delta_x = np.random.random((vars.shape, 2))
    # ----------------------  F  -------------
    F = Matrix(funcs)
    F = lambdify(vars, F, 'numpy')
    # --------------------  M   -----------------
    M = Matrix(funcs).jacobian(vars)
    M = lambdify(vars, M, 'numpy')  # numpy
    # --------------  calc  ------------------------
    for i in range(m):
        if M(*delta_x[0]) < 0 and M(*delta_x[1]) > 0:
            pass


if __name__ == '__main__':
    poly = [2, 3, 4]
