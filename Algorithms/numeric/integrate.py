import numpy as np
import sympy as sp
from sympy import integrate
import polynomial.interpolation as inter
import math


def newton_cotes(func=None, points=None, n=2, inter_method=inter.newton, range_=(-1, 1)):
    """
    numeric integral using interpolation

    :param
        func: function to integrate
        points: point for interpolation, note: if func specified the points will be chebyshev point
        n: rank of interpolation polynomial
        range_: the integral range (a,b)
        inter_method: method for interpolation, available method: newton,vandermoda, lagrange

    :return:
    integral on interpolation polynomial

    :complexity: O()
    """
    if not func and not points or func is not None and points is not None:
        print("enter point or function and range")
        return
    # get points from function
    if func:
        np.vectorize(func)
        roots = inter.chebyshev_root(n, *range_)
        # print(roots)
        points = np.concatenate((roots[:, None], func(roots)[:, None]), axis=1)

    poly = inter_method(points)
    # poly = np.around(np.array(inter_method(points), dtype=np.float64), decimals=1)
    x = sp.symbols('x')
    f = 0
    for i, c in enumerate(poly):
        f += c * x ** i
    return integrate(f, (x, *range_))


def newton_cotes_lagrange(func=None, points=None, n=2, range_=(-1, 1)):
    """
    numeric integral using interpolation of lagrang
            note: you can using newton_cotes() method with inter_method=inter.lagrange

    :param
        func: function to integrate
        points: point for interpolation, note: if func specified the points will be chebyshev point
        n: rank of interpolation polynomial
        range_: the integral range (a,b)

    :return:
    integral on interpolation polynomial

    :complexity: O()
    """
    if not func and not points or func is not None and points is not None:
        print("enter point or function and range")
        return
    if func:  # get point
        roots = inter.chebyshev_root(n, *range_)
        points = np.concatenate((roots[:, None], func(roots)[:, None]), axis=1)

    # general vars
    X, Y = points[:, 0], points[:, 1]
    x = sp.symbols('x')
    integral, numerator = 0, 1

    # calc numerator
    for x_i in X:
        numerator *= (x - x_i)

    # calc the integral
    for i, y in enumerate(Y):
        f = sp.expand(numerator / (x - X[i]))
        denominator = 1
        for j in range(len(X)):
            if j != i:
                denominator *= (X[i] - X[j])
        f /= denominator
        integral += y * integrate(f, (x, *range_))

    return integral


def trapeze(func, n=2, range_=(-1, 1)):  # riemann
    """
    numeric integral using lagrange interpolation in rank 1

    :param
        func: function to integrate
        n: rank of interpolation polynomial
        range_: the integral range (a,b)

    :return:
    integral of func in given range

    :complexity: O(n)

    :error: O(-(f''(c)/12)*((b-a)^3/n^2) ~ O(h^2)
    """
    h = (range_[1] - range_[0]) / n
    X = np.arange(n + 1) * h + range_[0]

    np.vectorize(func)
    integral = (h / 2) * (func(X[0]) + 2 * np.sum(func(X[1:-1])) + func(X[-1]))
    return integral


def simpson(func, n=2, range_=(-1, 1)):
    """
    numeric integral using lagrange interpolation in rank 2

    :param
        func: function to integrate
        n: rank of interpolation polynomial
        range_: the integral range (a,b)

    :return:
    integral of func in given range

    :complexity: O(n)

    :error: O(-(f''''(c)/180)*((b-a)/n)^4)
    """
    h = (range_[1] - range_[0]) / (n * 2)
    X = (np.arange(n * 2 + 1) * h + range_[0])
    np.vectorize(func)
    integral = (h / 3) * (func(X[0]) + 2 * np.sum(func(X[2:-1:2])) + 4 * np.sum(func(X[1:-1:2])) + func(X[-1]))
    return integral


def romberg(func, m=3, range_=(-1, 1), integrate_=simpson):
    """
    numeric integral using richardson

    :param
        func: function to integrate
        m: rank of approximation matrix R
        range_: the integral range (a,b)
        integrate_: integrate method, available method: trapeze, simpson

    :return:
        integral of func in given range

    :complexity: O()

    :error: O()
    """
    R = np.zeros((m, m))
    for i in range(m):
        R[i, 0] = integrate_(func, n=2 ** i, range_=range_)

    for i in range(1, m):
        R[i:, i] = 4 ** i / (4 ** i - 1) * R[i:, i - 1] - (1 / (4 ** i - 1)) * R[i - 1:-1, i - 1]

    return R


# def f(func, n=2, range_=(-1, 1)):


if __name__ == '__main__':
    x = sp.symbols('x')
    print('--------------------------------   test 1   --------------------------------')
    f, rang_ = sp.lambdify(x, sp.sin(x), "numpy"), (0, 1)
    print(newton_cotes(f, n=3, range_=rang_, inter_method=inter.lagrangh))
    print(newton_cotes_lagrange(f, range_=rang_, n=3))
    # for i in range(10):
    #     print(newton_cotes(f, n=i, range_=rang_, inter_method=inter.lagrangh))
    #     print(newton_cotes_lagrange(f, range_=rang_, n=i))
    print(-np.cos(1) + 1)
    # real_value = -np.cos(1) + 1

    print('--------------------------------   test trapeze   --------------------------------')
    f, rang_, real_value = sp.lambdify(x, 1 / (x + 1), "numpy"), (0, 1), math.log(2)
    print(f'real value={real_value}')
    err = []
    for i in range(1, 10):
        val = trapeze(f, n=i, range_=(0, 1))
        err.append(np.abs(val - real_value))
        print(f'for n={i}: val={val}, err={err[-1]}')

    print('--------------------------------   test simpson  --------------------------------')
    f, rang_, real_value = sp.lambdify(x, 1 / (x + 1), "numpy"), (0, 1), math.log(2)
    print(simpson(f, n=4, range_=(0, 1)))
    print(simpson(f, n=2, range_=(0, 1)))

    print('--------------------------------   test romberg  --------------------------------')
    f, rang_, real_value = sp.lambdify(x, 1 / (x + 1), "numpy"), (0, 1), math.log(2)
    print(romberg(f, range_=(0, 1), integrate_=trapeze, m=3)[-1, -1])
    print(f'real value={real_value}')
