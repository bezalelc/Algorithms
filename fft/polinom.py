import numpy as np
import sympy as sp


def eval_(A, x_0):
    """
    eval Polynomial in format

    @param:
        @A vector that represent Polynomial
        @x_0 value to insert to the Polynomial

    :complexity: O(n)
    """
    res, x = 0, 1
    for a in A:
        res += a * x
        x *= x_0
    return res


def f(A, B):
    """
    convert represent Polynomials

    @param:
        @A,B vectors that represent Polynomials

    :complexity: O(n^3)
    """
    x = np.array([-1, 0, 1, 2])
    X = np.vander(x, increasing=True)
    y = np.multiply([eval_(A, x) for x in x], [eval_(B, x) for x in x])
    return np.linalg.solve(X, y)


def f_(A, B):
    """
    convert represent Polynomials

    @param:
        @A,B vectors that represent Polynomials

    :complexity: O(n^2)
    """
    C = []
    n = max(len(A), len(B)) - 1
    for a, b, i in zip(A, B, range(n)):
        pass


if __name__ == '__main__':
    A, B = np.array([1, 2, 0]), np.array([-2, -2, 1])
    f(A, B)
    f_([2, -1, 0, 2], [1, 0, -2, 1])
