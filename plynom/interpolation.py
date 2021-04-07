"""
function for polynomials: (for now only one variable supported)
    interpolation: 1. vandermonde's matrix method
                   2. lagrangh method
                   3. newton method
                   4. FFT method
"""
import numpy as np
import sympy as sp
import multiply as mult


def vandermonde(points):
    """
    convert the representation of the polynomial from points to coefficients
    this function work with vandermonde matrix

    for now work only with one var x

    :param:
        points: (n+1)x2 Matrix of points when Matrix[:,1]=x points and Matrix[:,2] = y points,
            n is the rank of the polynomial

    :return:
        vector that represent the coefficients of the polynomial

    :Disadvantages:
        complexity:
        compute error: because vandermonde contain number in large range(=1 to x^n)

    :complexity: O((1/3)*n^3)
    """
    points = np.array(points)
    X = np.vander(points[:, 0], increasing=True)
    return np.linalg.solve(X, points[:, 1])


def lagrangh(points):
    """
    convert the representation of the polynomial from points to coefficients
    this function work with lagrangh method

    for now work only with one var x

    :param:
        points: (n+1)x2 Matrix of points when Matrix[:,1]=x points and Matrix[:,2] = y points,
            n is the rank of the polynomial

    :return:
        vector that represent the coefficients of the polynomial

    :complexity: O(n^2)
    """

    points = np.array(points)
    X, y = points[:, 0], points[:, 1]
    x = sp.symbols('x')
    F, numerator = 0, 1

    for x_i in X:
        numerator *= (x - x_i)

    numerator = sp.factor(numerator)
    for y, i in zip(y, range(len(X))):
        f = y * sp.expand(numerator / (x - X[i]))
        denominator = 1
        for j in range(len(X)):
            if j != i:
                denominator *= (X[i] - X[j])
        f /= denominator
        F += f

    return sp.Poly(F, x).all_coeffs()[::-1]


def newton(points):
    """
    convert the representation of the polynomial from points to coefficients
    this function work with newton method

    for now work only with one var x

    :param:
        points: (n+1)x2 Matrix of points when Matrix[:,1]=x points and Matrix[:,2] = y points,
            n is the rank of the polynomial

    :return:
        vector that represent the coefficients of the polynomial

    :complexity: O(n^2)
    """
    points = np.array(points)
    X, y = points[:, 0], points[:, 1]
    M = np.insert(np.zeros((points.shape[0], points.shape[0] - 1)), 0, y, axis=1)

    for i in range(M.shape[0]):
        for j in range(1, i + 1):
            M[i, j] = (M[i, j - 1] - M[i - 1, j - 1]) / (X[i] - X[i - j])

    x = sp.symbols('x')
    P, p = M[0, 0], 1
    for coef, x_i in zip(np.diag(M)[1:], X):
        p *= (x - x_i)
        P += coef * p

    return sp.Poly(P, x).all_coeffs()[::-1]


def fft(P):
    """
    FFT algorithm to


    :param P: polynomial in Coefficients representation

    :return:

    :complexity: O(n*log(n))
    """

    def fft_rec(P):
        """
        FFT recursive algorithm


        :param P: polynomial in Coefficients representation

        :return:

        :complexity: O(n) => for each recursion
        """
        if P.shape[0] == 1:
            return P

        P = P if P.shape[0] % 2 == 0 else np.append(P, 0)
        n = P.shape[0]

        y, y0, y1 = np.zeros((n,), dtype=np.complex256), fft_rec(P[::2]), fft_rec(P[1::2])
        roots = unity_roots(n)
        for k in range(n):
            y[k] = y0[k % (n // 2)] + y1[k % (n // 2)] * roots[k]

        return y

    def unity_roots(n):
        """
        compute the unity toots fo Rank(n)

        :param n: rank of complex polynomial x^n=1

        :return: unity roots

        :complexity: O(n)
        """
        k = np.arange(n)
        theta = (2 * np.pi * k) / n
        roots = np.around(np.cos(theta) + np.sin(theta) * 1j, decimals=10)
        # roots = np.cos(theta) + np.sin(theta) * 1j
        return roots

    P = np.array(P, dtype=np.complex256)
    return fft_rec(P)


if __name__ == '__main__':
    A, B = np.array([0, 0, 0, 1]), np.array([1, 2, 3, 4, 5])
    C, D = np.array([0, 1, 2, 3, 4, 5, 6]), np.array([0, 1, 2, 3, 4, 5, 6])
    E, F = np.array([1, 2, 0]), np.array([-2, -2, 1])
    G, T = np.array([3]), np.array([5])
    X, Y = np.array([3]), np.array([1, 2, 3, 4, 5])
    V, U = np.array([1, 2, 3, 4, 5]), np.array([4])
    I, J = np.array([1324, 25, 3543, 443, 534]), np.array([43, 45, 6, 5, 5])
    P, Q = np.array([1, -2, 3]), np.array([-4, 2])
    # [1, 2, 0],[-2, -2, 1],[-1, 0, 1, 2],[-2. -6. -3.  2.]

    print(np.round(mult.mult_coefficient(P, Q), 5))
    print('--------------------  mult polynomials test  ------------------------')
    print(np.array_equal(np.round(mult.mult_point(A, B), 5), np.round(mult.mult_coefficient(A, B), 5)))
    print(np.array_equal(np.round(mult.mult_point(C, D), 2), np.round(mult.mult_coefficient(C, D), 2)))
    print(np.array_equal(np.round(mult.mult_point(E, F), 5), np.round(mult.mult_coefficient(E, F), 5)))
    print(np.array_equal(np.round(mult.mult_point(G, T), 5), np.round(mult.mult_coefficient(G, T), 5)))
    print(np.array_equal(np.round(mult.mult_point(X, Y), 5), np.round(mult.mult_coefficient(X, Y), 5)))
    print(np.array_equal(np.round(mult.mult_point(V, U), 5), np.round(mult.mult_coefficient(V, U), 5)))
    print(np.array_equal(np.round(mult.mult_point(I, J), 5), np.round(mult.mult_coefficient(I, J), 5)))
    print(np.array_equal(np.round(mult.mult_point(P, Q), 5), np.round(mult.mult_coefficient(P, Q), 5)))

    print('--------------------  interpolation test  ------------------------')
    points = [(1, 12), (2, 15), (3, 16)]
    print(np.array_equal(vandermonde(points), lagrangh(points)))
    print(np.array_equal(vandermonde(points), newton(points)))
    points = [(5, 1), (-7, -23), (-6, -54)]
    print(np.array_equal(vandermonde(points), lagrangh(points)))
    print(np.array_equal(vandermonde(points), newton(points)))
    points = [(1, 2), (0, 2), (-1, 4)]
    print(np.array_equal(vandermonde(points), lagrangh(points)))
    print(np.array_equal(vandermonde(points), newton(points)))
    points = [(5, 1), (-7, -23), (-6, -54)]
    print(np.array_equal(vandermonde(points), lagrangh(points)))
    print(np.array_equal(vandermonde(points), newton(points)))

    # points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 2), (8, 2), (9, 3)]
    # print(interpolation_newton(points))
    points = [(-1, 4), (0, 1), (1, 0), (2, -5)]
    print(np.array_equal(vandermonde(points), lagrangh(points)))
    print(np.array_equal(vandermonde(points), newton(points)))
    print(vandermonde(points))

    print('--------------------------------------  fft test  ----------------------------------------------')
    P = [1, 2, 1]
    print(np.array_equal(fft(P).tolist(), [(4 + 0j), 2j, 0j, -2j]))
    print(P)
