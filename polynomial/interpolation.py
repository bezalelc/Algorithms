"""
function for polynomials: (for now only one variable supported)
    interpolation: 1. vandermonde's matrix method
                   2. lagrangh method
                   3. newton method
                   4. FFT method
"""
import numpy as np
import scipy as sc
import sympy as sp
import multiply as mult
import math
import cubic_spline
import fft


# **************************************  interpolation  *********************************************

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

    # numerator = sp.factor(numerator)
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

    return np.array(sp.Poly(P, x).all_coeffs()[::-1])


# **************************************  C  *********************************************
def C_poly(points, n, multiply):
    """
    calculate the best coefficients (=C that minimize the error) for given points

    @:param points: mx2 Matrix of points when Matrix[:,1]=x points and Matrix[:,2] = y points,
            n: rank of polynomial to return
            multiply: multiply that defined for the norm [norm1,norm2,norm_inf ...]

    :return: C: vector of coefficients for the polynomial

    :complexity: O((0.5n)^3) where n is rank of tha polynomial
     """
    points = np.array(points)
    x, y = points[:n, 0], points[:n, 1]
    L, rang = np.vander(x, increasing=True).T, np.arange(n)
    M, f = np.empty((n, n)), np.sum(y * L, axis=1)

    for i in range(n):
        for j in range(i, n):
            M[j, i] = M[i, j] = multiply(L[i], L[j])

    return np.linalg.solve(M, f)


def C_poly_itegration(x, f_arr, f_origin, rang, n):
    M, f = np.empty((n, n)), np.empty((n,))
    for i in range(n):
        f[i] = sp.integrate(f_arr[i] * f_origin, (x, *rang))
        for j in range(i, n):
            M[i, j] = M[j, i] = sp.integrate(f_arr[i] * f_arr[j], (x, *rang))
    return np.linalg.solve(M, f)


# **************************************  chebyshev  *********************************************


def chebyshev_root(n, start, end):
    """
    return roots points of Chebyshev's polynomial

    Chebyshev's polynomial: T_0(x)=1, T_1(x)=x,T_n(x)=2x*T_n_1(x)-T_n_2(x)
    Equivalent polynomial:  G_0(x)=1, G_1(x)=x,G_n(x)=cos(n*arcs(x))


    :param n: [int] the rank of the Chebyshev's polynomial
    :param start: [float] start point
    :param end: [float] end point

    :return:
        roots: roots points of Chebyshev's polynomial, array is size n

    :complexity: O(n)
    """
    i = np.arange(n + 1)
    roots = np.cos((2 * i[:-1] + 1) * np.pi / (2 * n)) * (end - start) / 2 + (end + start) / 2
    return roots


def chebyshev_extreme(n, start, end):
    """
    return extreme points of Chebyshev's polynomial

    Chebyshev's polynomial: T_0(x)=1, T_1(x)=x,T_n(x)=2x*T_n_1(x)-T_n_2(x)
    Equivalent polynomial:  G_0(x)=1, G_1(x)=x,G_n(x)=cos(n*arcs(x))


    :param n: [int] the rank of the Chebyshev's polynomial
    :param start: [float] start point
    :param end: [float] end point

    :return:
        extreme_points: extreme points of Chebyshev's polynomial, array is size n+1

    :complexity: O(n)
    """
    i = np.arange(n + 1)
    extreme = np.cos(i * np.pi / n) * (end - start) / 2 + (end + start) / 2
    return extreme


def chebyshev_T(n):
    """
    return Chebyshev's polynomial

    Chebyshev's polynomial: T_0(x)=1, T_1(x)=x,T_n(x)=2x*T_n_1(x)-T_n_2(x)
    Equivalent polynomial:  G_0(x)=1, G_1(x)=x,G_n(x)=cos(n*arcs(x))


    :param n: [int] the rank of the Chebyshev's polynomial

    :return: [sympy function] Chebyshev's polynomial in the Equivalent polynomial 'T' format

    :complexity: O(n)
    """
    x = sp.symbols('x')
    Tn_1, Tn = 1, x
    for i in range(2, n + 1):
        Tn_1, Tn = Tn, sp.factor(2 * x * Tn - Tn_1)

    return Tn if n > 1 else x if n == 1 else 1 if n == 0 else None


def chebyshev_G(n):
    """
    return Chebyshev's polynomial

    Chebyshev's polynomial: T_0(x)=1, T_1(x)=x,T_n(x)=2x*T_n_1(x)-T_n_2(x)
    Equivalent polynomial:  G_0(x)=1, G_1(x)=x,G_n(x)=cos(n*arcs(x))


    :param n: [int] the rank of the Chebyshev's polynomial

    :return: [sympy function] Chebyshev's polynomial in the Equivalent polynomial 'G' format

    :complexity: O(1)
    """
    x = sp.symbols('x')
    return 1 if n == 0 else x if n == 1 else sp.cos(n * sp.acos(x))


def chebyshev_Q(n):
    """
    return Chebyshev's polynomial in Normalized format

    extreme points: Q(x_i)=(-1)^i * 2^(1-n), x=[cos(pi*i/n) for i in range n]
    roots points:   Q(x_i)=0, x=[cos(pi*(i*2+1)/2n) for i in range n-1]


    Chebyshev's polynomial: T_0(x)=1, T_1(x)=x,T_n(x)=2x*T_n_1(x)-T_n_2(x)
    Equivalent polynomial:  G_0(x)=1, G_1(x)=x,G_n(x)=cos(n*arcs(x))


    :param n: [int] the rank of the Chebyshev's polynomial

    :return: [sympy function] Chebyshev's polynomial in the Equivalent polynomial 'G' format

    :complexity: O(1)
    """
    return chebyshev_G(n) * 2 ** (1 - n)


def err_max(n, points, df_n, c):
    """
    find the max error in interpolation

    :param n: rank of the interpolation
    :param points: interpolation points
    :param df_n: f^(n+1) derivative
    :param c: chosen point to get max value in df_n

    :return: [float] max error

    :complexity: O()
    """
    x = sp.symbols('x')
    f_err = (df_n(c) / math.factorial(n + 1)) * sp.expand(sp.prod(x - points))
    roots = np.array(sp.solve(sp.diff(f_err, x), x))
    f_err = sp.lambdify(x, f_err, 'numpy')
    return np.max(np.abs(f_err(roots)))


def chebyshev_err(n, start, end, df_n, c):
    """
     compute the error of interpolation using roots points of Chebyshev's polynomial

     Chebyshev's polynomial: T_0(x)=1, T_1(x)=x,T_n(x)=2x*T_n_1(x)-T_n_2(x)
     Equivalent polynomial:  G_0(x)=1, G_1(x)=x,G_n(x)=cos(n*arcs(x))


     :param n: [int] the rank of the Chebyshev's polynomial
     :param start: [float] start point
     :param end: [float] end point

     :return:
         [float] max error of interpolation

     :complexity: O(n)
     """
    return abs((1 / math.factorial(n + 1)) * df_n(c) * ((end - start) / 2) ** (n + 1) * 2 ** -n)


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

    points = [(-1, 0), (0, 0), (1, 2)]
    print(np.array_equal(vandermonde(points), lagrangh(points)))
    print(np.array_equal(vandermonde(points), newton(points)))
    print(vandermonde(points))

    print('--------------------------------------  fft test  ----------------------------------------------')
    P = [1, 2, 1]
    print(np.array_equal(fft.fft(P).tolist(), [(4 + 0j), 2j, 0j, -2j]))
    print(P)
    print('--------------------------------------  Chebyshev test  ----------------------------------------------')
    extreme_points, reset_points = chebyshev_extreme(2, -1, 1), chebyshev_root(2, -1, 1)
    print(extreme_points, '\n', reset_points)
    extreme_points, reset_points = chebyshev_extreme(2, 0, np.pi), chebyshev_root(2, 0, np.pi)
    print(extreme_points, '\n', reset_points)

    x = sp.symbols('x')
    start, end, n, f, c = 0, np.pi, 2, sp.sin(x), 1
    root = chebyshev_root(n + 1, start, end)
    exstrem = chebyshev_extreme(n, start, end)
    print(root)
    df_n = sp.lambdify(x, sp.diff(f, x, n + 1), 'numpy')
    err_ = err_max(n, root, df_n, c)
    print(err_)
    f = chebyshev_T(3)

    roots1, roots2 = np.array([0, np.pi / 2, np.pi]), chebyshev_root(n + 1, start, end)
    n, start, end, c, x = 2, 0, np.pi, 0, sp.symbols('x')
    f = sp.sin(x)
    f_x = sp.lambdify(x, f, 'numpy')
    points = [(xi, yi) for xi, yi in zip(roots1, f_x(roots1))]
    df_n = sp.lambdify(x, sp.diff(f, x, n + 1), 'numpy')
    p = lagrangh(points)
    # p /= p[-1]
    print('-------------------  regular roots  ----------------------------')
    print(err_max(n, roots1, df_n, c))

    print('-------------------  chebyshev roots error  ----------------------------')
    print(roots2)
    print('chebyshev roots=', err_max(n, roots2, df_n, c))
    print('chebyshev roots=', chebyshev_err(n, start, end, df_n, c))

    n, start, end, c, x = 2, 0, 2, 0, sp.symbols('x')
    roots = chebyshev_root(n + 1, start, end)
    f = x ** 3
    f_x = sp.lambdify(x, f, 'numpy')
    points = [(xi, yi) for xi, yi in zip(roots1, f_x(roots1))]
    df_n = sp.lambdify(x, sp.diff(f, x, n + 1), 'numpy')
    p = lagrangh(points)
    print('\nf=', f)
    print('roots=', roots)
    print('chebyshev roots=', err_max(n, roots, df_n, c))
    print('chebyshev roots=', chebyshev_err(n, start, end, df_n, c))

    print('-------------------    ----------------------------')
    points = [(-1, 0), (0, 0), (1, 2)]
    coeff = newton(points)
    n, start, end, c, x, f = 2, 0, np.pi, 0, sp.symbols('x'), 0
    f_ = x ** 3 + x ** 2
    for c, i in zip(coeff, range(len(coeff))):
        f += c * x ** i
    f_x = sp.lambdify(x, f, 'numpy')
    roots = np.array(points)[:, 0]
    df_n = sp.lambdify(x, sp.diff(f_, x, n + 1), 'numpy')
    print(err_max(n, roots, df_n, 0))
    print(3 / 8)

    print('-------------------  fft^-1 reverse test 1  ----------------------------')
    a = [-1, -1, 0, 1]
    DFT = fft.fft(a)
    print("DFT=", DFT)
    a_ = fft.fft_reverse(DFT)
    print(a_)
    print('-------------------  mult fft reverse test 2  ----------------------------')
    P1, P2 = [-1, 1], [1, 1]
    print(mult.mult_fft(P1, P2), mult.mult_coefficient(P1, P2))
    print(np.array_equal(mult.mult_fft(P1, P2), mult.mult_coefficient(P1, P2)))
    print(np.array_equal(np.around(mult.mult_fft(P1, P2), decimals=3)[:-1], mult.mult_point(P1, P2)))
    mult.mult_large_num(123, 456)

    print('-------------------  C_poly_itegration  ----------------------------')
    x, f_arr, f_origin, rang, n = sp.symbols('x'), [1, x], x ** 2, (0, 2), 2
    print(C_poly_itegration(x, f_arr, f_origin, rang, n))

    print('-------------------  newton  ----------------------------')
    # points = [(0, 1), (np.pi / 2, 0), (np.pi, -1)]
    points = chebyshev_root(3, 0, np.pi)
    points = [(x_i, np.cos(x_i)) for x_i in points]
    p, df, n, a, b, c = newton(points), sp.sin, 2, 0, np.pi, np.pi / 2
    print(chebyshev_err(2, a, b, df, c))

    points = [0, np.pi / 2, np.pi]
    points = [(x_i, np.cos(x_i)) for x_i in points]
    p, df, n, c = newton(points), sp.sin, 2, np.pi / 2
    # print(p)
    print(err_max(n, np.array(points)[:, 0], df, c))
