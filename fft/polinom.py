"""
function for polynomials: (for now only one variable supported)
    interpolation: 1. vandermonde's matrix method
                   2. lagrangh method
                   3. newton method

    Multiplication: 1. point representation
                    2. Coefficients representation

    eval: 1. coefficient representation

"""
import numpy as np
import sympy as sp


def eval_coefficient(A, x_0):
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


# ----------------------------  interpolation  -------------------------------------

def interpolation_vandermonde(points):
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


def interpolation_lagrangh(points):
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


def interpolation_newton(points):
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


# ----------------------------  multiply  -------------------------------------
def mult_coefficient(A, B):
    """
    multiply to Polynomials that represent by vector of values
        for example: Polynomial A= 1-2x+3x^2 => A=[1,-2,3]
                     Polynomial B= -4+2x     => B=[-4,2]
                     Polynomial C= A*B= -4+10x-16x^2+6^3 and the return will be: [ -4  10 -16   6]

    :param:
        @A,B vectors that represent Polynomials

    :return:
        vector of values that represent Polynomial C= A*B

    :complexity: O(n^2)
    """
    C = []
    for l in range(len(A) + len(B) - 1):
        c = 0
        # print(f'---------- iter {l} -------------------')
        for i, j in zip(range(min(l, len(A) - 1), -1, -1), range(max(0, l - len(A) + 1), min(l + 1, len(B)))):
            # print(f'i={i},j={j}')
            c += A[i] * B[j]
        C.append(c)

    return C


def mult_point(A, B, interpolation=interpolation_vandermonde):
    """
    multiply to Polynomials that represent by vector of values
        for example: Polynomial A= 1-2x+3x^2 => A=[1,-2,3]
                     Polynomial B= -4+2x     => B=[-4,2]
                     Polynomial C= A*B= -4+10x-16x^2+6^3 and the return will be: [ -4  10 -16   6]

    :param:
        @A,B vectors that represent Polynomials

    :return:
        vector of values that represent Polynomial C= A*B

    :complexity: O(n^2)
    """
    x = np.arange(-len(A), len(B) - 1)
    y = np.multiply([eval_coefficient(A, x) for x in x], [eval_coefficient(B, x) for x in x])
    # ev1 = np.vectorize(eval_(A,x_))
    # ev2 = np.vectorize(eval_())
    # y_ = np.multiply()
    # print("--", np.array_equal(y, y_))
    return interpolation(np.column_stack((x, y)))


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

    print(np.round(mult_coefficient(P, Q), 5))
    print('--------------------  mult polynomials test  ------------------------')
    print(np.array_equal(np.round(mult_point(A, B), 5), np.round(mult_coefficient(A, B), 5)))
    print(np.array_equal(np.round(mult_point(C, D), 2), np.round(mult_coefficient(C, D), 2)))
    print(np.array_equal(np.round(mult_point(E, F), 5), np.round(mult_coefficient(E, F), 5)))
    print(np.array_equal(np.round(mult_point(G, T), 5), np.round(mult_coefficient(G, T), 5)))
    print(np.array_equal(np.round(mult_point(X, Y), 5), np.round(mult_coefficient(X, Y), 5)))
    print(np.array_equal(np.round(mult_point(V, U), 5), np.round(mult_coefficient(V, U), 5)))
    print(np.array_equal(np.round(mult_point(I, J), 5), np.round(mult_coefficient(I, J), 5)))
    print(np.array_equal(np.round(mult_point(P, Q), 5), np.round(mult_coefficient(P, Q), 5)))

    print('--------------------  interpolation test  ------------------------')
    points = [(1, 12), (2, 15), (3, 16)]
    print(np.array_equal(interpolation_vandermonde(points), interpolation_lagrangh(points)))
    print(np.array_equal(interpolation_vandermonde(points), interpolation_newton(points)))
    points = [(5, 1), (-7, -23), (-6, -54)]
    print(np.array_equal(interpolation_vandermonde(points), interpolation_lagrangh(points)))
    print(np.array_equal(interpolation_vandermonde(points), interpolation_newton(points)))
    points = [(1, 2), (0, 2), (-1, 4)]
    print(np.array_equal(interpolation_vandermonde(points), interpolation_lagrangh(points)))
    print(np.array_equal(interpolation_vandermonde(points), interpolation_newton(points)))
    points = [(5, 1), (-7, -23), (-6, -54)]
    print(np.array_equal(interpolation_vandermonde(points), interpolation_lagrangh(points)))
    print(np.array_equal(interpolation_vandermonde(points), interpolation_newton(points)))

    # points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 2), (8, 2), (9, 3)]
    # print(interpolation_newton(points))
    points = [(-1, 4), (0, 1), (1, 0), (2, -5)]
    print(np.array_equal(interpolation_vandermonde(points), interpolation_lagrangh(points)))
    print(np.array_equal(interpolation_vandermonde(points), interpolation_newton(points)))
    print(interpolation_vandermonde(points))
