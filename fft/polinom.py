import numpy as np


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


def mult_(A, B):
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
    y = np.multiply([eval_(A, x) for x in x], [eval_(B, x) for x in x])
    # ev1 = np.vectorize(eval_(A,x_))
    # ev2 = np.vectorize(eval_())
    # y_ = np.multiply()
    # print("--", np.array_equal(y, y_))
    return interpolation_vandermonde(np.column_stack((x, y)))


def mult_coef(A, B):
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


def coef_to_point(X, points):
    pass


def point_to_coef():
    pass


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

    :complexity: O(n^3)
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

    :complexity: O(n^3)
    """
    points = np.array(points)
    pass
    # for i in points.shape[0]:


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

    print(np.round(mult_coef(P, Q), 5))

    print(np.array_equal(np.round(mult_(A, B), 5), np.round(mult_coef(A, B), 5)))
    print(np.array_equal(np.round(mult_(C, D), 2), np.round(mult_coef(C, D), 2)))
    print(np.array_equal(np.round(mult_(E, F), 5), np.round(mult_coef(E, F), 5)))
    print(np.array_equal(np.round(mult_(G, T), 5), np.round(mult_coef(G, T), 5)))
    print(np.array_equal(np.round(mult_(X, Y), 5), np.round(mult_coef(X, Y), 5)))
    print(np.array_equal(np.round(mult_(V, U), 5), np.round(mult_coef(V, U), 5)))
    print(np.array_equal(np.round(mult_(I, J), 5), np.round(mult_coef(I, J), 5)))
    print(np.array_equal(np.round(mult_(P, Q), 5), np.round(mult_coef(P, Q), 5)))

    points = [(1, 12), (2, 15), (3, 16)]
    print(interpolation_vandermonde(points))
