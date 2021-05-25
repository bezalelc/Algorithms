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
from Algorithms.polynomial import fft, interpolation as inter


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


def mult_fft(P1, P2):
    """
    multiply to Polynomials that represent by vector of Coefficients
        for example: Polynomial P1= 1-2x+3x^2 => P1=[1,-2,3]
                     Polynomial P2= -4+2x     => P2=[-4,2]
                     Polynomial C= P1*P2= -4+10x-16x^2+6^3 and the return will be: [ -4  10 -16   6]

    :param:
        @P1, P2 vectors that represent Polynomials

    :return:
        vector of Coefficients that represent Polynomial C= A*B

    :complexity: O(n*log(n))
    """
    P1, P2 = np.array(P1), np.array(P2)
    P1, P2 = np.append(P1, np.zeros((P2.shape[0],))), np.append(P2, np.zeros((P1.shape[0],)))
    DFT1, DFT2 = fft.fft(P1), fft.fft(P2)  # np.fft.fft(P1), np.fft.fft(P2)
    # DFT1, DFT2 = np.fft.fft(P1), np.fft.fft(P2)  # np.fft.fft(P1), np.fft.fft(P2)
    coeff = fft.fft_reverse(DFT1 * DFT2)
    # coeff = np.fft.ifft(DFT1 * DFT2)
    return coeff if coeff[-1] != 0 else coeff[:np.where(coeff == 0)[0][-1]]


def horner(P, x, version='rec'):
    """
    recursive version
    multiply polynomial with horner's rule: P=[a0,a1,a2,...,an]
        P(x)=a0+x(a1+x(a2+...x(an))))

    :param
        P: vectors that represent Polynomials
        x: x point to eval P(x)
        version: options: 'rec' => recursive horner algorithm
                           else => iterative horner algorithm


    :return: P(x)

    :complexity: O()
    """

    def iter(P, x):
        """
        iterative version
        """
        res = P[-1]
        for a in P[:-1][::-1]:
            res = a + x * res
        return res

    def rec(P, x, n):
        """
         recursive vesion
        """
        if n == 1:
            return P[-1] * x
        return x * (P[-n] + rec(P, x, n - 1))

    P = np.array(P)

    if version == 'rec':
        return P[0] + rec(P, x, P.shape[0] - 1)
    return iter(P, x)


def mult_point(A, B, interpolation_=inter.vandermonde):
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
    return interpolation_(np.column_stack((x, y)))


def mult_large_num(a, b):
    P1, P2 = np.array(list(map(int, str(a))))[::-1], np.array(list(map(int, str(b))))[::-1]
    return eval_coefficient(mult_fft(P1, P2), 10.)


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
    print(np.array_equal(inter.vandermonde(points), inter.lagrangh(points)))
    print(np.array_equal(inter.vandermonde(points), inter.newton(points)))
    points = [(5, 1), (-7, -23), (-6, -54)]
    print(np.array_equal(inter.vandermonde(points), inter.lagrangh(points)))
    print(np.array_equal(inter.vandermonde(points), inter.newton(points)))
    points = [(1, 2), (0, 2), (-1, 4)]
    print(np.array_equal(inter.vandermonde(points), inter.lagrangh(points)))
    print(np.array_equal(inter.vandermonde(points), inter.newton(points)))
    points = [(5, 1), (-7, -23), (-6, -54)]
    print(np.array_equal(inter.vandermonde(points), inter.lagrangh(points)))
    print(np.array_equal(inter.vandermonde(points), inter.newton(points)))

    # points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 2), (8, 2), (9, 3)]
    # print(interpolation_newton(points))
    points = [(-1, 4), (0, 1), (1, 0), (2, -5)]
    print(np.array_equal(inter.vandermonde(points), inter.lagrangh(points)))
    print(np.array_equal(inter.vandermonde(points), inter.newton(points)))
    print(inter.vandermonde(points))

    print('-------------------  mult fft mult_large_num test 1  ----------------------------')
    a, b = 6586546, 79658485
    print(a * b, mult_large_num(a, b))
