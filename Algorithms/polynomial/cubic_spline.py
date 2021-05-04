import numpy as np
import sympy as sp


def cubic_spline4_matrix(points):
    """
    calculate the coefficients for cubic spline

    points: (n+1)x2 Matrix of points when Matrix[:,1]=x points and Matrix[:,2] = y points,
            n is the rank of the polynomial

    :return: [lambdify(sympy Matrix)] foe each S=[s0,s1,...,sn] where s_i(x)=a_i*x^3+b_i*x^2+c_i*x+d_i

    :complexity: O((4n)^3) where n is points number -1
     """
    # init
    points = np.array(points)
    points = points[points[:, 0].argsort()]
    x, y, n = points[:, 0], points[:, 1], points.shape[0] - 1
    X = np.vander(x, N=4, increasing=False)

    # build M,A
    M, A, idx = np.zeros((4 * n, 4 * n)), np.zeros((4 * n,)), np.arange(n * 4).reshape((-1, 4))
    M[idx // 4, idx] = X[:-1, :]
    M[idx // 4 + n, idx] = X[1:, :]
    idx = idx[:-1, :-1]
    M[idx // 4 + n * 2, idx] = X[1:-1, 1:] * np.array([3, 2, 1])
    M[idx // 4 + n * 2, idx + 4] = -M[idx // 4 + n * 2, idx]
    idx = idx[:, :-1]
    M[idx // 4 + n * 3 - 1, idx] = X[1:-1, 2:] * np.array([6, 2])
    M[idx // 4 + n * 3 - 1, idx + 4] = -M[idx // 4 + n * 3 - 1, idx]
    M[-2, :2], M[-1, -4:-2] = X[0, 2:] * np.array([6, 2]), X[-1, 2:] * np.array([6, 2])
    A[:n], A[n:2 * n] = y[:-1], y[1:]

    # calculate the coefficients
    coeff = np.linalg.solve(M, A).reshape((-1, 4))
    x = sp.symbols('x')
    S = coeff[:, 0] * x ** 3 + coeff[:, 1] * x ** 2 + coeff[:, 2] * x + coeff[:, 3]

    # return function
    return map_S(S, x, points)


def cubic_spline4(points):
    """
    calculate the coefficients for cubic spline

    points: (n+1)x2 Matrix of points when Matrix[:,1]=x points and Matrix[:,2] = y points,
            n is the rank of the polynomial

    :return: [lambdify(sympy Matrix)] foe each S=[s0,s1,...,sn] where s_i(x)=a_i*x^3+b_i*x^2+c_i*x+d_i


    :complexity: O(n) where n is points number
     """
    # init
    points = np.array(points, dtype=np.float64)
    points = points[points[:, 0].argsort()]
    x, y, n = points[:, 0], points[:, 1], points.shape[0] - 1
    h = x[1:] - x[:-1]
    b = (y[1:] - y[:-1]) * (6 / h)
    u, v = 2 * (h[1:] + h[:-1]), b[1:] - b[:-1]

    # rank
    for i in range(1, n - 1):
        u[i] -= ((h[i] ** 2) / u[i - 1])
        v[i] -= (h[i] / u[i - 1]) * v[i - 1]

    z = np.empty((n + 1,))
    z[0], z[-1] = 0, 0

    # solve
    for i in range(n - 1, 0, -1):
        z[i] = (v[i - 1] - h[i] * z[i + 1]) / u[i - 1]

    # calc S
    c = y[1:] / h - (z[1:] * h) / 6
    d = y[:-1] / h - (z[:-1] * h) / 6
    x_ = sp.symbols('x')
    S = (z[:-1]) / (6 * h) * (x[1:] - x_) ** 3 + z[1:] / (6 * h) * (x_ - x[:-1]) ** 3
    S += c * (x_ - x[:-1]) + d * (x[1:] - x_)

    return map_S(S, x_, points)
    # return sp.lambdify(x_, sp.Matrix(S), 'numpy')


def map_S(S, x, points):
    func = [sp.lambdify(x, s, 'numpy') for s in S]
    # map_points = lambda p: int((p - start) // (range_ / n)) if p != end else n - 1
    map_points = lambda x0: np.searchsorted(points[:, 0], x0) - 1 if x0 != points[0, 0] else 0
    splines = np.vectorize(lambda x0: func[map_points(x0)](x0))
    return splines


if __name__ == '__main__':
    # print('-------------------  cubic spline: n=4  ----------------------------')
    # points = [(1, 1), (2, 2), (3, 3), (4, 4)]
    # print(cubic_spline4(points))
    # cubic_spline4(points)
    #
    # points = [(0, 0.3), (1, 1), (2, 5), (5, 7)]
    # print(cubic_spline4_matrix(points)(0.7))
    # print(cubic_spline4(points)(0.7))
    # print('-------------------  cubic spline: n=4, test 3  ----------------------------')
    # # points = [(3, 5), (1, 2), (2, 9), (3, -1)]
    # points = [(3, 5), (1, 2), (2, 9), (6, -1)]
    # print(cubic_spline4_matrix(points)(1))
    # print(cubic_spline4(points)(1))
    #
    # print('-------------------  cubic spline: n=4, test 4  ----------------------------')
    # points = [(4, 0), (1, 1), (2, 0), (3, -1)]
    # print(cubic_spline4_matrix(points)(1))
    # print(cubic_spline4(points)(1))
    print('-------------------  test 5  ----------------------------')
    points = np.linspace(7, 12, num=4)
    # points = np.arange(4)
    # print(points)
    x = sp.symbols('x')
    f = sp.lambdify(x, sp.sin(x), 'numpy')
    points = np.concatenate((points[:, None], f(points)[:, None]), axis=1)
    # print(points)
    real_points = np.linspace(7, 12, num=1000)
    # points = [(1, 1), (2, 0), (3, -1), (4, 0)]
    # print(cubic_spline4(points)(1))
    # spline_mat = cubic_spline4_matrix(points)
    # spline = cubic_spline4(points)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(real_points, spline_mat(real_points), c='k')
    # plt.plot(real_points, spline(real_points), c='b')
    # plt.plot(real_points, f(real_points), c='r')
    # plt.show()

    print('-------------------  test 5  ----------------------------')
    points = [(0.9, 1.3), (1.3, 1.5), (1.9, 1.85), (2.1, 2.1), (9, 0)]
    # points = [(0, 0),(1,1),(2,0),(3,-1)]
    # points = [(0,0),(1,-1),(3,0),(5,-1)]
    points = [(1, 1), (4, 2), (9, 3)]
    f = cubic_spline4(points)
    print(f(np.arange(1, 10)))
    # print(f(4.5))
    # points = np.array(points)[:, 0]
    # points = points[points.argsort()]
    # print(np.searchsorted(points, 1.1))
