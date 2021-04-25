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
    x, y, n = points[:, 0], points[:, 1], points.shape[0] - 1
    X = np.vander(x, N=4, increasing=False)

    # build M,A
    M, A, idx = np.zeros((4 * n, 4 * n)), np.zeros((4 * n,)), np.arange(4 * n)
    M[idx // 4, idx], A[idx[:n]] = X[:-1, :].reshape((-1,)), y[:-1]
    M[n + idx // 4, idx], A[idx[n:2 * n]] = X[1:, :].reshape((-1,)), y[1:]
    M[2 * n + idx[:-4] // 4, idx[:-4]] = (X[1:-1, :] * np.array([3, 2, 1, 0])).reshape((-1,))
    M[2 * n + idx[:-4] // 4, idx[4:]] = (X[1:-1, :] * np.array([3, 2, 1, 0])).reshape((-1,))
    M[3 * n - 1 + idx[:-4] // 4, idx[:-4]] = (X[1:-1, :] * np.array([6, 1, 0, 0])).reshape((-1,))
    M[3 * n - 1 + idx[:-4] // 4, idx[4:]] = (X[1:-1, :] * np.array([6, 1, 0, 0])).reshape((-1,))
    M[-2, 0:4] = np.array([6, 1, 0, 0])
    M[-1, n * 4 - 4:n * 4] = np.array([6, 1, 0, 0])

    # calculate the coefficients
    coeff = np.linalg.solve(M, A).reshape((-1, 4))
    x_ = sp.symbols('x')
    S = coeff[:, 0] * x_ ** 3 + coeff[:, 1] * x_ ** 2 + coeff[:, 2] * x_ + coeff[:, 3]

    # return function
    func = [sp.lambdify(x_, S[i], 'numpy') for i in range(len(S))]
    start, end = np.min(points[:, 0]), np.max(points[:, 0])
    rang = end - start
    map_points = lambda p: int((p - start) // (rang / n)) if p != end else n - 1
    splines = np.vectorize(lambda p: func[map_points(p)](p))
    return splines
    # return sp.lambdify(x_, sp.Matrix(S), 'numpy')


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
    x, y, n = points[:, 0], points[:, 1], points.shape[0] - 1
    h = x[1:] - x[:-1]
    b = (y[1:] - y[:-1]) * (6 / h)
    u, v = 2 * (h[1:] + h[:-1]), b[1:] - b[:-1]

    # rank
    u[1:] -= h[1:-1] ** 2 / u[:-1]
    v[1:] -= v[:-1] * (h[1:-1] / u[:-1])
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

    # return function
    func = [sp.lambdify(x_, S[i], 'numpy') for i in range(len(S))]
    start, end = np.min(points[:, 0]), np.max(points[:, 0])
    rang = end - start
    map_points = lambda p: int((p - start) // (rang / n)) if p != end else n - 1
    splines = np.vectorize(lambda p: func[map_points(p)](p))
    return splines
    # return sp.lambdify(x_, sp.Matrix(S), 'numpy')


if __name__ == '__main__':
    points = np.linspace(-6, 6, num=3)
    # points = np.vstack(np.linspace(-6, 6, num=3)
    # print(points)
    # cubic_spline4()
