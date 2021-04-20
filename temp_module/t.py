import numpy as np


# -------------------------  norm  --------------------------------------
def norm1(v):
    return np.sum(np.abs(v))


def norm2(v):
    return np.sqrt(np.sum(v ** 2))


def norm_inf(v):
    return np.max(np.abs(v))


# -------------------------  internal multiply  --------------------------------------
def skalar_mult(u, v):
    u, v = u.reshape((-1,)), v.reshape((-1,))
    return np.sum(u * v)


# -------------------------  norm_to_unit  --------------------------------------
def norm_to_unit(v, mult):
    return v / mult(v, v)


# -------------------------  orthonormal base  --------------------------------------
def base_orthonormal(V, u, mult, norm):
    if norm:
        V = np.array([v / norm(v) for v in V])
    eps = np.array([mult(u, v) for v in V])
    return eps


# -------------------------  pariah base   --------------------------------------
def pariah(n):
    pass


# -------------------------  function  --------------------------------------
# def
# -------------------------  C  --------------------------------------
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

    # print(x, y, f)
    # print(np.linalg.solve(M, f))
    return np.linalg.solve(M, f)


# -------------------------  main  --------------------------------------


if __name__ == '__main__':
    V = np.array([[1, 1], [2, -2]])
    u = np.array([4, 5]).reshape((-1,))
    V_ = np.array([v / norm2(v) for v in V])
    eps = np.array([skalar_mult(u, v) for v in V_])
    print(np.sum(base_orthonormal(V, u, skalar_mult, norm=norm2) * V_, axis=1))

    print('---------------------------  test C  ------------------------------')
    points = [(0, 1), (1, 3), (2, 7), (4, 4)]
    C_poly(points, 2, skalar_mult)
