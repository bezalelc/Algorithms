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
# -------------------------  main  --------------------------------------


if __name__ == '__main__':
    V = np.array([[1, 1], [2, -2]])
    u = np.array([4, 5]).reshape((-1,))
    V_ = np.array([v / norm2(v) for v in V])
    eps = np.array([skalar_mult(u, v) for v in V_])
    base_orthonormal(V, u, skalar_mult, norm=norm2)
    # print(eps[0] * V_[0] + eps[1] * V_[1])
    # print(V_.shape, eps.shape, u.shape)
    # print(np.sum(eps * V_, axis=1))
