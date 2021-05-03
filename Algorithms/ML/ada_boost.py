"""
example of ada boost algorithm
"""
import numpy as np
import sympy as sp
import math


def fit(X, y, F):
    # init
    m, F, F_len = X.shape[0], sp.Matrix(F), F.shape[0]
    func = sp.lambdify(x, F, 'numpy')
    P = np.array(func(X[:, 0]).reshape((-1, m)) == y, dtype=bool)
    H, W, eps, alpha = [], np.ones((m,)) / m, np.empty((F_len,)), []  #

    for i in range(F_len // 2):
        for j, p in zip(range(F_len), P):
            eps[j] = np.sum(W[p == False])
        h_min = np.argmin(eps)
        H.append(h_min), alpha.append(0.5 * math.log((1 - eps[h_min]) / eps[h_min]))
        W[P[h_min]] /= (1 - eps[h_min])
        W[P[h_min] == False] /= eps[h_min]
        W /= 2

    # print(sp.lambdify(x,F[0],'numpy')*9)
    # # predict = [ep * F[i] for ep, i in zip(eps, H)]
    # # print(predict)
    # # print(func[0] * eps[0])


if __name__ == '__main__':
    x = sp.symbols('x')
    F = np.array([x < 2, x < 4, x < 6, x > 2, x > 4, x > 6])

    X, y = np.array([[1, 5], [5, 5], [3, 3], [1, 1], [5, 1]]), np.array([1, 1, 0, 1, 1])
    # print(F(1))
    fit(X, y, F)
