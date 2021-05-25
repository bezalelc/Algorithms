import numpy as np


def lup(A, eps=1e-7):
    """
    Disassembly of a matrix to L,U,pi with Constraint diag(U)=1

    :param
        A: matrix
        eps: small value for numeric problems

    :return:  L,U,pi

    :complexity: O(n^3)
    """
    A = np.array(A)
    LU, pi, n = A.copy(), np.arange(A.shape[0]), A.shape[0]

    for k in range(n):
        p, k_ = np.max(LU[k:, k]), abs(np.argmax(LU[k:, k])) + k
        if abs(p) < eps:
            print("no disassembly, numeric issue")
            return None

        if k != k_:
            pi[k], pi[k_] = pi[k_], pi[k]
            LU[[k, k_]] = LU[[k_, k]]

        LU[k + 1:, k] /= LU[k, k]
        LU[k + 1:, k + 1:] = LU[k + 1:, k + 1:] - LU[k + 1:, k][:, None] @ LU[k, k + 1:][:, None].T

    L, U = np.tril(LU), np.triu(LU)
    L[np.diag_indices_from(L)] = 1

    return L, U, pi


def lup_(A):
    """

    :param A:

    :return:  L,U

    :complexity: O(n^3)
    """
    A = np.array(A)
    L, U, pi, n = np.zeros(A.shape), np.zeros(A.shape), np.arange(A.shape[0]), A.shape[0]

    for k in range(n):
        U[k, k], L[k, k] = A[k, k], 1
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / A[k, k]
            U[k, i] = A[k, i]

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i, j] -= L[i, k] * U[k, j]

    return L, U


def lup__(A, eps=1e-7):
    """

    :param A:

    :return:  L,U

    :complexity: O(n^3)
    """
    A = np.array(A)
    LU, pi, n = A.copy(), np.arange(A.shape[0]), A.shape[0]
    # L, U, pi, n = np.zeros(A.shape), np.zeros(A.shape), np.arange(A.shape[0]), A.shape[0]

    for k in range(n):
        p, k_ = 0, k
        for i in range(k, n):
            if abs(LU[i, k]) > p:
                p = abs(LU[i, k])
                k_ = i
        if p < eps:
            print("no solution")
            return False
        # print(f'--------  {k}  -------------')
        # print(LU)
        # print(p, pi, k, k_, '\n')
        pi[k], pi[k_] = pi[k_], pi[k]
        LU[[k, k_]] = LU[[k_, k]]
        # print(LU)
        # print(p, pi, k, k_, '\n')

        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            for j in range(k + 1, n):
                LU[i, j] -= LU[i, k] * LU[k, j]

    L, U = np.tril(LU), np.triu(LU)
    L[np.diag_indices_from(L)] = 1
    return L, U, pi


def solve(A, b, disassembly=None):
    """
    solve Ax=b linear system with LUP disassembly

    :param A: matrix n x n
    :param b: vector n x 1
    :param disassembly: L,U,pi disassembly of A

    :return: x: vector n x 1 of the solution of linear system

    :Algorithm:
        disassembly A to L,U,pi => L @ U = A[pi] =>  (L @ U) @ x = b[pi]
        solve L @ y = b[pi]
        solve U @ x = y


    :complexity: O(n^2) if L,U,pi given else O(n^3)
    """
    L, U, pi = disassembly if disassembly else lup(A)
    b = b[pi]
    x, y, n = np.empty(b.shape[0]), np.empty(b.shape[0]), b.shape[0]

    # solve
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - np.sum(L[i, :i] * y[:i])
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i + 1:] * x[i + 1:])) / U[i, i]

    return x


if __name__ == '__main__':
    A = np.array([[2, 3, 1, 5], [6, 13, 5, 19], [2, 19, 10, 23], [4, 10, 11, 31]], dtype=np.float64)
    # print(A, '\n')
    L, U = lup_(A)
    print(np.array_equal(L @ U, A))
    L, U, pi = lup__(A)
    print(np.array_equal(L @ U, A[pi]))  # [np.array([1, 2, 0])]
    L, U, pi = lup(A)
    print(np.array_equal(L @ U, A[pi]))

    A = np.array([[1, 2, 0], [3, 4, 4], [5, 6, 3]], dtype=np.float64)
    L, U = lup_(A)
    print(np.array_equal(L @ U, A))
    L, U, pi = lup__(A)
    print(np.array_equal(L @ U, A[pi]))  # [np.array([1, 2, 0])]
    L, U, pi = lup(A)
    print(np.array_equal(L @ U, A[pi]))
    # print(np.array_equal(lup(A), lup_(A)))
    # print(np.array_equal(lup(A), lup__(A)))
    # A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # pi = np.array([1, 2, 0])
    # # pi = (1, 2, 0)
    # print(A[pi, :])

    print('------------  solve  ----------------')
    A, b = np.array([[1, 2, 0], [3, 4, 4], [5, 6, 3]], dtype=np.float64), np.array([3, 7, 8])
    L, U, pi = lup(A)
    print(np.array_equal(L @ U, A[pi]))
    x = solve(A, b, disassembly=[L, U, pi])
    print(np.array_equal(np.around(x, decimals=10), np.around(np.linalg.solve(A, b), decimals=10)))
    print(x)
    print(np.linalg.solve(A, b))

    A, b = np.array([[2, 3, 1, 5], [6, 13, 5, 19], [2, 19, 10, 23], [4, 10, 11, 31]], dtype=np.float64), np.array(
        [3, 8, 7, 8])
    L, U, pi = lup(A)
    print(np.array_equal(L @ U, A[pi]))
    x = solve(A, b, disassembly=[L, U, pi])
    print(np.array_equal(np.around(x, decimals=10), np.around(np.linalg.solve(A, b), decimals=10)))
    print(x)
    print(np.linalg.solve(A, b))
