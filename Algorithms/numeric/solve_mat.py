import numpy as np


def thomas(A, b):
    """
    solve A @ x = b for Three-diagonal matrix


    :param A: matrix in size n x n
    :param b: vector in size n x 1

    :return: x => solution vector in size n x 1

    :complexity:
        run-time => O(n^3)
        memory => O(n)
    """
    A, b, n = np.array(A, dtype=np.float64), np.array(b, dtype=np.float64), len(b)
    A, b, d, a, r, x = np.c_[A, b], np.diag(A, k=-1), np.diag(A), np.diag(A, k=1).copy(), b.copy(), np.empty(n)

    # rank
    a[0] /= d[0]
    r[0] /= d[0]
    for i in range(1, n - 1):
        a[i] /= (d[i] - b[i] * a[i - 1])
        r[i] = (r[i] - b[i] * r[i - 1]) / (d[i] - b[i] * a[i - 1])
    # solve
    x[-1] = r[-1]
    for i in range(n - 2, -1, -1):
        x[i] = r[i] - a[i] * x[i + 1]

    return x


if __name__ == '__main__':
    A, b = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]]), np.array([1, 0, 1])
    # A, b = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]], dtype=np.float64), np.array(
    #     [1, 0, 0, 1])
    print(thomas(A, b))
    print(np.linalg.solve(A, b))
