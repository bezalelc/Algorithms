import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from general import normalize


def fit(X, k):
    """
    compress data X (mxn) to Z (mxk)

    :param X: (mxn) matrix, compressed data
    :param k: dimension to compress from n to k

    :return:
        Z: (mxn) matrix, compressed data
        U: Matrix of self-values vectors of Sigma(X) =  X.T @ X / m

    :complexity: O()
    """
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    m, n = X.shape
    X, mu, sigma = normalize.standard_deviation(X, ddof=1)
    sigma_ = X.T @ X / m  # covariance matrix
    U, S, D = np.linalg.svd(sigma_)
    Z = X @ U[:, :k]
    return Z, U, S


def recover(Z, U):
    """
    recover X data from the compressed data Z
    :param Z: (mxn) matrix, compressed data
    :param U: Matrix of self-values vectors of Sigma(X) =  X.T @ X / m

    :return: recover data: (mxn) matrix

    :complexity: O()
    """
    return Z @ U[:, :Z.shape[1]].T


def choose_k(X, eps):
    """
    choose the best k for compressed X (mxn) to Z (mxk) with loss <= eps

    :param X: (mxn) matrix, compressed data
    :param eps: max loss data

    :return:
        k: best k

    :complexity: O()
    """
    if len(X.shape) == 1:
        return 1
    k = X.shape[1]
    U, S = fit(X, 1024)[1:]
    while k > 1:
        X_rec = recover(X @ U[:, :k], U)
        J = cost(X, X_rec)
        if J >= eps:
            print(J)
            return k
        else:
            k -= 1


def choose_k_(X, eps):
    """
    choose the best k for compressed X (mxn) to Z (mxk) with loss <= eps

    :param X: (mxn) matrix, compressed data
    :param eps: max loss data

    :return:
        k: best k

    :complexity: O()
    """
    U, S = fit(X, 1024)[1:]
    sum_, s, k = np.sum(S), 0, len(S)

    for i in range(len(S)):
        s += S[i]
        if 1 - s / sum_ >= eps:
            print(1 - s / sum_, i)
            break
        else:
            k = i
    return k


def cost(X, X_rec):
    return np.sum((X - X_rec) ** 2) / np.sum(X ** 2)


def plot_images(X, r, c, h, w):
    fig = plt.figure(figsize=(10, 10))
    idx = np.random.randint(X.shape[0], size=(r * c,))
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            # print(i, j, (i - 1) * c + j)
            fig.add_subplot(r, c, (i - 1) * c + j)
            plt.imshow(X[idx[(i - 1) * c + j - 1], :].reshape((h, w)).T, cmap='gray')
            plt.axis('off')
            # plt.title("Third")

    plt.show()


if __name__ == '__main__':
    print('---------------------  test ex7data1  -----------------------------')
    data = io.loadmat('/home/bb/Documents/octave/week8/machine-learning-ex7/ex7/ex7data1.mat')
    X = data['X']

    # print(data)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    # fit(X, 1)

    print('---------------------  test ex7faces  -----------------------------')
    data = io.loadmat('/home/bb/Documents/octave/week8/machine-learning-ex7/ex7/ex7faces.mat')
    X = data['X']
    z, u, s = fit(X, 3)
    print(choose_k_(s, 0.01))
    # plot_images(X, 3, 3, 32, 32)
    # z, u = fit(X, 100)
    # X_rec = recover(z, u)
    # plot_images(X_rec, 3, 3, 32, 32)
