"""
Author: Bezalel Cohen
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs


class GMM():

    def __init__(self) -> None:
        super().__init__()
        self.X, self.EPOCHS = [], 100
        self.mu, self.sigma, self.pi = [], [], []

    def compile(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def cost(self):
        pass

    def loss(self):
        pass


def mixture(X, k, epoch=30):
    """
    Gaussian Mixture Models

    :param X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
           k: The desired number of dimensions

    :return:

    :efficiency:
    """
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    m, n = X.shape[0], X.shape[1]

    pi, mu, sigma = np.ones((k,)) / k, X[np.random.randint(m, size=k), :], np.array([np.eye(n) for i in range(k)])
    sigma_reg = sigma * 1e-6
    # init sigma
    # x = np.mean(X, axis=0)
    # x = ((X - x).T @ (X - x)) / m
    # for i in range(k):
    # sigma[i] = x / m
    # print(sigma[i])
    # ,np.random.rand(k, m),np.random.randint(np.min(X), np.max(X), size=(k, n)),np.ones((k,)) / k
    w, pr, y = np.empty((m, k)), np.empty((m, k)), np.random.randint(k, size=(m,))  # np.random.randint(k, size=(m,))
    # X_groups, y_groups = [], []

    while epoch:
        epoch -= 1
        # print(epoch)
        # for i in range(k):
        # idx = np.argwhere(y == i)

        # E-step
        sigma += sigma_reg
        pr = Pr(X, mu, sigma, pr=pr)
        print(pr[0:3])

        tmp = multivariate_normal(cov=sigma[0], mean=mu[0])
        print(tmp)
        w = pr * pi
        # print(w.shape)
        w = w / np.sum(w, axis=1)[:, None]

        # print(w)

        # M-step
        w_pere_k = np.sum(w, axis=0)
        pi = w_pere_k / m
        mu = (w.T @ X) / w_pere_k[:, None]
        # x = (X - mu).T @ (X - mu)
        # print(x.shape)

        for i in range(k):
            sigma[i] = (X - mu[i]).T @ (X - mu[i]) / w_pere_k[i]

        j = cost(pi, pr)
        # print(j)
        # if epoch == 27:
        break


def Pr(X, mu, sigma, pr=None):
    m, n, k = X.shape[0], mu.shape[0], mu.shape[1]
    pr = pr if pr is not None else np.empty((m, k))
    for i in range(k):
        pr[:, i] = np.exp(-0.5 * np.sum(((X - mu[i]) @ np.linalg.pinv(sigma[i])) * (X - mu[i]), axis=1))
        pr[:, i] /= (((2 * np.pi) ** (n / 2)) * np.sqrt(np.linalg.det(sigma[i])))
    return pr


def cost(pi, pr):
    return np.sum(pr * pi)


if __name__ == '__main__':
    print('\n\n===================================== test ex7data2 =====================================')
    import scipy.io

    X = scipy.io.loadmat('/home/bb/Documents/octave/week8/machine-learning-ex7/ex7/ex7data2.mat')['X']
    # mixture(X, 3)
    # print(len(X))
    # x = np.zeros((5, 7))
    # print(x is None)

    print('\n\n===================================== test  =====================================')
    X, Y = make_blobs(cluster_std=1.5, random_state=20, n_samples=500, centers=3)
    X = np.dot(X, np.random.RandomState(0).randn(2, 2))
    mixture(X, 3)
