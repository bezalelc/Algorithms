"""
Author: Bezalel Cohen
"""
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs

# np.seterr(all='raise')
from sklearn.mixture import GaussianMixture


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
    pi, mu, sigma = np.ones((k,)) / k, np.random.rand(k, n) * np.mean(X), np.zeros(
        (k, n, n)) + np.eye(n) * 5
    sigma_reg = np.eye(n) * 1e-6
    w = np.empty((m, k))
    j_best = cost(X, pi, mu, sigma, w)

    while epoch:
        epoch -= 1
        # E-step
        sigma += sigma_reg
        w = Pr(X, mu, sigma, pr=w) * pi
        w = w / np.sum(w, axis=1)[:, None]
        # M-step
        w_pere_k = np.sum(w, axis=0)
        pi = w_pere_k / np.sum(w_pere_k)
        mu = (w.T @ X) / w_pere_k[:, None]
        for i in range(k):
            sigma[i] = (((w[:, i].reshape((-1, 1)) * (X - mu[i])).T @ (X - mu[i])) + sigma_reg) / w_pere_k[i]

        j = cost(X, pi, mu, sigma, w)
        # print(j, j_best)
        if j == j_best:
            break
        elif j > j_best:
            j_best = j

    return pi, mu, sigma


def best_mu(X, k, epoch=20, max_iter=20):
    best_pi, best_mu, best_sigma = mixture(X, k, epoch=epoch)
    j_best = cost(X, best_pi, best_mu, best_sigma)
    for i in range(max_iter):
        pi, mu, sigma = mixture(X, k, epoch=epoch)
        j = cost(X, pi, mu, sigma)
        print(j, j_best)
        if j < j_best:
            j_best = j
            best_pi, best_mu, best_sigma = pi.copy(), mu.copy(), sigma.copy()
    return best_pi, best_mu, best_sigma


def Pr(X, mu, sigma, pr=None):
    m, n, k = X.shape[0], mu.shape[1], mu.shape[0]
    pr = pr if pr is not None else np.empty((m, k))
    for i in range(k):
        X_ = X - mu[i]
        pr[:, i] = np.exp(-(1 / 2) * (np.sum((X_ @ np.linalg.pinv(sigma[i])) * X_, axis=1)))
        pr[:, i] /= (np.sqrt(((2 * np.pi) ** n) * np.linalg.det(sigma[i])))
    return pr


def cost(X, pi, mu, sigma, w=None):
    return np.log(np.sum(Pr(X, mu, sigma, w) * pi))


def predict(X, pi, mu, sigma):
    return np.argmax(pi * Pr(X, mu, sigma), axis=1)


# --------------------------------------  debug  --------------------------------------------
def mixture_plot(X, k, epoch=5):
    """
    plot each step
    Gaussian Mixture Models

    :param X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
           k: The desired number of dimensions

    :return:

    :efficiency:
    """
    # init
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    m, n = X.shape[0], X.shape[1]
    pi, mu, sigma = np.ones((k,)) / k, np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(k, n)), np.zeros(
        (k, n, n)) + np.eye(n) * 5
    sigma_reg = np.eye(n) * 1e-6
    # mu = np.array([[-5, 17], [-6, 11], [3, 1]])  # ----
    w = np.empty((m, k))

    # plot vars
    J_history = [cost(X, pi, mu, sigma + sigma_reg, w)]
    w_plot = np.empty((m ** 2, k))
    X_sort = np.sort(X, axis=0)
    x, y = np.meshgrid(X_sort[:, 0], X_sort[:, 1])
    XY = np.array([np.array([x.flatten(), y.flatten()])]).T.reshape((m ** 2, n))

    # plt.figure(figsize=(8, 8))
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    fig0 = plt.figure(figsize=(10, 10))
    fig0_111 = fig0.add_subplot(111)
    fig0_111.scatter(X[:, 0], X[:, 1])
    fig0_111.set_title('initial vars')

    w_plot = Pr(XY, mu, sigma + sigma_reg, pr=w_plot)
    for mu_, s, i in zip(mu, sigma, range(k)):
        fig0_111.contour(X_sort[:, 0], X_sort[:, 1], w_plot[:, i].reshape((m, m)), colors='black', alpha=0.3)
        fig0_111.scatter(mu_[0], mu_[1], c='grey', zorder=10, s=100)
    plt.show()

    for i in range(epoch):

        # E-step
        sigma += sigma_reg
        w = Pr(X, mu, sigma, pr=w) * pi
        w = w / np.sum(w, axis=1)[:, None]

        # M-step
        w_pere_k = np.sum(w, axis=0)
        pi = w_pere_k / m
        mu = (w.T @ X) / w_pere_k[:, None]

        for j in range(k):
            sigma[j] = ((w[:, j].reshape((-1, 1)) * (X - mu[j])).T @ (X - mu[j])) / w_pere_k[j]

        J_history.append(cost(X, pi, mu, sigma + sigma_reg, w))

        fig1 = plt.figure(figsize=(10, 10))
        fig1_111 = fig1.add_subplot(111)
        fig1_111.scatter(X[:, 0], X[:, 1])
        fig1_111.set_title('iterations ' + str(i))

        w_plot = Pr(XY, mu, sigma, pr=w_plot)
        for mu_, s, i in zip(mu, sigma, range(k)):
            fig1_111.contour(X_sort[:, 0], X_sort[:, 1], w_plot[:, i].reshape((m, m)), colors='black', alpha=0.3)
            fig1_111.scatter(mu_[0], mu_[1], c='grey', zorder=10, s=100)
        plt.show()

    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111)
    ax2.plot(range(epoch + 1), J_history)
    ax2.set_title('Log Likelihood Values')
    # fig2.savefig('GMM2D Log Likelihood.png')
    plt.show()

    return pi, mu, sigma


# --------------------------------------  main  --------------------------------------------
if __name__ == '__main__':
    # print('\n\n===================================== test ex7data2 =====================================')
    import scipy.io

    # X = scipy.io.loadmat('/home/bb/Documents/octave/week8/machine-learning-ex7/ex7/ex7data2.mat')['X']
    # pi, mu, sigma = mixture(X, 3)
    # print(predict(X[0:13], pi, mu, sigma))

    # print('\n\n===================================== test blobs  =====================================')
    from sklearn.datasets import make_blobs
    from scipy.stats import multivariate_normal

    # X, Y = make_blobs(cluster_std=1.5, random_state=20, n_samples=500, centers=3)
    # X = np.dot(X, np.random.RandomState(0).randn(2, 2))
    # y = np.random.randint(-10, 20, size=(12, 2))
    # pi, mu, sigma = mixture(X, 3, epoch=50)
    # pi, mu, sigma = mixture(X, 3, epoch=50)
    #
    # # print(predict(y, pi, mu, sigma))
    #
    # GMM = GaussianMixture(n_components=3)
    # GMM.fit(X)
    # # Y = np.random.randint(-10, 20, size=(1, 2))
    # print(mu)
    # print()
    # print(GMM.means_)

    print('---------------------------------  Reducing Image size  --------------------------')

    # /home/bb/Documents/octave/week8/machine-learning-ex7/ex7/bird_small.png
    # /home/bb/Downloads/IMG-20180617-WA0018 (copy).jpg
    img = imageio.imread('/home/bb/Documents/octave/week8/machine-learning-ex7/ex7/bird_small.png')
    X = np.array(img, dtype=np.float128)
    origin, D = np.array(X.copy(), dtype=np.uint8), X.shape
    print(D)
    X = X.reshape((X.shape[0] * X.shape[1], -1))
    X = X / 255
    pi, mu, sigma = mixture_plot(X[:, :2], 3, epoch=30)
    # p = predict(X, pi, mu, sigma)
    # print(mu)
    #
    # mu = np.round(mu * 255)
    # img = np.array(np.reshape(mu[p, :], D), dtype=np.uint8)
    #
    # # Y = np.random.randint(-10, 20, size=(1, 2))
    # GMM = GaussianMixture(n_components=16)
    # GMM.fit(X)
    # # p = GMM.predict(X)
    # mu_ = GMM.means_
    # print()
    # print(mu_)
    # # mu = np.round(mu * 255)
    # # img = np.array(np.reshape(mu[p, :], D), dtype=np.uint8)
    #
    # img = Image.fromarray(img, 'RGB')
    # # img.save('/home/bb/Downloads/mixture_bird.png')
    # # # # img.show()
    # # # #
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(origin)
    # plt.title('origin image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(img)
    # plt.title('compressed image')
    # plt.show()
