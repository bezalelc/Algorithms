import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import loss


def gaussian(X):
    mu = np.mean(X, axis=0)
    sigma = np.mean((X - mu) ** 2, axis=0)
    return mu, sigma


def predict(X, mu, sigma):
    p = np.exp(-0.5 * ((X - mu) / sigma) ** 2) / ((2 * np.pi) ** 0.5 * sigma)
    p = np.prod(p, axis=1)
    return p


# --------------------------------------------  multivariate gaussian ----------------------------------------------
def multivariate_gaussian(X):
    mu = np.mean(X, axis=0)
    sigma = (X - mu).T @ (X - mu) / X.shape[0]
    return mu, sigma


def multivariate_predict(X, mu, sigma):
    m, n = X.shape[0], X.shape[1]
    X = X - mu
    p = ((2 * np.pi) ** n * np.linalg.det(sigma)) ** -0.5 * np.exp(
        -0.5 * (np.sum(X @ np.linalg.pinv(sigma) * X, axis=1)))
    return p


if __name__ == '__main__':
    data = io.loadmat('/home/bb/Documents/octave/week9/machine-learning-ex8/ex8/ex8data1.mat')
    X, Xval, yval = data['X'], data['Xval'], data['yval']

    m, n = X.shape
    mu, sigma = gaussian(X)
    print('mu:\n', mu, '\nsigma:\n', sigma)
    p = predict(X, mu, sigma)
    eps = 0.02

    mu_multy, sigma_multy = multivariate_gaussian(X)
    print('mu_multy:\n', mu_multy, '\nsigma_multy:\n', sigma_multy)
    p_multy = multivariate_predict(X, mu_multy, sigma_multy)

    # print(np.array_equal(np.around(p, decimals=1), np.around(p_multy, decimals=1)))

    # plot
    X_sort = np.sort(X, axis=0)
    x, y = np.meshgrid(X_sort[:, 0], X_sort[:, 1])
    XY = np.array(np.array([x.flatten(), y.flatten()]).T.reshape((m ** 2, n)))
    z = predict(XY, mu, sigma)
    z_multy = multivariate_predict(XY, mu_multy, sigma_multy)

    # plot hist
    plt.hist(X[:, 0], bins=100, alpha=0.6, color='r')
    plt.hist(X[:, 1], bins=100, alpha=0.3, color='b')
    plt.hist2d(X[:, 0], X[:, 1], bins=40)  # , bins=100, alpha=0.3, color='b'
    plt.show()
    # log(X+c),X^(1/c) #, bins=np.arange((np.min(X[:, 0]) - 1, np.max(X[:, 0] + 1)), 100)

    fig = plt.figure(figsize=(10, 10))
    fig111 = fig.add_subplot(121)
    # colors, area = np.random.rand(m), (20 * np.random.rand(m)) ** 2
    fig111.set_xlabel('Latency (ms)')
    fig111.set_ylabel('Throughput (mb/s)')
    fig111.set_title('normal')
    fig111.contour(X_sort[:, 0], X_sort[:, 1], z.reshape((m, m)))
    fig111.scatter(mu[0], mu[1], alpha=1, c='r', s=5, marker='o')
    colors = np.array(['r', 'b'])[np.array(p >= eps, dtype=np.uint8)]
    fig111.scatter(X[:, 0], X[:, 1], alpha=0.8, c=colors, s=3, marker='o')
    # print(f'p={p_} >= epsilon? {p_ >= eps}')

    fig112 = fig.add_subplot(122)
    # colors, area = np.random.rand(m), (20 * np.random.rand(m)) ** 2
    fig112.set_xlabel('Latency (ms)')
    fig112.set_ylabel('Throughput (mb/s)')
    fig112.set_title('multivariate')
    fig112.contour(X_sort[:, 0], X_sort[:, 1], z_multy.reshape((m, m)))
    fig112.scatter(mu_multy[0], mu_multy[1], alpha=1, c='r', s=5, marker='o')
    colors = np.array(['r', 'b'])[np.array(p_multy >= eps, dtype=np.uint8)]
    fig112.scatter(X[:, 0], X[:, 1], alpha=0.8, c=colors, s=3, marker='o')

    plt.show()

    # print loss
    print('normal:')
    p_val = predict(Xval, mu, sigma) <= eps
    print('accuracy', loss.accuracy(p_val, yval), end=', ')
    print('F_score', loss.F_score(p_val, yval), end=', ')
    print('recall', loss.recall(p_val, yval), end=', ')
    print('precision', loss.precision(p_val, yval), end=', ')
    print('false_positive_rate', loss.false_positive_rate(p_val, yval), '\n')

    # print loss
    print('multivariate:')
    p_val_multy = multivariate_predict(Xval, mu_multy, sigma_multy) <= eps
    print('accuracy', loss.accuracy(p_val_multy, yval), end=', ')
    print('F_score', loss.F_score(p_val_multy, yval), end=', ')
    print('recall', loss.recall(p_val_multy, yval), end=', ')
    print('precision', loss.precision(p_val_multy, yval), end=', ')
    print('false_positive_rate', loss.false_positive_rate(p_val_multy, yval))
