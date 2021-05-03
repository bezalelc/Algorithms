"""
Author: Bezalel Cohen
"""
import numpy as np

from general import load_data


def gaussian_one_feature(X, y):
    """
    compute the mean and correlations matrix for each class in X

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real class for each example (=row) in X

    :return:
        X_groups: list of X
        u: vector of mean(X) for each feature
        sigma: matrix 3D with correlations matrix for each feature

    :efficiency: O(k*m + m^2*n^2)
    """
    X = X.reshape((-1, 1))
    m, n = X.shape
    X_groups, y_groups = [], []
    classes = np.unique(y)
    u, sigma = np.zeros((len(classes), 1)), np.zeros((len(classes), 1))

    for clas, i in zip(classes, range(len(classes))):
        idx = np.where(y == clas)[0]
        X_groups.append(X[idx]), y_groups.append(y[idx])  # .reshape((-1,1)
        u[i], sigma[i] = np.mean(X_groups[-1]), np.std(X_groups[-1])
    x = X_groups[0][1]  # , y_groups[0][0]
    print(X_groups[0].shape, u.shape, sigma.shape)
    pr = predict_one_feature(X_groups[0], u, sigma)
    print(np.argmax(pr, axis=1))
    pr = predict_one_feature(X_groups[1], u, sigma)
    print(pr)
    print(np.argmax(pr, axis=1))

    # print(p(x, u, sigma))
    # X_groups[0]=X_groups[0].reshape((-1,1))
    # print((X_groups[0] - u))
    # print((X_groups[0] - u[:]).reshape((X_groups[0].shape[0],u.shape[0])))


def predict_one_feature(X, u, sigma):
    """
    predict the class for X

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        u: vector of mean(X) for each feature
        sigma: matrix 3D with correlations matrix for each feature

    :return:
        X_groups: list of X
        u: vector of mean(X) for each feature
        sigma: matrix 3D with correlations matrix for each feature

    :efficiency: O(k*(m*n+n^3+m*n^2+m*n^2)) ~ O(k*m*n^2)
    """
    u_, sigma_ = np.zeros((X.shape[0], u.shape[0])) + u.reshape(-1), np.zeros(
        (X.shape[0], sigma.shape[0])) + sigma.reshape(-1)
    X = X.reshape((X.shape[0], -1))
    return (1 / (np.sqrt(2 * np.pi) * sigma_)) * np.exp(- 0.5 * ((X - u_) / sigma_) ** 2)


def gaussian_bayes(X, y):
    """
    compute the mean and correlations matrix for each class in X

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real class for each example (=row) in X

    :return:
        X_groups: list of X
        u: vector of mean(X) for each feature
        sigma: matrix 3D with correlations matrix for each feature

    :efficiency: O(k*m + m^2*n^2)
    """
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    classes = np.unique(y)
    m, n, k = X.shape[0], X.shape[1], len(classes)
    X_groups, y_groups = [], []
    u, sigma = np.zeros((k, n)), np.zeros((k, n, n))

    for clas, i in zip(classes, range(len(classes))):
        idx = np.where(y == clas)[0]
        X_groups.append(X[idx]), y_groups.append(y[idx])  # .reshape((-1,1)
        u[i] = np.mean(X_groups[-1], axis=0)
        sigma[i] = ((X_groups[-1] - u[i]).T @ (X_groups[-1] - u[i])) / m

    return X_groups, u, sigma


def predict_gaussian_bayes(X, u, sigma):
    """
    predict the class for X

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        u: vector of mean(X) for each feature
        sigma: matrix 3D with correlations matrix for each feature

    :return:
        X_groups: list of X
        u: vector of mean(X) for each feature
        sigma: matrix 3D with correlations matrix for each feature

    :efficiency: O(k*(m*n+n^3+m*n^2+m*n^2)) ~ O(k*m*n^2)
    """
    m, n, k = X.shape[0] if len(X.shape) > 1 else 1, X.shape[0] if len(X.shape) == 1 else X.shape[1], u.shape[0]
    # if n == 1:
    #     u, sigma = np.zeros((m, n)) + u.reshape(-1), np.zeros((m, n)) + sigma.reshape(-1)
    X = X.reshape((m, -1))
    M = np.zeros((m, k))
    for i in range(k):
        M[:, i] = np.exp(-0.5 * np.sum(((X - u[i]) @ np.linalg.pinv(sigma[i])) * (X - u[i]), axis=1))
        M[:, i] /= (((2 * np.pi) ** (n / 2)) * np.sqrt(np.linalg.det(sigma[i])))  # (1/(2*n))
    return M


if __name__ == '__main__':
    print('\n\n===================================== test ex1data2 =====================================')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex2data1.txt')
    # np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    # f(X[:, 0], y)
    X_group, u, sigma = gaussian_bayes(X, y)
    # print((X_group[0]).shape)
    p = predict_gaussian_bayes(X_group[0][0:6], u, sigma)
    # print(p)
    print(np.mean(np.argmax(p, axis=1) == y))
