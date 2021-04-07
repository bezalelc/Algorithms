import numpy as np

from general import load_data


def naive_bayes(X, y):
    """
    compute the mean and correlations matrix for each class in X

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real class for each example (=row) in X

    :return:
        P: matrix of P_r(x=1|y=k)
        percent_k: Percent of each class

    :efficiency: O(k*n*m)
    """
    classes = np.unique(y)
    m, n, k = X.shape[0], X.shape[1], len(classes)
    P, percent_k = np.zeros((k, n)), np.zeros((k,))
    for i in range(k):
        idx = np.where(y == classes[i])[0]
        percent_k[i] = idx.shape[0] / m
        for j in range(n):
            P[i, j] = (np.where(X[idx, j] == 1)[0].shape[0] + 1) / (idx.shape[0] + k)  # 1,k
    return P, percent_k


def predict_naive_bayes(X, P, percent_k):
    """
    predict the class for X

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        P: matrix of P_r(x=1|y=k)
        percent_k: Percent of each class

    :return:
        pr: predicted class for all x_i in X

    :efficiency: O(m*n*k)
    """
    if len(X.shape) == 1:
        X = X.reshape((1, -1))
    m, k, n = X.shape[0], P.shape[0], P.shape[1]  # X.shape[0] if len(X.shape) > 1 else 1

    pr = np.zeros((m, 1))
    for i in range(m):
        x = X[i, :]
        p = P.copy()
        z = np.where(x == 0)[0]
        p[:, z] = 1 - p[:, z]
        pr[i] = np.argmax(np.prod(p, axis=1) * percent_k, axis=0)

    return pr


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
    # print(np.argmax(p, axis=1))
    print('\n\n===================================== test naive bayes =====================================')
    data = np.zeros((1000, 4))
    data[:500, -1], data[500:800, -1], data[800:, -1] = 0, 1, 2
    data[:400, 0], data[:350, 1], data[:450, 2] = 1, 1, 1
    data[500:650, 1], data[500:800, 2] = 1, 1
    data[800:900, 0], data[800:950, 1], data[800:850, 2] = 1, 1, 1
    X, y = data[:, :-1], data[:, -1]
    # print(data.tolist())
    P, sum_k = naive_bayes(X, y)
    pr = predict_naive_bayes(np.array([0, 1, 1]), P, sum_k)
    print(pr)
    pr = predict_naive_bayes(X, P, sum_k)
    # print(pr)
    # print(np.array([pr[900:].reshape((-1,)), y[900:]]))
    print(np.mean(pr.reshape((-1,)) == y))
