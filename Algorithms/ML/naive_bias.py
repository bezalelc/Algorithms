import numpy as np


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


if __name__ == '__main__':
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
    '''
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
'''
