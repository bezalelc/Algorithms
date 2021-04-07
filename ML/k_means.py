# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np


def find_mean(X, k, max_iter=100):
    """
    K-means algorithm:

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        k: The desired number of dimensions
        max_iter: max of iteration -> each iteration ini new random mean and try to find better mean

    :return: u: (k x n) vector with mean of n feature for each k

    efficiency: O(max_iter * (?) *k*m*n) while  (?) number of internal loops until we get the best mean
    """
    # useful vars
    m, n = X.shape[0], X.shape[1]
    u_res, w, y = np.zeros((k, n)), np.zeros((m, k)), np.zeros((m,))

    for i in range(max_iter):
        # init random k points to be the means
        idx = np.random.randint(m, size=k)
        u = X[idx, :]
        u_prev = u.copy()

        while True:
            # find class for each x
            for i in range(k):
                w[:, i] = np.sum((X - u[i]) ** 2, axis=1)
            y = np.argmin(w, axis=1)
            # plt.figure(figsize=(8, 6))
            # c = ['r', 'g', 'b']

            # update u
            for i in range(k):
                idx = np.argwhere(y == i).reshape((-1,))
                if idx.shape[0] == 0:
                    continue
                u[i] = np.mean(X[idx], axis=0)
                # plt.scatter(X[idx, 0], X[idx, 1], marker='*', linewidths=1, color=c[i])
                # plt.scatter(u[i, 0], u[i, 1], marker='+', linewidths=5, color=c[i])
            # print(J(X,u))
            # plt.show()

            if np.array_equal(u, u_prev):
                break
            else:
                u_prev = u

        u_res = u_prev if J(X, u_prev) < J(X, u_res) else u_res

    return u_res


def J(X, u):
    """
    compute yhe cost for K-means algorithm:

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        u: (k x n) vector with mean of n feature for each k

    :return: j: (k x n) ->  <float> : the sum of square of distances between specific mean
                                      and is closest points in X

    efficiency: O(k*m*n) * O(max_iter * (?) *k*m*n) while  (?) number of internal loops until we get the best mean
    """
    m, n, k, j, y = X.shape[0], X.shape[1], u.shape[0], 0, predict(X, u)
    for i in range(k):
        idx = np.argwhere(y == i).reshape((-1,))
        j += np.sum((X[idx, :] - u[i]) ** 2)
    return j


def predict(X, u):
    """
    compute for each point in X is class according to u

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        u: (k x n) vector with mean of n feature for each k

    :return: vector that contain the classes of X [class(X[0]),...,class(X[m])]

    efficiency: O(k*m*n)
    """
    if len(X.shape) == 1:
        X = X.reshape((1, -1))

    m, k, j = X.shape[0], u.shape[0], 0
    w = np.zeros((m, k))
    for i in range(k):
        w[:, i] = np.abs(np.sum((X - u[i]) ** 2, axis=1))

    return np.argmin(w, axis=1)


def best_mean(X, k_max=3):
    """
    compute for all k in range 2,...,k_max the best mean and plot the cost for each k
    this method can help to choose best k

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        k_max: max k to check

    efficiency: O(k_mak)*
    """
    plt.figure(figsize=(8, 6))
    j = []
    for k in range(2, k_max):
        j.append(J(X, find_mean(X, k, max_iter=3)))
    plt.plot(range(len(j)), j)
    plt.show()


if __name__ == '__main__':
    from ML.general import load_data
    from PIL import Image

    # from PIL import Image
    # from ML.general import load_data

    print('\n\n===================================== test ex1data2 =====================================')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex2data1.txt')
    X, y = data[:, :-1], data[:, -1:]
    u = find_mean(X, 2)
    p = predict(X, u)
    # print(np.mean(p == y))
    # best_mean(X,k_max=15)

    print('\n\n===================================== test ex1data2 =====================================')
    import scipy.io

    X = scipy.io.loadmat('/home/bb/Documents/octave/week8/machine-learning-ex7/ex7/ex7data2.mat')['X']
    u = find_mean(X, 3)
    # print(u)
    # print('---------------------------------  test best_mean()  --------------------------')
    # best_mean(X, k_max=14)

    print('---------------------------------  Reducing Image size  --------------------------')
    import imageio

    img = imageio.imread('/home/bb/Downloads/IMG-20180617-WA0018 (copy).jpg')
    X = np.array(img, dtype=np.float128)
    origin, D = np.array(X.copy(), dtype=np.uint8), X.shape
    print(D)
    X = X.reshape((X.shape[0] * X.shape[1], -1))
    X = X / 255

    # best_mean(X, 30)
    u = find_mean(X, 16, max_iter=10)
    p = predict(X, u)
    u = np.round(u * 255)
    img = np.array(np.reshape(u[p, :], D), dtype=np.uint8)


    img = Image.fromarray(img, 'RGB')
    img.save('/home/bb/Downloads/my.png')
    # img.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(origin)
    plt.title('origin image')
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('compressed image')
    plt.show()
