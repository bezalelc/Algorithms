import numpy as np
from model import Model
import model
from regularization import Regularization, L1, L2
# import numba as nb
# from numba import njit, jit, prange
# from numba.experimental import jitclass
# from numba.np.ufunc import parallel
from timeit import default_timer
from scipy.stats import mode


class KNearestNeighbor(Model):

    def __init__(self) -> None:
        super().__init__()

    def train(self, X: np.ndarray, y: np.ndarray):
        super().train(X, y)

    def predict(self, X_test, k=1, reg_func: Regularization = L2) -> np.ndarray:
        """
        k nearest neighbors algorithm:
            predict x according to the closest distance values
            this function is for classification predict


        :param X_test: data to predict
        :param k: range
        :param reg_func:  function for regularization

        :return: prediction

        :efficiency: O(m*n*test_size)
        """
        n, n_test = self.X.shape[0], X_test.shape[0]
        distances = np.empty((n_test, n))
        # for i in range(n_test):
        #     for j in range(n):
        # distances[i, j] = reg_func(X_test[i] - self.X[j])
        for i in range(n_test):
            distances[i, :] = reg_func(X_test[i] - self.X)
            distances[i, :] = np.sum((X_test[i] - self.X) ** 2, axis=1) ** 0.5

        # distances = reg_func(X_test[:, np.newaxis] - self.X)
        idx = np.argpartition(distances, k, axis=1)[:, :k].reshape((-1, k))
        neighbor = self.y[idx].reshape((-1, k))

        # pred = np.empty((n_test,), dtype=np.int64)
        # pred = np.array([np.bincount(neighbor[i]).argmax() for i in range(n_test)], dtype=np.int64)
        pred = mode(neighbor, axis=1)[0]
        # for i in range(n_test):
        #     pred[i] = np.bincount(neighbor[i]).argmax()

        return pred


if __name__ == '__main__':
    import os
    import metrics
    import Algorithms.ML_.data as data
    from Algorithms.ML_.data.load_data import data_paths, CIFAR_10, CIFAR_10_batch
    import Algorithms.ML_.helper.plot_ as plot_

    # *********************    load the dataset   ***********************
    # get train size 5000 and test size 500
    train_size, test_size = 100, 20
    # train_size, test_size = 5000, 500
    classes = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    path = data_paths['CIFAR10']
    # # Xtr, Ytr, Xte, Yte = CIFAR_10()
    X, y = CIFAR_10_batch(os.path.join(path, 'data_batch_1'))
    Xte, Yte = CIFAR_10_batch(os.path.join(path, 'test_batch'))

    # *********************    divide to X&y   ***********************
    # plot_.img_show_data(X_[0:9, ...].astype('uint8'), classes[Y[0:9].astype(int)])
    X, y = X[:train_size].reshape((train_size, -1)), y[:train_size].reshape((train_size, -1))
    Xte, Yte = Xte[:test_size].reshape((test_size, -1)), Yte[:test_size].reshape((test_size, -1))
    # print(X.shape, y.shape, Xte.shape, Yte.shape)

    # *********************    train   ***********************
    model = model.KNearestNeighbor()
    model.train(X, y)
    # start = default_timer()
    pred = model.predict(Xte, k=60)
    # end = default_timer()
    # print('time=', end - start)
    # *********************    predict   ***********************
    print(metrics.accuracy(Yte, pred))
    print(np.mean(metrics.null_accuracy(Yte)))
