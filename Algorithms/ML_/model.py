# import pickle
# from model import Model
from lasso import Regularization, L1, L2
# import numba as nb
# from numba import njit, jit, prange
# from numba.experimental import jitclass
# from numba.np.ufunc import parallel
from timeit import default_timer
from scipy.stats import mode
import numpy as np


class Model:
    def __init__(self) -> None:
        super().__init__()
        self.X, self.y = None, None

    def compile(self):
        pass

    def train(self, X, y):
        self.X, self.y = X, y

    def predict(self, X):
        pass

    def cost(self):
        pass

    def split(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


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
