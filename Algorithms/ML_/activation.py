import numpy as np
import abc


class Activation(metaclass=abc.ABCMeta):
    """
    interface for activation classes
    """

    @staticmethod
    @abc.abstractmethod
    def activation(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_grad_loop(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        pass


class Hinge(Activation):
    """
    hinge activation for svm
    """

    @staticmethod
    def activation(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return X @ W

    @staticmethod
    def grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        P = Hinge.activation(X, W)
        P = P - P[np.arange(m), y].reshape((-1, 1)) + 1
        P[P < 0], P[np.arange(m), y] = 0, 0
        P[P > 0] = 1
        P[np.arange(m), y] = - P.sum(axis=1)
        dW = X.T @ P / m
        return dW

    @staticmethod
    def loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[0]
        P = Hinge.activation(X, W)
        return np.sum(np.maximum(0, P - P[np.arange(m), y].reshape((-1, 1)) + 1)) / m - 1

    @staticmethod
    def loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        m = y.shape[0]
        P = Hinge.activation(X, W)
        P = P - P[np.arange(m), y].reshape((-1, 1)) + 1

        # loss
        P[P < 0], P[np.arange(m), y] = 0, 0
        L = np.sum(P) / m

        # grad
        P[P > 0] = 1
        P[np.arange(m), y] = - P.sum(axis=1)
        dW = X.T @ P / m

        return L, dW

    @staticmethod
    def loss_grad_loop(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        m, n, k = X.shape[0], X.shape[1], W.shape[1]

        P = Hinge.activation(X, W)
        P = P - P[np.arange(m), y].reshape((-1, 1)) + 1
        P[P < 0], P[np.arange(m), y] = 0, 0

        L, dW = 0, np.zeros(W.shape)
        for i in range(m):
            for j in range(k):
                if j != y[i] and P[i, j] > 0:
                    dW[:, y[i]] -= X[i]
                    dW[:, j] += X[i]
                    L += P[i, j]

        L, dW = L / m, dW / m
        return L, dW


class Softmax(Activation):

    @staticmethod
    def activation(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        P = X @ W
        P -= np.max(P, axis=1)[..., None]
        P = np.exp(P)
        P /= np.sum(P, axis=1)[..., None]
        return P

    @staticmethod
    def grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = y.shape[0]
        P = Softmax.activation(X, W)
        P[np.arange(m), y] -= 1
        dW = X.T @ P / m

        return dW

    @staticmethod
    def loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[0]
        P = Softmax.activation(X, W)
        L = np.sum(-np.log(P[np.arange(m), y])) / m
        return float(L)

    @staticmethod
    def loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        m = y.shape[0]
        P = Softmax.activation(X, W)

        # loss
        L = np.sum(-np.log(P[np.arange(m), y])) / m

        # grad
        P[np.arange(m), y] -= 1
        dW = X.T @ P / m

        return L, dW

    @staticmethod
    def loss_grad_loop(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """
        calculate the loss and gradient iin loop

        :param X:
        :param W:
        :param y:

        :return:
        """
        m, n, k = X.shape[0], X.shape[1], W.shape[1]

        P = Softmax.activation(X, W)

        L, dW = 0, np.zeros(W.shape)
        for i in range(m):
            L += -np.log(P[i, y[i]])
            dW[:, y[i]] -= X[i]

            for j in range(k):
                dW[:, j] += X[i] * P[i, j]

        L, dW = L / m, dW / m
        return L, dW

# from typing import NewType, Callable
# **********************   Activation   **********************************
# hidden layers
# Activation = NewType('Activation', Callable[[np.ndarray, int], np.ndarray])
# sigmoid = Activation(lambda X, axis=-1: 1. / (1. + np.exp(-X)))
# linear = Activation(lambda X, axis=-1: X)  # Regression
# relu = Activation(lambda X, axis=-1: np.maximum(0, X))
# leaky_relu = Activation(lambda X, axis=-1: np.where(X > 0, X, X * 0.01))
# tanh = Activation(lambda X, axis=-1: np.tanh(X))
# # # output layer
# softmax = Activation(lambda X, axis=-1: np.exp(X) / np.sum(np.exp(X), axis=axis)[..., None])
# logistic = Activation(lambda X, axis=-1: 1. / (1. + np.exp(-X)))
# # softmax_ = Activation(lambda X: x=,)  # “softer” version of argmax
#
# # **********************   Loss   **********************************
# Loss = NewType('Loss', Callable[[np.ndarray, np.ndarray], np.ndarray])
# hinge = Loss(lambda X, y: np.sum(np.maximum(0, X - X[np.arange(y.shape[0]), y].reshape((-1, 1)) + 1)) / y.shape[0] - 1)
# cross_entropy = Loss(lambda X, y: np.sum(-np.log(X)[np.arange(y.shape[0]), y]) / y.shape[0])
# # mse = Loss(lambda Z: np.sum() / Z.shape[1])
# # binary_cross_entropy = Loss()

# **********************   derivative   **********************************
# dLoss = NewType('dLoss', Callable[[np.ndarray, np.ndarray], np.ndarray])
# grad_1 = dLoss(lambda X, f, h: (f(X + h) - f(X)) / h)
# grad_2 = dLoss(lambda X, f, h: (f(X + h) - f(X - h)) / (2 * h))

# d_hinge = dLoss(lambda X )
