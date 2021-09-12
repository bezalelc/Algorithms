import numpy as np
from numpy import random
from typing import Union
from activation import Activation
from regularization import Regularization


class Layer:
    def __init__(self) -> None:
        super().__init__()


class Dense(Layer):

    def __init__(self, n, m, activation, reg, eps=1e-3, alpha=1e-5, lambda_=0) -> None:
        super().__init__()
        # layer params
        self.W: np.ndarray = random.randn(n, m) * eps
        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_
        # engine param
        self.activation: Activation() = activation
        self.reg: Regularization() = reg

    def predict(self, X) -> np.ndarray:
        """
        :param X: (m x n) matrix of data

        :return: (m x k) matrix of scores for each pairs of (example & class)

        :complexity: O(m*n*k) -> multiple 2 matrix + O(activation) + O(regularization)
        """
        W, Act = self.W, self.activation.activation
        return Act(X, W)

    def loss(self, X, y) -> float:
        """
        loss of the prediction

        :param X:  (m x n) matrix of data
        :param y: true labels for each example

        :return: total loss according to activation function

        :complexity: O(m*n*k) -> multiple 2 matrix + O(activation) + O(regularization) + O(loss)
        """
        m, Reg, lambda_, W, L = X.shape[0], self.reg.norm, self.lambda_, self.W, self.activation.loss
        return L(X, W, y) + lambda_ * Reg(W)  # np.sum(W * W)

    def grad(self, X, y, loss_=False) -> Union[np.ndarray, tuple[float, np.ndarray]]:
        """
        calculate the gradient for the activation function

        :param X:  (m x n) matrix of data
        :param y: true labels for each example
        :param loss_: if true calculate the loss also

        :return: if loss_=False: loss,dW
                 else: dW

        :complexity: O(m*n*k*2) -> multiple 2 matrix * 2 + O(activation) + O(regularization) + O(loss)
        """
        W, lambda_, Reg, dReg = self.W, self.lambda_, self.reg.norm, self.reg.d_norm
        Grad = self.activation.loss_grad if loss_ else self.activation.grad
        # Loss, Grad = self.activation.loss, self.activation.grad

        if loss_:
            # loss of all X
            # L, dW = Loss(self.X, W, self.y), Grad(X, W, y)
            # loss of only batch of X
            L, dW = Grad(X, W, y)
            L, dW = L + lambda_ * Reg(W), dW + lambda_ * dReg(W)  # np.sum(W * W),2*W
            return L, dW
        else:
            return Grad(X, W, y) + lambda_ * dReg(W)  # 2 * W
