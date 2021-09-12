import numpy as np
import abc
from typing import Union
from regularization import Regularization, L1, L2, L12
# from activation import softmax, Activation, linear, logistic, relu, leaky_relu, tanh, sigmoid
# from activation import Loss, hinge, cross_entropy
# from activation import dLoss
from activation import Activation, Hinge
from optimizer import Optimizer


class Model():
    def __init__(self) -> None:
        super().__init__()
        # general param
        self.m: int = 0
        self.n: int = 0
        self.k: int = 0
        # hyper param
        self.reg: Regularization() = None  # Regularization function
        # self.dReg: dRegularization = None  # derivative for regularization function
        # model param
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def compile(self, reg: Regularization = None) -> None:
        """
        restart hyper params
        """
        self.reg = reg
        # self.reg, self.d_reg, self.activation, self.loss_ = reg, d_reg, activation, loss_

    def train(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self.m, self.n = X.shape[0:2]
        self.k = np.max(y) + 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def loss(self, X: np.ndarray, y: np.ndarray):
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

    def predict(self, X_test: np.ndarray, k: int = 1, reg_func: Regularization = L2()) -> np.ndarray:
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
        distances = reg_func.norm(X_test[:, np.newaxis] - self.X)
        idx = np.argpartition(distances, k, axis=1)[:, :k].reshape((-1, k))
        neighbor = self.y[idx].reshape((-1, k))

        from scipy.stats import mode
        pred = mode(neighbor, axis=1)[0]
        return pred


class Regression(Model):

    def __init__(self) -> None:
        super().__init__()
        # hyper param
        self.reg: Regularization() = None  # Regularization function
        # self.dReg: dRegularization = None  # derivative for regularization function
        self.alpha: float = 1
        self.lambda_: float = 0
        # model param
        self.W: np.ndarray = np.array([])  # np.empty((1,))
        # engine param
        self.activation: Activation() = None  # Activation function
        # self.loss_: Loss = None  # Loss function for the activation
        # self.d_loss: dLoss = None  # derivative for Loss function

    def compile(self, alpha=0.001, lambda_=0, activation: Activation = None, reg: Regularization = None) -> None:
        super().compile(reg)


class SVM(Regression):

    def __init__(self) -> None:
        super().__init__()
        # hyper param
        self.c: int = 0
        self.gamma: int = 0

    def compile(self, alpha=0.001, lambda_=0, activation: Activation = Hinge(), reg: Regularization = L2(), c=1,
                gamma=0) -> None:
        super().compile(alpha, lambda_, activation, reg)
        self.c, self.gamma = c, gamma

    def train(self, X: np.ndarray, y: np.ndarray, iter_=1500, batch=32, eps=0.001, verbose=True) -> list[float]:
        super().train(np.hstack((np.ones((X.shape[0], 1)), X)), y)
        # unpacked param
        m, n, k, X, alpha = self.m, self.n, self.k, self.X, self.alpha

        if len(self.W) == 0:
            np.random.seed(1)  # -----
            self.W = np.random.randn(n, k) * eps
        W = self.W

        loss_history = []

        for i in range(iter_):
            batch_idx = np.random.choice(m, batch, replace=False)
            X_, y_ = X[batch_idx], y[batch_idx]

            if verbose:
                L, dW = self.grad(X_, y_, loss_=True)
                loss_history.append(L)
            else:
                dW = self.grad(X_, y_, loss_=False)

            W -= alpha * dW

        return loss_history

    def predict(self, X, add_ones=True) -> np.ndarray:
        W, Act = self.W, self.activation.activation

        if add_ones:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        return Act(X, W)

    def loss(self, X, y, add_ones=True) -> float:
        m, Reg, lambda_, W, L = X.shape[0], self.reg.norm, self.lambda_, self.W, self.activation.loss
        return L(X, W, y) + lambda_ * Reg(W)  # np.sum(W * W)

    def grad(self, X, y, loss_=False) -> Union[np.ndarray, tuple[float, np.ndarray]]:
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

    @staticmethod
    def best_alpha_lambda(X, y, Xv, Yv, alphas, lambdas, verbose=True):
        """
        choose the best hyper params alpha and lambda for svm

        :param X: train data
        :param y: classes for train data
        :param Xv: val data
        :param Yv: classes for val data
        :param verbose: print results
        :param alphas:
        :param lambdas:


        :return: best hyper params alpha and lambda
        """
        results = {}
        best_val = -1

        grid_search = [(lr, rg) for lr in alphas for rg in lambdas]
        for lr, rg in grid_search:
            # Create a new SVM instance
            model = SVM()
            model.compile(alpha=lr, lambda_=rg)
            train_loss = model.train(X, y, batch=200)

            # Predict values for training set
            pred = np.argmax(model.predict(model.X, add_ones=False), axis=1)
            train_accuracy = float(np.mean(pred == model.y))
            pred_v = np.argmax(model.predict(Xv), axis=1)
            val_accuracy = float(np.mean(pred_v == Yv))

            # Save results
            results[(lr, rg)] = (train_accuracy, val_accuracy)
            if best_val < val_accuracy:
                best_val = val_accuracy

        if verbose:
            for lr, reg in sorted(results):
                train_accuracy, val_accuracy = results[(lr, reg)]
                print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))

        return max(results, key=lambda x: results[x])


class GD(Model):
    pass


class NN:
    pass

# model = SVM()
# X = np.array([-15, 22, -44, 56.]).reshape((1, -1))
# y = np.array([2, ])
# W_ = np.array([[0, 0.2, -0.3], [0.01, 0.7, 0], [-0.05, 0.2, -0.45], [0.1, 0.05, -0.2], [0.05, 0.16, 0.03]])
# model.compile(activation=softmax, loss_=cross_entropy)
# # model.compile(activation=linear, loss_=hinge)
# model.train(X, y)
# model.W = W_
# # print(model.predict(X))
#
# print(model.loss(X, y))
# print(-np.log10(0.35338733))
# import svm
