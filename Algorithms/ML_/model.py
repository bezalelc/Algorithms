import numpy as np
from regularization import Regularization, L1, L2, L12
from activation import softmax, Activation, linear, logistic, relu, leaky_relu, tanh, sigmoid
from loss import Loss, hinge, cross_entropy
from optimizer import Optimizer
from gradient import Grad


class Model:
    def __init__(self) -> None:
        super().__init__()
        # general param
        self.m: int = 0
        self.n: int = 0
        self.k: int = 0
        # hyper param
        self.reg: Regularization = None
        self.alpha: int = 1
        self.lambda_: int = 0
        # model param
        self.X: np.ndarray = np.empty((1,))
        self.y: np.ndarray = np.empty((1,))
        self.W: np.ndarray = np.empty((1,))
        # engine param
        self.act: Activation = None
        self.loss_: Loss = None

    def compile(self, alpha=1, lambda_=0, reg=L2, activation=None, loss_=None) -> None:
        """
        restart hyper params
        """
        self.alpha, self.lambda_ = alpha, lambda_
        self.reg, self.act, self.loss_ = reg, activation, loss_

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
        distances = reg_func(X_test[:, np.newaxis] - self.X)
        idx = np.argpartition(distances, k, axis=1)[:, :k].reshape((-1, k))
        neighbor = self.y[idx].reshape((-1, k))

        from scipy.stats import mode
        pred = mode(neighbor, axis=1)[0]
        return pred


class SVM(Model):

    def __init__(self) -> None:
        super().__init__()
        # hyper param
        self.c: int = 0
        self.gamma: int = 0

    def compile(self, alpha=1, lambda_=0, reg=L2, activation=linear, loss_=hinge, c=1, gamma=0) -> None:
        super().compile(alpha, lambda_, reg, activation, loss_)
        self.c, self.gamma = c, gamma

    def train(self, X: np.ndarray, y: np.ndarray, eps=1e-4):
        super().train(np.hstack((np.ones((X.shape[0], 1)), X)), y)
        # unpacked param
        m, n, k = self.m, self.n, self.k

        self.W = np.random.randn(n, k) * eps

    def predict(self, X) -> np.ndarray:
        # unpacked param
        W, act = self.W, self.act

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return act(X @ W)

    def loss(self, X, y) -> float:
        # unpacked param
        reg, loss_, lambda_, W = self.reg, self.loss_, self.lambda_, self.W

        L = loss_(self.predict(X), y) + np.sum(lambda_ * reg(W))
        return float(L)
        # L = 0
        # for i in range(m):
        #     S = self.predict(X[i])
        #     for j in range(n):
        #         if j != y[i]:
        #             if S[y[i]] >= S[j] + 1:  # 1=> safety margin
        #                 L += 0
        #             else:
        #                 L += S[j] - S[y[i]] + 1
        #
        #             # L += max(0, S[j] - S[y[i]] + 1)
        # return float(np.sum(-np.log10(softmax(self.predict(X)))[np.arange(m), y]))


class GD(Model):
    pass


model = SVM()
X = np.array([-15, 22, -44, 56.]).reshape((1, -1))
y = np.array([2, ])
W_ = np.array([[0, 0.2, -0.3], [0.01, 0.7, 0], [-0.05, 0.2, -0.45], [0.1, 0.05, -0.2], [0.05, 0.16, 0.03]])
model.compile(activation=softmax, loss_=cross_entropy)
# model.compile(activation=linear, loss_=hinge)
model.train(X, y)
model.W = W_
# print(model.predict(X))

print(model.loss(X, y))
print(-np.log10(0.35338733))
