import matplotlib.pyplot as plt
import numpy as np
import abc
from typing import Union
from regularization import Regularization, L1, L2, L12
from activation import Activation_, Hinge_, Softmax_, Activation, Softmax
from optimizer import Optimizer, Vanilla
from layer import Layer, Dense, Dense1
import metrics


class Model:
    def __init__(self) -> None:
        super().__init__()
        # general param
        self.m: int = 0
        self.n: int = 0
        self.k: int = 0
        # hyper param
        self.treshold = 0.5
        # self.reg: Regularization() = None  # Regularization function
        # self.dReg: dRegularization = None  # derivative for regularization function
        # model param
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def compile(self) -> None:
        """
        restart hyper params
        """
        pass
        # self.reg = reg
        # self.reg, self.d_reg, self.activation, self.loss_ = reg, d_reg, activation, loss_

    def train(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self.m = X.shape[0]

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

    def describe(self):
        pass

    def k_fold(self, X, y, k: int = 5):
        """
        split dataset to k folders and train k times while in each loop the fold[i] if the validation
            data, this method is useful for a small dataset

        :param X:
        :param y:
        :param k:
        :return:
        """
        X, y = np.array_split(X, k), np.array_split(y, k)

        for i in range(k):
            X_, y_ = np.vstack([X[j] for j in range(k) if j != i]), np.hstack([y[j] for j in range(k) if j != i])
            self.train(X_, y_)


class KNearestNeighbor(Model):

    def __init__(self, reg: Regularization) -> None:
        super().__init__()
        self.reg = reg

    def train(self, X: np.ndarray, y: np.ndarray):
        super().train(X, y)
        self.m, self.n = X.shape[:2]

    def predict(self, X_test: np.ndarray, k: int = 1) -> np.ndarray:
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

        n, n_test, Reg = self.X.shape[0], X_test.shape[0], self.reg.norm
        distances = np.empty((n_test, n))
        for i in range(n_test):
            for j in range(n):
                distances[i, j] = Reg(X_test[i] - self.X[j]) ** 0.5
        # for i in range(n_test):
        # distances[i, :] = np.sum(self.reg.norm(X_test[i] - self.X)) ** 0.5
        # distances[i, :] = np.sum((X_test[i] - self.X) ** 2, axis=1) ** 0.5

        # distances = np.sum(Reg(X_test[:, np.newaxis] - self.X), axis=2) ** 0.5
        idx = np.argpartition(distances, k, axis=1)[:, :k].reshape((-1, k))
        neighbor = self.y[idx].reshape((-1, k))

        from scipy.stats import mode
        pred = mode(neighbor, axis=1)[0]
        return pred


class Regression(Model):

    def __init__(self, layers: list[Layer] = None, classes: list = None) -> None:
        super().__init__()

        # model param
        self.layers: list[Layer] = layers
        # self.layers_size: list = []

        # general params
        self.classes: list = classes

    def compile(self) -> None:
        layers = self.layers
        assert layers is not None and len(layers) > 0

        self.n, self.k = layers[0].input_shape[0], layers[-1].out_shape
        for i, layer in enumerate(layers):
            # self.layers_size.append((layer.input_shape, layer.out_shape))

            if i > 0 and (isinstance(layer, Dense) or isinstance(layer, Dense1)):
                layer.input_shape = (layers[i - 1].out_shape,)

    def train(self, X: np.ndarray, y: np.ndarray, val: tuple = None, iter_=1500, batch=32, return_loss=True,
              verbose=True) -> tuple[list, list]:
        super().train(X, y)  # np.hstack((np.ones((X.shape[0], 1)), X))
        # unpacked param
        m, n, k, X, y, layers = self.m, self.n, self.k, self.X, self.y, self.layers
        batch, epoch_size = min(batch, m), max(m / batch, 1)

        loss_history_t, loss_history_v = [], []
        for i in range(iter_):
            batch_idx = np.random.choice(m, batch, replace=False)
            X_, y_ = X[batch_idx], y[batch_idx]

            H = self.feedforward(X_)
            self.backpropagation(H, y_)

            if return_loss:
                loss_history_t.append(self.loss(X, y))
                if val:
                    loss_history_v.append(self.loss(*val))
            if verbose and i % epoch_size == 0:
                s = 'iteration %d / %d: loss %f' % (i, m, loss_history_t[-1])
                if val:
                    s += ' val loss %f' % (loss_history_v[-1])
                print(s)

        return loss_history_t, loss_history_v

    def feedforward(self, X: np.ndarray) -> list[np.ndarray]:
        layers = self.layers

        H = [X]
        for layer in layers:
            H.append(layer.forward(H[-1]))
        return H

    def backpropagation(self, H, y):
        # k, layers = self.k, self.layers
        # K = np.arange(k)
        # delta = H[-1] - np.array(y[:, None] == K)

        m, layers = H[0].shape[0], self.layers

        delta = self.layers[-1].act.delta(y, H[-1])
        for layer, h, i in zip(layers[::-1], H[:-1][::-1], range(len(layers))[::-1]):
            delta = layer.backward(delta, h, return_delta=bool(i))

    def grad(self, H: list[np.ndarray], y: np.ndarray) -> list[np.ndarray]:
        m, layers = H[0].shape[0], self.layers
        # K = np.arange(k)
        # delta = H[-1] - np.array(y[:, None] == K)

        delta = self.layers[-1].act.delta(y, H[-1])

        dW = []
        for layer, h, i in zip(self.layers[1:][::-1], H[1:-1][::-1], range(len(layers) - 1)[::-1]):
            w = layer.W
            # print(i, len(layers) - 1)
            # if i == len(layers) - 2 or not layers[i + 1].add_bias:
            #     dW.insert(0, h.T @ delta)
            #
            #
            # else:
            #     dW.insert(0, np.hstack((np.ones((h.shape[0], 1)), h)).T @ delta)
            dW.insert(0, np.hstack((np.ones((h.shape[0], 1)), h)).T @ delta)
            dW[0][1:, :] += layer.lambda_ * w[1:, :]
            dW[0] /= m
            delta = delta @ w[1:, :].T * h * (1 - h)

        dW.insert(0, np.hstack((np.ones((H[0].shape[0], 1)), H[0])).T @ delta)
        dW[0][1:, :] += layers[0].lambda_ * layers[0].W[1:, :]
        dW[0] /= m

        return dW

    def loss(self, X: np.ndarray, y: np.ndarray, pred: np.ndarray = None) -> float:
        if pred is None:
            pred = self.feedforward(X)[-1]
        m, Loss, layers = pred.shape[0], self.layers[-1].act.loss, self.layers
        # sigmoid loss
        # layers = self.layers
        # K = np.arange(k)
        #
        # pos = np.array(y == K[:, None]).T
        # J = -(np.sum(np.log(pred[pos])) + np.sum(np.log(1 - pred[~pos]))) / m
        # print(Loss)
        J = Loss(y, pred)
        J += sum(layer.regularize() for layer in layers)
        return J

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.argmax(self.feedforward(X)[-1], axis=1)

    def __iadd__(self, other: Layer):
        if isinstance(other, Dense):
            other.input_shape = (self.layers[-1].out_shape,)

        self.k = other.out_shape
        self.layers.append(other)
        # self.layers_size.append((other.input_shape, other.out_shape))
        return self

    def describe(self):
        layers = self.layers
        for i, layer in enumerate(layers):
            print(f'{i}. {layer.W.shape}')


class SVM(Model):

    def __init__(self) -> None:
        super().__init__()
        # hyper params
        self.alpha: float = 1
        self.lambda_: float = 0
        self.c: float = 0
        self.gamma: float = 0
        # model params
        self.W = None
        # engine params
        self.act: Activation_() = None
        self.reg: Regularization() = None
        self.opt: Optimizer() = None

    def compile(self, alpha=0.001, lambda_=0., activation: Activation_ = Hinge_(), reg: Regularization = L2(),
                opt: Optimizer = Vanilla, c=1, gamma=0) -> None:
        super().compile()
        self.act, self.reg, self.opt = activation, reg, opt
        self.alpha, self.lambda_, self.c, self.gamma = alpha, lambda_, c, gamma

    def train(self, X: np.ndarray, y: np.ndarray, iter_=1500, batch=32, eps=0.001, return_loss=True) -> list[float]:
        super().train(np.hstack((np.ones((X.shape[0], 1)), X)), y)
        self.n, self.k = self.X.shape[1], np.max(y) + 1
        # unpacked param
        m, n, k, X, alpha = self.m, self.n, self.k, self.X, self.alpha

        if self.W is None:
            # np.random.seed(1)  # -----
            self.W = np.random.randn(n, k) * eps
        W = self.W

        loss_history = []

        for i in range(iter_):
            batch_idx = np.random.choice(m, batch, replace=False)
            X_, y_ = X[batch_idx], y[batch_idx]

            if return_loss:
                L, dW = self.grad(X_, y_, loss_=True)
                loss_history.append(L)
            else:
                dW = self.grad(X_, y_, loss_=False)

            W -= alpha * dW

        return loss_history

    def predict(self, X, add_ones=True, threshold: float = 0.5) -> np.ndarray:
        W, Act = self.W, self.act.activation

        if add_ones:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        return np.argmax(Act(X, W), axis=1)

    def loss(self, X, y, add_ones=True) -> float:
        m, Reg, lambda_, W, L = X.shape[0], self.reg.norm, self.lambda_, self.W, self.act.loss
        if add_ones:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        print(X.shape)
        return L(X, W, y) + lambda_ * np.sum(Reg(W))  # np.sum(W * W)

    def grad(self, X, y, loss_=False) -> Union[np.ndarray, tuple[float, np.ndarray]]:
        W, lambda_, Reg, dReg = self.W, self.lambda_, self.reg.norm, self.reg.d_norm
        Grad = self.act.loss_grad if loss_ else self.act.grad
        # Loss, Grad = self.activation.loss, self.activation.grad

        if loss_:
            # loss of all X
            # L, dW = Loss(self.X, W, self.y), Grad(X, W, y)
            # loss of only batch of X
            L, dW = Grad(X, W, y)
            L, dW = L + lambda_ * np.sum(Reg(W)), dW + lambda_ * dReg(W)  # np.sum(W * W),2*W
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


class SVM1(Regression):

    def __init__(self) -> None:
        super().__init__()
        # hyper params
        self.alpha: float = 1
        self.lambda_: float = 0
        self.c: float = 0
        self.gamma: float = 0
        # model params
        self.layers = []
        # engine params
        self.act: Activation() = None
        self.reg: Regularization() = None
        self.opt: Optimizer() = None

    def compile(self, alpha=0.001, lambda_=0., activation: Activation = Softmax(), reg: Regularization = L2(),
                opt: Optimizer = Vanilla, c=1, gamma=0) -> None:
        # super().compile()
        self.act, self.reg, self.opt = activation, reg, opt
        self.alpha, self.lambda_, self.c, self.gamma = alpha, lambda_, c, gamma

    def train(self, X: np.ndarray, y: np.ndarray, val: tuple = None, iter_=1500, batch=32, eps=0.001,
              return_loss=True, verbose=True) -> tuple[list, list]:

        Model.train(self, X, y)
        self.n, self.k = self.X.shape[1], np.max(y) + 1
        if len(self.layers) == 0:
            # np.random.seed(1)  # -----
            self.layers.append(Dense1(self.k, input_shape=(self.n,), alpha=self.alpha, lambda_=self.lambda_,
                                      activation=self.act, reg=self.reg, opt=self.opt, eps=eps))

        # unpacked param
        # m, n, k, X, alpha = self.m, self.n, self.k, self.X, self.alpha
        # batch, epoch_size = min(batch, m), max(m / batch, 1)

        return super().train(X, y, val, iter_, batch, return_loss, verbose)

        # loss_history = []
        # for i in range(iter_):
        #     batch_idx = np.random.choice(m, batch, replace=False)
        #     X_, y_ = X[batch_idx], y[batch_idx]
        #
        #     if return_loss:
        #         L, dW = self.grad(X_, y_, loss_=True)
        #         loss_history.append(L)
        #     else:
        #         dW = self.grad(X_, y_, loss_=False)
        #
        #     W -= alpha * dW
        #
        # return loss_history

    # def predict(self, X, add_ones=True, threshold: float = 0.5) -> np.ndarray:
    #     W, Act = self.W, self.act.activation
    #
    #     if add_ones:
    #         X = np.hstack((np.ones((X.shape[0], 1)), X))
    #
    #     return np.argmax(Act(X, W), axis=1)

    # def loss(self, X, y, add_ones=True) -> float:
    #     m, Reg, lambda_, W, L = X.shape[0], self.reg.norm, self.lambda_, self.W, self.act.loss
    #     if add_ones:
    #         X = np.hstack((np.ones((X.shape[0], 1)), X))
    #     print(X.shape)
    #     return L(X, W, y) + lambda_ * np.sum(Reg(W))  # np.sum(W * W)

    # def grad(self, X, y, loss_=False) -> Union[np.ndarray, tuple[float, np.ndarray]]:
    #     W, lambda_, Reg, dReg = self.W, self.lambda_, self.reg.norm, self.reg.d_norm
    #     Grad = self.act.loss_grad if loss_ else self.act.grad
    #     # Loss, Grad = self.activation.loss, self.activation.grad
    #
    #     if loss_:
    #         # loss of all X
    #         # L, dW = Loss(self.X, W, self.y), Grad(X, W, y)
    #         # loss of only batch of X
    #         L, dW = Grad(X, W, y)
    #         L, dW = L + lambda_ * np.sum(Reg(W)), dW + lambda_ * dReg(W)  # np.sum(W * W),2*W
    #         return L, dW
    #     else:
    #         return Grad(X, W, y) + lambda_ * dReg(W)  # 2 * W

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


if __name__ == '__main__':
    from Algorithms.ML.neural_network.nn import NN
    import numpy as np
    from sklearn.datasets import make_blobs
    from layer import Dense1
    import normalize

    n_samples = 500
    X, Y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.40)
    X, _, _ = normalize.standard_deviation(X)
    slices = int(n_samples * 0.8)
    X, Y, Xv, Yv = X[:slices, :], Y[:slices], X[slices:, :], Y[slices:]
    import matplotlib.pyplot as plt

    # plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
    # plt.show()

    # np.random.seed(78)
    lambda_, alpha = 1e-1, 5
    # print('-------------  model NN  ----------------')
    # model = Regression([Dense1(5, input_shape=(2,), alpha=alpha, lambda_=lambda_), ])
    # model += Dense1(5, alpha=alpha, lambda_=lambda_)
    # model += Dense1(2, alpha=alpha, lambda_=lambda_)
    # model.compile()
    # # model.describe()
    #
    # history_t, history_v = model.train(X, Y, val=(Xv, Yv), iter_=1000, batch=32, return_loss=True)
    # pred = model.predict(X)
    # import metrics
    #
    # metrics.print_metrics(Y, pred)
    #
    # plt.plot(range(len(history_t)), history_t)
    # plt.plot(range(len(history_v)), history_v)
    #
    # print(history_t[0], history_t[-1])
    #
    # print('-------------  model SVM  ----------------')
    #
    # model2 = SVM()
    # model2.compile(lambda_=2.5e4, alpha=1e-7, activation=Softmax(), reg=L12())
    # history_svm_t = model2.train(X, Y, iter_=1000)
    # pred = model2.predict(X)
    # metrics.print_metrics(Y, pred)
    # plt.plot(range(len(history_svm_t)), history_svm_t)
    # plt.legend(['Train', 'Val', 'svm tr'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    # W1 = [np.random.randn(5, 3), np.random.randn(5, 6), np.random.randn(2, 6)]
    # model1 = NN(W=W1, alpha=alpha, landa=lambda_)
    # model1.fit(X, Y, max_iter=1000)
    # print(model.loss(X, Y))
    # print(model1.cost(X, Y))
    # model = Model()
    # model.k_fold(X, Y, k=5)
