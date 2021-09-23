import abc
import numpy as np
from numpy import random
from typing import Union, NamedTuple, NewType, Callable
from activation import Sigmoid, Activation, ReLU, Softmax, Linear
from regularization import Regularization, L2
from optimizer import Optimizer, Vanilla

InitWeights = NewType('InitWeights', Callable[[tuple, float], np.ndarray])
stdScale = InitWeights(lambda shape, std: random.randn(*shape) * std)


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, out_shape: int, input_shape: tuple = None) -> None:
        super().__init__()
        self.__input_shape: tuple = input_shape
        self.out_shape: int = out_shape

    @abc.abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True) -> np.ndarray:
        pass

    @abc.abstractmethod
    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        loss of activation if the activation is the last layer

        :param y: True classes
        :param h: input of the current layer

        :return: delta of the prediction
        """
        pass

    @abc.abstractmethod
    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        """
        loss of linear layer if the layer is the last layer

        :param y: True classes
        :param pred: input of the current layer

        :return: loss of the prediction
        """
        pass

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: tuple) -> None:
        self.__input_shape = input_shape_


class ActLayer(Layer):

    def __init__(self, out_shape: int, activation: Activation, input_shape: tuple = None) -> None:
        super().__init__(out_shape, input_shape)
        self.numeric_stability = False
        self.act: Activation = activation

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.numeric_stability:
            X -= np.max(X, axis=1, keepdims=True)
        return self.act.activation(X)

    def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True) -> np.ndarray:
        dAct = self.act.grad
        return delta * dAct(h)

    def grad(self, X: np.ndarray, h) -> np.ndarray:
        return self.act.grad(h)

    def delta(self, y: np.ndarray, pred: np.ndarray):
        return self.act.delta(y, pred)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return self.act.loss(y, pred)


class WeightLayer(Layer):
    """
    layer with weights and activation
    """

    def __init__(self, out_shape: int, reg: Regularization = L2,
                 opt: Optimizer = Vanilla(), opt_bias_: Optimizer = Vanilla(), eps=1e-3, alpha=1e-5,
                 input_shape: tuple = None, numeric_stability=True,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W=stdScale) -> None:
        super().__init__(out_shape, input_shape=input_shape)

        # general params
        self.numeric_stability: bool = numeric_stability
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.init_W = init_W
        self.W = None
        self.b = init_W((out_shape,), eps) if bias else None
        self.__input_shape = input_shape
        self.Z: np.ndarray = np.array([])
        if input_shape:
            assert len(input_shape) == 1
            self.W = init_W((input_shape[0], out_shape), eps)

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.reg: Regularization = reg
        self.opt: Optimizer() = opt
        self.opt_bias_: Optimizer() = opt_bias_

    def forward(self, X: np.ndarray) -> np.ndarray:
        W, b = self.W, self.b

        Z = X @ W
        if self.bias:
            Z += b

        if self.numeric_stability:
            Z -= np.max(Z, axis=1, keepdims=True)
        self.Z = Z

        return Z

    def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True) -> np.ndarray:
        dReg, Opt = self.reg.d_norm, self.opt
        W, b, Z, alpha, lambda_, m, bias = self.W, self.b, self.Z, self.alpha, self.lambda_, h.shape[0], self.bias

        dW = h.T @ delta / m + lambda_ * dReg(W) / 2
        if bias:
            db = delta.sum(axis=0) / m
            if self.reg_bias:
                db += lambda_ * dReg(b) / 2
            b -= self.opt_bias_.opt(db, alpha)

        if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
            delta = delta @ W.T

        W -= Opt.opt(dW, alpha)
        return delta

    def grad(self, delta: np.ndarray, h: np.ndarray) -> dict:
        """
        gradient for specific activation

        :param delta: gradient of the next layer
        :param h: input of the current layer

        :return: gradient of the current layer for the backward
        """
        dReg, Opt = self.reg.d_norm, self.opt
        W, b, Z, alpha, lambda_, m, bias = self.W, self.b, self.Z, self.alpha, self.lambda_, h.shape[0], self.bias
        grades = {'delta': delta @ W.T}

        grades['dW'] = h.T @ delta / m + lambda_ * dReg(W) / 2
        if bias:
            db = delta.sum(axis=0) / m
            if self.reg_bias:
                db += lambda_ * dReg(b) / 2
            grades['db'] = db

        return grades

    def delta(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        loss of activation if the activation is the last layer

        :param y: True classes
        :param h: input of the current layer

        :return: delta of the prediction
        """
        return Linear.delta(y, h)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        """
         loss of linear layer if the layer is the last layer

         :param y: True classes
         :param pred: input of the current layer

         :return: loss of the prediction
         """
        return Linear.loss(y, pred)

    def regularize(self) -> float:
        if not self.reg:
            return 0

        Reg, W, b, lambda_ = self.reg.norm, self.W, self.b, self.lambda_
        r = Reg(W) * lambda_
        if self.reg_bias:
            r += Reg(b) * lambda_

        return r

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: list):
        self.__input_shape = input_shape_
        bias, eps = self.bias, self.eps
        if input_shape_:
            assert len(input_shape_) == 1
            self.W = self.init_W((input_shape_[0], self.out_shape), eps)
            # self.W = random.randn(input_shape_[0], self.out_shape) * eps
            if bias:
                self.b = self.init_W((self.out_shape,), eps)
                # self.b = random.randn(self.out_shape) * eps


class BatchNorm(Layer):

    def __init__(self, out_shape: int, input_shape: tuple = None, eps=1e-5, axis=0, momentum=.9) -> None:
        super().__init__(out_shape, input_shape)
        self.gamma: np.ndarray = np.ones(out_shape)
        self.beta: np.ndarray = np.zeros(out_shape)
        self.mu = 0
        self.var = 0
        self.std = 0
        self.z = 0
        self.eps = eps
        self.momentum = momentum
        self.running_mu = np.zeros(out_shape, dtype=np.float64)
        self.running_var = np.zeros(out_shape, dtype=np.float64)
        self.axis = axis

    def forward(self, X: np.ndarray, mode='train') -> np.ndarray:
        gamma, beta, layer_norm, eps = self.gamma, self.beta, self.axis, self.eps

        if mode == 'train':
            layer_norm = self.axis

            mu, var = X.mean(axis=0), X.var(axis=0)
            std = var ** 0.5
            z = (X - mu) / (std + eps)
            out = gamma * z + beta
            if layer_norm == 0:
                momentum = self.momentum
                self.running_mu = momentum * self.running_mu + (1 - momentum) * mu
                self.running_var = momentum * self.running_var + (1 - momentum) * var

            self.std, self.var, self.mu, self.z = std, var, mu, z

        elif mode == 'test':
            out = gamma * (X - self.running_mu) / (self.running_var + eps) ** 0.5 + beta

        else:
            raise ValueError('Invalid forward batch norm mode "%s"' % mode)

        return out

    def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True, return_d=True) -> Union[np.ndarray, tuple]:
        z, gamma, beta, mu, var, std = self.z, self.gamma, self.beta, self.mu, self.var, self.std

        d_beta = delta.sum(axis=0)
        d_gamma = np.sum(delta * z, axis=0)

        m = delta.shape[0]
        df_dz = delta * gamma
        dx = (1 / (m * std)) * (m * df_dz - np.sum(df_dz, axis=0) - z * np.sum(df_dz * z, axis=0))

        return dx, d_gamma, d_beta

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass


class Dense(Layer):
    """
    layer with weights and activation
    """

    def __init__(self, out_shape: int, activation: Activation = Sigmoid, reg: Regularization = L2,
                 opt: Optimizer = Vanilla(), opt_bias_: Optimizer = Vanilla(), eps=1e-3, alpha=1e-5,
                 input_shape: tuple = None,
                 lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W=stdScale) -> None:
        super().__init__(out_shape, input_shape=input_shape)

        # general params
        self.m = 1
        self.eps: float = eps
        self.bias: bool = bias
        self.reg_bias: bool = reg_bias
        self.opt_bias: bool = opt_bias

        # layer params
        self.init_W = init_W
        self.W = None
        self.b = init_W((out_shape,), eps) if bias else None
        self.__input_shape = input_shape

        if input_shape:
            assert len(input_shape) == 1
            self.W = init_W((input_shape[0], out_shape), eps)

        # hyper params
        self.alpha, self.lambda_ = alpha, lambda_

        # engine param
        self.act: Activation = activation
        self.reg: Regularization = reg
        self.opt: Optimizer() = opt
        self.opt_bias_: Optimizer() = opt_bias_

        if isinstance(self.act, ReLU):
            self.numeric_stability = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        Act, W, b = self.act.activation, self.W, self.b

        Z = X @ W
        if self.bias:
            Z += b

        if self.numeric_stability:
            Z -= np.max(Z, axis=1, keepdims=True)
        self.Z = Z
        H = Act(Z)

        return H

    def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True) -> np.ndarray:
        dAct, dReg, Opt = self.act.grad, self.reg.d_norm, self.opt
        W, b, Z, alpha, lambda_, m, bias = self.W, self.b, self.Z, self.alpha, self.lambda_, h.shape[0], self.bias
        delta = delta * dAct(Z)  # activation grad

        dW = h.T @ delta / m + lambda_ * dReg(W) / 2
        if bias:
            db = delta.sum(axis=0) / m
            if self.reg_bias:
                db += lambda_ * dReg(b) / 2
            # print(db.shape)
            # print(Opt.opt(db, alpha).shape)
            # b -= Opt.opt(db, alpha)
            b -= self.opt_bias_.opt(db, alpha)

        if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
            delta = delta @ W.T

        # TODO remove next 3 line
        # print(dW*m, db*m, np.sum(dW) + np.sum(db))
        # print(np.sum(np.abs(dW)), np.sum(np.abs(db)), np.sum(dW) + np.sum(db))
        # print(np.su)

        W -= Opt.opt(dW, alpha)
        return delta

    def grad(self, delta: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        gradient for specific activation

        :param delta: gradient of the next layer
        :param pred: input of the current layer

        :return: gradient of the current layer for the backward
        """
        dAct, W, Z = self.act.grad, self.W, self.Z
        return delta @ W.T * dAct(Z)

    def delta(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        loss of activation if the activation is the last layer

        :param y: True classes
        :param h: input of the current layer

        :return: delta of the prediction
        """
        return self.act.delta(y, h)

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        return self.act.loss(y, pred)

    def regularize(self) -> float:
        if not self.reg:
            return 0

        Reg, W, b, lambda_ = self.reg.norm, self.W, self.b, self.lambda_
        r = Reg(W) * lambda_
        if self.reg_bias:
            r += Reg(b) * lambda_

        return r

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape_: list):
        self.__input_shape = input_shape_
        bias, eps = self.bias, self.eps
        if input_shape_:
            assert len(input_shape_) == 1
            self.W = self.init_W((input_shape_[0], self.out_shape), eps)
            # self.W = random.randn(input_shape_[0], self.out_shape) * eps
            if bias:
                self.b = self.init_W((self.out_shape,), eps)
                # self.b = random.randn(self.out_shape) * eps

    def __str__(self) -> str:
        s = f'W.shape: ' + str(self.W.shape)
        return s


# class Dense1(Layer):
#     """
#     layer with weights and activation
#     """
#
#     def __init__(self, out_shape: int, activation: Activation = Sigmoid, reg: Regularization = L2,
#                  opt: Optimizer = Vanilla(), opt_bias_: Optimizer = Vanilla(), eps=1e-3, alpha=1e-5,
#                  input_shape: tuple = None,
#                  lambda_=0, bias=True, reg_bias=False, opt_bias=True, init_W=stdScale) -> None:
#         super().__init__(out_shape, input_shape=input_shape)
#
#         # general params
#         self.m = 1
#         self.eps: float = eps
#         self.bias: bool = bias
#         self.reg_bias: bool = reg_bias
#         self.opt_bias: bool = opt_bias
#
#         # layer params
#         self.init_W = init_W
#         self.W = None
#         self.b = init_W((out_shape,), eps) if bias else None
#         self.__input_shape = input_shape
#
#         if input_shape:
#             assert len(input_shape) == 1
#             self.W = init_W((input_shape[0], out_shape), eps)
#
#         # hyper params
#         self.alpha, self.lambda_ = alpha, lambda_
#
#         # engine param
#         self.act: Activation = activation
#         self.reg: Regularization = reg
#         self.opt: Optimizer() = opt
#         self.opt_bias_: Optimizer() = opt_bias_
#
#         if isinstance(self.act, ReLU):
#             self.numeric_stability = False
#
#     def forward(self, X: np.ndarray) -> np.ndarray:
#         Act, W, b = self.act.activation, self.W, self.b
#
#         Z = X @ W
#         if self.bias:
#             Z += b
#
#         if self.numeric_stability:
#             Z -= np.max(Z, axis=1, keepdims=True)
#         self.Z = Z
#         H = Act(Z)
#
#         return H
#
#     def backward(self, delta: np.ndarray, h: np.ndarray, return_delta=True) -> np.ndarray:
#         dAct, dReg, Opt = self.act.grad, self.reg.d_norm, self.opt
#         W, b, Z, alpha, lambda_, m, bias = self.W, self.b, self.Z, self.alpha, self.lambda_, h.shape[0], self.bias
#         delta = delta * dAct(Z)  # activation grad
#
#         dW = h.T @ delta / m + lambda_ * dReg(W) / 2
#         if bias:
#             db = delta.sum(axis=0) / m
#             if self.reg_bias:
#                 db += lambda_ * dReg(b) / 2
#             # print(db.shape)
#             # print(Opt.opt(db, alpha).shape)
#             # b -= Opt.opt(db, alpha)
#             b -= self.opt_bias_.opt(db, alpha)
#
#         if return_delta:  # if the layer need to return is delta for chain rule, for example the first layer don't
#             delta = delta @ W.T
#
#         # TODO remove next 3 line
#         # print(dW*m, db*m, np.sum(dW) + np.sum(db))
#         # print(np.sum(np.abs(dW)), np.sum(np.abs(db)), np.sum(dW) + np.sum(db))
#         # print(np.su)
#
#         W -= Opt.opt(dW, alpha)
#         return delta
#
#     def grad(self, delta: np.ndarray, pred: np.ndarray) -> np.ndarray:
#         """
#         gradient for specific activation
#
#         :param delta: gradient of the next layer
#         :param pred: input of the current layer
#
#         :return: gradient of the current layer for the backward
#         """
#         dAct, W, Z = self.act.grad, self.W, self.Z
#         return delta @ W.T * dAct(Z)
#
#     def delta(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
#         """
#         loss of activation if the activation is the last layer
#
#         :param y: True classes
#         :param h: input of the current layer
#
#         :return: delta of the prediction
#         """
#         return self.act.delta(y, h)
#
#     def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
#         return self.act.loss(y, pred)
#
#     def regularize(self) -> float:
#         if not self.reg:
#             return 0
#
#         Reg, W, b, lambda_ = self.reg.norm, self.W, self.b, self.lambda_
#         r = Reg(W) * lambda_
#         if self.reg_bias:
#             r += Reg(b) * lambda_
#
#         return r
#
#     @property
#     def input_shape(self) -> tuple:
#         return self.__input_shape
#
#     @input_shape.setter
#     def input_shape(self, input_shape_: list):
#         self.__input_shape = input_shape_
#         bias, eps = self.bias, self.eps
#         if input_shape_:
#             assert len(input_shape_) == 1
#             self.W = self.init_W((input_shape_[0], self.out_shape), eps)
#             # self.W = random.randn(input_shape_[0], self.out_shape) * eps
#             if bias:
#                 self.b = self.init_W((self.out_shape,), eps)
#                 # self.b = random.randn(self.out_shape) * eps
#
#     def __str__(self) -> str:
#         s = f'W.shape: ' + str(self.W.shape)
#         return s

# filter_shape,num_filters,stride,padding,pad_size
# kW,kH,dW,dH,padW,padH
# from keras.layers import Con
# class LayerParams(NamedTuple):
#     # input_shape: tuple
#     # out_shape: int
#     trainable: bool = True
#     bias: bool = True
#     act: Activation = ReLU
#     reg: Regularization = L2
#     opt: Optimizer = Vanilla
#     reg_bias: bool = True
#     opt_bias: bool = True
#     numeric_stability: bool = True
#     # W: np.ndarray = None
#     alpha: float = 1e-7
#     lambda_: float = 0
#     eps: float = 1e-4
