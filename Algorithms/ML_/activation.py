import numpy as np
from typing import NewType, Callable

# **********************   Activation   **********************************
# hidden layers
Activation = NewType('Activation', Callable[[np.ndarray], np.ndarray])
sigmoid = Activation(lambda X: 1. / (1. + np.exp(-X)))
linear = Activation(lambda X: X)  # Regression
relu = Activation(lambda X: np.maximum(0, X))
leaky_relu = Activation(lambda X: np.where(X > 0, X, X * 0.01))
tanh = Activation(lambda X: np.tanh(X))
# output layer
softmax = Activation(lambda X: np.exp(X) / np.sum(np.exp(X)))  # “softer” version of argmax
logistic = Activation(lambda X: 1. / (1. + np.exp(-X)))
# **********************   Loss   **********************************
# Loss = NewType('Loss', Callable[[np.ndarray, np.ndarray], np.ndarray])
# # mse = Loss(lambda Z: np.sum() / Z.shape[1])
# # binary_cross_entropy = Loss()
# hinge = Loss(lambda X, y:
#              (np.sum(np.maximum(0, X - X[np.arange(y.shape[0]), y].reshape((-1, 1)) + 1)) - y.shape[0]) / y.shape[0])
# cross_entropy = Loss(lambda X, y: np.sum(-np.log10(X)[np.arange(y.shape[0]), y]))
# **********************   grad   **********************************
# Grad = NewType('Grad', Callable[[np.ndarray, np.ndarray], np.ndarray])
# grad_1 = Grad(lambda X, f, h: (f(X + h) - f(X)) / h)
# grad_2 = Grad(lambda X, f, h: (f(X + h) - f(X - h)) / (2 * h))
# **********************   optimizer   **********************************
# Optimizer = NewType('Optimizer', Callable[[np.ndarray, np.ndarray], np.ndarray])
# # vanilla = Optimizer()
# # momentum = Optimizer()
# # ada_grad = Optimizer()
# # adam = Optimizer()
