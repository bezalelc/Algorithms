import numpy as np
from typing import NewType, Callable

# hidden layers
Activation = NewType('Activation', Callable[[np.ndarray], np.ndarray])
sigmoid = Activation(lambda X: 1. / (1. + np.exp(-X)))
linear = Activation(lambda X: X)  # Regression
relu = Activation(lambda X: np.maximum(0, X))
leaky_relu = Activation(lambda X: np.where(X > 0, X, X * 0.01))
tanh = Activation(lambda X: np.tanh(X))
# output layer
softmax = Activation(lambda X: np.exp(X) / np.sum(np.exp(X), axis=0))  # “softer” version of argmax
logistic = Activation(lambda X: 1. / (1. + np.exp(-X)))

Loss = NewType('Loss', Callable[[np.ndarray], np.ndarray])
# mse = Loss(lambda Z: np.sum() / Z.shape[1])
# binary_cross_entropy = Loss()
# cross_entropy=Loss()
