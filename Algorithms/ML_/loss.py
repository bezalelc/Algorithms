import numpy as np
from typing import NewType, Callable

Loss = NewType('Loss', Callable[[np.ndarray, np.ndarray], np.ndarray])
hinge = Loss(lambda X, y:
             (np.sum(np.maximum(0, X - X[np.arange(y.shape[0]), y].reshape((-1, 1)) + 1)) - y.shape[0]) / y.shape[0])
cross_entropy = Loss(lambda X, y: np.sum(-np.log10(X)[np.arange(y.shape[0]), y]))
# mse = Loss(lambda Z: np.sum() / Z.shape[1])
# binary_cross_entropy = Loss()
