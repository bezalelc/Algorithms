import numpy as np
from typing import Callable, NewType

Regularization = NewType('Regularization', Callable[[np.ndarray, int], np.ndarray])
L1 = Regularization(lambda x, axis=-1: np.sum(np.abs(x), axis=axis))
L2 = Regularization(lambda x, axis=-1: np.sum(x ** 2, axis=axis) ** 0.5)
L12 = Regularization(lambda x, axis=-1: L1(x, axis) + L2(x, axis))  # elastic net
