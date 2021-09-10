import numpy as np
from typing import NewType, Callable

Grad = NewType('Grad', Callable[[np.ndarray, np.ndarray], np.ndarray])
grad_1 = Grad(lambda X, f, h: (f(X + h) - f(X)) / h)
grad_2 = Grad(lambda X, f, h: (f(X + h) - f(X - h)) / (2 * h))
