import numpy as np
from typing import NewType, Callable

Optimizer = NewType('Optimizer', Callable[[np.ndarray, np.ndarray], np.ndarray])
# vanilla = Optimizer()
# momentum = Optimizer()
# ada_grad = Optimizer()
# adam = Optimizer()
