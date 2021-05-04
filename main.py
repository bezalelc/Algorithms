import numpy as np
import matplotlib as plot
import sympy

x = np.arange(9).reshape((3, -1))
y = np.arange(3)
print(np.hstack((x, y[:, None])))
