import numpy as np
import matplotlib as plot
import sympy


# x = np.arange(9).reshape((3, -1))
# y = np.arange(3)
# print(np.hstack((x, y[:, None])))


def f(x):
    return x ** 2


f_ = lambda x: x ** 2 if x < 0 else 6

print(f_(4))
