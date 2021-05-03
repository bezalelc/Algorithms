import numpy as np
from polynomial import interpolation as inter
from typing import Union


def newton_cotes(func=None, points=None, n=2, inter_method=inter.newton, range_=(-1, 1)):
    if not func and not points or func is not None and points is not None:
        print("enter point or function adn range")
        return
    # get points from function
    if func:
        points = inter.chebyshev_root(n, *range_)
        print(points)
    # np.vectorize(func)

    # P_n=inter.


if __name__ == '__main__':
    newton_cotes()
