import numpy as np
import matplotlib as plot


# n+1 splines
# x,y,z -> n+1
# h,b -> n
# v,u -> n-1


# x = y = range(10)
# z = np.zeros(n)


def init(x, y):
    h = x[1:] + x[:-1]
    b = (6 / h) * (y[1:] - y[:-1])
    u, v = 2 * (h[1:] + h[:-1]), b[1:] - b[:-1]
    return u, v


def rowReduction(u, h):
    for i in range(1, n - 1):
        u[i] = u[i] - ((h[i - 1]) ** 2) / u[i - 1]
        # v[i] = b[i] - b[i - 1] - (((h[i - 1]) ** 2) / u[i - 1]) * v[i - 1]


#
# def solution:
#     for i in range(n - 1, 0, -1):
#         z[i] = (v[i] - (h[i - 1] * z[i + 1])) / u[i]
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    n = 5
    # h = np.empty((n,))
    x = np.arange(n + 1)
    h = x[1:] + x[:-1]
    z = np.empty((5,))

    print()
    print()
