import numpy as np
import matplotlib.pyplot as plt


def f(x):
    # return np.exp(x ** 2) - 2 * np.cos(x ** 2) ** 3 + (-x) ** x
    # return np.sin(x)
    return x ** 3  # + 7 - 8 * x ** 4 + (x / 5) ** 5


def ido(func, x0, x1, landa=1e-7, epsilon=1e-7, max_iter=100):
    for i in range(max_iter):
        df = (f(x1) - f(x0)) / (x1 - x0)

        # if df
        # try:
        x0,x1 =x1, x0 - f(x0) / df
        # raise
        print(x1)
        if np.abs(f(x1)) < landa or np.abs(x1 - x0) < epsilon:
            return x1




if __name__ == '__main__':
    # f(x) = e^x-2*cos(x^2)^3+x^7
    # f(x) = x^2-
    print(f(3))
    a = np.arange(-10, 10)
    np.vectorize(f)
    plt.figure(figsize=(8, 8))
    plt.plot(range(20), f(a))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    print(ido(f, 1, 2))
