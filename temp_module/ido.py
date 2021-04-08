import numpy as np
import matplotlib.pyplot as plt


def f(x):
    # return np.exp(x ** 2) - 2 * np.cos(x ** 2) ** 3 + (-x) ** x
    # return np.sin(x)
    # return x ** 3  # + 7 - 8 * x ** 4 + (x / 5) ** 5
    return x ** 2 - 17


def ido(func, x0, x1, landa=1e-3, epsilon=1e-3, max_iter=100):
    x_history, f_history = [x1], [f(x1)]
    x0, x1 = np.float128(x0), np.float128(x1)
    # x0, x1 = np.float128, np.float128
    for i in range(max_iter):
        df = (f(x1) - f(x0)) / (x1 - x0)

        # if df
        try:
            x0, x1 = x1, x0 - f(x0) / df  # df
            x_history.append(x1), f_history.append(f(x1))
        except ZeroDivisionError:
            print('choose other points')
            return

        # print(x1)
        if np.abs(f(x1)) < landa or np.abs(x1 - x0) < epsilon:
            return x1, x_history, f_history


if __name__ == '__main__':
    points = np.linspace(-10, 10, num=1000)
    np.vectorize(f)

    x, x_history, f_history = ido(f, 1, 2)
    print(f(x))

    plt.figure(figsize=(8, 8))
    plt.plot(range(points.shape[0]), f(points))
    plt.legend('f(x) graph')
    plt.scatter(x_history, f_history, marker='*', linewidths=3, color='r')
    print(x_history)

    # plt.scatter(len(f_history), f_history,marker='*', linewidths=1)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show(), x
