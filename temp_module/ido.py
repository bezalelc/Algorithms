import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


def f(x):
    # return np.exp(x ** 2) - 2 * np.cos(x ** 2) ** 3 + (-x) ** x
    # return np.sin(x)
    # return x ** 2 + 1
    # return x ** (3 - x ** 3)
    # return x ** 3  # + 7 - 8 * x ** 4 + (x / 5) ** 5
    # return x ** 2 - 17
    # return x ** 4 - 4 * x ** 3
    return x ** 4 - 4 * x ** 3 + x ** 2
    # return x**2-x**x


def ido(func, x0, x1, landa=1e-15, epsilon=1e-15, max_iter=100):
    x_history, f_history = [x1], [f(x1)]
    for i in range(max_iter):
        df = (func(x1) - func(x0)) / (x1 - x0)

        try:
            x0, x1 = x1, x0 - func(x0) / df  # df
            x_history.append(x1)
        except ZeroDivisionError:
            print('choose other points')
            return

        if np.abs(func(x1)) < landa or np.abs(x1 - x0) < epsilon:
            return x1, x_history


if __name__ == '__main__':
    points = np.linspace(-10, 10, num=1000)
    np.vectorize(f)

    x, x_history = ido(f, 1.0, 2.0)

    # plot the function
    plt.figure(figsize=(12, 8))
    plt.title('ido')
    plt.subplot(1, 2, 1)
    plt.plot(points, f(points))
    plt.legend('f(x) graph')
    plt.scatter(x_history[:-1], f(np.array(x_history[:-1])), marker='.', linewidths=2, color='black')
    plt.scatter(x_history[-1], f(np.array(x_history[-1])), marker='*', linewidths=3, color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    # plot the error
    plt.subplot(1, 2, 2)
    error = np.array([np.abs(x_ - x) for x_ in x_history])
    plt.plot(range(len(error)), error)
    plt.scatter(range(len(error)), error, marker='+', linewidths=2, color='black')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('error')

    print(f'x={x}, f({x})={f(x)}')
    plt.show()

    from scipy.optimize import root

    print(root(f, np.array([2., ])))
