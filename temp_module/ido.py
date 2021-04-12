import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 4 - 4 * x ** 3 + x ** 2


# derivative function of f
def df(x):
    return 4 * x ** 3 - 12 * x ** 2 + 2 * x


def secant(x0, x1):
    return (f(x1) - f(x0)) / (x1 - x0)


def root_calc(f, x0, method, lambd=1e-7, epsilon=1e-7, max_iter=100):
    """
    find root for function using newton's method

    :param f: function to find root
    :param x0: start x point
    :param method: method to calculate the derivative 'newton' for real derivative or else for secant
    :param lambd: minimal value for stop condition => if |f(x1)| < lambd: return
    :param epsilon: minimal value for stop condition => if |x1-x0| < epsilon: return
    :param max_iter: max iteration

    :return: x_history: list of all x that we found, the last element is the closest root
    """
    x1 = x0 + 1
    x_history = [x1]
    for i in range(max_iter):
        if method == 'newton':
            df_x = df(x1)
            x0 = x1
        else:
            df_x = secant(x0, x1)
        try:
            x0, x1 = x1, x0 - f(x0) / df_x
            x_history.append(x1)
        except ZeroDivisionError:
            break

        if np.abs(f(x1)) < lambd or np.abs(x1 - x0) < epsilon:
            break
    return x_history


if __name__ == '__main__':
    # plot graph of f(x)
    points = np.linspace(-1, 3.8, num=1000)
    np.vectorize(f)
    plt.plot(points, f(points))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = x^4 - 4x^3 + x^2')
    plt.show()

    x_history = root_calc(f, 1.0, 'newton')
    x = x_history[-1]
    x_history = np.array(x_history)
    print(f(x_history[-1]))
    error = x_history - x
    p1 = np.log(np.abs(error[2:-1] / error[1:-2])) / np.log(np.abs(error[1:-2] / error[:-3]))
    c1 = error[2:-1] / (error[1:-2])  # **p1

    # plot newton error
    plt.plot(range(len(error)), error)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.scatter(len(error) - 1, 0, linewidths=3)
    print('Newton: root =', x, '. Iterations =', len(error) - 1, '\nn\t\terror\t\t\t\t\t\t\tXn\t\t\t\t\t\t\t\tp')
    for i in range(len(error)):
        print(i, '\t', error[i], '\t\t\t', x_history[i], '\t\t\t',
              f"{p1[i - 2] if i >= 2 and i < len(error) - 1 else ''}")

    x_history = root_calc(f, 1.0, 'secant')
    x_history = np.array(x_history)
    error = x_history - x
    p2 = np.log(error[2:-1] / error[1:-2]) / np.log(error[1:-2] / error[:-3])
    c2 = error[2:-1] / (error[1:-2])

    # plot secant error
    plt.plot(range(len(error)), error)
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend(['Newton', 'Secant'])
    plt.scatter(len(error) - 1, 0, linewidths=3)
    print('\nSecant: root =', x, '. Iterations =', len(error) - 1, '\nn\t\terror\t\t\t\t\t\t\tXn\t\t\t\t\t\t\t\tp')
    for i in range(len(error)):
        print(i, '\t', error[i], '\t\t\t', x_history[i], '\t\t\t',
              f"{p2[i - 2] if i >= 2 and i < len(error) - 1 else ''}")
    plt.grid(ls='--')
    plt.show()

    # plot p
    plt.plot(range(2, len(p1) + 2), p1)
    plt.plot(range(2, len(p2) + 2), p2)
    plt.xlabel('Iterations')
    plt.ylabel('p')
    plt.title('The order of convergence')
    plt.legend(['p Newton', 'p Secant'])
    plt.grid(ls='--')
    plt.show()

    print('\n', c1.tolist())
    print(c2.tolist())
    print(df(1.0))
    print(f(1.0))

