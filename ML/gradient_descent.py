import matplotlib
import numpy as np
from general import load_data, normalize
import matplotlib.pyplot as plt


# --------------------------------------  optimizers  ---------------------------------------------
def simple(X, y, theta, alpha, optimizer_data=None):
    theta -= alpha * (1 / X.shape[0]) * (X.T @ (X @ theta - y))


def momentum(X, y, theta, V, alpha, beta):


# --------------------------------------  optimizers  ---------------------------------------------

def simple_linear_regression(X, y, theta, alpha=10e-7, num_iter=1000, batch=30, optimizer=simple, optimizer_data=None):
    """
    linear regression

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        alpha: learning rate by default alpha=10e-7 for prevent Entertainment, recommended range: [1e-5,10]
        num_iter: number of the iteration , should be between (1000-10e+9)
        batch: number of example to compute at once, the recommended is 30
        optimizer: <string>: algorithm to compute theta, optional [momentum,,,,adam]
        optimizer_data: <dict>: data for the specified optimizer

    optional optimizers algorithm: [simple,momentum,,,,adam]
        simple:
            theta[k+1] -= alpha*J'(theta[k])

        momentum:
            V[k] = beta*V[i-1] + alpha*J'(theta[k])
            theta[k+1] -= V[k]

            optional optimizer_data:
                beta: <number>: the Percentage of consideration in the previous gradient, recommended is 0.9
        adam:

    :return:
        theta
        J_history: the history of the cost function

    :efficiency: O(batch*n^2)
    """

    optimizer_data = optimizer_data.values() if optimizer_data else []
    J_history = []  # np.array([])

    m, start = X.shape[0], 0
    # batch = m
    for i in range(num_iter):
        # if i == 36:
        #     u = 9
        start = (i * batch) % (m - 1) if start + batch < m - 1 else 0
        end = start + batch if start + batch < m - 1 else m - 1

        # print(i, ": ", start, end)
        # theta -= alpha * (1 / batch) * (X[start:end].T @ (X[start:end] @ theta - y[start:end]))
        optimizer(X[start:end], y[start:end], theta, alpha, *optimizer_data)
        J_history.append(cost(X[start:end], y[start:end], theta))
    return theta, J_history


def normal_eqn(X, y):
    """

    :param X:
    :param y:

    :return:

    :efficiency:
    """
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)


def cost(X, y, theta):
    """
    compute the cost of the range between X*theta and y

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: vector of parameters for the feature

    :return: <float>: J cost of data x for the current theta

    :efficiency: O(m*n^2)
    """
    return (1 / (2 * X.shape[0])) * np.sum((X @ theta - y) ** 2)


def h_theta(X, theta):
    return X @ theta


# -----------------------------------------  debug function  --------------------------------------------
def test_ex1data1():
    print('===================================== test ex1data1 =====================================')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data1.txt')
    X, y = np.insert(data[:, :-1], 0, np.ones((data[:, :-1].shape[0]), dtype=data.dtype), axis=1), data[:, -1:]

    print('\n-------------------------------  iter on ex1data1.txt  ---------------------------------------')
    theta = np.zeros((X.shape[1], 1))
    # print(theta.shape)
    print(f'cost={cost(X, y, theta)} should be 32.072733877455676')
    theta = normal_eqn(X, y)
    print(f'theta={[float(t) for t in theta]} should be [-3.89578088, 1.19303364]')
    print(f'cost={cost(X, y, theta)} should be 4.476971375975179 ')
    print('mean theta error iter=', np.mean(np.abs(h_theta(X, theta) - y)), 'should be 2.1942453988270043')
    # print('error in octave=', np.mean(np.abs(h_theta(X, np.array([-3.6303, 1.1664])) - y)))
    # print('predict in octave=',h_theta(np.array([1, 7]), np.array([-3.6303, 1.1664])) * 10000)

    print('\n-------------------------------  regression  ---------------------------------------')
    theta = np.zeros((X.shape[1], 1))
    J_history = simple_linear_regression(X, y, theta, alpha=10e-3, num_iter=100000)[1]
    print(f'theta={[float(t) for t in theta]}')
    print(f'cost={cost(X, y, theta)}')

    plt.plot(range(len(J_history)), J_history)
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()


def test_ex1data2():
    print('\n\n===================================== test ex1data2 =====================================')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data2.txt')
    X, y = np.insert(data[:, :-1], 0, np.ones((data[:, :-1].shape[0]), dtype=data.dtype), axis=1), data[:, -1:]

    print('\n-------------------------------  normal_eqn  ---------------------------------------')
    theta = np.zeros((X.shape[1], 1))
    print(f'cost={cost(X, y, theta)} should be 32.072733877455676')
    theta = normal_eqn(X, y)
    print(f'theta={[float(t) for t in theta]} should be [ 89597.909542  139.210674  -8738.019112 ]')
    print(f'price={float(np.array([1, 1650, 3]) @ theta)}, should be 293081.464335')

    print('\n-------------------------------  regression with std normalize ---------------------------------------')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data2.txt')
    X, y = np.insert(data[:, :-1], 0, np.ones((data[:, :-1].shape[0]), dtype=data.dtype), axis=1), data[:, -1:]

    X, mu, sigma = normalize.standard_deviation(X)
    theta = np.zeros((X.shape[1], 1))
    J_history = simple_linear_regression(X, y, theta, alpha=10e-1, num_iter=100, batch=20)[1]
    print(f'theta={[float(t) for t in theta]}, should be [340412.659574 110631.050279 -6649.474271]')
    # predict
    x = np.array([1, 1650, 3], dtype=np.float64)
    x[1:] = (x[1:] - mu) / sigma
    print(f'price={float(x @ theta)}, should be 293081.464335')

    plt.plot(range(len(J_history)), J_history)
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()

    print('\n-------------------------------  regression with simple normalize ---------------------------------------')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data2.txt')
    X, y = np.insert(data[:, :-1], 0, np.ones((data[:, :-1].shape[0]), dtype=data.dtype), axis=1), data[:, -1:]

    X, max_, min_ = normalize.simple_normalize(X)
    theta = np.zeros((X.shape[1], 1))
    J_history = simple_linear_regression(X, y, theta, alpha=10e-1, num_iter=100, batch=15)[1]
    print(f'theta={[float(t) for t in theta]}, should be [199467.38469348644, 504777.9039879094, -34952.07644931053]')
    # predict
    x = np.array([1, 1650, 3], dtype=np.float64)
    x[1:] = (x[1:] - min_) / max_
    print(f'price={x @ theta}, should be 293081.464335')

    plt.plot(range(len(J_history)), J_history)
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()


# -----------------------------------------  debug function  --------------------------------------------

if __name__ == '__main__':
    """
    data for test:

    '/home/bb/Documents/python/ML/data/ex1data1.txt'
    '/home/bb/Documents/python/ML/data/ex1data2.txt' 
    '/home/bb/Documents/python/ML/data/ex2data1.txt' 
    '/home/bb/Documents/python/ML/data/ex2data2.txt'
    """

    # test_ex1data1()
    # test_ex1data2()
    # data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data1.txt')
    # X, y = np.insert(data[:, :-1], 0, np.ones((data[:, :-1].shape[0]), dtype=data.dtype), axis=1), data[:, -1:]
    # theta = np.zeros((X.shape[1], 1))
    # X, mu, sigma = normalize.standard_deviation(X)
    # print(f'cost={cost(X, y, theta)}')
    # J_history = simple_linear_regression(X, y, theta, alpha=10e-1, num_iter=100, batch=30)
    # print(np.array([[4, 9, 8], [-9, -8, -7]])[0:0,:])
