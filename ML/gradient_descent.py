import numpy as np
from general import load_data, normalize
import matplotlib.pyplot as plt


# --------------------------------------  optimizers  ---------------------------------------------
def simple(X, y, theta, alpha):
    """
    simple optimizer for linear regression

    this function improve theta according to the derivative of cost(X)
        theta[i+1] -= alpha*J'(X*theta[i])

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        alpha: learning rate

    """
    theta -= alpha * grad(X, y, theta)


def momentum(X, y, theta, V, alpha, beta):
    """
    momentum optimizer for linear regression

    this function improve theta according to the derivative of cost(X)
        V[i] = beta*V[i-1] + alpha*J'(X*theta[i])
        theta[k+1] -= V[i]

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        alpha: learning rate
        V: the mean of the theta until now
        beta: should be between [0.8,1)

    """
    V[:] = beta * V + alpha * grad(X, y, theta)
    theta -= V


def ada_grad(X, y, theta, G, const, epsilon):
    """
    Ada Grad optimizer for linear regression

    this function improve theta according to the derivative of cost(X)
        G[i] += G[i]+J'(X*theta[i])^2
        ALPHA = const / np.sqrt(G + epsilon)
        theta[i+1] -= ALPHA*theta[i]

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        G: the sum of the all theta^2 history
        cost: <float>: const number, recommended range [10e+4,10e+20]
        epsilon: small number for the fot not divide in zero
    """
    G += grad(X, y, theta) ** 2
    ALPHA = const / np.sqrt(G + epsilon)
    theta -= ALPHA * grad(X, y, theta)


def adam(X, y, theta, V, G, beta1, beta2, const):
    """
    Adam optimizer for linear regression, this optimizer is a Combination of momentum and Ada Grad

    this function improve theta according to the derivative of cost(X)
        V = beta1 * V + (1 - beta1) * '(X*theta[i])
        G = beta2 * G + (1 - beta2) * '(X*theta[i])^2
        V, G = 1 / (1 - beta1) * V, 1 / (1 - beta2) * G
        ALPHA = const / np.sqrt(G)
        theta -= ALPHA * V


    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        V: the mean of the theta until now
        beta1: should be between [0.8,1)
        beta2: recommended around 0.999
        cost: <float>: const number, recommended range [10e+4,10e+20]
    """
    grad_ = grad(X, y, theta)
    V[:] = (beta1 * V + (1 - beta1) * grad_)
    G[:] = (beta2 * G + ((1 - beta2) * (grad_ ** 2)))
    # V[:], G[:] = (1 / (1 - beta1)) * V, (1 / (1 - beta2)) * G
    ALPHA = const / np.sqrt(G + 10e-8)
    theta -= ALPHA * V


# --------------------------------------  linear regression  ---------------------------------------------

def linear_regression(X, y, theta, alpha=10e-7, num_iter=1000, batch=30, optimizer=simple, optimizer_data=None):
    """
    linear regression

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        alpha: learning rate by default alpha=10e-7 for prevent Entertainment, recommended range: [1e-5,10]
        num_iter: number of the iteration , should be between (1000-10e+9)
        batch: number of example to compute at once, the recommended is 30
        optimizer: <string>: algorithm to compute theta, optional [simple,momentum,ada_grad,adam]
        optimizer_data: <dict>: data for the specified optimizer

    optional optimizers algorithm: [simple,momentum,ada_grad,adam]
        simple:
            theta[i+1] -= alpha*J'(X*theta[i])

            optional optimizer_data:
                alpha: learning rate

        momentum:
            V[i] = beta*V[i-1] + alpha*J'(X*theta[i])
            theta[k+1] -= V[i]

            optional optimizer_data:
                V: the mean of the theta until now
                alpha: learning rate
                beta: <number>: the Percentage of consideration in the previous gradient, recommended is 0.9

        ada_grad:
            G[i] += G[i]+J'(X*theta[i])^2
            ALPHA = const / np.sqrt(G + epsilon)
            theta[i+1] -= ALPHA*theta[i]

            optional optimizer_data:
                G: the sum of the all theta^2 history
                cost: <float>: const number, recommended range [10e+4,10e+20]
                epsilon: small number for the fot not divide in zero

        adam:
            V = beta1 * V + (1 - beta1) * '(X*theta[i])
            G = beta2 * G + (1 - beta2) * '(X*theta[i])^2
            V, G = 1 / (1 - beta1) * V, 1 / (1 - beta2) * G
            ALPHA = const / np.sqrt(G)
            theta -= ALPHA * V

            optional optimizer_data:
                V: the mean of the theta until now
                G: the sum of the all theta^2 history
                beta1: recommended is around 0.9
                beta2: recommended is around 0.999
                cost: <float>: const number, recommended range [10e+4,10e+20]


    :return:
        theta
        J_history: the history of the cost function

    :efficiency: O(batch*n^2)
    """

    optimizer_data, data = optimizer_data if optimizer_data else {}, []
    if optimizer is simple:
        """
        data=[alpha]
        """
        data.append(alpha)
    elif optimizer is momentum:
        """
        data=[V,alpha,beta]
        """
        data.append(optimizer_data['V'] if 'V' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['alpha'] if 'alpha' in optimizer_data else alpha)
        data.append(optimizer_data['beta'] if 'beta' in optimizer_data else 0.9)
    elif optimizer is ada_grad:
        """
        data=[G,const, epsilon]
        """
        data.append(optimizer_data['G'] if 'G' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['const'] if 'const' in optimizer_data else 10e+13)
        data.append(optimizer_data['epsilon'] if 'epsilon' in optimizer_data else 10e-8)
    elif optimizer is adam:
        """
        data=[V, G, beta1, beta2, const]
        """
        data.append(optimizer_data['V'] if 'V' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['G'] if 'G' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['beta1'] if 'beta1' in optimizer_data else 0.9)
        data.append(optimizer_data['beta2'] if 'beta2' in optimizer_data else 0.999)
        data.append(optimizer_data['const'] if 'const' in optimizer_data else 10e+12)

    J_history = []
    m, start = X.shape[0], 0
    for i in range(num_iter):
        start = (i * batch) % (m - 1) if start + batch < m - 1 else 0
        end = start + batch if start + batch < m - 1 else m - 1

        optimizer(X[start:end], y[start:end], theta, *data)
        J_history.append(cost(X[start:end], y[start:end], theta))
    return theta, J_history


def normal_eqn(X, y):
    """

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X

    :return:
        theta: vector of parameters for the feature

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


def grad(X, y, theta):
    """
    compute the gradient of cost function

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: vector of parameters for the feature

    :return: cost'(X)

    :efficiency: O(m*n + m*n^2 + n)
    """
    return (1 / X.shape[0]) * (X.T @ (X @ theta - y))


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
    J_history = linear_regression(X, y, theta, alpha=10e-3, num_iter=100000)[1]
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
    theta, J_history = linear_regression(X, y, theta, alpha=0.01, num_iter=10000, batch=X.shape[0],
                                         optimizer=simple)
    print(f'    theta={[float(t) for t in theta]}\nshould be [340412.659574 110631.050279 -6649.474271]')
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
    J_history = linear_regression(X, y, theta, alpha=10e-1, num_iter=1000, batch=15)[1]
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


def test_general(file, func, optimizer=simple, norm=normalize.standard_deviation):
    data = load_data.load_from_file(file)
    X, y = np.insert(data[:, :-1], 0, np.ones((data[:, :-1].shape[0]), dtype=data.dtype), axis=1), data[:, -1:]

    print('\n\n===================================== test ex1data2 =====================================')
    print('-------------------------------  regression with momentum---------------------------------------')

    theta = np.zeros((X.shape[1], 1))
    X, mu, sigma = normalize.standard_deviation(X)
    theta, J_history = func(X, y, theta, alpha=0.001, num_iter=1000, batch=30, optimizer=ada_grad)
    print(
        f'    theta={[float(t) for t in theta]}\nshould be [340412.659574 110631.050279 -6649.474271]')
    print(f'cost={cost(X, y, theta)}')

    # predict
    x = np.array([1, 1650, 3], dtype=np.float64)
    x[1:] = (x[1:] - mu) / sigma
    print(f'price={float(x @ theta)}, should be 293081.464335')

    # plot
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
    test_general('/home/bb/Documents/python/ML/data/ex1data2.txt', func=linear_regression)
    # print(np.array([1, 2, 3]) * np.array([1, 2, 3]))
