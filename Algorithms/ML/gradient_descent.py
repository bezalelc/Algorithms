import matplotlib.pyplot as plt
import numpy as np

import optimizer as opt
import regularization as reg
from general import load_data, normalize


# class Model():
#     # pass global_default = {'cost': None, 'grad': None, 'reg_cost': None, 'reg_grad': None, 'alpha': [1e-4, ],
#     #                   'compute_alpha': opt.compute_alpha_simple, 'beta': 0.9,
#     #                   'beta1': 0.9, 'beta2': 0.99, 'beta_t': np.array([0.9, 0.99]),
#     #                   'compute_beta_t': opt.compute_beta_simple, 'epsilon': 10e-7, 'lambda': 0, 'const': 10e+12,
#     #                   'limit_class': 0.5}
#     def __init__(self) -> None:
#         self.cost=None,self.grad=None


# --------------------------------------  normal eqn  ---------------------------------------------

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


# --------------------------------------  poly feature   ----------------------------------------------
def poly_feature(X, ploy_array):
    X_ = X.copy()
    for p in ploy_array:
        X_ = np.concatenate((X_, X ** p), axis=1)
    return X_


# --------------------------------------  linear  -----------------------------------------------------
def linear_cost(X, y, theta, reg=None, lambda_=0):
    """
    compute the cost of the range between X*theta and y

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: vector of parameters for the feature
        reg: <function>: function for regularization the cost
        lambda: limit the search area of theta

    :return: <float>: J cost of data x for the current theta

    :efficiency: O(m*n^2)
    """
    J = np.sum((X @ theta - y) ** 2)
    if reg:
        J += lambda_ * reg(theta[1:])
    return (1 / (2 * X.shape[0])) * J


def linear_grad(X, y, theta, reg=None, lambda_=0):
    """
    compute the gradient of cost function

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: vector of parameters for the feature
        reg: <function>: function for regularization the gradient
        lambda: limit the search area of theta

    :return: cost'(X)

    :efficiency: O(m*n + m*n^2 + n)
    """
    grad = (X.T @ (X @ theta - y))
    if reg:
        grad[1:] += lambda_ * reg(theta[1:])
    return (1 / X.shape[0]) * (X.T @ (X @ theta - y))


def h_theta(X, theta):
    return X @ theta


# --------------------------------------  classification  ---------------------------------------------
def sigmoid(X, theta):
    return 1 / (1 + np.exp(-(X @ theta)))


def class_cost(X, y, theta, reg=None, lambda_=0):
    """
    compute the cost of the range between sigmoid(X*theta) and y

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: vector of parameters for the feature
        reg: <function>: function for regularization the cost
        lambda: limit the search area of theta

    :return: <float>: J cost of data x for the current theta

    :efficiency: O(m*n^2)
    """
    m = X.shape[0]
    Z = sigmoid(X, theta)
    J = (-1 / m) * (np.sum(np.log(Z[y == 1])) + np.sum(np.log(1 - Z[y == 0])))
    if reg:
        J += (lambda_ / (2 * m)) * reg(theta[1:])  # regularization
    return J


def class_grad(X, y, theta, reg=None, lambda_=0):
    """
    compute the gradient of cost function

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: vector of parameters for the feature
        reg: <function>: function for regularization the gradient
        lambda: limit the search area of theta

    :return: cost'(X)

    :efficiency: O(m*n + m*n^2 + n)
    """
    m = X.shape[0]
    grad = (1 / m)
    Z = sigmoid(X, theta)
    grad *= (X.T @ (Z - y))
    if reg:
        grad[1:] += (lambda_ / m) * reg(theta[1:])  # regularization gradient
    return grad


def one_vs_all(X, y):
    """
    classification one vs all with k classes

    :param X: training examples
    :param y: class of training example

    :return: theta vector with shape (n x k) wile n=number of attributes, k=number of classes

    :efficiency:
    """
    k = np.array(np.unique(y), dtype=np.uint8)
    Y = (y == k)
    theta = np.zeros((X.shape[1], k.shape[0]))
    theta, J = regression(X, Y, theta, class_grad, cost=class_cost, num_iter=1000, optimizer=opt.simple,
                          batch=X.shape[0])
    return theta, J


def one_vs_one(X, Y):
    """
    classification one vs one with k classes

    :param X: training examples
    :param y: class of training example

    :return: matrix of theta vectors with shape () wile n=number of attributes, k=number of classes

    :efficiency: O((k-1)^2*)  when k is number of classes
    """
    k = np.array(np.unique(Y), dtype=np.uint8)

    X_all, Y_all = [], []
    THETA = np.zeros((X.shape[1], (len(k) * (len(k) - 1)) // 2))
    # THETA = np.array([[3.180897341514672, 3.0838649814633614, -2.2265349457136954],
    #                   [-5.862294924547381, 4.222990185993943, 11.446614903798233],
    #                   [-2.658228735054151, -0.21796094330740157, -9.165301573076924],
    #                   [5.872049876694517, 3.420824982942488, -2.085844052882422],
    #                   [41.73943900102861, 21.179228027921013, -2.4104326262001194],
    #                   [15.267902977220565, -11.109668557210345, -3.246111842955075],
    #                   [-2.275997083921752, -1.6343439648746385, -0.9993453478703064],
    #                   [-28.841273581604323, -25.7485269040619, -0.12571893926167918]]
    #                  )

    for clas in k:
        idx = np.where(Y == clas)
        X_all.append(X[idx[0]]), Y_all.append(Y[idx[0]])

    r = 0
    for i in range(len(k) - 1):
        for j in range(i + 1, len(k)):
            x, y = X_all[i], Y_all[i] == k[i]
            x, y = np.insert(x, x.shape[0], X_all[j], axis=0), np.insert(y, x.shape[0], Y_all[j] == k[i], axis=0)
            t, J = regression(x, y, THETA[:, r].reshape((THETA.shape[0], 1)), class_grad, cost=class_cost,
                              num_iter=100, optimizer_data={'alpha': 0.5},
                              optimizer=opt.momentum,
                              batch=x.shape[0])
            THETA[:, r] = t.reshape((t.shape[0],))
            r += 1

            # cost
            # print(f'{i} vs {j}: cost={J[-1]}')

            # plot
            # plt.plot(range(len(J)), J)
            # plt.xlabel(xlabel='iter number')
            # plt.ylabel(ylabel='cost')
            # plt.title(f'regression class {i} vs {j}')
            # plt.show()

    # print(theta_.tolist())

    predict_one_vs_one(X, Y, THETA)
    return THETA


def predict_one_vs_one(X, Y, THETA):
    p = sigmoid(X, THETA)
    p = np.concatenate((p, 1 - p), axis=1)
    res = np.argmax(p, axis=1)
    k = np.unique(Y)

    def func_idx(x):
        midd = (len(k) * (len(k) - 1)) // 2
        if x < midd:
            return np.floor(np.roots([-1, 1 + 2 * (len(k) - 1), -2 * x])[1])
        else:
            n = np.floor(np.roots([-1, 1 + 2 * (len(k) - 1), -2 * (x - midd)])[1]) + 1
            return n + (x - midd) - (n - 1) * (2 * (len(k) - 1) - (n - 2)) / 2

    func = np.vectorize(func_idx)
    res = func(res)
    res = (res == (Y.reshape((Y.shape[0],)) - 1))
    print(np.mean(res))
    return res


# --------------------------------------  softmax  ------------------------------------------------
def softmax_cost(X, y, theta):
    # P = sigmoid(X, theta)
    Z = np.exp(X @ theta)
    J = Z * (1 / np.sum(Z, axis=0))
    idx = np.where(y != 0)
    return -np.sum(np.log(J[idx]))


def softmax_grad(X, y, theta):
    x = X[0]
    # print(y.shape)
    Z = np.exp(X @ theta)
    soft = Z * (1 / np.sum(Z, axis=0))

    print(soft.shape)
    print(soft)
    # dJ = np.diag(Z) - np.outer(Z, Z)
    dJ = np.diag(Z) - Z.T @ Z
    dJ /= X.shape[0]
    # print(dJ.shape)
    # print(theta @ dJ)
    # print(Z.shape)
    # print(Z)
    # print((np.diag(Z) - np.outer(Z, Z)))
    # A=np.diag(Z) - np.outer(Z, Z)
    # return A@theta
    pass


def softmax(X, y):
    # X_all, Y_all = [], []
    K = np.array(np.unique(y), dtype=np.uint8)
    m, n, k = X.shape[0], X.shape[1], K.shape[0]

    THETA = np.random.random((n, k))
    # print(K.dtype, y.dtype)
    Y = (y == K)
    # J = softmax_cost(X, Y, THETA)
    # dJ = softmax_grad(X, y, THETA)
    softmax_grad(X, y, THETA)
    # print(dJ)
    # THETA, J = regression(X, y, THETA, num_iter=1000, batch=X.shape[0], optimizer=simple, alpha=0.0001, grad=class_grad,
    #                       cost=class_cost)

    # return THETA, J


# --------------------------------------  regression  ---------------------------------------------

# global vars for regression
global_default = {'cost': None, 'grad': None, 'reg_cost': None, 'reg_grad': None, 'alpha': [1e-4, ],
                  'compute_alpha': opt.compute_alpha_simple, 'beta': 0.9,
                  'beta1': 0.9, 'beta2': 0.99, 'beta_t': np.array([0.9, 0.99]),
                  'compute_beta_t': opt.compute_beta_simple, 'epsilon': 10e-9, 'lambda': 0, 'const': 10e+12,
                  'limit_class': 0.5}


def predict(theta, x, data=None):
    x = np.array(x, dtype=np.float128)
    x = np.insert(x, 0, [1, ], axis=0)
    if not data:
        return theta @ x
    elif 'normalize' in data:
        if data['normalize'] is normalize.standard_deviation:
            x[1:] = (x[1:] - data['mu']) / data['sigma']
            return x @ theta
        elif data['normalize'] is normalize.simple_normalize:
            x[1:] = (x[1:] - data['min']) / data['max']
            return x @ theta
    else:
        print('need to specified normalize function')


def regression(X, y, theta, grad, cost=None, num_iter=1000, batch=30, optimizer=opt.simple,
               optimizer_data=None):
    """
    linear regression

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        alpha: learning rate by default alpha=10e-7 for prevent Entertainment, recommended range: [1e-5,10]
        num_iter: number of the iteration , should be between (1000-10e+9)
        batch: number of example to compute at once, the recommended is 30
        optimizer: <string_>: algorithm to compute theta, optional [simple,momentum,ada_grad,adam]
        optimizer_data: <dict>: data for the specified optimizer

    optional optimizers algorithm: [simple,momentum,ada_grad,adam]
        simple:
            theta[i+1] -= alpha*J'(X*theta[i])

            optional optimizer_data:
                alpha: learning rate
                lambda:
                grad: function to calculate the gradient

        momentum:
            V[i] = beta*V[i-1] + alpha*J'(X*theta[i])
            theta[k+1] -= V[i]

            optional optimizer_data:
                V: the mean of the theta until now
                alpha: learning rate
                beta: <number>: the Percentage of consideration in the previous gradient, recommended is 0.9
                grad: function to calculate the gradient

        ada_grad:
            G[i] += G[i]+J'(X*theta[i])^2
            ALPHA = const / np.sqrt(G + epsilon)
            theta[i+1] -= ALPHA*theta[i]

            optional optimizer_data:
                G: the sum of the all theta^2 history
                cost: <float>: const number, recommended range [10e+4,10e+20]
                epsilon: small number for the fot not divide in zero
                grad: function to calculate the gradient

        adam:
            V = beta1 * V + (1 - beta1) * J'(X*theta[i])
            G = beta2 * G + (1 - beta2) * J'(X*theta[i])^2
            V, G = 1 / (1 - beta1) * V, 1 / (1 - beta2) * G
            ALPHA = const / np.sqrt(G)
            theta -= ALPHA * V

            optional optimizer_data:
                V: the mean of the theta until now
                G: the sum of the all theta^2 history
                beta: <list>: (beta2, beta1): (recommended is around 0.9, recommended is around 0.999)
                cost: <float>: const number, recommended range [10e+4,10e+20]
                grad: function to calculate the gradient


    :return:
        theta
        J_history: the history of the cost function

    :efficiency: O(num_iter*batch*n^2)
    """

    optimizer_data, data = {**global_default, **optimizer_data} if optimizer_data else {**global_default}, []

    if optimizer is opt.simple:
        """
        data=[alpha, grad, reg, lambda_]
        """
        data.append(optimizer_data['alpha'])
        data.append(optimizer_data['compute_alpha'])
        data.append(grad)
        data.append(optimizer_data['reg_grad'])  # regularization gradient function
        data.append(optimizer_data['lambda'])
    elif optimizer is opt.momentum or optimizer is opt.momentum_w:
        """
        data=[V, alpha, beta, grad, reg, lambda_]
        """
        data.append(optimizer_data['V'] if 'V' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['alpha'])
        data.append(optimizer_data['compute_alpha'])
        data.append(optimizer_data['beta'] if 'beta' in optimizer_data else optimizer_data['beta1'])
        data.append(grad)
        data.append(optimizer_data['reg_grad'])  # regularization gradient function
        data.append(optimizer_data['lambda'])
    elif optimizer is opt.ada_grad:
        """
        data=[G, const, epsilon, grad, reg, lambda_]
        """
        data.append(optimizer_data['G'] if 'G' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['const'] if 'const' in optimizer_data else 10e+13)
        data.append(optimizer_data['epsilon'])
        data.append(grad)
        data.append(optimizer_data['reg_grad'])  # regularization gradient function
        data.append(optimizer_data['lambda'])
    elif optimizer is opt.adam or optimizer is opt.adam_w:
        """
        data=[V, G, alpha, compute_alpha, beta, beta_t, compute_beta_t, epsilon, grad, reg, lambda_]
        """
        data.append(optimizer_data['V'] if 'V' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['G'] if 'G' in optimizer_data else np.zeros(theta.shape))
        data.append(optimizer_data['alpha'])
        data.append(optimizer_data['compute_alpha'])
        data.append([optimizer_data['beta1'], optimizer_data['beta2']])
        data.append(optimizer_data['beta_t'])
        data.append(optimizer_data['compute_beta_t'])
        data.append(optimizer_data['epsilon'])
        data.append(grad)
        data.append(optimizer_data['reg_grad'])  # regularization gradient function
        data.append(optimizer_data['lambda'])

    J_history = []
    m, start = X.shape[0], 0
    for i in range(num_iter):
        start = (i * batch) % (m - 1) if start + batch < m - 1 else 0
        end = start + batch if start + batch < m - 1 else m - 1
        # print(X[start:end].shape,y[start:end].shape,theta.shape)
        # print(start, end)
        optimizer(X[start:end], y[start:end], theta, *data)
        if cost:
            J_history.append(
                cost(X[start:end], y[start:end], theta, optimizer_data['reg_cost'], optimizer_data['lambda']))
    return (theta, J_history) if cost else theta


# -----------------------------------------  debug function  --------------------------------------------
def test_ex1data1():
    print('===================================== test ex1data1 =====================================')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data1.txt')
    X, y = np.insert(data[:, :-1], 0, np.ones((data[:, :-1].shape[0]), dtype=data.dtype), axis=1), data[:, -1:]

    print('\n-------------------------------  iter on ex1data1.txt  ---------------------------------------')
    theta = np.zeros((X.shape[1], 1))
    # print(theta.shape)
    print(f'cost={linear_cost(X, y, theta, 0)} should be 32.072733877455676')
    theta = normal_eqn(X, y)
    print(f'theta={[float(t) for t in theta]} should be [-3.89578088, 1.19303364]')
    print(f'cost={linear_cost(X, y, theta, 0)} should be 4.476971375975179 ')
    print('mean theta error iter=', np.mean(np.abs(h_theta(X, theta) - y)), 'should be 2.1942453988270043')
    # print('error in octave=', np.mean(np.abs(h_theta(X, np.array([-3.6303, 1.1664])) - y)))
    # print('predict in octave=',h_theta(np.array([1, 7]), np.array([-3.6303, 1.1664])) * 10000)

    print('\n-------------------------------  regression  ---------------------------------------')
    theta = np.zeros((X.shape[1], 1))
    theta, J_history = regression(X, y, theta, linear_grad, optimizer_data={'alpha': [1e-2, ]}, num_iter=1000,
                                  batch=X.shape[0],
                                  cost=linear_cost)
    print(f'theta={[float(t) for t in theta]}')
    print(f'cost={linear_cost(X, y, theta)}')

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
    print(f'cost={linear_cost(X, y, theta)}')
    theta = normal_eqn(X, y)
    print(f'theta={[float(t) for t in theta]} should be ]')
    print(f'price={float(np.array([1, 1650, 3]) @ theta)}, should be 293081.464335')
    print(f'cost={linear_cost(X, y, theta)}')
    print(
        '\n-------------------------------  regression with std normalize ---------------------------------------')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data2.txt')
    X, y = data[:, :-1], data[:, -1:]
    X, mu, sigma = normalize.standard_deviation(X)
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    theta, J_history = regression(X, y, theta, linear_grad, optimizer_data={' alpha': 0.01}, num_iter=10000,
                                  batch=X.shape[0],
                                  optimizer=opt.simple, cost=linear_cost)
    theta_test = normal_eqn(X, y)
    print(f'theta={[float(t) for t in theta]}\n real={[float(t) for t in theta_test]}\n')
    print(f'cost={linear_cost(X, y, theta)}\nreal={linear_cost(X, y, theta_test)}')
    # predict
    x = np.array([1, 1650, 3], dtype=np.float64)
    x[1:] = (x[1:] - mu) / sigma
    print(f'price={float(x @ theta)}, should be 293081.464335')

    plt.plot(range(len(J_history)), J_history)
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()

    print(
        '\n-------------------------------  regression with simple normalize ---------------------------------------')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex1data2.txt')
    X, y = data[:, :-1], data[:, -1:]
    X, max_, min_ = normalize.simple_normalize(X)
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    theta = np.zeros((X.shape[1], 1))
    theta, J_history = regression(X, y, theta, linear_grad, optimizer_data={'alpha': [10e-1, ]}, num_iter=10000,
                                  batch=X.shape[0],
                                  cost=linear_cost)
    print(f'theta={[float(t) for t in theta]}\nreal= {[float(t) for t in normal_eqn(X, y)]}')
    # predict
    x = np.array([1650, 3], dtype=np.float64)
    x = (x - min_) / max_
    x = np.insert(x, 0, [1, ], axis=0)
    print(f'price={x @ theta}, should be 293081.464335')

    plt.plot(range(len(J_history)), J_history)
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()


def test_general(file, func):
    data = load_data.load_from_file(file)
    X, y = data[:, :-1], data[:, -1:]

    print('\n\n===================================== test ex1data2 =====================================')
    print('-------------------------------  regression with momentum---------------------------------------')
    X, mu, sigma = normalize.standard_deviation(X)
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    theta = np.zeros((X.shape[1], 1))
    theta_test = normal_eqn(X, y)
    data = {'alpha': [0.4, ], 'lambda': 1, 'reg_cost': reg.ridge_cost, 'reg_grad': reg.ridge_grad}
    theta, J_history = func(X, y, theta, linear_grad, optimizer_data=data, num_iter=1000, batch=30,
                            optimizer=opt.simple,
                            cost=linear_cost)
    print(f'theta={[float(t) for t in theta]}\n real={[float(t) for t in theta_test]}\n')
    print(f'cost={linear_cost(X, y, theta, reg.ridge_cost)}\nreal={linear_cost(X, y, theta_test, reg.ridge_cost)}')

    # predict
    x = np.array([1650, 3], dtype=np.float64)
    x = (x - mu) / sigma
    x = np.insert(x, 0, [1, ])
    print(f'price={float(x @ theta)}, should be 293081.464335')
    print('p=',
          predict(theta, [1650, 3], data={'normalize': normalize.standard_deviation, 'mu': mu, 'sigma': sigma}))

    # plot
    plt.plot(range(len(J_history)), J_history)
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()


def test_ex2data1():
    print('\n\n===================================== test ex1data2 =====================================')
    print('-------------------------------  regression with classification------------------------------')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex2data1.txt')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    # X, mu, sigma = normalize.standard_deviation(X)
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    theta = np.zeros((X.shape[1], 1))
    # theta = np.array([-24, 0.2, 0.2]).reshape((X.shape[1], 1))
    # print(theta.shape)
    # theta = np.array([-25.161, 0.206, 0.201]).reshape((X.shape[1], 1))
    # print(class_cost(X, y, theta))
    # print(class_grad(X, y, theta))
    theta = np.array([-25.06116393, 0.2054152, 0.2006545]).reshape((X.shape[1], 1))
    theta, J = regression(X, y, theta, class_grad, optimizer_data={'alpha': [0.0000002, ]}, num_iter=10000,
                          cost=class_cost,
                          optimizer=opt.momentum,
                          batch=X.shape[0])
    print(theta)
    print(J[-1:])
    print(np.mean(np.round(sigmoid(X, theta)) == y))

    # predict
    x = np.array([1, 45, 85])
    # x[1:] = (x[1:] - mu) / sigma
    p = (1 / (1 + np.exp(-x @ theta)))
    print(p)

    # plot
    # plt.plot(range(len(J[-30:])), J[-30:])
    # plt.xlabel(xlabel='iter number')
    # plt.ylabel(ylabel='cost')
    # plt.title('regression')
    # plt.show()

    # pos, neg = np.where(y == 1), np.where(y == 0)
    # Z = np.round(sigmoid(X, theta))
    # T, F = np.where((y == Z and y == 1)), np.where(y != Z)

    # print(X)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X[true_pos, 1], X[true_pos, 2], marker="*", color='g')
    # # plt.legend('pass')
    # plt.scatter(X[true_neg, 1], X[true_neg, 2], marker="o", color='y')
    # plt.scatter(X[err_pos, 1], X[err_pos, 2], marker="-", color='r')
    # plt.scatter(X[err_neg, 1], X[err_neg, 2], marker="+", color='r')
    # plt.legend('failed')
    # Z = sigmoid(X, theta)*100
    # # print(Z * 100)

    # plt.plot()
    # plt.xlabel(xlabel='exam 1')
    # plt.ylabel(ylabel='exan 2')
    # plt.title('regression')
    # plt.show()

    # print('-------------------------------  regression with regulation------------------------------')
    # data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex2data1.txt')
    # np.random.shuffle(data)
    # X, y = data[:, :-1], data[:, -1:]
    # # X, mu, sigma = normalize.standard_deviation(X)
    # X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    # theta = np.zeros((X.shape[1], 1))
    # # theta = np.array([-24, 0.2, 0.2]).reshape((X.shape[1], 1))
    # # print(theta.shape)
    # # theta = np.array([-25.161, 0.206, 0.201]).reshape((X.shape[1], 1))
    # # print(class_cost(X, y, theta))
    # # print(class_grad(X, y, theta))
    # theta = np.array([-25.06116393, 0.2054152, 0.2006545]).reshape((X.shape[1], 1))
    # theta, J = regression(X, y, theta, alpha=0.0000002, num_iter=100000, cost=class_cost, optimizer=momentum,
    #                       batch=X.shape[0], grad=class_grad)
    # print(theta)
    # print(J[-1:])
    # print(np.mean(np.round(sigmoid(X, theta)) == y))
    #
    # # predict
    # x = np.array([1, 45, 85])
    # # x[1:] = (x[1:] - mu) / sigma
    # p = (1 / (1 + np.exp(-x @ theta)))
    # print(p)
    #
    # # plot
    # plt.plot(range(len(J[-30:])), J[-30:])
    # plt.xlabel(xlabel='iter number')
    # plt.ylabel(ylabel='cost')
    # plt.title('regression')
    # plt.show()


def test_ex2data2():
    print('\n\n===================================== test ex2data2 =====================================')
    print('-------------------------------  regression with classification------------------------------')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex2data2.txt')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    p = np.arange(36)
    X = poly_feature(X, p)
    X, mu, sigma = normalize.standard_deviation(X)
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    theta = np.zeros((X.shape[1], 1))
    # print(X)
    # theta = np.array([-24, 0.2, 0.2]).reshape((X.shape[1], 1))
    # print(theta.shape)
    # theta = np.array([-25.161, 0.206, 0.201]).reshape((X.shape[1], 1))
    # print(class_cost(X, y, theta))
    # print(class_grad(X, y, theta))
    # theta = np.array([-25.06116393, 0.2054152, 0.2006545]).reshape((X.shape[1], 1))
    data_opt = {'alpha': [0.000001, ], 'lambda': 1, 'reg_cost': reg.ridge_cost, 'reg_grad': reg.ridge_grad}
    # data_opt = {'alpha': 0.0001}
    theta, J = regression(X, y, theta, class_grad, num_iter=100, cost=class_cost, optimizer=opt.adam,
                          batch=X.shape[0], optimizer_data=data_opt)  # , optimizer_data=data_opt
    print(theta)
    print('cost=', J[-1:])
    print('accuracy=', np.mean(np.round(sigmoid(X, theta)) == y))

    # predict
    # x = poly_feature(np.array([45, 85]), p)
    # x = np.insert(x, 0, [1, ])
    # x[1:] = (x[1:] - mu) / sigma
    # p = (1 / (1 + np.exp(-x @ theta)))
    # print(p)

    # plot
    plt.plot(range(len(J[-30:])), J[-30:])
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()


def test_stars():
    data = load_data.load_from_file('/home/bb/Downloads/data/archive2/6 class ready.csv')


def test_seeds():
    print('\n\n===================================== test seeds =====================================')
    data = load_data.load_from_file('/home/bb/Downloads/data/seeds_dataset.txt', delime='\t')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    k = np.array(np.unique(y), dtype=np.uint8)
    Y = (y == k)
    # print(k.dtype, y.dtype)

    print('-------------------------------  classification k-classes  ------------------------------')
    theta = np.zeros((X.shape[1], k.shape[0]))
    # theta = np.array([[-1.6829658687213866, -1.8543754265161039, 0.7754343585719218],
    #                   [-0.9215084632150828, 4.521276688115852, -4.004501938604477],
    #                   [2.648370316571783, -3.313060949916832, 0.16063859934981728],
    #                   [-0.05307786295000424, -2.4615062595012542, 1.3240942639424873],
    #                   [16.15559358997315, -11.585899548679317, -4.528678388339358],
    #                   [0.48474185667614517, -5.932838971987173, 4.974046769850393],
    #                   [-1.0259316117512303, 0.8043956856099687, 1.2596367701574402],
    #                   [-20.999903530911624, 11.891520812942074, 10.235602620923723]]
    #                  )

    data = {'alpha': [0.002], 'cost': class_cost, 'grad': class_grad, 'reg_cost': reg.ridge_cost,
            'reg_grad': reg.ridge_grad,
            'compute_alpha': opt.compute_alpha_simple, 'beta': 0.9,
            'beta1': 0.9, 'beta2': 0.99, 'beta_t': np.array([0.9, 0.99]),
            'compute_beta_t': opt.square_beta, 'epsilon': 10e-9, 'lambda': 1, 'const': 10e+12,
            'limit_class': 0.5}
    theta, J = regression(X, Y, theta, class_grad, cost=class_cost, num_iter=100, optimizer_data=data,
                          optimizer=opt.adam_w, batch=X.shape[0])
    print(J[0], J[-1:])
    print(theta.tolist())

    # plot
    plt.plot(range(len(J)), J)
    plt.xlabel(xlabel='iter number')
    plt.ylabel(ylabel='cost')
    plt.title('regression')
    plt.show()

    # print error
    res = np.array((np.round(np.argmax(sigmoid(X, theta), axis=1) + 1))).reshape((y.shape)) == y
    print(np.mean(res))
    print('accuracy=', np.mean(np.round(sigmoid(X, theta)) == Y))


def test_seeds_one_vs_one():
    print('\n\n===================================== test seeds =====================================')
    data = load_data.load_from_file('/home/bb/Downloads/data/seeds_dataset.txt', delime='\t')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    one_vs_one(X, y)
    # J, theta = one_vs_one(X, y)


def test_seeds_softmax():
    print('\n\n===================================== test seeds =====================================')
    data = load_data.load_from_file('/home/bb/Downloads/data/seeds_dataset.txt', delime='\t')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)

    classes = np.array(np.unique(y), dtype=np.uint8)
    # K = np.arange(classes.shape[0])
    # K_dict = {clas: k for clas, k in zip(classes, K)}
    # print(y.dtype)
    # y_ = np.array([K_dict[item] for item in y.reshape(y.shape[0], )], dtype=y.dtype)
    # print(y.dtype)
    softmax(X, y)

    # T, J = softmax(X, y)


# -----------------------------------------  main  --------------------------------------------

if __name__ == '__main__':
    """
    data for test:
    # linear
    '/home/bb/Documents/python/ML/data/ex1data1.txt'
    '/home/bb/Documents/python/ML/data/ex1data2.txt'
    #classification {0,1} k=2
    '/home/bb/Documents/python/ML/data/ex2data1.txt' 
    '/home/bb/Documents/python/ML/data/ex2data2.txt'
    #classification {0,5} k=6
    '/home/bb/Downloads/data/archive2/6 class ready.csv'
    #classification {0,1,2} k=3
    '/home/bb/Downloads/data/seeds_dataset.txt'
    """

    test_ex1data1()
    test_ex1data2()
    test_general('/home/bb/Documents/python/ML/data/ex1data2.txt', func=regression)
    test_ex2data1()
    test_ex2data2()
    test_seeds()
    # test_seeds_one_vs_one()
    # test_seeds_softmax()

    # Z = np.array([0.97337094, 0.85251098, 0.62495691, 0.63957056, 0.6969253])
    # S = np.array([0.16205871, 0.15308274, 0.22661096, 0.22634251, 0.23190508])
    # # Z = np.arange(9).reshape((3, 3))
    # # print(Z)
    # print(S)
    #
    # soft = np.reshape(S, (1, -1))
    # print(soft.shape)
    # # print(np.diag(Z))
    #
    # S_vector = S.reshape(S.shape[0], 1)
    # # print(S_vector.shape, S.shape)
    #
    # print(np.diag(S) - np.outer(S, S))
    # S_matrix = np.tile(S_vector, S.shape[0])
    # print()

    # np.where(a < limit, np.floor(a), np.ceil(a))
