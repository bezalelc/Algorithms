import numpy as np


def simple(X, y, theta, alpha, grad):
    """
    simple optimizer for linear regression

    this function improve theta according to the derivative of cost(X)
        theta[i+1] -= alpha*J'(X*theta[i])

    :param
        X: matrix of dataset [x(0).T,x(1).T,...,x(m).T]
        y: real value for each example (=row) in X
        theta: init vector of parameters for the feature
        alpha: learning rate
        grad: function to calculate the gradient

    """
    # print(grad(X, y, theta).shape, alpha, theta.shape)
    # print(theta.shape,grad(X, y, theta).shape)
    theta -= alpha * grad(X, y, theta)  # .reshape(theta.shape)


def momentum(X, y, theta, V, alpha, beta, grad):
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


def ada_grad(X, y, theta, G, const, epsilon, grad):
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
    ALPHA = const / (np.sqrt(G) + epsilon)
    theta -= ALPHA * grad(X, y, theta)


def adam(X, y, theta, V, G, beta, const, grad):
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
        beta: <list>: (beta2, beta1): (recommended is around 0.9, recommended is around 0.999)
        cost: <float>: const number, recommended range [10e+4,10e+20]
    """
    grad_ = grad(X, y, theta)
    V[:] = (beta[0] * V + (1 - beta[0]) * grad_)
    G[:] = (beta[1] * G + ((1 - beta[1]) * (grad_ ** 2)))
    V = (V / (1 - beta[0]))
    G = (G / (1 - beta[1]))
    beta[0], beta[1] = beta[0] ** 2, beta[1] ** 2
    # ALPHA = const / (np.sqrt(G + 10e-8))  # V-m,b1=0.9 , G-v,b2=0.999
    # ALPHA = 10 / (np.sqrt(G) + 10e-8)  # V-m , G-v
    # theta -= const * V / np.sqrt(G + 10e-8)
    theta -= (1e+4) * V / (np.sqrt(G) + 10e-8)
