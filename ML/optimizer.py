"""
    optional optimizers algorithm: [simple,momentum,ada_grad,adam]
        simple:
            theta[i+1] -= alpha*J'(X*theta[i])

            optional optimizer_data:
                alpha: learning rate
                grad: function to calculate the gradient
                reg: <function>: function for regularization the gradient
                lambda: limit the search area of theta

        momentum:
            V[i] = beta*V[i-1] + alpha*J'(X*theta[i])
            theta[k+1] -= V[i]

            optional optimizer_data:
                V: the mean of the theta until now
                alpha: learning rate
                beta: <number>: the Percentage of consideration in the previous gradient, recommended is 0.9
                grad: function to calculate the gradient
                reg: <function>: function for regularization the gradient
                lambda: limit the search area of theta

        ada_grad:
            G[i] += G[i]+J'(X*theta[i])^2
            ALPHA = const / np.sqrt(G + epsilon)
            theta[i+1] -= ALPHA*theta[i]

            optional optimizer_data:
                G: the sum of the all theta^2 history
                cost: <float>: const number, recommended range [10e+4,10e+20]
                epsilon: small number for the fot not divide in zero
                grad: function to calculate the gradient
                reg: <function>: function for regularization the gradient
                lambda: limit the search area of theta

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
                reg: <function>: function for regularization the gradient
                lambda: limit the search area of theta

"""
import numpy as np


def compute_beta_simple(beta_t):
    return beta_t


def square_beta(beta_t):
    return beta_t ** 2


def compute_alpha_simple(alpha):
    return alpha


def simple(X, y, theta, alpha, compute_alpha, grad, reg, lambda_):
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
        reg: <function>: function for regularization the gradient
        lambda: limit the search area of theta
    """
    alpha[0] = compute_alpha(alpha[0])
    theta -= alpha[0] * grad(X, y, theta, reg, lambda_)  # .reshape(theta.shape)


def momentum(X, y, theta, V, alpha, compute_alpha, beta, grad, reg, lambda_):
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
        grad: <function>: function for compute the gradient
        reg: <function>: function for regularization the gradient
        lambda: limit the search area of theta

    """
    V[:] = beta * V + alpha[0] * grad(X, y, theta, reg, lambda_)
    alpha[0] = compute_alpha(alpha[0])
    theta -= V


def momentum_w(X, y, theta, V, alpha, compute_alpha, beta, grad, reg, lambda_):
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
        grad: <function>: function for compute the gradient
        reg: <function>: function for regularization the gradient
        lambda: limit the search area of theta

    """
    V[:] = beta * V + alpha[0] * grad(X, y, theta, reg, lambda_)
    alpha[0] = compute_alpha(alpha[0])
    theta -= (V + (lambda_ / X.shape[0]) * reg(theta))


def ada_grad(X, y, theta, G, const, epsilon, grad, reg, lambda_):
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
        grad: <function>: function for compute the gradient
        reg: <function>: function for regularization the gradient
        lambda: limit the search area of theta
    """
    G += grad(X, y, theta, reg, lambda_) ** 2
    ALPHA = const / (np.sqrt(G) + epsilon)
    theta -= ALPHA * grad(X, y, theta, reg, lambda_)


def adam(X, y, theta, V, G, alpha, compute_alpha, beta, beta_t, compute_beta_t, epsilon, grad, reg, lambda_):
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
        lambda: limit the search area of theta
        cost: <float>: const number, recommended range [10e+4,10e+20]
        grad: <function>: function for compute the gradient
        reg: <function>: function for regularization the gradient
    """
    grad_ = grad(X, y, theta, reg, lambda_)
    V[:] = (beta[0] * V + (1 - beta[0]) * grad_)
    G[:] = (beta[1] * G + ((1 - beta[1]) * (grad_ ** 2)))
    V_ = (V / (1 - beta_t[0]))
    G_ = (G / (1 - beta_t[1]))
    alpha[0], beta_t[:] = compute_alpha(alpha[0]), compute_beta_t(beta_t)
    theta -= alpha[0] * V_ / (np.sqrt(G_) + epsilon)


def adam_w(X, y, theta, V, G, alpha, compute_alpha, beta, beta_t, compute_beta_t, epsilon, grad, reg, lambda_):
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
        lambda: limit the search area of theta
        cost: <float>: const number, recommended range [10e+4,10e+20]
        grad: <function>: function for compute the gradient
        reg: <function>: function for regularization the gradient
    """
    grad_ = grad(X, y, theta)
    V[:] = (beta[0] * V + (1 - beta[0]) * grad_)
    G[:] = (beta[1] * G + ((1 - beta[1]) * (grad_ ** 2)))
    V_ = (V / (1 - beta_t[0]))
    G_ = (G / (1 - beta_t[1]))
    alpha[0], beta_t[:] = compute_alpha(alpha[0]), compute_beta_t(beta_t)
    theta -= (alpha[0] * V_ / (np.sqrt(G_) + epsilon) + (lambda_ / X.shape[0]) * reg(theta))
