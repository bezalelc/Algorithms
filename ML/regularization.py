import numpy as np


def lasso_cost(theta):
    return np.sum(np.abs(theta))


def lasso_grad(theta):
    return np.sign(theta)


def ridge_cost(theta):
    return np.sum(theta ** 2)


def ridge_grad(theta):
    return theta
