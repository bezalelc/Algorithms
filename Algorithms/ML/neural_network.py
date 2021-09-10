import numpy as np
import scipy as sc
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd


# *******************************  activation  *****************************************
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.max(Z, 0)


def tanh_(Z):
    return np.tanh(Z)


# ***********************************  cost/grad  *****************************************
def cost(X, y, W, landa, activation):
    a = feedforward(W, X, activation)[-1]
    k, m = a.shape
    K = np.arange(k)
    pos = np.array(y == K[:, None])
    J = -(np.sum(np.log(a[pos])) + np.sum(np.log(1 - a[~pos])))
    J += (landa / 2) * np.sum([np.sum(w[:, 1:] ** 2) for w in W])
    J /= m
    return J


def grad(H, W, y, landa):
    m, k = y.shape[0], H[-1].shape[0]
    K = np.arange(k)
    delta = [H[-1] - np.array(y == K[:, None])]
    dW = [delta[0] @ H[-2].T]
    dW[0][1:, :] += landa * W[-1][1:, :]
    dW[0] /= m

    for h0, h1, w0, w1 in zip(H[:-2][::-1], H[1:-1][::-1], W[:-1][::-1], W[1:][::-1]):
        delta.insert(0, (w1.T[1:, :] @ delta[0]) * (h1[1:, :] * (1 - h1[1:, :])))
        dW.insert(0, delta[0] @ h0.T)
        dW[0][:, 1:] += landa * w0[:, 1:]
        dW[0] /= m

    return dW


# ***********************************  NN  *****************************************
def compile(layer_sizes, eps):
    W = []
    for l0, l1 in zip(layer_sizes[:-1], layer_sizes[1:]):
        W.append(np.random.rand(l1, l0 + 1) * 2 * eps - eps)
    return W


def feedforward(W, X, activation):
    H = [X.T]
    for w in W:
        H[-1] = np.insert(H[-1], 0, np.ones((H[-1].shape[1]), dtype=H[-1].dtype), axis=0)
        H.append(activation(w @ H[-1]))
    return H


def backpropagation(H, y, W, landa, alpha):
    dW = grad(H, W, y, landa)
    for w, dw in zip(W, dW):
        w -= alpha * dw
    return W


def predict(X, W, activation):
    h = feedforward(W, X, activation)[-1]
    p = np.argmax(h, axis=0)
    return p


# ***********************************  train  *****************************************
def fit(X, y, W, landa, max_iter, activation):
    for i in range(max_iter):
        H = feedforward(W, X, activation)
        W = backpropagation(H, y, W, landa, 0)
    return W
