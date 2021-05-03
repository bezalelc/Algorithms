import numpy as np
import scipy as sc
from scipy import io
import matplotlib.pyplot as plt


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
    dW[0][1:, :] += W[-1][1:, :]
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


if __name__ == '__main__':
    # init data/vars
    data = sc.io.loadmat('/home/bb/Documents/octave/week5/machine-learning-ex4/ex4/ex4data1.mat')
    X, y = data['X'], data['y']
    # labels = np.unique(y)
    # classes = {i: label for i, label in zip(range(len(labels)), labels)}
    y += 9
    y %= 10
    y = y.reshape((-1,))
    theta = sc.io.loadmat('/home/bb/Documents/octave/week5/machine-learning-ex4/ex4/ex4weights.mat')
    W = [theta['Theta1'], theta['Theta2']]

    # while True:
    #     plt.imshow(X[np.random.randint(5000), ::].reshape((20, 20)), cmap='gray')
    #     plt.colorbar()
    #     plt.show()
    # network_sizes = [X.shape[1], 70, 80, 90, 100, 90, 80, 70, 50, 40, 30, 25, 25, 25, 10]
    # W = compile(network_sizes, 0.12)
    W = np.load('/home/bb/Downloads/W0.npy', allow_pickle=True)
    # H = feedforward(W, X, sigmoid)
    # cost(X, y, W, 1, sigmoid)
    # grad(H, W, y, 0)
    landa = 1
    alpha = 1  # 0.001

    J, W_history = [], []
    for w in W:
        print(w.shape)

    print('before')
    print(cost(X, y, W, landa, sigmoid))
    p = predict(X, W, sigmoid)
    print(np.mean(p == y), '\n')
    for i in range(3):
        H = feedforward(W, X, sigmoid)
        W = backpropagation(H, y, W, landa, alpha)
        J.append(cost(X, y, W, landa, sigmoid))
    print('after')
    p = predict(X, W, sigmoid)
    # print(np.mean(p == y))
    # print(np.mean(p == y))
    # fit(X, y, W, landa, 34, sigmoid)
    plt.plot(range(len(J)), J)
    plt.show()
    print(J[-1])
    p = predict(X, W, sigmoid)
    print(np.mean(p == y))
    print(p)
    print(y)
    np.save('/home/bb/Downloads/W0.npy', W, allow_pickle=True)

    # dW = grad(H, W, y, landa)
    # import copy
    #
    # W0, W1, W2 = copy.deepcopy(W), copy.deepcopy(W), copy.deepcopy(W)
    # for i in range(len(W)):
    #     W0[i] -= dW[i]
    #     W2[i] += dW[i]
    #
    # J.append(cost(X, y, W0, landa, sigmoid))
    # J.append(cost(X, y, W1, landa, sigmoid))
    # J.append(cost(X, y, W2, landa, sigmoid))
    # print((J[2] - J[0]) / (2))
    # print(J[1])
