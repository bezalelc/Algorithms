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


if __name__ == '__main__':
    df = pd.read_csv('/home/bb/Downloads/data/iris.data')
    from scipy.stats import zscore

    z = np.abs(zscore(df.iloc[:, :4]))
    z = np.abs(zscore(df.iloc[:, 1]))

    median = df.iloc[np.where(z <= 3)[0], 1].median()
    df.iloc[np.where(z > 3)[0], 1] = np.nan
    df.fillna(median, inplace=True)
    # z = np.abs(zscore(df.iloc[:, :4]))

    # Save 6 samples as a holdout group to use only in the testing phase - don't touch it until you finish training and evaluating your model
    df0 = df.sample(frac=0.96, random_state=42)
    holdout = df.drop(df0.index)
    # Seperate the feature columns (first 4) from the labels column (5th column)
    x = df0.iloc[:, :4]
    y = df0.iloc[:, 4]
    x_standard = x.apply(zscore)
    species_names = np.unique(np.array(y))

    # one hot encode the labels since they are categorical
    y_cat = pd.get_dummies(y, prefix='cat')
    y_cat.sample(10, random_state=42)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x_standard, y_cat, test_size=0.5, random_state=42)
    x_train, y_train = x_train.to_numpy(), np.argmax(y_train.to_numpy(), axis=1)
    W = compile([4, 2, 2, 3], 0.12)
    # print(model.W)
    # print(model.predict(x_train))
    print(W[0])
    print('------------------------------------')
    # feedforward(W,x_train)
    # print(grad(y_train, landa=0)[0])
    # print('------------------------------------')
    # print(model.W[0])
