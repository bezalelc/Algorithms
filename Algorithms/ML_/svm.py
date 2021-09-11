import matplotlib.pyplot as plt
import numpy as np
from model import Model, SVM
from regularization import Regularization, L1, L2, dL2
# import numba as nb
# from numba import njit, jit, prange
# from numba.experimental import jitclass
# from numba.np.ufunc import parallel
from timeit import default_timer
from scipy.stats import mode
import os
import metrics
import Algorithms.ML_.data as data
from Algorithms.ML_.data.load_data import data_paths, CIFAR_10, CIFAR_10_batch
import Algorithms.ML_.helper.plot_ as plot_
# from activation import linear, sigmoid, softmax, cross_entropy
from activation import Activation, Hinge, Softmax

# *********************    load the dataset   ***********************
classes = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
path = data_paths['CIFAR10']
# # Xtr, Ytr, Xte, Yte = CIFAR_10()
X, y = CIFAR_10_batch(os.path.join(path, 'data_batch_1'))
Xte, Yte = CIFAR_10_batch(os.path.join(path, 'test_batch'))

# *********************    see data   ***********************
# plot_.img_show_data(X[0:9, ...].astype('uint8'), classes[y[0:9].astype(int)], wspace=0)
# *********************    divide to train,val,test,dev   ***********************
train_size, val_size, test_size, dev_size = 9000, 1000, 1000, 500
# # num_training, num_validation, num_test, num_dev = 49000, 1000, 1000, 500

# dev data
np.random.seed(1)
mask = np.random.choice(train_size, dev_size, replace=False)
Xd, Yd = X[mask].reshape((dev_size, -1)), y[mask].reshape((dev_size,))

Xv, Yv = X[train_size:].reshape((val_size, -1)), y[train_size:].reshape((val_size,))
X, y = X[:train_size].reshape((train_size, -1)), y[:train_size].reshape((train_size,))
# Xd, Yd = Xte[test_size:test_size + dev_size].reshape((dev_size, -1)), Yte[test_size:test_size + dev_size].reshape(
#     (dev_size,))
Xte, Yte = Xte[:test_size].reshape((test_size, -1)), Yte[:test_size].reshape((test_size,))

# *********************    data manipulation   ***********************
mu = X.mean(axis=0)
X, Xv, Xte, Xd = X - mu, Xv - mu, Xte - mu, Xd - mu

# *********************    train   ***********************
# model = SVM()
# model.compile(lambda_=2.5e4, alpha=1e-7)  # 1e-7, reg=2.5e4,
# loss_history = model.train(X, y, eps=0.001, batch=200, iter_=1500)
#
# plt.plot(range(len(loss_history)), loss_history)
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()
# print(loss_history[::100])
# lr, rg = SVM.ff(X, y, Xv, Yv, [1e-7, 1e-6],[2e4, 2.5e4, 3e4, 3.5e4, 4e4, 4.5e4, 5e4, 6e4])
# print(lr, rg)
model = SVM()
model.compile(alpha=1e-7, lambda_=2.5e4, activation=Softmax, Reg=L2, dReg=dL2)
# model.compile(alpha=0, lambda_=0, activation=Hinge, Reg=L2, dReg=dL2)
history = model.train(X, y, iter_=0, eps=0.0001)
print(model.loss(model.X, model.y, add_ones=False), np.sum(model.grad(model.X, model.y, False)))
L, dW = model.grad(model.X, model.y, True)
print(L, np.sum(dW))

# print(np.sum(model.grad(model.X, model.y, loss_=False)))
# print(np.sum(model.grad1(model.X, model.y)))
# L, dW = model.activation.loss_grad_loop(model.X, model.W, model.y)
# print(L, np.sum(dW))

loss_history = model.train(X, y, eps=0.0001, batch=200, iter_=1500)
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
print(loss_history[::100])
# *********************    metrics   ***********************
pred = np.argmax(model.predict(model.X, add_ones=False), axis=1)
pred_v = np.argmax(model.predict(Xv), axis=1)
pred_te = np.argmax(model.predict(Xte), axis=1)

# # *********************    metrics   ***********************
print(metrics.accuracy(y, pred))
print(metrics.accuracy(Yv, pred_v))
print(metrics.accuracy(Yte, pred_te))
# print(np.mean(metrics.null_accuracy(Yte)))
