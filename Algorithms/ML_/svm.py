import numpy as np
from model import Model, SVM
from regularization import Regularization, L1, L2
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
Xv, Yv = X[train_size:].reshape((val_size, -1)), y[train_size:].reshape((val_size, -1))
X, y = X[:train_size].reshape((train_size, -1)), y[:train_size].reshape((train_size, -1))
Xd, Yd = Xte[test_size:test_size + dev_size].reshape((dev_size, -1)), Yte[test_size:test_size + dev_size].reshape(
    (dev_size, -1))
Xte, Yte = Xte[:test_size].reshape((test_size, -1)), Yte[:test_size].reshape((test_size, -1))
# *********************    data manipulation   ***********************
mu = X.mean(axis=0)
X, Xv, Xte, Xd = X - mu, Xv - mu, Xte - mu, Xd - mu
X, Xv = np.hstack((X, np.ones((X.shape[0], 1)))), np.hstack((Xv, np.ones((Xv.shape[0], 1))))
Xte, Xd = np.hstack((Xte, np.ones((Xte.shape[0], 1)))), np.hstack((Xd, np.ones((Xd.shape[0], 1))))
# *********************    train   ***********************
W = np.random.randn(X.shape[1], len(classes)) * 1e-4
print(W.shape)
model = SVM()


# model.train(X, y)
# # start = default_timer()
# pred = model.predict(Xte, k=60)
# # end = default_timer()
# # print('time=', end - start)
# # *********************    predict   ***********************
# print(metrics.accuracy(Yte, pred))
# print(np.mean(metrics.null_accuracy(Yte)))
