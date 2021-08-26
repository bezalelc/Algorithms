import numpy as np
import scipy as sc
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm as sv
import metrics
import karnel

if __name__ == '__main__':
    # print('---------------------  ex6data1  -----------------------------')
    # data = sc.io.loadmat('/home/bb/Documents/octave/week7/machine-learning-ex6/ex6/ex6data1.mat')
    # X, y = data['X'], np.squeeze(data['y'])
    #
    # model = sv.SVC(kernel='linear', C=0.2)
    # model.fit(X, y)
    #
    # # plot Decision Boundary
    # min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    # min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(min1, max1, 0.01).flatten(), np.arange(min2, max2, 0.01).flatten())
    # grig = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
    # zz = model.predict(grig).reshape(xx.shape)
    # plt.contourf(xx, yy, zz, cmap='Paired')
    #
    # # plot points
    # idx0, idx1 = np.argwhere(y == 0), np.argwhere(y == 1)
    # plt.scatter(X[idx0, 0], X[idx0, 1], cmap='Paired')
    # plt.scatter(X[idx1, 0], X[idx1, 1], cmap='Paired')
    # plt.show()
    #
    # #loss
    # print(loss.accuracy(y, model.predict(X)))
    # print(loss.recall(y, model.predict(X)))
    # print(loss.confusion_matrix(y, model.predict(X)))
    # print(loss.F_score(y, model.predict(X)))

    # print('---------------------  ex6data2  -----------------------------')
    # data = sc.io.loadmat('/home/bb/Documents/octave/week7/machine-learning-ex6/ex6/ex6data2.mat')
    # X, y = data['X'], np.squeeze(data['y'])
    # model = sv.SVC(kernel='rbf', C=40, gamma=500)
    # model.fit(X, y)
    #
    # # plot Decision Boundary
    # min1, max1 = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    # min2, max2 = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    # xx, yy = np.meshgrid(np.arange(min1, max1, 0.01).flatten(), np.arange(min2, max2, 0.01).flatten())
    # grig = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
    # zz = model.predict(grig).reshape(xx.shape)
    # plt.contourf(xx, yy, zz, cmap='Paired')
    #
    # # plot points
    # idx0, idx1 = np.argwhere(y == 0), np.argwhere(y == 1)
    # plt.scatter(X[idx0, 0], X[idx0, 1], cmap='Paired')
    # plt.scatter(X[idx1, 0], X[idx1, 1], cmap='Paired')
    # plt.show()

    print('---------------------  ex6data3  -----------------------------')
    data = sc.io.loadmat('/home/bb/Documents/octave/week7/machine-learning-ex6/ex6/ex6data3.mat')
    X, y = data['X'], np.squeeze(data['y'])
    model = sv.SVC(kernel='rbf', C=5, gamma=52)
    model.fit(X, y)

    # plot Decision Boundary
    min1, max1 = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    min2, max2 = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    xx, yy = np.meshgrid(np.arange(min1, max1, 0.01).flatten(), np.arange(min2, max2, 0.01).flatten())
    grig = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
    zz = model.predict(grig).reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')

    # plot points
    idx0, idx1 = np.argwhere(y == 0), np.argwhere(y == 1)
    plt.scatter(X[idx0, 0], X[idx0, 1], cmap='Paired')
    plt.scatter(X[idx1, 0], X[idx1, 1], cmap='Paired')
    plt.show()
