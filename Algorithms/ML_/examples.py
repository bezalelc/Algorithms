"""
example how to use Machine Learning module,
for every algorithms in model.py there is a function of example here
"""
import numpy as np


def svm():
    # *********************    load the dataset and divide to X&y   ***********************
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(cluster_std=0.9, random_state=20, n_samples=1000, centers=10, n_features=10)

    from Algorithms.ML_.helper.data_helper import split_train_val_test
    X, Xv, y, Yv, Xt, Yt = split_train_val_test(X, Y)
    print(X.shape, y.shape, Xv.shape, Yv.shape, Xt.shape, Yt.shape)

    # *********************   build model    ***********************
    from model import SVM
    from activation import Activation, Softmax, Hinge
    from regularization import Regularization, L1, L2, L12
    from optimizer import Vanilla
    model = SVM()
    learning_rate, reg_rate = 1e-3, 5e-1
    model.compile(alpha=learning_rate, lambda_=reg_rate, activation=Softmax(), reg=L2(), opt=Vanilla())
    model.describe()
    # *********************    train   ***********************
    loss_train, loss_val = model.train(X, y, val=(Xv, Yv), iter_=1000, return_loss=True, verbose=True, eps=1e-3)
    import matplotlib.pyplot as plt
    plt.plot(range(len(loss_train)), loss_train)
    plt.plot(range(len(loss_val)), loss_val)
    plt.legend(['train', 'val'])
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.title('Training Loss history')
    plt.show()
    # *********************    predict   ***********************
    pred_train = model.predict(X)
    pred_val = model.predict(Xv)
    pred_test = model.predict(Xt)

    import metrics

    print('train accuracy=', metrics.accuracy(y, pred_train))
    print('val accuracy=', metrics.accuracy(Yv, pred_val))
    print('test accuracy=', metrics.accuracy(Yt, pred_test))
    print('null accuracy=', metrics.null_accuracy(y))
    import metrics
    metrics.print_metrics(Yt, pred_test)


def regression():
    # *********************    load the dataset and divide to X&y   ***********************
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(cluster_std=0.9, random_state=20, n_samples=1000, centers=10, n_features=10)

    from Algorithms.ML_.helper.data_helper import split_train_val_test
    X, Xv, y, Yv, Xt, Yt = split_train_val_test(X, Y)
    print(X.shape, y.shape, Xv.shape, Yv.shape, Xt.shape, Yt.shape)

    # *********************   build model    ***********************
    from model import Regression
    from layer import Layer, Dense
    from activation import Activation, Softmax, Sigmoid, ReLU
    from regularization import Regularization, L1, L2, L12
    from optimizer import Vanilla
    model = Regression()
    input_size = X.shape[1]
    hidden_size = 50
    num_classes = 10
    learning_rate, reg_rate = 1e-3, 0.5
    model = Regression(
        [Dense(hidden_size, input_shape=(input_size,), activation=ReLU(), alpha=learning_rate, lambda_=reg_rate), ])
    model += Dense(num_classes, activation=Softmax(), alpha=learning_rate, lambda_=reg_rate)  # add layer with +=
    model.compile()
    model.describe()
    # *********************    train   ***********************
    loss_train, loss_val = model.train(X, y, val=(Xv, Yv), iter_=5000, batch=32, return_loss=True, verbose=True)

    import matplotlib.pyplot as plt
    plt.plot(range(len(loss_train)), loss_train)
    plt.plot(range(len(loss_val)), loss_val)
    plt.legend(['train', 'val'])
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.title('Training Loss history')
    plt.show()
    # *********************    predict   ***********************
    pred_train = model.predict(X)
    pred_val = model.predict(Xv)
    pred_test = model.predict(Xt)

    import metrics
    print('train accuracy=', metrics.accuracy(y, pred_train))
    print('val accuracy=', metrics.accuracy(Yv, pred_val))
    print('test accuracy=', metrics.accuracy(Yt, pred_test))
    print('null accuracy=', metrics.null_accuracy(y))
    import metrics
    metrics.print_metrics(Yt, pred_test)


class Momentum1:

    def __init__(self) -> None:
        super().__init__()
        self.z = None
        self.eps: float = 1e-5
        self.momentum: float = .9
        self.axis = 0
        self.gamma, self.beta = 0, 0
        self.running_mean, self.running_var = 0, 0

    def train_norm(self, x: np.ndarray):
        z, gamma, beta, eps, axis = self.z, self.gamma, self.beta, self.eps, self.axis
        mu, var = x.mean(axis=axis), x.var(axis=axis) + eps
        std = var ** 0.5
        z = (x - mu) / std
        out = gamma * z + beta
        if axis == 0:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * (std ** 2)

        return out

    def test_norm(self, x: np.ndarray):
        running_mean, running_var = self.running_mean, self.running_var
        gamma, beta, eps = self.gamma, self.beta, self.eps
        z = (x - running_mean) / (running_var + eps) ** 0.5
        return gamma * z + beta

    def d_norm(self, dx: np.ndarray):
        # m = float()
        z, gamma, beta, eps, axis = self.z, self.gamma, self.beta, self.eps, self.axis
        d_gamma, d_beta = np.sum(dx * z, axis=axis), np.sum(dx, axis=axis)

        return dx, d_gamma, d_beta


def batch_norm_forward(a, gamma, beta, mode, momentum1):
    if mode == 'train':
        pass
    elif mode == 'test':
        pass


def check_batch_normalization():
    import numpy as np

    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    Norm = Momentum1()
    Norm.gamma, Norm.beta = np.asarray([1.0, 2.0, 3.0]), np.asarray([11.0, 12.0, 13.0])
    a = Norm.train_norm(a)
    print('  means: ', a.mean(axis=0))
    print('  stds:  ', a.std(axis=0))
    '''
      means:  [11. 12. 13.]
      stds:   [0.99999999 1.99999999 2.99999999]
    '''

    # Check the test-time forward pass by running the training-time
    # forward pass many times to warm up the running averages, and then
    # checking the means and variances of activations after a test-time
    # forward pass.
    print('--------------')
    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    gamma = np.ones(D3)
    beta = np.zeros(D3)
    Norm = Momentum1()
    Norm.gamma, Norm.beta = gamma, beta
    for t in range(50):
        X = np.random.randn(N, D1)
        a = np.maximum(0, X.dot(W1)).dot(W2)
        Norm.train_norm(a)
    # print(Norm.gamma, Norm.beta, Norm.eps)
    X = np.random.randn(N, D1)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    a = Norm.test_norm(a)
    # Means should be close to zero and stds close to one, but will be
    # noisier than training-time forward passes.
    print('  means: ', a.mean(axis=0))
    print('  stds:  ', a.std(axis=0))
    '''
     means:  [-0.03927354 -0.04349152 -0.10452688]
     stds:   [1.01531427 1.01238373 0.97819987]
    '''
    print('----------------')
    # Gradient check batchnorm backward pass
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    from model import Regression
    from layer import Dense, LayerParams
    from activation import ReLU, Linear

    # layer_param1 = LayerParams()
    # layer_param1.eps = 1
    # layer_param1.act =

    # model = Regression([Dense(D2, input_shape=(D1,)),
    #                     Dense(D3, input_shape=(D2,))])
    # model.compile()

    # print(np.sum(X))


# svm()
# regression()
check_batch_normalization()
