import numpy as np


def standard_deviation(data):
    m, n = data.shape
    U = 1 / m * np.sum(data, axis=0)
    sigma = np.sqrt(1 / m * np.sum((data - U) ** 2, axis=0))
    sigma[sigma == 0] = 1
    return (data - U) / sigma, U, sigma


def simple_normalize(data):
    max_, min_ = np.max(data), np.min(data)
    res = (data - min_) / (max_ - min_) if max_ != min_ else data
    return res, max_, min_


if __name__ == '__main__':
    t = np.array([[1, -5], [2, 15], [0, 20]])
    X = np.random.random((3, 3)) * 10
    # standard_deviation(X)
    standard_deviation(t)
    simple_normalize(t)
    print(t)
