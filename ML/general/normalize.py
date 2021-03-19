import numpy as np
from matplotlib.pyplot import axis


def standard_deviation(data):
    """
    Compute the standard deviation of data
        this function not normalize the column 0

    :param data: numpy array

    :return:
        data: numpy array of standard deviation of data
        mu: numpy array of the mean of every column (=attribute)
        sigma: numpy array of standard deviation of every column (=attribute)
    """
    '''
    m, n = data.shape
    mu = 1 / m * np.sum(data, axis=0)
    sigma = np.sqrt(1 / m * np.sum((data - mu) ** 2, axis=0))
    sigma[sigma == 0] = 1  // if sigma[i] == 0 => need to divide by 1 because there is not standard deviation  
    '''
    mu, sigma = np.mean(data[:, 1:], axis=0), np.std(data[:, 1:], axis=0)
    sigma[sigma == 0] = 1
    data[:, 1:] = (data[:, 1:] - mu) / sigma
    return data, mu, sigma


def simple_normalize(data):
    """
    Compute data between [-1,1]
        this function not normalize the column 0

    :param data: numpy array

    :return:
        data: numpy array of standard deviation of data
        max_: numpy array of max in every column (=attribute)
        min_: numpy array of min in every column (=attribute)
    """
    max_, min_ = np.max(data[:, 1:], axis=0), np.min(data[:, 1:], axis=0)
    div = (max_ - min_)
    div[div == 0] = 1
    data[:, 1:] = (data[:, 1:] - min_) / div
    return data, max_, min_


if __name__ == '__main__':
    t = np.array([[1, -5], [2, 15], [0, 20]])
    X = np.random.random((3, 3)) * 10
    # standard_deviation(X)
    # t, mu, sigma = standard_deviation(t)
    # print(t, '\n')
    # print((t * sigma) + mu)
    data, max_, min_ = simple_normalize(X)
    print(data, max_, min_)
    # print(t)
