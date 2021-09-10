import numpy as np


def predict(data, x, k=3):
    """
    k nearest neighbors algorithm:
        predict x according to the closest distance values
        this function is for linear predict


    :param data: dataset
    :param x: data to predict
    :param k: range

    :return: prediction

    :efficiency: O(m*n)
    """
    data = np.array(data, dtype=np.float128)
    distances = np.sum((data[:, :-1] - np.array(x, dtype=np.float128)) ** 2, axis=1)
    idx = np.argpartition(distances, 1)[:k]
    neighbors = data[:, -1:][idx].reshape((k,))
    weights = np.exp(-distances[idx] / (2 * np.std(neighbors) ** 2))
    W = sum(weights)
    return (1 / W) * np.sum(neighbors * weights), (1 / k) * np.sum(neighbors)


def predict_class(data, x, k=3):
    """
    k nearest neighbors algorithm:
        predict x according to the closest distance values
        this function is for classification predict


    :param data: dataset
    :param x: data to predict
    :param k: range

    :return: prediction

    :efficiency: O(m*n)
    """
    data = np.array(data, dtype=np.float128)
    distances = np.sum((data[:, :-1] - np.array(x, dtype=np.float128)) ** 2, axis=1)
    idx = np.argpartition(distances, 1)[:k]
    neighbors = data[:, -1:][idx].reshape((k,))
    weights = np.exp(-distances[idx] / (2 * np.std(neighbors) ** 2))
    W = np.sum(weights)
    return (1 / W) * np.sum(neighbors * weights), (1 / k) * np.sum(neighbors)


# D = np.array([[1, 2, 0], [4, 5, 1], [1, 2, 0]])
# print(D)
# print(predict_class(D, D[0, :], k=2))
