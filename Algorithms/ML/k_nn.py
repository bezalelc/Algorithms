import numpy as np

from Algorithms.ML.general import load_data


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


if __name__ == '__main__':
    """
    data for linear test:
    '/home/bb/Documents/python/ML/data/ex1data1.txt'
    '/home/bb/Documents/python/ML/data/ex1data2.txt' 

    
    data for class test:
    '/home/bb/Documents/python/ML/data/ex2data1.txt' 
    '/home/bb/Documents/python/ML/data/ex2data2.txt'
    """
    # data = [[1, 2, 3], [1, 0, 4], [2, 0, 0]]
    # x = [2, 3]
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex2data2.txt')
    i = 20
    # # x = np.array(data)[:, :-1]
    x = data[45][-1]
    print(predict(data, x, k=7))
    print(data[i][-1])
    # print(np.array([9,8,7]))
