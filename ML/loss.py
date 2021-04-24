"""
functions for Measure the error for classification:
    accuracy,confusion_matrix,recall,precision,false_positive_rate,F_score

"""
import numpy as np


def accuracy(y, predict):
    return np.mean(predict == y)


def confusion_matrix(y, predict):
    """
    compute the confusion matrix for classification

    :param
        y: true label for examples
        predict: predict label for examples

    :return: confusion matrix

    :efficiency: O(m^2) where m in number of examples
    """
    m, k = y.shape[0], np.unique(y).shape[0]
    y, predict = np.array(y[:], dtype=np.uint8).reshape((-1,)), np.array(predict[:], dtype=np.uint8).reshape((-1,))
    M = np.zeros((k, k), dtype=np.float64)
    np.add.at(M, (y, predict), 1)
    return M / m


def recall(y, predict):
    """
    compute the recall from the confusion matrix for classification

    :param
        y: true label for examples
        predict: predict label for examples

    :Formula: recall = true_positive/(true_positive+false_negative)


    :return: recall vector for every class [recall(class A),...,recall(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    M = confusion_matrix(y, predict)
    return np.diag(M) / (np.sum(M, axis=1))


def precision(y, predict):
    """
    compute the precision from the confusion matrix for classification

    :param
        y: true label for examples
        predict: predict label for examples

    :Formula: precision = true_positive/(true_positive+false_positive)

    :return: precision vector for every class [precision(class A),...,precision(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    M = confusion_matrix(y, predict)
    return np.diag(M) / (np.sum(M, axis=0))


def false_positive_rate(y, predict):
    """
    compute the false positive rate from the confusion matrix for classification

    :param
        y: true label for examples
        predict: predict label for examples

    :Formula: FPR = false_positive/(false_positive+true_negative)

    :return: precision vector for every class [FPR(class A),...,FPR(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    M = confusion_matrix(y, predict)
    FP = np.sum(M, axis=0) - np.diag(M)
    TN = np.sum(np.diag(M)) - np.diag(M)
    return FP / (FP + TN)


def F_score(y, predict, beta=1):
    """
    compute the F_score from the confusion matrix,precision and recall for classification

    :param
        y: true label for examples
        predict: predict label for examples
        beta: if |beta| < 1 : more weight for the recall
              if |beta| = 1 : same weight for precision and recall (= harmonic mean)
              if |beta| > 1 : more weight for the precision


    :Formula: F_score = ((1+beta^2) * recall * precision) / (beta ** 2 * precision + recall)

    :return: precision vector for every class [F_score(class A),...,F_score(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    recall_, precision_ = recall(y, predict), precision(y, predict)
    return ((1 + beta ** 2) * recall_ * precision_) / (beta ** 2 * precision_ + recall_)


if __name__ == '__main__':
    import gradient_descent as gd
    import optimizer as opt
    from general import load_data

    print('------------------------  test 1  ---------------------------')
    data = load_data.load_from_file('/home/bb/Documents/python/ML/data/ex2data1.txt')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    # X, mu, sigma = normalize.standard_deviation(X)
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    theta = np.zeros((X.shape[1], 1))
    # theta = np.array([-24, 0.2, 0.2]).reshape((X.shape[1], 1))
    # print(theta.shape)
    # theta = np.array([-25.161, 0.206, 0.201]).reshape((X.shape[1], 1))
    # print(class_cost(X, y, theta))
    # print(class_grad(X, y, theta))
    theta = np.array([-25.06116393, 0.2054152, 0.2006545]).reshape((X.shape[1], 1))
    theta, J = gd.regression(X, y, theta, gd.class_grad, optimizer_data={'alpha': [0.000002, ]}, num_iter=100,
                             cost=gd.class_cost,
                             optimizer=opt.momentum,
                             batch=X.shape[0])

    predict = np.round(gd.sigmoid(X, theta))
    print(confusion_matrix(y, predict))
    print(precision(y, predict))
    print(0.05 / (0.05 + 0.55))
    print('------------------------  test 2  ---------------------------')
    data = load_data.load_from_file('/home/bb/Downloads/data/seeds_dataset.txt', delime='\t')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    X = np.insert(X, 0, np.ones((X.shape[0]), dtype=X.dtype), axis=1)
    k = np.array(np.unique(y), dtype=np.uint8)
    Y = (y == k)

    theta = np.zeros((X.shape[1], k.shape[0]))
    theta = np.array([[-1.6829658687213866, -1.8543754265161039, 0.7754343585719218],
                      [-0.9215084632150828, 4.521276688115852, -4.004501938604477],
                      [2.648370316571783, -3.313060949916832, 0.16063859934981728],
                      [-0.05307786295000424, -2.4615062595012542, 1.3240942639424873],
                      [16.15559358997315, -11.585899548679317, -4.528678388339358],
                      [0.48474185667614517, -5.932838971987173, 4.974046769850393],
                      [-1.0259316117512303, 0.8043956856099687, 1.2596367701574402],
                      [-20.999903530911624, 11.891520812942074, 10.235602620923723]]
                     )

    # J = []
    # for i in range(len(k)):
    #     # print( X@theta[:, i])
    #     theta[:, i], j = regression(X, Y[:, i], theta[:, i], class_grad, cost=class_cost, num_iter=100, alpha=0.001,
    #                                 optimizer=momentum, batch=X.shape[0])
    #     J.append(j)
    theta, J = gd.regression(X, Y, theta, gd.class_grad, cost=gd.class_cost, num_iter=1000,
                             optimizer_data={'alpha': [0.001, ]},
                             optimizer=opt.simple, batch=X.shape[0])
    print(J[0] - J[-1:])
    print(theta.tolist())

    # # plot
    # plt.plot(range(len(J)), J)
    # plt.xlabel(xlabel='iter number')
    # plt.ylabel(ylabel='cost')
    # plt.title('regression')
    # plt.show()

    # print error
    res = np.array((np.round(np.argmax(gd.sigmoid(X, theta), axis=1) + 1))).reshape((y.shape))
    print(np.mean(res))
    print('accuracy=', np.mean(np.round(gd.sigmoid(X, theta)) == Y))
    print('accuracy=', np.mean(y == res))
    print('accuracy=', accuracy(y - 1, res - 1))
    print('confusion_matrix=', confusion_matrix(y - 1, res - 1))
    print('recall=', recall(y - 1, res - 1))
    print('precision=', precision(y - 1, res - 1))
    print('false_positive_rate=', false_positive_rate(y - 1, res - 1))

    # np.where(a < limit, np.floor(a), np.ceil(a))
