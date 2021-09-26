import numpy as np


def standard_deviation(data, ddof=0):
    """
    Compute the standard deviation of data

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
    mu, sigma = np.mean(data, axis=0), np.std(data, axis=0, ddof=ddof)
    sigma[sigma == 0] = 1
    data = (data - mu) / sigma
    return data, mu, sigma


def simple_normalize(data):
    """
    Compute data between [-1,1]

    :param data: numpy array

    :return:
        data: numpy array of standard deviation of data
        max_: numpy array of max in every column (=attribute)
        min_: numpy array of min in every column (=attribute)
    """
    max_, min_ = np.max(data, axis=0), np.min(data, axis=0)
    div = (max_ - min_)
    div[div == 0] = 1
    data = (data - min_) / div
    return data, max_, min_


class BatchNorm:

    def __init__(self, dim: int, eps=1e-5, axis=0, momentum=.9) -> None:
        self.gamma: np.ndarray = np.ones(dim)
        self.beta: np.ndarray = np.zeros(dim)
        self.mu = 0
        self.var = 0
        self.std = 0
        self.z = 0
        self.eps = eps
        self.momentum = momentum
        self.running_mu = np.zeros(dim, dtype=np.float64)
        self.running_var = np.zeros(dim, dtype=np.float64)
        self.axis = axis

    def forward(self, X: np.ndarray, mode='train') -> np.ndarray:
        gamma, beta, layer_norm, eps = self.gamma, self.beta, self.axis, self.eps

        if mode == 'train':
            layer_norm = self.axis

            mu, var = X.mean(axis=0), X.var(axis=0)
            std = var ** 0.5
            z = (X - mu) / (std + eps)
            out = gamma * z + beta
            if layer_norm == 0:
                momentum = self.momentum
                self.running_mu = momentum * self.running_mu + (1 - momentum) * mu
                self.running_var = momentum * self.running_var + (1 - momentum) * var

            self.std, self.var, self.mu, self.z = std, var, mu, z

        elif mode == 'test':
            out = gamma * (X - self.running_mu) / (self.running_var + eps) ** 0.5 + beta

        else:
            raise ValueError('Invalid forward batch norm mode "%s"' % mode)

        return out

    def backward(self, delta: np.ndarray, X: np.ndarray, return_delta=True) -> Union[np.ndarray, tuple]:
        grades = self.grad(delta, X)
        d_beta = grades['d_beta']
        d_gamma = grades['d_gamma']
        dx = grades['dx']

        return dx, d_gamma, d_beta

    def delta(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        pass

    def loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        pass

    def grad(self, delta: np.ndarray, X: np.ndarray) -> dict:
        z, gamma, beta, mu, var, std = self.z, self.gamma, self.beta, self.mu, self.var, self.std
        grades = {'d_beta': delta.sum(axis=0), 'd_gamma': np.sum(delta * z, axis=0)}

        m = delta.shape[0]
        df_dz = delta * gamma
        grades['dx'] = (1 / (m * std)) * (m * df_dz - np.sum(df_dz, axis=0) - z * np.sum(df_dz * z, axis=0))

        return grades
