import numpy as np
import abc


class Regularization(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def norm(x: np.ndarray) -> float:
        pass

    @staticmethod
    @abc.abstractmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        pass


class L1(Regularization):

    @staticmethod
    def norm(x: np.ndarray) -> float:
        return float(np.sum(np.abs(x)))

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return np.abs(x)


class L2(Regularization):

    @staticmethod
    def norm(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return x * 2


class L12(Regularization):

    @staticmethod
    def norm(x: np.ndarray) -> float:
        return L1.norm(x) + L2.norm(x)

    @staticmethod
    def d_norm(x: np.ndarray) -> np.ndarray:
        return L1.d_norm(x) + L2.d_norm(x)

# from typing import Callable, NewType
# Regularization = NewType('Regularization', Callable[[np.ndarray, int], np.ndarray])
# L1 = Regularization(lambda x, axis=-1: np.sum(np.abs(x), axis=axis))
# L2 = Regularization(lambda x, axis=-1: np.sum(x ** 2))
# L12 = Regularization(lambda x, axis=-1: L1(x, axis) + L2(x, axis))  # elastic net
#
# dRegularization = NewType('dRegularization', Callable[[np.ndarray, int], np.ndarray])
# dL2 = dRegularization(lambda x, axis=-1: 2 * x)
