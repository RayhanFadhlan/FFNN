import numpy as np
from abc import ABC, abstractmethod


class Regularizer(ABC):
    @abstractmethod
    def loss(self, weights):
        pass

    @abstractmethod
    def gradient(self, weights):
        pass


class NoRegularizer(Regularizer):
    def loss(self, weights):
        return 0

    def gradient(self, weights):
        return np.zeros_like(weights)

    def __str__(self):
        return "NoRegularizer"


# https://www.geeksforgeeks.org/regularization-in-machine-learning/
class L1Regularizer(Regularizer):
    def __init__(self, lambda_param=0.01):
        self.lambda_param = lambda_param

    def loss(self, weights):
        #  lambda * sum(|weights|)
        return self.lambda_param * np.sum(np.abs(weights))

    def gradient(self, weights):
        #  lambda * sign(weights)
        return self.lambda_param * np.sign(weights)

    def __str__(self):
        return f"L1Regularizer(lambda={self.lambda_param})"


class L2Regularizer(Regularizer):
    def __init__(self, lambda_param=0.01):
        self.lambda_param = lambda_param

    def loss(self, weights):
        # lambda * sum(weights^2)
        return  self.lambda_param * np.sum(np.square(weights))

    def gradient(self, weights):
        # 2 * lambda * weights
        return 2 * self.lambda_param * weights

    def __str__(self):
        return f"L2Regularizer(lambda={self.lambda_param})"
