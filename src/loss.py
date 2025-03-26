import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self, y_pred, y_true):
        # dl/dy_pred
        pass

class MSE(Loss):
    def forward(self, y_pred, y_true):
        # (1/n) * sum((y_true - y_pred )^2)
        return np.mean(np.square(y_true - y_pred))

    def backward(self, y_pred, y_true):
        # (2/n) * (y_pred - y_true)
        n_samples = y_pred.shape[0]
        return 2 * (y_pred - y_true) / n_samples

    def __str__(self):
        return "MSE"

class BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        # -1/n * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        epsilon = 1e-15
        # Biar ga infinite
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def backward(self, y_pred, y_true):
        # -1/n  * (y_pred - y_true / (y_pred * (1 - y_pred)))
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n_samples = y_pred.shape[0]

        gradient = (y_pred - y_true) / (y_pred * (1 - y_pred))

        return -(1/n_samples) * gradient

    def __str__(self):
        return "BinaryCrossEntropy"

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        #  -1/n * sum(y_true * log(y_pred))
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n_samples = y_pred.shape[0]

        return -np.sum(y_true * np.log(y_pred)) / n_samples

    def backward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n_samples = y_pred.shape[0]

        # -1/n * (y_true / y_pred)
        # biar ga infinite
        gradient = np.zeros_like(y_pred)
        mask = y_true > 0
        gradient[mask] = -y_true[mask] / (y_pred[mask] * n_samples)

        return gradient

    def __str__(self):
        return "CategoricalCrossEntropy"
