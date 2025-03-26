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
        # (1/n) * sum((y_true - y_pred)^2)
        return np.mean(np.square(y_true - y_pred))

    def backward(self, y_pred, y_true):
        # 2/n * (y_pred - y_true)
        n_samples = y_pred.shape[0] if len(y_pred.shape) > 1 else 1
        return 2 * (y_pred - y_true) / n_samples

    def __str__(self):
        return "MSE"

# TODO: coba cek lagi juga
class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # -1/n * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        epsilon = 1e-15
        # Biar ga infinite
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if len(y_pred.shape) == 1 or y_pred.shape[0] == 1:
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            return -np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))

    def backward(self, y_pred, y_true):
        # -1/n  * (y_pred - y_true / (y_pred * (1 - y_pred)))
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n_samples = y_pred.shape[0]

        gradient = (y_pred - y_true) / (y_pred * (1 - y_pred))

        return -(1/n_samples) * gradient

    def __str__(self):
        return "BinaryCrossEntropy"

# TODO: coba cek lagi juga
class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # -1/n * sum(y_true * log(y_pred))
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(1, -1)
            y_true = y_true.reshape(1, -1)

        n_samples = y_pred.shape[0]

        return -np.sum(y_true * np.log(y_pred)) / n_samples

    def backward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(1, -1)
            y_true = y_true.reshape(1, -1)

        n_samples = y_pred.shape[0]

        # -1/n * (y_true / y_pred)
        return -y_true / (y_pred * n_samples)

    def __str__(self):
        return "CategoricalCrossEntropy"
