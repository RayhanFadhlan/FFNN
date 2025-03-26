import numpy as np
from abc import ABC, abstractmethod

class Normalization(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, grad_output):
        pass

class NoNormalization(Normalization):
    def forward(self, x):
        return x

    def backward(self, x, grad_output):
        return grad_output

    def __str__(self):
        return "NoNormalization"

class RMSNorm(Normalization):
    # https://dl.acm.org/doi/pdf/10.5555/3454287.3455397
    # https://drive.google.com/file/d/1vdkQhdWBFzmp8B9EOP6YDLq2DSYwhI0y/view?usp=sharing
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.cached_rms = None

        # y = x / sqrt(1/n * sum(x^2) + epsilon)
    def forward(self, x):
        self.cached_rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + self.epsilon)
        return x / self.cached_rms

    def backward(self, x, grad_output):
        n_features = x.shape[1]
        rms_squared = np.square(self.cached_rms)
        # dx 1/rms
        dx = grad_output / self.cached_rms

        # 1/ (n * rms^2) * (sum(x))
        sum_term = np.sum(x * grad_output, axis=1, keepdims=True) / (n_features * rms_squared)
        # dx -= x * sum_term / rms
        dx -= x * sum_term / self.cached_rms

        return dx

    def __str__(self):
        return f"RMSNorm(epsilon={self.epsilon})"
