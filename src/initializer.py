import numpy as np
from abc import ABC, abstractmethod

class Initializer(ABC):
    @abstractmethod
    def initialize(self, shape):
        pass

class ZeroInitializer(Initializer):
    def initialize(self, shape):
        return np.zeros(shape)

    def __str__(self):
        return "ZeroInitializer"

class UniformInitializer(Initializer):
    def __init__(self, low=-0.1, high=0.1, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.low, self.high, shape)

    def __str__(self):
        return f"UniformInitializer(low={self.low}, high={self.high}, seed={self.seed})"

class NormalInitializer(Initializer):
    def __init__(self, mean=0.0, std=0.1, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(self.mean, self.std, shape)

    def __str__(self):
        return f"NormalInitializer(mean={self.mean}, std={self.std}, seed={self.seed})"

# Bonus initialization methods bagian 3
class XavierInitializer(Initializer):
    # limit = sqrt(6 / (fan_in + fan_out)), weights = U(-limit, limit)
    # https://www.geeksforgeeks.org/xavier-initialization/
    def __init__(self, seed=None):
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)

        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else 1
        limit = np.sqrt(6 / (fan_in + fan_out))

        return np.random.uniform(-limit, limit, shape)

    def __str__(self):
        return f"XavierInitializer(seed={self.seed})"


class HeInitializer(Initializer):
    #  std = sqrt(2 / fan_in), weights = N(0, std²)
    def __init__(self, seed=None):
        self.seed = seed

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)

        fan_in = shape[0]
        std = np.sqrt(2 / fan_in)

        return np.random.normal(0, std, shape)

    def __str__(self):
        return f"HeInitializer(seed={self.seed})"
