import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, grad_output):
        pass

class Linear(Activation):
    # f(x) = x
    def forward(self, x):
        return x

    # df/dx = 1
    def backward(self, x, grad_output):
        return grad_output

    def __str__(self):
        return "Linear"

class ReLU(Activation):
    # f(x) = max(0, x)
    def forward(self, x):
        return np.maximum(0, x)

    # df/dx = 1 if x > 0, 0 otherwise
    def backward(self, x, grad_output):
        return grad_output * (x > 0)

    def __str__(self):
        return "ReLU"

class Sigmoid(Activation):
    # f(x) = 1 / (1 + e^(-x))
    def forward(self, x):
        # Biar gak overflow computation nya, x di clip ke [-500, 500]
        # TODO: for the boundary try it out using others
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    # df/dx = f(x) * (1 - f(x))
    def backward(self, x, grad_output):
        sigmoid_output = self.forward(x)
        return grad_output * sigmoid_output * (1 - sigmoid_output)

    def __str__(self):
        return "Sigmoid"

class Tanh(Activation):
    # f(x) = tanh(x)
    def forward(self, x):
        return np.tanh(x)

    # df/dx = 1 - tanh^2(x)
    def backward(self, x, grad_output):
        return grad_output * (1 - np.tanh(x)**2)

    def __str__(self):
        return "Tanh"

# TODO: Coba cek lagi ini
class Softmax(Activation):
    # f(x_i) = e^(x_i) / sum(e^(x_j))
    # Buat gak overflow, x_i di shift dulu ke x_i - max(x), jadi secara matematis
    # f(x_i) = e^(x_i) * e^(-max(x)) / sum(e^(x_j) * e^(-max(x)))
    # Based on this books: https://www.deeplearningbook.org/contents/numerical.html halaman 79
    def forward(self, x):
        x_array = np.array(x, dtype=np.float64)

        if x_array.size == 0:
            return np.array([])

        if x_array.ndim == 0:
            return np.array([1.0])

        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)

        shifted_x = x_array - np.max(x_array, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        softmax_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        if x_array.shape[0] == 1 and (not hasattr(x, 'ndim') or x.ndim == 1):
            return softmax_output[0]

        return softmax_output

    def backward(self, x, grad_output):
        x_array = np.array(x, dtype=np.float64)

        if x_array.ndim == 0:
            return np.zeros_like(x_array)
        if x_array.ndim == 1:
            # dari [x1, x2, x3] jadi [[x1], [x2], [x3]]
            x_array = x_array.reshape(1, -1)

        if grad_output.ndim == 1 and x_array.ndim == 2:
            grad_output = grad_output.reshape(1, -1)

        softmax_output = self.forward(x_array)
        if softmax_output.ndim == 1:
            softmax_output = softmax_output.reshape(1, -1)

        batch_size = x_array.shape[0]

        # Inisiasi 0 ke array dx
        dx = np.zeros_like(x_array)

        for i in range(batch_size):
             # bikin diagonal matrix dari softmax_output jadi [[x1, 0, 0], [0, x2, 0], [0, 0, x3]]
            diag_softmax = np.diag(softmax_output[i])

            # J_ij = y_i * (delta_ij - y_j)
            jacobian = diag_softmax - np.outer(softmax_output[i], softmax_output[i])

            #  grad_output[i] @ J
            dx[i] = grad_output[i] @ jacobian

        if x.ndim == 1:
            return dx[0]

        return dx

    def __str__(self):
        return "Softmax"

# Bonus activation functions number 2
# https://medium.com/analytics-vidhya/activation-function-c762b22fd4da
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    # f(x) = max(alpha*x, x)
    def forward(self, x):
        return np.maximum(self.alpha * x, x)

    # df/dx = alpha if x < 0, 1 otherwise
    def backward(self, x, grad_output):
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return grad_output * dx

    def __str__(self):
        return f"LeakyReLU(alpha={self.alpha})"

class ELU(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    # f(x) = x if x > 0, alpha*(e^x - 1) otherwise
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -500, 0)) - 1))

    # df/dx = 1 if x > 0, alpha*e^x otherwise
    def backward(self, x, grad_output):
        dx = np.ones_like(x)
        mask = x <= 0
        dx[mask] = self.alpha * np.exp(np.clip(x, -500, 0))[mask]
        return grad_output * dx

    def __str__(self):
        return f"ELU(alpha={self.alpha})"
