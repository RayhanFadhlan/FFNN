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
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

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

class Softmax(Activation):
    # f(x_i) = e^(x_i) / sum(e^(x_j))
    # Buat gak overflow, x_i di shift dulu ke x_i - max(x), jadi secara matematis
    # f(x_i) = e^(x_i) * e^(-max(x)) / sum(e^(x_j) * e^(-max(x)))
    # Based on this books: https://www.deeplearningbook.org/contents/numerical.html halaman 79
    def forward(self, x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x, grad_output):
        softmax_output = self.forward(x)
        batch_size = softmax_output.shape[0]

        # Inisiasi 0 ke array dx
        dx = np.zeros_like(softmax_output)

        for i in range(batch_size):
            # dari [x1, x2, x3] jadi [[x1], [x2], [x3]]
            s = softmax_output[i].reshape(-1, 1)
            # bikin diagonal matrix dari softmax_output jadi [[x1, 0, 0], [0, x2, 0], [0, 0, x3]]
            # sigma ij * softmax(x_i)
            outer = np.diagflat(s)
            # softmax(x_i) * softmax(x_j)
            inner = np.dot(s, s.T)
            # softmax(x_i) * (sigma ij - softmax(x_j))
            J = outer - inner
            dx[i] = np.dot(J, grad_output[i])

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
