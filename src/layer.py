import numpy as np
from activation import Linear
from initializer import HeInitializer, ZeroInitializer
from regularizers import NoRegularizer
from normalization import NoNormalization

class Layer:
    def __init__(
        self,
        input_size,
        output_size,
        activation=None,
        weight_initializer=None,
        bias_initializer=None,
        regularizer=None,
        normalization=None
    ):
        # Default values
        self.activation = activation if activation is not None else Linear()
        self.weight_initializer = weight_initializer if weight_initializer is not None else HeInitializer()
        self.bias_initializer = bias_initializer if bias_initializer is not None else ZeroInitializer()
        self.regularizer = regularizer if regularizer is not None else NoRegularizer()
        self.normalization = normalization if normalization is not None else NoNormalization()

        self.weights = self.weight_initializer.initialize((input_size, output_size))
        self.biases = self.bias_initializer.initialize(output_size)

        self.weights_grad = np.zeros_like(self.weights)
        self.biases_grad = np.zeros_like(self.biases)

        self.input = None
        self.output_before_activation = None
        self.output = None

        # Add optimization parameters for faster convergence
        self.weight_cache = np.zeros_like(self.weights)
        self.bias_cache = np.zeros_like(self.biases)

        # Velocity decay for momentum
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

        # Store layer dimensions for compatibility
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        self.input = x

        # x @ w + b
        self.output_before_activation = np.dot(x, self.weights) + self.biases

        # Apply normalization
        normalized_output = self.normalization.forward(self.output_before_activation)

        # activation
        self.output = self.activation.forward(normalized_output)

        return self.output

    def backward(self, grad_output):
        # dE/dnet = dE/dO * dO/dnet
        grad_before_activation = self.activation.backward(self.output_before_activation, grad_output)

        grad_before_norm = self.normalization.backward(self.output_before_activation, grad_before_activation)

        batch_size = max(1, self.input.shape[0])  # Biar gak 0 division

        # dE/dw = dE/dnet * dnet/dw
        self.weights_grad = np.dot(self.input.T, grad_before_norm) / batch_size

        self.weights_grad += self.regularizer.gradient(self.weights)

        # w.r.t biases
        self.biases_grad = np.mean(grad_before_norm, axis=0)

        # dE/dx = dE/dnet dnet/dx = dE/dnet w^T
        grad_input = np.dot(grad_before_norm, self.weights.T)

        return grad_input

    def update_weights(self, learning_rate, use_momentum=True, momentum=0.9, use_rmsprop=True,
                       decay_rate=0.99, epsilon=1e-8, clip_value=1.0):
        # https://www.geeksforgeeks.org/adam-optimizer/
        if use_momentum:
            # vt = beta·v{t-1} + (1-beta)·dw
            self.weight_momentum = momentum * self.weight_momentum + (1 - momentum) * self.weights_grad
            self.bias_momentum = momentum * self.bias_momentum + (1 - momentum) * self.biases_grad

            weight_update = self.weight_momentum
            bias_update = self.bias_momentum
        else:
            # delta_w = dw
            weight_update = self.weights_grad
            bias_update = self.biases_grad

        # https://www.geeksforgeeks.org/gradient-descent-with-rmsprop-from-scratch/
        if use_rmsprop:
            self.weight_cache = decay_rate * self.weight_cache + (1 - decay_rate) * np.square(self.weights_grad)
            self.bias_cache = decay_rate * self.bias_cache + (1 - decay_rate) * np.square(self.biases_grad)

            # delta_w = delta_w / sqrt(cache + epsilon)
            weight_update = weight_update / (np.sqrt(self.weight_cache) + epsilon)
            bias_update = bias_update / (np.sqrt(self.bias_cache) + epsilon)

        weight_update = np.clip(weight_update, -clip_value, clip_value)
        bias_update = np.clip(bias_update, -clip_value, clip_value)

        self.weights -= learning_rate * weight_update
        self.biases -= learning_rate * bias_update

    def get_weights(self):
        return self.weights, self.biases