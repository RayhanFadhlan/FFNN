import numpy as np
from activation import Activation, Linear, ReLU, Sigmoid, Tanh, Softmax
from loss import Loss, MSE, BinaryCrossEntropy, CategoricalCrossEntropy

class Value:
    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, Value):
            self.data = data.data
        else:
            self.data = np.array(data, dtype=np.float64)

        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.shape = self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power musti int atau float"
        if power == 0:
            return Value(np.ones_like(self.data))

        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            if power < 0:
                eps = 1e-10
                safe_data = np.where(np.abs(self.data) < eps, eps * np.sign(self.data), self.data)
                self.grad += (power * safe_data ** (power - 1)) * out.grad
            else:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        eps = 1e-15
        safe_other = Value(np.where(np.abs(other.data) < eps,
                                   eps * np.sign(other.data + 1e-20),
                                   other.data))
        return self * (safe_other ** -1)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        eps = 1e-15
        safe_self = Value(np.where(np.abs(self.data) < eps,
                                  eps * np.sign(self.data + 1e-20),
                                  self.data))
        return other * (safe_self ** -1)

    def backward(self):
        topo = []
        visited = set()

        # Bikin urutan topologi buat backward
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


class AutoDiffActivation:
    # Wrapper buat fungsi aktivasi
    def __init__(self, activation):
        self.activation = activation
        self.name = str(activation)
        self.last_input = None
        self.last_output = None

    def apply(self, x):
        self.last_input = x.data if isinstance(x, Value) else x

        if not isinstance(x, Value):
            x = Value(x)

        result_data = self.activation.forward(x.data)
        result = Value(result_data, (x,), self.name)
        self.last_output = result_data

        def _backward():
            grad_input = self.activation.backward(x.data, result.grad)
            x.grad += grad_input

        result._backward = _backward
        return result

    def __str__(self):
        return self.name


class AutoDiffLoss:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.name = str(loss_fn)

    def calculate(self, y_pred, y_true):
        # Pastiin y_pred jadi Value object
        if not isinstance(y_pred, Value):
            if isinstance(y_pred, list) and all(isinstance(p, Value) for p in y_pred):
                y_pred_data = np.array([p.data.item() if p.data.ndim == 0 else p.data.flatten()[0]
                                       for p in y_pred])
                original_values = y_pred
                y_pred = Value(y_pred_data)
            else:
                y_pred = Value(y_pred)
                original_values = None
        else:
            original_values = None

        if isinstance(y_true, Value):
            y_true_data = y_true.data
        else:
            y_true_data = np.array(y_true, dtype=np.float64)

        # Hitung loss
        loss_value = self.loss_fn.forward(y_pred.data, y_true_data)
        result = Value(loss_value, (y_pred,), self.name)

        #  backward
        def _backward():
            grad_pred = self.loss_fn.backward(y_pred.data, y_true_data)

            # Kasus khusus buat categorical cross-entropy + softmax
            if isinstance(self.loss_fn, CategoricalCrossEntropy) and original_values is not None:
                grad_raw = y_pred.data - y_true_data
                for i, val in enumerate(original_values):
                    if i < len(grad_raw):
                        if val.data.ndim == 0:  # skalar
                            grad_val = grad_raw[i] * result.grad
                            min_grad = 0.01
                            if abs(grad_val) < min_grad:
                                grad_val = min_grad * np.sign(grad_val + 1e-15)
                            val.grad += grad_val
                        else:  # vektor/tensor
                            reshaped_grad = np.reshape(grad_raw[i], val.data.shape)
                            val.grad += reshaped_grad * result.grad
            else:
                # Kasus standar buat nilai langsung
                y_pred.grad += grad_pred * result.grad

        result._backward = _backward
        return result

    def __str__(self):
        return self.name


class AutoDiffNeuron:
    def __init__(self, input_size, activation=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        scale = np.sqrt(2.0 / max(1, input_size))
        self.w = [Value(np.random.randn() * scale) for _ in range(input_size)]
        self.b = Value(0.0)

        if activation is None:
            self.activation = AutoDiffActivation(Linear())
        elif isinstance(activation, Activation):
            self.activation = AutoDiffActivation(activation)
        else:
            self.activation = activation

    def __call__(self, x):
        act = self.b
        for i, xi in enumerate(x):
            if i < len(self.w):
                act = act + (self.w[i] * xi)

        return self.activation.apply(act)

    def parameters(self):
        return self.w + [self.b]


class AutoDiffLayer:
    # Layer buat jaringan neural
    def __init__(self, input_size, output_size, activation=None, seed=None):
        if activation is None:
            activation = ReLU()

        # Pake seed beda buat tiap neuron
        base_seed = seed if seed is not None else np.random.randint(0, 10000)
        self.neurons = [
            AutoDiffNeuron(input_size, activation, seed=base_seed+i if seed else None)
            for i in range(output_size)
        ]

        self.activation_name = str(activation)
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, x):
        # Forward lewat semua neuron
        outputs = [n(x) for n in self.neurons]
        return outputs if len(outputs) > 1 else outputs[0]

    def parameters(self):
        # Ambil semua parameter layer
        return [p for neuron in self.neurons for p in neuron.parameters()]


class AutoDiffMLP:
    def __init__(self, layer_sizes, activations, seed=42):
        sizes = layer_sizes
        acts = activations

        if len(acts) != len(sizes) - 1:
            raise ValueError("Jumlah fungsi aktivasi harus satu kurang dari jumlah layer")

        # Bikin layers
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(
                AutoDiffLayer(
                    sizes[i],
                    sizes[i+1],
                    acts[i],
                    seed=seed+i*100 if seed else None
                )
            )

        self.layer_sizes = layer_sizes
        self.activations = activations

        # Parameter optimizer
        self.learning_rate_decay = 0.99
        self.min_learning_rate = 0.001

    def __call__(self, x):
        if not all(isinstance(xi, Value) for xi in x):
            x = [Value(float(xi)) for xi in x]

        for i, layer in enumerate(self.layers):
            layer_output = layer(x)
            if isinstance(layer_output, list):
                x = layer_output
            else:
                x = [layer_output]

        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train_step(self, X, y, learning_rate=0.01):
        loss_fn = AutoDiffLoss(CategoricalCrossEntropy())

        for p in self.parameters():
            p.grad = 0

        batch_loss = 0
        batch_size = len(X)

        for i, (x_i, y_i) in enumerate(zip(X, y)):
            x_values = [Value(float(x_j)) for x_j in x_i]

            # Forward
            y_pred = self(x_values)

            if not isinstance(y_pred, list):
                y_pred = [y_pred]

            # Hitung loss
            try:
                loss = loss_fn.calculate(y_pred, y_i)
                batch_loss += loss.data

                # Backward
                loss.backward()

            except Exception as e:
                continue

        batch_loss /= batch_size

        for p in self.parameters():
            p.grad = np.clip(p.grad, -5.0, 5.0)

            min_grad = 1e-4
            if np.abs(p.grad).sum() < min_grad:
                p.grad = p.grad + np.random.randn(*p.grad.shape) * min_grad

            effective_lr = max(learning_rate, self.min_learning_rate)
            p.data = p.data - effective_lr * p.grad

        return batch_loss

    def fit(self, X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32,
            validation_data=None, verbose=True, early_stopping_patience=None):
        losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        current_lr = learning_rate

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        n_samples = len(X_train)

        for epoch in range(epochs):
            # Acak data tiap epoch
            indices = np.random.permutation(n_samples)

            num_batches = (n_samples + batch_size - 1) // batch_size
            X_batches = [X_train[indices[i*batch_size:(i+1)*batch_size]] for i in range(num_batches)]
            y_batches = [y_train[indices[i*batch_size:(i+1)*batch_size]] for i in range(num_batches)]

            epoch_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
                try:
                    batch_loss = self.train_step(X_batch, y_batch, learning_rate=current_lr)
                    epoch_loss += batch_loss * len(X_batch) / n_samples

                    if verbose and batch_idx % max(1, num_batches // 5) == 0:
                        print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{num_batches} - Loss: {batch_loss:.6f}")
                except Exception as e:
                    continue

            losses.append(epoch_loss)

            current_lr *= self.learning_rate_decay
            current_lr = max(current_lr, self.min_learning_rate)

            if validation_data is not None:
                try:
                    X_val, y_val = validation_data
                    val_preds = self.predict(X_val)

                    # Hitung validation loss
                    if y_val.shape[1] > 1:  # Klasifikasi
                        val_loss = CategoricalCrossEntropy().forward(val_preds, y_val)
                    else:  # Regresi
                        val_loss = np.mean((val_preds - y_val) ** 2)

                    val_losses.append(val_loss)

                    # Early stopping
                    if early_stopping_patience and len(val_losses) > 1:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Stop lebih awal di epoch {epoch+1}")
                            break
                except Exception as e:
                    continue

            # Report
            if verbose:
                val_str = f", val_loss = {val_losses[-1]:.6f}" if validation_data is not None and val_losses else ""
                print(f"Epoch {epoch+1}/{epochs}: loss = {epoch_loss:.6f}{val_str}")

        history = {'loss': losses}
        if validation_data is not None and val_losses:
            history['val_loss'] = val_losses

        return history


def get_autodiff_activation(activation_name):
    activation_map = {
        'linear': Linear(),
        'relu': ReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'softmax': Softmax()
    }

    if activation_name.lower() in activation_map:
        act = activation_map[activation_name.lower()]
        return AutoDiffActivation(act)
    else:
        raise ValueError(f"Activation function '{activation_name}' not found")


def get_autodiff_loss(loss_name):
    loss_map = {
        'mse': MSE(),
        'binary_cross_entropy': BinaryCrossEntropy(),
        'categorical_cross_entropy': CategoricalCrossEntropy()
    }

    if loss_name.lower() in loss_map:
        loss_fn = loss_map[loss_name.lower()]
        return AutoDiffLoss(loss_fn)
    else:
        raise ValueError(f"Loss function '{loss_name}' not found")
