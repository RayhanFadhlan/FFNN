import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from layer import Layer
from loss import MSE
from activation import Softmax

class FFNN:

    def __init__(self, layer_sizes, activations, loss=None, use_autodiff=False):
        if len(layer_sizes) < 2:
            raise ValueError("At least 2 layers (input and output) are required")

        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must match number of layers (except input layer)")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss if loss is not None else MSE()
        self.use_autodiff = use_autodiff

        # Pake standard, input output layering
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activations[i]))

        self.autodiff_model = None
        if use_autodiff:
            try:
                from autodiff import AutoDiffMLP, AutoDiffActivation, AutoDiffSoftmax

                # Force pake autodiff buat activationnya
                autodiff_activations = []
                for act in activations:
                    if isinstance(act, Softmax):
                        autodiff_activations.append(AutoDiffSoftmax())
                    elif not isinstance(act, AutoDiffActivation):
                        autodiff_activations.append(AutoDiffActivation(act))
                    else:
                        autodiff_activations.append(act)

                self.autodiff_model = AutoDiffMLP(
                    layer_sizes,
                    autodiff_activations,
                    seed=42
                )
            except ImportError:
                print("AutoDiff module not available. Using standard implementation.")
                self.use_autodiff = False

    def forward(self, X):
        if self.use_autodiff and self.autodiff_model:
            # Handle different input shapes for autodiff
            if X.ndim == 1:  # Single sample
                x_values = [X[i] for i in range(len(X))]
                result = self.autodiff_model(x_values)
                if isinstance(result, list):
                    return np.array([r.data for r in result])
                else:
                    return result.data
            else:  # Batch of samples
                results = []
                for x_i in X:
                    x_values = [x_i[j] for j in range(len(x_i))]
                    result = self.autodiff_model(x_values)
                    if isinstance(result, list):
                        results.append([r.data for r in result])
                    else:
                        results.append(result.data)
                return np.array(results)
        else:
            # Standard forward pass
            output = X
            for layer in self.layers:
                output = layer.forward(output)
            return output

    def backward(self, y_pred, y_true):
        # autodiff
        if self.use_autodiff and self.autodiff_model:
            return self.loss_function.forward(y_pred, y_true)
        else:
            # Standard backward pass
            loss = self.loss_function.forward(y_pred, y_true)
            grad = self.loss_function.backward(y_pred, y_true)

            for layer in reversed(self.layers):
                grad = layer.backward(grad)

            return loss

    def update_weights(self, learning_rate):
        if not self.use_autodiff:
            for layer in self.layers:
                layer.update_weights(learning_rate)

    def fit(self, X_train, y_train, batch_size=32, learning_rate=0.01, epochs=100,
            validation_data=None, verbose=1, early_stopping_patience=None):
        history = {
            'loss': [],
            'val_loss': []
        }

        # Case pake autodiff
        if self.use_autodiff and self.autodiff_model:
            try:
                autodiff_history = self.autodiff_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    verbose=(verbose == 1),
                    early_stopping_patience=early_stopping_patience
                )

                # Transfer history
                history['loss'] = autodiff_history['loss']
                if 'val_loss' in autodiff_history:
                    history['val_loss'] = autodiff_history['val_loss']

                return history
            except Exception as e:
                print(f"Error in autodiff training: {e}")
                print("Falling back to standard implementation")
                self.use_autodiff = False

        # Standard training
        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            epoch_loss = 0

            if verbose == 1:
                batch_iterator = tqdm(range(0, n_samples, batch_size),
                                    desc=f"Epoch {epoch+1}/{epochs}",
                                    unit="batch")
            else:
                batch_iterator = range(0, n_samples, batch_size)

            for start_idx in batch_iterator:
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Backward pass
                batch_loss = self.backward(y_pred, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples

                # Update weights
                self.update_weights(learning_rate)

                if verbose == 1:
                    batch_iterator.set_postfix({
                        'loss': f"{batch_loss:.4f}"
                    })

            # Record training loss
            history['loss'].append(epoch_loss)

            # Hitung validation loss
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function.forward(y_val_pred, y_val)
                history['val_loss'].append(val_loss)

                # Early stopping
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        if verbose == 1:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

            # Print epoch summary
            if verbose == 1:
                val_str = f", val_loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}{val_str}")

        return history

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        return self.loss_function.forward(y_pred, y)

    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Utils
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'loss_function': self.loss_function,
            'use_autodiff': self.use_autodiff,
            'layers': self.layers if not self.use_autodiff else None
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                print(f"Saved model to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, filepath):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

                model = cls(
                    model_data['layer_sizes'],
                    model_data['activations'],
                    model_data['loss_function'],
                    use_autodiff=model_data.get('use_autodiff', False)
                )

                if not model.use_autodiff and 'layers' in model_data:
                    model.layers = model_data['layers']

                return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # Visualization methods
    def visualize_model(self, simplified=False, save_path=None):
        try:
            from visualization import plot_simplified_model_graph, plot_model_graph
            if simplified:
                plot_simplified_model_graph(self, save_path)
            else:
                plot_model_graph(self, save_path)
        except ImportError:
            print("Visualization module not available. Cannot visualize model.")

    def visualize_weight_distributions(self, layers_to_plot=None, save_path=None):
        try:
            from visualization import plot_weight_distributions
            plot_weight_distributions(self, layers_to_plot, save_path)
        except ImportError:
            print("Visualization module not available. Cannot visualize weight distributions.")

    def visualize_gradient_distributions(self, layers_to_plot=None, save_path=None):
        try:
            from visualization import plot_gradient_distributions
            plot_gradient_distributions(self, layers_to_plot, save_path)
        except ImportError:
            print("Visualization module not available. Cannot visualize gradient distributions.")

    def visualize_learning_curves(self, history, save_path=None):
        try:
            from visualization import plot_learning_curves
            plot_learning_curves(history, save_path)
        except ImportError:
            print("Visualization module not available. Cannot visualize learning curves.")

    def visualize_predictions(self, y_true, y_pred, y_sklearn=None, save_path=None):
        try:
            from visualization import plot_prediction_comparison
            plot_prediction_comparison(y_true, y_pred, y_sklearn, save_path)
        except ImportError:
            print("Visualization module not available. Cannot visualize predictions.")
