import numpy as np
import pickle
import os
from loss import MSE
from activation import Softmax

class FFNNAutodiff:
    def __init__(self, layer_sizes, activations, loss=None):
        if len(layer_sizes) < 2:
            raise ValueError("At least 2 layers (input and output) are required")

        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must match number of layers (except input layer)")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss if loss is not None else MSE()

        try:
            from autodiff import AutoDiffMLP, AutoDiffActivation

            autodiff_activations = []
            for act in activations:
                if isinstance(act, Softmax):
                    from autodiff import AutoDiffSoftmax
                    autodiff_activations.append(AutoDiffSoftmax())
                elif not isinstance(act, AutoDiffActivation):
                    autodiff_activations.append(AutoDiffActivation(act))
                else:
                    autodiff_activations.append(act)

            self.model = AutoDiffMLP(
                layer_sizes,
                autodiff_activations,
                seed=42
            )
        except ImportError:
            raise ImportError("AutoDiff module not available. Use FFNN instead.")

    def forward(self, X):
        if X.ndim == 1:
            x_values = [X[i] for i in range(len(X))]
            result = self.model(x_values)
            if isinstance(result, list):
                return np.array([r.data for r in result])
            else:
                return result.data
        else:
            results = []
            for x_i in X:
                x_values = [x_i[j] for j in range(len(x_i))]
                result = self.model(x_values)
                if isinstance(result, list):
                    results.append([r.data for r in result])
                else:
                    results.append(result.data)
            return np.array(results)

    def backward(self, y_pred, y_true):
        return self.loss_function.forward(y_pred, y_true)

    def fit(self, X_train, y_train, batch_size=32, learning_rate=0.01, epochs=100,
            validation_data=None, verbose=1, early_stopping_patience=None):
        try:
            autodiff_history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=(verbose == 1),
                early_stopping_patience=early_stopping_patience
            )

            history = {
                'loss': autodiff_history['loss'],
                'val_loss': autodiff_history.get('val_loss', [])
            }

            return history
        except Exception as e:
            print(f"Error in autodiff training: {e}")
            raise

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        return self.loss_function.forward(y_pred, y)

    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'loss_function': self.loss_function
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
                    model_data['loss_function']
                )

                return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

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
