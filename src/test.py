import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model import FFNN
from activation import ReLU, Sigmoid, Softmax, Tanh
from loss import CategoricalCrossEntropy
from utils import one_hot_encode
from initializer import HeInitializer, XavierInitializer

# Create output directory if it doesn't exist
os.makedirs('out', exist_ok=True)

def load_mnist_dataset():
    try:
        print("Loading MNIST dataset using fetch_openml...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading from OpenML: {str(e)}")
        print("Loading from local file instead...")

        from scipy.io import arff
        try:
            data, meta = arff.loadarff('mnist_784.arff')
            data_array = np.array(data.tolist(), dtype=object)
            X = np.array(data_array[:, :-1], dtype=float)
            y = data_array[:, -1].astype(str)
            print("Dataset loaded from local file.")
        except Exception as local_error:
            print(f"Error loading from local file: {str(local_error)}")
            print("Falling back to digits dataset...")
            from sklearn.datasets import load_digits
            digits = load_digits()
            X, y = digits.data, digits.target
            print("Digits dataset loaded successfully.")

    if isinstance(y[0], str):
        y = np.array([int(label) for label in y])

    return X, y

def load_and_prepare_data():
    print("Loading MNIST dataset...")
    X, y = load_mnist_dataset()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    if len(X) > 10000:
        print("Using a subset of the data for faster training...")
        subset_size = 10000
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Subset shape: {X.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_classes = len(np.unique(y))
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)

    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"One-hot encoded y_train shape: {y_train_onehot.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot

def create_and_train_model(X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot,
                          use_autodiff=False):
    print("\n" + "="*50)
    implementation_type = "AUTODIFF" if use_autodiff else "STANDARD"
    print(f"{implementation_type} IMPLEMENTATION (3-Layer Architecture)")
    print("="*50)

    input_size = X_train.shape[1]

    num_classes = y_train_onehot.shape[1]
    model = FFNN(
        layer_sizes=[input_size, 128, 64, num_classes],
        activations=[ReLU(), Tanh(), Softmax()],
        loss=CategoricalCrossEntropy(),
        use_autodiff=use_autodiff
    )

    if not use_autodiff:
        from layer import Layer
        model.layers = []
        for i in range(len(model.layer_sizes) - 1):
            initializer = HeInitializer(seed=42) if i == 0 else XavierInitializer(seed=42)
            model.layers.append(Layer(
                input_size=model.layer_sizes[i],
                output_size=model.layer_sizes[i + 1],
                activation=model.activations[i],
                weight_initializer=initializer
            ))

    start_time = time.time()

    print(f"\nTraining {implementation_type.lower()} model...")
    history = model.fit(
        X_train,
        y_train_onehot,
        batch_size=32,
        learning_rate=0.01,
        epochs=5,
        validation_data=(X_test, y_test_onehot),
        verbose=1
    )

    train_time = time.time() - start_time
    print(f"Training took {train_time:.2f} seconds")

    y_pred, accuracy = evaluate_model(model, X_test, y_test, y_test_onehot, implementation_type.lower())

    plot_learning_curves(history, implementation_type.lower())

    return model, history, train_time, accuracy

def train_sklearn_model(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("SKLEARN IMPLEMENTATION (3-Layer Architecture)")
    print("="*50)

    print("\nCreating sklearn MLPClassifier...")
    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='tanh',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.01,
        max_iter=5,
        random_state=42,
        verbose=True
    )

    start_time = time.time()

    print("\nTraining sklearn MLPClassifier...")
    sklearn_model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"Training took {train_time:.2f} seconds")

    history = {'loss': sklearn_model.loss_curve_}

    y_pred, accuracy = evaluate_sklearn_model(sklearn_model, X_test, y_test)
    plot_learning_curves(history, 'sklearn')

    return sklearn_model, history, train_time, accuracy

def evaluate_model(model, X_test, y_test, y_test_onehot, model_name):
    print(f"\nEvaluating {model_name} model...")

    y_pred = model.predict(X_test)

    y_pred_indices = np.argmax(y_pred, axis=1)

    accuracy = np.mean(y_pred_indices == y_test)
    print(f"\n{model_name.capitalize()} model test accuracy: {accuracy:.4f}")

    plot_confusion_matrix(y_test, y_pred_indices, model_name)

    print(f"\nClassification Report ({model_name.capitalize()}):")
    print(classification_report(y_test, y_pred_indices))

    return y_pred, accuracy

def evaluate_sklearn_model(model, X_test, y_test):
    print("\nEvaluating sklearn model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSklearn model test accuracy: {accuracy:.4f}")

    plot_confusion_matrix(y_test, y_pred, 'sklearn')

    print("\nClassification Report (Sklearn):")
    print(classification_report(y_test, y_pred))

    return model.predict_proba(X_test), accuracy

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({model_name.capitalize()})')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'out/{model_name}_cm.png')
    plt.show()

def plot_learning_curves(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name.capitalize()} Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'out/{model_name}_loss.png')
    plt.show()

def compare_models(standard_results, sklearn_results, autodiff_results=None):
    print("\n" + "="*50)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*50)

    std_model, std_history, std_time, std_accuracy = standard_results
    sklearn_model, sklearn_history, sklearn_time, sklearn_accuracy = sklearn_results

    print("\n{:<20} {:<12} {:<12} {:<15}".format("Model", "Accuracy", "Train Time", "Parameters"))
    print("-" * 60)

    if autodiff_results:
        autodiff_model, autodiff_history, autodiff_time, autodiff_accuracy = autodiff_results
        autodiff_params = "N/A"
        print("{:<20} {:<12.4f} {:<12.2f}s {:<15}".format("AutoDiff", autodiff_accuracy, autodiff_time, autodiff_params))

    std_params = sum(layer.weights.size + layer.biases.size for layer in std_model.layers)
    print("{:<20} {:<12.4f} {:<12.2f}s {:<15}".format("Standard", std_accuracy, std_time, std_params))

    sklearn_params = sum(c.size for c in sklearn_model.coefs_) + sum(i.size for i in sklearn_model.intercepts_)
    print("{:<20} {:<12.4f} {:<12.2f}s {:<15}".format("Sklearn", sklearn_accuracy, sklearn_time, sklearn_params))

    plt.figure(figsize=(12, 6))

    plt.plot(std_history['loss'], label='Standard Loss')

    x = np.linspace(0, len(std_history['loss']), len(sklearn_history['loss']))
    plt.plot(x, sklearn_history['loss'], label='Sklearn Loss')

    if autodiff_results:
        plt.plot(autodiff_history['loss'], label='AutoDiff Loss')

    plt.title('Comparison of Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('out/combined_loss.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    models = []
    accuracies = []

    if autodiff_results:
        models.append('AutoDiff')
        accuracies.append(autodiff_accuracy)

    models.extend(['Standard', 'Sklearn'])
    accuracies.extend([std_accuracy, sklearn_accuracy])

    plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim([0.9 * min(accuracies), 1.0])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('out/accuracy_comparison.png')
    plt.show()

def main():
    print("FFNN Demo on MNIST Dataset - Model Comparison")
    print("="*50)

    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = load_and_prepare_data()

    standard_model, standard_history, standard_time, standard_accuracy = create_and_train_model(
        X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot,
        use_autodiff=False
    )

    sklearn_model, sklearn_history, sklearn_time, sklearn_accuracy = train_sklearn_model(
        X_train, X_test, y_train, y_test
    )

    try:
        autodiff_model, autodiff_history, autodiff_time, autodiff_accuracy = create_and_train_model(
            X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot,
            use_autodiff=True
        )

        compare_models(
            (standard_model, standard_history, standard_time, standard_accuracy),
            (sklearn_model, sklearn_history, sklearn_time, sklearn_accuracy),
            (autodiff_model, autodiff_history, autodiff_time, autodiff_accuracy)
        )
    except Exception as e:
        print(f"\nError with autodiff implementation: {e}")
        print("Comparing standard and sklearn implementations only...")

        compare_models(
            (standard_model, standard_history, standard_time, standard_accuracy),
            (sklearn_model, sklearn_history, sklearn_time, sklearn_accuracy)
        )

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
