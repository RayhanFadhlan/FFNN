import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def one_hot_encode(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1

    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
    return one_hot


def calculate_accuracy(y_pred, y_true):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = (y_pred > 0.5).astype(int).flatten()

    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    return accuracy_score(y_true_indices, y_pred_indices)


def load_mnist_data(subset_size=None):
    """
    Load MNIST dataset and preprocess it

    Note: subset size to be used for faster training
    """

    print("Loading MNIST dataset...")
    try:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto")
        X = X.astype(float)
        y = y.astype(int)
    except Exception as e:
        print(f"Error loading from OpenML: {str(e)}")

    X = np.array(X)
    y = np.array(y)

    X = X / 255.0

    # If subset_size is provided
    if subset_size is not None and subset_size < len(X):
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # One-hot encode the labels
    num_classes = len(np.unique(y))
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)

    print(f"Data loaded: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data: X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot


def train_and_evaluate(
    model,
    X_train,
    X_test,
    y_train_onehot,
    y_test,
    y_test_onehot,
    epochs=10,
    batch_size=32,
    learning_rate=0.01,
    model_name="Model",
):
    """
    Train and evaluate a model, returning its history and accuracy
    """
    print(f"\nTraining {model_name}...")
    start_time = time.time()

    history = model.fit(
        X_train,
        y_train_onehot,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        validation_data=(X_test, y_test_onehot),
        verbose=1,
    )

    training_time = time.time() - start_time
    print(f"Training took {training_time:.2f} seconds")

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)

    print(f"{model_name} accuracy: {accuracy:.4f}")

    return history, accuracy, training_time, model
