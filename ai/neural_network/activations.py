import numpy as np


def relu(x):
    """Apply ReLU element-wise."""
    return np.maximum(0.0, np.asarray(x, dtype=float))


def relu_derivative(x):
    """Return d(ReLU) / dx for a cached pre-activation tensor."""
    return (np.asarray(x, dtype=float) > 0).astype(float)


def softmax(logits):
    """
    Convert logits to probabilities with a numerically stable softmax.

    The function accepts either a single sample with shape (classes,) or a
    batch with shape (samples, classes). The return shape matches the input.
    """
    values = np.asarray(logits, dtype=float)
    is_vector = values.ndim == 1
    if is_vector:
        values = values.reshape(1, -1)

    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return probabilities[0] if is_vector else probabilities
