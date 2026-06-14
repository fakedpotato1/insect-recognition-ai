import numpy as np

from ai.neural_network.activations import softmax


def _validate_class_labels(y, num_classes):
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional integer label array")
    if labels.size == 0:
        raise ValueError("y cannot be empty")

    labels = labels.astype(int, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("class labels must be in the range [0, num_classes)")
    return labels


def cross_entropy_loss(probabilities, y, epsilon=1e-12):
    """
    Calculate mean multi-class cross-entropy from probabilities and labels.

    y must be a one-dimensional integer array encoded from 0.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape (samples, classes)")

    labels = _validate_class_labels(y, probs.shape[1])
    if probs.shape[0] != labels.size:
        raise ValueError("probabilities and y must contain the same number of samples")

    clipped = np.clip(probs, epsilon, 1.0)
    sample_losses = -np.log(clipped[np.arange(labels.size), labels])
    return float(np.mean(sample_losses))


def softmax_cross_entropy(logits, y):
    """
    Return loss, gradient with respect to logits, and probabilities.

    For softmax followed by cross-entropy, dL/dz simplifies to
    (probabilities - one_hot_labels) / batch_size.
    """
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2:
        raise ValueError("logits must have shape (samples, classes)")

    labels = _validate_class_labels(y, values.shape[1])
    if values.shape[0] != labels.size:
        raise ValueError("logits and y must contain the same number of samples")

    probabilities = softmax(values)
    loss = cross_entropy_loss(probabilities, labels)

    grad_logits = probabilities.copy()
    grad_logits[np.arange(labels.size), labels] -= 1.0
    grad_logits /= labels.size
    return loss, grad_logits, probabilities
