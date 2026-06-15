from dataclasses import dataclass

import numpy as np

from ai.evaluation.accuracy import accuracy_score
from ai.evaluation.precision_recall import classification_report_stats


@dataclass(frozen=True)
class BaselineResult:
    """Metrics produced by one sklearn baseline model."""

    name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    predictions: np.ndarray


def _as_feature_matrix(values, name):
    features = np.asarray(values, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, features)")
    return features


def _as_label_vector(values, name):
    labels = np.asarray(values, dtype=int)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return labels


def _validate_xy(features, labels, feature_name, label_name):
    if features.shape[0] != labels.size:
        raise ValueError(
            f"{feature_name} and {label_name} must contain the same number of samples"
        )
    if labels.size == 0:
        raise ValueError(f"{label_name} cannot be empty")


def _infer_num_classes(*label_arrays):
    combined = np.concatenate([labels.ravel() for labels in label_arrays if labels.size])
    if combined.size == 0:
        raise ValueError("at least one label is required")
    if np.any(combined < 0):
        raise ValueError("class labels must be non-negative integers")
    return int(np.max(combined)) + 1


def evaluate_baseline(name, estimator, X_train, y_train, X_test, y_test):
    """
    Fit a sklearn-style estimator and return the same metrics as the custom AI.

    Baselines are intentionally library models. They are comparison points for
    the hand-written BP, PSO-BP, and GA-BP implementations, not the project's
    main AI contribution.
    """
    train_features = _as_feature_matrix(X_train, "X_train")
    train_labels = _as_label_vector(y_train, "y_train")
    test_features = _as_feature_matrix(X_test, "X_test")
    test_labels = _as_label_vector(y_test, "y_test")
    _validate_xy(train_features, train_labels, "X_train", "y_train")
    _validate_xy(test_features, test_labels, "X_test", "y_test")

    estimator.fit(train_features, train_labels)
    predictions = np.asarray(estimator.predict(test_features), dtype=int)
    num_classes = _infer_num_classes(train_labels, test_labels, predictions)
    stats = classification_report_stats(test_labels, predictions, num_classes)

    return BaselineResult(
        name=name,
        accuracy=float(accuracy_score(test_labels, predictions)),
        macro_precision=float(stats["macro_precision"]),
        macro_recall=float(stats["macro_recall"]),
        macro_f1=float(stats["macro_f1"]),
        predictions=predictions,
    )
