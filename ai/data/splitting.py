from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def _validate_ratios(ratios):
    values = np.asarray(ratios, dtype=float)
    if values.shape != (3,):
        raise ValueError("ratios must contain train, validation, and test values")
    if np.any(values < 0) or not np.any(values > 0):
        raise ValueError("ratios must be non-negative and contain at least one positive value")
    return values / np.sum(values)


def _ratio_counts(total, ratios):
    raw_counts = ratios * total
    counts = np.floor(raw_counts).astype(int)
    remainder = int(total - np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for index in order[:remainder]:
            counts[index] += 1
    return counts


def _split_indices(indices, ratios):
    train_count, val_count, _ = _ratio_counts(indices.size, ratios)
    train_end = train_count
    val_end = train_count + val_count
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _rebalance_to_counts(split_indices, target_counts):
    split_lists = [list(indices) for indices in split_indices]
    while True:
        sizes = np.asarray([len(indices) for indices in split_lists], dtype=int)
        deltas = sizes - target_counts
        if np.all(deltas == 0):
            break

        source_candidates = np.flatnonzero(deltas > 0)
        target_candidates = np.flatnonzero(deltas < 0)
        if source_candidates.size == 0 or target_candidates.size == 0:
            break

        source = source_candidates[np.argmax(deltas[source_candidates])]
        target = target_candidates[np.argmin(deltas[target_candidates])]
        split_lists[target].append(split_lists[source].pop())

    return tuple(np.asarray(indices, dtype=int) for indices in split_lists)


def train_val_test_split(
    X,
    y,
    ratios=(0.7, 0.2, 0.1),
    shuffle=True,
    random_state=None,
    stratify=True,
):
    """
    Split features and labels into train/validation/test arrays.

    The default ratio is 7:2:1. When stratify is True, each label is split
    independently first, then global split sizes are adjusted to match the
    requested ratio as closely as possible.
    """
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    if features.ndim != 2:
        raise ValueError("X must have shape (samples, features)")
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if features.shape[0] != labels.size:
        raise ValueError("X and y must contain the same number of samples")
    if labels.size == 0:
        raise ValueError("cannot split an empty dataset")

    normalized_ratios = _validate_ratios(ratios)
    rng = np.random.default_rng(random_state)

    if stratify:
        train_parts = []
        val_parts = []
        test_parts = []
        for label in np.unique(labels):
            label_indices = np.flatnonzero(labels == label)
            if shuffle:
                rng.shuffle(label_indices)
            label_train, label_val, label_test = _split_indices(
                label_indices, normalized_ratios
            )
            train_parts.append(label_train)
            val_parts.append(label_val)
            test_parts.append(label_test)

        train_indices = np.concatenate(train_parts)
        val_indices = np.concatenate(val_parts)
        test_indices = np.concatenate(test_parts)
        train_indices, val_indices, test_indices = _rebalance_to_counts(
            (train_indices, val_indices, test_indices),
            _ratio_counts(labels.size, normalized_ratios),
        )
    else:
        indices = np.arange(labels.size)
        if shuffle:
            rng.shuffle(indices)
        train_indices, val_indices, test_indices = _split_indices(
            indices, normalized_ratios
        )

    if shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

    return DatasetSplits(
        X_train=features[train_indices],
        y_train=labels[train_indices],
        X_val=features[val_indices],
        y_val=labels[val_indices],
        X_test=features[test_indices],
        y_test=labels[test_indices],
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
