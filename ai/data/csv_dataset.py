import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CsvDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple
    label_name: str
    label_mapping: dict
    inverse_label_mapping: dict


def _resolve_column(column, headers, column_count, name):
    if isinstance(column, str):
        if headers is None:
            raise ValueError(f"{name} cannot be a name when has_header is False")
        if column not in headers:
            raise ValueError(f"unknown {name}: {column}")
        return headers.index(column)

    index = int(column)
    if index < 0:
        index += column_count
    if index < 0 or index >= column_count:
        raise ValueError(f"{name} index out of range: {column}")
    return index


def _resolve_feature_columns(feature_columns, label_index, headers, column_count):
    if feature_columns is None:
        return [index for index in range(column_count) if index != label_index]

    indices = [
        _resolve_column(column, headers, column_count, "feature column")
        for column in feature_columns
    ]
    if label_index in indices:
        raise ValueError("feature_columns must not include the label column")
    if len(set(indices)) != len(indices):
        raise ValueError("feature_columns contains duplicates")
    return indices


def _build_label_mapping(raw_labels):
    unique_labels = set(raw_labels)
    try:
        ordered_labels = sorted(unique_labels, key=lambda value: int(value))
    except ValueError:
        ordered_labels = sorted(unique_labels)
    return {label: index for index, label in enumerate(ordered_labels)}


def _normalize_label_mapping(label_mapping):
    normalized = {str(key): int(value) for key, value in label_mapping.items()}
    values = sorted(normalized.values())
    expected = list(range(len(values)))
    if values != expected:
        raise ValueError("label_mapping values must be contiguous integers from 0")
    return normalized


def _parse_feature(value, row_number, column_name):
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number} has non-numeric feature in column {column_name}: {value}"
        ) from exc


def load_csv_dataset(
    csv_path,
    label_column=-1,
    feature_columns=None,
    has_header=True,
    delimiter=",",
    encoding="utf-8-sig",
    label_mapping=None,
):
    """
    Load a feature CSV into NumPy arrays for the hand-written classifiers.

    By default the last column is treated as the label, and every other column
    is treated as a numeric feature. String labels are encoded to integers from
    0, and the mapping is returned with the dataset.
    """
    path = Path(csv_path)
    with path.open("r", newline="", encoding=encoding) as csv_file:
        rows = [row for row in csv.reader(csv_file, delimiter=delimiter) if row]

    if not rows:
        raise ValueError("CSV file is empty")

    if has_header:
        headers = tuple(cell.strip() for cell in rows[0])
        data_rows = rows[1:]
    else:
        headers = None
        data_rows = rows

    if not data_rows:
        raise ValueError("CSV file does not contain data rows")

    column_count = len(data_rows[0])
    if column_count == 0:
        raise ValueError("CSV file does not contain columns")

    if headers is not None and len(headers) != column_count:
        raise ValueError("header column count does not match data column count")

    for offset, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != column_count:
            raise ValueError(
                f"row {offset} has {len(row)} columns, expected {column_count}"
            )

    label_index = _resolve_column(label_column, headers, column_count, "label column")
    feature_indices = _resolve_feature_columns(
        feature_columns, label_index, headers, column_count
    )
    if not feature_indices:
        raise ValueError("at least one feature column is required")

    column_names = (
        headers if headers is not None else tuple(f"column_{i}" for i in range(column_count))
    )

    features = []
    raw_labels = []
    for offset, row in enumerate(data_rows, start=2 if has_header else 1):
        features.append(
            [
                _parse_feature(row[index].strip(), offset, column_names[index])
                for index in feature_indices
            ]
        )
        raw_labels.append(row[label_index].strip())

    if label_mapping is None:
        label_mapping = _build_label_mapping(raw_labels)
    else:
        label_mapping = _normalize_label_mapping(label_mapping)

    missing_labels = sorted(set(raw_labels) - set(label_mapping))
    if missing_labels:
        raise ValueError(f"label_mapping is missing labels: {missing_labels}")

    y = np.asarray([label_mapping[label] for label in raw_labels], dtype=int)
    inverse_label_mapping = {index: label for label, index in label_mapping.items()}

    return CsvDataset(
        X=np.asarray(features, dtype=float),
        y=y,
        feature_names=tuple(column_names[index] for index in feature_indices),
        label_name=column_names[label_index],
        label_mapping=dict(label_mapping),
        inverse_label_mapping=inverse_label_mapping,
    )
