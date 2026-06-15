from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.data import load_csv_dataset
from ai.inference.artifact import InsectClassifierArtifact
from ai.neural_network import MLPClassifier


@dataclass(frozen=True)
class TrainingResult:
    artifact_path: Path
    samples: int
    input_dim: int
    output_dim: int
    training_accuracy: float
    history: dict


def _normalization_stats(features):
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def train_mlp_classifier_from_csv(
    csv_path,
    output_path,
    hidden_dims=(64,),
    epochs=100,
    batch_size=None,
    learning_rate=0.01,
    random_state=42,
    label_column=-1,
    feature_columns=None,
):
    dataset = load_csv_dataset(
        csv_path,
        label_column=label_column,
        feature_columns=feature_columns,
    )
    if dataset.X.shape[0] == 0:
        raise ValueError("training CSV contains no samples")

    output_dim = int(np.max(dataset.y)) + 1
    model = MLPClassifier(
        input_dim=dataset.X.shape[1],
        hidden_dims=tuple(hidden_dims),
        output_dim=output_dim,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    feature_mean, feature_scale = _normalization_stats(dataset.X)
    normalized_features = (dataset.X - feature_mean) / feature_scale
    history = model.fit(
        normalized_features,
        dataset.y,
        epochs=epochs,
        batch_size=batch_size,
    )
    predictions = model.predict(normalized_features)
    accuracy = float(np.mean(predictions == dataset.y))

    artifact = InsectClassifierArtifact(
        model=model,
        feature_names=dataset.feature_names,
        label_mapping=dataset.label_mapping,
        inverse_label_mapping=dataset.inverse_label_mapping,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        metadata={
            "source_csv": str(Path(csv_path)),
            "epochs": int(epochs),
            "batch_size": batch_size,
            "random_state": random_state,
            "training_accuracy": accuracy,
        },
    )
    artifact.save(output_path)

    return TrainingResult(
        artifact_path=Path(output_path),
        samples=dataset.X.shape[0],
        input_dim=dataset.X.shape[1],
        output_dim=output_dim,
        training_accuracy=accuracy,
        history=history,
    )
