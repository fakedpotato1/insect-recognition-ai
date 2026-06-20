from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.data import load_csv_dataset, train_val_test_split
from ai.inference.artifact import InsectClassifierArtifact
from ai.neural_network import MLPClassifier, cross_entropy_loss


@dataclass(frozen=True)
class TrainingResult:
    artifact_path: Path
    samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    input_dim: int
    output_dim: int
    training_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    training_loss: float
    validation_loss: float
    test_loss: float
    artifact_training_accuracy: float
    history: dict
    split_ratios: tuple
    refit_full: bool


def _normalization_stats(features):
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def _normalized(features, mean, scale):
    return (features - mean) / scale


def _metrics(model, features, labels):
    if labels.size == 0:
        return None, None
    probabilities = model.predict_proba(features)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == labels))
    loss = float(cross_entropy_loss(probabilities, labels))
    return accuracy, loss


def _train_model(
    X,
    y,
    hidden_dims,
    epochs,
    batch_size,
    learning_rate,
    random_state,
    optimizer,
    l2_penalty,
):
    output_dim = int(np.max(y)) + 1
    model = MLPClassifier(
        input_dim=X.shape[1],
        hidden_dims=tuple(hidden_dims),
        output_dim=output_dim,
        learning_rate=learning_rate,
        random_state=random_state,
        optimizer=optimizer,
        l2_penalty=l2_penalty,
    )
    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
    )
    return model, history


def train_mlp_classifier_from_csv(
    csv_path,
    output_path,
    hidden_dims=(64,),
    epochs=100,
    batch_size=None,
    learning_rate=0.01,
    random_state=42,
    optimizer="adam",
    l2_penalty=0.0,
    label_column=-1,
    feature_columns=None,
    split_ratios=(0.7, 0.2, 0.1),
    refit_full=False,
):
    dataset = load_csv_dataset(
        csv_path,
        label_column=label_column,
        feature_columns=feature_columns,
    )
    if dataset.X.shape[0] == 0:
        raise ValueError("training CSV contains no samples")

    splits = train_val_test_split(
        dataset.X,
        dataset.y,
        ratios=split_ratios,
        random_state=random_state,
        stratify=True,
    )

    feature_mean, feature_scale = _normalization_stats(splits.X_train)
    X_train = _normalized(splits.X_train, feature_mean, feature_scale)
    X_val = _normalized(splits.X_val, feature_mean, feature_scale)
    X_test = _normalized(splits.X_test, feature_mean, feature_scale)

    model, history = _train_model(
        X_train,
        splits.y_train,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        optimizer=optimizer,
        l2_penalty=l2_penalty,
    )

    training_accuracy, training_loss = _metrics(model, X_train, splits.y_train)
    validation_accuracy, validation_loss = _metrics(model, X_val, splits.y_val)
    test_accuracy, test_loss = _metrics(model, X_test, splits.y_test)
    artifact_training_accuracy = training_accuracy

    if refit_full:
        feature_mean, feature_scale = _normalization_stats(dataset.X)
        normalized_features = _normalized(dataset.X, feature_mean, feature_scale)
        model, history = _train_model(
            normalized_features,
            dataset.y,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
            optimizer=optimizer,
            l2_penalty=l2_penalty,
        )
        artifact_training_accuracy, _ = _metrics(model, normalized_features, dataset.y)

    output_dim = int(np.max(dataset.y)) + 1
    ratios = tuple(float(value) for value in split_ratios)

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
            "learning_rate": float(learning_rate),
            "optimizer": optimizer,
            "l2_penalty": float(l2_penalty),
            "random_state": random_state,
            "split_ratios": ratios,
            "refit_full": bool(refit_full),
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "test_loss": test_loss,
            "artifact_training_accuracy": artifact_training_accuracy,
            "train_samples": int(splits.y_train.size),
            "validation_samples": int(splits.y_val.size),
            "test_samples": int(splits.y_test.size),
        },
    )
    artifact.save(output_path)

    return TrainingResult(
        artifact_path=Path(output_path),
        samples=dataset.X.shape[0],
        train_samples=int(splits.y_train.size),
        validation_samples=int(splits.y_val.size),
        test_samples=int(splits.y_test.size),
        input_dim=dataset.X.shape[1],
        output_dim=output_dim,
        training_accuracy=training_accuracy,
        validation_accuracy=validation_accuracy,
        test_accuracy=test_accuracy,
        training_loss=training_loss,
        validation_loss=validation_loss,
        test_loss=test_loss,
        artifact_training_accuracy=artifact_training_accuracy,
        history=history,
        split_ratios=ratios,
        refit_full=bool(refit_full),
    )
