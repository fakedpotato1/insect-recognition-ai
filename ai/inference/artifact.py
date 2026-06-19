import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.inference.classes import class_metadata
from ai.neural_network import MLPClassifier


ARTIFACT_VERSION = "1.0"
MODEL_TYPE = "mlp"


@dataclass(frozen=True)
class PredictionResult:
    class_id: int
    label: str
    insect_name: str
    confidence: float
    probabilities: np.ndarray


@dataclass
class InsectClassifierArtifact:
    model: MLPClassifier
    feature_names: tuple
    label_mapping: dict
    inverse_label_mapping: dict
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    metadata: dict

    def _transform_features(self, values):
        features = np.asarray(values, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2:
            raise ValueError("features must have shape (samples, features)")
        if features.shape[1] != self.feature_mean.size:
            raise ValueError(
                f"expected {self.feature_mean.size} features, got {features.shape[1]}"
            )
        return (features - self.feature_mean) / self.feature_scale

    def predict_proba(self, values):
        return self.model.predict_proba(self._transform_features(values))

    def predict_one(self, values):
        probabilities = self.predict_proba(values)[0]
        encoded_index = int(np.argmax(probabilities))
        raw_label = self.inverse_label_mapping.get(encoded_index, str(encoded_index))
        class_id, label, insect_name = class_metadata(raw_label, encoded_index)

        return PredictionResult(
            class_id=class_id,
            label=label,
            insect_name=insect_name,
            confidence=float(probabilities[encoded_index]),
            probabilities=probabilities,
        )

    def save(self, artifact_path):
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            **self.metadata,
            "artifact_version": ARTIFACT_VERSION,
            "model_type": MODEL_TYPE,
            "input_dim": self.model.input_dim,
            "hidden_dims": list(self.model.hidden_dims),
            "output_dim": self.model.output_dim,
            "learning_rate": self.model.learning_rate,
            "feature_names": list(self.feature_names),
            "label_mapping": {str(key): int(value) for key, value in self.label_mapping.items()},
            "inverse_label_mapping": {
                str(key): str(value)
                for key, value in self.inverse_label_mapping.items()
            },
        }

        np.savez_compressed(
            path,
            parameters=self.model.get_parameters_vector(),
            feature_mean=np.asarray(self.feature_mean, dtype=float),
            feature_scale=np.asarray(self.feature_scale, dtype=float),
            metadata=json.dumps(metadata, sort_keys=True),
        )


def _load_metadata(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    return json.loads(str(value))


def load_artifact(artifact_path):
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        metadata = _load_metadata(data["metadata"])
        if metadata.get("model_type") != MODEL_TYPE:
            raise ValueError(f"unsupported model type: {metadata.get('model_type')}")

        model = MLPClassifier(
            metadata["input_dim"],
            tuple(metadata["hidden_dims"]),
            metadata["output_dim"],
            learning_rate=metadata.get("learning_rate", 0.01),
            random_state=0,
        )
        model.set_parameters_vector(data["parameters"])

        inverse_label_mapping = {
            int(key): value
            for key, value in metadata["inverse_label_mapping"].items()
        }

        return InsectClassifierArtifact(
            model=model,
            feature_names=tuple(metadata["feature_names"]),
            label_mapping=dict(metadata["label_mapping"]),
            inverse_label_mapping=inverse_label_mapping,
            feature_mean=np.asarray(data["feature_mean"], dtype=float),
            feature_scale=np.asarray(data["feature_scale"], dtype=float),
            metadata=metadata,
        )
