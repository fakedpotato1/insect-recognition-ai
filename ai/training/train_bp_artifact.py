import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.evaluation.precision_recall import classification_report_stats
from ai.inference.artifact import InsectClassifierArtifact
from ai.neural_network import MLPClassifier, cross_entropy_loss


def parse_hidden_dims(value):
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save the hand-written BP classifier artifact."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dims", default="128,64", type=parse_hidden_dims)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--l2-penalty", default=0.001, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


def normalize_split(value):
    value = value.strip().lower()
    return "val" if value in {"valid", "validation"} else value


def load_feature_csv(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        feature_names = [name for name in reader.fieldnames if name.startswith("feat_")]
        X, y, splits = [], [], []
        class_names = {}
        for row in reader:
            X.append([float(row[name]) for name in feature_names])
            label = int(float(row["label"]))
            y.append(label)
            splits.append(normalize_split(row["split"]))
            if row.get("class_name"):
                class_names[str(label)] = row["class_name"]

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(splits),
        tuple(feature_names),
        class_names,
    )


def normalization_stats(X_train):
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    return mean, np.where(scale < 1e-8, 1.0, scale)


def normalized(X, mean, scale):
    return (X - mean) / scale


def split_data(X, y, splits):
    result = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        if not np.any(mask):
            raise ValueError(f"missing split: {split}")
        result[split] = (X[mask], y[mask])
    return result


def evaluate(model, X, y, num_classes):
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    stats = classification_report_stats(y, predictions, num_classes)
    return {
        "accuracy": float(np.mean(predictions == y)),
        "macro_f1": float(stats["macro_f1"]),
        "loss": float(cross_entropy_loss(probabilities, y)),
    }


def main():
    args = parse_args()
    X, y, splits, feature_names, class_names = load_feature_csv(args.csv)
    data = split_data(X, y, splits)
    # Fit normalization on the train split only to avoid leaking validation or
    # test statistics into the saved model artifact.
    mean, scale = normalization_stats(data["train"][0])
    norm_data = {
        split: (normalized(values[0], mean, scale), values[1])
        for split, values in data.items()
    }

    output_dim = int(np.max(y)) + 1
    model = MLPClassifier(
        input_dim=X.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=output_dim,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        optimizer=args.optimizer,
        l2_penalty=args.l2_penalty,
    )

    start = time.time()
    history = model.fit(
        norm_data["train"][0],
        norm_data["train"][1],
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    fit_seconds = time.time() - start

    metrics = {
        split: evaluate(model, values[0], values[1], output_dim)
        for split, values in norm_data.items()
    }
    label_mapping = {str(index): index for index in range(output_dim)}
    inverse_label_mapping = {index: str(index) for index in range(output_dim)}

    artifact = InsectClassifierArtifact(
        model=model,
        feature_names=feature_names,
        label_mapping=label_mapping,
        inverse_label_mapping=inverse_label_mapping,
        feature_mean=mean,
        feature_scale=scale,
        metadata={
            "source_csv": str(Path(args.csv)),
            "feature_extractor": "dinov2",
            "class_names": class_names,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "l2_penalty": args.l2_penalty,
            "random_state": args.random_state,
            "fit_seconds": fit_seconds,
            "metrics": metrics,
            "history": history,
        },
    )
    artifact.save(args.output)

    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "artifact": str(Path(args.output)),
                "samples": int(X.shape[0]),
                "input_dim": int(X.shape[1]),
                "output_dim": output_dim,
                "fit_seconds": fit_seconds,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote model: {args.output}")
    print(f"Wrote metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
