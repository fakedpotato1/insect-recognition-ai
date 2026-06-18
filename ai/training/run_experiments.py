import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.evaluation.precision_recall import classification_report_stats
from ai.neural_network import MLPClassifier


@dataclass(frozen=True)
class ResultRow:
    model: str
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_macro_f1: float
    val_macro_f1: float
    test_macro_f1: float
    fit_seconds: float


def parse_hidden_dims(value):
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hand-written BP and sklearn baselines on a feature CSV."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-prefix", default="feat_")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--hidden-dims", default="128,64", type=parse_hidden_dims)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--l2-penalty", default=0.001, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--knn-neighbors", default=5, type=int)
    parser.add_argument("--extra-trees", default=200, type=int)
    return parser.parse_args()


def normalized_split_name(value):
    value = value.strip().lower()
    return "val" if value in {"valid", "validation"} else value


def load_feature_csv(csv_path, feature_prefix, label_column, split_column):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        feature_names = [
            name for name in reader.fieldnames if name.startswith(feature_prefix)
        ]
        if not feature_names:
            raise ValueError(f"No feature columns start with {feature_prefix!r}")

        X, y, splits = [], [], []
        for row in reader:
            X.append([float(row[name]) for name in feature_names])
            y.append(int(float(row[label_column])))
            splits.append(normalized_split_name(row[split_column]))

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(splits),
        tuple(feature_names),
    )


def split_arrays(X, y, splits):
    masks = {
        "train": splits == "train",
        "val": splits == "val",
        "test": splits == "test",
    }
    missing = [name for name, mask in masks.items() if not np.any(mask)]
    if missing:
        raise ValueError(f"Missing split rows: {missing}")
    return {name: (X[mask], y[mask]) for name, mask in masks.items()}


def normalization_stats(X_train):
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    return mean, np.where(scale < 1e-8, 1.0, scale)


def normalize(X, mean, scale):
    return (X - mean) / scale


def metrics(y_true, y_pred, num_classes):
    stats = classification_report_stats(y_true, y_pred, num_classes)
    return float(accuracy_score(y_true, y_pred)), float(stats["macro_f1"])


def evaluate_bp(args, train, val, test, num_classes, input_dim):
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=num_classes,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        optimizer=args.optimizer,
        l2_penalty=args.l2_penalty,
    )
    start = time.time()
    model.fit(train[0], train[1], epochs=args.epochs, batch_size=args.batch_size)
    fit_seconds = time.time() - start

    train_acc, train_f1 = metrics(train[1], model.predict(train[0]), num_classes)
    val_acc, val_f1 = metrics(val[1], model.predict(val[0]), num_classes)
    test_acc, test_f1 = metrics(test[1], model.predict(test[0]), num_classes)
    return ResultRow(
        "BP",
        train_acc,
        val_acc,
        test_acc,
        train_f1,
        val_f1,
        test_f1,
        fit_seconds,
    )


def evaluate_sklearn(name, model, train, val, test, num_classes):
    start = time.time()
    model.fit(train[0], train[1])
    fit_seconds = time.time() - start

    train_acc, train_f1 = metrics(train[1], model.predict(train[0]), num_classes)
    val_acc, val_f1 = metrics(val[1], model.predict(val[0]), num_classes)
    test_acc, test_f1 = metrics(test[1], model.predict(test[0]), num_classes)
    return ResultRow(
        name,
        train_acc,
        val_acc,
        test_acc,
        train_f1,
        val_f1,
        test_f1,
        fit_seconds,
    )


def write_results(output_dir, rows, metadata):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "results.csv"
    json_path = output / "results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path.write_text(
        json.dumps(
            {"metadata": metadata, "results": [asdict(row) for row in rows]},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


def print_results(rows):
    print(
        "model,train_accuracy,val_accuracy,test_accuracy,"
        "train_macro_f1,val_macro_f1,test_macro_f1,fit_seconds"
    )
    for row in rows:
        print(
            f"{row.model},{row.train_accuracy:.4f},{row.val_accuracy:.4f},"
            f"{row.test_accuracy:.4f},{row.train_macro_f1:.4f},"
            f"{row.val_macro_f1:.4f},{row.test_macro_f1:.4f},"
            f"{row.fit_seconds:.2f}"
        )


def main():
    args = parse_args()
    X, y, split_values, feature_names = load_feature_csv(
        args.csv,
        args.feature_prefix,
        args.label_column,
        args.split_column,
    )
    split_data = split_arrays(X, y, split_values)
    num_classes = int(np.max(y)) + 1

    mean, scale = normalization_stats(split_data["train"][0])
    norm_data = {
        name: (normalize(values[0], mean, scale), values[1])
        for name, values in split_data.items()
    }

    rows = [
        evaluate_bp(
            args,
            norm_data["train"],
            norm_data["val"],
            norm_data["test"],
            num_classes,
            X.shape[1],
        )
    ]

    baselines = [
        ("Dummy-most-frequent", DummyClassifier(strategy="most_frequent")),
        ("GaussianNB", GaussianNB()),
        (
            f"KNN-{args.knn_neighbors}",
            KNeighborsClassifier(
                n_neighbors=args.knn_neighbors,
                weights="distance",
                n_jobs=-1,
            ),
        ),
        (
            "DecisionTree-balanced",
            DecisionTreeClassifier(
                random_state=args.random_state,
                class_weight="balanced",
            ),
        ),
        (
            f"ExtraTrees-{args.extra_trees}-balanced",
            ExtraTreesClassifier(
                n_estimators=args.extra_trees,
                random_state=args.random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
    for name, model in baselines:
        rows.append(
            evaluate_sklearn(
                name,
                model,
                norm_data["train"],
                norm_data["val"],
                norm_data["test"],
                num_classes,
            )
        )

    metadata = {
        "csv": str(Path(args.csv)),
        "samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "num_classes": int(num_classes),
        "train_samples": int(split_data["train"][1].size),
        "val_samples": int(split_data["val"][1].size),
        "test_samples": int(split_data["test"][1].size),
        "feature_columns": len(feature_names),
        "hidden_dims": args.hidden_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "l2_penalty": args.l2_penalty,
        "random_state": args.random_state,
    }
    csv_path, json_path = write_results(args.output_dir, rows, metadata)
    print_results(rows)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
