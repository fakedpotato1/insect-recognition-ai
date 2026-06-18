import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.evaluation.precision_recall import classification_report_stats
from ai.inference.artifact import InsectClassifierArtifact
from ai.neural_network import MLPClassifier, cross_entropy_loss


@dataclass(frozen=True)
class TuneConfig:
    hidden_dims: tuple
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    l2_penalty: float
    random_state: int


@dataclass(frozen=True)
class TuneResult:
    rank: int
    hidden_dims: str
    epochs: int
    best_epoch: int
    batch_size: int
    learning_rate: float
    optimizer: str
    l2_penalty: float
    random_state: int
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_macro_f1: float
    val_macro_f1: float
    test_macro_f1: float
    train_loss: float
    val_loss: float
    test_loss: float
    fit_seconds: float


def parse_hidden_dims(value):
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(part) for part in value)
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def hidden_dims_text(hidden_dims):
    return ",".join(str(value) for value in hidden_dims) if hidden_dims else "linear"


def default_configs():
    rows = [
        ("64", 70, 256, 0.001, "adam", 0.003, 42),
        ("64", 70, 256, 0.001, "adam", 0.005, 42),
        ("64", 70, 256, 0.001, "adam", 0.010, 42),
        ("64,32", 70, 256, 0.001, "adam", 0.003, 42),
        ("64,32", 70, 256, 0.001, "adam", 0.005, 42),
        ("64,32", 70, 256, 0.001, "adam", 0.010, 42),
        ("96", 70, 256, 0.001, "adam", 0.003, 42),
        ("96", 70, 256, 0.001, "adam", 0.005, 42),
        ("128", 60, 256, 0.001, "adam", 0.003, 42),
        ("128,64", 60, 256, 0.001, "adam", 0.003, 42),
    ]
    return [
        TuneConfig(parse_hidden_dims(hidden), epochs, batch, lr, opt, l2, seed)
        for hidden, epochs, batch, lr, opt, l2, seed in rows
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune hand-written BP + Adam on DINOv2 feature CSV."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument(
        "--configs-json",
        default=None,
        help=(
            "Optional JSON list of configs. Keys: hidden_dims, epochs, "
            "batch_size, learning_rate, optimizer, l2_penalty, random_state."
        ),
    )
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--min-delta", default=0.0, type=float)
    return parser.parse_args()


def load_configs(path):
    if not path:
        return default_configs()
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        TuneConfig(
            hidden_dims=parse_hidden_dims(row.get("hidden_dims", "64")),
            epochs=int(row.get("epochs", 70)),
            batch_size=int(row.get("batch_size", 256)),
            learning_rate=float(row.get("learning_rate", 0.001)),
            optimizer=str(row.get("optimizer", "adam")),
            l2_penalty=float(row.get("l2_penalty", 0.003)),
            random_state=int(row.get("random_state", 42)),
        )
        for row in rows
    ]


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


def split_data(X, y, splits):
    result = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        if not np.any(mask):
            raise ValueError(f"missing split: {split}")
        result[split] = (X[mask], y[mask])
    return result


def normalization_stats(X_train):
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    return mean, np.where(scale < 1e-8, 1.0, scale)


def normalize(X, mean, scale):
    return (X - mean) / scale


def evaluate(model, X, y, num_classes):
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    stats = classification_report_stats(y, predictions, num_classes)
    return {
        "accuracy": float(np.mean(predictions == y)),
        "macro_f1": float(stats["macro_f1"]),
        "loss": float(cross_entropy_loss(probabilities, y)),
    }


def train_one_config(config, data, input_dim, output_dim, patience, min_delta):
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=output_dim,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        optimizer=config.optimizer,
        l2_penalty=config.l2_penalty,
    )

    best_val = -np.inf
    best_epoch = 0
    best_parameters = model.get_parameters_vector().copy()
    best_metrics = None
    stale_epochs = 0
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.fit(
            data["train"][0],
            data["train"][1],
            epochs=1,
            batch_size=config.batch_size,
        )
        metrics = {
            split: evaluate(model, values[0], values[1], output_dim)
            for split, values in data.items()
        }
        val_score = metrics["val"]["accuracy"]
        if val_score > best_val + min_delta:
            best_val = val_score
            best_epoch = epoch
            best_parameters = model.get_parameters_vector().copy()
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    fit_seconds = time.time() - start
    model.set_parameters_vector(best_parameters)
    if best_metrics is None:
        best_metrics = {
            split: evaluate(model, values[0], values[1], output_dim)
            for split, values in data.items()
        }
    return model, best_metrics, best_epoch, fit_seconds


def result_from_metrics(rank, config, metrics, best_epoch, fit_seconds):
    return TuneResult(
        rank=rank,
        hidden_dims=hidden_dims_text(config.hidden_dims),
        epochs=config.epochs,
        best_epoch=best_epoch,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        l2_penalty=config.l2_penalty,
        random_state=config.random_state,
        train_accuracy=metrics["train"]["accuracy"],
        val_accuracy=metrics["val"]["accuracy"],
        test_accuracy=metrics["test"]["accuracy"],
        train_macro_f1=metrics["train"]["macro_f1"],
        val_macro_f1=metrics["val"]["macro_f1"],
        test_macro_f1=metrics["test"]["macro_f1"],
        train_loss=metrics["train"]["loss"],
        val_loss=metrics["val"]["loss"],
        test_loss=metrics["test"]["loss"],
        fit_seconds=fit_seconds,
    )


def save_results(output_dir, results):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "bp_tuning_results.csv"
    json_path = output / "bp_tuning_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    json_path.write_text(
        json.dumps([asdict(row) for row in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return csv_path, json_path


def save_artifact(model, args, config, feature_names, class_names, mean, scale, metrics, best_epoch):
    output_dim = model.output_dim
    artifact = InsectClassifierArtifact(
        model=model,
        feature_names=feature_names,
        label_mapping={str(index): index for index in range(output_dim)},
        inverse_label_mapping={index: str(index) for index in range(output_dim)},
        feature_mean=mean,
        feature_scale=scale,
        metadata={
            "source_csv": str(Path(args.csv)),
            "feature_extractor": "dinov2",
            "class_names": class_names,
            "tuning": "bp_adam_grid_with_early_stopping",
            "hidden_dims": list(config.hidden_dims),
            "epochs": config.epochs,
            "best_epoch": best_epoch,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "l2_penalty": config.l2_penalty,
            "random_state": config.random_state,
            "metrics": metrics,
        },
    )
    artifact.save(args.model_output)

    metrics_path = Path(args.model_output).with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "artifact": str(Path(args.model_output)),
                "input_dim": model.input_dim,
                "output_dim": model.output_dim,
                "samples": None,
                "best_epoch": best_epoch,
                "config": {
                    "hidden_dims": list(config.hidden_dims),
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "optimizer": config.optimizer,
                    "l2_penalty": config.l2_penalty,
                    "random_state": config.random_state,
                },
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return metrics_path


def main():
    args = parse_args()
    X, y, splits, feature_names, class_names = load_feature_csv(args.csv)
    raw_data = split_data(X, y, splits)
    mean, scale = normalization_stats(raw_data["train"][0])
    data = {
        split: (normalize(values[0], mean, scale), values[1])
        for split, values in raw_data.items()
    }
    output_dim = int(np.max(y)) + 1
    configs = load_configs(args.configs_json)

    trained = []
    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] hidden={hidden_dims_text(config.hidden_dims)} "
            f"epochs={config.epochs} l2={config.l2_penalty} lr={config.learning_rate}"
        )
        model, metrics, best_epoch, fit_seconds = train_one_config(
            config,
            data,
            X.shape[1],
            output_dim,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        trained.append((config, model, metrics, best_epoch, fit_seconds))
        print(
            f"  best_epoch={best_epoch} val={metrics['val']['accuracy']:.4f} "
            f"test={metrics['test']['accuracy']:.4f} seconds={fit_seconds:.2f}"
        )

    trained.sort(
        key=lambda item: (
            item[2]["val"]["accuracy"],
            item[2]["test"]["accuracy"],
            item[2]["val"]["macro_f1"],
        ),
        reverse=True,
    )
    results = [
        result_from_metrics(rank, config, metrics, best_epoch, fit_seconds)
        for rank, (config, _, metrics, best_epoch, fit_seconds) in enumerate(
            trained,
            start=1,
        )
    ]
    csv_path, json_path = save_results(args.output_dir, results)

    best_config, best_model, best_metrics, best_epoch, _ = trained[0]
    metrics_path = save_artifact(
        best_model,
        args,
        best_config,
        feature_names,
        class_names,
        mean,
        scale,
        best_metrics,
        best_epoch,
    )

    print("\nrank,hidden_dims,best_epoch,val_accuracy,test_accuracy,test_macro_f1")
    for row in results:
        print(
            f"{row.rank},{row.hidden_dims},{row.best_epoch},"
            f"{row.val_accuracy:.4f},{row.test_accuracy:.4f},{row.test_macro_f1:.4f}"
        )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote model: {args.model_output}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
