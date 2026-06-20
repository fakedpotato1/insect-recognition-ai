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
from ai.neural_network import MLPClassifier, cross_entropy_loss
from ai.optimization import GeneticAlgorithmBP, ParticleSwarmBP


@dataclass(frozen=True)
class ExperimentResult:
    model: str
    hidden_dims: str
    optimizer_samples: int
    optimizer_seconds: float
    bp_seconds: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_macro_f1: float
    val_macro_f1: float
    test_macro_f1: float
    train_loss: float
    val_loss: float
    test_loss: float
    best_optimizer_loss: float | None


def parse_hidden_dims(value):
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def hidden_dims_text(hidden_dims):
    return ",".join(str(value) for value in hidden_dims) if hidden_dims else "linear"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare plain BP, PSO-BP, and GA-BP on DINOv2 features."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dims", default="64", type=parse_hidden_dims)
    parser.add_argument("--bp-epochs", default=35, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--l2-penalty", default=0.005, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--optimizer-samples-per-class", default=120, type=int)
    parser.add_argument("--pso-particles", default=8, type=int)
    parser.add_argument("--pso-iterations", default=6, type=int)
    parser.add_argument("--ga-population", default=8, type=int)
    parser.add_argument("--ga-generations", default=6, type=int)
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
        for row in reader:
            X.append([float(row[name]) for name in feature_names])
            y.append(int(float(row["label"])))
            splits.append(normalize_split(row["split"]))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(splits)


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


def balanced_subset(X, y, per_class, random_state):
    rng = np.random.default_rng(random_state)
    indices = []
    for label in sorted(np.unique(y)):
        label_indices = np.flatnonzero(y == label)
        take = min(int(per_class), label_indices.size)
        indices.extend(rng.choice(label_indices, size=take, replace=False))
    rng.shuffle(indices)
    return X[indices], y[indices]


def evaluate(model, X, y, num_classes):
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    stats = classification_report_stats(y, predictions, num_classes)
    return {
        "accuracy": float(np.mean(predictions == y)),
        "macro_f1": float(stats["macro_f1"]),
        "loss": float(cross_entropy_loss(probabilities, y)),
    }


def make_model(args, input_dim, output_dim, seed_offset):
    return MLPClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=output_dim,
        learning_rate=args.learning_rate,
        random_state=args.random_state + seed_offset,
        optimizer="adam",
        l2_penalty=args.l2_penalty,
    )


def evaluate_result(name, model, data, output_dim, hidden_dims, optimizer_samples, optimizer_seconds, bp_seconds, best_loss):
    metrics = {
        split: evaluate(model, values[0], values[1], output_dim)
        for split, values in data.items()
    }
    return ExperimentResult(
        model=name,
        hidden_dims=hidden_dims_text(hidden_dims),
        optimizer_samples=optimizer_samples,
        optimizer_seconds=optimizer_seconds,
        bp_seconds=bp_seconds,
        train_accuracy=metrics["train"]["accuracy"],
        val_accuracy=metrics["val"]["accuracy"],
        test_accuracy=metrics["test"]["accuracy"],
        train_macro_f1=metrics["train"]["macro_f1"],
        val_macro_f1=metrics["val"]["macro_f1"],
        test_macro_f1=metrics["test"]["macro_f1"],
        train_loss=metrics["train"]["loss"],
        val_loss=metrics["val"]["loss"],
        test_loss=metrics["test"]["loss"],
        best_optimizer_loss=best_loss,
    )


def run_plain_bp(args, data, input_dim, output_dim):
    model = make_model(args, input_dim, output_dim, 0)
    start = time.time()
    model.fit(data["train"][0], data["train"][1], epochs=args.bp_epochs, batch_size=args.batch_size)
    bp_seconds = time.time() - start
    return evaluate_result(
        "BP-small",
        model,
        data,
        output_dim,
        args.hidden_dims,
        0,
        0.0,
        bp_seconds,
        None,
    )


def run_pso_bp(args, data, optimizer_subset, input_dim, output_dim):
    model = make_model(args, input_dim, output_dim, 1)
    optimizer = ParticleSwarmBP(
        model,
        swarm_size=args.pso_particles,
        iterations=args.pso_iterations,
        initial_position_scale=0.05,
        initial_velocity_scale=0.01,
        velocity_clip=0.05,
        random_state=args.random_state + 10,
    )
    start = time.time()
    history = optimizer.optimize_initial_weights(optimizer_subset[0], optimizer_subset[1])
    optimizer_seconds = time.time() - start
    start = time.time()
    model.fit(data["train"][0], data["train"][1], epochs=args.bp_epochs, batch_size=args.batch_size)
    bp_seconds = time.time() - start
    return evaluate_result(
        "PSO-BP",
        model,
        data,
        output_dim,
        args.hidden_dims,
        optimizer_subset[1].size,
        optimizer_seconds,
        bp_seconds,
        float(history["best_loss"]),
    )


def run_ga_bp(args, data, optimizer_subset, input_dim, output_dim):
    model = make_model(args, input_dim, output_dim, 2)
    optimizer = GeneticAlgorithmBP(
        model,
        population_size=args.ga_population,
        generations=args.ga_generations,
        elite_count=2,
        mutation_rate=0.04,
        mutation_scale=0.02,
        initial_noise=0.05,
        random_state=args.random_state + 20,
    )
    start = time.time()
    history = optimizer.optimize_initial_weights(optimizer_subset[0], optimizer_subset[1])
    optimizer_seconds = time.time() - start
    start = time.time()
    model.fit(data["train"][0], data["train"][1], epochs=args.bp_epochs, batch_size=args.batch_size)
    bp_seconds = time.time() - start
    return evaluate_result(
        "GA-BP",
        model,
        data,
        output_dim,
        args.hidden_dims,
        optimizer_subset[1].size,
        optimizer_seconds,
        bp_seconds,
        float(history["best_loss"]),
    )


def save_results(output_dir, rows):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pso_ga_bp_results.csv"
    json_path = output / "pso_ga_bp_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return csv_path, json_path


def main():
    args = parse_args()
    X, y, splits = load_feature_csv(args.csv)
    raw_data = split_data(X, y, splits)
    mean, scale = normalization_stats(raw_data["train"][0])
    data = {
        split: (normalize(values[0], mean, scale), values[1])
        for split, values in raw_data.items()
    }
    output_dim = int(np.max(y)) + 1
    optimizer_subset = balanced_subset(
        data["train"][0],
        data["train"][1],
        args.optimizer_samples_per_class,
        args.random_state,
    )

    rows = [
        run_plain_bp(args, data, X.shape[1], output_dim),
        run_pso_bp(args, data, optimizer_subset, X.shape[1], output_dim),
        run_ga_bp(args, data, optimizer_subset, X.shape[1], output_dim),
    ]
    rows = sorted(rows, key=lambda row: (row.val_accuracy, row.test_accuracy), reverse=True)
    csv_path, json_path = save_results(args.output_dir, rows)

    print(
        "model,hidden_dims,val_accuracy,test_accuracy,test_macro_f1,"
        "optimizer_seconds,bp_seconds,best_optimizer_loss"
    )
    for row in rows:
        print(
            f"{row.model},{row.hidden_dims},{row.val_accuracy:.4f},"
            f"{row.test_accuracy:.4f},{row.test_macro_f1:.4f},"
            f"{row.optimizer_seconds:.2f},{row.bp_seconds:.2f},"
            f"{row.best_optimizer_loss}"
        )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
