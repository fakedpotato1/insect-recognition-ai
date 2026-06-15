import argparse
import json

from ai.inference.training import train_mlp_classifier_from_csv


def _parse_hidden_dims(value):
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_column(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(
        description="Train a production MLP classifier artifact from features.csv."
    )
    parser.add_argument("csv_path", help="CSV file produced by feature extraction")
    parser.add_argument(
        "--output",
        default="models/insect_classifier.npz",
        help="Path for the saved model artifact",
    )
    parser.add_argument("--hidden-dims", default="64", type=_parse_hidden_dims)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--label-column", default="-1", type=_parse_column)
    parser.add_argument("--feature-columns", nargs="*", default=None)
    args = parser.parse_args()

    result = train_mlp_classifier_from_csv(
        args.csv_path,
        args.output,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        label_column=args.label_column,
        feature_columns=args.feature_columns,
    )

    print(json.dumps({
        "artifact_path": str(result.artifact_path),
        "samples": result.samples,
        "input_dim": result.input_dim,
        "output_dim": result.output_dim,
        "training_accuracy": result.training_accuracy,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
