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


def _parse_split_ratios(value):
    ratios = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(ratios) != 3:
        raise argparse.ArgumentTypeError("split ratios must be train,val,test")
    return ratios


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
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--l2-penalty", default=0.008, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--label-column", default="-1", type=_parse_column)
    parser.add_argument("--feature-columns", nargs="*", default=None)
    parser.add_argument(
        "--split-ratios",
        default=(0.7, 0.2, 0.1),
        type=_parse_split_ratios,
        help="Train, validation, and test split ratios",
    )
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="Evaluate with train/val/test, then retrain the saved artifact on all rows",
    )
    args = parser.parse_args()

    result = train_mlp_classifier_from_csv(
        args.csv_path,
        args.output,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        l2_penalty=args.l2_penalty,
        random_state=args.random_state,
        label_column=args.label_column,
        feature_columns=args.feature_columns,
        split_ratios=args.split_ratios,
        refit_full=args.refit_full,
    )

    print(json.dumps({
        "artifact_path": str(result.artifact_path),
        "samples": result.samples,
        "train_samples": result.train_samples,
        "validation_samples": result.validation_samples,
        "test_samples": result.test_samples,
        "input_dim": result.input_dim,
        "output_dim": result.output_dim,
        "training_accuracy": result.training_accuracy,
        "validation_accuracy": result.validation_accuracy,
        "test_accuracy": result.test_accuracy,
        "training_loss": result.training_loss,
        "validation_loss": result.validation_loss,
        "test_loss": result.test_loss,
        "artifact_training_accuracy": result.artifact_training_accuracy,
        "split_ratios": result.split_ratios,
        "refit_full": result.refit_full,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
