import argparse
from dataclasses import dataclass

import numpy as np

from ai.baselines import evaluate_default_baselines
from ai.data import load_csv_train_val_test
from ai.evaluation.accuracy import accuracy_score
from ai.evaluation.precision_recall import classification_report_stats
from ai.neural_network import MLPClassifier
from ai.optimization import build_ga_bp_classifier, build_pso_bp_classifier


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Metrics for one model in the comparison experiment."""

    name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    predictions: np.ndarray
    history: object = None


def _as_feature_matrix(values, name):
    features = np.asarray(values, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, features)")
    return features


def _as_label_vector(values, name):
    labels = np.asarray(values, dtype=int)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return labels


def _validate_xy(features, labels, feature_name, label_name):
    if features.shape[0] != labels.size:
        raise ValueError(
            f"{feature_name} and {label_name} must contain the same number of samples"
        )
    if labels.size == 0:
        raise ValueError(f"{label_name} cannot be empty")


def _infer_output_dim(*label_arrays):
    combined = np.concatenate([labels.ravel() for labels in label_arrays if labels.size])
    if combined.size == 0:
        raise ValueError("at least one label is required")
    if np.any(combined < 0):
        raise ValueError("class labels must be non-negative integers")
    return int(np.max(combined)) + 1


def _history_from_fit(classifier, fit_result):
    if hasattr(classifier, "history_"):
        return classifier.history_
    if isinstance(fit_result, dict):
        return fit_result
    return None


def evaluate_classifier(name, classifier, X_train, y_train, X_test, y_test, fit_kwargs=None):
    """Fit a classifier and calculate the project evaluation metrics."""
    train_features = _as_feature_matrix(X_train, "X_train")
    train_labels = _as_label_vector(y_train, "y_train")
    test_features = _as_feature_matrix(X_test, "X_test")
    test_labels = _as_label_vector(y_test, "y_test")
    _validate_xy(train_features, train_labels, "X_train", "y_train")
    _validate_xy(test_features, test_labels, "X_test", "y_test")

    fit_result = classifier.fit(train_features, train_labels, **(fit_kwargs or {}))
    predictions = np.asarray(classifier.predict(test_features), dtype=int)
    output_dim = _infer_output_dim(train_labels, test_labels, predictions)
    stats = classification_report_stats(test_labels, predictions, output_dim)

    return ModelEvaluationResult(
        name=name,
        accuracy=float(accuracy_score(test_labels, predictions)),
        macro_precision=float(stats["macro_precision"]),
        macro_recall=float(stats["macro_recall"]),
        macro_f1=float(stats["macro_f1"]),
        predictions=predictions,
        history=_history_from_fit(classifier, fit_result),
    )


def _seed_offset(random_state, offset):
    return None if random_state is None else int(random_state) + offset


def _model_kwargs(base_kwargs, random_state):
    kwargs = dict(base_kwargs or {})
    kwargs.setdefault("random_state", random_state)
    return kwargs


def build_default_ai_models(
    input_dim,
    output_dim,
    hidden_dims=(16,),
    random_state=42,
    bp_model_kwargs=None,
    pso_kwargs=None,
    ga_kwargs=None,
):
    """Build the hand-written AI models used against the sklearn baselines."""
    bp_kwargs = _model_kwargs(bp_model_kwargs, random_state)
    pso_model_kwargs = _model_kwargs(bp_model_kwargs, _seed_offset(random_state, 1))
    ga_model_kwargs = _model_kwargs(bp_model_kwargs, _seed_offset(random_state, 2))

    pso_options = dict(pso_kwargs or {})
    pso_options.setdefault("random_state", _seed_offset(random_state, 3))

    ga_options = dict(ga_kwargs or {})
    ga_options.setdefault("random_state", _seed_offset(random_state, 4))

    return (
        ("BP", MLPClassifier(input_dim, hidden_dims, output_dim, **bp_kwargs)),
        (
            "PSO-BP",
            build_pso_bp_classifier(
                input_dim,
                hidden_dims,
                output_dim,
                model_kwargs=pso_model_kwargs,
                **pso_options,
            ),
        ),
        (
            "GA-BP",
            build_ga_bp_classifier(
                input_dim,
                hidden_dims,
                output_dim,
                model_kwargs=ga_model_kwargs,
                **ga_options,
            ),
        ),
    )


def run_model_comparison(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_dims=(16,),
    bp_epochs=100,
    batch_size=None,
    random_state=42,
    bp_model_kwargs=None,
    pso_kwargs=None,
    ga_kwargs=None,
    include_baselines=True,
    knn_neighbors=5,
):
    """Evaluate BP, PSO-BP, GA-BP, and optional sklearn baselines."""
    train_features = _as_feature_matrix(X_train, "X_train")
    train_labels = _as_label_vector(y_train, "y_train")
    test_features = _as_feature_matrix(X_test, "X_test")
    test_labels = _as_label_vector(y_test, "y_test")
    _validate_xy(train_features, train_labels, "X_train", "y_train")
    _validate_xy(test_features, test_labels, "X_test", "y_test")

    output_dim = _infer_output_dim(train_labels, test_labels)
    models = build_default_ai_models(
        input_dim=train_features.shape[1],
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        random_state=random_state,
        bp_model_kwargs=bp_model_kwargs,
        pso_kwargs=pso_kwargs,
        ga_kwargs=ga_kwargs,
    )

    results = []
    for name, classifier in models:
        fit_kwargs = {"batch_size": batch_size}
        if name == "BP":
            fit_kwargs["epochs"] = bp_epochs
        else:
            fit_kwargs["bp_epochs"] = bp_epochs
        results.append(
            evaluate_classifier(
                name,
                classifier,
                train_features,
                train_labels,
                test_features,
                test_labels,
                fit_kwargs=fit_kwargs,
            )
        )

    if include_baselines:
        baseline_results = evaluate_default_baselines(
            train_features,
            train_labels,
            test_features,
            test_labels,
            knn_neighbors=knn_neighbors,
            random_state=random_state,
        )
        results.extend(
            ModelEvaluationResult(
                name=result.name,
                accuracy=result.accuracy,
                macro_precision=result.macro_precision,
                macro_recall=result.macro_recall,
                macro_f1=result.macro_f1,
                predictions=result.predictions,
            )
            for result in baseline_results
        )

    return tuple(results)


def run_csv_model_comparison(
    csv_path,
    label_column=-1,
    feature_columns=None,
    split_ratios=(0.7, 0.2, 0.1),
    random_state=42,
    hidden_dims=(16,),
    bp_epochs=100,
    batch_size=None,
    knn_neighbors=5,
):
    """Load feature CSV output and run the full model comparison suite."""
    data = load_csv_train_val_test(
        csv_path,
        label_column=label_column,
        feature_columns=feature_columns,
        split_ratios=split_ratios,
        random_state=random_state,
    )
    return run_model_comparison(
        data.splits.X_train,
        data.splits.y_train,
        data.splits.X_test,
        data.splits.y_test,
        hidden_dims=hidden_dims,
        bp_epochs=bp_epochs,
        batch_size=batch_size,
        random_state=random_state,
        knn_neighbors=knn_neighbors,
    )


def _parse_column(value):
    try:
        return int(value)
    except ValueError:
        return value


def _parse_hidden_dims(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _format_results(results):
    rows = ["model,accuracy,macro_precision,macro_recall,macro_f1"]
    for result in results:
        rows.append(
            f"{result.name},{result.accuracy:.4f},{result.macro_precision:.4f},"
            f"{result.macro_recall:.4f},{result.macro_f1:.4f}"
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare BP/PSO-BP/GA-BP against sklearn baselines."
    )
    parser.add_argument("csv_path", help="CSV file produced by feature extraction")
    parser.add_argument("--label-column", default="-1", type=_parse_column)
    parser.add_argument("--feature-columns", nargs="*", default=None)
    parser.add_argument("--hidden-dims", default="16", type=_parse_hidden_dims)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--knn-neighbors", default=5, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()

    results = run_csv_model_comparison(
        args.csv_path,
        label_column=args.label_column,
        feature_columns=args.feature_columns,
        random_state=args.random_state,
        hidden_dims=args.hidden_dims,
        bp_epochs=args.epochs,
        batch_size=args.batch_size,
        knn_neighbors=args.knn_neighbors,
    )
    print(_format_results(results))


if __name__ == "__main__":
    main()
