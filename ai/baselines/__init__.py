from ai.baselines.common import BaselineResult, evaluate_baseline
from ai.baselines.decision_tree import build_decision_tree_baseline
from ai.baselines.knn import build_knn_baseline
from ai.baselines.naive_bayes import build_naive_bayes_baseline


def evaluate_default_baselines(
    X_train,
    y_train,
    X_test,
    y_test,
    knn_neighbors=5,
    random_state=42,
):
    """Train and evaluate the three sklearn baselines used for comparison."""
    estimators = (
        ("KNN", build_knn_baseline(n_neighbors=knn_neighbors)),
        ("Decision Tree", build_decision_tree_baseline(random_state=random_state)),
        ("Naive Bayes", build_naive_bayes_baseline()),
    )
    return tuple(
        evaluate_baseline(name, estimator, X_train, y_train, X_test, y_test)
        for name, estimator in estimators
    )


__all__ = [
    "BaselineResult",
    "build_decision_tree_baseline",
    "build_knn_baseline",
    "build_naive_bayes_baseline",
    "evaluate_baseline",
    "evaluate_default_baselines",
]
