__all__ = [
    "ModelEvaluationResult",
    "build_default_ai_models",
    "evaluate_classifier",
    "run_csv_model_comparison",
    "run_model_comparison",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from ai.experiments import model_comparison

    return getattr(model_comparison, name)
