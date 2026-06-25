from pathlib import Path
import sys

try:
    from config import Config
except ModuleNotFoundError:
    from backend.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_artifact_path():
    artifact_path = Path(Config.MODEL_ARTIFACT_PATH)
    if artifact_path.is_absolute():
        return artifact_path

    backend_dir = Path(__file__).resolve().parent
    return (backend_dir / artifact_path).resolve()


def predict(image_base64: str):
    artifact_path = _resolve_artifact_path()
    if not artifact_path.exists():
        raise NotImplementedError(
            "Model artifact is not available. Train it with "
            "`python ai/training/train_bp_artifact.py --csv <features.csv> "
            "--output ai/model/dinov2_bp_classifier.npz` or set "
            "MODEL_ARTIFACT_PATH."
        )

    feature_extractor = Config.MODEL_FEATURE_EXTRACTOR.lower()
    if feature_extractor == "dinov2":
        from ai.inference.dinov2_predictor import predict_base64_image_dinov2

        return predict_base64_image_dinov2(
            image_base64,
            artifact_path=artifact_path,
            localization_mode=Config.LOCALIZATION_MODE,
            min_area_ratio=Config.LOCALIZATION_MIN_AREA_RATIO,
            dinov2_model=Config.DINOV2_MODEL,
            device=Config.DINOV2_DEVICE,
        )

    if feature_extractor != "classic":
        raise ValueError(
            "MODEL_FEATURE_EXTRACTOR must be 'dinov2' or 'classic'"
        )

    from ai.inference import predict_base64_image

    return predict_base64_image(
        image_base64,
        artifact_path=artifact_path,
        localization_mode=Config.LOCALIZATION_MODE,
        min_area_ratio=Config.LOCALIZATION_MIN_AREA_RATIO,
    )
