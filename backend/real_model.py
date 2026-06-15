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
    from ai.inference import predict_base64_image

    artifact_path = _resolve_artifact_path()
    if not artifact_path.exists():
        raise NotImplementedError(
            "Model artifact is not available. Train it with "
            "`python -m ai.inference.train_classifier features.csv --output "
            "models/insect_classifier.npz` or set MODEL_ARTIFACT_PATH."
        )

    return predict_base64_image(
        image_base64,
        artifact_path=artifact_path,
        localization_mode=Config.LOCALIZATION_MODE,
        min_area_ratio=Config.LOCALIZATION_MIN_AREA_RATIO,
    )
