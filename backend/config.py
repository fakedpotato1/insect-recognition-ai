import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    USE_MOCK_MODEL = (
        os.getenv("USE_MOCK_MODEL", "false")
        .lower()
        == "true"
    )

    PORT = int(
        os.getenv("PORT", 5000)
    )

    MODEL_ARTIFACT_PATH = os.getenv(
        "MODEL_ARTIFACT_PATH",
        "../ai/model/dinov2_bp_classifier.npz"
    )

    MODEL_FEATURE_EXTRACTOR = os.getenv(
        "MODEL_FEATURE_EXTRACTOR",
        "dinov2"
    )

    LOCALIZATION_MODE = os.getenv(
        "LOCALIZATION_MODE",
        "auto"
    )

    LOCALIZATION_MIN_AREA_RATIO = float(
        os.getenv("LOCALIZATION_MIN_AREA_RATIO", "0.002")
    )

    DINOV2_MODEL = os.getenv(
        "DINOV2_MODEL",
        "facebook/dinov2-small"
    )

    DINOV2_DEVICE = os.getenv(
        "DINOV2_DEVICE",
        "auto"
    )
