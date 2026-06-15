import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    USE_MOCK_MODEL = (
        os.getenv("USE_MOCK_MODEL", "true")
        .lower()
        == "true"
    )

    PORT = int(
        os.getenv("PORT", 5000)
    )

    MODEL_ARTIFACT_PATH = os.getenv(
        "MODEL_ARTIFACT_PATH",
        "../models/insect_classifier.npz"
    )

    LOCALIZATION_MODE = os.getenv(
        "LOCALIZATION_MODE",
        "auto"
    )

    LOCALIZATION_MIN_AREA_RATIO = float(
        os.getenv("LOCALIZATION_MIN_AREA_RATIO", "0.002")
    )
