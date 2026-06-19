import base64
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ai.inference.artifact import load_artifact
from ai.inference.localization import locate_insect_crop


def decode_base64_image(image_base64):
    image_bytes = base64.b64decode(image_base64, validate=True)
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("base64 image could not be decoded")
    return image_bgr


def extract_feature_vector(
    image_bgr,
    localization_mode="auto",
    min_area_ratio=0.002,
):
    from ai.feature_extraction.feature_extraction import FEATURE_DIM, extract_features

    crop = locate_insect_crop(
        image_bgr,
        mode=localization_mode,
        min_area_ratio=min_area_ratio,
    )
    features = extract_features(crop.image)
    if features.shape[0] != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} features, got {features.shape[0]}")
    return features, crop


@lru_cache(maxsize=2)
def _cached_artifact(artifact_path):
    return load_artifact(Path(artifact_path))


def predict_image_array(
    image_bgr,
    artifact_path,
    localization_mode="auto",
    min_area_ratio=0.002,
    use_cache=True,
):
    artifact = (
        _cached_artifact(str(Path(artifact_path)))
        if use_cache
        else load_artifact(artifact_path)
    )
    features, crop = extract_feature_vector(
        image_bgr,
        localization_mode=localization_mode,
        min_area_ratio=min_area_ratio,
    )
    prediction = artifact.predict_one(features)

    return {
        "insect_name": prediction.insect_name,
        "confidence": prediction.confidence,
        "class_id": prediction.class_id,
        "label": prediction.label,
        "crop_box": {
            "x": int(crop.bbox[0]),
            "y": int(crop.bbox[1]),
            "width": int(crop.bbox[2]),
            "height": int(crop.bbox[3]),
        },
        "localization_mode": crop.mode,
        "localization_fallback": crop.used_fallback,
    }


def predict_base64_image(
    image_base64,
    artifact_path,
    localization_mode="auto",
    min_area_ratio=0.002,
):
    image_bgr = decode_base64_image(image_base64)
    return predict_image_array(
        image_bgr,
        artifact_path=artifact_path,
        localization_mode=localization_mode,
        min_area_ratio=min_area_ratio,
    )
