from config import Config

import mock_model
import real_model


CONTRACT_VERSION = "1.0"

CLASS_NAMES = [
    "ant",
    "bed-bug",
    "bee",
    "beetle",
    "bernsteinschabe",
    "cockroach",
    "fly",
    "fruitfly",
    "grasshopper",
    "hornet",
    "housefly",
    "ladybug",
    "mosquito",
    "moth",
    "silverfish",
    "slug",
    "snail",
    "spider",
    "tiger mosquito",
    "wasp",
]


def _display_name(label):
    return label.replace("-", " ").title()


def get_model_status():
    if Config.USE_MOCK_MODEL:
        return {
            "ready": True,
            "mode": "mock",
            "status": "mock",
            "message": "Mock model is active for frontend/backend integration.",
        }

    return {
        "ready": False,
        "mode": "real",
        "status": "not_implemented",
        "message": "Real AI model adapter is not implemented yet.",
    }


def get_ai_contract():
    return {
        "contract_version": CONTRACT_VERSION,
        "detect": {
            "method": "POST",
            "paths": ["/detect", "/api/detect"],
            "content_type": "application/json",
            "request_body": {
                "image": "base64 encoded image bytes without a data URL prefix",
            },
            "success_response": {
                "insect_name": "display name for the predicted class",
                "confidence": "float from 0.0 to 1.0",
                "class_id": "integer class id, or null when unavailable",
                "label": "machine-readable class label",
                "model_status": "mock, ready, or not_implemented",
                "contract_version": CONTRACT_VERSION,
            },
            "error_responses": {
                "400": "missing or invalid request image",
                "501": "real model adapter has not been implemented",
                "500": "unexpected server error",
            },
        },
        "feature_extraction": {
            "purpose": "Produce the CSV consumed by training and evaluation code.",
            "input": "dataset/{train,valid,test}/{images,labels} with YOLO txt labels",
            "output_csv": {
                "feature_columns": "feat_0 through feat_460",
                "label_column": "label",
                "row_granularity": "one row per insect crop",
            },
        },
        "classes": [
            {
                "class_id": class_id,
                "label": label,
                "insect_name": _display_name(label),
            }
            for class_id, label in enumerate(CLASS_NAMES)
        ],
    }


def _normalize_prediction(raw_prediction):
    if not isinstance(raw_prediction, dict):
        raise ValueError("model prediction must be a dictionary")

    if "insect_name" not in raw_prediction or "confidence" not in raw_prediction:
        raise ValueError("model prediction requires insect_name and confidence")

    confidence = float(raw_prediction["confidence"])
    if confidence < 0 or confidence > 1:
        raise ValueError("model confidence must be between 0.0 and 1.0")

    class_id = raw_prediction.get("class_id")
    if class_id is not None:
        class_id = int(class_id)

    label = raw_prediction.get("label")

    if label is None and class_id is not None and 0 <= class_id < len(CLASS_NAMES):
        label = CLASS_NAMES[class_id]

    status = get_model_status()

    return {
        "insect_name": str(raw_prediction["insect_name"]),
        "confidence": confidence,
        "class_id": class_id,
        "label": label,
        "model_status": status["status"],
        "model_mode": status["mode"],
        "contract_version": CONTRACT_VERSION,
    }


def predict(image):
    if Config.USE_MOCK_MODEL:
        return _normalize_prediction(mock_model.predict(image))

    return _normalize_prediction(real_model.predict(image))
