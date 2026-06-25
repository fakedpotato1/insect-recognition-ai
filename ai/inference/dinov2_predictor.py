from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ai.inference.artifact import load_artifact
from ai.inference.localization import locate_insect_crop
from ai.inference.predictor import decode_base64_image


DEFAULT_DINOV2_MODEL = "facebook/dinov2-small"


def _select_device(choice):
    import torch

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if choice == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=2)
def _cached_dinov2(model_name, device_name):
    from transformers import AutoImageProcessor, AutoModel

    import torch

    device = torch.device(device_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model


@lru_cache(maxsize=2)
def _cached_artifact(artifact_path):
    return load_artifact(Path(artifact_path))


def _crop_to_pil(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _extract_feature(image, processor, model, device):
    import torch

    inputs = processor(images=[image], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        # The CLS token is the 384-dimensional image descriptor used by the
        # training CSV, so inference extracts and normalizes it the same way.
        vector = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector = vector / norm
    return vector.astype(float)


def predict_image_array_dinov2(
    image_bgr,
    artifact_path,
    localization_mode="auto",
    min_area_ratio=0.002,
    dinov2_model=DEFAULT_DINOV2_MODEL,
    device="auto",
    use_cache=True,
):
    resolved_device = _select_device(device)
    processor, feature_model = _cached_dinov2(dinov2_model, str(resolved_device))
    artifact = (
        _cached_artifact(str(Path(artifact_path)))
        if use_cache
        else load_artifact(artifact_path)
    )

    crop = locate_insect_crop(
        image_bgr,
        mode=localization_mode,
        min_area_ratio=min_area_ratio,
    )
    features = _extract_feature(
        _crop_to_pil(crop.image),
        processor,
        feature_model,
        resolved_device,
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
        "feature_extractor": "dinov2",
        "dinov2_model": dinov2_model,
        "device": str(resolved_device),
    }


def predict_base64_image_dinov2(
    image_base64,
    artifact_path,
    localization_mode="auto",
    min_area_ratio=0.002,
    dinov2_model=DEFAULT_DINOV2_MODEL,
    device="auto",
):
    image_bgr = decode_base64_image(image_base64)
    return predict_image_array_dinov2(
        image_bgr,
        artifact_path=artifact_path,
        localization_mode=localization_mode,
        min_area_ratio=min_area_ratio,
        dinov2_model=dinov2_model,
        device=device,
    )
