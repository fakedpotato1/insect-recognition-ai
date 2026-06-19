from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray
    bbox: tuple
    mode: str
    used_fallback: bool = False


def _full_image_crop(image_bgr, mode):
    height, width = image_bgr.shape[:2]
    return CropResult(
        image=image_bgr,
        bbox=(0, 0, width, height),
        mode=mode,
        used_fallback=True,
    )


def _padded_bbox(x, y, width, height, image_width, image_height, padding_ratio):
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + width + pad_x)
    y2 = min(image_height, y + height + pad_y)
    return x1, y1, x2 - x1, y2 - y1


def locate_insect_crop(
    image_bgr,
    mode="auto",
    min_area_ratio=0.002,
    max_area_ratio=0.95,
    padding_ratio=0.12,
):
    """
    Locate a likely insect crop in a single uploaded image.

    "auto" is a conservative OpenCV contour heuristic for MVP inference. If no
    convincing crop is found, the full image is used so classification can still
    run for already-cropped insect photos.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image is empty")

    if mode == "full":
        return _full_image_crop(image_bgr, mode="full")
    if mode != "auto":
        raise ValueError("localization mode must be 'auto' or 'full'")

    image_height, image_width = image_bgr.shape[:2]
    image_area = image_height * image_width
    min_area = image_area * float(min_area_ratio)
    max_area = image_area * float(max_area_ratio)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation_mask = cv2.inRange(hsv[:, :, 1], 35, 255)
    mask = cv2.bitwise_or(edges, saturation_mask)
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    center_x = image_width / 2
    center_y = image_height / 2
    diagonal = np.hypot(image_width, image_height)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue

        bbox_center_x = x + width / 2
        bbox_center_y = y + height / 2
        center_distance = np.hypot(bbox_center_x - center_x, bbox_center_y - center_y)
        center_score = 1.0 - min(center_distance / (diagonal + 1e-6), 1.0)
        score = area * (0.75 + 0.25 * center_score)
        candidates.append((score, x, y, width, height))

    if not candidates:
        return _full_image_crop(image_bgr, mode="auto")

    _, x, y, width, height = max(candidates, key=lambda item: item[0])
    x, y, width, height = _padded_bbox(
        x,
        y,
        width,
        height,
        image_width,
        image_height,
        padding_ratio,
    )
    crop = image_bgr[y : y + height, x : x + width]
    return CropResult(image=crop, bbox=(x, y, width, height), mode="auto")
