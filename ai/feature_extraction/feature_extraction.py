"""
feature_extraction.py
---------------------
Pipeline per image:
  Read Image → Read YOLO Labels → Crop Each Insect →
  Feature Extraction → One Feature Vector Per Insect → CSV

Each row in features.csv represents ONE insect crop, not one image.

Feature breakdown (461 dims total)
-----------------------------------
  HSV Histogram   :  32-dim   (Hue 16 + Sat 8 + Val 8 bins)
  LBP             :  64-dim   (default LBP, P=8, R=1, 64-bin histogram)
  HOG             : 324-dim   (9 orientations, 32x32 cells, 2x2 blocks)
  Hu Moments      :  10-dim   (7 log Hu + 3 central moment ratios)
  Contour Stats   :   7-dim   (Area, Perimeter, Aspect Ratio, Circularity,
                               Solidity, Extent, Equivalent Diameter)
  GLCM            :  24-dim   (6 props x 4 angles x 1 distance)
  ─────────────────────────────────────────────────────
  Total           : 461 dimensions

YOLO Label Format (each line in .txt file)
------------------------------------------
  class_id  cx  cy  w  h
  Example:  0 0.50 0.40 0.20 0.30
"""

import csv
import logging
import multiprocessing
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG 
# ─────────────────────────────────────────────────────────────
DATASET_ROOT = "./dataset"
OUTPUT_CSV   = "features.csv"
IMG_SIZE     = (128, 128)
N_WORKERS    = 8        # parallel processes
MIN_CROP_PX  = 20       # crops smaller than this (px) are skipped

CLASS_NAMES = [
    "ant",             # 0
    "bed-bug",         # 1
    "bee",             # 2
    "beetle",          # 3
    "bernsteinschabe", # 4
    "cockroach",       # 5
    "fly",             # 6
    "fruitfly",        # 7
    "grasshopper",     # 8
    "hornet",          # 9
    "housefly",        # 10
    "ladybug",         # 11
    "mosquito",        # 12
    "moth",            # 13
    "silverfish",      # 14
    "slug",            # 15
    "snail",           # 16
    "spider",          # 17
    "tiger mosquito",  # 18
    "wasp",            # 19
]
# ─────────────────────────────────────────────────────────────

FEATURE_DIM = 461  # HSV(32) + LBP(64) + HOG(324) + Hu(10) + Contour(7) + GLCM(24)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════

# ── FEATURE 1: HSV HISTOGRAM  →  32 dims ───────────────────────
def extract_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
    feat   = np.concatenate([h_hist, s_hist, v_hist])
    total  = feat.sum()
    if total > 0:
        feat /= total
    return feat.astype(np.float32)   # 32-dim

# ── FEATURE 2: LBP HISTOGRAM  →  64 dims ───────────────────────
_LBP_P      = 8
_LBP_R      = 1.0
_LBP_N_BINS = 64

def extract_lbp(gray: np.ndarray) -> np.ndarray:
    lbp  = local_binary_pattern(gray, _LBP_P, _LBP_R, method="default")
    hist, _ = np.histogram(lbp.ravel(), bins=_LBP_N_BINS,
                           range=(0, 2 ** _LBP_P), density=True)
    return hist.astype(np.float32)   # 64-dim


# ── FEATURE 3: HOG  →  324 dims ────────────────────────────────
def extract_hog(gray: np.ndarray) -> np.ndarray:
    feat, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    return feat.astype(np.float32)   # 324-dim

# ── FEATURE 4: HU MOMENTS  →  10 dims ──────────────────────────
def extract_hu_moments(gray: np.ndarray) -> np.ndarray:
    _, bw   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(bw)

    hu     = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    m00  = moments["m00"] + 1e-10
    nu20 = moments["mu20"] / (m00 ** 2.0)
    nu02 = moments["mu02"] / (m00 ** 2.0)
    nu11 = moments["mu11"] / (m00 ** 2.0)

    return np.concatenate(
        [hu_log, np.array([nu20, nu02, nu11], dtype=np.float32)]
    ).astype(np.float32)   # 10-dim


# ── FEATURE 5: CONTOUR STATS  →  7 dims ────────────────────────
def extract_contour(gray: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_h * img_w
    diag     = np.sqrt(img_h ** 2 + img_w ** 2)

    if len(contours) == 0:
        return np.zeros(7, dtype=np.float32)

    cnt         = max(contours, key=cv2.contourArea)
    area        = cv2.contourArea(cnt)
    perimeter   = cv2.arcLength(cnt, True)
    _, _, bw_, bh_ = cv2.boundingRect(cnt)
    aspect_ratio   = bw_ / (bh_ + 1e-6)
    extent         = area / (bw_ * bh_ + 1e-6)
    circularity    = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    hull           = cv2.convexHull(cnt)
    solidity       = area / (cv2.contourArea(hull) + 1e-6)
    equiv_diam     = np.sqrt(4 * area / np.pi)

    return np.array([
        area       / (img_area + 1e-6),
        perimeter  / (diag + 1e-6),
        aspect_ratio,
        np.clip(circularity, 0.0, 1.0),
        np.clip(solidity,    0.0, 1.0),
        np.clip(extent,      0.0, 1.0),
        equiv_diam / (diag + 1e-6),
    ], dtype=np.float32)   # 7-dim


# ── FEATURE 6: GLCM  →  24 dims ────────────────────────────────
_GLCM_DISTANCES = [1]
_GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_GLCM_LEVELS    = 32
_GLCM_PROPS     = ["contrast", "dissimilarity", "homogeneity",
                   "energy", "correlation", "ASM"]

def extract_glcm(gray: np.ndarray) -> np.ndarray:
    gray_q = (gray // 8).astype(np.uint8)   # quantise 256 → 32 levels
    glcm   = graycomatrix(gray_q,
                          distances=_GLCM_DISTANCES,
                          angles=_GLCM_ANGLES,
                          levels=_GLCM_LEVELS,
                          symmetric=True, normed=True)
    feat = [graycoprops(glcm, p).flatten() for p in _GLCM_PROPS]
    return np.concatenate(feat).astype(np.float32)   # 24-dim


# ═══════════════════════════════════════════════════════════════
# COMBINED PIPELINE  →  461 dims
# ═══════════════════════════════════════════════════════════════
def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Resize crop → compute gray ONCE → extract all features.
    Returns float32 vector of length 461.
    """
    img  = cv2.resize(img_bgr, IMG_SIZE)
    h, w = img.shape[:2]

    # ── Precompute grayscale ONCE — shared by all extractors ─────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ─────────────────────────────────────────────────────────────

    hsv     = extract_hsv(img)             #  32 
    lbp     = extract_lbp(gray)            #  64  
    hog_f   = extract_hog(gray)            # 324  
    hu      = extract_hu_moments(gray)     #  10  
    contour = extract_contour(gray, h, w)  #   7  
    glcm    = extract_glcm(gray)           #  24  
                                           # ─────
                                           # 461 total
    return np.concatenate(
        [hsv, lbp, hog_f, hu, contour, glcm]
    ).astype(np.float32)

# ═══════════════════════════════════════════════════════════════
# YOLO HELPERS
# ═══════════════════════════════════════════════════════════════
def parse_yolo_label(label_path: Path):
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                cx, cy   = float(parts[1]), float(parts[2])
                bw, bh   = float(parts[3]), float(parts[4])
                annotations.append((class_id, cx, cy, bw, bh))
            except ValueError:
                continue
    return annotations


def crop_insect(img_bgr: np.ndarray,
                cx: float, cy: float,
                bw: float, bh: float):
    img_h, img_w = img_bgr.shape[:2]
    x1 = max(0, int((cx - bw / 2) * img_w))
    y1 = max(0, int((cy - bh / 2) * img_h))
    x2 = min(img_w, int((cx + bw / 2) * img_w))
    y2 = min(img_h, int((cy + bh / 2) * img_h))
    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
        return None
    return img_bgr[y1:y2, x1:x2]

# ═══════════════════════════════════════════════════════════════
# WORKER FUNCTION  –  runs inside each subprocess
# Must be a top-level function (not nested) for pickle/multiprocessing
# ═══════════════════════════════════════════════════════════════
def process_single_image(args):
    """
    Processes one (img_path, label_path) pair.
    Returns (rows, skips):
      rows  — list of row-dicts (one per valid insect crop)
      skips — list of skip-dicts describing every skipped crop with reason
    """
    img_path, label_path = args
    rows  = []
    skips = []

    # ── Read image ───────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        skips.append({
            "source_image": str(img_path),
            "crop_index":   -1,
            "class_id":     -1,
            "reason":       "cv2.imread failed — file unreadable or corrupted",
        })
        return rows, skips

    # ── Read bounding boxes ──────────────────────────────────────
    annotations = parse_yolo_label(label_path)
    if not annotations:
        skips.append({
            "source_image": str(img_path),
            "crop_index":   -1,
            "class_id":     -1,
            "reason":       "label file empty — no bounding boxes found",
        })
        return rows, skips

    # ── Process each insect crop ─────────────────────────────────
    for crop_idx, (class_id, cx, cy, bw, bh) in enumerate(annotations):

        # Compute pixel coords for skip reporting
        ih, iw = img.shape[:2]
        x1 = max(0, int((cx - bw / 2) * iw))
        y1 = max(0, int((cy - bh / 2) * ih))
        x2 = min(iw, int((cx + bw / 2) * iw))
        y2 = min(ih, int((cy + bh / 2) * ih))
        crop_w = x2 - x1
        crop_h = y2 - y1

        skip_base = {
            "source_image": str(img_path),
            "crop_index":   crop_idx,
            "class_id":     class_id,
        }

        # Check: crop too small
        if crop_w < MIN_CROP_PX or crop_h < MIN_CROP_PX:
            skips.append({**skip_base,
                          "reason": f"crop too small ({crop_w}x{crop_h}px "
                                    f"< {MIN_CROP_PX}px threshold)"})
            continue

        crop = img[y1:y2, x1:x2]

        # Check: feature extraction error
        try:
            features = extract_features(crop)
        except Exception as exc:
            skips.append({**skip_base,
                          "reason": f"feature extraction exception: {exc}"})
            continue

        # Check: wrong output dimension
        if features.shape[0] != FEATURE_DIM:
            skips.append({**skip_base,
                          "reason": f"wrong feature dim {features.shape[0]} "
                                    f"(expected {FEATURE_DIM})"})
            continue

        class_name = (CLASS_NAMES[class_id]
                      if class_id < len(CLASS_NAMES)
                      else f"class_{class_id}")

        row = {f"feat_{i}": features[i] for i in range(FEATURE_DIM)}
        row["label"] = class_id
        rows.append(row)

    return rows, skips

# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════
def run_diagnostics(root: str):
    root_path = Path(root)
    log.info("=" * 65)
    log.info("DIAGNOSTIC SCAN  –  %s", root_path.resolve())
    log.info("=" * 65)

    all_files  = list(root_path.rglob("*"))
    all_images = [f for f in all_files
                  if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]

    log.info("Total files in dataset  : %d", len(all_files))
    log.info("Total images found      : %d", len(all_images))
    log.info("-" * 65)

    log.info("Images per top-level sub-folder:")
    for cdir in sorted([d for d in root_path.iterdir() if d.is_dir()]):
        imgs = [f for f in cdir.rglob("*")
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]
        log.info("  %-35s %d images", cdir.name, len(imgs))
    log.info("-" * 65)

    ext_counts = Counter(f.suffix for f in all_images)
    log.info("Extension breakdown:")
    for ext, count in sorted(ext_counts.items()):
        flag = "  ← uppercase! may be missed" if ext != ext.lower() else ""
        log.info("  %-12s %d%s", ext, count, flag)

    root_depth   = len(root_path.parts)
    depth_counts = Counter(len(f.parts) - root_depth for f in all_images)
    log.info("Depth breakdown:")
    for depth, count in sorted(depth_counts.items()):
        flag = "  ← TOO DEEP" if depth > 2 else ""
        log.info("  depth %-4d %d images%s", depth, count, flag)
    log.info("=" * 65)

def pre_scan_insects(pairs: list):
    class_counter  = Counter()
    multi_insect   = 0
    total_insects  = 0
    empty_labels   = 0

    log.info("=" * 65)
    log.info("PRE-SCAN  –  counting insects across all label files ...")
    log.info("=" * 65)

    for _, label_path in tqdm(pairs, desc="Pre-scanning labels"):
        annotations = parse_yolo_label(label_path)
        n = len(annotations)
        if n == 0:
            empty_labels += 1
            continue
        if n > 1:
            multi_insect += 1
        for class_id, *_ in annotations:
            class_counter[class_id] += 1
            total_insects += 1

    log.info("Total images scanned          : %d", len(pairs))
    log.info("Total insect instances found  : %d", total_insects)
    log.info("Images with multiple insects  : %d", multi_insect)
    log.info("Images with empty label file  : %d", empty_labels)
    log.info("-" * 65)
    log.info("Insects per class (before extraction):")

    max_count = max(class_counter.values()) if class_counter else 1
    for class_id in sorted(class_counter):
        name  = (CLASS_NAMES[class_id]
                 if class_id < len(CLASS_NAMES) else f"class_{class_id}")
        count = class_counter[class_id]
        bar   = "█" * max(1, round(count / max_count * 30))
        log.info("  [%2d] %-20s %5d  %s", class_id, name, count, bar)
    log.info("=" * 65)


def collect_image_label_pairs(root: str):
    root_path = Path(root)
    pairs     = []
    no_label  = 0

    for split in ("train", "valid", "test"):
        img_dir   = root_path / split / "images"
        label_dir = root_path / split / "labels"
        if not img_dir.exists():
            log.warning("Split folder not found, skipping: %s", img_dir)
            continue
        imgs = sorted([f for f in img_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS])
        for img_path in imgs:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                no_label += 1
                continue
            pairs.append((img_path, label_path))

    log.info("Image-label pairs: %d  |  Skipped (no label): %d",
             len(pairs), no_label)
    return pairs

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    run_diagnostics(DATASET_ROOT)
    pairs = collect_image_label_pairs(DATASET_ROOT)
    if not pairs:
        log.error("No image-label pairs found. Check DATASET_ROOT.")
        return

    pre_scan_insects(pairs)

    log.info("Starting feature extraction with %d workers ...", N_WORKERS)

    fieldnames = (
        [f"feat_{i}" for i in range(FEATURE_DIM)]
        + ["label"]
    )

    written       = 0
    class_counter = Counter()
    all_skips     = []   # collects every skipped crop with its reason

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with multiprocessing.Pool(processes=N_WORKERS) as pool:
            for rows, skips in tqdm(
                pool.imap(process_single_image, pairs),
                total=len(pairs),
                desc="Processing images",
            ):
                for row in rows:
                    writer.writerow(row)
                    written += 1
                    class_counter[row["label"]] += 1

                all_skips.extend(skips)

    # ── Skip report ───────────────────────────────────────────────
    if all_skips:
        reason_counter = Counter(s["reason"].split(":")[0].split("(")[0].strip()
                                 for s in all_skips)
        log.info("SKIP REPORT  (%d crops skipped):", len(all_skips))
        for reason, count in reason_counter.most_common():
            log.info("  %-50s  %d", reason, count)

    # ── Final summary ─────────────────────────────────────────────
    log.info("=" * 65)
    log.info("EXTRACTION COMPLETE")
    log.info("=" * 65)
    log.info("Total insect instances (pre-scan) : see PRE-SCAN above")
    log.info("Total crops written               : %d", written)
    log.info("Total crops skipped               : %d", len(all_skips))
    log.info("-" * 65)
    log.info("Insect counts per class (written):")
    max_count = max(class_counter.values()) if class_counter else 1
    for class_id in sorted(class_counter):
        name  = (CLASS_NAMES[class_id]
                 if class_id < len(CLASS_NAMES) else f"class_{class_id}")
        count = class_counter[class_id]
        bar   = "█" * max(1, round(count / max_count * 30))
        log.info("  [%2d] %-20s %5d  %s", class_id, name, count, bar)
    log.info("=" * 65)
    log.info("CSV saved → %s", OUTPUT_CSV)


if __name__ == "__main__":
    # Required on Windows — prevents spawning recursive subprocesses
    multiprocessing.freeze_support()
    main()