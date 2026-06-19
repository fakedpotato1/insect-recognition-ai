# Production Inference Workflow

This workflow connects feature extraction output to the Flask `/detect` path.

## 1. Produce Features

The feature extraction owner should run the extraction script against the local
image dataset and YOLO labels:

```bash
python -m ai.feature_extraction.feature_extraction
```

The output CSV should have 461 feature columns and one label column:

```text
feat_0,feat_1,...,feat_460,label
```

Large generated CSV files should not be committed unless the team explicitly
decides to version a small sample. Prefer sharing the full dataset artifact
through release assets, cloud storage, or another agreed data handoff.

## 2. Train A Saved Classifier

Train a loadable artifact from the CSV:

```bash
python -m ai.inference.train_classifier features.csv \
  --output models/insect_classifier.npz \
  --hidden-dims 64 \
  --epochs 120 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --optimizer adam \
  --l2-penalty 0.008 \
  --split-ratios 0.7,0.2,0.1
```

Training reports train, validation, and test metrics. Use those metrics for
model selection. After choosing hyperparameters, add `--refit-full` if you want
the saved production artifact to be retrained on every row after reporting split
metrics.

The artifact stores:

- MLP model weights
- feature normalization statistics
- feature names
- label mappings
- train/validation/test metrics and training metadata

You can test the saved artifact locally before connecting the backend:

```bash
python -m ai.inference.predict_image path/to/insect.jpg \
  --model models/insect_classifier.npz \
  --localization-mode auto
```

## 3. Configure Backend

Set backend environment variables:

```env
USE_MOCK_MODEL=false
MODEL_ARTIFACT_PATH=../models/insect_classifier.npz
LOCALIZATION_MODE=auto
LOCALIZATION_MIN_AREA_RATIO=0.002
```

`LOCALIZATION_MODE=auto` uses an OpenCV contour heuristic to crop the most likely
insect region before feature extraction. Use `LOCALIZATION_MODE=full` when users
upload already-cropped single-insect images.

## 4. Runtime Prediction

The runtime path is:

```text
base64 image
  -> OpenCV decode
  -> locate crop or use full image
  -> extract 461-dim feature vector
  -> normalize features from artifact stats
  -> MLP predict_proba
  -> backend JSON response
```

The response includes the predicted class, confidence, and crop box metadata.

## Remaining Inputs Needed

To make this production-ready, the project still needs:

- the final `features.csv` generated from the agreed dataset
- the trained `models/insect_classifier.npz` artifact
- a decision on whether uploaded images are already cropped or need stronger
  object detection than the current OpenCV heuristic
