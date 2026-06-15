# AI API Contract

This branch keeps the current frontend flow working while giving the AI,
backend, and frontend code a shared interface to build against.

## Detection

`POST /api/detect`

Legacy alias: `POST /detect`

Request:

```json
{
  "image": "base64 encoded image bytes without a data URL prefix"
}
```

Success response:

```json
{
  "insect_name": "Ant",
  "confidence": 0.95,
  "class_id": 0,
  "label": "ant",
  "model_status": "mock",
  "model_mode": "mock",
  "contract_version": "1.0"
}
```

Error response:

```json
{
  "error": "Invalid base64 image"
}
```

## AI Status

`GET /api/ai/status`

Response:

```json
{
  "ready": true,
  "mode": "mock",
  "status": "mock",
  "message": "Mock model is active for frontend/backend integration.",
  "contract_version": "1.0"
}
```

When `USE_MOCK_MODEL=false`, the endpoint currently reports the real adapter as
not implemented until `backend/real_model.py` is connected to trained AI code.

## Contract Discovery

`GET /api/ai/contract`

Returns detection request/response fields, feature extraction CSV expectations,
and the class list shared by backend and AI code.

## Feature Extraction Boundary

Feature extraction consumes the image dataset and YOLO `.txt` labels:

```text
dataset/train/images + dataset/train/labels
dataset/valid/images + dataset/valid/labels
dataset/test/images  + dataset/test/labels
```

It produces a CSV for model training and evaluation:

```text
feat_0,feat_1,...,feat_460,label
```

Each row represents one insect crop. Model training and evaluation code should
read that CSV, with `label` as the target column and every `feat_*` column as a
numeric feature.
