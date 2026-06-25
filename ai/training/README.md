# BP Training, Tuning, and Baseline Reproduction

This directory contains the training scripts, DINOv2 feature CSV, tuning
configuration, and saved experiment results needed to reproduce the AI model.
The model currently used by the backend is stored at:

```text
ai/model/dinov2_bp_classifier.npz
```

## Input Data

The hand-written BP model is trained from the extracted DINOv2 feature CSV:

```text
ai/training/data/pest_dinov2_features_fast.csv
```

The feature-generation pipeline is:

```text
Roboflow YOLO images
-> YOLO bounding-box crop
-> facebook/dinov2-small frozen feature extractor
-> 384-dimensional feature CSV
```

Supporting files:

```text
ai/training/data/label_mapping.json
ai/training/data/feature_stats.json
```

## Final Model Pipeline

The deployed model uses this pipeline:

```text
DINOv2 feature CSV
-> train-split mean/std normalization
-> hand-written BP network
-> hand-written Adam optimizer
-> L2 regularization
-> validation early stopping
-> ai/model/dinov2_bp_classifier.npz
```

Final configuration:

```text
hidden_dims: 96
optimizer: adam
learning_rate: 0.001
l2_penalty: 0.005
batch_size: 256
random_state: 7
best_epoch: 16
```

Final metrics:

```text
train accuracy: 0.9907
val accuracy:   0.7825
test accuracy:  0.7758
test macro F1:  0.7723
```

## Reproduce BP Tuning

```powershell
.\.venv-ml\Scripts\python.exe .\ai\training\tune_bp.py `
  --csv .\ai\training\data\pest_dinov2_features_fast.csv `
  --output-dir .\ai\training\outputs\bp_tuning_verify `
  --model-output .\ai\model\dinov2_bp_classifier.npz `
  --configs-json .\ai\training\configs\bp_tuning_round2.json `
  --patience 14
```

Saved tuning results:

```text
ai/training/results/bp_tuning_round1/
ai/training/results/bp_tuning_round2/
```

## Reproduce PSO-BP and GA-BP

PSO-BP and GA-BP do not use separate BP implementations. They reuse the same
hand-written `ai.neural_network.MLPClassifier`. The only difference is that PSO
or GA first searches for better initial BP weights, then the same BP + Adam
training process continues.

```powershell
.\.venv-ml\Scripts\python.exe .\ai\training\run_pso_ga_bp.py `
  --csv .\ai\training\data\pest_dinov2_features_fast.csv `
  --output-dir .\ai\training\outputs\pso_ga_bp `
  --hidden-dims 64 `
  --bp-epochs 35 `
  --batch-size 256 `
  --learning-rate 0.001 `
  --l2-penalty 0.005 `
  --optimizer-samples-per-class 120 `
  --pso-particles 8 `
  --pso-iterations 6 `
  --ga-population 8 `
  --ga-generations 6 `
  --random-state 42
```

Saved results:

```text
ai/training/results/pso_ga_bp/
```

## Reproduce Baselines

```powershell
.\.venv-ml\Scripts\python.exe .\ai\training\run_experiments.py `
  --csv .\ai\training\data\pest_dinov2_features_fast.csv `
  --output-dir .\ai\training\outputs\baselines_final `
  --hidden-dims 96 `
  --epochs 16 `
  --batch-size 256 `
  --learning-rate 0.001 `
  --optimizer adam `
  --l2-penalty 0.005 `
  --random-state 7 `
  --extra-trees 200
```

Report-ready baseline metrics:

```text
GaussianNB test accuracy:              0.7100
ExtraTrees-200-balanced test accuracy: 0.7428
```

Saved results:

```text
ai/training/results/baselines_final/
```
