# Insect Recognition AI

## Overview

This project is an insect image recognition system for a practical AI
coursework. The final pipeline uses a frozen DINOv2 feature extractor and a
hand-written NumPy BP neural network classifier. PSO and GA are implemented as
optional initial-weight optimizers for the same BP model, while KNN, decision
tree, Naive Bayes, and Extra Trees are kept as comparison baselines.

The target pipeline is:

```text
Image input
  -> Image preprocessing
  -> DINOv2 feature extraction
  -> optional GA/PSO weight initialization
  -> hand-written NumPy BP neural network classifier
  -> Prediction result and insect information
```

The deployed backend model artifact is stored in
`ai/model/dinov2_bp_classifier.npz`.

## Architecture

```text
insect-recognition-ai/
|-- ai/          AI algorithms, baselines, optimization, evaluation, and tests
|-- backend/     Flask API and model service integration
|-- frontend/    Vite React user interface
|-- docs/        API and production inference notes
|-- README.md    Project overview and setup guide
```

## AI Design

The classifier side follows a decoupled feature-to-classification interface:

```text
X: NumPy array with shape (samples, feature_dim)
y: one-dimensional integer labels with shape (samples,)
```

Labels must be encoded from `0` to `num_classes - 1`. This keeps the
cross-entropy and gradient calculation simple and avoids shape conflicts during
NumPy indexing.

Planned algorithm responsibilities:

- `ai/neural_network/`: hand-written Dense layers, forward propagation,
  backpropagation, Softmax, cross-entropy loss, and an MLP classifier.
- `ai/evolutionary/`: hand-written GA and PSO optimizers.
- `ai/optimization/`: GA-BP and PSO-BP integration that maps all network
  weights and biases into one vector for global initialization.
- `ai/baselines/`: sklearn KNN, decision tree, and Naive Bayes models for
  comparison only.
- `ai/evaluation/`: accuracy, confusion matrix, precision, recall, and F1
  metrics.

## Dataset and Features

The dataset is sourced from Roboflow and is available as a fork at:

**https://universe.roboflow.com/chingyang-tan/pest-detection-vuziq-mzuwk**

It contains images of **20 common household insect classes** with bounding-box
annotations for object detection and preprocessing. Source images are not
required to run the submitted application. The reproducible DINOv2 feature CSV
used for training and baseline comparison is stored at:

```text
ai/training/data/pest_dinov2_features_fast.csv
```


## Development

### Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer
- npm
- Git

### Clone

```bash
git clone https://github.com/fakedpotato1/insect-recognition-ai.git
cd insect-recognition-ai
```

### Backend

Open one terminal from the project root.

PowerShell:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r .\backend\requirements.txt
Copy-Item .\backend\.env.example .\backend\.env
.\.venv-ml\Scripts\python.exe .\backend\app.py
```

macOS/Linux:

```bash
python3 -m venv .venv-ml
./.venv-ml/bin/python -m pip install --upgrade pip
./.venv-ml/bin/python -m pip install -r ./backend/requirements.txt
cp ./backend/.env.example ./backend/.env
./.venv-ml/bin/python ./backend/app.py
```

The backend runs at:

```text
http://127.0.0.1:5000
```

### Frontend

Open a second terminal from the project root.

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Then open:

```text
http://127.0.0.1:5173/
```

The frontend sends uploaded images to the backend API at
`http://127.0.0.1:5000`.

### AI Tests

```bash
python -m unittest discover -s ai/tests -p "*_tests.py"
```

### Model Comparison

Compare the hand-written BP model and sklearn baselines with:

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

## Coursework Notes

The implementation should prioritize:

- Clear, documented, and functional code.
- AI algorithms that are implemented directly and can be explained from their
  mathematical steps.
- A complete demonstration path from image upload to classification result once
  the feature extraction and knowledge base tasks are merged.
