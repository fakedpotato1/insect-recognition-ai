# Insect Recognition AI

## Overview

This project is an insect image recognition system for a practical AI
coursework. The system is designed around traditional image features and
hand-written machine learning algorithms instead of pre-trained deep learning
models. Feature extraction can use external models, while the core classifier
uses the project's BP neural network with PSO/GA weight initialization. KNN,
decision tree, and Naive Bayes are kept as sklearn comparison baselines.

The target pipeline is:

```text
Image input
  -> Image preprocessing
  -> Feature extraction
  -> GA/PSO weight initialization
  -> NumPy BP neural network classifier
  -> Prediction result and insect information
```

The insect information database and extracted dataset files are expected to be
added by separate project tasks.

## Architecture

```text
insect-recognition-ai/
|-- ai/          AI algorithms, baselines, optimization, evaluation, and tests
|-- backend/     Flask API and model service integration
|-- frontend/    Vite React user interface
|-- docs/        Reports, design notes, and presentation material
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

## Dataset

The current planned image source is IP102, which contains pest and insect image
categories suitable for multi-class classification experiments. Final processed
features should be committed separately from the source images when the dataset
task is ready.

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

After feature extraction writes a CSV file, compare the hand-written AI models
against sklearn baselines with:

```bash
python -m ai.experiments.model_comparison features.csv --label-column label
```

## Coursework Notes

The implementation should prioritize:

- Clear, documented, and functional code.
- AI algorithms that are implemented directly and can be explained from their
  mathematical steps.
- A complete demonstration path from image upload to classification result once
  the feature extraction and knowledge base tasks are merged.
