# AI Module

This folder contains the AI work for the insect recognition project.

## Scope

- `neural_network/`: hand-written BP neural network components built with NumPy.
- `evolutionary/`: hand-written PSO and GA optimizers.
- `optimization/`: PSO-BP and GA-BP adapters that optimize BP initial weights before backpropagation.
- `baselines/`: sklearn KNN, decision tree, and Gaussian Naive Bayes models used only as comparison baselines.
- `data/`: CSV feature loading and train/validation/test splitting helpers.
- `evaluation/`: project metrics for accuracy, confusion matrix, precision, recall, and F1.

The feature extraction stage may call external models, but the core AI classifier
for this coursework is the manually implemented BP/PSO/GA code above.

## Run Tests

```bash
python -m unittest discover -s ai/tests -p "*_tests.py"
```

## Baseline Comparison

```python
from ai.experiments import run_csv_model_comparison

results = run_csv_model_comparison("features.csv", label_column="label")
for result in results:
    print(result.name, result.accuracy, result.macro_f1)
```
