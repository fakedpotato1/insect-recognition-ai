# DINOv2 + tuned hand-written BP model

This directory contains the current model used by the backend.

## Pipeline

```text
Roboflow YOLO images
-> YOLO bbox crop
-> facebook/dinov2-small frozen feature extractor
-> ai/training/data/pest_dinov2_features_fast.csv
-> hand-written BP classifier in ai.neural_network.MLPClassifier
-> hand-written Adam optimizer
-> ai/model/dinov2_bp_classifier.npz
```

## Final tuned BP config

```text
hidden_dims: 96
optimizer: adam
learning_rate: 0.001
l2_penalty: 0.005
batch_size: 256
random_state: 7
best_epoch: 16
```

## Final metrics

```text
train accuracy: 0.9907
val accuracy:   0.7825
test accuracy:  0.7758
test macro F1:  0.7723
```

Previous model test accuracy was `0.7583`.

## PSO/GA note

PSO-BP and GA-BP use the same hand-written BP network. PSO and GA search for better initial network weights, then the same BP + Adam training continues.

In the latest comparison:

```text
GA-BP test accuracy:  0.7730
PSO-BP test accuracy: 0.7680
```

The deployed model remains tuned BP + Adam because it achieved the best validation and test accuracy in this run.
