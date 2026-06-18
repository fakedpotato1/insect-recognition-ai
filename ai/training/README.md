# BP 训练、调参与 baseline 复现说明

本目录保存用于复现实验的训练脚本、DINOv2 特征 CSV、调参配置和报告结果。当前前后端对接模型位于：

```text
ai/model/dinov2_bp_classifier.npz
```

## 数据输入

手写 BP 训练使用已经提取好的 DINOv2 特征 CSV：

```text
ai/training/data/pest_dinov2_features_fast.csv
```

这条 CSV 的生成流程是：

```text
Roboflow YOLO images
-> YOLO bbox crop
-> facebook/dinov2-small frozen feature extractor
-> 384-dim feature CSV
```

配套文件：

```text
ai/training/data/label_mapping.json
ai/training/data/feature_stats.json
```

## 最终模型训练线

最终部署模型使用：

```text
DINOv2 feature CSV
-> train split mean/std normalization
-> hand-written BP network
-> hand-written Adam optimizer
-> L2 regularization
-> validation early stopping
-> ai/model/dinov2_bp_classifier.npz
```

最终配置：

```text
hidden_dims: 96
optimizer: adam
learning_rate: 0.001
l2_penalty: 0.005
batch_size: 256
random_state: 7
best_epoch: 16
```

最终指标：

```text
train accuracy: 0.9907
val accuracy:   0.7825
test accuracy:  0.7758
test macro F1:  0.7723
```

## 复现 BP 调参

```powershell
.\.venv-ml\Scripts\python.exe .\ai\training\tune_bp.py `
  --csv .\ai\training\data\pest_dinov2_features_fast.csv `
  --output-dir .\ai\training\outputs\bp_tuning_verify `
  --model-output .\ai\model\dinov2_bp_classifier.npz `
  --configs-json .\ai\training\configs\bp_tuning_round2.json `
  --patience 14
```

调参结果已保存到：

```text
ai/training/results/bp_tuning_round1/
ai/training/results/bp_tuning_round2/
```

## 复现 PSO-BP / GA-BP

PSO-BP 和 GA-BP 不是三套不同 BP 写法。它们复用同一个手写 `ai.neural_network.MLPClassifier`，区别是 PSO/GA 先搜索 BP 网络的初始权重，然后继续运行手写 BP + Adam。

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

结果已保存到：

```text
ai/training/results/pso_ga_bp/
```

## 复现 baseline

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

报告可使用的两个 baseline：

```text
GaussianNB test accuracy:              0.7100
ExtraTrees-200-balanced test accuracy: 0.7428
```

结果已保存到：

```text
ai/training/results/baselines_final/
```
