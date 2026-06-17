# BP 训练与 baseline 复现说明

本目录只放可复现训练脚本和小型说明文件。最终前后端对接模型在：

```text
ai/model/dinov2_bp_classifier.npz
```

## 重要说明

手写 BP 重新训练需要 DINOv2 特征 CSV，当前已经保留在仓库：

```text
ai/training/data/pest_dinov2_features_fast.csv
```

这份 CSV 来自 `outputs.rar` 里的原始输出：

```text
outputs/pest_dinov2_features_fast.csv
```

配套标签和统计文件：

```text
ai/training/data/label_mapping.json
ai/training/data/feature_stats.json
```

## 运行 baseline

```powershell
.\.venv-ml\Scripts\python.exe .\ai\training\run_experiments.py `
  --csv .\ai\training\data\pest_dinov2_features_fast.csv `
  --output-dir .\ai\training\outputs\dinov2_baselines `
  --hidden-dims 128,64 `
  --epochs 80 `
  --batch-size 256 `
  --learning-rate 0.001 `
  --optimizer adam `
  --l2-penalty 0.001 `
  --extra-trees 200
```

其中 `BP` 使用仓库手写 `ai.neural_network.MLPClassifier`；`Dummy`、`GaussianNB`、`KNN`、`DecisionTree`、`ExtraTrees` 是 sklearn baseline。

## 重新训练最终 BP artifact

```powershell
.\.venv-ml\Scripts\python.exe .\ai\training\train_bp_artifact.py `
  --csv .\ai\training\data\pest_dinov2_features_fast.csv `
  --output .\ai\model\dinov2_bp_classifier.npz `
  --hidden-dims 128,64 `
  --epochs 80 `
  --batch-size 256 `
  --learning-rate 0.001 `
  --optimizer adam `
  --l2-penalty 0.001
```

训练完成会同时生成：

```text
ai/model/dinov2_bp_classifier.metrics.json
```
