# DINOv2 + 手写 BP 模型说明

本目录是当前前后端对接使用的最终模型目录。

## 文件

- `dinov2_bp_classifier.npz`：最终分类模型 artifact。输入是 384 维 DINOv2 特征，分类器是仓库手写 `ai.neural_network.MLPClassifier`。
- `dinov2_bp_classifier.metrics.json`：训练后在 train/val/test 上的指标。
- `label_mapping.json`：类别编号到昆虫类别名的映射。
- `feature_stats.json`：生成特征 CSV 时的数据量、类别分布和 DINOv2 配置。

## 当前指标

- train accuracy: `1.0000`
- validation accuracy: `0.7668`
- test accuracy: `0.7583`
- test macro F1: `0.7540`

## 后端配置

后端默认读取：

```powershell
MODEL_ARTIFACT_PATH=../ai/model/dinov2_bp_classifier.npz
MODEL_FEATURE_EXTRACTOR=dinov2
DINOV2_MODEL=facebook/dinov2-small
DINOV2_DEVICE=auto
USE_MOCK_MODEL=false
```

启动后端：

```powershell
cd D:\insect-recognition-ai
.\.venv-ml\Scripts\python.exe .\backend\app.py
```

如果新机器没有依赖，先安装：

```powershell
.\.venv-ml\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

## 标签映射

```text
0 ant
1 bed-bug
2 bee
3 beetle
4 bernsteinschabe
5 cockroach
6 fly
7 fruitfly
8 grasshopper
9 hornet
10 housefly
11 ladybug
12 mosquito
13 moth
14 silverfish
15 slug
16 snail
17 spider
18 tiger_mosquito
19 wasp
```
