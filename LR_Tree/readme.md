## 风控算法学习

这个目录用于基于信用卡欺诈识别数据集学习常见风控模型。

### 数据集介绍

本项目使用的是经典的信用卡欺诈检测数据集 Credit Card Fraud Detection。该数据集常用于风控、反欺诈、异常检测和极度不平衡二分类问题的学习。

数据文件位于：

```text
fengkong/data/raw/creditcard.csv
```

数据集基本情况：

- 样本量：284,807 笔信用卡交易
- 正常交易：284,315 笔，`Class=0`
- 欺诈交易：492 笔，`Class=1`
- 欺诈样本占比约 0.1727%，属于非常典型的类别不平衡问题
- 目标变量：`Class`
- 特征数量：30 个

字段说明：

- `Time`：当前交易距离数据集中第一笔交易的时间间隔，单位通常理解为秒
- `Amount`：交易金额
- `V1` 到 `V28`：经过 PCA 降维和脱敏处理后的匿名特征，原始业务含义不可见
- `Class`：标签列，`0` 表示正常交易，`1` 表示欺诈交易

建模注意点：

- 不能只看 Accuracy。因为欺诈样本极少，即使模型全部预测为正常，准确率也会非常高。
- 更应该关注 Recall、Precision、F1、ROC-AUC、PR-AUC 和混淆矩阵。
- 在风控场景中，阈值选择很重要。降低阈值通常可以提高欺诈召回率，但也会带来更多误报。
- 逻辑回归、SVM 等模型通常需要特征标准化；树模型和集成树模型一般不强依赖标准化。

### 目录结构

```text
fengkong/
├── data/
│   └── raw/                         # 原始数据
│       ├── creditcard.csv.zip
│       └── creditcard.csv
├── experiments/                     # 每个模型一个独立实验脚本
│   ├── 01_logistic_regression.py
│   ├── 01_logistic_regression_from_scratch.py
│   ├── 02_decision_tree.py
│   ├── 03_random_forest.py
│   ├── 04_lightgbm.py
│   └── 05_xgboost.py
├── outputs/                         # 每个模型单独保存图像和结果
│   ├── logistic_regression/
│   ├── logistic_regression_from_scratch/
│   ├── decision_tree/
│   ├── random_forest/
│   ├── lightgbm/
│   └── xgboost/
├── src/                             # 可复用的数据加载、指标评估工具
│   ├── config.py
│   ├── data_utils.py
│   └── metrics_utils.py
└── 1.LR.py                          # 兼容旧入口，转发到逻辑回归实验
```

### 当前已实现

- 逻辑回归 Logistic Regression
- 手写逻辑回归，不调用 sklearn 的 LogisticRegression
- 决策树 Decision Tree
- 随机森林 Random Forest
- LightGBM 脚本
- XGBoost 脚本
- 训练集 / 测试集指标评估
- Accuracy、Precision、Recall、F1、ROC-AUC、PR-AUC
- 混淆矩阵和 classification report
- 不同阈值下的 precision / recall / f1 对比
- ROC 曲线和 PR 曲线保存

### 运行逻辑回归

```bash
/mnt/lfd/anaconda3/envs/UFO/bin/python fengkong/experiments/01_logistic_regression.py
```

也可以继续使用旧入口：

```bash
/mnt/lfd/anaconda3/envs/UFO/bin/python fengkong/1.LR.py
```

运行手写逻辑回归：

```bash
/mnt/lfd/anaconda3/envs/UFO/bin/python fengkong/experiments/01_logistic_regression_from_scratch.py
```

### 运行树模型

当前也可以使用 `droplet_seg` 环境运行实验：

```bash
/mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/02_decision_tree.py
```

```bash
/mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/03_random_forest.py
```

```bash
/mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/04_lightgbm.py
```

```bash
/mnt/lfd/anaconda3/envs/droplet_seg/bin/python fengkong/experiments/05_xgboost.py
```

如果 `droplet_seg` 环境没有安装 LightGBM 或 XGBoost，需要先安装：

```bash
/mnt/lfd/anaconda3/envs/droplet_seg/bin/python -m pip install lightgbm
```

```bash
/mnt/lfd/anaconda3/envs/droplet_seg/bin/python -m pip install xgboost
```

### 实验结果基线

下面结果来自手写逻辑回归：

```bash
/mnt/lfd/anaconda3/envs/UFO/bin/python fengkong/experiments/01_logistic_regression_from_scratch.py
```

默认参数：

- `learning_rate=0.1`
- `epochs=500`
- `l2=0.001`
- 默认分类阈值 `threshold=0.5`

训练损失变化：

```text
epoch=   1, loss=0.700901
epoch=  50, loss=0.255847
epoch= 100, loss=0.203692
epoch= 150, loss=0.180210
epoch= 200, loss=0.167376
epoch= 250, loss=0.159434
epoch= 300, loss=0.154089
epoch= 350, loss=0.150270
epoch= 400, loss=0.147418
epoch= 450, loss=0.145214
epoch= 500, loss=0.143465
```

测试集指标：

```text
Accuracy:  0.9767
Precision: 0.0640
Recall:    0.9184
F1:        0.1196
ROC-AUC:   0.9766
PR-AUC:    0.7054
```

测试集混淆矩阵：

```text
行=真实类别，列=预测类别

[[55547  1317]
 [    8    90]]
```

含义：

- `55547`：正常交易，被正确预测为正常
- `1317`：正常交易，被误判为欺诈，属于误报
- `8`：欺诈交易，被预测为正常，属于漏报
- `90`：欺诈交易，被成功识别

在默认阈值 `0.5` 下，模型抓住了 `90 / 98` 笔欺诈交易，召回率较高；但同时误报了 `1317` 笔正常交易，精确率较低。

阈值对比：

```text
threshold | precision | recall | f1
     0.90 |    0.3671 | 0.8878 | 0.5194
     0.70 |    0.1252 | 0.8878 | 0.2194
     0.50 |    0.0640 | 0.9184 | 0.1196
     0.30 |    0.0238 | 0.9184 | 0.0464
     0.10 |    0.0048 | 0.9694 | 0.0096
```

结果分析：

- 训练损失持续下降，说明手写梯度下降正常工作。
- 测试集 `ROC-AUC=0.9766`，说明模型整体排序能力较好。
- 测试集 `PR-AUC=0.7054`，在极度不平衡数据上比 Accuracy 更有参考价值。
- 默认阈值 `0.5` 下 Recall 高，但 Precision 很低，说明模型偏向多报警、少漏报。
- 当前结果里 `threshold=0.90` 的 F1 最高，Precision 明显提升，同时 Recall 仍保持在 `0.8878`，可以作为后续模型对比时的一个业务阈值参考。
- 训练集和测试集表现差距不大，暂时没有明显过拟合迹象。

### 后续计划

- 决策树 Decision Tree
- 随机森林 Random Forest
- LightGBM
- XGBoost
- CatBoost
- 支持向量机 SVM
- 神经网络
