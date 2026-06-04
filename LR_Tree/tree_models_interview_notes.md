# 树模型面试总结：随机森林与梯度提升

这份笔记用于回答面试里常见的问题：

- 随机森林是什么？
- GBDT / XGBoost / LightGBM 是什么？
- 它们有什么区别？
- 给定一个数据和业务场景，应该怎么选模型？

## 1. 树模型总览

树模型的基础是决策树。

决策树可以理解成一组 `if-else` 规则：

```text
如果 V14 <= -4.2：
    如果 V10 <= -2.5：
        判为欺诈
    否则：
        判为正常
否则：
    继续判断其他条件
```

单棵决策树优点是可解释，缺点是容易过拟合、稳定性较差。

所以实际业务里更常用集成树模型：

- 随机森林 Random Forest
- 梯度提升树 GBDT
- XGBoost
- LightGBM
- CatBoost

## 2. 随机森林 Random Forest

### 2.1 核心思想

随机森林是很多棵决策树的集成。

```text
随机森林 = 多棵决策树 + 随机样本采样 + 随机特征采样 + 平均/投票
```

分类任务中：

```text
每棵树给一个预测概率
最终概率 = 所有树预测概率的平均值
```

### 2.2 随机性来自哪里

第一，样本随机。

每棵树训练时，会从训练集中有放回抽样，这叫 bootstrap。

```text
原始训练集有 N 条样本
每棵树也抽 N 次
有些样本会重复出现
有些样本不会被抽中
```

第二，特征随机。

每个节点分裂时，不一定看全部特征，而是随机选一部分特征。

分类任务中常见做法：

```text
每次分裂只看 sqrt(特征数) 个特征
```

### 2.3 为什么随机森林比单棵树稳定

单棵树容易受数据扰动影响。

随机森林通过多棵树平均，降低单棵树的波动。

```text
单棵树：方差高，容易过拟合
随机森林：多棵树平均，降低方差
```

### 2.4 常见参数

```text
n_estimators：树的数量
max_depth：每棵树最大深度
min_samples_leaf：叶子节点最少样本数
max_features：每次分裂考虑多少特征
class_weight：类别不平衡权重
```

### 2.5 优缺点

优点：

- 稳定，默认效果通常不错
- 不需要特征标准化
- 对异常值不太敏感
- 能处理非线性关系
- 可用 `feature_importances_` 看特征重要性

缺点：

- 模型较大，预测速度可能慢
- 可解释性不如单棵树
- 对极高维稀疏特征不一定最优
- 很难像 boosting 那样持续降低偏差

## 3. 梯度提升树 GBDT

### 3.1 核心思想

GBDT 也是很多棵树，但训练方式和随机森林不同。

随机森林是并行思想：

```text
多棵树独立训练，最后平均
```

GBDT 是串行思想：

```text
一棵树一棵树地训练
后一棵树修正前面模型的错误
```

整体预测：

```text
最终预测 = 第1棵树 + 第2棵树 + ... + 第K棵树
```

### 3.2 每棵树在学什么

以二分类为例，当前模型会输出一个概率 `p`。

如果真实标签是欺诈：

```text
y = 1
p = 0.1
```

说明模型低估了欺诈风险，下一棵树要把预测往上修正。

如果真实标签是正常：

```text
y = 0
p = 0.9
```

说明模型误报了，下一棵树要把预测往下修正。

所以 GBDT 每一轮都在学习当前模型的残差或梯度方向。

### 3.3 GBDT 与随机森林的本质区别

```text
随机森林：降低方差
GBDT：降低偏差
```

随机森林中树与树相对独立。

GBDT 中后一棵树依赖前面所有树的结果。

## 4. XGBoost

### 4.1 XGBoost 是什么

XGBoost 是一种高性能梯度提升树实现。

它在 GBDT 基础上增强了：

- 一阶梯度
- 二阶梯度
- 正则化
- 列采样
- 行采样
- 缺失值处理
- 高效分裂算法

### 4.2 XGBoost 怎么优化

对于二分类 logloss：

```text
g = p - y
h = p * (1 - p)
```

其中：

- `g` 是一阶梯度，表示预测错误方向
- `h` 是二阶梯度，表示修正时的曲率信息

XGBoost 每次分裂节点时，不是用 Gini，而是计算这个切分能让目标函数下降多少。

这个下降量通常叫 gain。

直观理解：

```text
哪个切分能让损失下降最多，就选择哪个切分
```

### 4.3 XGBoost 常见参数

```text
n_estimators：树的数量
learning_rate：学习率
max_depth：树最大深度
min_child_weight：叶子节点最小二阶梯度和
subsample：样本采样比例
colsample_bytree：特征采样比例
reg_lambda：L2 正则
reg_alpha：L1 正则
scale_pos_weight：正负样本不平衡权重
```

风控二分类中，`scale_pos_weight` 很常用：

```text
scale_pos_weight = 负样本数 / 正样本数
```

## 5. LightGBM

### 5.1 LightGBM 是什么

LightGBM 也是梯度提升树框架，和 XGBoost 同属 GBDT 家族。

它的特点是训练速度快、内存占用低，适合较大规模表格数据。

### 5.2 LightGBM 的核心特点

第一，Histogram 算法。

它会把连续特征离散成桶：

```text
连续值 -> 若干个 bin
```

这样找切分点更快。

第二，leaf-wise 生长。

传统树常见是 level-wise：

```text
一层一层生长
```

LightGBM 默认是 leaf-wise：

```text
每次优先分裂收益最大的叶子
```

leaf-wise 拟合能力强，但也更容易过拟合，所以常配合：

```text
num_leaves
max_depth
min_child_samples
```

### 5.3 LightGBM 常见参数

```text
n_estimators：树的数量
learning_rate：学习率
num_leaves：叶子数量上限
max_depth：最大深度
min_child_samples：叶子节点最少样本数
subsample：样本采样比例
colsample_bytree：特征采样比例
reg_lambda：L2 正则
class_weight：类别权重
```

## 6. 随机森林、XGBoost、LightGBM 的区别

| 方法 | 训练方式 | 树之间关系 | 分裂依据 | 主要优势 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| 随机森林 | 并行 | 树相互独立 | Gini / Entropy | 稳定，调参简单 | 模型大，提升空间有限 |
| GBDT | 串行 | 后树修正前树错误 | 梯度下降损失 | 精度通常更高 | 更容易过拟合 |
| XGBoost | 串行 | 后树修正前树错误 | 一阶/二阶梯度 gain | 强正则，效果稳 | 参数较多 |
| LightGBM | 串行 | 后树修正前树错误 | 梯度 gain + histogram | 快，适合大数据 | leaf-wise 容易过拟合 |

## 7. 面试中怎么选择模型

### 7.1 数据量不大，想要稳定 baseline

优先选择：

```text
随机森林
```

理由：

- 调参相对简单
- 不需要标准化
- 默认效果通常不错
- 不容易因为轻微调参失误而崩

回答方式：

```text
如果我需要快速建立一个稳定 baseline，我会先用随机森林。
它通过 bootstrap 和特征随机采样降低方差，比单棵树稳定。
```

### 7.2 表格数据，追求更高精度

优先选择：

```text
XGBoost 或 LightGBM
```

理由：

- 对表格数据效果强
- 能学习非线性和特征交互
- 可以通过 boosting 持续修正错误

回答方式：

```text
如果是结构化表格数据，并且目标是提升效果，我会考虑 XGBoost 或 LightGBM。
它们通过逐轮拟合梯度来降低损失，通常比随机森林有更强的拟合能力。
```

### 7.3 数据量较大，训练速度重要

优先选择：

```text
LightGBM
```

理由：

- histogram 加速
- 训练快
- 内存效率高

回答方式：

```text
如果数据量较大，我会优先考虑 LightGBM，因为它的 histogram 和 leaf-wise 策略训练效率更高。
但我会控制 num_leaves、max_depth 和 min_child_samples 防止过拟合。
```

### 7.4 类别极度不平衡，比如欺诈检测

优先考虑：

```text
XGBoost / LightGBM / 随机森林都可以
```

重点不是只选模型，而是评估指标和阈值。

应该关注：

```text
PR-AUC
Recall
Precision
F1
混淆矩阵
同一 Recall 下的 Precision
```

不应该只看：

```text
Accuracy
```

因为全预测正常也会有很高准确率。

回答方式：

```text
在欺诈检测这类极度不平衡场景，我会优先看 PR-AUC，而不是 Accuracy。
模型上可以先用随机森林作为 baseline，再尝试 XGBoost 或 LightGBM。
同时需要调 class_weight 或 scale_pos_weight，并根据业务成本选择阈值。
```

### 7.5 业务强解释性要求高

优先选择：

```text
逻辑回归 / 单棵浅决策树 / 可解释性增强的树模型
```

随机森林和 boosting 可以看特征重要性，但整体不如逻辑回归和浅树直观。

回答方式：

```text
如果业务强依赖解释性，我会先考虑逻辑回归或浅层决策树。
如果必须用复杂树模型，我会结合特征重要性、SHAP 等方法解释。
```

## 8. 面试高频对比回答

### 8.1 随机森林和 GBDT 的区别

```text
随机森林是 Bagging 思想，多棵树并行训练，主要降低方差。
GBDT 是 Boosting 思想，多棵树串行训练，后一棵树修正前面模型的错误，主要降低偏差。
```

### 8.2 XGBoost 和 GBDT 的区别

```text
XGBoost 是增强版 GBDT。
它使用一阶和二阶梯度，加入正则化，并支持行采样、列采样和高效分裂算法。
所以它通常更稳定，泛化能力更好。
```

### 8.3 XGBoost 和 LightGBM 的区别

```text
XGBoost 通常更稳，LightGBM 通常更快。
LightGBM 使用 histogram 算法和 leaf-wise 生长，训练效率高，但如果不限制 num_leaves 和 max_depth，更容易过拟合。
```

### 8.4 为什么样本不平衡时 PR-AUC 更重要

```text
因为 PR-AUC 关注 Precision 和 Recall，直接反映报警质量和欺诈召回。
ROC-AUC 在负样本极多时可能显得过于乐观，Accuracy 更不可靠。
```

## 9. 一句话选择策略

```text
先用逻辑回归或随机森林做 baseline；
表格数据追求效果时试 XGBoost / LightGBM；
数据量大优先 LightGBM；
不平衡场景重点看 PR-AUC 和阈值；
解释性要求高时优先简单模型或配合 SHAP。
```
