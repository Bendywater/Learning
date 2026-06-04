# Learning

这是一个个人机器学习、深度学习和大模型应用方向的学习仓库，用来整理不同主题的代码实验、课程笔记、项目实践和模型验证方法。

仓库内容覆盖传统机器学习、树模型、无监督学习、计算机视觉、RAG、LangChain/LangGraph、多模态检索、风控场景建模等方向。各子目录相对独立，适合按主题逐步学习和复盘。

## 目录结构

```text
.
├── Model_test/                         # 模型验证、交叉验证与调参方法
├── LR_Tree/                            # 逻辑回归、决策树、随机森林、LightGBM、XGBoost
├── Unsupervised_Learning/              # PCA、聚类、降维等无监督学习实验
├── Electric-Load-Forecasting/          # 电力负荷预测项目
├── RAG/                                # RAG 与多模态 RAG 学习实践
├── Langchain1.0-Langgraph1.0-Learning/ # LangChain 1.0 与 LangGraph 1.0 系统学习
├── CV/                                 # 深度学习、PyTorch、CV、NLP 等课程笔记
├── shipinhaofengkong/                  # 视频/图文风控相关实验
├── assist-agent/                       # Agent 与 LLM 应用实践
└── Claude-Code-Source-Study/           # Claude Code 源码学习笔记
```

## 主要内容

### Model_test

整理模型验证和调参的基础方法，包括：

- Hold-out 留出法
- K-Fold 交叉验证
- Stratified K-Fold 分层交叉验证
- Grid Search 网格搜索
- Random Search 随机搜索
- 贝叶斯优化思路
- 过拟合与数据泄露规避

相关文件：

- `Model_test/readme.md`
- `Model_test/kfold_gridsearchCV.py`

### LR_Tree

围绕信用卡欺诈检测数据集，实践常见分类模型：

- Logistic Regression
- 手写 Logistic Regression
- Decision Tree
- Random Forest
- LightGBM
- XGBoost

同时包含 ROC、PR 曲线等评估结果输出。

### Unsupervised_Learning

包含无监督学习相关实验，重点包括：

- PCA 降维
- KMeans 聚类
- 聚类可视化
- 零售数据分析实验

### Electric-Load-Forecasting

电力负荷预测项目，包含数据处理、模型训练、评估脚本和配置文件，适合作为时间序列预测项目模板参考。

### RAG

整理检索增强生成相关内容，包括基础 RAG、多模态 RAG、向量检索、PDF 文档处理等实践。

### LangChain / LangGraph

系统学习 LangChain 1.0 与 LangGraph 1.0，内容按阶段组织：

- 基础概念
- Prompt、Message、Tool
- Memory 与 Checkpoint
- RAG 基础与进阶
- Agent 和多 Agent
- 多模态输入
- 错误处理与 LangSmith
- 项目实践

### CV

包含大量 Jupyter Notebook 学习笔记，覆盖：

- PyTorch 基础
- 卷积神经网络
- 经典 CNN 架构
- 目标检测
- 语义分割
- RNN、LSTM、Transformer
- BERT
- 深度学习课程作业与测验
- RAG 与大模型 API 入门

## 数据与大文件说明

本仓库中的大文件数据集、压缩包、虚拟环境和 Python 缓存文件不建议上传到 GitHub，已通过 `.gitignore` 忽略。

例如：

- `LR_Tree/data/raw/creditcard.csv`
- `LR_Tree/data/raw/creditcard.csv.zip`
- `Electric-Load-Forecasting/data/raw/electricityloaddiagrams20112014.zip`
- `venv/`
- `.venv/`
- `__pycache__/`

如需复现实验，请根据各子目录 README 或代码说明自行下载对应数据集，并放到约定的数据目录下。

## 使用方式

建议按主题进入对应目录查看说明：

```bash
cd Model_test
```

或运行具体 Python 脚本：

```bash
python kfold_gridsearchCV.py
```

不同子项目依赖不完全相同，建议为每个项目单独创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果某个子项目没有 `requirements.txt`，可以根据报错逐步安装所需依赖。

## 仓库目标

这个仓库主要用于：

- 沉淀机器学习和深度学习学习笔记
- 复现经典算法和项目流程
- 整理模型验证、调参和评估方法
- 积累 RAG、Agent、多模态等大模型应用实践
- 为后续项目开发和面试复习提供可回看的材料

