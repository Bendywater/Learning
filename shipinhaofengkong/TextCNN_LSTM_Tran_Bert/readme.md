# 用Transformer架构完成一个风控业务场景的检测
比如现在有一些文本数据，这个文本包含了标题，简介，内容以及评论，或者有一些是有举报理由的，那现在你怎么用这批数据去检测出哪些是有风险的数据呢？

## 1. 项目背景

在内容平台里，很多风险并不是只靠一个关键词就能判断出来的。比如标题看起来正常，但正文里有诱导交易；正文正常，但评论区出现引流、辱骂、色情暗示；或者用户举报理由里已经包含了“诈骗”“低俗”“虚假宣传”等强信号。

这个项目的目标是：基于纯文本信息，使用 Transformer 架构训练一个文本风控检测模型，判断一条内容是否存在风险，并进一步识别风险类型和风险等级。

可以把它理解成一个“文本内容安全审核模型”。

输入数据可能包括：

- 标题：视频标题、帖子标题、商品标题
- 简介：内容摘要、发布说明
- 正文：长文本、视频 ASR 转写、OCR 文本
- 评论：高赞评论、最新评论、被举报评论
- 举报理由：用户举报时填写的原因

输出结果包括：

- 是否风险：`risk / normal`
- 风险类型：色情低俗、辱骂攻击、诈骗引流、虚假宣传、违法违规、涉政敏感、暴恐危险等
- 风险等级：低危、中危、高危
- 模型分数：用于线上阈值控制和人工审核排序

## 2. 业务问题定义

这个任务不能简单地理解成“文本二分类”。真实风控里通常需要同时解决几个问题：

1. 是否有风险
2. 是哪一类风险
3. 风险有多严重
4. 模型是否足够可解释，能不能给审核员提供辅助证据
5. 线上是追求高召回，还是追求高精度

所以本项目采用多任务建模方式：

```text
输入文本
  -> Transformer Encoder
  -> 风险二分类 head
  -> 风险多分类 head
  -> 风险等级 head
  -> 风险分数
```

其中：

- 二分类 head 判断是否需要进入风控链路
- 多分类 head 判断风险类型
- 等级 head 判断处置强度

这样比只做一个 `0/1` 分类更接近真实业务。

## 3. 数据设计

### 3.1 原始数据格式

假设每条样本长这样：

```json
{
  "content_id": "v_10001",
  "title": "兼职副业日入几百，点我了解",
  "summary": "分享一个轻松赚钱的方法",
  "content": "不需要经验，添加联系方式即可领取教程...",
  "comments": [
    "真的能赚钱吗",
    "加了之后让我先交保证金",
    "这个像诈骗"
  ],
  "report_reason": "疑似诈骗引流",
  "label_risk": 1,
  "label_type": "fraud",
  "label_level": "high"
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `title` | 标题，通常短但强表达 |
| `summary` | 简介，补充上下文 |
| `content` | 正文、OCR、ASR 等主体文本 |
| `comments` | 评论区文本，可能包含用户反馈 |
| `report_reason` | 举报理由，是强监督信号 |
| `label_risk` | 是否风险 |
| `label_type` | 风险类型 |
| `label_level` | 风险等级 |

### 3.2 风险类型设计

可以先设计 7 个一级风险类别：

| 类别 | 说明 | 示例 |
|---|---|---|
| `normal` | 正常内容 | 普通分享、知识科普 |
| `porn` | 色情低俗 | 暗示性内容、低俗擦边 |
| `abuse` | 辱骂攻击 | 人身攻击、歧视、网暴 |
| `fraud` | 诈骗引流 | 兼职骗局、刷单、投资诈骗 |
| `ads` | 广告导流 | 加微信、私域引流、站外交易 |
| `illegal` | 违法违规 | 赌博、违禁品、危险行为 |
| `misinfo` | 虚假宣传 | 夸大疗效、虚假金融收益 |

真实业务里可以继续细分二级类，比如：

- `fraud_job`：兼职刷单诈骗
- `fraud_investment`：投资理财诈骗
- `ads_contact`：联系方式导流
- `porn_soft`：软色情擦边
- `abuse_personal_attack`：人身攻击

### 3.3 文本拼接方式

Transformer 输入通常是一段 token 序列。对于多字段文本，可以使用字段标记拼接：

```text
[TITLE] 兼职副业日入几百，点我了解
[SUMMARY] 分享一个轻松赚钱的方法
[CONTENT] 不需要经验，添加联系方式即可领取教程...
[COMMENTS] 真的能赚钱吗 [SEP] 加了之后让我先交保证金 [SEP] 这个像诈骗
[REPORT] 疑似诈骗引流
```

这样做的好处是模型可以区分不同字段的语义来源。

比如同样出现“诈骗”：

- 正文里说“教你如何诈骗别人”是高风险
- 评论里说“这个像诈骗”是用户反馈信号
- 举报理由里说“疑似诈骗”是强监督信号

字段标记能帮助模型理解这些差异。

## 4. 模型方案

### 4.1 基线模型

项目可以先做几个基线：

| 模型 | 作用 |
|---|---|
| TF-IDF + LR | 最简单的文本分类基线 |
| TextCNN | 捕捉局部关键词和短语 |
| BiLSTM | 捕捉一定顺序关系 |
| BERT / RoBERTa | Transformer 语义建模 |

最终主模型使用中文预训练 Transformer，例如：

- `hfl/chinese-roberta-wwm-ext`
- `bert-base-chinese`
- `macbert-base-chinese`
- 如果业务文本更长，可以用 Longformer / BigBird 类结构

### 4.2 Transformer 架构

核心结构：

```text
多字段文本
   |
Tokenizer
   |
Input IDs + Attention Mask + Segment Embedding
   |
Pretrained Transformer Encoder
   |
[CLS] 向量 / Pooling 向量
   |
多任务分类头
   |-------------------- 是否风险
   |-------------------- 风险类型
   |-------------------- 风险等级
```

多任务 loss：

```text
Loss = a * Loss_risk + b * Loss_type + c * Loss_level
```

其中：

- `Loss_risk`：二分类交叉熵
- `Loss_type`：多分类交叉熵
- `Loss_level`：等级分类交叉熵
- `a/b/c`：不同任务的权重

例如初期可以设置：

```text
Loss = 1.0 * 二分类Loss + 1.0 * 类型Loss + 0.5 * 等级Loss
```

### 4.3 为什么用 Transformer

相比 TextCNN / LSTM，Transformer 更适合这个任务：

1. 能建模长距离语义关系  
   比如标题说“赚钱教程”，正文后面才出现“先交保证金”。

2. 能理解上下文  
   “加我”单独看不一定违规，但和“稳赚”“高收益”“私聊”一起出现时风险升高。

3. 对变体表达更鲁棒  
   风控文本里经常有谐音、缩写、错别字、分隔符规避，比如“薇”“v我”“+微”“看主页”。

4. 可以利用预训练知识  
   中文预训练模型已经学到大量语义关系，少量标注数据也能微调出不错效果。

## 5. 训练流程

### 5.1 数据处理

主要步骤：

1. 清洗 HTML、特殊符号、重复空格
2. 保留有风险含义的符号，比如 `+V`、微信号、链接、金额
3. 评论过多时选择 Top-K 条评论
4. 文本过长时按字段优先级截断
5. 按内容 ID 划分训练集、验证集、测试集，避免同一内容泄漏

字段优先级可以这样设：

```text
举报理由 > 标题 > 简介 > 正文前半段 > 高风险评论 > 普通评论
```

因为举报理由和标题通常信息密度更高。

### 5.2 样本构造

风控数据常见问题是类别不平衡：

- 正常样本很多
- 风险样本少
- 高危风险更少
- 新型风险样本更少

处理方式：

- 对少数类过采样
- 对正常样本欠采样
- 使用 class weight
- 使用 focal loss
- 做 hard negative mining

Hard negative 示例：

```text
“反诈骗宣传：不要相信刷单日赚几百”
```

这句话包含“刷单”“日赚几百”，但它是正常的反诈内容。模型如果只学关键词就会误杀，所以这类样本很重要。

### 5.3 训练伪代码

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class RiskTextTransformer(nn.Module):
    def __init__(self, model_name, num_types, num_levels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.risk_head = nn.Linear(hidden_size, 2)
        self.type_head = nn.Linear(hidden_size, num_types)
        self.level_head = nn.Linear(hidden_size, num_levels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]

        risk_logits = self.risk_head(cls)
        type_logits = self.type_head(cls)
        level_logits = self.level_head(cls)

        return risk_logits, type_logits, level_logits
```

输入拼接函数：

```python
def build_text(sample):
    comments = " [SEP] ".join(sample.get("comments", [])[:5])
    return (
        f"[TITLE] {sample.get('title', '')} "
        f"[SUMMARY] {sample.get('summary', '')} "
        f"[CONTENT] {sample.get('content', '')} "
        f"[COMMENTS] {comments} "
        f"[REPORT] {sample.get('report_reason', '')}"
    )
```

训练时：

```python
loss_risk = ce_loss(risk_logits, label_risk)
loss_type = ce_loss(type_logits, label_type)
loss_level = ce_loss(level_logits, label_level)

loss = loss_risk + loss_type + 0.5 * loss_level
loss.backward()
optimizer.step()
```

## 6. 评估指标

风控场景不能只看 Accuracy，因为正常样本太多，模型全预测正常也可能准确率很高。

重点看：

| 指标 | 作用 |
|---|---|
| Precision | 命中的风险里有多少是真的风险 |
| Recall | 真实风险里有多少被模型抓住 |
| F1 | Precision 和 Recall 的平衡 |
| PR-AUC | 类别不平衡时比 ROC-AUC 更有参考价值 |
| 高危 Recall | 高危风险漏判率必须低 |
| 分类型 Recall | 每个风险类别是否都能召回 |
| 人审通过率 | 送审内容里真正违规的比例 |

不同链路的指标重点不同：

- 初筛模型：更关注 Recall，宁可多送审，不要漏掉风险
- 复筛模型：更关注 Precision，减少误杀
- 高危模型：关注高危 Recall 和响应速度

## 7. 阈值策略

模型输出风险概率后，不应该只用一个固定阈值，而是做分层处置。

示例：

| 风险分数 | 处置 |
|---|---|
| `score >= 0.95` | 高危，直接拦截或强制人审 |
| `0.80 <= score < 0.95` | 中高危，进入人工审核队列 |
| `0.50 <= score < 0.80` | 低危，结合规则和账号特征判断 |
| `score < 0.50` | 放行，但保留日志 |

如果模型判断为 `fraud`，且文本里存在联系方式、金额、转账、保证金等规则命中，可以提高风险等级。

```text
最终分数 = 模型分数 + 规则加权 + 账号风险加权
```

这样比纯模型更稳。

## 8. 线上系统设计

线上链路可以设计为：

```text
用户发布内容
   |
文本抽取
   |-- 标题
   |-- 简介
   |-- 正文
   |-- OCR/ASR
   |-- 评论
   |-- 举报理由
   |
规则初筛
   |
Transformer 风控模型
   |
风险分数融合
   |
处置策略
   |-- 放行
   |-- 限流
   |-- 人审
   |-- 拦截
   |-- 封禁/处罚
   |
审核结果回流
   |
训练数据更新
```

线上需要关注：

- 推理延迟
- 模型版本管理
- 阈值动态调整
- 风险分布监控
- 漏判 case 回流
- 误杀 case 回流
- 新型黑产样本的快速迭代

## 9. 可解释性设计

审核员不只需要一个分数，还需要知道为什么被判风险。

可以做几种解释：

1. 高风险关键词高亮  
   比如“加微”“稳赚”“保证金”“私聊”“裸聊”。

2. 字段级贡献  
   告诉审核员风险主要来自标题、正文、评论还是举报理由。

3. Top-K 相似违规样本  
   检索历史相似违规内容，辅助人工判断。

4. Attention 或梯度归因  
   用于离线分析模型关注了哪些 token。

示例输出：

```json
{
  "content_id": "v_10001",
  "risk_score": 0.972,
  "risk_type": "fraud",
  "risk_level": "high",
  "evidence": [
    {"field": "title", "text": "兼职副业日入几百"},
    {"field": "content", "text": "添加联系方式"},
    {"field": "comments", "text": "让我先交保证金"}
  ],
  "action": "manual_review"
}
```

## 10. 项目亮点

这个项目可以重点讲这些亮点：

1. 多字段建模  
   不只是把标题拿去分类，而是融合标题、简介、正文、评论、举报理由。

2. 多任务学习  
   同时预测是否风险、风险类型和风险等级，更符合真实业务。

3. 风控阈值分层  
   根据风险分数做放行、人审、拦截，不是简单分类。

4. 类别不平衡处理  
   使用 class weight、过采样、hard negative mining 解决长尾风险。

5. 规则 + 模型融合  
   模型负责语义泛化，规则负责强约束和可解释兜底。

6. 数据回流闭环  
   人审结果、举报结果、误杀漏判样本可以持续回流训练。

## 11. 面试回答模板

如果面试官问：

> 给你标题、简介、正文、评论、举报理由这些文本，你怎么检测风险？

可以这样回答：

我会先把这个问题定义成一个多字段文本风控任务，而不是简单二分类。每条内容会包含标题、简介、正文、评论和举报理由，我会用特殊字段标记把它们拼接起来，比如 `[TITLE]`、`[CONTENT]`、`[COMMENTS]`、`[REPORT]`，再送入中文预训练 Transformer。

模型上我会用 BERT/RoBERTa 这类 Encoder 结构，取 `[CLS]` 向量做多任务分类：一个 head 判断是否风险，一个 head 判断风险类型，比如诈骗、低俗、辱骂、广告导流等，另一个 head 判断风险等级。训练时把三个 loss 加权求和。

数据上我会特别关注类别不平衡和 hard negative。比如“刷单日赚几百”可能是诈骗，但“反诈骗宣传：不要相信刷单日赚几百”是正常内容，如果没有 hard negative，模型容易只记关键词，线上误杀会很严重。

评估上不会只看 accuracy，而是重点看风险召回率、Precision、F1、PR-AUC，以及高危风险的召回。线上我会根据模型分数做分层处置，高分直接拦截或人审，中分结合规则和账号风险，低分放行并留日志。

最后我会把人工审核、用户举报、误杀漏判 case 回流到训练集，持续迭代模型。这样整个系统不是一个孤立分类器，而是一个规则、模型、人工审核和数据回流结合的风控闭环。

## 12. 可以落地的目录结构

```text
TextCNN_LSTM_Tran_Bert/
  README.md
  data/
    train.jsonl
    valid.jsonl
    test.jsonl
  src/
    preprocess.py
    dataset.py
    model.py
    train.py
    evaluate.py
    inference.py
  configs/
    risk_text_bert.yaml
  outputs/
    checkpoints/
    reports/
```

核心模块说明：

| 文件 | 作用 |
|---|---|
| `preprocess.py` | 清洗文本、拼接多字段、构造标签 |
| `dataset.py` | 封装 PyTorch Dataset |
| `model.py` | Transformer 多任务模型 |
| `train.py` | 训练入口 |
| `evaluate.py` | 评估 Precision / Recall / F1 / PR-AUC |
| `inference.py` | 单条内容风险预测 |
| `risk_text_bert.yaml` | 模型、数据、训练参数配置 |

## 13. 后续优化方向

如果基础模型已经跑通，可以继续优化：

- 使用领域继续预训练：用平台历史文本做 MLM 继续预训练
- 引入对抗样本：处理谐音、错别字、拆字、符号插入
- 使用长文本模型：解决正文和评论过长的问题
- 加入账号特征：账号历史违规率、发布频率、互动异常
- 加入图谱特征：同设备、同联系方式、同团伙内容
- 使用蒸馏模型：把大模型蒸馏成小模型，降低线上延迟
- 做主动学习：优先标注模型最不确定的样本

最终可以从“单条文本检测”升级成“内容 + 账号 + 行为 + 关系”的综合风控系统。
