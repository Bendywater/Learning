# CLIP 
CLIP 一个把图像和文本映射到用一个语义空间的多模态模型
它和普通图像分类模型最大的区别是：

  - 普通图像分类模型：输出固定类别
  - CLIP：输出“图像和一段文本有多匹配”

所以在风控里，CLIP 很适合做：

- 零样本/少样本风险识别
- 图文语义匹配
- 抽帧后的语义打分
- 相似违规内容召回



一个视频风控系统常见是这样：

  视频
  -> 抽帧
  -> 每帧跑视觉模型
  -> 多帧结果聚合
  -> 视频级风险判断

  这里“每帧跑什么模型”可以有很多选择：

  - 普通图像分类模型
  - 目标检测模型
  - OCR
  - NSFW/暴恐/广告专用模型
  - CLIP
  - 更强的视频模型

  所以抽帧只是“把视频变成若干张图”，不是等于 CLIP。


3. CLIP 在风控里怎么用

  这才是你面试里要会讲的。


  方法一：零样本分类
  比如你没有专门训练“低俗擦边”分类器，但你可以写一些文本提示词：

  "a pornographic image"
  "a violent image"
  "a screenshot containing QR code"
  "a normal daily-life image"

  然后把抽出的帧和这些文本分别算相似度。

  如果某一帧和 "a screenshot containing QR code" 很接近，说明它可能含导流元素。

  这叫 zero-shot classification。

  那再截帧之后，推理的时候就会给他对应的图片和一堆文本，然后得倒对应的相似度的值，可以判断

  ———

  方法二：做视觉语义特征
  不是直接拿 CLIP 做最终判断，而是把 CLIP 的图像 embedding 当作特征。

  流程像这样：

  抽帧
  -> 每帧过 CLIP image encoder
  -> 得到帧特征
  -> 多帧聚合
  -> 接一个下游分类器

  例如：

  - 平均池化多帧向量
  - max pooling
  - attention pooling
  - 再接 MLP / XGBoost / Transformer

  这个更像工业做法。

  ———

  方法三：图文一致性检查
  视频号场景里很重要。

  比如：

  - 标题说“生活记录”
  - 帧里全是擦边、导流、二维码
  - OCR/ASR 和画面不一致

  这时候可以做：

  - 帧图像 embedding
  - 标题/OCR/ASR 文本 embedding
  - 看图文相似关系是否异常

  这个在“挂羊头卖狗肉”的识别上很有价值。


## 4. 本目录代码说明

我这里做了一个可以跑通的 CLIP 风控 demo。数据是脚本自动生成的模拟图片和文本，覆盖 4 类：

| 类别 | 含义 |
|---|---|
| `normal` | 正常生活、学习、旅行内容 |
| `qr_ad` | 二维码、私域导流、广告引流 |
| `fraud` | 兼职赚钱、保证收益、先交保证金等诈骗风险 |
| `violence` | 危险、暴力、威胁类风险 |

目录结构：

```text
CLIP_related/
  requirements.txt
  readme.md
  data/
    images/                 # 自动生成的模拟图片
    train.jsonl             # 自动生成的训练数据
    test.jsonl              # 自动生成的测试数据
    all.jsonl
  src/
    generate_demo_data.py   # 造数据
    clip_utils.py           # CLIP 加载、向量抽取、相似度计算
    zero_shot_risk.py       # CLIP 零样本风险识别
    text_image_consistency.py # 图文一致性检查
    train_clip_classifier.py # CLIP embedding + LR 分类器
    run_demo.sh             # 一键运行
  outputs/
    zero_shot_predictions.csv
    consistency_scores.csv
    classifier_report.txt
```

## 5. 运行方式

进入目录：

```bash
cd /mnt/lfd/learning/shipinhaofengkong/CLIP_related
```

生成模拟数据不需要安装第三方库，直接运行：

```bash
python3 src/generate_demo_data.py
```

如果要运行 CLIP 推理，再安装完整依赖：

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements.txt
```

如果没有 `uv`，也可以用 pip：

```bash
python3 -m pip install -r requirements.txt
```

运行零样本风险识别：

```bash
.venv/bin/python src/zero_shot_risk.py
```

运行图文一致性检查：

```bash
.venv/bin/python src/text_image_consistency.py
```

训练一个轻量下游分类器：

```bash
.venv/bin/python src/train_clip_classifier.py
```

也可以一键运行：

```bash
bash src/run_demo.sh
```

第一次运行 CLIP 时会从 HuggingFace 下载 `openai/clip-vit-base-patch32`，需要网络。如果机器已经缓存过模型，就会直接使用本地缓存。

## 6. 这个 demo 对应真实业务怎么讲

真实视频风控里，一条视频会先抽帧，比如每 1 秒抽一帧，或者按场景变化抽关键帧。每一帧都可以走这个 demo 里的 CLIP 流程：

```text
视频
  -> 抽帧
  -> CLIP image encoder
  -> 和风险 prompt 计算相似度
  -> 多帧分数聚合
  -> 视频级风险判断
```

零样本方式适合冷启动：

```text
图片 embedding vs 风险文本 prompt embedding
```

如果已经有标注数据，可以用更工业的方式：

```text
图片
  -> CLIP image embedding
  -> Logistic Regression / MLP / XGBoost
  -> 风险分类
```

如果标题、OCR、ASR 和画面不一致，可以做图文一致性检查：

```text
图像 embedding vs 文本 embedding
```

例如标题说“学习资料分享”，但画面里是二维码和“加私聊赚钱”，图文相似度低，同时风险 prompt 分数高，就可以进入人工审核或直接限流。
