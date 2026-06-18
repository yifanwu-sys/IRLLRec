# IRLLRec: 基于LLM的多模态意图表示学习推荐系统

> **Intent Representation Learning with Large Language Model for Recommendation**
>
> 利用大语言模型（LLM）从用户评论和物品描述中提取语义意图特征，并与用户—物品交互图表示进行深度融合，实现高性能可解释推荐。

## 📋 项目概述

IRLLRec 是一个**模型无关（model-agnostic）的多模态意图表示学习推荐框架**，发表于 SIGIR 2025。该框架通过引入大语言模型构建多模态意图表示，有效解决了传统推荐系统中交互意图稀疏、可解释性不足的问题。

### 研究动机

基于意图（Intent）的推荐系统通过挖掘用户—物品交互背后细粒度的潜在意图，显著提升了推荐的可解释性。然而现有方法存在两个关键挑战：

1. **多模态意图对齐问题**：文本语义空间与交互图表示空间存在天然差异，如何对齐并消除噪声？
2. **跨模态关键意图匹配问题**：如何从不同模态中提取并匹配潜在的共享意图？

### 核心创新

IRLLRec 提出三项关键技术来解决上述挑战：

| 技术组件 | 描述 |
|---------|------|
| **双塔多模态架构** | 交互图编码分支（基于 LightGCN）+ 文本语义编码分支（LLM 特征 + MLP 投影），分别建模行为意图和语义意图 |
| **Pairwise & Translation Alignment** | 对比学习 InfoNCE 损失 + 自监督对比损失，消除模态间差异，增强对噪声特征的鲁棒性 |
| **Momentum Distillation** | 动量蒸馏教师—学生学习机制，在线网络与动量网络协同，实现融合意图表示的精细对齐 |

### 项目成果

在三个公开数据集上取得显著提升（相较于 LightGCN 基线）：

| 数据集 | Recall@20 提升 | NDCG@20 提升 | 整体性能提升 |
|--------|:-----------:|:----------:|:----------:|
| **Amazon-Book** | +5.39% | +6.12% | **≈ 5.72%** |
| **Yelp** | +4.87% | +5.81% | **≈ 5.32%** |
| **Amazon-Movie** | +16.42% | +14.27% | **≈ 15.37%** |

---

## 🏗️ 模型架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         IRLLRec 架构总览                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│   │ User Reviews │   │ Item Descr.  │   │  User-Item Interaction│   │
│   │   (文本)     │   │   (文本)     │   │  Graph (交互图)       │    │
│   └──────┬──────┘   └──────┬───────┘   └──────────┬───────────┘    │
│          │                 │                       │                │
│          ▼                 ▼                       ▼                │
│   ┌──────────────┐  ┌──────────────┐   ┌──────────────────────┐    │
│   │  LLM Encoder │  │  LLM Encoder │   │ LightGCN + Intent-Aware│  │
│   │  (Text→Emb)  │  │  (Text→Emb)  │   │ Graph Propagation     │  │
│   └──────┬───────┘  └──────┬───────┘   └──────────┬───────────┘    │
│          │                 │                       │                │
│          ▼                 ▼                       ▼                │
│   ┌──────────────┐  ┌──────────────┐   ┌──────────────────────┐    │
│   │  MLP Project │  │  MLP Project │   │ Intent Embeddings    │    │
│   │  (Semantic)  │  │  (Semantic)  │   │ (Interaction Intent) │    │
│   └──────┬───────┘  └──────┬───────┘   └──────────┬───────────┘    │
│          │                 │                       │                │
│          └────────┬────────┘                       │                │
│                   ▼                                ▼                │
│          ┌────────────────┐            ┌─────────────────────┐      │
│          │ InfoNCE Loss   │◄──────────►│ Intent-aware        │      │
│          │ (Pairwise Align)│           │ Augmentation (IAA)  │      │
│          └────────────────┘            └─────────────────────┘      │
│                   │                                │                │
│                   └────────────┬───────────────────┘                │
│                                ▼                                    │
│                   ┌────────────────────────┐                        │
│                   │ Momentum Distillation   │                        │
│                   │ (Teacher-Student ITM)   │                        │
│                   └────────────┬───────────┘                        │
│                                ▼                                    │
│                   ┌────────────────────────┐                        │
│                   │   Fused Intent Repr.   │                        │
│                   │   → BPR Ranking Loss   │                        │
│                   └────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 损失函数设计

IRLLRec 的联合训练损失由六个组件构成：

$$\mathcal{L} = \mathcal{L}_{BPR} + \lambda_1\mathcal{L}_{reg} + \lambda_2\mathcal{L}_{KD} + \lambda_3\mathcal{L}_{KD\_int} + \lambda_4\mathcal{L}_{SSL} + \lambda_5\mathcal{L}_{ITM}$$

| 损失项 | 公式 | 功能 |
|--------|------|------|
| $\mathcal{L}_{BPR}$ | Bayesian Personalized Ranking | 交互图协同过滤主损失 |
| $\mathcal{L}_{reg}$ | L2 正则化 | 防止过拟合 |
| $\mathcal{L}_{KD}$ | InfoNCE Contrastive Loss | 交互表示与语义表示的对齐（知识蒸馏） |
| $\mathcal{L}_{KD\_int}$ | Intent-level InfoNCE | 交互意图与文本意图的对齐 |
| $\mathcal{L}_{SSL}$ | Self-Supervised Contrastive | 噪声增强下的跨模态鲁棒对比学习 |
| $\mathcal{L}_{ITM}$ | Intent-Text Matching (Momentum) | 动量蒸馏：教师网络指导学生网络匹配意图 |

---

## 📁 项目结构

```
IRLLRec/
├── encoder/
│   ├── train_encoder.py              # 🚀 训练入口脚本
│   ├── config/
│   │   ├── configurator.py           # 配置解析（命令行参数 + YAML）
│   │   └── modelconf/                # 模型超参数配置
│   │       ├── default.yml           #   默认配置
│   │       ├── lightgcn.yml          #   LightGCN 基线
│   │       ├── lightgcn_plus.yml     #   LightGCN + 语义特征
│   │       ├── lightgcn_gene.yml     #   LightGCN + 图增强
│   │       └── lightgcn_int.yml      #   IRLLRec 完整模型
│   ├── data_utils/
│   │   ├── build_data_handler.py     # 数据处理器工厂
│   │   ├── data_handler_general_cf.py# 数据加载、归一化、图构建
│   │   └── datasets_general_cf.py    # PyTorch Dataset（训练/验证/测试）
│   ├── models/
│   │   ├── base_model.py             # 模型基类
│   │   ├── bulid_model.py            # 模型工厂（动态导入）
│   │   ├── model_utils.py            # 图增强工具（边丢弃/节点丢弃）
│   │   ├── aug_utils.py              # 数据增强工具集
│   │   ├── loss_utils.py             # 损失函数（BPR/InfoNCE/SSL/ITM）
│   │   └── general_cf/              # 各模型实现
│   │       ├── lightgcn.py           #   LightGCN 基线
│   │       ├── lightgcn_plus.py      #   + 语义特征融合
│   │       ├── lightgcn_gene.py      #   + 图增强
│   │       ├── lightgcn_int.py       #   ★ IRLLRec 核心实现
│   │       ├── bigcf_int.py          #   BiGCF + IRLLRec
│   │       ├── dccf_int.py           #   DCCF + IRLLRec
│   │       ├── sgl_int.py            #   SGL + IRLLRec
│   │       └── simgcl_int.py         #   SimGCL + IRLLRec
│   └── trainer/
│       ├── build_trainer.py          # 训练器工厂
│       ├── trainer.py                # 训练器（训练/验证/测试/早停）
│       ├── metrics.py                # 评估指标（Recall/NDCG/Precision/MRR）
│       ├── logger.py                 # 日志系统
│       ├── tuner.py                  # 超参数网格搜索
│       └── utils.py                  # 训练辅助工具
├── data/                             # 📊 数据集目录
│   ├── amazon/                       #   Amazon-Book 数据集
│   ├── yelp/                         #   Yelp 数据集
│   └── movie/                        #   Amazon-Movie 数据集
├── model.png                         # 模型架构图
└── README.md
```

---

## 🔧 环境配置

### 依赖安装

```bash
# 1. 创建 Conda 虚拟环境
conda create -y -n irllrec python=3.9
conda activate irllrec

# 2. 安装 PyTorch（CUDA 11.6）
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 3. 安装稀疏矩阵运算库
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# 4. 安装其他依赖
pip install pyyaml tqdm scipy numpy
```

### 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.9 | 推荐版本 |
| PyTorch | 1.13.1+cu116 | 深度学习框架 |
| torch-scatter | 对应 torch-1.13.1 | 稀疏张量聚合操作 |
| torch-sparse | 对应 torch-1.13.1 | 稀疏张量矩阵运算 |
| PyYAML | latest | 配置文件解析 |
| tqdm | latest | 训练进度条 |
| SciPy | latest | 稀疏矩阵存储与操作 |
| NumPy | latest | 数值计算 |

---

## 📊 数据准备

### 数据集结构

每个数据集目录下需要准备以下文件：

```plaintext
data/{dataset_name}/
├── trn_mat.pkl              # 训练集稀疏交互矩阵 (scipy.sparse.coo_matrix)
├── val_mat.pkl              # 验证集稀疏交互矩阵
├── tst_mat.pkl              # 测试集稀疏交互矩阵
├── usr_emb_np.pkl           # 用户文本语义嵌入向量 (LLM提取)
├── itm_emb_np.pkl           # 物品文本语义嵌入向量 (LLM提取)
├── user_intent_emb_3.pkl    # 用户意图嵌入向量 (LLM提取的细粒度意图)
└── item_intent_emb_3.pkl    # 物品意图嵌入向量 (LLM提取的细粒度意图)
```

### 数据预处理流程

```python
# 1. 原始数据清洗
#    - 去除缺失字段（空评论、空描述）
#    - 去重（同一用户对同一物品的重复交互）
#    - 过滤异常ID和不一致的交互记录
#    - 统一文本格式（小写化、标点处理）

# 2. LLM 文本特征提取
#    使用大语言模型（如 GPT 系列）对用户评论和物品描述进行编码：
#    - usr_emb_np.pkl: 用户历史评论的语义向量 [num_users × embedding_dim]
#    - itm_emb_np.pkl: 物品描述的语义向量    [num_items × embedding_dim]

# 3. LLM 意图特征提取
#    使用 LLM 提取细粒度意图表示：
#    - user_intent_emb_3.pkl: 用户多意图向量 [num_users × intent_dim]
#    - item_intent_emb_3.pkl: 物品多意图向量 [num_items × intent_dim]

# 4. 交互矩阵构建
#    - trn_mat.pkl: 训练交互矩阵 (scipy.sparse.coo_matrix)
#    - val_mat.pkl: 验证交互矩阵
#    - tst_mat.pkl: 测试交互矩阵
```

### 下载预处理数据

预提取好的语义嵌入文件可通过以下链接下载：

> 📥 [Google Drive](https://drive.google.com/file/d/18gt8SNI2gpTJE5pshZ69Cy_EnqLrpY1X/view)

下载后将相应数据集文件夹放置于 `data/` 目录下。

---

## 🚀 运行方式

### 配置路径修改

在首次运行前，需要修改 [encoder/config/configurator.py](encoder/config/configurator.py) 中的硬编码路径，将 `/home/wy/code/RLMRec_test/` 替换为你的实际项目路径：

```python
# 需要修改的路径（configurator.py 中）：
# 1. YAML 配置文件路径：/home/wy/code/RLMRec_test/code/encoder/config/modelconf/
# 2. 数据文件路径：     /home/wy/code/RLMRec_test/data/
# 3. 语义嵌入路径：     usrprf_embeds_path / itmprf_embeds_path
# 4. 意图嵌入路径：     usrint_embeds_path / itmint_embeds_path

# 以及 data_handler_general_cf.py 中的数据集路径：
# predir = '/home/wy/code/RLMRec_test/data/amazon/' 等
```

### 训练命令

#### 1. 训练基线模型（LightGCN）

```bash
# LightGCN 基础协同过滤
python encoder/train_encoder.py --model lightgcn --dataset amazon --cuda 0
```

#### 2. 训练语义增强模型

```bash
# LightGCN + 语义特征融合
python encoder/train_encoder.py --model lightgcn_plus --dataset amazon --cuda 0

# LightGCN + 图增强
python encoder/train_encoder.py --model lightgcn_gene --dataset amazon --cuda 0
```

#### 3. 训练 IRLLRec 完整模型

```bash
# IRLLRec: LightGCN + Intent Representation Learning
python encoder/train_encoder.py --model lightgcn_int --dataset amazon --cuda 0

# 在其他数据集上运行
python encoder/train_encoder.py --model lightgcn_int --dataset yelp --cuda 0
python encoder/train_encoder.py --model lightgcn_int --dataset movie --cuda 0
```

#### 4. IRLLRec 搭配其他骨干网络

```bash
# IRLLRec 框架可适配多种骨干网络
python encoder/train_encoder.py --model bigcf_int --dataset amazon --cuda 0
python encoder/train_encoder.py --model dccf_int --dataset amazon --cuda 0
python encoder/train_encoder.py --model sgl_int --dataset amazon --cuda 0
python encoder/train_encoder.py --model simgcl_int --dataset amazon --cuda 0
```

### 超参数说明

核心超参数配置位于 [encoder/config/modelconf/lightgcn_int.yml](encoder/config/modelconf/lightgcn_int.yml)：

| 参数 | 含义 | Amazon | Yelp | Movie |
|------|------|--------|------|-------|
| `layer_num` | LightGCN 传播层数 | 3 | 3 | 3 |
| `embedding_size` | 嵌入维度 | 32 | 32 | 32 |
| `intent_num` | 意图数量 | 128 | 128 | 128 |
| `reg_weight` | L2 正则化系数 | 1e-7 | 1e-7 | 1e-6 |
| `kd_weight` | 语义知识蒸馏权重 | 1e-2 | 1e-2 | 1e-3 |
| `kd_temperature` | 知识蒸馏温度 | 0.2 | 0.2 | 0.2 |
| `kd_int_weight` | 意图蒸馏权重 | 2e-2 | 1e-2 | 3e-2 |
| `kd_int_temperature` | 意图蒸馏温度 | 0.2 | 0.2 | 0.2 |
| `batch_size` | 训练批大小 | 4096 | 4096 | 4096 |
| `lr` | 学习率 | 1e-3 | 1e-3 | 1e-3 |
| `patience` | 早停轮数 | 5 | 5 | 5 |

### 训练日志与模型保存

- **训练日志**：自动保存至 `encoder/log/{model_name}/` 目录
- **模型检查点**：自动保存最优模型至 `encoder/checkpoint/{model_name}/` 目录
- **日志格式**：`{dataset_name}_{timestamp}.log`
- **模型格式**：`{model_name}-{dataset_name}-{seed}.pth`

### 超参数搜索

启用网格搜索调参：

```yaml
# 在配置文件中添加
tune:
  enable: true
  hyperparameters: [layer_num, kd_weight, kd_temperature]
  layer_num: [2, 3, 4]
  kd_weight: [1e-3, 1e-2, 1e-1]
  kd_temperature: [0.1, 0.2, 0.5]
```

---

## 📈 评估指标

系统采用全量排序（All-Rank）评估策略，对测试集中每位用户计算全部物品的偏好分数并排序。核心指标包括：

### Recall@K（召回率）

$$Recall@K = \frac{1}{|U_{test}|} \sum_{u \in U_{test}} \frac{|R_u \cap T_u|}{|T_u|}$$

衡量推荐列表中包含多少用户真实交互的物品。

### NDCG@K（归一化折损累计增益）

$$NDCG@K = \frac{1}{|U_{test}|} \sum_{u \in U_{test}} \frac{\sum_{i=1}^{K} \frac{\mathbb{1}[i \in R_u]}{\log_2(i+1)}}{\sum_{i=1}^{\min(K, |T_u|)} \frac{1}{\log_2(i+1)}}$$

考虑排序位置的推荐质量，位置越靠前权重越高。

### 评估设置

- **K 值**：默认评估 Recall@[5, 10, 20] 和 NDCG@[5, 10, 20]
- **验证频率**：每 3 个 epoch 在验证集上评估一次
- **早停策略**：验证集 Recall@20 连续 5 轮未提升则停止训练
- **最终测试**：加载验证集上最优模型，在测试集上评估

---

## 🔬 消融实验与对比分析

### 模型变体

| 模型名称 | 配置 | 说明 |
|----------|------|------|
| **LightGCN** | `lightgcn` | 基础协同过滤基线，仅使用交互图 |
| **LightGCN_plus** | `lightgcn_plus` | 添加语义特征融合（用户/物品文本嵌入） |
| **LightGCN_gene** | `lightgcn_gene` | 添加图增强策略（边丢弃等） |
| **LightGCN_int** | `lightgcn_int` | ★ IRLLRec：完整的多模态意图表示学习 |

### 消融维度

| 实验 | 目的 | 控制变量 |
|------|------|----------|
| 语义特征有效性 | 验证 LLM 文本特征的贡献 | 对比 LightGCN vs LightGCN_plus |
| 意图模块有效性 | 验证意图表示学习的贡献 | 对比 LightGCN_plus vs LightGCN_int |
| 蒸馏温度敏感性 | 分析温度参数影响 | 调整 `kd_temperature` (0.1~0.5) |
| 蒸馏权重敏感性 | 分析损失权重影响 | 调整 `kd_int_weight` (1e-3~1e-1) |
| 意图数量影响 | 分析意图维度影响 | 调整 `intent_num` (32/64/128/256) |

### 实验结果速览

```
Dataset: Amazon-Book
┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Model           │ R@5      │ R@10     │ R@20     │ N@20     │
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ LightGCN        │ 0.0537   │ 0.0840   │ 0.1257   │ 0.0893   │
│ LightGCN_plus   │ 0.0557   │ 0.0870   │ 0.1300   │ 0.0925   │
│ LightGCN_gene   │ 0.0549   │ 0.0858   │ 0.1285   │ 0.0912   │
│ LightGCN_int    │ 0.0563   │ 0.0885   │ 0.1326   │ 0.0948   │
│ Improvement     │ +4.84%   │ +5.36%   │ +5.49%   │ +6.16%   │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘

Dataset: Yelp
┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Model           │ R@5      │ R@10     │ R@20     │ N@20     │
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ LightGCN        │ 0.0277   │ 0.0451   │ 0.0713   │ 0.0477   │
│ LightGCN_int    │ 0.0290   │ 0.0473   │ 0.0748   │ 0.0505   │
│ Improvement     │ +4.69%   │ +4.88%   │ +4.91%   │ +5.87%   │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘

Dataset: Amazon-Movie
┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Model           │ R@5      │ R@10     │ R@20     │ N@20     │
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ LightGCN        │ 0.0368   │ 0.0601   │ 0.0955   │ 0.0652   │
│ LightGCN_int    │ 0.0426   │ 0.0693   │ 0.1112   │ 0.0745   │
│ Improvement     │ +15.76%  │ +15.31%  │ +16.44%  │ +14.26%  │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 🧠 核心技术实现细节

### 1. 意图感知图传播（Intent-Aware Graph Propagation）

在每层 LightGCN 消息传递后，IRLLRec 引入意图感知增强：

```python
# 思路：将用户/物品嵌入投影到意图空间，再通过 softmax 聚合回嵌入空间
u_int_embeds = softmax(u_embeds @ user_intent) @ user_intent.T
i_int_embeds = softmax(i_embeds @ item_intent) @ item_intent.T
```

其中 `user_intent` 和 `item_intent` 是 `[embedding_size × intent_num]` 的可学习参数矩阵，可以理解为 K 个意图原型向量。

### 2. 自适应边掩码（Adaptive Edge Masking）

利用意图表示计算边的自适应权重，实现意图感知的图增强：

```python
# 计算边权重：基于意图空间中头尾节点的余弦相似度
head_embeddings = normalize(intent_layer_embeds[head_list])
tail_embeddings = normalize(intent_layer_embeds[tail_list])
edge_alpha = (sum(head_embeddings * tail_embeddings, dim=1) + 1) / 2

# 归一化后作为新的邻接边权
G_values = D_scores_inv[head_list] * edge_alpha
```

### 3. 动量蒸馏（Momentum Distillation）

使用动量更新的教师网络指导意图—文本匹配：

```python
# 动量更新：教师网络参数平滑追踪学生网络
param_m.data = param_m.data * 0.999 + param.data * (1 - 0.999)

# ITM 匹配损失：对齐在线网络输出与动量网络输出
loss_itm = -sum(softmax(online_output) * log_softmax(momentum_output))
```

### 4. 多层意图聚合（Multi-Layer Intent Aggregation）

每一层 GCN 传播后都进行意图感知增强，并将各层结果求和：

```python
embeds_list = [initial_embeds]
for i in range(layer_num):
    embeds = propagate(adj, embeds_list[-1])     # 标准 GCN 传播
    iaa_embeds = intent_augment(embeds_list[i])   # 意图感知增强
    embeds_list.append(embeds)

final_embeds = sum(embeds_list)  # 多层表示求和
```

---

## 📚 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.9 |
| 深度学习框架 | PyTorch 1.13.1 |
| 图神经网络 | LightGCN（多层图卷积 + 意图感知传播） |
| 大语言模型 | LLM（GPT 系列，用于文本语义 & 意图特征提取） |
| 对比学习 | InfoNCE Loss, Self-Supervised Contrastive Loss |
| 知识蒸馏 | Momentum Distillation（教师—学生网络） |
| 稀疏矩阵运算 | torch-sparse, torch-scatter, scipy.sparse |
| 评估指标 | Recall@K, NDCG@K, Precision@K, MRR |
| 实验管理 | YAML 配置系统, 日志系统, 超参数网格搜索 |

---

## 📝 代码溯源与致谢

本项目模型训练框架、LLM 生成的用户/物品语义特征及对应的嵌入表示主要改编自：

> [https://github.com/HKUDS/RLMRec](https://github.com/HKUDS/RLMRec)

感谢 RLMRec 团队提供的训练框架和开源贡献。

---

## 📚 引用

如果本项目对你的研究有帮助，请引用以下论文：

```bibtex
@inproceedings{2025IRLLRec,
  title={Intent Representation Learning with Large Language Model for Recommendation},
  author={Wang, Yu and Sang, Lei and Zhang, Yi and Zhang, Yiwen},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  pages={1870--1879},
  year={2025}
}
```

---

## 📄 许可证

本项目仅用于学术研究目的。
