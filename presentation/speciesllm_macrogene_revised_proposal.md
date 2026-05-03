# SpeciesLLM Macrogene：跨物种 Gene Space 前置对齐方案

## 0. 目标定位

本方案的目标不是训练 perturbation world model，而是训练一个 **跨物种 cell embedding backbone**。

核心思想是：不要让 backbone 直接面对每个物种独立的 gene vocabulary，而是在 backbone 之前加入一个 **macrogene construction layer**，先把不同物种的 gene space 映射到统一的 biological / evolutionary / functional macrogene space，再训练 cell embedding backbone。

整体路线为：

```text
gene sequence embedding
+ protein sequence embedding
+ gene function text embedding
        ↓
cross-species gene representation
        ↓
soft gene-to-macrogene assignment
        ↓
cell × macrogene expression matrix
        ↓
cell embedding backbone
```

因此，backbone 的核心输入不再是原始的 `cell × gene` 表达矩阵，而是跨物种对齐后的：

```text
cell × macrogene expression matrix
```

backbone 的核心学习目标是：

```text
cell × macrogene expression → universal cell embedding
```

---

## 1. 方案核心优点

这个方案适合构建跨物种 embedding backbone，主要优点包括三点。

### 1.1 把跨物种 gene vocabulary 对齐前置化

不同物种的基因集合并不一致。若直接把所有物种的 gene vocabulary 输入 backbone，模型需要同时学习：

```text
gene identity
+ gene function
+ cross-species homology
+ cell state representation
```

这会让 backbone 承担过多职责。

macrogene layer 的作用是先把不同物种的 gene space 压缩到统一的跨物种功能空间，使 backbone 不必直接处理每个物种独立的 gene vocabulary。

### 1.2 macrogene 比 raw gene 更适合作为跨物种输入单位

macrogene 可以吸收以下几类关系：

```text
one-to-one ortholog
many-to-many ortholog
remote homolog
protein family similarity
functional similarity
pathway / GO similarity
```

因此，macrogene 不是简单的 gene cluster，而是跨物种共享的 biological / evolutionary / functional unit。

### 1.3 backbone 任务更清晰

经过 macrogene layer 后，embedding backbone 只需要学习：

```text
cell × macrogene expression → cell embedding
```

它不再需要直接解决跨物种基因命名、同源关系、远缘同源、低资源物种 annotation 稀疏等问题。

这使模型分工更清楚：

```text
macrogene layer：负责跨物种 gene space 对齐
embedding backbone：负责学习 cell state representation
```

---

## 2. 原公式的维度问题与推荐重写

原文设定为：

```text
M ∈ R^{N × d}
B ∈ R^{N × K}
A ∈ R^{K × d}

M ≈ Z A
Z = M B
SRE = ||M - M B A||_F^2
```

这里存在维度问题：

```text
M: N × d
B: N × K
M B: (N × d) × (N × K) 不能相乘
```

因此，`Z = M B` 和 `M B A` 在当前定义下都不成立。

建议把公式重写为 gene-to-macrogene soft assignment 的形式。

### 2.1 推荐符号定义

设：

```text
E ∈ R^{G × d}              所有物种合并后的 gene embedding
W ∈ R^{G × K}              gene-to-macrogene soft assignment matrix
C ∈ R^{K × d}              macrogene prototype embedding
X_gene ∈ R^{cells × G}     原始 gene expression matrix
X_macro ∈ R^{cells × K}    macrogene expression matrix
```

其中：

```text
G：所有物种合并后的 gene 数量
d：gene embedding 维度
K：macrogene 数量
W_{gk}：gene g 分配到 macrogene k 的权重
```

### 2.2 Gene embedding reconstruction

用 macrogene prototype 重构 gene embedding：

$$
\hat{E} = W C
$$

其中：

```text
W: G × K
C: K × d
WC: G × d
E: G × d
```

维度是合法的。

### 2.3 推荐目标函数

建议使用：

$$
\mathcal{L}_{proto}
=
\|E - WC\|_F^2
+
\lambda_{graph}\operatorname{Tr}(W^\top L W)
+
\lambda_{sparse}\|W\|_1
+
\lambda_{entropy}\sum_{g,k} W_{gk}\log W_{gk}
$$

约束为：

$$
W_{gk} \ge 0, \qquad \sum_{k=1}^{K} W_{gk}=1
$$

解释如下：

```text
||E - WC||_F^2：让 macrogene prototype 能重构 gene embedding
Tr(W^T L W)：保留 gene embedding KNN 图上的局部平滑结构
||W||_1：鼓励 assignment 稀疏
entropy regularization：控制 assignment 的软硬程度
```

其中，`L` 是基于 gene representation KNN graph 得到的 graph Laplacian。

### 2.4 表达矩阵聚合

得到 `W` 后，将原始 gene expression 聚合为 macrogene expression：

$$
X_{macro} = X_{gene} W
$$

维度为：

```text
X_gene:  cells × genes
W:       genes × macrogenes
X_macro: cells × macrogenes
```

这才是维度正确、语义清楚的写法。

---

## 3. Soft Gene-to-Macrogene Assignment

不要把每个 gene hard assign 到单个 macrogene。

不推荐：

```text
gene g → only one macrogene k
```

推荐：

```text
gene g → top-r macrogenes with soft weights
```

也就是说：

$$
W_{gk} > 0 \quad \text{only for top-r nearest macrogenes}
$$

并且：

$$
\sum_k W_{gk}=1
$$

### 3.1 相似度计算

对 gene embedding $e_g$ 和 macrogene prototype $c_k$ 计算相似度。

可以使用 cosine similarity：

$$
s_{gk} = \frac{e_g^\top c_k}{\|e_g\|\|c_k\|}
$$

也可以使用负欧氏距离：

$$
s_{gk} = -\|e_g - c_k\|_2^2
$$

### 3.2 Top-r sparse softmax

对每个 gene 只保留最相近的 top-r 个 macrogene，其余位置 mask 掉：

```text
Top-r mask:
保留每个 gene 最相近的 r 个 macrogene，其余权重置为 -∞
```

然后在 top-r 集合内做 softmax：

$$
W_{gk}
=
\frac{\exp(s_{gk}/\tau)}
{\sum_{k' \in TopR(g)} \exp(s_{gk'}/\tau)}
$$

其中：

```text
r：每个 gene 允许连接的 macrogene 数量
τ：temperature，越小越接近 hard assignment，越大越平滑
```

### 3.3 推荐默认值

建议从以下配置开始：

```text
top-r = 3 或 5
τ = 0.05 ~ 0.2
```

推荐主实验：

```text
top-r = 5
τ = 0.1
```

这样可以同时满足三点：

```text
1. gene 可以属于多个功能模块
2. macrogene expression 不会过度离散
3. assignment 仍然保持稀疏，计算成本可控
```

---

## 4. 三种 Gene Embedding 的整合方式

本方案使用三类 gene representation：

```text
1. gene sequence embedding
2. protein sequence embedding
3. gene function text description embedding
```

不建议直接 concat 后聚类。原因是三类 embedding 的尺度、噪声、覆盖率和偏倚不同。

推荐使用 **multi-view gated fusion**。

### 4.1 各模态投影到统一维度

先将三类 embedding 分别投影到相同维度：

$$
h_{dna} = \operatorname{MLP}_{dna}(e_{dna})
$$

$$
h_{prot} = \operatorname{MLP}_{prot}(e_{prot})
$$

$$
h_{text} = \operatorname{MLP}_{text}(e_{text})
$$

### 4.2 学习模态权重

使用 gating network 学习不同 gene 对不同模态的依赖程度：

$$
\alpha
=
\operatorname{Softmax}
(
\operatorname{MLP}_{gate}([h_{dna}, h_{prot}, h_{text}, m])
)
$$

其中，`m` 是 modality mask，用来表示某个 gene 是否缺失某类 embedding。

### 4.3 融合 gene embedding

最终 gene representation 为：

$$
h_{gene}
=
\alpha_{dna}h_{dna}
+
\alpha_{prot}h_{prot}
+
\alpha_{text}h_{text}
$$

### 4.4 推荐模态优先级

建议遵循以下原则：

```text
protein embedding：主轴，负责跨物种功能和远缘同源关系
gene sequence embedding：补充演化和序列层信息
text embedding：弱语义先验，不要让它支配整个空间
```

### 4.5 为什么 text embedding 要谨慎使用

gene function text description 在 human / mouse 中通常更丰富，在低资源物种中更稀疏。

如果 text embedding 权重过大，macrogene space 可能变成：

```text
human/mouse annotation bias space
```

而不是：

```text
cross-species evolutionary functional space
```

因此建议加入：

```text
modality dropout
```

即在训练或构建 macrogene 时随机丢弃某些模态，使 fusion module 不会过度依赖 text description。

---

## 5. Macrogene 数量 K 的三档方案

`K` 是 macrogene layer 的核心超参数。

如果 `K` 太小，macrogene 过粗，会损失 cell-type marker 和细胞状态差异。

如果 `K` 太大，压缩效果变弱，跨物种对齐收益下降，backbone 计算成本也更高。

建议设置三档：

| 档位 | K | 用途 |
|---|---:|---|
| Compact | 512–1024 | 快速验证、低算力、初步跨物种整合 |
| Base | 2048–4096 | 主推荐方案，表达能力和压缩率较平衡 |
| Large | 8192+ | 高分辨率版本，适合资源充足时做 scaling |

主实验建议从：

```text
K = 2048 或 K = 4096
```

开始。

理由是：

```text
macrogene 不是 raw gene，而是压缩后的功能 / 演化单元。
```

因此，2048–4096 个 macrogene 通常比直接输入 20k raw genes 更适合 embedding backbone。

---

## 6. Backbone 预训练 Loss

本方案不纳入 Species / batch invariance regularization。

保留两个主 loss：

```text
L_pretrain = L_masked_macro + λ L_contrastive
```

其中：

```text
L_masked_macro：学习 cell 内部 macrogene 共表达结构
L_contrastive：塑造 cell embedding 空间结构
```

---

## 6.1 Masked Macrogene Modeling

目标：

```text
让模型理解一个细胞内部 macrogene 之间的共表达结构。
```

做法：

```text
1. 输入 X_macro
2. 随机 mask 一部分 macrogene expression
3. 让模型根据未 mask 的 macrogene 预测被 mask 的 macrogene expression
```

公式为：

$$
\mathcal{L}_{masked}
=
\operatorname{ReconLoss}
(
\hat{X}_{masked},
X_{masked}
)
$$

loss 可以选择：

```text
MSE
Huber
Poisson
Negative Binomial
ZINB
discretized cross entropy
```

推荐原则：

```text
如果 macrogene expression 是连续 log-normalized value：使用 MSE 或 Huber。
如果保留 count-like 特性：使用 NB 或 ZINB。
```

对 embedding backbone 的第一版实现，建议先用：

```text
Huber loss
```

它比 MSE 对极端表达值更稳健，又比 NB / ZINB 简单。

---

## 6.2 Contrastive Learning

目标：

```text
直接塑造 cell embedding space。
```

masked modeling 让模型学会补全表达结构，但不保证 cell embedding 的几何结构一定适合检索、聚类、跨物种映射和 downstream transfer。

contrastive learning 直接约束：

```text
相似细胞 embedding 更近
不相似细胞 embedding 更远
```

可使用 InfoNCE：

$$
\mathcal{L}_{contrastive}
=
-
\log
\frac{
\exp(\operatorname{sim}(z_i,z_i^+)/\tau)
}{
\sum_j \exp(\operatorname{sim}(z_i,z_j)/\tau)
}
$$

其中：

```text
z_i：cell embedding
z_i^+：正样本 cell embedding
z_j：batch 内其他负样本或弱负样本
τ：temperature
```

### 6.2.1 正样本构造

适合本方案的正样本包括：

```text
同一个 cell 的两种 mask view
同一个 cell 的两种 dropout / noise augmentation view
同一 cell type / tissue / annotation group 内的细胞
同一物种或跨物种中已知可对齐的细胞类型
```

如果不希望引入 species / batch invariance regularization，则不要使用 adversarial species loss，也不要强制 species mixing。

contrastive learning 在这里只作为 cell embedding geometry 的塑形目标。

### 6.2.2 为什么 masked modeling 和 contrastive learning 可以共存

两者关注点不同：

```text
masked modeling：学习 cell 内部 macrogene 共表达结构
contrastive learning：学习 cell-level embedding 空间结构
```

因此它们互补：

```text
masked modeling 负责“读懂表达结构”
contrastive learning 负责“把细胞放到合理的 embedding 空间”
```

对 embedding backbone 来说，两者一起用通常比只做 masked reconstruction 更合理。

---

## 7. Macrogene Layer Sanity Check

在正式训练 backbone 前，应先验证 macrogene 是否构建成功。

建议保留以下 5 个检查。

| 检查 | 目的 |
|---|---|
| Ortholog enrichment | 同源基因是否更容易落在相同或相邻 macrogene |
| Protein family enrichment | PFAM / eggNOG family 是否在 macrogene 内富集 |
| GO / pathway coherence | macrogene 内部是否功能一致 |
| Species mixing score | macrogene 是否被单一物种垄断 |
| Marker preservation | 经典 cell-type marker 聚合后是否仍能区分细胞类型 |

最重要的是：

```text
Marker preservation
```

原因是：如果 macrogene 聚合后把 cell-type marker 信号抹掉，那么后续 embedding backbone 再强也很难恢复这些信息。

---

## 8. 推荐最小可行实现版本

第一版可以按以下设置实现：

```text
Gene representation:
  protein embedding: ESM2
  gene sequence embedding: DNA model embedding
  text embedding: gene function description embedding
  fusion: multi-view gated fusion + modality dropout

Macrogene assignment:
  K = 2048 或 4096
  top-r = 5
  τ = 0.1
  W row-normalized soft assignment

Macrogene expression:
  X_macro = X_gene W

Backbone pretraining:
  L_pretrain = L_masked_macro + λ L_contrastive
  L_masked_macro: Huber 或 MSE
  L_contrastive: InfoNCE

Sanity check:
  ortholog enrichment
  protein family enrichment
  GO / pathway coherence
  species mixing score
  marker preservation
```

---

## 9. 推荐最终表述

本方案可以概括为：

```text
SpeciesLLM Macrogene 方案通过多模态 gene representation
构建跨物种 gene-to-macrogene soft assignment，
将不同物种的 gene expression matrix 映射到统一的 macrogene expression space。
在此基础上，embedding backbone 只需学习 cell × macrogene expression 到 universal cell embedding 的映射，
从而把跨物种 gene vocabulary 对齐问题从 backbone 内部前置到 macrogene construction layer。
```

一句话版本：

```text
先用多模态 gene embedding 构建跨物种 macrogene space，
再在 cell × macrogene expression 上训练 universal cell embedding backbone。
```

