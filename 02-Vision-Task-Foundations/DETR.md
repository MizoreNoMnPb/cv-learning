# DETR 系目标检测器：从 DETR 到 Grounding DINO

> 本文比较集合预测、可变形注意力、去噪训练与文本条件检测。首次阅读按 DETR、[Deformable DETR](#deformable-detr)、[DINO](#dino)、[Grounding DINO](#grounding-dino)的顺序阅读；第 1.8 节的 Swin 背景可在需要了解骨干网络时补读。

前置知识：[R-CNN 系列](rcnn.md)、[注意力](../01-Foundations/attention.md)和 [交叉注意力](cross-attention.md)。

---

## 一、DETR (DEtection TRansformer)

> **[论文发现]** DETR（DEtection TRansformer，检测 Transformer）通过一对一集合监督直接预测检测结果；本节结构与目标来自 [1，第 5–8 页]。

### 1.1 动机：抛弃 NMS 和 Anchor

传统检测器（Faster R-CNN, YOLO）依赖：
- 大量 anchor boxes / proposals
- NMS（非极大值抑制）后处理
- 手工设计的组件（RPN, RoI Pooling）

DETR 的目标：将目标检测变成一个**端到端的集合预测问题**——直接输出无序的检测结果集合。

### 1.2 图像输入序列化：2D 特征图 → 1D 序列

这是 DETR 最关键的表示转换。Transformer 接收序列表示，而卷积特征保留二维空间网格；DETR 通过以下流程连接二者。

#### 完整流程

```
原始图像 (3, H₀, W₀)  例如: (3, 800, 1066)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 1: CNN Backbone 特征提取                │
│   ResNet-50 去掉最后的 avgpool 和 FC 层      │
│   保留空间结构                                │
└─────────────────────┬───────────────────────┘
                      │
    (2048, H, W)     H=H₀/32, W=W₀/32
    例如: (2048, 25, 34)    ← 800/32≈25, 1066/32≈34
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ Step 2: 1×1 卷积降维                         │
│   1×1 卷积：2048 输入通道 → 256 输出通道    │
│   将 2048 通道压缩到 d_model=256              │
└─────────────────────┬───────────────────────┘
                      │
    (256, 25, 34)    C=256(即d_model), H=25, W=34
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ Step 3: 空间展平 (Flatten)                   │
│   (256, 25, 34) → (256, 850) → (850, 256)   │
│   每个空间位置变成一个 256 维的 token          │
│   总共 25×34=850 个 image tokens             │
└─────────────────────┬───────────────────────┘
                      │
    (850, 256)        850 个 token，每个 256 维
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ Step 4: 加位置编码 (Positional Encoding)      │
│   基线采用固定二维正弦位置编码                │
│   每个 (row, col) 位置生成唯一的 256 维编码     │
│   形状也是 (850, 256)，加入每层注意力的Q/K     │
└─────────────────────┬───────────────────────┘
                      │
    (850, 256)        这是送入 Transformer Encoder 的最终序列
                      │
                      ▼
              Transformer Encoder
```

#### Step 1: Backbone 特征提取


**为什么选 stride=32 的特征层**：
- 太低层（stride=8）：token 数量 $H/8 \times W/8$ 太大，Encoder self-attention 是 $O(N^2)$ 不可承受
- 太高层（stride=64）：空间分辨率太低，小物体位置信息丢失严重
- stride=32 是 DETR 在精度和效率之间的平衡（Deformable DETR 后来通过多尺度特征 + deformable attention 解决了这个问题）

#### Step 2: 1×1 卷积降维

ResNet 输出的 2048 维对于 Transformer 来说太宽（计算量大）。用 1×1 卷积降到 DETR 的 `d_model=256`：


1×1 卷积本质是对每个空间位置做独立的线性变换：将 2048 维特征向量投影到 256 维，不同位置共享同一个投影矩阵。

#### Step 3: 空间展平——序列化

这是**最关键**的一步：将 2D 特征图塌陷为 1D 序列。


**展平前 vs 展平后**：

```
展平前 (2D 空间结构):
位置 (0,0)  →  [0.2, -0.5, ..., 0.1]  (256维)
位置 (0,1)  →  [0.3,  0.1, ..., 0.4]
...
位置 (24,33)→  [-0.1, 0.2, ..., 0.3]

展平后 (1D 序列):
Token 0:  位置 (0,0)  的 256 维特征
Token 1:  位置 (0,1)  的 256 维特征
Token 2:  位置 (0,2)  的 256 维特征
...
Token 849: 位置 (24,33) 的 256 维特征
```

**[数学推导]** 展平只是可逆的索引重排，不会删除特征值。问题在于不带位置输入的自注意力对 token 排列等变，无法仅从这个运算规则判断原二维邻接关系；位置编码提供坐标信息。

#### Step 4: 位置编码——保留空间信息

DETR 使用**固定的 2D 正弦位置编码**（对行和列分别编码再拼接）：


**[论文发现]** 原文比较固定与可学习位置编码，基线采用固定正弦编码，并在每层注意力中加入位置表示。[1，第 13 页 Table 3、第 20 页 Figure 10] **[分析判断]** 正弦规则可在新坐标求值，可学习编码也可通过插值等方式适配不同尺寸；是否泛化取决于具体结构与训练，不能说可学习编码只能处理固定尺寸。

#### 序列化后的尺寸流转

```
原始图像:    (B, 3, 800, 1066)
Backbone:    (B, 2048, 25, 34)     H=25, W=34
1×1 Conv:    (B, 256, 25, 34)      C: 2048→256
Flatten:     (B, 850, 256)         N=25×34=850 tokens
PosEnc:      (B, 850, 256)         保留为独立位置输入
    │
    ▼
Transformer Encoder 输入:
  src: (B, 850, 256)   ← 每个 token 是图像一个空间位置的特征
  pos: (B, 850, 256)   ← 该位置的 2D 位置编码
    │
    ▼
Encoder 输出:
  memory: (B, 850, 256) ← 经过全局 self-attention 增强后的特征
    │
    ▼
Decoder Cross-Attention:
  query: (B, 100, 256)  ← Object Queries 从 memory 中"检索"物体
  key/value: memory
```

#### 为什么这个序列化方案巧妙

1. **无信息损失**：展平是可逆的（知道 H, W 就能还原 2D 结构），本质上只是改变了内存排列
2. **显式位置输入**：位置表示与内容特征分别提供给注意力；不能假定骨干特征完全不含位置信息。
3. **序列接口**：复用 Transformer 的注意力与前馈模块，但目标查询并行解码，不采用原始文本 Transformer 的自回归因果解码。[1，第 7 页]

### 1.3 核心架构

```
输入图像 (3, H, W)
    │
    ▼
CNN Backbone (ResNet-50) → 特征图 (2048, H/32, W/32)
    │
    ▼
1×1 Conv 降维 → (256, H/32, W/32)
    │
    ▼
Flatten → 特征序列 (H_f W_f, 256)，另提供位置编码
    │
    ▼
Transformer Encoder × 6 层 → 全局上下文增强的特征序列
    │
    ▼
100 个 Object Queries（可学习的预测槽位，可输出 no object）
    │
    ▼
Transformer Decoder × 6 层 → 每个 query 解码出一个检测结果
    │  ├── Self-Attention: queries 之间交互
    │  └── Cross-Attention: queries 从 encoder 输出(memory)检索信息
    │
    ▼
Prediction Heads (共享 FFN)，每个 query 独立预测:
    ├── Class: (100, num_classes+1)  ← +1 是 "no object" (∅) 类
    └── BBox:  (100, 4) 中心点 (cx, cy, w, h) 归一化到 [0,1]
```

### 1.4 核心创新一：Object Queries


- 这 100 个 queries 是**可学习的参数**（不是从图像计算的）
- **[论文发现]** 原文可视化观察到不同槽位具有位置与尺寸偏好。[1，第 14 页 Figure 7]
- **[分析判断]** 这种偏好是训练结果，不是每个查询必须对应固定位置或固定类别的保证。

### 1.5 核心创新二：Bipartite Matching (匈牙利匹配)

DETR 固定输出 N=100 个预测，但真实物体数通常远小于 100。怎么配对？

**匈牙利算法**：找预测集合和 GT 集合之间的**最优一对一匹配**：

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})$$

其中匹配代价：
$$\mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i \neq \emptyset\}} \cdot \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} \cdot \mathcal{L}_{box}(b_i, \hat{b}_{\sigma(i)})$$

**直观理解**：
1. 算所有 (pred_i, gt_j) 对的损失矩阵 (100 × M)
2. 用匈牙利算法找一对一最优匹配
3. 未匹配的 pred 优化为 "no object" (∅)


### 1.6 损失函数

匹配后的损失 = 分类损失 + 框回归损失：

$$\mathcal{L} = \sum_{i=1}^N \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} \cdot \left( \lambda_{L1} \|b_i - \hat{b}\|_1 + \lambda_{giou} \cdot (1 - \text{GIoU}) \right) \right]$$

- 分类：交叉熵，∅ 类也参与；原文将 ∅ 类项权重降为其他类别的 1/10，上式未展开该加权。[1，第 5–6 页]
- 框回归：L1 Loss + GIoU Loss（两者互补）

### 1.7 DETR 的优缺点

**结构特点**：

- 使用固定数量的目标查询和集合预测目标。
- 通过一对一匹配分配训练监督。
- 通过注意力让查询与图像特征交互。

**缺点**：
- **训练收敛成本**：应结合数据、训练轮数和配置比较，不能仅凭两种方法的轮数判断
- 小物体检测差（单尺度特征 + 全局 attention 对细节不敏感）
- 计算量大（encoder 在 HW 尺度做 self-attention，$O(H^2W^2)$）

---

### 1.8 补充阅读：Swin Transformer 层次化骨干网络

> **[论文发现]** Swin 的层次结构与移位窗口来自 [2，第 3–5 页]。下图以 Swin-T 为例。

本节补充 Swin Transformer 的窗口与层次化表示，用于理解采用这一骨干网络的检测器变体。首次阅读方法演进时，可先跳到 [Deformable DETR](#deformable-detr)，之后再回看。

#### 标准 ViT 用作检测 backbone 的限制

把输出单一尺度特征的标准视觉 Transformer 用作检测骨干网络时，需要考虑以下结构差异。它们说明需要怎样适配特征，不能直接推出这种骨干网络不适用于密集预测：

| 问题 | 原因 | 检测需要 |
|------|------|---------|
| **单尺度输出** | ViT 全程保持 16×16 patch → 始终一个分辨率 | 多尺度特征（FPN 需要 stride 8/16/32） |
| **计算量 $O(N^2)$** | 全局 self-attention，token 数 N 固定 | 高分辨率特征层 N 很大，计算不可承受 |
| **无局部归纳偏置** | 纯全局 attention，对局部结构不敏感 | 检测需要精细的局部定位 |

#### Swin 的核心创新一：层次化结构 (Hierarchical)

Swin 不是全程维持一个分辨率，而是像 CNN 一样**逐阶段降采样**：

```
输入图像 (3, H, W)
    │
    ▼
Patch Partition: 切成 4×4 patches (每个 patch = 4×4×3 = 48 维)
    │
    ▼
Stage 1: Linear Embedding + Swin Blocks × 2
    │ → (C, H/4, W/4)    C=96
    │
    ▼
Patch Merging: 2×2 邻域拼接 → 4C → Linear → 2C
    │
    ▼
Stage 2: Swin Blocks × 2
    │ → (2C, H/8, W/8)    2C=192   ← stride 8 的骨干特征
    │
    ▼
Patch Merging
    │
    ▼
Stage 3: Swin Blocks × 6
    │ → (4C, H/16, W/16)  4C=384   ← stride 16 的骨干特征
    │
    ▼
Patch Merging
    │
    ▼
Stage 4: Swin Blocks × 2
    │ → (8C, H/32, W/32)  8C=768   ← stride 32 的骨干特征
```

这产生不同步幅的骨干特征。FPN（Feature Pyramid Network，特征金字塔网络）还包含横向与自顶向下融合，不能把原始 stage 输出直接称为 FPN 的 P3/P4/P5。

#### Swin 的核心创新二：Shifted Window Attention

标准 ViT 的全局 attention 是 $O(H^2W^2)$。Swin 用**窗口 attention + shifted 窗口**来高效近似：

##### 窗口划分（Window Partition）

将特征图分成不重叠的 $M \times M$ 窗口（默认 M=7），attention 只在窗口内计算：

```
特征图 (56 × 56) → 分成 64 个 7×7 窗口
每个窗口内独立做 self-attention

计算量：O(HW × M²) vs 标准 attention O(H²W²)
对 56×56 特征图:
  ViT: O(56⁴) ≈ 9.8M × d
  Swin: O(56² × 7²) ≈ 154K × d  → 减少 64 倍
```


**问题**：纯窗口 attention 失去了窗口之间的信息交互 → 感受野被限制在 7×7 局部。

##### Shifted Window（交替偏移）

Swin 的**连续两个 block** 采用交替的窗口划分：

```
Block 1 (W-MSA):  标准 7×7 窗口
┌──────┬──────┐
│  W1  │  W2  │
├──────┼──────┤
│  W3  │  W4  │
└──────┴──────┘

Block 2 (SW-MSA): 窗口偏移 (⌊M/2⌋, ⌊M/2⌋) = (3, 3) 像素
┌──┬──────────┐
│a │     b    │
├──┼────┬─────┤
│  │ W1'│ W2' │
│c ├────┼─────┤
│  │ W3'│ W4' │
└──┴────┴─────┘
   d
```

- Block 1：标准窗口内的 attention（局部建模）
- Block 2：偏移窗口 → 每个窗口包含了不同原窗口的边界 → 实现**跨窗口信息交互**
- 交替执行 → 等价于全局感受野的逐步扩大


**Cyclic Shift + Mask 的机制**：

Shifted window 导致窗口大小不均（边角窗口只有部分），直接 attention 会浪费计算。Swin 用 **cyclic shift** 把所有小块拼成完整窗口，再用 attention mask 阻止语义上不应交互的区域通信：

```
原始特征图            Cyclic Shift 后
┌──┬──────────┐       ┌─────┬─────┐
│a │     b    │       │ W1' │ W2' │   a移到右下，d移到左上
├──┼────┬─────┤  →    ├─────┼─────┤   但 a 和 W1' 的其余部分不应交互
│  │ W1'│ W2' │       │ W3' │ W4' │   → 用 mask 屏蔽
│c ├────┼─────┤       └─────┴─────┘
│  │ W3'│ W4' │
└──┴────┴─────┘
   d
```


#### Swin 的变体

| 变体 | dim (C) | Stage depths | Params | 使用场景 |
|------|---------|-------------|--------|---------|
| **Swin-T** | 96 | {2,2,6,2} | 28M | 轻量 |
| **Swin-S** | 96 | {2,2,18,2} | 50M | 中等 |
| **Swin-B** | 128 | {2,2,18,2} | 88M | Grounding DINO |
| **Swin-L** | 192 | {2,2,18,2} | 197M | 原文的大容量变体 |

#### 在 DETR 系中的使用

DINO 和 Grounding DINO 用 Swin 替换 ResNet backbone 的方式：


**Swin 天然输出多尺度特征，不需要额外构造图像金字塔**。DINO/Grounding DINO 仍在不同 stage
输出后使用轻量投影，将各尺度通道统一到 256。

#### ViT vs Swin 对比

| | ViT (如 CLIP ViT-B/16) | Swin Transformer |
|---|---|---|
| **结构** | 224×224 输入、16×16 像素块时为 14×14 token 网格 | 层次化，4 个 stage 逐步降采样 |
| **Attention** | 全局 self-attention，$O(N^2)$ | 窗口内 attention + Shifted window，$O(N \cdot M^2)$ |
| **多尺度输出** | 无（需额外模块） | 天然多尺度（4 个 stage） |
| **计算效率** | 全局关系项随 token 数平方增长 | 固定窗口大小时，关系项随 token 数线性增长 |
| **适用任务** | 分类、CLIP 对比学习 | **检测、分割（密集预测任务）** |
| **网络深度** | 各层保持同一 token 分辨率 | stage 深度由变体指定，并非单调递增 |

---

<a id="deformable-detr"></a>

## 二、Deformable DETR

> **[论文发现]** 可变形注意力、多尺度结构与比较结果来自 [3，第 5–8 页]。

### 2.1 解决 DETR 的两个核心问题

1. **慢收敛**：DETR 的 cross-attention 需要从全局搜索 → Deformable DETR 用多尺度可变形 attention，聚焦于参考点附近
2. **小物体差**：DETR 只用单尺度特征 → Deformable DETR 用多尺度特征图

### 2.2 Deformable Attention

标准 Attention：
$$\text{Attn}(z_q, x) = \sum_{m=1}^{HW} \text{softmax}(q \cdot k_m) \cdot v_m$$

**[数学推导]** 单个查询遍历 $HW$ 个位置，关系计算关于位置数为线性；若查询也有 $HW$ 个，全部自注意力才产生 $(HW)^2$ 个关系项。还需乘以特征宽度并计入投影。

**Deformable Attention**：
$$\text{DeformAttn}(z_q, \hat{p}_q, x) = \sum_{h=1}^{H} W_h \left[ \sum_{k=1}^{K} A_{hqk} \cdot W_h' x(\hat{p}_q + \Delta p_{hqk}) \right]$$

- 每个 query 只在 K 个**可学习采样点**上做 attention（K=4, 远小于 HW）
- 采样点由 reference point $\hat{p}_q$ + 可学习偏移 $\Delta p$ 确定
- 多尺度：在多尺度特征图上分别采样


**为什么 Deformable Attention 更快收敛**：
- 标准 cross-attention 需要从所有位置中学习关注正确区域（500 epochs）
- Deformable attention 从 reference point 附近开始搜索（局部先验），大大缩小搜索空间（50 epochs）

### 2.3 多尺度特征

**[论文发现]** 原文用 ResNet 的 C3/C4/C5 特征经 1×1 投影，再从 C5 经 stride-2 卷积得到第四层；四个尺度通道统一为 256，对应步幅 8、16、32、64。多尺度注意力在各层分别采样并归一化权重，同时加入尺度编码。它没有使用额外的 FPN 自顶向下结构。[3，第 6 页公式 (3)]


### 2.4 与 DETR 的对比

| | DETR | Deformable DETR |
|---|---|---|
| **原文训练轮数** | 500 epochs | 50 epochs；轮数比不等于耗时比 |
| **小物体 AP** | 20.5 | 26.4；COCO val2017、基础 ResNet-50 配置 [3，第 8 页 Table 1] |
| **Encoder Self-Attn** | 全局 (HW × HW) | Deformable (HW × K) |
| **Decoder Cross-Attn** | 全局 (N × HW) | 多尺度 Deformable (N × K × L) |
| **特征** | 单尺度 | 多尺度 |

---

<a id="dino"></a>

## 三、DINO (DETR with Improved DeNoising anchOr boxes)

> **[论文发现]** 本节按归档 arXiv:2203.03605v4 解读；正式发表为 ICLR 2023，页码不能与其他版本混用。[4]

### 3.1 核心改进

DINO 在 Deformable DETR 基础上做了三处关键改进，进一步加速收敛和提升性能。

#### (1) Contrastive DeNoising (CDN)

训练时，在 object queries 中添加**噪声 GT**（略微偏移真实框）：


**为什么 CDN 有效**：
- 标准 DETR 中，decoder 需要从头学习"从随机 query 到物体"的映射
- CDN 提供了一个**显式的去噪任务**：给定近似位置 + 噪声，恢复精确位置
- Positive 和 Negative 组的对比增强了模型对定位精度的敏感度

#### (2) Mixed Query Selection

**[论文发现]** 两阶段 Deformable DETR 用编码器选出的特征初始化位置与内容；DINO 的 Mixed Query Selection（混合查询选择）只从高分编码器位置初始化动态锚框，内容查询仍为可学习嵌入。这样位置依赖当前图像，而初始内容不直接复制尚未精炼的局部特征。[4，第 9–10 页 §3.4] 一阶段 Deformable DETR 与两阶段变体应分开。


#### (3) Look Forward Twice

**[论文发现]** 原始 DETR 已对每个解码层加入辅助预测损失。[1，第 7–8 页] DINO 的 Look Forward Twice（向前看两次）针对迭代框细化的梯度路径：让第 $i$ 层框偏移受到本层与相邻后一层框损失的共同监督，而不是只接受本层框损失。它不是简单取消所有 stop-gradient，更不是“原 DETR 只在最后一层反传”。[4，第 10–11 页 Figure 6、§3.5]

### 3.2 DINO 整体架构

```
图像 → Backbone → 多尺度特征
                    │
            Transformer Encoder (Deformable Self-Attn)
                    │
            ┌───────┴────────┐
            │                 │
    Content Queries    从 Encoder 选 Top-K 位置 → Position Queries
    (可学习)                │
            │                 │
            └───────┬────────┘
                    │
            CDN Queries (GT + 噪声) ─────┐
                    │                    │
            ┌───────┴────────┐          │
            │  Decoder × 6   │←─────────┘
            │  (Deformable   │
            │   Cross-Attn)  │
            │  Look Forward   │
            │  Twice         │
            └───────┬────────┘
                    │
            Prediction Heads
                    │
            Hungarian Matching → Loss
```

### 3.3 性能对比

| 模型 | Backbone | Epochs | AP (COCO) |
|------|----------|--------|-----------|
| DETR | R50 | 500 | 42.0 |
| Deformable DETR | R50 | 50 | 43.8 |
| DINO-4scale | R50 | 12 | 49.0；归档 v4 第 12 页 Table 1 |
| DINO-5scale | R50 | 12 | 49.4；归档 v4 第 12 页 Table 1 |

**[论文发现]** 以上是对应论文在 COCO val2017 的报告值；DETR/Deformable DETR 取自 [3，第 8 页 Table 1]。多尺度数、增强、查询与训练方法不同，不能把 AP 差异全部归因于某一项改进，也不能由轮数直接推出计算成本。

---

<a id="grounding-dino"></a>

## 四、Grounding DINO

> **[论文发现]** Grounding DINO 的三个跨模态融合环节来自 [5，第 3–5 页]。

### 4.1 核心目标

将闭集检测的固定类别输出扩展为文本条件的区域—词语匹配；可查询的类别不必限于下游检测标注集合，但不能保证任意文本类别都能被正确检测。

### 4.2 架构创新

#### 两流特征提取

```
图像 → Image Backbone (Swin-B) → 多尺度图像特征 (4个尺度)
文本 → Text Backbone (BERT)    → 文本 token 特征
```

#### Feature Enhancer

```
图像特征 → Deformable Self-Attention → 增强的图像特征
文本特征 → 标准 Self-Attention       → 增强的文本特征
        ↓
图像特征 + 文本特征 → Cross-Attention → 跨模态融合特征
```

**Cross-Attention 的作用**：
- 图像特征作为 Q，文本特征作为 K, V
- 让图像的每个位置去"检索"相关的文本语义
- 例如：汽车的图像区域会与 "car" 的文本 token 高度响应

#### Language-Guided Query Selection

**[论文发现]** 语言引导查询选择先比较各图像位置与文本 token 的相似度，再按每个图像位置的最大文本响应选择高分候选，作为解码器查询的位置来源。[5，第 4 页公式 (1)]


**与传统 Query Selection 的区别**：
- DINO：依据闭集分类头的候选类别分数选择编码器位置
- Grounding DINO：根据图像-文本的**跨模态响应**选择（开集）

#### Cross-Modality Decoder

Decoder 的每层包含三种 attention：
1. **Self-Attention**：queries 之间交互（抑制重复检测）
2. **Image Cross-Attention**：queries → 图像特征（定位）
3. **Text Cross-Attention**：queries → 文本特征（识别）


#### 文本输入处理

Grounding DINO 的文本输入格式：用 "." 分隔类别名：
```
"car . bus . truck . van . freight car ."
```

每个类别的名称被 tokenize 后送入 BERT，所有类别共享同一个文本序列。

## 五、DETR 系列对比总结

| | DETR | Deformable DETR | DINO | Grounding DINO |
|---|---|---|---|---|
| **Backbone** | ResNet-50 | ResNet-50 | ResNet-50 / **Swin-L** | **Swin-B** |
| **Query** | 可学习 (100) | 可学习 (300) | Mixed (Content 可学习 + Position 从 Encoder) | 语言引导的 Query Selection |
| **Attention** | 标准 Self/Cross | Deformable | Deformable | Deformable + Text Cross-Attn |
| **收敛** | 500 epochs | 50 epochs | 12 epochs (R50) / 36 epochs (Swin-L) | — |
| **训练技巧** | — | — | CDN + Look Forward Twice | Grounded Pre-Training |
| **是否开集** | 否 | 否 | 否 | 是 |
| **文本输入** | — | — | — | BERT-encoded class names |
| **多尺度特征** | 无（单尺度 stride=32） | 多尺度 (stride 8~64) | 多尺度 | 多尺度 (来自 Swin 层次结构) |

---

## 六、常见理论问题

### Q: DETR 的 Object Queries 为什么每个会自动专精于不同位置？

**A**: 匈牙利匹配的一对一性质起了关键作用——如果两个 query 都预测同一个物体，只有一个能匹配到 GT（因为匹配是一对一的），另一个会被惩罚为 ∅。经过训练，queries 自动分化以避免竞争，每个 query 学会关注不同的空间区域和尺度。

### Q: Deformable Attention 的复杂度从 $O(N^2)$ 降到多少？

**[数学推导]** 仅统计关系采样项时，单尺度由 $N^2$ 变为每头 $NK$；多尺度还需乘尺度数。完整模块仍含值投影与输出投影，不能把关系项比例当作整体 FLOPs 或耗时比例。[3，第 5–6 页及附录 A.1]

### Q: Grounding DINO 和 GLIP 的主要区别？

**A**: 
- **GLIP**：基于 DyHead + BERT，将检测重新表述为 phrase grounding（区域-短语匹配），效果强但架构较重
- **Grounding DINO**：以 DINO 为基础，在特征增强、查询选择、解码三个阶段引入语言信息；性能需注明预训练数据、零样本或微调协议后比较。[5，第 3–6 页]
- Grounding DINO 的 Text Cross-Attention 在 decoder 每一层都做，GLIP 偏向在特征层做 fusion

### Q: Swin Transformer 的核心创新是什么？为什么比 ViT 更适合做检测 backbone？

**A**: 两个核心创新：
1. **层次化结构**：4 个 stage 逐步降采样，产生不同分辨率骨干特征；是否添加 FPN 由检测头决定。
2. **Shifted Window Attention**：窗口内计算注意力，交替偏移使信息逐层跨窗口传播；有限层数不等于单层全局注意力。

**[分析判断]** Swin 提供直接可用的层次特征与局部计算结构；ViT 也可经特征重组、窗口注意力等方式用于密集任务，不能从原始单尺度输出推出“不适合检测”。

### Q: Shifted Window 的 cyclic shift + mask 是做什么的？

**A**: 窗口偏移后，特征图边缘产生不完整窗口。Cyclic shift 将其循环移位拼成完整窗口（便于批量 attention），同时用 attention mask 屏蔽"不应交互"的区域对——如原本不在同一窗口的两个 patch 被错误拼到一个窗口时，mask 阻止它们之间的 attention。这比给每个碎片窗口单独做 attention 更高效。


## 论文出处

> [1] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko. “End-to-End Object Detection with Transformers.” ECCV, 2020。arXiv:2005.12872v3，第 5–8 页损失与架构，第 13–14 页位置编码和查询可视化，第 20 页结构图。[原文](https://arxiv.org/pdf/2005.12872v3)
>
> [2] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo. “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” ICCV, 2021。归档版第 3–5 页层次化结构、窗口注意力和模型尺寸。[归档原文](<../01-Foundations/Backbone/[ICCV 2021] Swin-T - Swin Transformer - Hierarchical Vision Transformer using Shifted Windows/paper.pdf>)
>
> [3] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai. “Deformable DETR: Deformable Transformers for End-to-End Object Detection.” ICLR, 2021。arXiv:2010.04159v4，第 5–6 页公式 (2)–(3)，第 8 页 Table 1。[原文](https://arxiv.org/pdf/2010.04159v4)
>
> [4] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum. “DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.” ICLR, 2023。归档为 arXiv:2203.03605v4（2022），第 9–11 页混合查询与梯度路径，第 12 页 Table 1。[归档原文](<Object-Detection/[ICLR 2023] DINO - DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection/paper.pdf>)
>
> [5] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang. “Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.” ECCV, 2024。公开预印本第 3–5 页总体架构、语言引导查询与解码。[归档原文](<../04-Domain-Adaptation/VLM-Prior/[ECCV 2024] GDINO - Grounding DINO Marrying DINO with Grounded Pre-Training for Open-Set Object Detection/paper.pdf>)
