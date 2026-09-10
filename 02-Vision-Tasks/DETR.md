# DETR 系目标检测器：从 DETR 到 Grounding DINO

> 本文比较集合预测、可变形注意力、去噪训练与文本条件检测。首次阅读按 DETR、[Deformable DETR](#deformable-detr)、[DINO](#dino)、[Grounding DINO](#grounding-dino)的顺序阅读；第 1.8 节的 Swin 背景可在需要了解骨干网络时补读。

前置知识：[R-CNN 系列](rcnn.md)、[注意力](../01-Foundations/attention.md)和 [交叉注意力](cross-attention.md)。

---

## 一、DETR (DEtection TRansformer)

> Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020

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
│   Conv2d(2048, 256, kernel_size=1)           │
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
│   固定正弦位置编码 (不是可学习的)               │
│   每个 (row, col) 位置生成唯一的 256 维编码     │
│   形状也是 (850, 256)，直接 element-wise 相加  │
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

**关键问题**：展平丢失了空间位置关系——Token 0 和 Token 1 在图像上是相邻的，但 Transformers 不知道。必须通过位置编码补偿。

#### Step 4: 位置编码——保留空间信息

DETR 使用**固定的 2D 正弦位置编码**（对行和列分别编码再拼接）：


**为什么用正弦位置编码而不是可学习的位置编码**：
- 正弦编码具有天然的**相对位置**归纳能力：$\text{PE}(pos+k)$ 可以表示为 $\text{PE}(pos)$ 的线性函数
- 模型可以学会关注相对位置关系（如"上方的物体"），而不需要记住每个绝对位置
- 对未见过的图像尺寸也能泛化（可学习编码只能处理固定尺寸）

#### 序列化后的尺寸流转

```
原始图像:    (B, 3, 800, 1066)
Backbone:    (B, 2048, 25, 34)     H=25, W=34
1×1 Conv:    (B, 256, 25, 34)      C: 2048→256
Flatten:     (B, 850, 256)         N=25×34=850 tokens
+ PosEnc:    (B, 850, 256)         element-wise 加
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
2. **位置编码解耦**：空间位置信息和语义特征分离——展平的特征只编码"看到什么"，位置编码只编码"在哪儿"——Transformer 可以灵活组合两者
3. **类 NLP 的接口**：序列化后直接复用标准 Transformer（不做任何修改），DETR 的 Encoder/Decoder 就是标准 Transformer 结构

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
Flatten + Positional Encoding → 序列 (HW, 256)
    │
    ▼
Transformer Encoder × 6 层 → 全局上下文增强的特征序列
    │
    ▼
100 个 Object Queries (可学习的，代表"寻找 100 个物体")
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
- 训练后，不同 query 会自然地**专精于不同位置和尺度**的物体
- 例如：query_0 偏向检测大物体在图像中央，query_45 偏向检测小物体在右下角

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

- 分类：标准 CrossEntropy（∅ 类也算）
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

> Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021 (Best Paper)

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
    │ → (C, H/4, W/4)    C=128
    │
    ▼
Patch Merging: 2×2 邻域拼接 → 4C → Linear → 2C
    │
    ▼
Stage 2: Swin Blocks × 2
    │ → (2C, H/8, W/16)   2C=256   ← 对应 FPN 的 P3
    │
    ▼
Patch Merging
    │
    ▼
Stage 3: Swin Blocks × 6
    │ → (4C, H/16, W/16)  4C=512   ← 对应 FPN 的 P4
    │
    ▼
Patch Merging
    │
    ▼
Stage 4: Swin Blocks × 2
    │ → (8C, H/32, W/32)  8C=1024  ← 对应 FPN 的 P5
```

这天然生成了**多尺度特征**（stages 2/3/4 的输出恰好对应 FPN 的 P3/P4/P5），无需额外构建图像金字塔。

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
| **Swin-L** | 192 | {2,2,18,2} | 197M | DINO 最强版本 |

#### 在 DETR 系中的使用

DINO 和 Grounding DINO 用 Swin 替换 ResNet backbone 的方式：


**Swin 天然输出多尺度特征，不需要额外构造图像金字塔**。DINO/Grounding DINO 仍在不同 stage
输出后使用轻量投影，将各尺度通道统一到 256。

#### ViT vs Swin 对比

| | ViT (如 CLIP ViT-B/16) | Swin Transformer |
|---|---|---|
| **结构** | 单尺度，全程 14×14 patches | 层次化，4 个 stage 逐步降采样 |
| **Attention** | 全局 self-attention，$O(N^2)$ | 窗口内 attention + Shifted window，$O(N \cdot M^2)$ |
| **多尺度输出** | 无（需额外模块） | 天然多尺度（4 个 stage） |
| **计算效率** | 高分辨率下不可承受 | 分辨率增加时计算量线性增长 |
| **适用任务** | 分类、CLIP 对比学习 | **检测、分割（密集预测任务）** |
| **网络深度** | uniform（全层同样） | 递增（浅层少 block、深层多 block） |

---

<a id="deformable-detr"></a>

## 二、Deformable DETR

> Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021

### 2.1 解决 DETR 的两个核心问题

1. **慢收敛**：DETR 的 cross-attention 需要从全局搜索 → Deformable DETR 用多尺度可变形 attention，聚焦于参考点附近
2. **小物体差**：DETR 只用单尺度特征 → Deformable DETR 用多尺度特征图

### 2.2 Deformable Attention

标准 Attention：
$$\text{Attn}(z_q, x) = \sum_{m=1}^{HW} \text{softmax}(q \cdot k_m) \cdot v_m$$

计算量 $O(HW)$，且与特征图大小平方增长。

**Deformable Attention**：
$$\text{DeformAttn}(z_q, \hat{p}_q, x) = \sum_{h=1}^{H} W_h \left[ \sum_{k=1}^{K} A_{hqk} \cdot W_h' x(\hat{p}_q + \Delta p_{hqk}) \right]$$

- 每个 query 只在 K 个**可学习采样点**上做 attention（K=4, 远小于 HW）
- 采样点由 reference point $\hat{p}_q$ + 可学习偏移 $\Delta p$ 确定
- 多尺度：在多尺度特征图上分别采样


**为什么 Deformable Attention 更快收敛**：
- 标准 cross-attention 需要从所有位置中学习关注正确区域（500 epochs）
- Deformable attention 从 reference point 附近开始搜索（局部先验），大大缩小搜索空间（50 epochs）

### 2.3 多尺度特征


### 2.4 与 DETR 的对比

| | DETR | Deformable DETR |
|---|---|---|
| **Convergence** | 500 epochs | 50 epochs (10× 更快) |
| **小物体 AP** | ~10 | ~25 |
| **Encoder Self-Attn** | 全局 (HW × HW) | Deformable (HW × K) |
| **Decoder Cross-Attn** | 全局 (N × HW) | 多尺度 Deformable (N × K × L) |
| **特征** | 单尺度 | 多尺度 |

---

<a id="dino"></a>

## 三、DINO (DETR with Improved DeNoising anchOr boxes)

> Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection", ICLR 2023

### 3.1 核心改进

DINO 在 Deformable DETR 基础上做了三处关键改进，进一步加速收敛和提升性能。

#### (1) Contrastive DeNoising (CDN)

训练时，在 object queries 中添加**噪声 GT**（略微偏移真实框）：


**为什么 CDN 有效**：
- 标准 DETR 中，decoder 需要从头学习"从随机 query 到物体"的映射
- CDN 提供了一个**显式的去噪任务**：给定近似位置 + 噪声，恢复精确位置
- Positive 和 Negative 组的对比增强了模型对定位精度的敏感度

#### (2) Mixed Query Selection

DETR 的 decoder queries 是纯可学习参数。Deformable DETR 用 encoder 输出的 top-K 特征初始化 queries 的 content 部分。DINO 进一步改进：


#### (3) Look Forward Twice

标准 DETR 中，梯度只从最后一层反传。DINO 让每层 decoder 的梯度独立反传，且 box 预测的梯度**不反传到下一层**：


新的修正方法中，box 预测的梯度不截断，但用一个辅助 loss 来稳定。

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
| DINO | R50 | 12 | 49.0 |
| DINO | Swin-L | 36 | 58.5 |

DINO 仅 12 epochs 就超越了 Deformable DETR 50 epochs 的结果。

---

<a id="grounding-dino"></a>

## 四、Grounding DINO

> Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", ECCV 2024

### 4.1 核心目标

将**闭集检测器 DINO** 扩展为**开放词汇检测器**——输入文本描述，检测任意类别的物体。

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


**与传统 Query Selection 的区别**：
- DINO：根据 encoder 输出的 objectness score 选择（闭集）
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

**A**: $O(N \cdot K)$，其中 K 是采样点数（通常 4）。对于 N=10K 的特征图位置，从 $O(10^8)$ 降到 $O(4 \times 10^4)$，降低了 2500 倍。

### Q: Grounding DINO 和 GLIP 的主要区别？

**A**: 
- **GLIP**：基于 DyHead + BERT，将检测重新表述为 phrase grounding（区域-短语匹配），效果强但架构较重
- **Grounding DINO**：基于 DINO 架构，将开放词汇能力融入 DETR 系列，在 COCO 上 AP 相当但架构更优雅
- Grounding DINO 的 Text Cross-Attention 在 decoder 每一层都做，GLIP 偏向在特征层做 fusion

### Q: Swin Transformer 的核心创新是什么？为什么比 ViT 更适合做检测 backbone？

**A**: 两个核心创新：
1. **层次化结构**：4 个 stage 逐步降采样，天然生成多尺度特征（P3/P4/P5），无需额外 FPN
2. **Shifted Window Attention**：窗口内做 self-attention（高效），交替偏移窗口实现跨窗口信息交互（等价全局感受野）

检测需要多尺度特征和高分辨率 → ViT 的单尺度 + $O(H^2W^2)$ 复杂度不适合，Swin 的层次化 + 窗口 attention 完美适配。

### Q: Shifted Window 的 cyclic shift + mask 是做什么的？

**A**: 窗口偏移后，特征图边缘产生不完整窗口。Cyclic shift 将其循环移位拼成完整窗口（便于批量 attention），同时用 attention mask 屏蔽"不应交互"的区域对——如原本不在同一窗口的两个 patch 被错误拼到一个窗口时，mask 阻止它们之间的 attention。这比给每个碎片窗口单独做 attention 更高效。
