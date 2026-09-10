# 注意力机制：查询、键、值与多头计算

> 本文从加权汇总解释注意力，推导点积缩放与梯度，再介绍自注意力和多头计算。阅读需要了解矩阵乘法、方差与链式法则。

论文提出的结构用 **[论文发现]** 标注；由公式得到的性质用 **[数学推导]** 标注。两个序列之间的信息传递另见 [交叉注意力](../02-Vision-Tasks/cross-attention.md)。

---

## 一、注意力机制的基本直觉

注意力（Attention）根据当前查询，从一组参考信息中计算加权汇总：

- **查询（Query，$Q$）**：表示当前位置需要匹配的信息。
- **键（Key，$K$）**：供查询比较的特征；它是模型表示，不是人工类别标签。
- **值（Value，$V$）**：被加权汇总的内容，与键逐项对应。

先计算查询与键的匹配分数，再用 Softmax 将每个查询对应的分数转换为和为 1 的权重，最后对值加权求和。

**[论文发现]** 不加掩码的全局自注意力允许任意两个位置在一层内直接交互。[1，第 6 页 Table 1] 这里的“一层内交互”描述信息传递路径，不能理解为总计算量与序列长度无关；复杂度见第 4.3 节。

---

## 二、缩放点积注意力的计算与推导

### 2.1 原始公式

**[论文发现]** 缩放点积注意力（Scaled Dot-Product Attention）用缩放后的点积生成权重：[1，第 4 页公式 (1)]

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：

- $Q \in \mathbb{R}^{n \times d_k}$：$n$ 个查询，每个维度为 $d_k$
- $K \in \mathbb{R}^{m \times d_k}$：$m$ 个键，每个维度为 $d_k$
- $V \in \mathbb{R}^{m \times d_v}$：$m$ 个值，每个维度为 $d_v$
- $d_k$：键与查询的维度

Softmax 沿键的位置维度计算，即对每个查询分别归一化。

### 2.2 分步拆解

**[数学推导]** 根据矩阵乘法的定义，计算过程可分为三步。

首先计算每个查询与所有键的点积：

$$
S=QK^T\in\mathbb R^{n\times m}.
$$

随后按键与查询的维度缩放，并在每个查询对应的行上归一化：

$$
A=\operatorname{softmax}\left(\frac{S}{\sqrt{d_k}}\right),
\qquad
\sum_{j=1}^{m}A_{ij}=1.
$$

最后以注意力权重汇总值：

$$
O=AV\in\mathbb R^{n\times d_v}.
$$

因此，$QK^T$ 决定“读取哪里”，$AV$ 决定“读出什么”。

<a id="attention-scaling"></a>

### 2.3 为什么除以 √d_k？

**[数学推导]** 假设查询向量 $q$ 和键向量 $k$ 的各分量相互独立，均值为 0、方差为 1。对每一项：

$$
\mathbb E[q_i k_i]=\mathbb E[q_i]\mathbb E[k_i]=0,
\qquad
\mathbb E[(q_i k_i)^2]
=\mathbb E[q_i^2]\mathbb E[k_i^2]=1.
$$

因此 $\operatorname{Var}(q_i k_i)=1$。各项独立，求和后得到：

$$
\operatorname{Var}(q\cdot k)
=\sum_{i=1}^{d_k}\operatorname{Var}(q_i k_i)=d_k.
$$

点积的标准差为 $\sqrt{d_k}$。例如 $d_k=64$ 时，标准差为 8；除以 $\sqrt{d_k}$ 后：

$$
\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{d_k}}\right)
=\frac{\operatorname{Var}(q\cdot k)}{d_k}=1.
$$

这解释了缩放因子的选择。上述方差结论依赖独立性与单位方差假设，不能保证训练过程中方差始终等于 1。论文在第 4 页脚注 4 使用这一假设说明点积尺度随维度增长的问题。[1]

### 2.4 饱和如何影响梯度

**[数学推导]** Softmax 输出 $a_i=e^{z_i}/Z$，其中 $Z=\sum_j e^{z_j}$。对输入 $z_j$ 求导：

$$
\frac{\partial a_i}{\partial z_j}
=\frac{\delta_{ij}e^{z_i}Z-e^{z_i}e^{z_j}}{Z^2}
=a_i(\delta_{ij}-a_j).
$$

$\delta_{ij}$ 在 $i=j$ 时为 1，否则为 0。这给出雅可比矩阵（各输出对各输入的偏导数）：

$$\frac{\partial a_i}{\partial z_j} = a_i (\delta_{ij} - a_j)$$

当某个 $z_i$ 远大于其他 $z_j$ 时，$a_i \to 1$，$a_j \to 0$（j≠i）。此时：

- $\partial a_i / \partial z_i = a_i(1-a_i) \to 0$
- $\partial a_j / \partial z_i = -a_i \cdot a_j \to 0$

当权重接近独热分布时，Softmax 的局部导数趋近于零。在上游梯度有界的条件下，经由这一环节传回查询和键的梯度可能变小，影响权重的学习。

这不能推出整个模型停止学习。由 $O_{ia}=\sum_j A_{ij}V_{ja}$ 对 $V_{jb}$ 求导，得到 $\partial O_{ia}/\partial V_{jb}=A_{ij}\delta_{ab}$；即使某个注意力权重接近 1，值分支仍可以获得梯度。

---

## 三、自注意力：三组表示来自同一序列

自注意力（Self-Attention）中，查询、键和值都由同一个输入序列投影得到。[1，第 5 页]

给定输入 $X \in \mathbb{R}^{n \times d_{model}}$：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

其中 $W_Q,W_K\in\mathbb R^{d_{model}\times d_k}$，
$W_V\in\mathbb R^{d_{model}\times d_v}$。三组投影共享输入，但具有不同参数。

**[数学推导]** 如果直接令 $Q=K=V=X$，匹配分数就固定为输入向量的点积。引入三组可学习投影后，模型可以分别调整用于匹配的特征和用于汇总的内容。

---

## 四、多头注意力：分别计算后再合并

### 4.1 数学形式

**[论文发现]** 多头注意力（Multi-Head Attention）用多组投影分别计算注意力，再拼接输出。以下采用 $h$ 个头、每头 $d_k=d_v=d_{model}/h$ 的设置：[1，第 4–5 页]

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$$

$$\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

其中 $W_Q^i,W_K^i,W_V^i\in\mathbb R^{d_{model}\times d_k}$，$W_O\in\mathbb R^{h d_v\times d_{model}}$。

### 4.2 头部划分与合并

先用各头的投影矩阵得到查询、键和值，再在每个头中独立计算注意力。忽略批次维度，自注意力的序列长度为 $n$，第 $i$ 个头的形状为：

$$
Q_i\in\mathbb R^{n\times d_k},\quad
K_i\in\mathbb R^{n\times d_k},\quad
V_i\in\mathbb R^{n\times d_v}.
$$

划分和拼接只改变张量的分组方式；真正让不同头学习不同关系的是各自独立的投影矩阵。

### 4.3 计算复杂度

**[数学推导]** 长度为 $n$ 的单头自注意力先计算 $n^2$ 个点积，每个点积包含 $d_k$ 个乘积；汇总值时，每个输出分量累加 $n$ 项。因此：

| 操作 | 复杂度 |
|---|---|
| $QK^T$ | $O(n^2d_k)$ |
| Softmax | $O(n^2)$ |
| 加权汇总 $AV$ | $O(n^2d_v)$ |
| 单头注意力合计 | $O(n^2(d_k+d_v))$ |

取 $d_k=d_v=d_{model}/h$，将 $h$ 个头的工作量相加，注意力部分为 $O(n^2d_{model})$。输入投影与输出投影还需要 $O(nd_{model}^2)$：每个位置都进行宽度约为 $d_{model}$ 的线性变换。两部分应分开计算，不能把单头复杂度当作整个多头模块的复杂度。

---

## 五、继续阅读

- [交叉注意力](../02-Vision-Tasks/cross-attention.md)：查询与键值来自不同序列时，信息流和形状如何变化。
- [DETR 系列](../02-Vision-Tasks/DETR.md)：目标查询如何读取图像特征。
- [CLIP](../02-Vision-Tasks/clip.md)：文本与图像如何分别编码，再计算相似度。
- [扩散模型](../02-Vision-Tasks/diffusion-models.md)：条件信息如何进入去噪网络。

## 论文出处

> [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. “Attention Is All You Need.” NeurIPS, 2017。缩放点积及方差解释见公开版第 4 页公式 (1) 与脚注 4，多头结构见第 4–5 页，注意力用途见第 5 页，计算复杂度与路径长度见第 6 页 Table 1。[原文](https://arxiv.org/pdf/1706.03762)
