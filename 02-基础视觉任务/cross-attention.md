# 交叉注意力：两个序列之间的信息传递

交叉注意力（Cross-Attention）用一个序列产生查询，从另一个序列的键和值中读取信息。本文重点解释两个序列的角色、输出形状与计算量。缩放点积和多头的基础推导见 [注意力机制](../01-%E5%9F%BA%E7%A1%80%E7%90%86%E8%AE%BA/attention.md)。

## 一、哪个序列提供查询

设 $X$ 是需要更新的序列，$Y$ 是供它读取的参考序列。查询由 $X$ 产生，键和值由 $Y$ 产生：

$$
Q=XW_Q,\qquad K=YW_K,\qquad V=YW_V.
$$

自注意力与交叉注意力的区别在于表示的来源：

| 项目 | 自注意力 | 交叉注意力 |
|---|---|---|
| 查询来源 | $X$ | $X$ |
| 键和值来源 | $X$ | $Y$ |
| 信息流 | 汇总同一序列的信息 | 为 $X$ 的每个位置汇总 $Y$ 的信息 |
| 输出位置数 | $X$ 的长度 | $X$ 的长度 |

**[论文发现]** Transformer 的编码器—解码器注意力使用解码器状态产生查询，使用编码器输出产生键和值。[1，第 5 页] 在这一用法中，每个解码位置都从编码序列读取与当前预测有关的信息。

## 二、从形状理解计算过程

**[数学推导]** 设两个输入的形状为 $X\in\mathbb R^{N_q\times D_x}$、$Y\in\mathbb R^{N_k\times D_y}$。取投影矩阵：

$$
W_Q\in\mathbb R^{D_x\times d_k},\quad
W_K\in\mathbb R^{D_y\times d_k},\quad
W_V\in\mathbb R^{D_y\times d_v}.
$$

于是查询与键具有相同的特征维度，可以计算点积；键和值具有相同的位置数，可以逐项对应。使用缩放点积公式 [1，第 4 页公式 (1)]：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right),
\qquad O=AV.
$$

| 步骤 | 形状 | 含义 |
|---|---|---|
| 查询 $Q$ | $N_q\times d_k$ | 每个待更新位置的查询特征 |
| 键 $K$ | $N_k\times d_k$ | 每个参考位置的匹配特征 |
| 值 $V$ | $N_k\times d_v$ | 每个参考位置提供的内容 |
| 点积 $QK^T$ | $N_q\times N_k$ | 每个查询与每个参考位置的匹配分数 |
| 权重 $A$ | $N_q\times N_k$ | 每一行沿 $N_k$ 个位置归一化 |
| 输出 $O$ | $N_q\times d_v$ | 每个查询得到的加权汇总 |

例如，$X$ 有 2 个查询、$Y$ 有 3 个参考位置，则权重矩阵有 2 行 3 列。每一行决定一个查询怎样组合 3 个值向量，最终得到 2 个输出向量。输出位置数由查询决定。

缩放因子控制点积的尺度，其方差推导依赖独立性等假设，详见 [注意力机制第 2.3 节](../01-%E5%9F%BA%E7%A1%80%E7%90%86%E8%AE%BA/attention.md#attention-scaling)。

## 三、多头如何组合

**[论文发现]** 多头结构为各头设置不同投影，分别计算后拼接，再作输出投影。[1，第 4–5 页] 将同一结构用于两个输入序列，第 $i$ 个头为：

$$
\operatorname{head}_i
=\operatorname{Attention}(XW_Q^{(i)},YW_K^{(i)},YW_V^{(i)}),
$$

$$
O=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O.
$$

**[数学推导]** 若每头输出维度为 $d_v$，拼接结果为 $N_q\times(hd_v)$；取 $W_O\in\mathbb R^{hd_v\times D_{\text{out}}}$，最终输出为 $N_q\times D_{\text{out}}$。

多头允许模型学习不同的匹配方式，但结构本身没有规定“某一头负责位置，另一头负责语义”。这是需要从训练结果中观察的性质。

## 四、计算量取决于两个序列的长度

**[数学推导]** 单头需要计算 $N_qN_k$ 个长度为 $d_k$ 的点积，再为每个查询累加 $N_k$ 个 $d_v$ 维值向量：

$$
O(N_qN_kd_k)+O(N_qN_kd_v)
=O\bigl(N_qN_k(d_k+d_v)\bigr).
$$

若有 $h$ 个头且 $d_k=d_v=D/h$，注意力部分总计为 $O(N_qN_kD)$。仅当两序列长度都为 $N$ 时，才简化为 $O(N^2D)$。

若输入与输出宽度均为 $D$，线性投影还需要 $O((N_q+N_k)D^2)$；因为各序列位置都要乘以相应的投影矩阵。

当 $N_q\ll N_k$ 时，这部分注意力计算量小于在 $N_k$ 个位置上做全局自注意力的 $O(N_k^2D)$。比较时需明确基准序列长度与特征宽度。

## 论文出处

> [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. “Attention Is All You Need.” NeurIPS, 2017。缩放点积公式见公开版第 4 页公式 (1)，多头结构见第 4–5 页，编码器—解码器注意力见第 5 页第 3.2.3 节。[原文](https://arxiv.org/pdf/1706.03762)
