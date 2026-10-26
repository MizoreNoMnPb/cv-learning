# 图卷积与多模态类比推理：从谱域到两阶段消息传递

> 本文区分一般图卷积与 S²AR 预印本中的具体图算子。GCN（Graph Convolutional Network，图卷积网络）沿图边聚合节点表示；S²AR（Semantic-Spatial Analogical Reasoning，语义—空间类比推理）是下文所引 GRID 可见光—红外检测方法中的推理模块，不是领域自适应问题的通用解法。

一般图卷积的出处为 [1]；S²AR 结构来自尚未同行评审的 [2]，页码按归档 PDF 的物理页计数。数学性质与方法效果须分别理解。

---

## 一、图神经网络基础

### 1.1 什么是图（Graph）

一个图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：
- $\mathcal{V}$：节点集合，每个节点有一个特征向量
- $\mathcal{E}$：边集合，每条边连接两个节点
- 邻接矩阵 $A \in \mathbb{R}^{N \times N}$：$A_{ij} > 0$ 表示节点 i 和 j 之间有边

在多模态类比推理的一个典型建模中：
- **类别节点**：文本嵌入 $H^c \in \mathbb{R}^{K \times D}$
- **像素节点**：选定的 RGB/IR 位置的特征 $H^{v,p} \in \mathbb{R}^{M \times D}$
- **边**：语义关系边（文本↔像素）和空间关系边（像素↔像素）

RGB（Red, Green, Blue，红绿蓝）在此指可见光图像，IR（Infrared，红外）指红外图像；不同模态的节点是否空间对齐属于额外数据假设。

### 1.2 为什么需要图

**[分析判断]** 全局注意力需要计算全部节点对的关系，图结构可以显式限定哪些节点通信。但图也可以是稠密的，注意力也可以使用局部窗口或稀疏掩码；不能仅凭“图卷积”这一名称断言它比注意力快。

---

## 二、谱域图卷积 (Spectral Graph Convolution)

### 2.1 动机：从图像卷积到图卷积

**图像卷积**：在规则网格上，卷积核在空间邻域内操作
$$(f * g)(x) = \sum_{y \in \mathcal{N}(x)} f(y) \cdot g(x-y)$$

**图卷积**：一般图不要求规则坐标网格，邻域由边定义；若节点本身具有空间坐标，仍可利用空间邻域构图。

### 2.2 图拉普拉斯矩阵

$$L = D - A$$

其中 $D$ 是度矩阵（对角矩阵，$D_{ii} = \sum_j A_{ij}$）。

对无向、非负权重且非孤立的节点，**对称归一化拉普拉斯**定义为：
$$L_{norm} = I - D^{-1/2} A D^{-1/2}$$

**[数学推导]** 令 $z=D^{-1/2}x$，则 $x^TL_{norm}x=\frac12\sum_{ij}A_{ij}(z_i-z_j)^2\geq0$；又由 $(z_i-z_j)^2\leq2z_i^2+2z_j^2$ 得 $x^TL_{norm}x\leq2\|x\|^2$，所以特征值在 $[0,2]$。这里作为谱域基的是特征向量，特征值描述对应频率。孤立节点需要另行定义归一化规则；有向或负权图不能直接套用这些结论。

### 2.3 一阶图卷积 (Kipf & Welling, ICLR 2017)

**[论文发现]** Kipf–Welling GCN 的传播规则为：[1，第 2–3 页公式 (2)、(8)]

$$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$

其中：
- $\tilde{A} = A + I$：邻接矩阵 + 自连接
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$：$\tilde{A}$ 的度矩阵
- $W^{(l)}$：可学习的权重矩阵
- $\sigma$：激活函数（通常是 ReLU）
- $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$：对称归一化的邻接矩阵

**直观理解**：
1. 先形成 $S=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}\in\mathbb R^{N\times N}$，其元素为 $S_{ij}=\tilde A_{ij}/\sqrt{\tilde d_i\tilde d_j}$。
2. 再计算 $SH$，为各节点按归一化权重聚合邻居。不能把 $N\times D$ 的 $\tilde A H$ 再从右侧乘 $N\times N$ 的度矩阵。
3. $W$：对聚合后的特征做线性变换
4. $\sigma$：非线性激活

---

## 三、动态邻接与两阶段类比推理

### 3.1 从邻接矩阵计算到归一化

**[论文发现]** S²AR 预印本式 (11) 使用 $I - D^{-1/2} A D^{-1/2}$ 作为图拉普拉斯，而不是直接使用
$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$：

$$H_{new} = \text{ReLU}((I - D^{-1/2} A D^{-1/2}) H W)$$

该式见 [2，第 13 页公式 (11)]。**[数学推导]** 对满足第 2.2 节条件的归一化拉普拉斯特征向量 $u$，若 $Lu=\lambda u$，线性部分将该模式乘以 $\lambda$；零频模式被消去，较大特征值模式受更强作用。动态构图若不满足对称性等条件，则不能直接套用这一频谱解释。该算子与 GCN 的归一化邻接聚合不同，不能不加区分地解释成“同类平均”。


### 3.2 邻接矩阵的 Metric Learning 计算

邻接矩阵也可以由 Metric Learning（度量学习）根据输入动态计算，而不是预先固定：


**Metric Learning 中 $\Sigma$ 的作用**：
- 标准内积 $H_i \cdot H_j$ 假设所有特征维度同等重要
- $\Sigma$ 是一个对角矩阵，对角元素是可学习的维度权重
- $H_i\Sigma H_j^T$ 是可学习的双线性相似度，不是距离。$\Sigma$ 对称半正定时，对非负二次型 $(H_i-H_j)\Sigma(H_i-H_j)^T$ **开平方**才得到马氏半距离；$\Sigma$ 严格正定时，该平方根是通常的马氏距离。未开平方的量一般不满足三角不等式。
- AvgPool 聚合全局信息后生成 $\Sigma$，使其具有**上下文感知能力**

以上动态权重来自 [2，第 11–12 页公式 (7)–(8)]；原文的卷积映射包含 ReLU，因此其邻接权重非负。

### 3.3 S²AR 的 Induction（图推理）流程

将类别节点与单一模态的像素节点拼接为：

$$
H=[H^c;H^{v,p}],
$$

再以动态邻接 $A(H)$ 执行图卷积，使同类语义通过类别节点在不同像素之间传播。

### 3.4 Dynamic Gating（动态门控）跨模态空间融合

**[数学推导]** 设两个模态提供的空间关系矩阵分别为 $A_1,A_2$，单个分支可用门控系数融合：

$$
A_{\mathrm{fused}}
=\gamma A_1+(1-\gamma)A_2,
\qquad 0\leq\gamma\leq1.
$$

这是一支输出的简化记号；S²AR 分别为 RGB 与 IR 计算两组归一化权重，而不是两支共用一个门控标量。[2，第 12 页公式 (9)–(10)]

### 3.5 特征回注到密集特征图

图推理只更新被选中的 $M$ 个节点。更新结果按照原空间索引写回密集特征图，未被选择的位置保持原特征，
再与原密集特征按可学习门控融合。[2，第 13 页公式 (12)] 这里“少量选中节点”不等于“节点之间的邻接矩阵稀疏”。

---

## 四、两阶段推理与全图推理

### 全图推理（Full Graph Reasoning，FGR）
```
所有节点 (K 文本 + M RGB + M IR) → 一个大图 → 一次图卷积 → 输出
```
**[分析判断]** 需要关注：
- 节点总数为 K + 2M，邻接矩阵为 (K+2M)²
- 文本和视觉节点的异质性是否引入不利信息，需要通过受控实验判断。
- 不同边类型承载不同信息，合并建图会改变消息传递路径。

### 两阶段推理 (S²AR)
```
阶段1: (K 文本 + M RGB) → 语义推理 → 更新 RGB 像素
       (K 文本 + M IR)  → 语义推理 → 更新 IR 像素

阶段2: (M RGB 像素) → 空间推理(含 IR 空间信息) → 最终 RGB 像素
       (M IR 像素)  → 空间推理(含 RGB 空间信息) → 最终 IR 像素
```
**[论文发现]** 原文将语义与空间关系分阶段处理，并报告其消融结果。[2，第 13 页方法说明及第 21–22 页 Figure 5] **[分析判断]** 子图更小不保证总计算更低：两次 $(K+M)$ 节点图加两次 $M$ 节点图，其稠密关系项总数为 $2(K+M)^2+2M^2$；单次 $(K+2M)$ 图为 $(K+2M)^2$，前者还多出 $K^2$ 项。比较还必须考虑重复投影、实际边数、层数与融合操作。

### 设计动机的例子

**Intra-Modal** 阶段：远处的小 "car" 在 RGB 中不清晰 → 通过文本 "car" 节点 → 吸收近处大 "car" 的 RGB 特征 → 变清晰

**Inter-Modal** 阶段：RGB 中因光照看不清的 "car" → 利用 IR 中清晰的热信号（通过 $\gamma_{ir}$ 控制的空间边）→ 补充 RGB 特征

两个阶段**分工明确**：第一阶段解决"同类物体在模态内的语义传播"，第二阶段解决"利用另一模态的空间信息补充当前模态"。

---

## 五、常见理论问题

### Q: 为什么用图卷积而不是 Transformer 做类比推理？

**[分析判断]** 图结构便于把关系类型显式写入边，但 Transformer 也能通过关系嵌入或掩码区分边。S²AR 的主要规模控制来自先选择少量像素节点；其动态邻接仍按节点对计算，不能直接套用稀疏图复杂度。节点预算是方法设置，不应把一个示例值当成通用定值。

### Q: 图卷积中邻接矩阵 A 是可学习的还是固定的？

**A**: 在动态邻接方案中，$A$ 由当前输入计算，因此是 data-dependent；生成 $A$ 的度量参数则可训练。

$$A = \text{ReLU}(\text{Conv}(H) \cdot \Sigma(H) \cdot \text{Conv}(H)^T)$$

- A 依赖当前输入 H，所以是动态的
- 但 Conv 的权重和 Σ 的生成网络是可学习的
- 这比固定 A（如 k-NN）更灵活，能根据上下文自适应

### Q: 计算复杂度是多少？如何保证效率？

**[数学推导]** 对 $N=K+M$ 个、宽度为 $D$ 的节点，稠密双线性邻接与传播的主要乘加量为 $O(ND^2+N^2D)$，还要累计各子图与阶段。即使只取 $N=64,D=256$，$N^2D=1{,}048{,}576$，而 $ND^2=4{,}194{,}304$；只报前一项会漏掉更大的投影项。这是乘加项数的估计，不是实际耗时或完整 FLOPs 测量。

**[分析判断]** 节点数量还影响覆盖率：预算过小可能漏掉重要区域，预算增大会提高关系计算与存储成本。

## 论文出处

> [1] Thomas N. Kipf, Max Welling. “Semi-Supervised Classification with Graph Convolutional Networks.” ICLR, 2017。arXiv:1609.02907，第 2–3 页公式 (2)–(8) 给出谱域近似与重归一化传播。[原文](https://arxiv.org/pdf/1609.02907)
>
> [2] Ting Li, Maoyuan Zhao, Haiwen Xu, Hongkai Liu, Nianxing Li, Shuaifeng Li, Mao Ye. “Language-Guided Analogical Reasoning for Visible-Infrared Object Detection.” SSRN 预印本，2026（官方页面于 2026-07-20 发布），尚未同行评审。DOI: 10.2139/ssrn.7149771。PDF 第 11–13 页（正文印刷第 10–12 页）公式 (5)–(12) 给出选点、动态图、门控与两阶段传播。[归档原文](<../04-Domain-Adaptation/Fusion-and-Mixing/[SSRN preprint] S2AR - Language-Guided Analogical Reasoning for Visible-Infrared Object Detection/paper.pdf>)；[书目与版本](<../04-Domain-Adaptation/Fusion-and-Mixing/[SSRN preprint] S2AR - Language-Guided Analogical Reasoning for Visible-Infrared Object Detection/sources.md>)。
