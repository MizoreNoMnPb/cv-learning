# PatchCore 精读：从样本 Memory Bank 到参数化记忆

> **阅读草稿**：原文解读与应用拓展仍需核验；拓展设想不作为原论文结论。

> 原论文：*Towards Total Recall in Industrial Anomaly Detection*，CVPR 2022
> 归档原文：[PDF](paper.pdf)
> 官方代码：[amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)

本文用 **[论文发现]** 标注所引论文直接报告的内容，用 **[数学推导]** 标注由明确前提推出的性质，用 **[业界共识]** 标注通用概念，用 **[分析判断]** 标注方法解释与应用设想。后续工作与应用设想会在相应段落中说明，不归因于原论文。

## 阅读建议

首次阅读先看第 1–5 节的 [任务与计算流程](#paper-main)，再阅读第 6–8 节的原文变体、实验和实现比较。第 9–11 节的 [方法比较与应用讨论](#applications)可在理解基本方法后查阅。

---

## 0. 概念对照：记忆库、覆盖与参数

### 0.1 PatchCore 的 Memory Bank 不是模型参数

> Memory Bank（记忆库）：保存正常训练图像的局部特征，供测试特征执行最近邻检索的外部数据结构。

**[论文发现]** PatchCore 冻结 ImageNet 预训练特征提取器，把正常图像的 patch 特征加入集合
$\mathcal M$，再用 coreset 压缩。整个过程不需要在目标数据集上反向传播。[1, pp.3-5,12]

更严格地说：

| 对象                                         | 是否由训练样本产生 | 是否进入优化器 | 是否有梯度 | 本质               |
| -------------------------------------------- | -----------------: | -------------: | ---------: | ------------------ |
| PatchCore 的$\mathcal M$                   |                 是 |             否 |         否 | 样本特征集合       |
| PaDiM 的$\mu,\Sigma$                       |                 是 |             否 |         否 | 统计估计量         |
| K-means / FINCH 原型                         |                 是 |             否 |         否 | 聚类摘要           |
| $P\in\mathbb R^{K\times d}$ 作为可训练张量 |       可随机初始化 |             是 |         是 | 参数化 Memory Bank |

因此，“Memory Bank”只描述功能，不自动意味着它是可训练参数。

### 0.2 “Total Recall”不等于保留全部训练特征

**[论文发现]** 作者的目标是尽量保留测试时可用的正常上下文，但核心方法又明确使用 coreset
把原始 bank 压缩到 25%、10% 甚至 1%。[1, pp.2,4-7]

准确表述是：

> PatchCore 用大量局部正常特征扩大 nominal context（正常上下文），再用覆盖导向的 coreset
> 删除冗余；“Total Recall”表达的是尽量覆盖正常特征空间，而不是逐项保存全部特征。

### 0.3 参数化 Memory 早于 PatchCore，但来自另一条路线

**[论文发现]** MemAE 在 2019 年已经把固定数量的 memory slots 与编码器、解码器共同训练；
MNAD、DAAD 又分别发展了显式读写和多尺度 block-wise memory。[3, pp.2-5; 4, pp.4-6;
5, pp.2,4-6]

所以历史不是“PatchCore 首先提出 bank，后来才有人参数化”，而是两条路线在 2022 年交汇：

- 检索路线：保存预训练特征，靠距离检测，如 SPADE、PaDiM、PatchCore。
- 重建路线：用可读写或可训练槽位限制解码器，如 MemAE、MNAD、DAAD。

PatchCore 第 7 页的 learned proxy 实验，才是它在自身检索框架内直接尝试参数化 bank 的证据。[1, p.7]

---

<a id="paper-main"></a>

## 1. 论文解决什么问题

### 1.1 问题设定

> OCC（One-Class Classification，单类分类）：训练阶段只观察一个类别，这里是正常样本，测试时识别偏离该类别的样本。
> OOD（Out-of-Distribution，分布外）：测试样本位于训练分布支撑集之外。

**[论文发现]** PatchCore 处理 cold-start industrial anomaly detection（冷启动工业异常检测）：

$$
\mathcal X_N=\{x_i:y_i=0\}
$$

只含正常训练图像；测试集 $\mathcal X_T$ 同时可能含正常与异常图像。方法需要输出：

- 图像级异常分数；
- 像素级异常热图。

问题定义和符号来自原论文第 1、3 页。[1, pp.1,3]

### 1.2 作者指出的三组矛盾

**[论文发现]** 预训练特征方法有效，但直接使用存在以下矛盾：[1, pp.1-4]

1. 深层特征语义强、感受野大，却更偏向 ImageNet 类别且空间分辨率低。
2. 浅层特征分辨率高，却缺少判断结构异常所需的上下文。
3. 保存更多正常特征可以扩大正常分布覆盖，但 bank、检索时间和显存会随数据量增长。

PatchCore 的三个对应设计是：

1. 使用中层特征，默认 WideResNet50 的 block 2 和 block 3；
2. 在特征图上聚合局部邻域，默认 neighbourhood size $p=3$；
3. 使用 greedy coreset 保留特征空间覆盖并减少冗余。

---

## 2. 完整流程先看一遍

> 原论文 Figure 2（第 4 页）：正常图像被拆成局部 patch 特征，组成并压缩 Memory Bank；测试时以 patch 到 bank 的最近邻距离生成图像分数和定位图。[1, p.4]

### 2.1 构建 Memory Bank

1. 冻结 ImageNet 预训练 backbone。
2. 从每张正常图像提取中层特征图。
3. 对每个空间位置聚合周围 $p\times p$ 的特征。
4. 把不同层级缩放到同一网格并融合。
5. 收集全部正常 patch embedding，得到原始 $\mathcal M$。
6. 用 greedy coreset 得到较小的 $\mathcal M_C$。

### 2.2 测试

1. 用同一个冻结网络提取测试 patch embedding。
2. 对每个测试 patch，在 $\mathcal M_C$ 中搜索最近邻。
3. 最近邻距离作为该 patch 的异常分数。
4. 全图最大的 patch 分数作为图像级异常分数。
5. 将 patch 分数还原为空间网格，双线性插值到输入大小，再做高斯平滑。

**[分析判断]** PatchCore 没有显式“正常/异常分类器”。它实现的是：

$$
\text{异常程度}
=\text{测试局部特征到已知正常局部特征集合的距离}。
$$

---

## 3. 局部 patch 特征：Equation (1)-(4)

### 3.1 特征图与局部邻域

设预训练网络为 $\phi$，图像 $x_i$ 在层级 $j$ 的特征图为：

$$
\phi_{i,j}=\phi_j(x_i)\in\mathbb R^{c\times h\times w}.
$$

位置 $(h,w)$ 的通道向量记为 $\phi_{i,j}(h,w)\in\mathbb R^c$。

**[论文发现]** 对奇数 patch size $p$，局部邻域定义为：[1, p.3, Eq.(1)]

$$
\mathcal N_p^{(h,w)}
=\left\{
(a,b)\ \middle|\
a\in\left[h-\left\lfloor\frac p2\right\rfloor,
h+\left\lfloor\frac p2\right\rfloor\right],
b\in\left[w-\left\lfloor\frac p2\right\rfloor,
w+\left\lfloor\frac p2\right\rfloor\right]
\right\}.
$$

原文公式的排版使用区间写法；这里把奇数 $p$ 的离散边界明确写成 floor 形式，含义不变。

当 $p=3$ 时，一个位置不只表示单个特征向量，而是观察一个 $3\times3$ 特征邻域。

### 3.2 邻域聚合

**[论文发现]** 局部感知特征为：[1, p.3, Eq.(2)]

$$
\phi_{i,j}\!\left(\mathcal N_p^{(h,w)}\right)
=f_{\mathrm{agg}}
\left(
\left\{\phi_{i,j}(a,b):(a,b)\in\mathcal N_p^{(h,w)}\right\}
\right).
$$

PatchCore 使用 adaptive average pooling（自适应平均池化）作为 $f_{\mathrm{agg}}$。[1, p.3]

这一步不是在原图上切出 $3\times3$ 像素，而是在 CNN 特征图上聚合相邻位置。每个特征位置已经对应原图中较大的感受野，因此局部聚合进一步加入周围上下文。

### 3.3 stride 只负责取点，不负责扩大感受野

**[论文发现]** 层级 $j$ 的 patch 集合为：[1, p.3, Eq.(3)]

$$
\mathcal P_{s,p}(\phi_{i,j})
=\left\{
\phi_{i,j}\!\left(\mathcal N_p^{(h,w)}\right)
\ \middle|\
h\bmod s=0,\ w\bmod s=0
\right\}.
$$

- $p$：每个 patch embedding 聚合多大的局部邻域。
- $s$：在特征网格上每隔多少位置取一个 embedding。

**[论文发现]** 默认 $s=1$；把 $s$ 提高到 2、3 会降低空间覆盖和定位分辨率，图像 AUROC
分别降到 97.6%、96.8%。[1, pp.3,7]

因此，用 stride 缩小 bank 和用 coreset 缩小 bank 不等价：

- stride 先规则丢弃空间位置，某些局部模式从未进入候选集；
- coreset 先看到全部候选，再按特征空间覆盖选择代表点。

### 3.4 多层融合与原始 bank

**[论文发现]** 对高层特征的 patch 网格做双线性缩放，使其空间位置数与低层相同，再融合对应位置的特征。默认使用相邻的两个中层层级 2 和 3。[1, pp.3-4,7]

所有正常图像的 patch 集合并起来得到：[1, p.4, Eq.(4)]

$$
\mathcal M
=\bigcup_{x_i\in\mathcal X_N}
\mathcal P_{s,p}\!\left(\phi_j(x_i)\right).
$$

这里的 $\phi_j$ 已包含论文所述的多层融合步骤。

### 3.5 一个具体尺寸例子

**[分析判断]（公开实现核验）** 官方推荐配置为 224 输入、WideResNet50、`layer2 + layer3`、`patchsize=3`、
最终 embedding 维度 1024。`layer2` 的高分辨率网格为 $28\times28$，所以每张图产生：

$$
28\times28=784
$$

个 patch embedding。推荐命令和具体聚合实现见官方仓库 README 与
`src/patchcore/patchcore.py`（commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`）。[13]

**[数学推导]** 若某类别有 250 张正常图、每个 embedding 为 1024 维 `float32`，未压缩 bank
仅特征矩阵约占：

$$
250\times784\times1024\times4
=802{,}816{,}000\ \text{bytes}
\approx765.6\ \text{MiB}.
$$

10% coreset 理论上把这部分降到约 76.6 MiB，尚未计入检索索引和临时张量。这说明 coreset
首先是工程必需品，不只是一个精度技巧。

---

## 4. Coreset：Equation (5)

> Coreset（核心集）：从大集合中选择较小子集，使原问题在该子集上的解尽量逼近完整集合上的解。
> Minimax facility location（最小最大设施选址）：让每个原始点到最近代表点的最远距离尽可能小。

### 4.1 目标函数

**[论文发现]** PatchCore 选择：[1, p.4, Eq.(5)]

$$
\mathcal M_C^*
=\arg\min_{\mathcal M_C\subset\mathcal M}
\max_{m\in\mathcal M}
\min_{n\in\mathcal M_C}
\lVert m-n\rVert_2.
$$

从内向外读：

1. 对每个原始特征 $m$，找到 coreset 中最近的 $n$；
2. 找到所有原始特征中“被代表得最差”的那个；
3. 选择 $\mathcal M_C$，让这个最坏距离尽可能小。

**[数学推导]** 定义覆盖半径：

$$
r(\mathcal M_C)
=\max_{m\in\mathcal M}\min_{n\in\mathcal M_C}\lVert m-n\rVert_2.
$$

若 $r$ 很小，则每个被删除的正常特征都至少有一个保留特征位于半径 $r$ 内。它控制的是
**最坏覆盖误差**，正好对应 PatchCore 不希望漏掉稀有正常模式的目标。

### 4.2 贪心 farthest-first

Equation (5) 的精确求解是 NP-hard。**[论文发现]** 作者使用贪心近似：[1, pp.4-5, Algorithm 1]

1. 初始化少量 anchor；
2. 计算每个候选点到当前 coreset 的最近距离；
3. 选择这个最近距离最大的点；
4. 重复直到达到目标大小。

每次选择的都是当前覆盖最差的正常特征，因此常称 farthest-first traversal（最远优先遍历）或
k-center greedy（k 中心贪心）。

### 4.3 随机投影为什么可以加速

> JL（Johnson-Lindenstrauss）定理：一组有限高维点可随机投影到较低维，并以高概率近似保持两两距离。

**[论文发现]** coreset 选择前使用随机线性投影 $\psi:\mathbb R^d\to\mathbb R^{d'}$，
$d'<d$，只在低维空间中决定索引；最终保存的仍是相应原始 embedding。[1, p.4]

**[分析判断]（公开实现核验）** 官方 sampler 用一个未训练、无 bias 的 `Linear` 将特征随机投影到 128 维；
README 推荐 approximate greedy coreset，并以 10 个随机起点近似初始化覆盖距离。[13]

### 4.4 为什么随机采样明显更差

**[数学推导]** 随机采样优化的是平均意义上的抽样代表性，不直接约束最坏覆盖半径。若正常分布
包含一个占比很小的模式，其被随机遗漏的概率会很高；farthest-first 则会因为该模式距离已有中心远，
主动选择它。

这正是“主流正常模式”和“稀有但合格的正常模式”并存时，工业误报问题的核心。

---

## 5. 最近邻异常分数：Equation (6)-(7)

### 5.1 patch 分数与图像分数

设测试图像的 patch 集合为 $\mathcal P(x_{\mathrm{test}})$。对每个测试 patch $q$，
最近正常特征距离为：

$$
d(q,\mathcal M_C)
=\min_{m\in\mathcal M_C}\lVert q-m\rVert_2.
$$

**[论文发现]** 先选出最近邻距离最大的测试 patch $q^*$ 及其最近 bank 特征 $m^*$：[1, p.5, Eq.(6)]

$$
(q^*,m^*)
=\arg\max_{q\in\mathcal P(x_{\mathrm{test}})}
\arg\min_{m\in\mathcal M_C}
\lVert q-m\rVert_2,
$$

$$
s^*=\lVert q^*-m^*\rVert_2.
$$

第一行是原文的嵌套 `arg max / arg min` 记法；更无歧义的等价标量式是：

$$
s^*
=\max_{q\in\mathcal P(x_{\mathrm{test}})}
\min_{m\in\mathcal M_C}\lVert q-m\rVert_2.
$$

**[分析判断]** 图像只要存在一个明显异常 patch，就应被判为异常。这是 PatchCore 的关键归纳偏置，
也使它对很小的局部缺陷敏感。[1, pp.2,5]

### 5.2 邻域重加权

**[论文发现]** 作者没有直接把 $s^*$ 作为最终图像分数，而是检查 $m^*$ 周围的 $b$ 个 bank
邻居 $\mathcal N_b(m^*)$：[1, p.5, Eq.(7)]

$$
s
=\left(
1-
\frac{\exp\left(\lVert q^*-m^*\rVert_2\right)}
{\displaystyle\sum_{m\in\mathcal N_b(m^*)}
\exp\left(\lVert q^*-m\rVert_2\right)}
\right)s^*.
$$

令括号内权重为 $w$。因为分母包含分子对应项且所有项为正：

$$
0\leq w<1,
\qquad
0\leq s<s^*.
$$

所以原文所说的“increase the anomaly score”不是让 $s$ 超过 $s^*$，而是：与位于稠密正常簇中的
匹配相比，若 $m^*$ 本身是孤立、稀有的正常点，则其他支持点到 $q^*$ 更远，分母增大，$w$ 更接近 1，
最终分数被**更少地压低**。[1, p.5]

### 5.3 定位图

**[论文发现]** 每个测试 patch 的最近邻距离按原空间位置排回网格，然后：[1, p.5]

1. 双线性插值到输入分辨率；
2. 使用 $\sigma=4$ 的 Gaussian smoothing（高斯平滑）；
3. 论文说明该参数没有专门优化。

图像分数取最异常 patch，像素图保留所有 patch 分数。二者共享一次检索，但聚合方式不同。

---

## 6. 论文已经做过参数化 Bank：Equation (8)

这是理解后续工作的关键，不能只把 PatchCore 看成“作者从未考虑可学习 Memory”。

### 6.1 learned basis proxies

**[论文发现]** 作者采样一组可学习 proxy：

$$
\mathcal P=\{p_k\}_{k=1}^{K},
\qquad
p_k\in\mathbb R^d,
\qquad
K=p_{\mathrm{target}}|\mathcal M|.
$$

对每个原始 bank 特征 $m_i$，Equation (8) 定义：[1, p.7, Eq.(8)]

$$
\mathcal L_{\mathrm{rec}}(m_i)
=\left\lVert
m_i-
\sum_{p_k\in\mathcal P}
\frac{
\exp\left(\lVert m_i-p_k\rVert_2\right)
}{
\displaystyle\sum_{p_j\in\mathcal P}
\exp\left(\lVert m_i-p_j\rVert_2\right)
}
p_k
\right\rVert_2^2.
$$

这是一组固定数量、通过可微重构目标学习的 basis proxies。按“是否进入优化目标”定义，它就是
PatchCore 自己的参数化 Memory Bank 尝试。

### 6.2 原式的正距离指数非常可疑

**[原论文原式]** 分子和分母明确写为正的 $\exp(\lVert m_i-p_k\rVert_2)$，没有负号，不是常见的负距离 softmax。[1, p.7]

**[数学推导]** 若两个 proxy 与 $m_i$ 的距离分别为 1 和 3，则较远 proxy 的权重为：

$$
\frac{e^3}{e^1+e^3}\approx0.881,
$$

较近 proxy 的权重反而只有：

$$
\frac{e^1}{e^1+e^3}\approx0.119.
$$

这与相似度注意力通常使用 $\exp(-d)$ 的方向相反。现有证据不能判断它是印刷错误还是作者确实按此训练；
论文没有公开这部分实现。[1, p.7; 13]

### 6.3 即使补上负号，目标仍与最近邻检测错位

这是比符号更根本的问题。

Equation (8) 优化的是“多个 proxy 的加权和能否重构 $m_i$”：

$$
m_i\approx\sum_k a_{ik}p_k.
$$

PatchCore 推理关心的却是“是否至少有一个独立 proxy 接近 $m_i$”：

$$
\min_k\lVert m_i-p_k\rVert_2.
$$

**[数学推导]** 在一维空间取 $p_1=-1$、$p_2=1$、$m=0$。若权重各为 $1/2$：

$$
\frac12p_1+\frac12p_2=0=m,
$$

重构误差为 0；但最近邻距离为：

$$
\min(|0-(-1)|,|0-1|)=1.
$$

所以低重构误差不保证低最近邻覆盖误差。learned proxies 可以用多个远点的组合重构样本，
却仍不适合作为 PatchCore 的单点最近邻参照。

### 6.4 实验结论

> 原论文 Figure 5（第 7 页）：橙色为 greedy coreset，绿色为随机采样，灰色为 learned proxies；learned proxies 在图像级和像素级 AUROC 上都明显更差。[1, p.7]

**[论文发现]** 作者还报告：[1, p.7]

- 不压缩 bank 时，测试中实际成为最近邻的 bank 样本不足 30%；
- 压到 1% 后，约 95% 的 coreset 样本会在测试中被使用；
- 10%-50% 的某些压缩区间甚至能略优于不压缩 bank；
- coreset 明显优于随机采样和 learned proxies。

**[分析判断]** Figure 5 不能推出“所有参数化 Memory Bank 都无效”。它只否定了这一组具体组合：

- 冻结的 PatchCore 特征空间；
- Equation (8) 的 basis reconstruction 目标；
- 相同 bank 预算下的 learned proxy；
- 原文未充分披露的训练实现。

它真正说明的是：**对最近邻异常检测，覆盖目标比可重构目标更匹配。**

---

## 7. 实验结果应该怎样读

### 7.1 默认 PatchCore，不要和 ensemble 混写

**[论文发现]** MVTec AD 默认 224 输入、WideResNet50、中层 2+3 的结果为：[1, pp.6-7,
Tables 1-3]

| Bank 保留比例 | Image AUROC | Pixel AUROC |  PRO |
| ------------: | ----------: | ----------: | ---: |
|           25% |        99.1 |        98.1 | 93.4 |
|           10% |        99.0 |        98.1 | 93.5 |
|            1% |        99.0 |        98.0 | 93.1 |

> PRO（Per-Region Overlap，逐区域重叠）：按缺陷连通区域衡量定位覆盖，减少大缺陷像素数对指标的主导。

### 7.2 99.6% 是高分辨率多 backbone ensemble

**[论文发现]** 99.6 image AUROC、98.2 pixel AUROC、94.9 PRO 来自 320 输入、
DenseNet201 + ResNeXt101 + WideResNet101 的 ensemble，不是默认单模型。[1, p.7, Table 4]

另一个 WideResNet101、280 输入的 1% bank 配置为 99.4 / 98.2 / 94.4。[1, p.7, Table 4]

### 7.3 速度结果的条件

**[论文发现]** Table 5 报告的 MVTec 单图平均推理时间包括 backbone 前向：[1, p.7]

| 配置           | 时间/图 | 对应分数（Image, Pixel, PRO） |
| -------------- | ------: | ----------------------------- |
| PatchCore-100% |  0.60 s | 99.1, 98.0, 93.3              |
| PatchCore-10%  |  0.22 s | 99.0, 98.1, 93.5              |
| PatchCore-1%   |  0.17 s | 99.0, 98.0, 93.1              |

补充材料第 12 页将硬件逐字写为 `Nvidia Tesla V4 GPUs`，但没有给出更具体的型号信息；本文按原文
照录，不能擅自改成 V100，也不能据此建立可复现的硬件基线。表中时间还依赖当时的 PyTorch、FAISS
和实现版本，不是现代 GPU 或当前代码的固定延迟。[1, pp.7,12]

### 7.4 论文实际使用的数据集

**[论文发现]** 论文实验只包含：[1, pp.5,8]

- MVTec AD：主要工业基准；
- MTD（Magnetic Tile Defects，磁瓦缺陷数据集）：图像 AUROC 97.9；
- mSTC（mini ShanghaiTech Campus，迷你上海科技大学校园数据集）：像素 AUROC 91.8。

原论文没有 VisA 或 BTAD 实验。将后续复现结果写成 PatchCore 原论文结果是错误的。

### 7.5 原论文明确留下的限制

**[论文发现]** PatchCore 的有效性受预训练特征能否迁移到工业域限制；作者把“结合特征适配”留作未来工作。[1, p.8]

**[分析判断]** 工业部署还要额外验证：

- 正常训练集污染会把缺陷写入 bank，造成假阴性；
- 新光照、相机、批次会让所有正常 patch 到 bank 的距离整体抬升；
- `max` 聚合对单个噪声 patch 敏感；
- 局部最近邻擅长纹理和局部结构缺陷，但不天然理解“零件数量不对、位置关系不对”等逻辑异常；
- bank 大小、索引后端和特征维度直接影响 RAM、显存和延迟。

---

## 8. 论文与官方代码并不完全相同

### 8.1 官方代码版本

**[分析判断]（公开实现核验）** 官方仓库 Apache-2.0。下文对照所用代码版本：`fcaa92f124fb1ad74a7acf56726decd4b27cbcad`。[13]

### 8.2 关键对应与差异

| 项目       | 论文                     | 官方代码                                                         |
| ---------- | ------------------------ | ---------------------------------------------------------------- |
| backbone   | 冻结预训练网络           | 特征提取全程`torch.no_grad()`                                  |
| bank 类型  | 正常 patch 特征集合      | detach 后转 NumPy，再写入 FAISS 索引                             |
| 推荐层级   | WideResNet50 block 2+3   | README 命令为`layer2 + layer3`                                 |
| 局部邻域   | 默认$p=3$              | `Unfold(kernel_size=3, padding=1)`                             |
| coreset    | 随机投影 + greedy        | 未训练`Linear` 投影到 128 维；README 用 approximate greedy 10% |
| 最近邻数   | Equation (6) 为 1-NN     | README 推荐`--anomaly_scorer_num_nn 1`                         |
| CLI 默认   | 论文没有这个软件默认值   | `run_patchcore.py` 的 CLI 默认却是 5-NN                        |
| 图像分数   | Eq.(7) 邻域重加权        | `_predict()` 对 patch kNN 均值取全图最大值，未实现 Eq.(7)      |
| 定位后处理 | 双线性插值，$\sigma=4$ | `RescaleSegmentor` 与论文一致                                  |

最重要的 paper/code mismatch 是 **官方仓库没有 Equation (7) 的图像级重加权**。当 README 使用
1-NN 时，代码实际执行：

$$
s_{\mathrm{code}}
=\max_q\min_{m\in\mathcal M_C}\lVert q-m\rVert_2,
$$

即 Equation (6) 的 $s^*$。[13]

### 8.3 anomalib 的实现选择不同

**[分析判断]（公开实现核验）** `open-edge-platform/anomalib` 维护 PatchCore，默认：

- `wide_resnet50_2`；
- `layer2 + layer3`；
- 3×3 average pooling；
- `num_neighbors=9`；
- 显式实现 Equation (7) 的邻域权重；
- bank 注册为 buffer，不是 `Parameter`，并明确说明无需反向传播。[14]

因此，复现实验必须报告使用的是论文公式、Amazon 官方代码还是 anomalib；“都叫 PatchCore”不代表
图像级分数完全相同。

### 8.4 最小复现配置

建议先复现论文主设置，而不是先改参数化 bank：

```text
backbone: wide_resnet50_2, ImageNet pretrained, frozen
layers: layer2 + layer3
input: resize 256, center crop 224
patch size / stride: 3 / 1
embedding: 1024
nearest neighbour: 1
coreset: approximate greedy, 10%
augmentation: none
```

必须同时记录：bank 向量数、实际内存、建库时间、单图检索时间和代码 commit。

---

<a id="applications"></a>

## 9. Memory Bank 的演进：不要只按名称分类

### 9.1 三条分类轴

| 轴       | 可能取值                                             |
| -------- | ---------------------------------------------------- |
| 存什么   | 原始样本特征、加权样本、统计量、聚类原型、可训练槽位 |
| 怎么更新 | 不更新、统计估计、聚类、EMA、显式读写、梯度下降      |
| 怎么使用 | 1-NN、kNN、马氏距离、soft projection、重建、对比损失 |

只有“可训练槽位 + 由损失反传更新”才是严格意义上的参数化 Memory Bank。

### 9.2 代表性时间线

|      年份 | 方法      | Memory 形态                        | 更新方式                          | 与工业 PatchCore 的关系                  |
| --------: | --------- | ---------------------------------- | --------------------------------- | ---------------------------------------- |
|      2019 | MemAE     | 固定数量 memory items              | 与 AE 端到端梯度训练              | 参数化记忆的基础，但走重建路线           |
|      2020 | MNAD      | 正常原型 items                     | 显式读写；compact / separate 约束 | 展示在线更新与防坍塌，但主要是视频 AD    |
| 2020/2021 | PaDiM     | 每个位置的$\mu_{ij},\Sigma_{ij}$ | 样本统计估计                      | 参数化分布，不是可学习 bank              |
|      2021 | DAAD      | 多尺度 block-wise memory           | 与编码器、解码器反向传播          | 在 MVTec 上直接验证可学习工业 memory     |
|      2022 | PatchCore | coreset 样本库                     | 非参数；另做 learned proxy 消融   | learned proxy 不如覆盖导向 coreset       |
|      2022 | CFA       | 固定规模中心 bank                  | 首图 K-means，逐图 EMA            | bank 非梯度；梯度训练 patch descriptor   |
| 2022/2023 | SoftPatch | 样本 + outlier weight              | 去噪、coreset、固定权重           | 解决污染，不做参数化                     |
|      2023 | PMB       | 分区、多层 memory units            | 与 AE 目标联合学习                | 工业/视觉重建式参数化 memory             |
| 2022/2026 | AnoMem    | 多尺度 Hopfield 权重               | 对比学习联合优化                  | 参数化、多尺度，但不是工业专用证据       |
|      2023 | LeMO      | $K=10$ prototype vectors         | 对比目标 + 重分配/聚类更新        | 最接近在线工业参数化 PatchCore           |
|      2024 | ProtoAD   | FINCH 聚类原型                     | 聚类后固定                        | 原型转 1×1 conv kernel，但没有梯度训练  |
|      2026 | ProCon    | 正常样本 memory                    | 不训练，soft projection           | 前沿重新转向“改查询”，而非“改成参数” |

年份和方法性质来自各原文指定页，具体证据见下面分节及文末引用。[2-12]

### 9.3 PaDiM：统计参数不等于可训练参数

**[论文发现]** PaDiM 在每个空间位置 $(i,j)$ 上，用正常 patch 集合估计多元高斯：

$$
\mathcal N(\mu_{ij},\Sigma_{ij}),
$$

并用马氏距离检测测试 patch。$\mu$ 和 $\Sigma$ 由样本均值、协方差直接计算，没有梯度更新；
其存储和推理复杂度不随训练图像数增长，但依赖空间对齐。[2, pp.2-4]

这类方法属于 parametric distribution model（参数分布模型），不属于 learnable memory slots。

### 9.4 MemAE：真正可训练的槽位

**[论文发现]** MemAE 定义：

$$
M\in\mathbb R^{N\times C},
$$

每一行 $m_i$ 是 memory item。编码 $z$ 与各 item 做 cosine similarity，经 softmax 得到权重，
再用少量 memory items 组合成送入解码器的 $\hat z$。memory、encoder、decoder 在正常数据重建目标下
共同更新；entropy regularizer 和 hard shrinkage 使寻址稀疏，降低异常通过大量正常 item 组合而被良好
重建的机会。[3, pp.2-5, Eq.(3)-(10)]

它解决的是“限制解码器能使用的潜变量”，不是“缩短 PatchCore 最近邻列表”。

### 9.5 MNAD：显式读写不是普通梯度参数

**[论文发现]** MNAD 为每个 query 对所有 memory items 计算读权重，同时定义另一组写权重；训练和测试时
都可以更新 memory。compactness loss 拉近 query 与最近 item，separateness loss 拉开最近与次近 item，
防止所有原型坍塌到一起。[4, pp.4-6, Eq.(1)-(12)]

MNAD 的 item 更新主要由显式归一化写规则完成，不应简单描述成“把整个 bank 放入 optimizer”。
它对在线工业 bank 的启发是：更新必须有异常门控，否则测试异常会被写成新的正常模式。[4, p.5]

### 9.6 DAAD 与 PMB：工业图像中的可学习重建记忆

> DAAD（Divide-and-Assemble Anomaly Detection，分而组装异常检测）：把特征划分成块，再从对应正常 memory 重组。
> PMB（Partition Memory Bank，分区记忆库）：为不同空间分区和低层特征配置独立 memory units。

**[论文发现]** DAAD 认为单像素级 memory 太容易用正常局部元素拼出异常；过大的整体 block 又会连正常图
也重建不好。因此使用多尺度、中等大小的 block-wise memory，并将 memory items 与网络参数一起反向传播。
它在 MVTec AD 上直接实验，是“参数化 memory 可用于工业 AD”的明确证据。[5, pp.2,4-6]

**[论文发现]** PMB 继续将低层特征按空间分区，每个分区用独立 memory unit，并通过重建损失联合学习；
目标是保存具有语义完整性的正常低层模式，减少异常被逻辑重组后仍能重建的问题。[7, pp.2,4-6]

但两者依赖“异常不能由正常记忆良好重建”的假设。神经解码器仍可能泛化到未见异常，且像素重建误差
容易受到光照和纹理噪声影响。这与 PatchCore 的距离检索风险不同。

### 9.7 CFA 与 ProtoAD：固定原型不能误标为梯度 Bank

**[论文发现]** CFA 的 bank $C$：先对第一张正常图的 patch 做 K-means，再逐图寻找匹配特征并用 EMA
更新中心；通过梯度学习的是 1×1 CoordConv patch descriptor。其 bank 大小与数据集样本数无关，
但更新本身不是 optimizer gradient。[6, pp.3-5, Algorithm 1]

**[论文发现]** ProtoAD 用 FINCH 非参数聚类得到正常原型，将 L2 归一化原型直接作为 1×1 convolution
kernel，以通道最大值完成 cosine-similarity 检索；论文明确说网络无需训练阶段。[11, pp.3,6-7]

二者都说明“固定规模原型”有价值，但不能作为“可学习参数优于 coreset”的证据。

### 9.8 LeMO：最贴近当前问题的直接尝试

> LeMO（Learning Memory for Online anomaly detection，面向在线异常检测的记忆学习）：
> 用很小的可学习原型库和局部 adapter 处理流式正常图像。

**[论文发现]** LeMO 冻结 WideResNet50，训练 1×1 local patch adapter，并设置：

$$
P=(p_1,\ldots,p_K)\in\mathbb R^{K\times D},
\qquad K=10.
$$

memory 可由单张图 K-means 初始化，也可将随机矩阵做 QR 分解，以正交噪声初始化；随后 AnoNCE
对比目标同时更新特征 $z$ 与原型 $P$，再通过平衡重分配、合并和 K-means 拆分缓解原型失衡。
[9, pp.3-6, Eq.(1)-(4)]

这条路线的重要变化是：不再拟合 PatchCore Equation (8) 的加权重构，而是直接让正常特征靠近相关
prototype、远离无关 prototype，推理仍使用最近原型距离。[9, pp.5-6]

**[证据边界]** LeMO 为 arXiv 预印本，未检索到作者公开代码，不能只凭论文表格
把其结果当作已被独立复现的工程结论。[9]

### 9.9 多尺度可学习记忆与 2026 年的新趋势

**[论文发现]** AnoMem 把每一层的正常原型表示为 learnable modern Hopfield memory，并在对比学习中联合
更新 encoder 与多尺度 memory 权重；同时加入方差项防止 prototype collapse。其公开版本主要验证通用
one-class、半监督和人脸攻击检测，不是 MVTec 工业证据。[8, pp.2-5]

**[分析判断]** 该工作在 2026 年以 *Unified Anomaly Detection via Multi-Scale Contrasted Memory*
发表于 IEEE Transactions on Image Processing。[8]

**[论文发现]** 2026 年 7 月的 ProCon 反而保持 training-free sample memory，用测试 patch 在局部正常
邻域上的 soft projection residual 代替硬 1-NN，并通过多 bank 与多层一致性稳定分数；作者明确强调
没有 decoder、backbone fine-tuning 或 learned fusion weight。[12, p.1]

**[分析判断]** 当前进展不是单向地“非参数 bank 必然被可学习 bank 淘汰”。至少存在三条并行方向：

1. 学习更好的特征，但保留样本/中心 bank；
2. 学习固定容量 memory slots；
3. 保持 training-free bank，改进检索、投影、鲁棒性和流式构建。

---

## 10. 参数化 Memory Bank 的收益与代价

### 10.1 潜在收益

**[分析判断]** 固定 $K\times d$ 的可训练 bank 可以：

- 让内存与训练集规模解耦；
- 与 adapter 联合适配当前产品域；
- 用连续优化吸收重复模式；
- 支持在线小步更新，而不必重跑全量 coreset；
- 将检索转成矩阵乘法或小规模距离计算。

### 10.2 核心风险

1. **目标错位**：重构组合好，不等于单个原型覆盖好，PatchCore Equation (8) 已展示。
2. **稀有正常模式丢失**：平均损失被高频模式主导，尾部正常模式容易变成误报。
3. **prototype collapse**：多个槽位收敛到相同模式，名义容量大、有效容量小。
4. **异常污染**：在线更新时若异常被当正常写入，后续同类缺陷会变成近邻。
5. **特征坐标系漂移**：adapter 或 backbone 更新后，旧 bank 与新特征不再处于同一表示空间。
6. **小样本过拟合**：工业类别每类正常图不多，额外参数未必比冻结预训练特征稳定。
7. **复现复杂度**：结果开始依赖初始化、学习率、更新顺序、原型利用率和停止条件。

### 10.3 为什么“可学习”不是自动更强

PatchCore coreset 直接近似：

$$
\min_{\mathcal M_C}
\max_m\min_n\lVert m-n\rVert_2,
$$

它明确保护最坏覆盖。常见 prototype 平均目标近似：

$$
\min_P\frac1{|\mathcal M|}
\sum_{m\in\mathcal M}\min_{p\in P}\lVert m-p\rVert_2^2,
$$

更接近 K-means，主要降低平均量化误差。一个很小但很远的正常簇对平均式贡献有限，却会主导 PatchCore
式最大异常分数。两者优化对象不同，这就是工业长尾正常模式下必须正视的取舍。

---

## 11. 面向当前 Memory Bank AD 的实验顺序

以下是**[分析判断]**，不是 PatchCore 原论文结论。

### 11.1 先建立不可缺失的基线

| 组 | Bank                                       | 特征       | 目的                             |
| -- | ------------------------------------------ | ---------- | -------------------------------- |
| A  | 完整 bank                                  | 冻结       | 测上限、内存和延迟               |
| B  | random 10% / 1%                            | 冻结       | 验证覆盖的重要性                 |
| C  | greedy coreset 10% / 1%                    | 冻结       | PatchCore 主基线                 |
| D  | K-means / FINCH，相同向量数                | 冻结       | 区分均值原型与 k-center 覆盖     |
| E  | PatchCore Eq.(8) 原式                      | 冻结       | 复现原论文 learned proxy 失败    |
| F  | Eq.(8) 改为负距离                          | 冻结       | 单独判断正号是否为关键问题       |
| G  | 最近原型距离目标 + separation / usage 约束 | 冻结       | 测真正面向检索的参数化 bank      |
| H  | G + 轻量 1×1 adapter                      | 局部可训练 | 判断 feature adaptation 是否必要 |

E 和 F 必须分开，否则无法区分“公式符号问题”和“重构目标错位”。

### 11.2 公平比较约束

- 各方法使用相同 backbone、层级、输入分辨率、embedding 维度和 bank 向量数。
- 每次特征提取器发生更新，都必须用同一版本特征重建或同步更新 bank。
- 除标准 MVTec AD 外，加入正常训练污染、目标域光照漂移和 few-shot 三种设置。
- 至少运行多个随机种子；参数化 bank 对初始化更敏感。
- 不只报告 AUROC，还报告固定召回率下误报、每类最差结果和正常尾部距离。

### 11.3 必须增加的 Memory 指标

| 指标                    | 说明                                        |
| ----------------------- | ------------------------------------------- |
| Bank bytes              | 真正占用的 RAM / VRAM，而不只报向量比例     |
| Build time              | 特征提取、采样或优化总时间                  |
| Retrieval latency       | 排除和包含 backbone 的两种延迟              |
| Utilization             | 测试时被选为最近邻的 bank 向量比例          |
| Coverage radius         | $\max_m\min_p\lVert m-p\rVert_2$          |
| Mean quantization error | $\frac1M\sum_m\min_p\lVert m-p\rVert_2^2$ |
| Prototype pair distance | 监测原型是否坍塌                            |
| Assignment entropy      | 监测少数原型垄断全部正常 patch              |
| Contamination retention | 注入缺陷后有多少异常 patch 留在 bank        |

Coverage radius 与 mean quantization error 必须同时看：前者对应 PatchCore 的最坏覆盖，后者对应原型的
平均拟合，单独一个都不完整。

---

## 12. 当前阶段结论

1. **[论文发现]** PatchCore 的核心不是“保存全部特征”，而是中层局部特征、覆盖导向 coreset 与 patch
   最近邻最大分数的组合。[1, pp.2-7]
2. **[论文发现]** 默认单模型是 99.0-99.1 image AUROC；99.6 是高分辨率多 backbone ensemble，
   不能混写。[1, pp.6-7]
3. **[论文发现]** PatchCore 已尝试可学习 basis proxies，但 Equation (8) 的正距离 softmax 可疑，
   且重构目标与 1-NN 覆盖存在结构性错位；实验上 learned proxies 不如 coreset。[1, p.7]
4. **[分析判断]（公开实现核验）** Amazon 官方代码的 backbone 和 bank 完全非参数化，并未实现论文 Equation (7)；
   anomalib 当前实现了该重加权。[13,14]
5. **[文献结论]** 可学习 memory slots 在 MemAE、DAAD、PMB、AnoMem、LeMO 中持续发展，但需要区分
   梯度参数、显式读写、EMA 中心和统计参数。[3-9]
6. **[分析判断]** 对当前基于 PatchCore 的工业 AD，第一优先级不是直接替换成参数 bank，而是用相同
   bank 预算复现 `coreset / cluster / learned proxy`，并同时测最坏覆盖、正常尾部误报和污染鲁棒性。

---

## 论文引用

> [1] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler.
> “Towards Total Recall in Industrial Anomaly Detection.” *IEEE/CVF Conference on Computer Vision and
> Pattern Recognition (CVPR)*, pp.14298-14308, 2022. DOI: 10.1109/CVPR52688.2022.01392.
> 本文使用 arXiv:2106.08265v2 的公开分页：动机 pp.1-3；Equation (1)-(4) pp.3-4；Equation (5)、
> Algorithm 1、Equation (6)-(7) pp.4-5；主结果 p.6；Figure 5 与 Equation (8)、速度和 ensemble p.7；
> MTD、mSTC 与限制 p.8；实现细节 p.12。

> [2] Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier.
> “PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.”
> *International Conference on Pattern Recognition Workshops (ICPR)*, pp.475-489, 2021.
> arXiv:2011.08785v1 public version pp.1-7.
> 本文引用位置：位置高斯、统计估计与马氏距离 pp.2-4，Equation (1)-(2) p.3。

> [3] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh,
> Anton van den Hengel. “Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for
> Unsupervised Anomaly Detection.” *IEEE/CVF International Conference on Computer Vision (ICCV)*,
> pp.1705-1714, 2019. DOI: 10.1109/ICCV.2019.00179.
> arXiv:1904.02639v2 public version：联合更新与方法动机 p.2；memory 定义、寻址、稀疏化和训练 pp.3-5，
> Equation (1)-(10)。

> [4] Hyunjong Park, Jongyoun Noh, Bumsub Ham. “Learning Memory-Guided Normality for Anomaly Detection.”
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.14360-14369, 2020.
> DOI: 10.1109/CVPR42600.2020.01438.
> arXiv:2003.13228v1 public version：memory read/write pp.4-5；compactness、separateness 与测试更新门控
> pp.5-6，Equation (1)-(12)。

> [5] Jinlei Hou, Yingying Zhang, Qiaoyong Zhong, Di Xie, Shiliang Pu, Hong Zhou.
> “Divide-and-Assemble: Learning Block-Wise Memory for Unsupervised Anomaly Detection.”
> *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp.8771-8780, 2021.
> DOI: 10.1109/ICCV48922.2021.00867.
> arXiv:2107.13118v1 public version：block size 动机与端到端 memory p.2；multi-scale block-wise module
> pp.4-5；损失、MVTec 设置与结果 pp.5-6。

> [6] Sungwook Lee, Seunghyun Lee, Byung Cheol Song.
> “CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization.”
> *IEEE Access*, vol.10, pp.78446-78454, 2022. DOI: 10.1109/ACCESS.2022.3193699.
> arXiv:2206.04325v1 public version：adapter 损失 p.3；K-means + EMA memory Algorithm 1 p.4；
> scoring 与实验设置 p.5。

> [7] Peng Xing, Zechao Li. “Visual Anomaly Detection via Partition Memory Bank Module and Error
> Estimation.” *IEEE Transactions on Circuits and Systems for Video Technology*, vol.33, no.8,
> pp.3596-3607, 2023. DOI: 10.1109/TCSVT.2023.3237562.
> arXiv:2209.12441v1 public version：PMB 动机与联合学习 p.2；分区 memory 结构与寻址 pp.4-5；
> 重建目标与实验 p.5-6。

> [8] Loic Jezequel, Ngoc-Son Vu, Jean Beaudet, Aymeric Histace.
> “Anomaly Detection via Multi-Scale Contrasted Memory.” arXiv:2211.09041v2, pp.1-10, 2022.
> 该工作扩展版以 “Unified Anomaly Detection via Multi-Scale Contrasted Memory” 发表于
> *IEEE Transactions on Image Processing*, vol.35, pp.2802-2815, 2026,
> DOI: 10.1109/TIP.2026.3663923。
> 本文引用公开预印本：Hopfield memory 定义 p.2；多尺度对比 memory 与防坍塌 pp.3-5。

> [9] Han Gao, Huiyuan Luo, Fei Shen, Zhengtao Zhang.
> “Towards Total Online Unsupervised Anomaly Detection and Localization in Industrial Vision.”
> arXiv:2305.15652v1, pp.1-12, 2023.
> 本文引用位置：在线问题和方法定位 pp.1-3；正交初始化、$K=10$ bank 和 AnoNCE 联合更新 pp.4-6；
> 在线实验 pp.6-9。

> [10] Xi Jiang, Jianlin Liu, Jinbao Wang, Qian Nie, Kai Wu, Yong Liu, Chengjie Wang, Feng Zheng.
> “SoftPatch: Unsupervised Anomaly Detection with Noisy Data.”
> *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, pp.15433-15445, 2022.
> arXiv:2403.14233v1 public version pp.1-17.
> 本文引用位置：污染问题和 patch 去噪 pp.1-2；weighted coreset pp.4-6，Equation (1)-(8)。

> [11] Chao Huang, Zhao Kang, Hong Wu. “A Prototype-Based Neural Network for Image Anomaly Detection
> and Localization.” *Neural Processing Letters*, vol.56, article 169, 2024.
> DOI: 10.1007/s11063-024-11466-7. arXiv:2310.02576v2 public version pp.1-20.
> 本文引用位置：非参数 prototype 与无需训练 p.3；FINCH 选择和结构 pp.6-7。

> [12] Joongwon Chae, Lihui Luo, Yang Liu, Dongmei Yu, Peiwu Qin, Runming Wang, Ilmoon Chae.
> “ProCon: Projection-Consistency Memory for Training-Free Anomaly Detection.”
> arXiv:2607.04894v1, 2026.
> 本文仅引用第 1 页摘要中的 soft projection、training-free 设定和公开代码声明。

## 代码与网络资源

> [13] Amazon Science. `patchcore-inspection`, Apache-2.0.
> https://github.com/amazon-science/patchcore-inspection

> [14] Open Edge Platform. `anomalib`, PatchCore implementation, Apache-2.0.
> https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/patchcore

> 其他对应实现：
>
> - MemAE：https://github.com/donggong1/memae-anomaly-detection
> - CFA：https://github.com/sungwool/CFA_for_anomaly_localization
> - SoftPatch：https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch
> - ProCon：https://github.com/jw-chae/Procon
