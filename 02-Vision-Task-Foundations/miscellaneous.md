# 其余关键技术详解

> 本文暂作跨主题查阅页：可变形卷积、空间条件归一化、多实例学习、局部敏感性、参数平均与 Fourier 特征。Grounding DINO 的完整结构见 [DETR 串讲](DETR.md#grounding-dino)。各主题后续应并入对应串讲；本文不再充当新主题的默认收纳页。

---

## 一、可变形卷积（Deformable Convolution）

可变形卷积根据输入预测采样偏移，使卷积采样位置可以离开固定网格。

> 出处：Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei. “Deformable Convolutional Networks.” *IEEE International Conference on Computer Vision (ICCV)*, pp.764-773, 2017。标准卷积与可变形卷积定义见公开版 pp.3-4, Equations (1)-(2)。

### 1.1 核心问题

标准卷积在**固定规则网格**上采样，无法自适应物体的几何变形（方向、尺度、形变）。

### 1.2 原理

标准卷积：
$$y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n)$$

其中 $\mathcal{R}$ 是固定网格（如 3×3 卷积的 $(-1,-1),(-1,0),...,(1,1)$）。

**Deformable Conv** 为每个采样位置学习一个**偏移量**：
$$y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n + \Delta p_n)$$

其中 $\Delta p_n$ 由另一个卷积网络预测（二维情况下，每个空间位置输出 $2N$ 个偏移值，$N$ 为核内采样点数量）。非整数位置通过双线性插值取样。[论文发现，原文第 3 页公式 (2)–(4)]

---

## 二、SPADE (Spatially-Adaptive Denormalization)

SPADE（Spatially-Adaptive Denormalization，空间自适应反归一化）根据语义条件为不同空间位置预测不同的缩放与偏移。

> 出处：Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu. “Semantic Image Synthesis with Spatially-Adaptive Normalization.” *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.2337-2346, 2019。无参数归一化与 SPADE 定义见公开版 pp.3-4, Equations (1)-(2)。

### 2.1 核心思想

标准 BatchNorm 的仿射变换是**全局的**（每个通道一对 γ, β）。SPADE 让 γ 和 β 变成**空间变化的**（spatially-varying），由条件输入（如 layout/分割图）预测。

### 2.2 公式

标准 BN：
$$y = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

SPADE：
$$y = \gamma(c)_{h,w} \cdot \frac{x - \mu}{\sigma} + \beta(c)_{h,w}$$

其中 $\gamma(c)$ 和 $\beta(c)$ 是从条件 c 经过卷积网络预测的**空间变化**参数。

---

## 三、Grounding DINO

> 出处：Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang. “Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.” *European Conference on Computer Vision (ECCV)*, Lecture Notes in Computer Science, pp.38-55, 2024。整体架构、feature enhancer、language-guided query selection 与 cross-modality decoder 见公开预印本 pp.3-5。

### 3.1 核心思想

Grounding DINO 将**开放词汇目标检测**与**视觉 grounding** 结合：
- 输入：图像 + 文本查询（如 "car . truck . bus"）
- 输出：检测框 + 类别匹配（每个框对应哪个文本查询）

### 3.2 架构简述

```
图像 → Image Backbone (Swin-B) → 多尺度特征
                                         │
文本 → Text Backbone (BERT) → 文本特征   │
                                         │
                    ┌────────────────────┘
                    ▼
        Feature Enhancer (Deformable Self-Attn + Cross-Attn)
                    │
                    ▼
        Language-Guided Query Selection
                    │
                    ▼
        Cross-Modality Decoder
                    │
                    ▼
        Detection: boxes + text-matched classes
```

## 四、MIL Loss (Multiple Instance Learning Loss)

> MIL（Multiple Instance Learning，多实例学习）：训练标签只说明一个样本包中是否含有正实例，不要求给出每个实例的精确标签。

### 4.1 为什么需要 MIL

当弱标签或伪标签可能漏检、误检时，标准 CrossEntropy 假设每个 proposal 都有精确类别，因而对噪声敏感。

MIL 的核心改变：**不指定哪个 proposal 属于哪个类别**，而是说"这组 proposals 中至少有一个属于 class k"。

### 4.2 MIL 与标准 CrossEntropy

| | CrossEntropy | MILCrossEntropy |
|---|---|---|
| **标签要求** | 每个候选区域一个实例标签 | 每个样本包一个或多个标签，不指明包内具体哪个实例为正 |
| **对噪声的容忍** | 取决于实例标签质量和目标函数 | 可容忍实例位置未知；不能自动修复错误的包级类别标签 |
| **优化目标** | proposal-level 精确分类 | bag-level 至少有一个正确的 |
| **适用场景** | 干净标注 | 伪标签、弱监督 |

**[数学推导]** 例如对二分类包 $B=\{x_i\}_{i=1}^m$，在给定实例正类概率 $p_i$ 且条件独立的假设下，所有实例均为负的概率为 $\prod_i(1-p_i)$，于是“至少一个为正”的包概率为 $P_B=1-\prod_i(1-p_i)$。包标签 $y_B\in\{0,1\}$ 对应损失 $-y_B\log P_B-(1-y_B)\log(1-P_B)$。这只是明确假设下的一种聚合形式，MIL 不存在唯一通用的损失公式；多热编码与标签不确定性也不是同一概念。

---

## 五、Jacobian 与奇异值分解：局部敏感性

Jacobian（雅可比矩阵）汇集向量函数各输出对各输入的偏导；SVD（Singular Value Decomposition，奇异值分解）将线性映射分解为输入方向、缩放系数和输出方向。

### 5.1 理论背景

对于分类网络 $f: \mathbb{R}^D \rightarrow \mathbb{R}^K$（D 为输入维度，K 为类别数），**输出关于输入的 Jacobian 矩阵**：

$$J = \frac{\partial f(x)}{\partial x} = \begin{pmatrix} 
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_D} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_K}{\partial x_1} & \cdots & \frac{\partial f_K}{\partial x_D}
\end{pmatrix} \in \mathbb{R}^{K \times D}$$

- $J_{ij}$：输入的第 j 个像素变化时，对第 i 个类别 logit 的影响
- 它描述固定输入附近的输出敏感性；仅凭静态分类网络的 Jacobian，不能推出存在连续吸引子。

### 5.2 为什么 Jacobian SVD 能反映局部几何结构

对输入 $x$ 附近的小扰动 $\delta$：
$$f(x + \delta) \approx f(x) + J \cdot \delta$$

**[数学推导]** 设 $J=U\Sigma V^T$，$v_i,u_i$ 为一对右、左奇异向量，奇异值为 $\sigma_i$。令 $\delta=\epsilon v_i$，则

$$f(x+\epsilon v_i)-f(x)=\epsilon\sigma_i u_i+o(|\epsilon|).$$

因此大奇异值表示该输入方向上一阶变化大，小奇异值表示一阶变化小。$\sigma_i=0$ 只消去一阶项；例如 $f(x)=x^2$ 在 $x=0$ 的导数为零，但有限扰动仍改变输出。分类边界还取决于类别分数差，不能由某个大奇异值直接断言“边界清晰”。

**[分析判断]** 吸引子属于动力系统概念，必须另行定义状态递推 $h_{t+1}=F(h_t)$ 或微分方程，并检查不变集合及其附近的稳定性；静态映射 $f:\mathbb R^D\to\mathbb R^K$ 的低秩不足以提供这些条件。

---

## 六、EMA Teacher

> EMA（Exponential Moving Average，指数移动平均）：用历史参数的指数加权平均构造变化更平滑的 teacher。

### 6.1 为什么用 EMA

**[论文发现]** Mean Teacher 使用学生参数的指数移动平均构造教师，并让带扰动的学生与教师预测保持一致；教师参数通过平均更新，不通过一致性损失直接优化。[1，第 3 页] 参数平均减少相邻步教师权重的变化，但不保证其伪标签正确，也不能据此断言直接复制参数必然导致“作弊”。

teacher 参数 $\theta_T$ 的更新为：

$$
\theta_T\leftarrow m\theta_T+(1-m)\theta_S,
\qquad 0\leq m<1,
$$

其中 $\theta_S$ 是当前 student 参数，$m$ 越接近 1，teacher 变化越慢。

### 6.2 EMA momentum 的影响

**[数学推导]** 展开更新可得 $\theta_{T,t}=m^t\theta_{T,0}+(1-m)\sum_{j=1}^t m^{t-j}\theta_{S,j}$。忽略初始项后，历史权重按几何级数衰减，典型时间尺度约为 $1/(1-m)$ 步。$m=0.9,0.99,0.999$ 分别对应约 $10,100,1000$ 步；$m=0$ 才是直接复制，$m=1$ 是固定教师这一边界情形。最优平滑程度取决于学生变化速度。

---

## 七、Fourier Features / Positional Encoding

### 7.1 为什么需要位置编码

**[论文发现]** Fourier 特征工作研究低维坐标回归中 MLP（Multi-Layer Perceptron，多层感知机）的频谱偏置，用正弦和余弦特征改变其对不同频率的学习能力。[2，第 1–4 页] 这与 Transformer 用位置编码打破无位置注意力的排列对称性是相关但不同的问题；坐标本身不宜称为“低频数据”。

### 7.2 正弦位置编码

对位置 $p$ 和通道索引 $i$：

$$
\operatorname{PE}(p,2i)=\sin\left(p/10000^{2i/d}\right),
$$

$$
\operatorname{PE}(p,2i+1)=\cos\left(p/10000^{2i/d}\right).
$$

**[论文发现]** 上式来自 Transformer 的固定位置编码，其中 $d$ 为编码宽度。[3，第 6 页]

### 7.3 Fourier 特征映射

对坐标 $x$ 和频率矩阵 $B$：

$$
\gamma(x)=[\sin(2\pi Bx),\cos(2\pi Bx)].
$$

**Fourier Features 的关键性质**：
- 坐标 $(x,y)$ → 一组不同频率的正弦、余弦特征 → 改变多层感知机拟合空间细节时的频率偏好；坐标本身不应被称为“低频数据”
- 类似核方法中的 RBF kernel approximation
- 在 NeRF 中也大量使用

---

## 八、常见理论问题

### Q: Deformable Conv 和普通 Conv 的计算开销对比？

**A**: Deformable Conv 额外开销来自：
1. 偏移预测网络（一个额外的卷积层）
2. 连续坐标上的双线性插值采样

它比标准卷积多出偏移预测与插值采样两部分计算，因此计算量和访存开销都会增加。

### Q: MIL Loss 和标准 CrossEntropy 在什么情况下差异最大？

**[分析判断]** 当包标签可靠但实例位置未知时，两者监督粒度不同，MIL 可以避免把所有候选区域都强行赋予同一实例标签。若整包的真实类别是 car 却被误标成 truck，MIL 仍会受错误包标签误导；它不是普遍的抗伪标签噪声机制。

### Q: EMA 的 momentum 为什么通常接近 1？

**A**: 
- Teacher 模型应该提供**慢变的、稳定的**学习 target
- 较大的 momentum 使 teacher 对较长时间范围内的 student 参数做平滑平均
- momentum 太小时，teacher 会快速跟随 student，稳定目标的作用减弱

### Q: Jacobian 的奇异值为 0 意味着什么？

**[数学推导]** 对应右奇异向量方向的一阶响应为零；高阶变化仍可能存在。只有在定义了动力系统并验证不变性与稳定性之后，才有依据讨论吸引子及其切空间。

## 补充论文出处

> [1] Antti Tarvainen, Harri Valpola. “Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results.” NeurIPS, 2017。第 3 页 Figure 2、教师参数更新与一致性目标。[归档原文](<../04-Domain-Adaptation/Mean-Teacher/[NeurIPS 2017] MeanTeacher - Mean Teachers Are Better Role Models/paper.pdf>)
>
> [2] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng. “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.” NeurIPS, 2020。arXiv:2006.10739，第 1–4 页讨论频谱偏置、特征映射及频率尺度。[原文](https://arxiv.org/pdf/2006.10739)
>
> [3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. “Attention Is All You Need.” NeurIPS, 2017。第 6 页 §3.5 位置编码。[原文](https://arxiv.org/pdf/1706.03762)
