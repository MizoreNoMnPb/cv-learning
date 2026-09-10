# 其余关键技术详解

> 覆盖：Deformable Convolution / SPADE / Grounding DINO / MIL Loss / Gradient Discrepancy Loss / Jacobian SVD / EMA Update / Fourier Features

---

## 一、Deformable Convolution

> 出处：Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei. “Deformable Convolutional Networks.” *IEEE International Conference on Computer Vision (ICCV)*, pp.764-773, 2017。标准卷积与可变形卷积定义见公开版 pp.3-4, Equations (1)-(2)。

### 1.1 核心问题

标准卷积在**固定规则网格**上采样，无法自适应物体的几何变形（方向、尺度、形变）。

### 1.2 原理

标准卷积：
$$y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n)$$

其中 $\mathcal{R}$ 是固定网格（如 3×3 卷积的 $(-1,-1),(-1,0),...,(1,1)$）。

**Deformable Conv** 为每个采样位置学习一个**偏移量**：
$$y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n + \Delta p_n)$$

其中 $\Delta p_n$ 由另一个卷积网络预测（通常是 2N 个偏移值，N 为核大小）。

---

## 二、SPADE (Spatially-Adaptive Denormalization)

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
| **标签要求** | 每个 proposal 一个确定标签 | 可以为多热编码（不确定） |
| **对噪声的容忍** | 低（噪声标签直接误导） | 高（只需正类总体有响应） |
| **优化目标** | proposal-level 精确分类 | bag-level 至少有一个正确的 |
| **适用场景** | 干净标注 | 伪标签、弱监督 |

---

## 五、Jacobian SVD

### 5.1 理论背景

对于分类网络 $f: \mathbb{R}^D \rightarrow \mathbb{R}^K$（D 为输入维度，K 为类别数），**输出关于输入的 Jacobian 矩阵**：

$$J = \frac{\partial f(x)}{\partial x} = \begin{pmatrix} 
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_D} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_K}{\partial x_1} & \cdots & \frac{\partial f_K}{\partial x_D}
\end{pmatrix} \in \mathbb{R}^{K \times D}$$

- $J_{ij}$：输入的第 j 个像素变化时，对第 i 个类别 logit 的影响
- 如果所有输入方向上的扰动都使输出变化相似 → Jacobian 的奇异值分布**集中**（少数大奇异值）→ 存在**连续吸引子**
- 如果每个方向独立影响输出 → 奇异值分布**分散** → 无明显的连续吸引子结构

### 5.2 为什么 Jacobian SVD 能反映局部几何结构

对输入 $x$ 附近的小扰动 $\delta$：
$$f(x + \delta) \approx f(x) + J \cdot \delta$$

- **大奇异值对应的方向**：输入沿该方向变化时，输出变化剧烈（→ 对扰动敏感，分类边界清晰）
- **小奇异值对应的方向**：输入沿该方向变化时，输出几乎不变（→ 对扰动不敏感，可能是"平坦"的吸引子区域）
- 奇异值的**分布特征**（如均值、方差、极差、最大最小比）反映了网络的"吸引子景观"
- 不同架构（CNN, RNN, MLP）的奇异值分布模式不同 → 反映了架构结构对吸引子行为的影响

---

## 六、EMA Teacher

> EMA（Exponential Moving Average，指数移动平均）：用历史参数的指数加权平均构造变化更平滑的 teacher。

### 6.1 为什么用 EMA

直接将 student 参数复制给 teacher（hard update）会导致：
- Teacher 随 student 快速变化 → 伪标签不稳定
- Student 容易"作弊"（adversarially exploit teacher）

EMA 让 teacher 缓慢跟随 student，提供更稳定的学习目标。

teacher 参数 $\theta_T$ 的更新为：

$$
\theta_T\leftarrow m\theta_T+(1-m)\theta_S,
\qquad 0\leq m<1,
$$

其中 $\theta_S$ 是当前 student 参数，$m$ 越接近 1，teacher 变化越慢。

### 6.2 EMA momentum 的影响

| momentum | 含义 | 效果 |
|----------|------|------|
| 1.0 | Teacher 固定（不更新） | Baseline |
| 0.999 | Teacher 极慢变化 | 最稳定，但可能跟不上 student |
| 0.99 | Teacher 缓慢变化 | 稳定性与跟随速度折中 |
| 0.9 | Teacher 较快变化 | 接近直接复制，不太稳定 |

---

## 七、Fourier Features / Positional Encoding

### 7.1 为什么需要位置编码

神经网络（特别是 MLP）难以学习**高频函数**——直接输入坐标 (x, y) 很难让网络学到精确的空间位置。Fourier 特征映射将低频坐标映射到高频空间，使 MLP 能更好地表示空间细节。

### 7.2 正弦位置编码

对位置 $p$ 和通道索引 $i$：

$$
\operatorname{PE}(p,2i)=\sin\left(p/10000^{2i/d}\right),
$$

$$
\operatorname{PE}(p,2i+1)=\cos\left(p/10000^{2i/d}\right).
$$

### 7.3 Fourier 特征映射

对坐标 $x$ 和频率矩阵 $B$：

$$
\gamma(x)=[\sin(2\pi Bx),\cos(2\pi Bx)].
$$

**Fourier Features 的关键性质**：
- 低频坐标 (x,y) → 映射到高频空间 → MLP 可以更容易地学习空间变化
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

**A**: 当**伪标签有噪声**时差异最大：
- 如果 GDINO 将一个 "car" 误标为 "truck"，CrossEntropy 会强行优化该 proposal 向 "truck" 方向，导致特征混淆
- MIL 不要求每个 proposal 精确分类，只要求在一组 proposals 中 "truck" 的正类总分高即可，容忍个别错误

### Q: EMA 的 momentum 为什么通常接近 1？

**A**: 
- Teacher 模型应该提供**慢变的、稳定的**学习 target
- 较大的 momentum 使 teacher 对较长时间范围内的 student 参数做平滑平均
- momentum 太小时，teacher 会快速跟随 student，稳定目标的作用减弱

### Q: Jacobian 的奇异值为 0 意味着什么？

**A**:
- 存在输入方向，沿该方向扰动时**任意类别的输出都不变化**
- 这意味着网络在该输入附近有一个"平坦"的方向
- 从动力系统角度看，这对应**连续吸引子**的 tangent space
- 奇异值越小（越接近 0），说明该方向的输出不敏感度越高，吸引子结构越明显
