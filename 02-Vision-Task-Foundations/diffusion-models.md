# 扩散模型：DDPM 的概率过程、潜空间与条件生成

> DDPM（Denoising Diffusion Probabilistic Models，去噪扩散概率模型）用逐步加噪与学习反向去噪来建模数据；LDM（Latent Diffusion Model，潜空间扩散模型）把扩散过程放在编码器得到的潜变量空间。本文介绍核心公式与边界，不包含完整的变分下界推导。

原论文设计标为 **[论文发现]**，自行整理的计算标为 **[数学推导]**，模型选择与局限标为 **[分析判断]**。

---

## 一、概率扩散模型的核心直觉

扩散模型的核心思想分两步：
1. **前向过程（Forward）**：向数据逐步添加高斯噪声；当残留信号足够小时，末端分布近似标准高斯。
2. **反向过程（Reverse）**：学习一个神经网络来逐步去噪，从噪声中恢复数据

```
Forward (固定，不需要学习):
x₀ → x₁ → x₂ → ... → x_T  (数据 → 纯噪声)
  +噪声  +噪声      +噪声

Reverse (需要学习):
x_T → x_{T-1} → ... → x₀  (纯噪声 → 数据)
  去噪     去噪      去噪
```

---

## 二、DDPM 数学推导

### 2.1 前向过程

**[论文发现]** DDPM 的前向过程是固定的马尔可夫链，每一步向数据添加高斯噪声：[1，第 2 页公式 (2)]

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$

其中 $\beta_t \in (0, 1)$ 是噪声日程。原论文的 $T=1000$ 实验采用从 $10^{-4}$ 到 $0.02$ 的线性日程；这是一组实验设置，并非扩散模型的定义。[1，第 6 页]

**重参数化技巧**：直接从 $x_0$ 一步计算 $x_t$（不需要迭代）：
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$$

即：
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

**[数学推导]** 递推 $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t$，其中各步噪声独立，信号系数乘积为 $\sqrt{\bar\alpha_t}$。累计噪声方差满足 $v_t=\alpha_t v_{t-1}+\beta_t$、$v_0=0$；代入可得 $v_t=1-\bar\alpha_t$。独立高斯的线性组合仍为高斯，因此得到上述边缘分布。[1，第 2 页公式 (4)]

### 2.2 反向过程

反向过程也是一个马尔可夫链，但需要学习：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})$$

DDPM 将 $\sigma_t^2$ 固定为 $\beta_t$ 或 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$（无学习），只学习 $\mu_\theta$。

### 2.3 训练目标

**[论文发现]** DDPM 比较不同反向均值参数化与训练目标，使用简化的噪声预测目标：[1，第 4–5 页公式 (11)–(14)]

$$\mathcal{L}_{simple}(\theta) = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

其中：
- $x_0 \sim q(x_0)$：从数据分布采样
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：随机噪声
- $t \sim \text{Uniform}(1, ..., T)$：随机时间步
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

**[论文发现]** 固定 $x_t,t$ 时，噪声与反向均值存在下节所示的仿射对应。但参数化可换算，不代表不加权的平方误差具有相同优化目标。原变分目标中，噪声误差的权重为 $\beta_t^2/[2\sigma_t^2\alpha_t(1-\bar\alpha_t)]$；$\mathcal L_{simple}$ 去掉这一时间步权重，改变了各噪声水平对训练的贡献。[1，第 4–5 页]

### 2.4 DDPM 采样过程

从预测的噪声 $\epsilon_\theta$ 到 $x_{t-1}$ 的均值和采样：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t \mathbf{I})$$

其中均值由预测噪声反推：
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

**[论文发现]** 采样从 $x_T\sim\mathcal N(0,I)$ 开始，按 $t=T,\ldots,1$ 迭代：[1，第 4 页 Algorithm 2]
$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z,\qquad z\sim\mathcal N(0,I)\ (t>1),\quad z=0\ (t=1).$$
最后一步不再加入随机噪声；上式是原论文展示样本的采样规则。

---

## 三、U-Net 噪声预测网络

### 3.1 为什么这些工作使用 U-Net

扩散模型的去噪任务本质上是**像素级预测**（逐像素预测噪声），需要：
- 大感受野（理解全局图像结构）
- 保持空间分辨率（精确定位噪声）
- U-Net 的编码器、解码器和跳跃连接提供一种结合上下文与空间细节的结构；这不意味着扩散网络只能采用 U-Net。[1，第 6 页；2，第 4 页]

### 3.2 时间步 t 的注入方式

模型需要知道当前处于哪个扩散步（不同步的噪声水平不同）：

常见做法先把标量 $t$ 映射为正弦位置编码，再通过可学习映射得到时间嵌入 $e_t$。对第 $i$ 组频率：

$$
e_t^{(2i)}=\sin\left(t\omega_i\right),
\qquad
e_t^{(2i+1)}=\cos\left(t\omega_i\right).
$$

**[分析判断]** 时间嵌入可以通过相加或缩放与平移调制残差块。下式仅表示后一种一般参数化，不是原始 DDPM 每一层都采用的规定：

$$
h'=\gamma(e_t)\odot\operatorname{Norm}(h)+\beta(e_t).
$$

### 3.3 条件注入：Cross-Attention

**[论文发现]** LDM 用交叉注意力注入文本等条件；拼接条件也是原文讨论的方式，所以条件生成并非必须使用交叉注意力：[2，第 4–5 页]

$$
\operatorname{CrossAttn}(h,c)
=\operatorname{softmax}\left(
\frac{(hW_Q)(cW_K)^T}{\sqrt{d_k}}
\right)cW_V,
$$

其中图像特征 $h$ 产生 query，条件序列 $c$ 产生 key 和 value。

---

## 四、Latent Diffusion Model (LDM / Stable Diffusion)

### 4.1 为什么需要 LDM

DDPM 在**像素空间**做扩散，特征张量的空间规模随图像分辨率增长，因此计算量与中间激活规模都很大。

**[论文发现]** LDM 在预训练自编码器的潜空间做扩散。原论文比较 KL 正则化和向量量化两类潜空间；VAE（Variational Autoencoder，变分自编码器）通过变分目标学习带概率约束的潜表示，不能把所有 LDM 都等同于同一种 VAE 配置。[2，第 4–5 页]

```
像素空间 (3×H×W)  →  编码器  →  潜空间 (c×H/f×W/f)
                                         ↓
                                    扩散/去噪过程（低维计算）
                                         ↓
                                    解码器  →  像素空间
```

**[数学推导]** 若举例取 $c=4,f=8$，$512\times512\times3$ 图像变为 $64\times64\times4$ 潜变量，标量数之比为 $(512^2\times3)/(64^2\times4)=48$。这只说明输入张量元素数减少 48 倍，不能推出网络浮点运算或耗时同样减少 48 倍；它们还受通道宽度、注意力、层数和编解码器影响。

### 4.2 LDM 训练

1. 用编码器把图像映射到潜变量 $z_0$。
2. 在潜空间按 DDPM 前向过程构造 $z_t$。
3. 训练去噪网络预测加入的噪声。
4. 生成时从潜空间噪声逐步去噪，再由解码器还原图像。

其简化目标与像素空间 DDPM 形式相同，只是把 $x$ 替换为 $z$：

$$
\mathcal L_{\mathrm{LDM}}
=\mathbb E_{z_0,\epsilon,t}
\left[
\left\lVert
\epsilon-\epsilon_\theta(z_t,t,c)
\right\rVert_2^2
\right].
$$

---

## 五、采样加速与条件引导是两类操作

### 5.1 DDIM：去噪扩散隐式模型

DDIM（Denoising Diffusion Implicit Models，去噪扩散隐式模型）构造与 DDPM 共享训练目标的一族生成过程。**[论文发现]** 令随机性参数为零可得到下列确定性更新；这只是该族的一个特例。[3，第 4–6 页公式 (12)]

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{"预测的 }x_0\text{"}} + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta$$

DDIM 允许**跳跃采样**（如只用 50 步而非 1000 步），大幅加速。

### 5.2 CFG：无分类器引导

CFG（Classifier-Free Guidance，无分类器引导）在不另外训练分类器的情况下组合有条件与无条件预测，以调节条件匹配和样本多样性的权衡。**[论文发现]** 原文训练时随机丢弃部分条件，采样时混合两种预测：[4，第 4–5 页]

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

此处 $w$ 是引导尺度，$w=1$ 得到普通条件预测，$w>1$ 放大两者之差；它对应原文式 (6) 中的 $1+w_{\mathrm{paper}}$。CFG 本身不减少采样步数，通常需要条件和无条件两次网络求值；它不是一种采样加速算法。引导尺度应结合模型与任务选择。

### 5.3 DPM-Solver

**[论文发现]** DPM-Solver（Diffusion Probabilistic Model Solver，扩散概率模型求解器）针对扩散常微分方程（Ordinary Differential Equation，ODE）的结构构造高阶数值求解方法，以减少网络求值次数。[5，第 3–5 页] 它可与条件引导组合，但两者解决的问题不同。

---

## 六、常见理论问题

### 扩散模型是否总比生成对抗网络好？

**[分析判断]** 不能从训练目标直接推出这种普遍排序。GAN（Generative Adversarial Network，生成对抗网络）通过生成器与判别器的对抗目标建模数据；扩散模型使用去噪目标，优化问题不同，但平方误差并不保证训练始终稳定。比较应固定数据、模型规模、训练预算、采样预算与评估指标。

### Q: 为什么预测噪声 ε 而不是直接预测 x₀？

**[分析判断]** 原始样本 $x_0\sim q(x_0)$ 的分布不随采样的时间步变化，变化的是带噪分布 $q(x_t)$。噪声参数化与数据参数化可以相互换算，但损失权重、信噪比与模型结构影响优化；原 DDPM 的实验不能证明噪声预测在所有模型中都最优。[1，第 4–6 页]

### Q: 如何处理条件扩散中的"无分类器引导"（CFG）？

**A**: 训练时以一定概率随机丢弃条件（设为空嵌入），让模型同时学习有条件和无条件去噪。推理时用 CFG 公式混合两种预测，放大条件信号。

## 论文出处

> [1] Jonathan Ho, Ajay Jain, Pieter Abbeel. “Denoising Diffusion Probabilistic Models.” NeurIPS, 2020。arXiv:2006.11239v2，第 2 页公式 (2)、(4)，第 3–5 页公式 (7)–(14)，第 4 页 Algorithm 2，第 6 页实验设置。[原文](https://arxiv.org/pdf/2006.11239v2)
>
> [2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer. “High-Resolution Image Synthesis with Latent Diffusion Models.” CVPR, 2022。arXiv:2112.10752v2，第 4–5 页介绍潜空间正则、条件注入与公式 (1)–(3)。[原文](https://arxiv.org/pdf/2112.10752v2)
>
> [3] Jiaming Song, Chenlin Meng, Stefano Ermon. “Denoising Diffusion Implicit Models.” ICLR, 2021。arXiv:2010.02502v4，第 4–6 页公式 (12) 与加速采样讨论。[原文](https://arxiv.org/pdf/2010.02502v4)
>
> [4] Jonathan Ho, Tim Salimans. “Classifier-Free Diffusion Guidance.” arXiv:2207.12598, 2022；短版发表于 NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications。第 4–5 页 Algorithm 1–2 与公式 (6)。[原文](https://arxiv.org/pdf/2207.12598)
>
> [5] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu. “DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.” NeurIPS, 2022。arXiv:2206.00927，第 3–5 页扩散常微分方程与高阶求解器。[原文](https://arxiv.org/pdf/2206.00927)
