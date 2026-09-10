# Diffusion Model 详解：从 DDPM 到 LDM 的数学推导

> 覆盖：DDPM 前向与反向过程、噪声预测目标、U-Net 条件注入、LDM 与采样加速

---

## 一、概率扩散模型的核心直觉

扩散模型的核心思想分两步：
1. **前向过程（Forward）**：向数据逐步添加高斯噪声，直到变成纯噪声
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

前向过程是一个**固定的马尔可夫链**，每一步向数据添加高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$

其中 $\beta_t \in (0, 1)$ 是 noise schedule（通常线性从 $\beta_1=10^{-4}$ 增加到 $\beta_T=0.02$）。

**重参数化技巧**：直接从 $x_0$ 一步计算 $x_t$（不需要迭代）：
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$$

即：
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### 2.2 反向过程

反向过程也是一个马尔可夫链，但需要学习：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})$$

DDPM 将 $\sigma_t^2$ 固定为 $\beta_t$ 或 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$（无学习），只学习 $\mu_\theta$。

### 2.3 训练目标

DDPM 发现，不直接预测 $\mu_\theta$，而是**预测噪声 $\epsilon$** 效果更好：

$$\mathcal{L}_{simple}(\theta) = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

其中：
- $x_0 \sim q(x_0)$：从数据分布采样
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：随机噪声
- $t \sim \text{Uniform}(1, ..., T)$：随机时间步
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

**为什么预测噪声比预测 $\mu$ 好**：
- $\mu$ 是噪声的线性变换，两者在数学上等价
- 但噪声的方差是 1，$\mu$ 的方差随时间 t 变化，直接预测噪声更容易（目标分布更统一）
- 这就是"preconditioning"的思想

### 2.4 DDPM 采样过程的完整推导

从预测的噪声 $\epsilon_\theta$ 到 $x_{t-1}$ 的均值和采样：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t \mathbf{I})$$

其中均值由预测噪声反推：
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

采样时（加上随机噪声以保持多样性）：
$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I}), \quad t > 0$$

---

## 三、U-Net 噪声预测网络

### 3.1 为什么用 U-Net

扩散模型的去噪任务本质上是**像素级预测**（逐像素预测噪声），需要：
- 大感受野（理解全局图像结构）
- 保持空间分辨率（精确定位噪声）
- U-Net 的 encoder-decoder + skip connection 完美满足

### 3.2 时间步 t 的注入方式

模型需要知道当前处于哪个扩散步（不同步的噪声水平不同）：

常见做法先把标量 $t$ 映射为正弦位置编码，再通过可学习映射得到时间嵌入 $e_t$。对第 $i$ 组频率：

$$
e_t^{(2i)}=\sin\left(t\omega_i\right),
\qquad
e_t^{(2i+1)}=\cos\left(t\omega_i\right).
$$

在 U-Net 中，时间嵌入通过**加或 scale+shift** 注入每个 ResNet block：

$$
h'=\gamma(e_t)\odot\operatorname{Norm}(h)+\beta(e_t).
$$

### 3.3 条件注入：Cross-Attention

对于**条件扩散模型**（如文本到图像），需要通过 cross-attention 注入条件：

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

LDM 的核心改进：**在 VAE 的 latent space 中做扩散**。

```
像素空间 (3×H×W)  →  VAE Encoder  →  latent space (4×H/8×W/8)
                                         ↓
                                    扩散/去噪过程（低维计算）
                                         ↓
                                    VAE Decoder  →  像素空间
```

- latent 通常是 4 通道，空间维度下采样 8×
- 如 512×512×3 图像 → 64×64×4 latent
- 计算量降低约 **48 倍**（512²×3 / 64²×4 ≈ 48）

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

## 五、扩散模型的三种采样加速方法

### 5.1 DDIM (Denoising Diffusion Implicit Models)

将 DDPM 的马尔可夫采样改为**非马尔可夫**确定性采样：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{"预测的 }x_0\text{"}} + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta$$

DDIM 允许**跳跃采样**（如只用 50 步而非 1000 步），大幅加速。

### 5.2 Classifier-Free Guidance (CFG)

在条件扩散中，通过混合条件和无条件预测来增强条件控制：

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

其中 $w$ 是 guidance scale（通常 7.5），$w>1$ 时放大条件信号。

### 5.3 DPM-Solver

将扩散采样视为常微分方程（ODE）求解，用高阶数值方法加速。

---

## 六、常见理论问题

### Q: 为什么扩散模型比 GAN 生成质量好？

**A**: 
- GAN 的 minimax 训练不稳定（模式崩塌、不收敛）
- 扩散模型是简单的回归目标（预测噪声），训练极其稳定
- 扩散模型逐步生成，每个步骤都是小幅修改，能更好的覆盖数据分布的细节
- 代价是推理速度慢（需要多步迭代）

### Q: 为什么预测噪声 ε 而不是直接预测 x₀？

**A**:
- 噪声的分布是标准正态 N(0,I)，方差恒定
- x₀ 的真实分布很复杂（自然图像的流形），且在时间步之间变化剧烈
- 预测噪声等价于预测 score function（score-based model 的角度），目标更均匀、更易优化

### Q: 如何处理条件扩散中的"无分类器引导"（CFG）？

**A**: 训练时以一定概率随机丢弃条件（设为空嵌入），让模型同时学习有条件和无条件去噪。推理时用 CFG 公式混合两种预测，放大条件信号。
