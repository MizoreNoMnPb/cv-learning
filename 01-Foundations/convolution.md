# 卷积详解：从数学、信号与系统到神经网络

> 覆盖：连续与离散卷积 -> 互相关 -> 线性时不变系统 -> 频域滤波 -> 多通道卷积 -> $3\times3$ 特征提取 -> 下采样与通道扩张 -> 转置卷积与上采样 -> $1\times1$ 分类头 -> 像素级 Softmax。  
> 关联阅读：[U-Net: Convolutional Networks for Biomedical Image Segmentation 精读](../02-Vision-Tasks/Segmentation/paper/%5BMICCAI%202015%5D%20U-Net%20-%20Convolutional%20Networks%20for%20Biomedical%20Image%20Segmentation/U-Net%20Convolutional%20Networks%20for%20Biomedical%20Image%20Segmentation.md)。

## 阅读建议

首次阅读可以先看第 0 节的计算链，再进入 [多通道卷积](#cnn-convolution)、[局部模式算例](#local-patterns)与 [下采样](#downsampling)。这条路线先回答“算了什么、形状怎样变化、为什么能提取局部特征”。

希望理解数学与信号系统时，再读第 1–2 节；希望理解分割网络时，继续读 [上采样](#upsampling)、[特征融合](#feature-fusion)、[分类输出](#classification-output)和第 9 节的完整尺寸示例。第 10–11 节用于检查设计选择与常见误解。

本文用 **[论文发现]** 标注原文报告的内容，用 **[数学推导]** 标注从明确前提推出的性质，用 **[分析判断]** 标注结构解释及应用判断。定义直接列出，连续使用同类证据时不重复解释标签。

---

## 0. 先建立完整计算链

一条典型的语义分割计算链是：

```text
输入图像 X: [B, C_in, H, W]
        |
        v
3x3 卷积 + 非线性：提取局部模式
        |
        v
下采样：减少 H、W，扩大感受野
        |
        v
增加通道：用更多特征图描述更抽象的模式
        |
        v
上采样：恢复空间采样网格
        |
        +------ 编码器高分辨率特征（skip connection，跳跃连接）
        |
        v
3x3 卷积：融合上下文与定位信息
        |
        v
1x1 卷积：把每个像素的 C 维特征映射为 K 个类别 logit
        |
        v
Softmax（沿类别轴）：得到每个像素上的 K 类概率
```

下文分别解释每一步的作用。首次阅读时，先区分空间尺寸、通道数与输出概率，再结合算例理解各运算。

### 0.1 本文统一记号

> LTI（Linear Time-Invariant，线性时不变）：既满足线性，又满足输入平移导致输出等量平移的一类系统。  
> CNN（Convolutional Neural Network，卷积神经网络）：使用局部连接和空间权重共享处理网格数据的神经网络。  
> MAC（Multiply-Accumulate，乘加）：一次乘法及其向累加和的加法，常用于统计卷积计算量。  
> FLOPs（Floating-Point Operations，浮点运算次数）：浮点加、减、乘等操作数量；不同工具是否把一个 MAC 计为 1 或 2 FLOPs 并不统一。

| 符号 | 含义 |
|---|---|
| $B$ | 批次大小 |
| $C_{\text{in}}$、$C_{\text{out}}$ | 输入、输出通道数 |
| $H,W$ | 输入特征图高、宽 |
| $K_h,K_w$ | 卷积核高、宽 |
| $S_h,S_w$ | stride（步幅） |
| $P_h,P_w$ | 两侧对称填充的大小 |
| $D_h,D_w$ | dilation（膨胀率） |
| $H_{\text{out}},W_{\text{out}}$ | 输出特征图高、宽 |
| $G$ | groups（分组数） |

“维度”这个词容易造成混乱。本文采用 $[B,C,H,W]$ 的批、通道、高、宽排列：

- tensor rank（张量阶数）始终是 4；
- 空间分辨率由 $H,W$ 表示；
- 通道维度由 $C$ 表示；
- 单个位置的特征维数是 $C$；
- 整张特征图的标量数量是 $C\times H\times W$。

所以“降分辨率、增加维度”应准确说成：**减小空间尺寸 $H,W$，同时增大通道数 $C$**。

---

## 1. 数学上的卷积是什么

### 1.1 连续时间卷积

**定义** 对两个连续函数 $x(t)$ 和 $h(t)$，卷积定义为：

$$
(x*h)(t)
=\int_{-\infty}^{+\infty}
x(\tau)h(t-\tau)\,d\tau.
$$

固定一个输出位置 $t$，计算过程可以拆成：

1. 将 $h(\tau)$ 翻转为 $h(-\tau)$；
2. 平移到 $h(t-\tau)$；
3. 与 $x(\tau)$ 逐点相乘；
4. 对所有 $\tau$ 积分。

卷积输出不是“某个位置的原值”，而是输入在一段邻域内与核函数的匹配结果。

### 1.2 离散时间卷积

**定义** 对离散序列 $x[n]$ 和 $h[n]$：

$$
(x*h)[n]
=\sum_{m=-\infty}^{+\infty}
x[m]h[n-m].
$$

若 $h$ 只有 $K$ 个非零元素，每个输出最多只需要 $K$ 次乘法。有限长度卷积核因此是一种局部运算。

### 1.3 二维卷积

图像是二维离散信号。**定义** 严格的二维卷积为：

$$
(X*K)[i,j]
=\sum_m\sum_n
X[m,n]K[i-m,j-n].
$$

等价地，在以 $(i,j)$ 为输出位置的局部窗口中，需要把卷积核在高、宽两个方向都翻转后再做点积。

### 1.4 深度学习中的“卷积”实际是互相关

> Cross-correlation（互相关）：不翻转卷积核，直接在输入上滑动并计算局部点积。

深度学习文献中称为“卷积”的二维算子通常计算：

$$
Y[i,j]
=\sum_m\sum_n
X[i+m,j+n]K[m,n].
$$

这在数学上是互相关，而不是上一节的严格卷积。[1, pp.3-5]

用一维 valid 运算看得最直接。设：

$$
x=[1,2,3,4],
\qquad
k=[1,0,-1].
$$

互相关直接使用 $k$：

$$
[1,2,3]\cdot[1,0,-1]=-2,
$$

$$
[2,3,4]\cdot[1,0,-1]=-2.
$$

若采用严格卷积，需要先把核翻转为 $[-1,0,1]$，在相同 valid 窗口记法下得到 $[2,2]$。

**[数学推导]** 对神经网络而言，卷积核参数是从数据中学习的。设严格卷积想学习核 $K$，互相关层只需学习旋转 $180^\circ$ 后的核：

$$
K'[m,n]=K[-m,-n].
$$

$K\leftrightarrow K'$ 是一一对应的参数重排，所以使用互相关不会减少可表示的局部线性映射集合。只有在手工指定 Sobel 等固定方向核时，是否翻转才会直接改变响应符号或方向。

### 1.5 线性与平移等变性

#### 线性

设局部卷积算子为 $\mathcal C_K$。由求和的分配律：

$$
\begin{aligned}
\mathcal C_K(aX_1+bX_2)
&=\sum_{m,n}(aX_1+bX_2)K\\
&=a\sum_{m,n}X_1K+b\sum_{m,n}X_2K\\
&=a\mathcal C_K(X_1)+b\mathcal C_K(X_2).
\end{aligned}
$$

因此单个不含激活函数的卷积层是线性映射；加上 bias 后是仿射映射。

#### 平移等变，而不是平移不变

定义平移算子：

$$
(T_\Delta X)[i,j]=X[i-\Delta_i,j-\Delta_j].
$$

对无限网格、stride 1 且没有边界截断的卷积：

$$
\mathcal C_K(T_\Delta X)
=T_\Delta\mathcal C_K(X).
$$

含义是：输入向右移动 3 个像素，输出特征图也向右移动 3 个像素。输出并没有保持不变，所以这是 **equivariance（等变性）**，不是 invariance（不变性）。

**[理论边界]** 零填充产生的人造边界、stride 大于 1 的采样相位、池化以及裁剪都会破坏严格平移等变性。现代 CNN 只能说具有局部权重共享带来的平移等变倾向，不能笼统地说“卷积天然平移不变”。

### 1.6 卷积的矩阵视角

对长度为 5 的输入和长度为 3 的一维 valid 互相关：

$$
x=
\begin{bmatrix}
x_0&x_1&x_2&x_3&x_4
\end{bmatrix}^{\mathsf T},
$$

可以写为：

$$
y=Cx,
$$

其中：

$$
C=
\begin{bmatrix}
w_0&w_1&w_2&0&0\\
0&w_0&w_1&w_2&0\\
0&0&w_0&w_1&w_2
\end{bmatrix}.
$$

这个矩阵具有两个特征：

- **局部连接**：每行只有相邻的 3 个非零权重；
- **权重共享**：每一行重复使用同一组 $w_0,w_1,w_2$。

实际实现不会显式构造这个巨大稀疏矩阵，但该视角解释了三件事：

1. 卷积本质上仍是线性代数；
2. 权重共享为何显著减少参数；
3. 转置卷积为什么对应 $C^{\mathsf T}$，而不对应 $C^{-1}$。

---

## 2. 信号与系统角度：卷积为什么代表滤波

### 2.1 LTI 系统由冲激响应完全决定

> Impulse response（冲激响应）：系统对单位冲激输入的输出；在离散系统中记为 $h[n]=\mathcal H\{\delta[n]\}$。

任意离散信号都可以分解为平移冲激的加权和：

$$
x[n]
=\sum_m x[m]\delta[n-m].
$$

设系统 $\mathcal H$ 同时满足线性和时不变性。由线性：

$$
\mathcal H\{x[n]\}
=\sum_m x[m]\mathcal H\{\delta[n-m]\}.
$$

由时不变性，冲激平移多少，响应就平移多少：

$$
\mathcal H\{\delta[n-m]\}=h[n-m].
$$

因此：

$$
\boxed{
y[n]
=\mathcal H\{x[n]\}
=\sum_m x[m]h[n-m]
=(x*h)[n]
}
$$

这不是类比，而是从 LTI 两条定义推出的结果：**任何 LTI 系统都可以由输入与冲激响应的卷积描述**。

对图像而言，若一个二维滤波器在所有位置使用相同规则，那么卷积核就是它的二维冲激响应。

### 2.2 频域中的卷积定理

> DTFT（Discrete-Time Fourier Transform，离散时间傅里叶变换）：把离散信号表示为不同角频率复指数的叠加。

定义：

$$
X(e^{j\omega})
=\sum_n x[n]e^{-j\omega n},
\qquad
H(e^{j\omega})
=\sum_n h[n]e^{-j\omega n}.
$$

已知 $y[n]=\sum_m x[m]h[n-m]$，对其做 DTFT：

$$
\begin{aligned}
Y(e^{j\omega})
&=\sum_n\sum_m x[m]h[n-m]e^{-j\omega n}.
\end{aligned}
$$

令 $r=n-m$，即 $n=r+m$：

$$
\begin{aligned}
Y(e^{j\omega})
&=\sum_m x[m]\sum_r h[r]e^{-j\omega(r+m)}\\
&=\left(\sum_m x[m]e^{-j\omega m}\right)
  \left(\sum_r h[r]e^{-j\omega r}\right)\\
&=X(e^{j\omega})H(e^{j\omega}).
\end{aligned}
$$

所以：

$$
\boxed{
x*h
\quad\Longleftrightarrow\quad
X(e^{j\omega})H(e^{j\omega})
}
$$

**[数学推导]** 空间域中的卷积，等价于频域中让输入的每个频率分量乘上 $H(e^{j\omega})$：

- $|H|$ 决定各频率被放大还是抑制；
- $\arg H$ 决定各频率的相位变化；
- 保留低频、抑制高频得到平滑；
- 抑制直流和低频、突出快速变化得到边缘响应。

二维图像的推导完全相同。二维卷积核的频率响应为：

$$
H(\omega_x,\omega_y)
=\sum_m\sum_n
h[m,n]e^{-j(\omega_xm+\omega_yn)}.
$$

对多通道线性卷积层，在忽略有限图像边界的条件下：

$$
Y_o(\omega_x,\omega_y)
=\sum_{c=1}^{C_{\text{in}}}
H_{o,c}(\omega_x,\omega_y)
X_c(\omega_x,\omega_y).
$$

**[数学推导]** 因此卷积层在每个空间频率上既做频率选择，也通过 $H_{o,c}$ 做输入通道到输出通道的线性混合。$1\times1$ 卷积没有空间位移项，其 $H_{o,c}$ 不随 $(\omega_x,\omega_y)$ 变化，所以它能混合通道，却不能在当前层单独实现具有空间频率选择性的滤波。

> DFT（Discrete Fourier Transform，离散傅里叶变换）：把有限离散序列映射为有限个离散频率系数。  
> FFT（Fast Fourier Transform，快速傅里叶变换）：高效计算 DFT 的一类算法，而不是一种不同的变换。

卷积定理也给出另一条计算路线：先对输入和核做 DFT，在频域逐元素相乘，再做逆 DFT。有限长度 DFT 默认对应 circular convolution（循环卷积）；若要得到长度分别为 $N,K$ 的线性卷积，两个序列至少应补零到 $N+K-1$，避免首尾环绕。对很大的核或全局卷积，FFT 路线的渐近计算量可能低于空间域直接滑窗；对 CNN 中常见的 $3\times3$ 小核，频域变换与 padding 本身也有计算代价。算法选择不改变卷积的数学定义。

### 2.3 用核系数判断滤波倾向

#### 均值核：低通平滑

$$
K_{\text{mean}}
=\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}.
$$

所有系数之和为 1，因此常量输入 $X[i,j]=c$ 经过它仍输出 $c$。局部快速变化被邻域平均削弱，所以它具有低通倾向。

#### Sobel $x$ 方向导数核：突出左右变化

$$
K_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}.
$$

系数之和为 0，所以常量区域响应为 0。它计算左右方向的亮度差，因而主要响应垂直边缘。这里“$x$ 方向梯度”和“垂直边缘”描述的是两个不同对象，不能混为一谈。

#### Laplacian 核：二阶变化

$$
K_{\text{lap}}=
\begin{bmatrix}
0&1&0\\
1&-4&1\\
0&1&0
\end{bmatrix}.
$$

其系数和同样为 0。中心与四邻域相近时响应小，中心相对邻域快速凸起或凹陷时响应大。

### 2.4 “CNN 卷积核是滤波器”成立到什么程度

**[分析判断]** 一个固定卷积层在激活函数之前，确实可以看成一组多输入、多输出线性滤波器。

但完整 CNN 通常包含 ReLU、归一化、池化、门控和多层复合，因此整体不再是 LTI 系统。尤其是非线性激活会产生新的频率成分，所以不能为整个 CNN 定义一个与输入无关的单一频率响应。

准确表述是：

- 单个固定的线性卷积层可以用滤波器和频率响应分析；
- 经过训练的卷积核可能表现出低通、带通、方向选择或通道组合倾向；
- 多层非线性 CNN 学到的是输入相关的分段线性映射，不能简化为一个固定 Sobel 或高斯滤波器。

### 2.5 边界条件是运算定义的一部分

卷积窗口移到图像边缘时，窗口的一部分会落到图像外。常见处理包括：

| 方法 | 图像外数值 | 直接影响 |
|---|---|---|
| valid / padding 0 | 不计算缺少上下文的位置 | 输出缩小，只保留完整窗口 |
| zero padding | 填 0 | 尺寸可保持，但边界产生人造强度突变 |
| reflect padding | 镜像反射 | 边界通常更连续，但引入对称假设 |
| replicate padding | 复制边缘值 | 避免跳到 0，但可能形成平台 |
| circular padding | 从另一侧循环取值 | 适合周期信号，不适合多数普通图像 |

**[分析判断]** 工业图像中若缺陷可能紧贴视野边缘，padding 策略会直接影响边界异常分数。模型在零填充边缘学到的响应，可能反映“图像结束了”，而不是“出现了缺陷”。

### 2.6 下采样为什么会混叠

> Aliasing（混叠）：采样率不足时，不同高频信号在采样后变成相同或错误的低频外观，因而无法从采样结果唯一恢复。

一维按 2 倍下采样为：

$$
y[n]=x[2n].
$$

它只保留偶数位置。如果输入交替为：

$$
x=[1,-1,1,-1,\ldots],
$$

那么只取偶数位置后可能得到常量序列 $[1,1,1,\ldots]$，原来的最高频振荡被误认为直流分量。

**[分析判断]** 经典采样理论要求下采样前先低通，去除新的 Nyquist frequency（奈奎斯特频率，即该采样率下可无混叠表示的最高频率）以上的成分。CNN 的 stride-2 卷积或池化不一定严格满足这一条件；而 ReLU 产生的特征也不保证带限。因此，下采样造成的信息丢失和对像素级平移的敏感性是实际问题。抗混叠下采样工作正是针对这一点。[10, pp.1-4]

---

<a id="cnn-convolution"></a>

## 3. 神经网络中的多通道卷积

### 3.1 一个输出像素到底算了什么

> NCHW（Batch, Channel, Height, Width，批、通道、高、宽）：一种按批、通道、高、宽排列四阶图像张量的记号。

输入张量采用 NCHW 排列：

$$
X\in\mathbb R^{B\times C_{\text{in}}\times H\times W}.
$$

标准二维卷积权重为：

$$
W\in
\mathbb R^{C_{\text{out}}\times C_{\text{in}}\times K_h\times K_w},
$$

bias 为：

$$
\beta\in\mathbb R^{C_{\text{out}}}.
$$

忽略 groups 时，一个输出元素为：

$$
\begin{aligned}
Z[b,o,i,j]
=\beta[o]
&+\sum_{c=0}^{C_{\text{in}}-1}
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
W[o,c,u,v]\\
&\qquad\cdot
X[b,c,iS_h+uD_h-P_h,\ jS_w+vD_w-P_w].
\end{aligned}
$$

越界位置按照 padding 规则处理。这个公式说明：

- 一个输出通道 $o$ 对应一个形状为 $C_{\text{in}}\times K_h\times K_w$ 的卷积核；
- 它同时查看全部输入通道，而不是只查看一个通道；
- 同一组权重在所有空间位置共享；
- $C_{\text{out}}$ 个不同卷积核产生 $C_{\text{out}}$ 张输出特征图。

例如，一个输入通道数为 64、输出通道数为 128、核尺寸为 $3\times3$ 的卷积层拥有 128 个卷积核，每个核的形状是 $64\times3\times3$，不是“64 个 $3\times3$ 核”。

### 3.2 输出尺寸公式及完整推导

dilation 为 $D$ 时，长度为 $K$ 的核实际覆盖：

$$
K_{\text{eff}}=D(K-1)+1
$$

个输入位置跨度。

以高度方向为例，padding 后可用长度为 $H+2P_h$。第 $q$ 个输出窗口从 $qS_h$ 开始，窗口必须满足：

$$
qS_h+K_{\text{eff}}\le H+2P_h.
$$

因此最大合法索引为：

$$
q_{\max}
=\left\lfloor
\frac{H+2P_h-K_{\text{eff}}}{S_h}
\right\rfloor.
$$

索引从 0 开始，所以输出数量为 $q_{\max}+1$：

$$
\boxed{
H_{\text{out}}
=\left\lfloor
\frac{H+2P_h-D_h(K_h-1)-1}{S_h}
\right\rfloor+1
}
$$

宽度同理：

$$
\boxed{
W_{\text{out}}
=\left\lfloor
\frac{W+2P_w-D_w(K_w-1)-1}{S_w}
\right\rfloor+1
}
$$

该推导与常见卷积算术总结一致。[1, pp.3-10]

常见 $3\times3$、dilation 1 情况：

| kernel | padding | stride | dilation | 高度变化 |
|---:|---:|---:|---:|---:|
| 3 | 0 | 1 | 1 | $H\to H-2$ |
| 3 | 1 | 1 | 1 | $H\to H$ |
| 3 | 1 | 2 | 1 | $H\to\lceil H/2\rceil$ |
| 3 | 2 | 1 | 2 | $H\to H$ |

原始 U-Net 的第一次编码块正是：

$$
572\xrightarrow{3\times3,\ P=0}570
\xrightarrow{3\times3,\ P=0}568.
$$

每次卷积是上下各少 1、左右各少 1，所以高宽各少 2。[5, pp.2,4]

### 3.3 参数量、MAC 与内存

标准卷积参数量为：

$$
N_{\text{param}}
=C_{\text{out}}
\left(
\frac{C_{\text{in}}}{G}K_hK_w
+\mathbf{1}_{\text{bias}}
\right).
$$

其中 $\mathbf{1}_{\text{bias}}=1$ 表示启用 bias，关闭 bias 时取 0。

单张输出特征图的 MAC 数近似为：

$$
N_{\text{MAC}}
=H_{\text{out}}W_{\text{out}}C_{\text{out}}
\frac{C_{\text{in}}}{G}K_hK_w.
$$

batch 为 $B$ 时再乘 $B$。若一个工具把一次乘法和一次加法分别计数，则 FLOPs 约为 $2N_{\text{MAC}}$；报告性能时必须说明口径。

例如，输入通道数为 64、输出通道数为 128、核尺寸为 $3\times3$ 且不计 bias 时：

$$
N_{\text{param}}=128\times64\times3\times3=73{,}728.
$$

若输出为 $128\times128$：

$$
N_{\text{MAC}}
=128\times128\times128\times64\times9
\approx1.208\times10^9.
$$

参数量与输入图像尺寸无关，但计算量和激活内存随空间尺寸增长。

### 3.4 卷积的三个主要归纳偏置

> Inductive bias（归纳偏置）：模型结构预先假定哪些规律更可能成立，从而限制搜索空间。

#### 局部性

一个 $3\times3$ 卷积只直接查看局部邻域，假设相近像素之间的关系比任意远处像素更重要。

#### 权重共享

同一个核在整张图滑动，假设一种局部模式无论出现在左上角还是右下角，都应使用相同检测规则。

LeNet-5 对局部感受野、共享权重和逐层下采样的早期 CNN 结构给出了系统实例。[2, pp.2290-2295]

#### 层级组合

浅层局部模式经过多层组合形成更大范围模式。这个过程扩大感受野，但是否得到“边缘 -> 纹理 -> 部件 -> 物体”的整齐语义层级取决于数据、目标和优化，不能把它当作每个 CNN 必然出现的固定顺序。

### 3.5 stride、padding、dilation 和 groups 分别控制什么

| 参数 | 控制对象 | 不直接控制什么 |
|---|---|---|
| kernel size | 每次局部采样范围 | 输出通道数 |
| stride | 卷积窗口移动间隔、输出采样密度 | 卷积核数量 |
| padding | 边界上下文和输出尺寸 | 感受野内部权重值 |
| dilation | 核内采样点间隔、有效覆盖范围 | 采样点数量 |
| groups | 输入输出通道连接方式 | 空间分辨率 |
| $C_{\text{out}}$ | 输出特征图数量 | 空间采样间隔 |

#### dilation

$3\times3$ 核在 dilation 2 时，采样位置之间隔一个像素，有效覆盖范围为：

$$
K_{\text{eff}}=2(3-1)+1=5.
$$

它以 9 个参数覆盖 $5\times5$ 范围，但没有观察其中所有 25 个位置。膨胀卷积可在不继续下采样的情况下扩大上下文，连续使用不合适的 dilation 模式也可能产生稀疏网格效应。[8, pp.1-3]

#### grouped convolution

当 $G>1$ 时，输入与输出通道被分组，每个输出通道只连接 $C_{\text{in}}/G$ 个输入通道。它减少参数和计算，但同时限制跨组信息交换。

#### depthwise convolution

> Depthwise convolution（逐通道卷积）：令 $G=C_{\text{in}}$，每个输入通道独立进行空间卷积。

若 depth multiplier（深度倍率，即每个输入通道产生的输出通道数）为 1，则 depthwise $3\times3$ 参数量只有：

$$
9C_{\text{in}}.
$$

随后通常用 pointwise convolution（逐点卷积），即 $1\times1$ 卷积，进行通道混合：

$$
C_{\text{in}}C_{\text{out}}.
$$

合计：

$$
9C_{\text{in}}+C_{\text{in}}C_{\text{out}},
$$

小于标准 $3\times3$ 卷积的 $9C_{\text{in}}C_{\text{out}}$。这是 MobileNet 类轻量网络的核心计算分解之一。[7, p.3]

---

<a id="local-patterns"></a>

## 4. $3\times3$ 卷积核如何提取特征

### 4.1 本质是局部向量内积

先看单通道 $3\times3$ patch：

$$
P=
\begin{bmatrix}
p_{00}&p_{01}&p_{02}\\
p_{10}&p_{11}&p_{12}\\
p_{20}&p_{21}&p_{22}
\end{bmatrix},
$$

卷积核为：

$$
K=
\begin{bmatrix}
k_{00}&k_{01}&k_{02}\\
k_{10}&k_{11}&k_{12}\\
k_{20}&k_{21}&k_{22}
\end{bmatrix}.
$$

将它们拉平成 9 维向量，卷积响应就是：

$$
z=\langle\operatorname{vec}(P),\operatorname{vec}(K)\rangle+b
=\sum_{u=0}^{2}\sum_{v=0}^{2}p_{uv}k_{uv}+b.
$$

如果 patch 的数值排列与核的正负权重模式一致，内积绝对值就大；如果彼此抵消，响应就小。卷积核因此可以看成一个局部模式模板。

### 4.2 一个边缘响应的数值例子

考虑左暗右亮的 patch：

$$
P=
\begin{bmatrix}
0&0&1\\
0&0&1\\
0&0&1
\end{bmatrix},
$$

以及 Sobel $x$ 方向核：

$$
K_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}.
$$

互相关响应为：

$$
z=1+2+1=4.
$$

若 patch 是常量 $P_{uv}=c$，则：

$$
z=c\sum_{u,v}K_x[u,v]=0.
$$

所以这个核抑制均匀区域，突出左右亮度变化。把核取负，响应符号反转，便偏好相反方向的边缘。

### 4.3 学习卷积核不是手工选择卷积核

训练时，卷积权重由损失函数的梯度更新：

$$
W\leftarrow W-\eta\frac{\partial L}{\partial W},
$$

其中 $\eta$ 是学习率。对于单个响应：

$$
z=\sum_{u,v}W[u,v]X[u,v],
$$

有：

$$
\frac{\partial z}{\partial W[u,v]}=X[u,v].
$$

根据链式法则：

$$
\frac{\partial L}{\partial W[u,v]}
=\frac{\partial L}{\partial z}X[u,v].
$$

同一个权重在所有空间位置共享，因此它的总梯度是所有使用位置贡献之和：

$$
\frac{\partial L}{\partial W[u,v]}
=\sum_{i,j}
\frac{\partial L}{\partial Z[i,j]}
X[i+u,j+v].
$$

**[数学推导]** 如果某类局部排列经常能降低任务损失，与它相关的权重就会被反复沿一致方向更新。这就是卷积核从数据中形成方向、纹理、颜色组合或更抽象响应的基本机制。

**[经验观察]** 可视化研究发现，很多自然图像 CNN 的第一层会出现类似边缘、颜色对比和方向选择的核；更深层特征需要结合激活最大化或反投影解释。[12, pp.818-827] 这是一类训练结果，不是 $3\times3$ 结构在数学上保证的固定语义。

### 4.4 多通道核提取的是“空间模式 + 通道组合”

当输入有 $C_{\text{in}}$ 个通道时，一个输出通道计算：

$$
z_o[i,j]
=\sum_{c=1}^{C_{\text{in}}}
\left\langle
P_c[i,j],K_{o,c}
\right\rangle+b_o.
$$

因此一个 $3\times3$ 核并非只有 9 个权重，而是有 $9C_{\text{in}}$ 个空间-通道权重。它可以同时表达：

- 某个通道上的局部梯度；
- 两个通道之间的相反响应；
- 多个上游特征共同出现时才激活的组合；
- 对某些输入通道完全忽略。

以 RGB 输入为例，一个核可能对 R 通道中心给正权、对 G 通道给负权，从而检测颜色对比；深层输入通道不再对应固定颜色，而对应上一层学到的特征。

### 4.5 一个卷积层仍然只是线性/仿射映射

如果连续叠加 stride-1 卷积但不加入非线性，并暂时忽略有限边界处理：

$$
Y=\mathcal C_{K_2}\left(\mathcal C_{K_1}(X)\right),
$$

两个线性局部算子的复合仍可写成一个更大核的线性局部算子：

$$
Y=\mathcal C_{K_{\text{eff}}}(X).
$$

多层仍可合并成一个更大的线性卷积核；它不会形成真正的分段选择能力。

> ReLU（Rectified Linear Unit，修正线性单元）：逐元素计算 $\max(0,z)$，将负响应截断为 0。

加入 ReLU：

$$
Y=K_2*\operatorname{ReLU}(K_1*X+b_1)+b_2,
$$

就不能再合并为一个固定线性核。第一层核的响应是否大于 0 会改变后续有效计算路径，网络因而成为分段线性函数。

**[分析判断]** “卷积提取特征”实际是三部分共同完成的：

1. 卷积计算局部线性响应；
2. bias 和归一化调整响应基准与尺度；
3. 非线性决定哪些响应继续传递并允许多层组合。

### 4.6 为什么常用 $3\times3$

连续两个 stride-1 的 $3\times3$ 卷积具有 $5\times5$ 理论感受野：

$$
3+(3-1)=5.
$$

连续三个具有 $7\times7$ 理论感受野：

$$
3+(3-1)+(3-1)=7.
$$

若输入输出通道都为 $C$，一个 $5\times5$ 卷积参数量为：

$$
25C^2.
$$

两个 $3\times3$ 卷积参数量为：

$$
2\times9C^2=18C^2.
$$

它不仅参数更少，中间还可以插入一次非线性。VGG（Visual Geometry Group，牛津大学视觉几何组）网络系统采用小 $3\times3$ 核堆叠，并明确讨论了用多个小核获得大感受野的参数和非线性优势。[3, pp.2-3]

但 $3\times3$ 不是无条件最优：

- 大核可以更直接聚合较远上下文；
- depthwise 大核的参数代价远低于标准大核；
- 高频细节任务可能需要避免过早扩大 stride；
- 参数量与乘加次数只描述计算复杂度的部分侧面，不能由此断言某种卷积在所有条件下都更快。

### 4.7 感受野的逐层推导

> Receptive field（感受野）：某个特征位置在理论上可能依赖的输入空间范围。

定义第 $l$ 层：

- $r_l$：理论感受野大小；
- $j_l$：相邻特征位置在原输入上的间隔，也称 jump；
- $K_{\text{eff},l}=D_l(K_l-1)+1$。

初始输入满足：

$$
r_0=1,
\qquad
j_0=1.
$$

每经过一层：

$$
\boxed{
j_l=j_{l-1}S_l
}
$$

$$
\boxed{
r_l=r_{l-1}+\left(K_{\text{eff},l}-1\right)j_{l-1}
}
$$

第二式的原因是：新层最左和最右采样点相差 $K_{\text{eff}}-1$ 个旧层间隔，每个旧层间隔对应原图上的 $j_{l-1}$ 个像素。

以“两次 $3\times3$ 卷积 + $2\times2$ 池化 + 两次 $3\times3$ 卷积”为例：

| 操作 | $r$ | $j$ |
|---|---:|---:|
| 输入 | 1 | 1 |
| 3×3, s=1 | 3 | 1 |
| 3×3, s=1 | 5 | 1 |
| 2×2 pool, s=2 | 6 | 2 |
| 3×3, s=1 | 10 | 2 |
| 3×3, s=1 | 14 | 2 |

下采样以后，同样一个 $3\times3$ 核会让原图感受野增加 $2j$，所以深层上下文扩张更快。

**[证据边界]** 理论感受野只说明“可能影响”。训练后各位置贡献通常并不均匀，中心区域影响可能更集中，这被称为 effective receptive field（有效感受野）。[13, pp.1-5]

---

<a id="downsampling"></a>

## 5. 如何降低分辨率，又为什么增加通道

### 5.1 三种常见下采样

#### stride-2 convolution

例如：


当 $H$ 为偶数时：

$$
H_{\text{out}}
=\left\lfloor\frac{H+2-3}{2}\right\rfloor+1
=\frac H2.
$$

卷积同时完成可学习滤波和降采样。

#### max pooling

$2\times2$ max pooling、stride 2 在每个窗口保留最大值：

$$
y[i,j]
=\max_{0\le u,v<2}
x[2i+u,2j+v].
$$

它没有卷积权重，强调局部强响应，但会丢弃其他数值。

#### average pooling

$$
y[i,j]
=\frac14\sum_{u=0}^{1}\sum_{v=0}^{1}
x[2i+u,2j+v].
$$

它在采样前带有简单低通作用，但也会削弱窄小峰值。

### 5.2 通道增加不是下采样的结果

下面两个操作完全可以独立设置：


stride 决定 $H,W$，卷积核数量决定 $C_{\text{out}}$。二者同时变化只是常见架构约定，不存在“面积缩小后信息自动进入通道”的机制。

### 5.3 为什么常见设计让通道翻倍

#### 原因一：增加特征类型

浅层可能只需描述有限的局部颜色和梯度。深层位置覆盖更大输入区域，需要同时表示更多组合模式，设计者通常为它分配更多输出核。

这是容量分配逻辑，不是“深层一定需要恰好两倍通道”的定律。

#### 原因二：控制激活内存

下采样前标量数量为：

$$
N_{\text{act}}=HWC.
$$

高宽减半、通道翻倍以后：

$$
N'_{\text{act}}
=\frac H2\frac W2(2C)
=\frac12HWC.
$$

所以通道翻倍后，激活数量实际上仍减半。

若只从标量数量看，要在空间面积变为 $1/4$ 后保持总数量不变，需要通道变成 $4C$：

$$
\frac H2\frac W2(4C)=HWC.
$$

即使总数量不变，也不代表映射可逆；权重结构和采样过程仍可能丢失信息。

#### 原因三：让同尺度卷积计算量大致稳定

在 $H\times W$ 上做 $C\to C$ 的 $3\times3$ 卷积：

$$
N_{\text{MAC}}=9HWC^2.
$$

下一尺度为 $H/2\times W/2$，通道为 $2C\to2C$：

$$
\begin{aligned}
N'_{\text{MAC}}
&=9\frac H2\frac W2(2C)(2C)\\
&=9HWC^2.
\end{aligned}
$$

因此“空间减半、通道翻倍”能让同类卷积层的 MAC 近似保持不变，同时扩大感受野并提高每个位置的表示维数。

### 5.4 下采样的信息损失不可忽略

从 $H\times W\times C$ 映射到 $H/2\times W/2\times2C$，输出标量只有输入的一半。对任意输入建立一一对应通常不可能。

更具体地看，stride 2 只在某个采样相位上产生输出。即使卷积窗口有重叠，映射仍可能把多个不同高频输入映射为相同输出。解码器后续扩大尺寸，并不意味着它恢复了原始像素；它依据低分辨率特征、训练先验和 skip feature 预测高分辨率结果。

### 5.5 对小目标和工业缺陷的影响

**[分析判断]** 一条只有 1 至 2 像素宽的划痕主要表现为局部高频变化。连续下采样可能使它：

- 在平均池化中被周围正常像素稀释；
- 在 stride 采样中落在不利相位而响应显著改变；
- 在深层特征图上小于一个位置；
- 最终只能由解码器根据弱证据“猜回去”。

因此工业小缺陷任务常需要验证：

- 是否保留足够高分辨率的中间特征；
- 第一次 stride-2 是否过早；
- 下采样前是否需要抗混叠滤波；
- skip connection 是否真的保留缺陷，而不是只带来背景纹理；
- 评估是否按缺陷尺寸分桶，而不是只看总体平均指标。

这些是由采样机制推出的实验假设，不是某个网络结构必然优于另一个结构的结论。

---

<a id="upsampling"></a>

## 6. 上采样与转置卷积

> Up-convolution（上采样卷积）：用于扩大特征图空间尺寸的可学习操作名称；不同论文可能用它指转置卷积，也可能指插值后接普通卷积，必须结合具体定义判断。

### 6.1 上采样只扩大网格，不自动恢复信息

设低分辨率特征为：

$$
X\in\mathbb R^{C\times H\times W}.
$$

上采样 2 倍产生：

$$
Y\in\mathbb R^{C'\times2H\times2W}.
$$

输出位置变多不等于信息变多。新增位置必须由以下来源估计：

- 固定插值规则；
- 可学习卷积参数；
- 编码器传来的高分辨率 skip feature；
- 训练数据提供的结构先验。

### 6.2 最近邻与双线性插值

#### nearest-neighbor interpolation

最近邻上采样直接复制最近的低分辨率值。2 倍上采样的一维示意为：

$$
[a,b,c]
\longrightarrow
[a,a,b,b,c,c].
$$

它不引入平滑，但可能产生块状边界。

#### bilinear interpolation

> Bilinear interpolation（双线性插值）：先在一个方向线性插值，再在另一个方向线性插值，以四个邻近采样点的距离权重估计新值。

它输出较平滑，但高频细节仍是插值结果，不是对真实细节的观测。

插值负责确定几何网格，后续卷积负责学习局部重建和通道变换。

### 6.3 转置卷积为什么叫“转置”

上一节把卷积写成：

$$
y=Cx.
$$

转置卷积对输入 $z$ 计算：

$$
x'=C^{\mathsf T}z.
$$

它叫 transpose convolution，是因为使用了卷积线性算子的转置矩阵。一般情况下：

$$
C^{\mathsf T}\ne C^{-1}.
$$

原因包括：

- $C$ 往往不是方阵；
- stride 大于 1 时已经减少输出维数；
- 即使是方阵，也不保证满秩或正交；
- $C^{\mathsf T}C$ 通常不等于单位矩阵。

所以“deconvolution（反卷积）”这个俗称容易误导。转置卷积不是在数学上撤销前一层卷积。

它与反向传播还有直接关系。若前向卷积为：

$$
y=Cx,
$$

标量损失为 $L(y)$，根据链式法则：

$$
\boxed{
\frac{\partial L}{\partial x}
=C^{\mathsf T}
\frac{\partial L}{\partial y}
}
$$

因此，固定同一组权重时，普通卷积对输入的反向传播使用的正是转置卷积型算子。这解释了名称来源，但仍不意味着它是 $C^{-1}$。

### 6.4 转置卷积输出尺寸

对一维或单个空间方向，转置卷积输出为：

$$
\boxed{
H_{\text{out}}
=(H_{\text{in}}-1)S
-2P
+D(K-1)
+P_{\text{out}}
+1
}
$$

其中 $P_{\text{out}}$ 是输出侧的尺寸修正量。该公式可以从前向卷积的输入输出尺寸关系反解候选输入尺寸得到；$P_{\text{out}}$ 用于区分 stride 大于 1 时多个输入尺寸映射到同一输出尺寸的歧义。[1, pp.19-26]

例如，取 $K=2$、$S=2$、$P=0$、$D=1$、$P_{\text{out}}=0$，则：

$$
H_{\text{out}}
=(H_{\text{in}}-1)2+2
=2H_{\text{in}}.
$$

因此 $28\to56$。

$P_{\text{out}}$ 只选择输出形状，不等价于在最终输出周围实际补一圈普通零值。

### 6.5 “插零再卷积”的等价视角

stride 为 2 的转置卷积可以理解为：

1. 在相邻输入值之间插入 1 个零；
2. 根据原前向卷积的 padding 关系处理边界；
3. 用学习到的核进行 stride-1 卷积式累加。

例如一维输入：

$$
[a,b,c]
\longrightarrow
[a,0,b,0,c]
\longrightarrow
\text{learned filtering}.
$$

中间的零不是最终希望输出的像素，而是描述转置线性算子如何把低分辨率值散布到高分辨率网格。

### 6.6 checkerboard artifact 从哪里来

> Checkerboard artifact（棋盘格伪影）：上采样输出中出现周期性交替的亮暗或响应强弱网格。

当 kernel size 不能被 stride 整除时，不同输出位置可能接收不同数量的卷积核重叠贡献。例如 $K=3,S=2$ 时，有些位置被更多窗口覆盖，初始和训练后的响应都可能不均匀。二维两个方向叠加后形成棋盘格。[9, “Overlap & Learning”]

降低风险的常见方式：

- 使用 kernel size 可被 stride 整除的配置，例如 $K=2,S=2$；
- 使用 bilinear/nearest interpolation 后接普通卷积；
- 可视化常量输入和真实特征的上采样响应；
- 不只观察最终损失，还检查频谱和局部周期纹理。

这只能降低结构性风险，不能保证模型绝不产生纹理伪影。

### 6.7 Pixel Shuffle

> Pixel Shuffle（像素重排）：先在低分辨率位置预测 $r^2$ 倍通道，再把这些通道按确定规则重排到 $r\times r$ 的空间子像素位置。

$$
[B,Cr^2,H,W]
\longrightarrow
[B,C,rH,rW].
$$

它本身只是无参数重排；产生 $Cr^2$ 通道的前置卷积负责学习各子像素值。它常见于超分辨率，也可用于其他密集预测 decoder。

### 6.8 原始 U-Net 的 up-convolution

**[论文发现]** 原始 U-Net 将扩张路径的一步描述为上采样后接 $2\times2$ convolution，并称为 up-convolution；该操作同时把通道数减半。随后把编码器特征裁剪到相同空间尺寸，在通道维拼接，再做两个 $3\times3$ valid convolution。[5, pp.2,4]

后续 U 形网络常用 $K=2,S=2$ 的转置卷积完成这一尺寸变化，也可能采用插值后接卷积。两者的参数化与频率响应不同，不能仅凭“up-convolution”这一名称判断数学操作。

---

<a id="feature-fusion"></a>

## 7. 解码器中的特征拼接与残差相加

> Concatenation（拼接）：沿指定张量维度并排保留多个张量。  
> Residual connection（残差连接）：把变换分支与恒等或投影分支逐元素相加。

设编码器特征为：

$$
E\in\mathbb R^{B\times C_e\times H\times W},
$$

解码器特征为：

$$
D\in\mathbb R^{B\times C_d\times H\times W}.
$$

### 7.1 concat

$$
Y_{\text{cat}}=\operatorname{Concat}(E,D)
\in\mathbb R^{B\times(C_e+C_d)\times H\times W}.
$$

它只要求空间尺寸一致，不要求 $C_e=C_d$。两份信息保留在不同通道，后续卷积学习如何融合；代价是后续卷积输入通道和计算量增加。

### 7.2 addition

若形状完全一致：

$$
Y_{\text{add}}=E+D
\in\mathbb R^{B\times C\times H\times W}.
$$

若通道不同，需要投影：

$$
Y_{\text{add}}=P(E)+D,
$$

其中 $P$ 常由 $1\times1$ 卷积实现。加法不增加通道，计算和内存更小，但它预设两个分支的对应通道可以直接叠加，且相加后无法再单独访问原始 $E,D$。

### 7.3 residual 的梯度路径

标准残差形式为：[6, p.3, Eq.(1)]

$$
Y=X+F(X).
$$

由链式法则：

$$
\frac{\partial L}{\partial X}
=\frac{\partial L}{\partial Y}
\left(
I+\frac{\partial F}{\partial X}
\right).
$$

$I$ 提供不经过 $F$ 内部层的直接梯度项。这是残差连接的优化意义。U-Net 的 encoder-to-decoder concat 重点则是多尺度特征融合；两者都跨层传信息，但数学操作和主要目的不同。

---

<a id="classification-output"></a>

## 8. $1\times1$ 卷积、logit 与 Softmax 输出

### 8.1 $1\times1$ 卷积在每个位置做通道线性变换

设 decoder 最终特征为：

$$
F\in\mathbb R^{B\times C\times H\times W}.
$$

固定像素 $(i,j)$，它对应一个 $C$ 维向量：

$$
f_{i,j}\in\mathbb R^C.
$$

使用 $K$ 个 $1\times1$ 卷积核，权重为：

$$
W_{\text{head}}\in\mathbb R^{K\times C},
\qquad
b_{\text{head}}\in\mathbb R^K.
$$

输出为：

$$
\boxed{
z_{i,j}
=W_{\text{head}}f_{i,j}+b_{\text{head}}
\in\mathbb R^K
}
$$

同一个 $W_{\text{head}}$ 在所有像素位置共享。因此，$1\times1$ 卷积等价于对每个像素的特征向量应用同一个线性分类器。

它有三个关键性质：

1. 不直接混合相邻空间位置；
2. 可以任意混合和改变通道数；
3. 输出位置仍继承上游网络已经建立的感受野。

第三点很重要：$1\times1$ 只表示当前这一层不扩大感受野，不表示最终预测只看一个原图像素。$f_{i,j}$ 可能已经包含很大范围的编码器上下文。

### 8.2 $1\times1$ 卷积的典型用途

#### 通道降维或升维

$$
[B,2048,H,W]
\xrightarrow{1\times1}
[B,256,H,W].
$$

参数量为 $2048\times256$，远小于同通道映射的 $3\times3$ 参数量 $9\times2048\times256$。

#### bottleneck（瓶颈映射）

先用 $1\times1$ 降通道，再做昂贵的空间卷积，最后升回通道，可以降低中间计算。是否损失信息取决于瓶颈宽度和训练结果。

#### segmentation head

若有 $K$ 个互斥语义类别：

$$
[B,C,H,W]
\xrightarrow{1\times1}
[B,K,H,W].
$$

全卷积网络与 U-Net 都使用卷积分类头产生空间类别分数。[4, pp.2-4; 5, pp.2,4]

### 8.3 logit 不是概率

> Logit（未归一化类别分数）：分类头直接输出的任意实数；不同类别 logit 的相对大小决定分类结果。

对像素 $(i,j)$：

$$
z_{i,j}=[z_1,z_2,\ldots,z_K].
$$

每个 $z_k$ 可以是负数，也不要求总和为 1。$1\times1$ 卷积完成的只是线性打分。

### 8.4 像素级 Softmax 如何参数化类别分布

对同一个像素的类别维进行：

$$
\boxed{
p_k(i,j)
=\frac{\exp(z_k(i,j))}
{\sum_{q=1}^{K}\exp(z_q(i,j))}
}
$$

于是：

$$
p_k(i,j)>0,
\qquad
\sum_{k=1}^{K}p_k(i,j)=1.
$$

Softmax 只在类别轴耦合 $K$ 个类别，不在空间轴归一化。对 NCHW 张量，类别轴是 $C$；若最后一维表示宽度 $W$，沿最后一维归一化会错误地让同一行的空间位置相互竞争。

**[论文发现]** 原始 U-Net 也在每个像素位置对最终通道使用 Softmax。[5, p.4]

<a id="softmax-stability"></a>

### 8.5 为什么计算时要减去最大 logit

设：

$$
m=\max_k z_k.
$$

则：

$$
\frac{e^{z_k-m}}{\sum_qe^{z_q-m}}
=\frac{e^{z_k}/e^m}{\sum_qe^{z_q}/e^m}
=\frac{e^{z_k}}{\sum_qe^{z_q}}.
$$

概率不变，但所有指数输入都不大于 0，可以避免 $e^{z_k}$ 在大正数时溢出。这是数值稳定的 Softmax 计算方式。

### 8.6 Softmax 与交叉熵的梯度

若真实类别为 $y$，单像素交叉熵为：

$$
L=-\log p_y.
$$

Softmax 的偏导为：

$$
\frac{\partial p_y}{\partial z_k}
=p_y(\mathbf{1}[k=y]-p_k).
$$

代入链式法则：

$$
\begin{aligned}
\frac{\partial L}{\partial z_k}
&=-\frac1{p_y}
\frac{\partial p_y}{\partial z_k}\\
&=p_k-\mathbf{1}[k=y].
\end{aligned}
$$

所以正确类别的梯度为 $p_y-1$，推动其 logit 上升；其他类别梯度为 $p_k$，推动它们下降。

交叉熵通常直接由 logits 通过 log-sum-exp 形式计算：

$$
L=-z_y+\log\sum_{k=1}^{K}e^{z_k}.
$$

该式与 $-\log p_y$ 等价，并可结合减去最大 logit 的技巧稳定计算；不需要先显式得到 Softmax 概率。

### 8.7 二分类：两个 Softmax logit 与一个 sigmoid logit

两个类别 logit 为 $z_0,z_1$ 时：

$$
p_1
=\frac{e^{z_1}}{e^{z_0}+e^{z_1}}
=\frac1{1+e^{-(z_1-z_0)}}
=\sigma(z_1-z_0).
$$

所以二类 Softmax 的概率只依赖 logit 差。可以选择：

- 输出 2 通道并使用 cross-entropy；
- 输出 1 通道并使用 binary cross-entropy with logits。

二者参数化不同但都能表达二分类。标签、输出通道和损失函数必须配套。

### 8.8 多标签为什么用 sigmoid

若一个像素可以同时属于多个非互斥标签，Softmax 的概率和为 1 会强迫标签竞争。此时对每个通道独立使用：

$$
p_k=\sigma(z_k)=\frac1{1+e^{-z_k}},
$$

并用逐通道二元交叉熵。语义分割中的互斥类别通常用 Softmax；实例 mask 或多标签属性则可能使用独立 sigmoid。不能仅凭“任务是分割”决定激活函数。

---

## 9. 从输入到分割结果：一条完整 U-Net-like 尺寸流

> U-Net-like：具有编码器、解码器和同尺度 skip fusion 的 U 形网络；不等同于 2015 年原始 U-Net。

下面使用现代常见的 same-padding 版本说明各操作职责。输入为：

$$
X\in\mathbb R^{B\times3\times256\times256}.
$$

每个 `DoubleConv` 包含两个 $3\times3$、stride 1、padding 1 卷积，因此空间尺寸不变。

| 阶段 | 操作 | 输出形状 |
|---|---|---|
| Input | RGB image | $[B,3,256,256]$ |
| Enc 0 | DoubleConv $3\to64$ | $[B,64,256,256]$ |
| Down 0 | Pool $2\times2$ | $[B,64,128,128]$ |
| Enc 1 | DoubleConv $64\to128$ | $[B,128,128,128]$ |
| Down 1 | Pool $2\times2$ | $[B,128,64,64]$ |
| Enc 2 | DoubleConv $128\to256$ | $[B,256,64,64]$ |
| Down 2 | Pool $2\times2$ | $[B,256,32,32]$ |
| Enc 3 | DoubleConv $256\to512$ | $[B,512,32,32]$ |
| Down 3 | Pool $2\times2$ | $[B,512,16,16]$ |
| Bottleneck | DoubleConv $512\to1024$ | $[B,1024,16,16]$ |
| Up 3 | Transposed conv $1024\to512$ | $[B,512,32,32]$ |
| Fuse 3 | concat Enc 3 | $[B,1024,32,32]$ |
| Dec 3 | DoubleConv $1024\to512$ | $[B,512,32,32]$ |
| Up 2 | Transposed conv $512\to256$ | $[B,256,64,64]$ |
| Fuse 2 | concat Enc 2 | $[B,512,64,64]$ |
| Dec 2 | DoubleConv $512\to256$ | $[B,256,64,64]$ |
| Up 1 | Transposed conv $256\to128$ | $[B,128,128,128]$ |
| Fuse 1 | concat Enc 1 | $[B,256,128,128]$ |
| Dec 1 | DoubleConv $256\to128$ | $[B,128,128,128]$ |
| Up 0 | Transposed conv $128\to64$ | $[B,64,256,256]$ |
| Fuse 0 | concat Enc 0 | $[B,128,256,256]$ |
| Dec 0 | DoubleConv $128\to64$ | $[B,64,256,256]$ |
| Head | $1\times1$, $64\to K$ | $[B,K,256,256]$ |
| Probability | Softmax over $K$ | $[B,K,256,256]$ |

这条链中各部件的职责不能互换：

- $3\times3$ 卷积：局部空间与通道特征提取；
- pooling/stride：改变采样密度并扩大后续 jump；
- 增加通道：分配更多特征类型和容量；
- upsampling：恢复输出网格；
- concat：重新引入高分辨率编码特征；
- decoder convolution：融合 skip 与深层上下文；
- $1\times1$ convolution：把每个位置的 64 维特征映射为 $K$ 个类别分数；
- Softmax：只负责把 $K$ 个分数变成互斥类别概率。

### 9.1 原始 U-Net 为什么还需要 crop

**[论文发现]** 2015 原始 U-Net 的 $3\times3$ 卷积均为 padding 0。编码器特征在每次卷积后缩小；decoder 上采样后的特征与对应 encoder feature 尺寸不同，所以必须从 encoder feature 中央裁剪后再 concat。[5, pp.2,4]

例如：

$$
568\times568
\xrightarrow{\text{center crop}}
392\times392,
$$

才能与最后一次上采样得到的 $392\times392$ decoder feature 拼接。现代 padding 1 版本通常不需要这种 crop，但它已经改变了原始 U-Net 的边界行为。

### 9.2 decoder 为什么离不开 encoder 证据

低分辨率 bottleneck 具有较大上下文，但精确位置已经被下采样压缩。仅靠上采样只能生成更密的网格，不能知道原图某条边缘究竟位于两个低分辨率采样点之间的哪里。

skip feature 提供高分辨率候选位置，decoder feature 提供“这是什么”的上下文。concat 后的卷积再决定：

$$
\text{high-resolution location}
+\text{low-resolution context}
\longrightarrow
\text{dense prediction}.
$$

这解释了 U-Net 的结构动机，但不保证所有 skip feature 都有益；浅层背景纹理也可能被一并传入 decoder。

---

## 10. 卷积设计对训练与输出的具体影响

### 10.1 padding 改变的不只是尺寸

padding 0、1 都可能得到可训练模型，但它们给模型提供的边界证据不同：

- padding 0：不预测缺少完整上下文的位置，输出不断缩小；
- zero padding 1：保持尺寸，但模型可以感知到零边界；
- reflect padding：引入镜像连续假设；
- 非对称 padding：可能造成输出坐标偏移。

对定位任务必须同时考察输出尺寸和坐标对齐，不能只判断运算是否有有效输出。

### 10.2 偶数卷积核的 same padding 可能不对称

stride 1 时若希望保持尺寸，需要总 padding：

$$
P_{\text{total}}=K_{\text{eff}}-1.
$$

$K=3$ 时总 padding 为 2，可以左右各 1；$K=2$ 时总 padding 为 1，无法对称分到两侧。此时任何保持尺寸的补齐规则都必须在一侧补 0、另一侧补 1，这会影响特征中心对齐。

### 10.3 初始化决定深层信号是否爆炸或衰减

设一个卷积输出神经元的 fan-in 为：

$$
N=\frac{C_{\text{in}}}{G}K_hK_w.
$$

假设输入独立、零均值、二阶矩为 $q$，权重独立、零均值、方差为 $\sigma_w^2$。卷积前激活二阶矩近似为：

$$
N\sigma_w^2q.
$$

若激活分布近似关于 0 对称，ReLU 约保留一半二阶矩：

$$
q_{\text{out}}
\approx\frac12N\sigma_w^2q.
$$

令输入输出尺度近似相同：

$$
\frac12N\sigma_w^2q=q,
$$

得到：

$$
\boxed{
\sigma_w^2=\frac2N
}
$$

这就是 ReLU 网络常用 He initialization（何恺明初始化）的核心方差推导；原论文从前向/反向信号方差角度给出相应初始化方法。[11, pp.4-5]

### 10.4 normalization 会改变“卷积输出”的统计含义

若卷积后立即归一化，卷积响应的绝对均值和尺度会被重新调整。此时单独查看 kernel 系数不足以判断最终激活强弱，还要看：

- normalization 的统计量；
- 可学习 scale 和 bias；
- 后续激活函数；
- train/eval 模式差异。

小 batch 分割训练中，BatchNorm 的 batch 统计可能噪声较大；GroupNorm 等方法不依赖 batch 维统计，但也会改变归一化假设。选择应由 batch 大小和验证实验决定。

### 10.5 更大感受野不等于更精确定位

下采样和深层卷积扩大上下文，却同时降低空间采样密度。对分类，这通常可以接受；对分割、关键点和微小缺陷，定位精度可能下降。

因此 dense prediction（密集预测）经常结合：

- 高分辨率 skip feature；
- 多尺度 feature pyramid；
- dilation 保持分辨率；
- decoder 上采样；
- 边界或区域损失。

这些机制是在上下文范围、中间激活规模、计算量和定位精度之间做不同取舍，并不存在免费恢复细节的方法。

### 10.6 softmax 概率不等于可信度

Softmax 保证输出为正且和为 1，但不保证概率已经校准，也不保证输入来自训练分布。模型遇到训练分布之外的输入时，仍可能给出高度集中的类别概率。所有 logit 同时增加相同常数不会改变 Softmax；关键是类别分数之间的差异，推导见第 8.5 节。

**[分析判断]** Softmax 是参数化归一化，不是异常检测器，也不是不确定性证明。在分布偏移条件下，概率校准与拒识能力需要作为独立性质分析。

## 11. 阅读后的自查问题

### Q1：为什么原始 U-Net 的一次 $3\times3$ conv 让尺寸减少 2？

因为 $K=3,S=1,P=0,D=1$：

$$
H_{\text{out}}
=H-3+1
=H-2.
$$

本质是卷积中心不能位于最外侧一圈；上、下各少 1，左、右各少 1。

### Q2：为什么 $3\times3$ 能提取边缘？

只有当核权重学成或被设成“邻域一侧为负、另一侧为正、系数和接近 0”的差分模式时，它才会突出边缘。$3\times3$ 只提供局部参数化空间，不自动等于边缘检测器。

### Q3：为什么降采样后通道经常翻倍？

这是容量和计算预算设计：空间面积变为 $1/4$ 后，通道从 $C$ 变为 $2C$，激活量减半；同尺度 $3\times3$ 的 $2C\to2C$ 计算量则与上一级 $C\to C$ 近似相同。它不是下采样的数学必然结果。

### Q4：转置卷积是不是把 encoder 卷积反过来？

不是。它计算卷积矩阵的转置 $C^{\mathsf T}$，不是逆矩阵 $C^{-1}$。它恢复空间尺寸，但丢失内容只能依靠学习先验和 skip feature 估计。

### Q5：$1\times1$ 卷积为什么能分类？

因为每个空间位置已有一个 $C$ 维特征向量，$K$ 个 $1\times1$ 核正好组成 $K\times C$ 的线性分类器，将该向量映射为 $K$ 个类别 logit。

### Q6：为什么分割 Softmax 要沿类别轴计算？

因为每个空间位置都需要一个独立的类别分布。对 NCHW 张量，需要保证每个 $(b,i,j)$ 上：

$$
\sum_{k=1}^{K}p[b,k,i,j]=1.
$$

对宽度或高度做 Softmax 会变成像素位置之间竞争，含义完全不同。

### Q7：concat 与 residual addition 的最短区别是什么？

concat 把两份特征保留为更多通道，后续层再学习融合；addition 直接把对应元素合成一份，要求通道语义和形状能够对齐。标准 residual 还提供恒等梯度路径。

---

## 12. 结论

1. **定义** 卷积是翻转、平移、逐点乘积与求和；深度学习中的二维“卷积”通常采用不翻核的互相关定义。
2. **[数学推导]** LTI 系统输出必然可写成输入与冲激响应的卷积；空间卷积在频域对应逐频率相乘。
3. **[分析判断]** 神经网络中的特征提取来自卷积局部响应、跨通道组合、非线性和多层训练共同作用，而不是 $3\times3$ 核自动拥有某种语义。
4. **[数学推导]** 输出尺寸由 kernel、padding、stride、dilation 决定；输出通道数由卷积核数量决定，二者独立。
5. **[分析判断]** 空间减半、通道翻倍能在扩大上下文时控制激活量，并使同尺度卷积计算量大致稳定，但会丢失高分辨率信息。
6. **[数学推导]** 转置卷积是卷积线性算子的转置，不是一般意义上的逆；上采样输出是重建或预测，不是原像素的自动恢复。
7. **[数学推导]** $1\times1$ 卷积是共享的逐像素通道线性变换；Softmax 再把类别 logit 变成互斥概率。
8. **[分析判断]** 小缺陷、高频纹理和精确边界对 stride、padding、抗混叠与 skip feature 特别敏感，必须按尺寸和位置进行针对性验证。

---

## 论文与资料引用

> [1] Vincent Dumoulin, Francesco Visin. “A Guide to Convolution Arithmetic for Deep Learning.” arXiv:1603.07285, 2016, revised 2018, public version pp.1-31. 普通卷积算术见 pp.3-10；转置卷积见 pp.19-26。  
> [2] Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner. “Gradient-Based Learning Applied to Document Recognition.” *Proceedings of the IEEE*, 86(11):2278-2324, 1998. 卷积网络与共享权重结构见 pp.2290-2295。  
> [3] Karen Simonyan, Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” *International Conference on Learning Representations (ICLR)*, 2015. arXiv:1409.1556, public version pp.1-14；小 $3\times3$ 卷积核与堆叠设计见 pp.2-3。  
> [4] Jonathan Long, Evan Shelhamer, Trevor Darrell. “Fully Convolutional Networks for Semantic Segmentation.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.3431-3440, 2015. 全卷积分类器转换、空间 score 与上采样见公开版本 pp.2-5。  
> [5] Olaf Ronneberger, Philipp Fischer, Thomas Brox. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science (LNCS) 9351, pp.234-241, 2015. arXiv:1505.04597v1, public version pp.1-8；网络尺寸和 Figure 1 见 p.2，架构、up-convolution、pixel-wise Softmax 见 p.4。  
> [6] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. “Deep Residual Learning for Image Recognition.” *CVPR*, pp.770-778, 2016. 残差映射 $y=F(x)+x$ 见公开版本 p.3, Equation (1)。  
> [7] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.” arXiv:1704.04861, 2017, public version pp.1-9；depthwise separable convolution 见 p.3。  
> [8] Fisher Yu, Vladlen Koltun. “Multi-Scale Context Aggregation by Dilated Convolutions.” *ICLR*, 2016. arXiv:1511.07122, public version pp.1-13；dilated convolution 定义与感受野说明见 pp.1-3。  
> [9] Augustus Odena, Vincent Dumoulin, Chris Olah. “Deconvolution and Checkerboard Artifacts.” *Distill*, 2016, no pagination；不均匀重叠解释见 “Overlap & Learning” section. DOI: 10.23915/distill.00003。  
> [10] Richard Zhang. “Making Convolutional Networks Shift-Invariant Again.” *Proceedings of the 36th International Conference on Machine Learning (ICML)*, Proceedings of Machine Learning Research (PMLR) 97:7324-7334, 2019. 下采样、混叠与抗混叠动机见公开版本 pp.1-4。  
> [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.” *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp.1026-1034, 2015. ReLU 网络初始化推导见公开版本 pp.4-5。  
> [12] Matthew D. Zeiler, Rob Fergus. “Visualizing and Understanding Convolutional Networks.” *European Conference on Computer Vision (ECCV)*, LNCS 8689, pp.818-833, 2014. 特征可视化方法与层级观察见 pp.818-827。  
> [13] Wenjie Luo, Yujia Li, Raquel Urtasun, Richard Zemel. “Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.” *Advances in Neural Information Processing Systems 29 (NeurIPS)*, pp.4898-4906, 2016. 理论与有效感受野差异见公开版本 pp.1-5。
