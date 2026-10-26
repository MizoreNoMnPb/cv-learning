# 计算机视觉基础算法

> 本文按图像滤波、特征提取、几何变换与评估指标组织查阅。卷积的完整推导见 [卷积文档](../01-Foundations/convolution.md)，边界框损失见 [损失函数](../01-Foundations/losses.md)。

---

## 一、卷积与图像滤波

### 1.1 2D 卷积的数学定义

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) \cdot K(m, n)$$

深度学习中的"卷积"实际是**互相关**（cross-correlation），省去了 kernel 翻转：

$$\operatorname{CrossCorr}(I, K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)$$


### 1.2 常用滤波核

| 滤波器 | 核 | 用途 |
|--------|---|------|
| **均值滤波** | $\frac{1}{9}\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}$ | 去噪（但会模糊边缘） |
| **高斯滤波** | $\frac{1}{16}\begin{pmatrix}1&2&1\\2&4&2\\1&2&1\end{pmatrix}$ | 去噪 + 保留更多结构 |
| **Sobel（$x$ 方向梯度）** | $\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix}$ | 比较左右亮度，主要响应垂直边缘 |
| **Laplacian** | $\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix}$ | 全方向边缘检测 |

### 1.3 高斯滤波的核心性质

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

- **可分离**：2D 高斯 = 两个 1D 高斯卷积 → 计算量从 $O(k^2)$ 降到 $O(2k)$
- $\sigma$ 增大通常带来更强平滑，同时损失细节；不能不考虑噪声类型就判断“去噪更好”。
- Canny 边缘检测的第一步就是高斯滤波

### 1.4 卷积输出尺寸公式

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

其中 P=padding, K=kernel_size, S=stride。

---

## 二、边缘检测

### 2.1 Canny 边缘检测（最经典的算法）

五步流程：

```
灰度图像
    │
1. 高斯滤波 (σ=1.4)          ← 去噪
    │
2. 计算梯度 (Sobel)           ← 梯度幅值 + 方向
    │
3. NMS (非极大值抑制)           ← 细边缘（只保留梯度方向上的局部最大值）
    │
4. 双阈值检测 (low=50, high=150)  ← 强边缘/弱边缘/非边缘
    │
5. 边缘连接 (hysteresis)      ← 弱边缘邻接强边缘 → 保留；否则丢弃
    │
最终边缘图
```


**双阈值 + 边缘连接（hysteresis）**：
- 梯度 > high → **强边缘**（确信保留）
- 梯度 < low → **非边缘**（丢弃）
- low < 梯度 < high → **弱边缘**：若沿 8-邻域弱边缘链可到达强边缘则保留，不限于直接邻接强边缘的第一圈像素。
- 这个设计兼顾了去噪（low 阈值滤掉弱噪声）和完整性（弱但连接强边缘的被保留）

### 2.2 梯度方向量化

**[分析判断]** 一种离散实现把梯度方向量化为 0°、45°、90°、135°，沿对应方向与两侧邻居比较；也可沿真实梯度方向插值比较。四方向量化不是 Canny 定义中不可更改的条件，上节的尺度和阈值也仅是示例。


---

## 三、特征点检测与描述

### 3.1 Harris 角点检测

**核心思想**：在图像上滑动一个窗口，观察窗口内像素的变化：

$$E(u, v) = \sum_{x,y} w(x,y) \cdot [I(x+u, y+v) - I(x, y)]^2$$

Taylor 展开后：

$$E(u,v) \approx [u \; v] \; M \; \begin{bmatrix} u \\ v \end{bmatrix}, \quad M = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

**角点判定**（通过 M 的特征值 $\lambda_1, \lambda_2$）：

| 条件 | 区域类型 |
|------|---------|
| $\lambda_1 \approx 0, \lambda_2 \approx 0$ | 平坦区域（两个方向都没变化） |
| $\lambda_1 \gg \lambda_2$ 或反之 | 边缘（一个方向有变化） |
| $\lambda_1 \gg 0, \lambda_2 \gg 0$ | **角点**（两个方向都显著变化） |

实际常用 Harris response 避免显式特征值分解：
$$R = \det(M) - k \cdot \text{trace}(M)^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$


### 3.2 SIFT：尺度不变特征变换

SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）在尺度空间检测局部极值，并用局部方向直方图描述关键点。**[论文发现]** 下列步骤来自 [1，§3–6]。

**四步流程**：

```
1. Scale-Space Extrema Detection
   ├── 构建高斯金字塔（不同 σ 的模糊 + 降采样）
   └── DoG (Difference of Gaussian): 相邻尺度相减 → 极值点检测

2. Keypoint Localization
   └── 去掉低对比度和边缘响应过强的点（比例取决于图像与阈值）

3. Orientation Assignment
   └── 在关键点周围计算梯度方向直方图（36 bins）
       取主方向（和超过 80% 的次方向）
       后续所有计算旋转到主方向 → 旋转不变性

4. Keypoint Descriptor
   └── 16×16 邻域 → 分成 4×4 子块
       每块计算 8 方向梯度直方图
       → 4×4×8 = 128 维描述子
        归一化与截断 → 减弱部分亮度、对比度变化的影响
```

**SIFT 的不变性**：
- **平移不变**：局部检测
- **旋转不变**：方向归一化
- **尺度不变**：DoG 多尺度空间极值检测
- **光照鲁棒性**：梯度与归一化可抵消理想的加性亮度偏移和正比例缩放，但非线性照明、阴影、饱和仍会改变描述子。[1，§6]
- **视角变化**：部分鲁棒（仿射变化）

### 3.3 ORB (Oriented FAST and Rotated BRIEF)

ORB（Oriented FAST and Rotated BRIEF，带方向的 FAST 与旋转 BRIEF）用带方向的角点检测和二值比较构造局部描述子。FAST（Features from Accelerated Segment Test，加速线段测试特征）用圆周连续像素比较检测角点；BRIEF（Binary Robust Independent Elementary Features，二值鲁棒独立基本特征）用局部像素对的亮度比较构造二值编码。

ORB 是比 SIFT 更轻量的二值特征方案：

- **FAST 角点**：圆周 16 个像素中必须存在连续的一段像素，全部比中心亮至少阈值或全部暗至少阈值。FAST-9 的连续段长为 9；不是任意 9 个绝对差较大的像素即可。[2，第 4–5 页]
- **BRIEF 描述子**：256 对随机点对的灰度比较 → 256 bit。极快但不含旋转不变性

ORB = FAST (加尺度金字塔 + 灰度质心方向) + rBRIEF (旋转感知的 BRIEF)：


### 3.4 特征匹配


**[论文发现]** Lowe 比值检验比较最近邻与次近邻距离 $d_1/d_2$，原论文采用 0.8 阈值。[1，§7.1] 较低比值表示最近邻相对更突出，但不保证它正确；阈值是可选择的，仍需几何验证。0.7 是可能的设置，不能称为原论文固定定义。

---

## 四、几何变换

### 4.1 变换模型

| 变换 | 自由度 | 保留性质 | 矩阵形式 |
|------|--------|---------|---------|
| **平移** | 2 | 方向、长度、角度、平行 | $2\times3$ |
| **欧几里得** | 3 | 长度、角度、平行 | $2\times3$ |
| **相似** | 4 | 角度、平行 | $2\times3$ |
| **仿射** | 6 | 平行 | $2\times3$ |
| **投影（单应）** | 8 | 直线 | $3\times3$ |

### 4.2 单应矩阵 (Homography)

**[数学推导]** 在齐次坐标中，平面间投影满足 $\lambda\tilde x'=H\tilde x$，其中 $\lambda\neq0$ 表示齐次尺度。若 $H\tilde x=(u,v,w)^T$ 且 $w\neq0$，欧氏坐标为 $(u/w,v/w)$，不能将齐次等价直接写成两个欧氏坐标的线性等式。

**至少需要 4 对非退化匹配点**（每对贡献 2 个独立约束，单应矩阵除整体比例外有 8 个自由度）；共线等退化配置不能唯一确定单应。


### 4.3 RANSAC：随机采样一致性

RANSAC（RANdom SAmple Consensus，随机采样一致性）通过反复抽取小样本拟合模型，并用全体数据的一致性筛选候选模型。

解决"有 outliers 时的模型拟合"：

```
1. 随机选最小样本集（如算 H 需要 4 对点）
2. 用这些样本拟合模型
3. 测试所有数据，统计 inliers（误差 < threshold）
4. 重复 N 次，保留 inliers 最多的模型
5. 用所有 inliers 重新拟合最终模型
```

**[数学推导]** 假设每次采样独立，单点为内点的概率为 $w$，抽取 $s$ 点全部为内点的概率近似为 $w^s$。$N$ 次均未抽中全内点组的概率为 $(1-w^s)^N$。要求成功概率至少为 $p$，得到：
$$N\geq\left\lceil\frac{\log(1-p)}{\log(1-w^s)}\right\rceil,\qquad 0<w,p<1.$$

- p = 期望的成功概率（通常 0.99）
- w = inlier 比例（估计值）
- s = 最小样本数

例如内点比例 $w=0.5$、$s=4$、$p=0.99$ 时，需要向上取整为 72 次。该概率计算还假设全内点组非退化且能产生有效模型，不是实际成功的无条件保证。

### 4.4 图像配准流程

```
Image A ──→ 特征检测 (SIFT/ORB) ──→ 特征描述
                                           │
Image B ──→ 特征检测 (SIFT/ORB) ──→ 特征描述
                                           │
                         特征匹配 (FLANN + Lowe's ratio test)
                                           │
                         几何验证 (RANSAC 估计 H / F 矩阵)
                                           │
                         图像变换（透视变换 / 仿射变换）
```

---

## 五、NMS (Non-Maximum Suppression)

NMS（Non-Maximum Suppression，非极大值抑制）按分数保留预测并抑制高度重叠框；它常用于密集检测器，但不是所有检测器的必需环节，例如 DETR 采用集合预测避免依赖 NMS。[3，第 1、5–7 页]

### 5.1 算法流程

```
输入：N 个检测框，每个有 score
1. 按 score 降序排序
2. 取最高分框 B_max → 加入保留列表
3. 移除与 B_max 的 IoU > threshold 的其他框
4. 重复 2-3 直到没有剩余框
```

### 5.2 Soft-NMS

标准 NMS 直接删除重叠框 → 对密集场景（人群、车辆）可能误删真实物体。Soft-NMS 降低重叠框的 score 而非删除：

$$s_i = \begin{cases} s_i, & \text{IoU} < \tau \\ s_i \cdot (1 - \text{IoU}), & \text{IoU} \geq \tau \end{cases}$$

**适用场景**：密集行人检测、车辆检测。

### 5.3 变体对比

| 方法 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **NMS** | 直接删除 | 快 | 密集场景漏检 |
| **Soft-NMS** | 降低 score | 减少漏检 | 慢一点 |
| **Weighted Boxes Fusion** | 按权重融合多份重叠预测 | 可用于模型集成 | 效果依赖预测质量、匹配与融合规则，不能保证最精确 |

---

## 六、IoU 及其变体

### 6.1 IoU 计算

IoU（Intersection over Union，交并比）衡量两个区域的相对重叠。**[数学推导]** 对正面积轴对齐矩形 $A=(x_1,y_1,x_2,y_2)$ 与 $B=(u_1,v_1,u_2,v_2)$：

$$w_I=\max(0,\min(x_2,u_2)-\max(x_1,u_1)),\quad h_I=\max(0,\min(y_2,v_2)-\max(y_1,v_1)),$$
$$|I|=w_Ih_I,\qquad\operatorname{IoU}(A,B)=\frac{|I|}{(x_2-x_1)(y_2-y_1)+(u_2-u_1)(v_2-v_1)-|I|}.$$

分母来自集合容斥公式。这里使用连续坐标区域面积；若采用离散像素索引，需要明确端点约定。

### 6.2 Rotated IoU

OBB（Oriented Bounding Box，有向边界框）用中心、尺寸和方向表示可旋转矩形。
两个旋转矩形（OBB）的交集面积计算**远复杂于轴对齐矩形**。核心挑战：两个旋转矩形的交集是多边形（最多 8 边形），需要：

1. **求交点**：A 的每条边与 B 的每条边的交叉点
2. **判断包含**：A 的顶点是否在 B 内，B 的顶点是否在 A 内
3. **凸包排序**：将所有交点和内点按角度排序
4. **多边形面积**：用 Shoelace 公式


### 6.3 OBB 的角度歧义性

同一个旋转矩形可以采用不同角度约定：

- **边顺序约定**：先固定 width/height 的边顺序，再限制 $\theta$ 的取值区间；
- **长边约定**：令 $\theta$ 表示长边与 x 轴夹角，通常限制在长度为 $\pi$ 的区间内。

因此，比较或变换旋转框前必须同时统一边顺序与角度区间。

---

## 七、图像金字塔与多尺度

### 7.1 高斯金字塔

```
原始图像 (640×480)
    ↓ Gaussian blur + 降采样 2×
Level 1 (320×240)
    ↓
Level 2 (160×120)
    ↓
...
```


### 7.2 在检测中的应用

**多尺度测试**（传统方法）：

**[分析判断]** 将输入缩放为若干尺寸，分别提取特征并检测，再把预测框映射回同一坐标系进行合并。它增加了多次图像处理的代价；特征金字塔则在一次网络内融合不同分辨率特征，两者不能视为同一操作。

**图像金字塔在 SIFT 中的角色**：每层（octave）是上一层降采样 2× 的结果，每层内又有多个 DoG 尺度 → 实现尺度不变性。

---

## 八、颜色空间

### 8.1 常见颜色空间

| 空间 | 通道 | 特点 | 使用场景 |
|------|------|------|---------|
| **RGB** | Red, Green, Blue | 红绿蓝通道表示；不等同于原始传感器读数 | 显示、常规输入 |
| **HSV** | Hue, Saturation, Value | 色相、饱和度与明度分量 | 按颜色阈值分割，但不保证照明不变 |
| **Lab** | L, a, b | 感知均匀（ΔE 接近人眼感知） | 色彩差异计算 |
| **YCbCr** | Luma, Cb, Cr | 亮度、色度分离 | 视频压缩、肤色检测 |

### 8.2 RGB → HSV

**[数学推导]** 设 $r,g,b\in[0,1]$，$M=\max(r,g,b)$、$m=\min(r,g,b)$、$\Delta=M-m$。则 $V=M$，$M>0$ 时 $S=\Delta/M$（$M=0$ 时约定 $S=0$）。当 $\Delta>0$ 时，按最大通道定义色相：

$$H=60^\circ\begin{cases}((g-b)/\Delta)\bmod6,&M=r,\\(b-r)/\Delta+2,&M=g,\\(r-g)/\Delta+4,&M=b.\end{cases}$$

$\Delta=0$ 时颜色无彩，色相没有唯一意义。理想正比例缩放 $(r,g,b)\mapsto a(r,g,b)$ 且不发生饱和时，$H,S$ 不变；若光源改变通道比例、出现加性偏移或饱和，$H,S$ 也可能变化。

**[分析判断]** 因此 HSV 便于表达颜色阈值，却不能把光照变化保证隔离到 V 通道。

---

## 九、图像分割基础

### 9.1 阈值分割


**Otsu 算法的原理**：遍历所有可能阈值 t，找使**类间方差最大**的 t：
$$\sigma_B^2(t) = w_0(t)w_1(t)[\mu_0(t) - \mu_1(t)]^2$$

其中 $w_0, w_1$ 是前后景像素占比，$\mu_0, \mu_1$ 是前后景平均灰度。

### 9.2 连通域分析

在二值图上，连通域由 4-邻域或 8-邻域关系定义。对每个连通分量可计算面积、质心、外接框与形状统计量。

---

## 十、直方图

### 10.1 灰度直方图

对离散灰度级 $r_k$，归一化直方图为：

$$
p(r_k)=\frac{n_k}{N},
$$

其中 $n_k$ 是灰度为 $r_k$ 的像素数，$N$ 是总像素数。

### 10.2 直方图均衡化

增强对比度——使直方图尽量均匀分布：


**原理**：对像素值做 CDF（累积分布函数）映射，使输出近似均匀分布。

**CLAHE** (Contrast Limited Adaptive Histogram Equalization)：在局部小区域做均衡化 + 限制对比度放大（防止噪声过度增强）。

### 10.3 HOG (Histogram of Oriented Gradients)

传统行人检测的核心特征（Deformable Part Models 的基础）：

```
1. 计算梯度幅值和方向
2. 将图像分成 cells（如 8×8）
3. 每个 cell 统计 9 方向梯度直方图
4. 将 2×2 cells 组成一个 block，做归一化
5. 所有 block 的 HOG 特征拼接 → 最终描述子
```


---

## 十一、性能评估指标

### 11.1 Precision, Recall, F1

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

### 11.2 PR Curve 与 AP

对检测器在**不同 confidence 阈值**下，绘制 Precision-Recall 曲线：

AP（Average Precision，平均精度）汇总不同召回率下的精确率；具体离散定义依评价协议而定。VOC 2007 使用 11 点插值形式：

$$\text{AP}_{11}=\frac{1}{11}\sum_{r\in\{0,0.1,\ldots,1\}}\max_{\tilde r\geq r}P(\tilde r).$$

不能把这一公式当作所有 AP 指标的统一定义。COCO 的标准汇总还涉及 101 个召回采样点、多个 IoU 阈值以及目标尺度等设置；报告数字时应写明数据集、匹配规则与具体指标。该段仍需在评价专题中补齐官方协议的逐项出处。

### 11.3 mAP

$$\text{mAP} = \frac{1}{K} \sum_{k=1}^K \text{AP}_k$$

---

## 十二、常见理论问题

### Q: 为什么用 Canny 的 hysteresis 而不是单阈值？

**A**: 单阈值会把断续的边缘切断。Hysteresis 通过"弱边缘如果连接强边缘就保留"的规则，在**保持边缘连续性**的同时滤掉噪声孤点。

### Q: SIFT vs ORB 什么时候用哪个？

**[分析判断]** 比较应依据视角、尺度、纹理、匹配精度与计算预算，不能固定排列为某种描述子始终更好。SIFT 的梯度描述与 ORB 的二值比较提供不同取舍；学习式描述子也需要在目标数据上验证。许可与专利状态不作为本理论页的选型依据。

### Q: RANSAC 在 outlier 比例很高时怎么办？

**[数学推导]** RANSAC 没有统一的 50% 外点硬阈值。由第 4.3 节可知，成功率取决于内点比例、最小样本数和迭代数；当内点比例很小时，需要的次数快速增长。最小样本数由模型决定，不能任意减小。更可靠的匹配先验和采样策略可能改善效率，但仍需验证退化与误匹配。

## 论文出处与证据边界

> [1] David G. Lowe. “Distinctive Image Features from Scale-Invariant Keypoints.” International Journal of Computer Vision, 60(2):91–110, 2004。作者公开稿 §3–6（PDF 第 7–17 页）描述关键点与描述子，§7.1（第 19–20 页）介绍最近邻比值检验。[作者原文](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
>
> [2] Edward Rosten, Tom Drummond. “Machine Learning for High-Speed Corner Detection.” ECCV, 2006。作者公开稿第 4–5 页定义连续圆周像素测试。[作者原文](https://www.edwardrosten.com/work/rosten_2006_machine.pdf)
>
> [3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko. “End-to-End Object Detection with Transformers.” ECCV, 2020。arXiv:2005.12872v3，第 1、5–7 页介绍无 NMS 集合预测。[原文](https://arxiv.org/pdf/2005.12872v3)

**[分析判断]** 本文仍是跨主题速查材料；Canny、Harris、ORB、Otsu、HOG 与评估协议的完整原文对照应在相应专题中继续补齐，不应把本文当作这些方法均已逐项完成证据核验的综述。
