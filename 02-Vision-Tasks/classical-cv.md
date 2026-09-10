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
| **Sobel (水平)** | $\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix}$ | 水平边缘检测 |
| **Laplacian** | $\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix}$ | 全方向边缘检测 |

### 1.3 高斯滤波的核心性质

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

- **可分离**：2D 高斯 = 两个 1D 高斯卷积 → 计算量从 $O(k^2)$ 降到 $O(2k)$
- $\sigma$ 越大 → 模糊越强 → 去噪越好但细节丢失越多
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
- low < 梯度 < high → **弱边缘**：如果与强边缘 8-邻接 → 保留；否则丢弃
- 这个设计兼顾了去噪（low 阈值滤掉弱噪声）和完整性（弱但连接强边缘的被保留）

### 2.2 梯度方向量化

Sobel 梯度的方向被量化为 4 个方向（0°, 45°, 90°, 135°），因为 NMS 只需要在这些方向上的前后邻居：


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


### 3.2 SIFT (Scale-Invariant Feature Transform)

**四步流程**：

```
1. Scale-Space Extrema Detection
   ├── 构建高斯金字塔（不同 σ 的模糊 + 降采样）
   └── DoG (Difference of Gaussian): 相邻尺度相减 → 极值点检测

2. Keypoint Localization
   └── 去掉低对比度和边缘上的点（丢弃 ~90% 候选点）

3. Orientation Assignment
   └── 在关键点周围计算梯度方向直方图（36 bins）
       取主方向（和超过 80% 的次方向）
       后续所有计算旋转到主方向 → 旋转不变性

4. Keypoint Descriptor
   └── 16×16 邻域 → 分成 4×4 子块
       每块计算 8 方向梯度直方图
       → 4×4×8 = 128 维描述子
       归一化 → 光照不变性
```

**SIFT 的不变性**：
- **平移不变**：局部检测
- **旋转不变**：方向归一化
- **尺度不变**：DoG 多尺度空间极值检测
- **光照不变**：描述子归一化
- **视角变化**：部分鲁棒（仿射变化）

### 3.3 ORB (Oriented FAST and Rotated BRIEF)

ORB 是比 SIFT 更轻量的二值特征方案：

- **FAST 角点**：如果像素 p 与周围 16 个像素中 ≥ 9 个差异大 → 角点。极快但不含尺度和方向
- **BRIEF 描述子**：256 对随机点对的灰度比较 → 256 bit。极快但不含旋转不变性

ORB = FAST (加尺度金字塔 + 灰度质心方向) + rBRIEF (旋转感知的 BRIEF)：


### 3.4 特征匹配


**Lowe's ratio test 的原理**：对于 query 中的每个点，找最近邻 m 和次近邻 n。如果 $d(m)/d(n) < 0.7$，则 m 是可信匹配（因为离次近邻足够远，有唯一性）。如果比值 > 0.7，说明 m 和 n 都差不多远 → 可能是重复纹理/噪声。

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

描述两个平面之间的投影变换：
$$x' = Hx, \quad H_{3\times3}$$

**至少需要 4 对匹配点**（每对贡献 2 个方程，8 个未知数 = 8/2 = 4）。


### 4.3 RANSAC (RANdom SAmple Consensus)

解决"有 outliers 时的模型拟合"：

```
1. 随机选最小样本集（如算 H 需要 4 对点）
2. 用这些样本拟合模型
3. 测试所有数据，统计 inliers（误差 < threshold）
4. 重复 N 次，保留 inliers 最多的模型
5. 用所有 inliers 重新拟合最终模型
```

**迭代次数 N 的确定**：
$$N = \frac{\log(1-p)}{\log(1-w^s)}$$

- p = 期望的成功概率（通常 0.99）
- w = inlier 比例（估计值）
- s = 最小样本数

当 outlier 比 50% 时，s=4：$N = \frac{\log(0.01)}{\log(1-0.5^4)} \approx 72$ 次迭代就足够。

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

目标检测推理的**必须环节**，用于去除对同一物体的重复检测。

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
| **Weighted Boxes Fusion** | 融合重叠框 | 最精确 | 最慢 |

---

## 六、IoU 及其变体

### 6.1 IoU 计算


### 6.2 Rotated IoU

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

**图像金字塔在 SIFT 中的角色**：每层（octave）是上一层降采样 2× 的结果，每层内又有多个 DoG 尺度 → 实现尺度不变性。

---

## 八、颜色空间

### 8.1 常见颜色空间

| 空间 | 通道 | 特点 | 使用场景 |
|------|------|------|---------|
| **RGB** | Red, Green, Blue | 原始传感器输出，强光照耦合 | 显示、常规输入 |
| **HSV** | Hue, Saturation, Value | 颜色、饱和度、亮度分离 | 颜色分割（独立于光照） |
| **Lab** | L, a, b | 感知均匀（ΔE 接近人眼感知） | 色彩差异计算 |
| **YCbCr** | Luma, Cb, Cr | 亮度、色度分离 | 视频压缩、肤色检测 |

### 8.2 RGB → HSV


HSV 的优势：颜色由 Hue 单一维度决定，不受亮度/饱和度变化影响。光照变化只影响 V 通道。

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

$$\text{AP} = \int_0^1 P(R) \, dR \approx \frac{1}{11} \sum_{r \in \{0,0.1,...,1\}} \max_{\tilde{r} \geq r} P(\tilde{r})$$

（VOC 2007 的 11-point interpolation，COCO 用 101-point）

### 11.3 mAP

$$\text{mAP} = \frac{1}{K} \sum_{k=1}^K \text{AP}_k$$

---

## 十二、常见理论问题

### Q: 为什么用 Canny 的 hysteresis 而不是单阈值？

**A**: 单阈值会把断续的边缘切断。Hysteresis 通过"弱边缘如果连接强边缘就保留"的规则，在**保持边缘连续性**的同时滤掉噪声孤点。

### Q: SIFT vs ORB 什么时候用哪个？

**A**: SIFT 精度高但慢 + 有专利。ORB 快 + 免费。嵌入式/实时系统用 ORB，高精度离线匹配用 SIFT。神经网络时代两者都被 SuperPoint 等 learned descriptor 超越。

### Q: RANSAC 在 outlier 比例很高时怎么办？

**A**: RANSAC 对 outlier 比例 < 50% 工作良好。超过 50% 时：减少最小样本集大小、增加迭代次数、或用更鲁棒的变体如 MSAC（M-estimator SAC）、PROSAC（带先验的采样）。
