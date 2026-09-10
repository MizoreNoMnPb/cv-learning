# YOLO 系列目标检测器：从 v1 到 v11

> You Only Look Once — 单阶段目标检测的演进

---

## 一、YOLOv1 (2016)

> Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016

### 1.1 核心思想

把目标检测重新定义为一个**回归问题**：直接从图像像素预测边界框和类别概率。

```
输入图像 448×448
    │
    ▼
CNN (24 卷积层 + 2 FC层)
    │
    ▼
输出: 7×7×30 的张量
    │
    ▼
7×7 网格，每个网格预测:
  - 2 个 bounding box (x, y, w, h, confidence) → 2×5 = 10
  - 20 个类别概率 (PASCAL VOC)          → 20
  → 总共 30 个值/网格
```

### 1.2 预测格式

每个网格预测 B=2 个 bbox，每个 bbox 包含 5 个值：

$$(x, y, w, h, \text{confidence})$$

- $(x, y)$：bbox 中心相对于网格单元的偏移（归一化到 [0,1]）
- $(w, h)$：bbox 宽高相对于整张图像的比例（归一化到 [0,1]）
- confidence：$\text{Pr}(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$

### 1.3 损失函数

$$\mathcal{L} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$

**关键设计**：
- $\lambda_{coord} = 5$：加重位置损失的权重（位置比分类重要）
- $\lambda_{noobj} = 0.5$：减轻无物体网格的置信度损失（大多数网格没有物体）
- $\sqrt{w}, \sqrt{h}$：平方根使小物体的尺寸偏差被更重地惩罚

### 1.4 局限性

- 每个网格只能预测 B 个物体（密集场景差）
- 对不常见的长宽比泛化差
- 小物体检测效果差（7×7 的粗网格）
- 无 anchor，直接回归 bbox 坐标

---

## 二、YOLOv2 / YOLO9000 (2017)

> Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger", CVPR 2017

### 2.1 关键改进

| 改进 | 说明 |
|------|------|
| **Batch Normalization** | 每层卷积后加 BN，mAP +2% |
| **高分辨率分类器** | 先在 448×448 上微调分类器 10 epochs |
| **Anchor Boxes** | 引入 anchor，不再直接回归坐标 |
| **Dimension Clusters** | K-Means 聚类确定 anchor 尺寸（不用手工选） |
| **直接位置预测** | 预测相对 grid cell 的偏移（而非相对整图） |
| **Fine-Grained Features** | Passthrough layer：将 26×26 特征拼接到 13×13 |
| **多尺度训练** | 每 10 batches 换一个输入尺寸 |

### 2.2 Anchor + 直接位置预测

每个 grid cell 预测 5 个 anchor boxes，每个预测：
$$b_x = \sigma(t_x) + c_x$$
$$b_y = \sigma(t_y) + c_y$$
$$b_w = p_w \cdot e^{t_w}$$
$$b_h = p_h \cdot e^{t_h}$$

其中 $(c_x, c_y)$ 是 grid cell 左上角坐标，$(p_w, p_h)$ 是 anchor 的预设宽高。这种参数化将 bbox 中心限制在 grid cell 内，训练更稳定。

---

## 三、YOLOv3 (2018)

> Redmon & Farhadi, "YOLOv3: An Incremental Improvement"

### 3.1 关键改进

**Darknet-53 Backbone**：
- 53 层卷积，有残差连接（ResNet 风格）
- 无 pooling（用 stride=2 的卷积下采样）

**多尺度预测（FPN 核心思想）**：

```
Backbone
    │
    ├──→ 13×13 特征 (stride 32) → 大物体检测
    ├──→ 26×26 特征 (stride 16) → 中物体检测（上采样自 13×13 + skip）
    └──→ 52×52 特征 (stride 8)  → 小物体检测（上采样自 26×26 + skip）
```

**多标签分类**：用 Sigmoid + BCE（而非 Softmax），允许一个物体有多个标签。

---

## 四、YOLOv4 (2020)

> Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection"

### 4.1 关键改进

**Bag of Freebies (训练技巧，不增加推理开销)**：
- **Mosaic 数据增强**：将 4 张图拼成 1 张，增加小物体和上下文多样性
- **CIoU Loss**：取代 IoU Loss，同时考虑重叠面积 + 中心距离 + 长宽比
- **CmBN** (Cross mini-Batch Normalization)
- **Self-Adversarial Training**

**Bag of Specials (增加少许推理开销)**：
- **Mish 激活函数**：$f(x) = x \cdot \tanh(\ln(1+e^x))$
- **CSPDarknet53**：CSP (Cross Stage Partial) 减少梯度重复计算
- **SPP** (Spatial Pyramid Pooling)：多尺度 pooling 增加感受野
- **PANet** (Path Aggregation Network)：改进 FPN，加自底向上路径

### 4.2 CIoU Loss

$$\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

其中：
- $\rho$ 是预测框和 GT 框中心点的欧氏距离
- $c$ 是包围两框的最小矩形的对角线长度
- $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$（长宽比一致性）
- $\alpha = \frac{v}{(1-\text{IoU})+v}$

---

## 五、YOLOv5 (2020)

> Ultralytics 团队（非官方 YOLO 作者）

### 5.1 结构与训练改动

- 模型缩放：YOLOv5n/s/m/l/x 五种尺寸
- 自动 anchor 聚类
- 自适应图像缩放（letterbox）

### 5.2 架构

```
Input → Focus (切片下采样) → CSPDarknet53 → SPPF → PANet → Detect Head
```

---

## 六、YOLOv6 / YOLOv7 (2022)

### YOLOv7

- **E-ELAN** (Extended Efficient Layer Aggregation Network)
- **模型重参数化**（RepConv）：训练时多分支，推理时合并为单分支
- **辅助头训练**：深层辅助检测头帮助浅层学习

### YOLOv6

- **RepVGG 风格** backbone
- **SimOTA** label assignment（动态正样本匹配）

---

## 七、YOLOv8 (2023)

> Ultralytics, Jocher et al., 2023

### 7.1 核心改进

**与 v5 的主要区别**：
- **Anchor-Free**：不再使用 anchor box，直接预测 bbox 中心点 + 宽高
- **解耦头 (Decoupled Head)**：分类和回归分支完全分离
- **C2f 模块**（取代 C3）：更多的残差连接，更好的梯度流
- **TaskAlignedAssigner**：对齐分类和定位质量的样本分配

### 7.2 Anchor-Free Decoupled Head

```
FPN 特征 (B, C, H, W) 为每个尺度:
    │
    ├──→ Cls分支: Conv → Conv → Conv2d(K)        → Cls Logits (B, K, H, W)
    └──→ Reg分支: Conv → Conv → Conv2d(4×reg_max) → BBox Distribution (B, 4×reg_max, H, W)
```

**Anchor-Free 预测**：每个特征图位置预测：
- 该位置是否有物体中心
- 物体中心到 bbox 四条边的距离 (left, top, right, bottom)


### 7.3 TaskAlignedAssigner

匹配 anchor points 和 GT boxes 时，同时考虑：
- **分类得分**：该位置和 GT 类别的匹配度
- **定位质量**：该位置预测框和 GT 框的 IoU

$$\text{align\_score} = \text{cls\_score}^\alpha \cdot \text{IoU}^\beta$$

选 Top-K 个最高分的 anchor points 作为正样本。

### 7.4 Distribution Focal Loss (DFL)

不直接回归边界框的四个值，而是回归一条**分布**（每个边的可能值）：


### 7.5 YOLOv8 变体

| 模型 | Params | 适用 |
|------|--------|------|
| YOLOv8n | 3.2M | 移动端 |
| YOLOv8s | 11.2M | 轻量 |
| YOLOv8m | 25.9M | 中等 |
| YOLOv8l | 43.7M | 通用 |
| YOLOv8x | 68.2M | 高性能 |

### 7.6 OBB 版本 (YOLOv8-obb)

用于旋转框检测（如 DroneVehicle、DOTA）：
- 在 bbox head 中增加角度预测通道（从 xyxy 4 通道变为 xyxyθ 5 通道）
- OBB 模式的输出：$(x, y, w, h, \theta)$
- 使用 ProbIoU 或 Rotated IoU Loss

**角度表示**：

- long-edge 135：$\theta \in [-\frac{\pi}{4}, \frac{3\pi}{4}]$；
- long-edge 90：$\theta \in [-\frac{\pi}{2}, 0]$。

转换关系：如果 $\theta > \frac{\pi}{2}$，则 $\theta_{le90} = \theta - \pi$

---

## 八、YOLOv9 / YOLOv10 / YOLOv11 (2024)

### YOLOv9

- **GELAN** (Generalized ELAN)：通用化的高效层聚合网络
- **PGI** (Programmable Gradient Information)：通过辅助可逆分支保留更多梯度信息，解决深层网络的**信息瓶颈**问题

**核心 insight**：深层网络中，随着层数增加，梯度信息会逐渐丢失。PGI 通过一个辅助的可逆分支提供额外的梯度路径。

### YOLOv10

- **NMS-Free**：首次在 YOLO 系列实现无 NMS 推理（使用一对一匹配训练）
- 双头训练：一对一匹配头 + 一对多匹配头。推理时只用一对一匹配头

### YOLOv11

- YOLOv8 的改进版，Ultralytics 维护
- C3k2 模块（v8 C2f 的升级）
- 改进的特征融合
- 更好的 hyprid backbone

---

## 九、YOLO 系列演进总结

| 版本 | 年份 | Backbone | 关键技术 |
|------|------|----------|---------|
| v1 | 2016 | Custom CNN | 网格回归 |
| v2 | 2017 | Darknet-19 | Anchor + BN + Passthrough |
| v3 | 2018 | Darknet-53 | FPN + 多尺度 |
| v4 | 2020 | CSPDarknet53 | Mosaic + CIoU + Mish |
| v5 | 2020 | CSPDarknet53 | 模型缩放 + 自动 anchor |
| v8 | 2023 | CSPDarknet53 | Anchor-Free + DFL + Decoupled Head |
| v8-obb | 2023 | 同上 + 角度头 | OBB 旋转框 |
| v11 | 2024 | 改进 CSPDarknet | 结构与特征融合优化 |

---

## 十、常见理论问题

### Q: YOLO 为什么叫 "You Only Look Once"？

**A**: 与两阶段检测器（R-CNN 系列）形成对比——YOLO 只需**一次前向传播**同时输出物体位置和类别，不需要 region proposal → feature extraction → classification 的多步流程。这是它速度快的关键。

### Q: Anchor-Free vs Anchor-Based 的优缺点？

**A**:
- **Anchor-Free (YOLOv8)**：不需要预设 anchor，减少超参数，对小物体/非常规长宽比更灵活
- **Anchor-Based (YOLOv3-v7)**：利用先验知识（预设常见长宽比），训练更稳定，在常见物体上精度更高
- YOLOv8 选 Anchor-Free 的原因：简化设计 + DFL 提供了足够的定位精度

### Q: 单阶段 vs 两阶段检测器的本质区别？

**A**: 单阶段（YOLO）直接在特征图上分类+回归，速度快但精度略低（特别是小物体）。两阶段（Faster R-CNN）先生成 proposals 再精细分类/回归，精度高但慢。近年来这个界限在模糊——YOLOv8+ 的精度已接近两阶段检测器。
