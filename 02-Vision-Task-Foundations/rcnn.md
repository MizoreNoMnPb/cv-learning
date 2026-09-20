# R-CNN 系列目标检测器：从 R-CNN 到 Mask R-CNN

> 两阶段目标检测的演进路线：R-CNN (2014) → Fast R-CNN (2015) → Faster R-CNN (2015) → Mask R-CNN (2017)

---

## 一、R-CNN (2014)

> Girshick et al., "Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation", CVPR 2014

### 1.1 核心流程

R-CNN = **Region proposals + CNN 分类**

```
输入图像
    │
    ▼
Selective Search → ~2000 个 Region Proposals（类别无关）
    │
    ▼
每个 proposal warp 到 227×227
    │
    ▼
CNN (AlexNet) → 提取 4096 维特征向量
    │
    ▼
SVM 分类器 → 对每个类别打分
    │
    ▼
Bounding Box Regressor → 精修框坐标
    │
    ▼
NMS (Non-Maximum Suppression) → 最终检测结果
```

### 1.2 训练流程（多阶段）

1. **CNN 预训练**：ImageNet 分类（1000 类）
2. **CNN 微调**：在 warped proposals 上微调（21 类 = 20 + background）
3. **SVM 训练**：用 CNN 特征训练每个类别的二分类 SVM（"hard negative mining"）
4. **BBox Regressor 训练**：训练类别特定的框回归器

### 1.3 局限性

- **极慢**：每张图需要 2000 次 CNN 前向（~47s/image on GPU）
- **多阶段训练**：CNN、SVM、BBox Regressor 分别训练
- **Selective Search 是瓶颈**：不可学习的 proposal 生成
- **Warping 丢失信息**：强制拉伸 proposal 到 227×227 改变了长宽比

---

## 二、Fast R-CNN (2015)

> Girshick, "Fast R-CNN", ICCV 2015

### 2.1 核心改进：共享特征计算

R-CNN 为每个 proposal 独立计算 CNN，Fast R-CNN **一次前向、共享特征**：

```
输入图像
    │
    ▼
CNN → 整张图的特征图 (不是 2000 次，只算 1 次)
    │
    ▼
Selective Search Proposals → 映射到特征图上
    │
    ▼
RoI Pooling：每个 proposal → 固定尺寸 (7×7) 的特征块
    │
    ▼
FC 层 → 两个 sibling 输出:
    ├── Softmax 分类 (K+1 类，含背景)
    └── BBox Regressor (每个类别 4 个值)
```

### 2.2 RoI Pooling

将任意大小的 RoI 映射到固定尺寸的特征图。

```
输入：特征图 (C, H, W) + RoI (x1, y1, x2, y2)
    │
1. 将 RoI 量化为特征图网格
2. 将 RoI 分成 h_out × w_out 个子窗口（如 7×7）
3. 每个子窗口内做 Max Pooling
    │
输出：(C, h_out, w_out) 固定尺寸
```


**RoI Pooling 的问题**：量化操作（取整）导致 misalignment。这个被后来的 RoI Align 解决。

### 2.3 Multi-Task Loss

Fast R-CNN 实现了端到端的联合训练：

$$
\mathcal{L} = \mathcal{L}_{cls}(p, u) + \lambda \cdot [u \geq 1] \cdot \mathcal{L}_{loc}(t^u, v)
$$

其中：

- $\mathcal{L}_{cls}$：Softmax CrossEntropy
- $\mathcal{L}_{loc}$：Smooth L1 Loss（只在非背景类别上计算）
- $u \geq 1$：当该 RoI 是正样本（非背景）时才计算回归损失

---

## 三、Faster R-CNN (2015)

> Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015

### 3.1 核心创新：RPN (Region Proposal Network)

**抛弃 Selective Search**，用一个可学习的子网络生成 proposals：

```
输入图像
    │
    ▼
CNN Backbone → 共享特征图 (C, H, W)
    │
    ├──→ RPN (Region Proposal Network):
    │       在特征图的每个位置
    │       - k 个 anchor boxes (不同尺度×长宽比)
    │       - 每个 anchor: 2 个 cls 得分(前景/背景) + 4 个回归值
    │       - 输出: ~300 个 proposals
    │
    └──→ RoI Pooling + Fast R-CNN Head (同上):
            对 RPN 生成的 proposals 做精细分类和回归
```

### 3.2 RPN 的 Anchor 设计

每个滑动窗口位置对应 **k = 9** 个 anchor boxes：

- 3 个尺度：128², 256², 512²
- 3 个长宽比：1:1, 1:2, 2:1


### 3.3 RPN 损失函数

$$
\mathcal{L}_{RPN} = \frac{1}{N_{cls}} \sum_i \mathcal{L}_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* \cdot \mathcal{L}_{reg}(t_i, t_i^*)
$$

- $\mathcal{L}_{cls}$：Binary CrossEntropy（前景 vs 背景）
- $\mathcal{L}_{reg}$：Smooth L1 Loss（只对正样本 anchor）
- $p_i^* = 1$ 如果 anchor 与某个 GT 的 IoU > 0.7（正样本）
- $p_i^* = 0$ 如果 anchor 与所有 GT 的 IoU < 0.3（负样本）

### 3.4 四步交替训练

1. 训练 RPN（ImageNet 预训练 backbone）
2. 训练 Fast R-CNN（用 RPN 的 proposals，也 ImageNet 预训练）
3. 冻结 backbone 共享层，只微调 RPN
4. 冻结 backbone 共享层，只微调 Fast R-CNN head

> 后续方法通常采用**联合训练**（joint training），不再需要四步交替。

### 3.5 Smooth L1 Loss

$$
\text{smooth}_{L1}(x) = \begin{cases} 0.5 x^2 \cdot \frac{1}{\beta}, & \text{if } |x| < \beta \\ |x| - 0.5\beta, & \text{otherwise} \end{cases}
$$

- 比 L2 Loss 对异常值更鲁棒
- 比 L1 Loss 在原点附近更平滑（梯度不会突变）

---

## 四、RoI Align (Mask R-CNN 中的改进)

> He et al., "Mask R-CNN", ICCV 2017

### 4.1 RoI Pooling 的问题

RoI Pooling 两次量化导致 misalignment：

1. RoI 坐标取整（浮点 → 整数）
2. Bin 边界取整

对于分割任务，这种像素级错位会严重影响 mask 精度。

### 4.2 RoI Align

**无量化操作**：使用双线性插值在连续坐标上采样：

若采样点 $(x,y)$ 位于四个整数网格点之间，则其特征由四个邻点加权得到：

$$
f(x,y)=\sum_{i\in\{0,1\}}\sum_{j\in\{0,1\}}
w_{ij}(x,y)f(x_i,y_j),
\qquad
\sum_{i,j}w_{ij}=1.
$$


---

## 五、Mask R-CNN (2017)

> He et al., "Mask R-CNN", ICCV 2017

### 5.1 核心创新

在 Faster R-CNN 的基础上添加了**第三个分支**——Mask Prediction：

```
Faster R-CNN 输出:
    ├── 分类 (K+1 类)
    └── BBox 回归 (4×K)

Mask R-CNN 新增:
    └── Mask 预测 (K × 28 × 28)
```

### 5.2 Mask Head

每个 RoI 的固定尺寸特征先经过若干保持空间尺寸的卷积，再上采样到更高分辨率，最后为每个类别输出一张独立的 mask logit 图。该分支不与分类分支共享最终预测头。

### 5.3 关键设计

**Decoupled Mask and Class Prediction**：

- 分类分支输出类别标签
- Mask 分支输出 **K 个独立的二值 mask**（每类一个，Sigmoid 独立激活）
- 推理时：选分类分支预测的类别对应的那个 mask

**为什么是 per-class sigmoid 而非 per-pixel softmax**：

- Per-pixel softmax 要求不同类别的 mask 互相排斥
- Per-class sigmoid 允许重叠（两个物体的 mask 可以有交集）
- 对重叠物体更友好

### 5.4 Mask Loss

$$
\mathcal{L}_{mask} = -\frac{1}{N} \sum_{i} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
$$

只计算**GT 类别对应的 mask**的 BCE Loss（不对所有 K 个 mask 计算）。


---

## 六、R-CNN 系列演进总结

| 方法               | 年份 | Proposal         | 特征           | 分类器     | 速度  | 新增能力        |
| ------------------ | ---- | ---------------- | -------------- | ---------- | ----- | --------------- |
| R-CNN              | 2014 | Selective Search | 独立 CNN       | SVM        | ~47s  | 首次用 CNN 检测 |
| Fast R-CNN         | 2015 | Selective Search | 共享 CNN       | Softmax    | ~2s   | 端到端训练      |
| Faster R-CNN       | 2015 | RPN (可学习)     | 共享 CNN       | Softmax    | ~0.2s | 全可学习        |
| Mask R-CNN         | 2017 | RPN              | 共享 CNN + FPN | Softmax    | ~0.2s | +实例分割       |

---

## 七、常见理论问题

### Q: RoI Pooling vs RoI Align 的区别？哪个更好？

**A**:

- **RoI Pooling**：两次量化取整，损失空间精度，速度快
- **RoI Align**：双线性插值，无量化，精度高，但稍慢
- RoI Align 对分割任务是必须的（像素级精度），对检测也有小改善（~1% AP）
- RoI Align 已成为需要高精度空间对齐时的标准选择

### Q: RPN 的 anchor 怎么选？不合适的 anchor 有什么影响？

**A**: 通常通过分析数据集中的 bbox 长宽比和尺度分布来选择（如 K-Means 聚类）。不合适的 anchor 会导致：

- 太多 anchor 不匹配任何 GT（正负样本失衡）
- 小物体或非常规长宽比的物体难以覆盖

### Q: Faster R-CNN 和 YOLO 的本质区别？

**A**: Faster R-CNN 是**两阶段**：RPN 初筛 proposals → ROI Head 精细分类回归。YOLO 是**单阶段**：直接在特征图上预测。两阶段精度高（特别是小物体），单阶段速度快。近年差距缩小（YOLOv8 已接近）。

### Q: Mask R-CNN 中为什么 mask 分支不做 softmax？

**A**: Per-pixel softmax 强制每个像素只能属于一个类别。Per-class sigmoid 允许两个物体的 mask 有重叠区域。对于大多数场景（如两个物体部分遮挡），sigmoid 更合理。
