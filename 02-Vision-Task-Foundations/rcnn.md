# R-CNN 系列目标检测器：从 R-CNN 到 Mask R-CNN

> 两阶段目标检测的演进路线：R-CNN (2014) → Fast R-CNN (2015) → Faster R-CNN (2015) → Mask R-CNN (2017)

R-CNN（Regions with Convolutional Neural Network Features，基于区域的卷积神经网络特征）先提取候选区域，再进行分类与定位。CNN（Convolutional Neural Network，卷积神经网络）用局部卷积提取特征；RoI（Region of Interest，感兴趣区域）是待进一步识别的候选区域；SVM（Support Vector Machine，支持向量机）是基于分类间隔的判别模型。

各代方法的结构与目标属于 **[论文发现]**，分别对应文末 [1]–[4]；结构取舍的解释属于 **[分析判断]**。这里的两阶段指候选生成与候选识别，不代表一定要进行两次整图骨干前向。

---

## 一、R-CNN (2014)

> 原文 [1，第 2–4 页]。

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

- **重复计算**：每张图约 2000 个候选分别经过 CNN，区域间的重叠图像内容没有共享卷积特征。[1，第 2–3 页]
- **多阶段训练**：CNN、SVM、BBox Regressor 分别训练
- **Selective Search 是瓶颈**：不可学习的 proposal 生成
- **Warping 丢失信息**：强制拉伸 proposal 到 227×227 改变了长宽比

---

## 二、Fast R-CNN (2015)

> 原文 [2，第 2–3 页]。

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

**[论文发现]** 在候选区域已经给定的条件下，Fast R-CNN 联合训练共享特征、分类和框回归；Selective Search 本身不参与梯度训练。[2，第 2–3 页公式 (1)]

$$
\mathcal{L} = \mathcal{L}_{cls}(p, u) + \lambda \cdot [u \geq 1] \cdot \mathcal{L}_{loc}(t^u, v)
$$

其中：

- $\mathcal{L}_{cls}$：Softmax CrossEntropy
- $\mathcal{L}_{loc}$：Smooth L1 Loss（只在非背景类别上计算）
- $u \geq 1$：当该 RoI 是正样本（非背景）时才计算回归损失

---

## 三、Faster R-CNN (2015)

> 原文 [3，§3.1–3.2]。

### 3.1 核心创新：RPN (Region Proposal Network)

RPN（Region Proposal Network，区域候选网络）在共享特征图上预测类别无关的前景分数与候选框。

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
- $p_i^* = 1$：与某个真实框具有最高 IoU 的 anchor，或与某个真实框的 IoU > 0.7（正样本）。最高 IoU 规则用于避免某些真实框没有任何正候选。[3，§3.1.2]
- $p_i^* = 0$ 如果 anchor 未被归为正样本，且与所有 GT 的 IoU < 0.3（负样本）

GT（Ground Truth，真实标注）是训练监督；IoU（Intersection over Union，交并比）是两个框交集面积与并集面积之比。除上述正负样本外，其他 anchor 不参与该分类损失。

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

> 原文 [4，第 3–4 页]。

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

- **[论文发现]** 对一个 RoI，mask 分支只学习“给定类别下哪些像素属于该实例”，类别判断由独立分类分支负责。[4，第 3 页]
- 逐像素 sigmoid 避免不同类别的 mask 通道相互竞争，训练只监督真实类别通道。
- 这不是为了允许“两个实例重叠”：不同 RoI 的 mask 本来就独立输出，实例之间是否重叠与同一 RoI 内是否做类别 softmax 是两件事。

### 5.4 Mask Loss

$$
\mathcal{L}_{mask} = -\frac{1}{N} \sum_{i} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
$$

只计算**GT 类别对应的 mask**的 BCE Loss（不对所有 K 个 mask 计算）。


---

## 六、R-CNN 系列演进总结

| 方法 | 年份 | 候选生成 | 特征计算 | 主要变化 |
|---|---|---|---|---|
| R-CNN | 2014 | Selective Search | 每个区域独立卷积 | 将预训练卷积特征用于候选区域识别 |
| Fast R-CNN | 2015 | Selective Search | 整图共享卷积 | 在给定候选上联合分类与回归 |
| Faster R-CNN | 2015 | 可学习 RPN | 与检测头共享卷积 | 学习类别无关的候选生成 |
| Mask R-CNN | 2017 | RPN | 共享骨干，支持多种特征结构 | RoIAlign 与独立实例 mask 分支 |

**[分析判断]** 不同骨干、输入尺寸、候选数量和运行条件下的耗时不能直接横比。原始 R-CNN 的 AlexNet 流程也不应配上另一种骨干的速度数字。

---

## 七、常见理论问题

### Q: RoI Pooling vs RoI Align 的区别？哪个更好？

**A**:

- **RoI Pooling**：区域及分箱边界量化可能产生位置偏差。
- **RoI Align**：在连续坐标处插值，避免上述量化；原文通过消融观察了对 mask 与框预测的改善。[4，第 6–7 页]
- 它不是所有分割网络的必需模块，例如 U-Net 不通过 RoI 提取特征；具体计算开销与效果要结合配置。

### Q: RPN 的 anchor 怎么选？不合适的 anchor 有什么影响？

**A**: 通常通过分析数据集中的 bbox 长宽比和尺度分布来选择（如 K-Means 聚类）。不合适的 anchor 会导致：

- 太多 anchor 不匹配任何 GT（正负样本失衡）
- 小物体或非常规长宽比的物体难以覆盖

### Q: Faster R-CNN 和 YOLO 的本质区别？

**[分析判断]** Faster R-CNN 先生成候选，再对候选区域识别与回归；典型 YOLO 检测器直接在密集特征位置产生预测。阶段数说明信息流，不给出精度或速度的普遍排序；比较必须固定数据、骨干、输入尺寸与评估协议。

### Q: Mask R-CNN 中为什么 mask 分支不做 softmax？

**[论文发现]** 分类分支已经预测 RoI 的类别，mask 分支只需学习该类别的二值空间形状。使用独立 sigmoid 并仅监督真实类别通道，避免在 mask 目标中重复进行类别竞争。[4，第 3 页]

## 论文出处

> [1] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. “Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation.” CVPR, 2014。公开版第 2–4 页描述候选、卷积特征和训练。[原文](https://arxiv.org/pdf/1311.2524)
>
> [2] Ross Girshick. “Fast R-CNN.” ICCV, 2015。公开版第 2–3 页 RoI pooling 与多任务目标公式 (1)。[原文](https://arxiv.org/pdf/1504.08083)
>
> [3] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” NeurIPS, 2015；扩展版发表于 TPAMI, 2017。arXiv:1506.01497v3 第 3–6 页 §3.1–3.2 说明 RPN、正负样本及联合训练方案。[原文](https://arxiv.org/pdf/1506.01497v3)
>
> [4] Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. “Mask R-CNN.” ICCV, 2017。arXiv:1703.06870v3，第 3–4 页定义 mask 目标与 RoIAlign，第 6–7 页消融。[原文](https://arxiv.org/pdf/1703.06870v3)
