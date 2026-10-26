# YOLO 系列目标检测器：从 YOLOv1 到 YOLO11

> YOLO（You Only Look Once，只看一次）：直接从图像特征预测目标框与类别的一类单阶段检测器。

本文选取截至 2024 年的代表性设计，按框参数化、特征融合、样本分配与去重方式组织。不同版本来自不同团队，不是一条由同一作者持续发布的严格继承链。论文结论标为“论文发现”；版本信息以维护者资料为补充，不作为性能优劣的证据。

---

## 一、YOLOv1 (2016)

> [论文发现] 本节对应文献 [1] 第 2–4 页。

### 1.1 核心思想

把目标检测重新定义为一个**回归问题**：直接从图像像素预测边界框和类别概率。

```
输入图像 448×448
    │
    ▼
卷积神经网络（24 卷积层 + 2 全连接层）
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

其中 IoU（Intersection over Union，交并比）是预测框与真实框交集面积除以并集面积。每个网格只预测一组共享的条件类别概率，并非为两个框各预测一组类别分布。

### 1.3 损失函数

为避免原论文公式 (3) 的计数边界与省略下标造成歧义，这里将网格编号统一为 $i=1,\ldots,S^2$，框编号为 $j=1,\ldots,B$，带帽量为监督目标：

$$
\begin{aligned}
\mathcal L={}&\lambda_{\mathrm{coord}}\sum_{i=1}^{S^2}\sum_{j=1}^{B}\mathbb1_{ij}^{\mathrm{obj}}
\left[(x_{ij}-\hat x_{ij})^2+(y_{ij}-\hat y_{ij})^2
+(\sqrt{w_{ij}}-\sqrt{\hat w_{ij}})^2+(\sqrt{h_{ij}}-\sqrt{\hat h_{ij}})^2\right]\\
&+\sum_{i=1}^{S^2}\sum_{j=1}^{B}\mathbb1_{ij}^{\mathrm{obj}}(C_{ij}-\hat C_{ij})^2
+\lambda_{\mathrm{noobj}}\sum_{i=1}^{S^2}\sum_{j=1}^{B}\mathbb1_{ij}^{\mathrm{noobj}}(C_{ij}-\hat C_{ij})^2\\
&+\sum_{i=1}^{S^2}\mathbb1_i^{\mathrm{obj}}\sum_c(p_i(c)-\hat p_i(c))^2.
\end{aligned}
$$

$\mathbb1_{ij}^{\mathrm{obj}}$ 表示该框预测器负责这个目标，$\mathbb1_i^{\mathrm{obj}}$ 表示目标中心落在该网格。训练时，在该网格的候选框中选择当前 IoU 最高者承担框回归；不能把“网格有目标”与“其中每个框都负责目标”混为一谈。[1，第 3–4 页]

**关键设计**：
- $\lambda_{coord} = 5$：论文中用于平衡坐标误差与其他损失的经验权重，不表示定位在所有任务中比分类重要
- $\lambda_{noobj} = 0.5$：减轻无物体网格的置信度损失（大多数网格没有物体）
- $\sqrt{w}, \sqrt{h}$：平方根使小物体的尺寸偏差被更重地惩罚

### 1.4 局限性

- 每个网格只有一组类别分布，且可用框数量有限，多个目标中心落入同一网格时受到限制
- 对不常见的长宽比泛化差
- 小物体检测效果差（7×7 的粗网格）
- 未使用预设形状的 anchor box（锚框）；这是参数化方式，本身并不是缺点

---

## 二、YOLOv2 / YOLO9000 (2017)

> [论文发现] 本节对应文献 [2] 第 2–5 页。

### 2.1 关键改进

| 改进 | 说明 |
|------|------|
| **Batch Normalization（批归一化，BN）** | 归一化中间激活；论文在其消融配置中报告平均精度均值提升超过 2 个百分点，不能视为通用增益 |
| **高分辨率分类器** | 先在 448×448 上微调分类器 10 epochs |
| **Anchor Boxes** | 引入预设框形状，仍回归位置与宽高，只是改用相对锚框的参数化 |
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

> [论文发现] 本节对应文献 [3] 第 1–3 页。

### 3.1 关键改进

**Darknet-53 Backbone**：
- 53 层卷积，有残差连接（ResNet 风格）
- 无 pooling（用 stride=2 的卷积下采样）

**多尺度预测**：借鉴 FPN（Feature Pyramid Network，特征金字塔网络）的跨尺度融合思想，在不同分辨率特征上预测；以下尺寸对应 $416\times416$ 输入，并非硬性物体尺寸分区。

```
Backbone
    │
    ├──→ 13×13 特征 (stride 32) → 大物体检测
    ├──→ 26×26 特征 (stride 16) → 中物体检测（上采样自 13×13 + skip）
    └──→ 52×52 特征 (stride 8)  → 小物体检测（上采样自 26×26 + skip）
```

**多标签分类**：使用独立 sigmoid 与 BCE（Binary Cross-Entropy，二元交叉熵），允许一个物体同时有多个类别标签；定义见[损失函数](../01-Foundations/losses.md)。

---

## 四、YOLOv4 (2020)

> [论文发现] 本节对应文献 [4] 第 2–5 页。下面列举组件作用，不表示每一项都是该论文首次提出。

### 4.1 关键改进

**Bag of Freebies (训练技巧，不增加推理开销)**：
- **Mosaic 数据增强**：将 4 张图拼成 1 张，增加小物体和上下文多样性
- **CIoU（Complete IoU，完全交并比）损失**：同时考虑重叠面积、中心距离与长宽比
- **CmBN（Cross mini-Batch Normalization，跨小批次归一化）**：在同一批次内汇集多个小批次的统计量
- **Self-Adversarial Training**

**Bag of Specials (增加少许推理开销)**：
- **Mish 激活函数**：$f(x) = x \cdot \tanh(\ln(1+e^x))$
- **CSPDarknet53**：CSP（Cross Stage Partial，跨阶段部分连接）将部分特征分支分开处理后融合
- **SPP（Spatial Pyramid Pooling，空间金字塔池化）**：用不同尺度的池化聚合上下文
- **PANet（Path Aggregation Network，路径聚合网络）**：补充自底向上的特征融合路径

### 4.2 CIoU Loss

$$\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

其中：
- $\rho$ 是预测框和 GT 框中心点的欧氏距离
- $c$ 是包围两框的最小矩形的对角线长度
- $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$（长宽比一致性）
- $\alpha = \frac{v}{(1-\text{IoU})+v}$

出处是提出该损失的文献 [5] 第 4 页公式 (8)–(11)，并非 YOLOv4 原创。$b^{gt}$ 表示真实框中心；完全重合时按连续延拓令附加惩罚为零，避免把 $0/0$ 当成已定义数值。

---

## 五、YOLOv5 (2020)

> Ultralytics 团队维护的模型家族；名称不表示原 YOLO 作者的连续论文版本。

### 5.1 结构与训练改动

- 模型缩放：YOLOv5n/s/m/l/x 五种尺寸
- 自动 anchor 聚类
- 自适应图像缩放（letterbox）

### 5.2 架构与版本边界

```
输入 → 卷积下采样 → C3 特征阶段 → SPPF → 多尺度路径聚合 → 检测头
```

这里以官方 **v6.0 发布标签**中的模型结构为例：C3 是基于 CSP 思想的特征模块，SPPF（Spatial Pyramid Pooling - Fast，快速空间金字塔池化）使用串联池化聚合上下文。早期版本的 Focus 输入切片层与后来的 SPPF 不能无条件拼成一个统一“YOLOv5 架构”；也不能把所有版本的骨干直接写成 CSPDarknet53。[官方固定版本配置](https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml)

---

## 六、YOLOv6 / YOLOv7 (2022)

### YOLOv7

- **E-ELAN（Extended Efficient Layer Aggregation Network，扩展高效层聚合网络）**：设计特征聚合与梯度传播路径
- **模型重参数化**（RepConv）：训练时多分支，推理时合并为单分支
- **辅助头训练**：中间层辅助头增加监督，由主检测头的预测引导标签分配；粗标签供辅助头、细标签供主头。[6，第 5 页 Figure 5]

### YOLOv6

- 采用可重参数化卷积的骨干与解耦检测头。
- 初期使用 SimOTA（Simplified Optimal Transport Assignment，简化最优传输分配），根据预测质量动态选择正样本；**2022 年技术报告已改用 TAL（Task Alignment Learning，任务对齐学习）作为默认分配**，所以不能用 SimOTA 概括整个 YOLOv6。[7，第 4–5 页]

---

## 七、YOLOv8 (2023)

> Ultralytics 于 2023 年发布；此处用官方资料说明模型特征，损失与分配思想回到原始方法论文。[官方模型说明](https://docs.ultralytics.com/models/yolov8/)

### 7.1 核心改进

**与 v5 的主要区别**：
- **Anchor-Free**：不使用预设宽高的锚框，仍使用特征图上的参考点；框可由参考点到四条边的距离表示
- **解耦头 (Decoupled Head)**：分类和回归各用自己的预测分支，仍共享上游图像特征
- **C2f 模块**：汇集多个中间特征的 CSP 风格模块；不能仅凭连接数断言任何任务上的梯度或精度必然更好
- **TaskAlignedAssigner**：对齐分类和定位质量的样本分配

### 7.2 Anchor-Free Decoupled Head

```
FPN 特征 (B, C, H, W) 为每个尺度:
    │
    ├──→ 分类分支：卷积映射 → K 个类别分数        → (B, K, H, W)
    └──→ 回归分支：卷积映射 → 4 组离散距离分布    → (B, 4R, H, W)
```

$B$ 是批量大小，$K$ 是类别数，$R$ 是每条边距离分布的离散取值数量。

**Anchor-Free 预测**：每个特征图位置预测：
- 各类别的预测得分
- 参考点到框四条边的距离 (left, top, right, bottom)

参考点不必等于目标中心。给定 $(a_x,a_y)$，四个距离可解码为 $(a_x-l,a_y-t,a_x+r,a_y+b)$；若距离以特征网格为单位，还需乘对应步幅。这是参数化的代数转换。


### 7.3 TaskAlignedAssigner

匹配参考点与 GT（Ground Truth，真实标注）框时，同时考虑：
- **分类得分**：该位置和 GT 类别的匹配度
- **定位质量**：该位置预测框和 GT 框的 IoU

$$\text{align\_score} = \text{cls\_score}^\alpha \cdot \text{IoU}^\beta$$

这是 TOOD（Task-aligned One-stage Object Detection，任务对齐单阶段目标检测）提出的任务对齐度量：[8，第 5 页公式 (9)]。基本思想是对每个真实框选取得分高的候选点；具体检测器还要处理候选点范围、一个点匹配多个框的冲突，以及监督权重归一化，不能只用“全图 Top-K”概括完整分配流程。

### 7.4 Distribution Focal Loss (DFL)

DFL（Distribution Focal Loss，分布焦点损失）监督每条边到参考点的离散距离分布。设取值为 $0,\ldots,R-1$，预测概率 $p_j\geq0$ 且 $\sum_jp_j=1$，解码距离为：

$$\hat y=\sum_{j=0}^{R-1}j p_j.$$

对落在相邻整数 $i,i+1$ 之间的目标 $y$，损失为：

$$\mathcal L_{\mathrm{DFL}}=-(i+1-y)\log p_i-(y-i)\log p_{i+1}.$$

[数学推导] 两个监督权重非负且和为 1；令 $p_i=i+1-y$、$p_{i+1}=y-i$，其期望为 $i(i+1-y)+(i+1)(y-i)=y$。因此它能用离散概率表示连续的监督坐标。整数端点可用退化的单点目标；超出离散区间的目标另需范围约定。该公式出自 GFL（Generalized Focal Loss，广义焦点损失）论文 [9] 第 5 页公式 (5)–(6)，不是 YOLOv8 首创，也不是对预测不确定性已经完成校准的证明。


### 7.5 YOLOv8 变体

常见 n/s/m/l/x 变体改变网络宽度和深度。参数量取决于具体任务、类别数与发布版本；不能仅凭参数量判断某个应用上的速度、内存或精度。这里保留缩放概念，不把缺少统一配置的参数表当作模型选择结论。

### 7.6 OBB 版本 (YOLOv8-obb)

OBB（Oriented Bounding Box，有向边界框）用于表示旋转矩形。常用几何参数是 $(x,y,w,h,\theta)$。在距离分布预测上增加角度分支，不等于把内部的四个标量通道简单改成五个；概率距离和旋转几何需要共同解码。具体损失与角度约定需对应固定版本，不能把所有旋转框检测器混写为一套规则。

**角度表示**：

- 采用长边为方向并规定 $w\geq h$ 时，同一个矩形的方向以 $\pi$ 为周期；可选半开区间 $[-\frac{\pi}{4},\frac{3\pi}{4})$（常称 long-edge 135）；
- 也可选 $[-\frac{\pi}{2},\frac{\pi}{2})$（常称 long-edge 90）。原稿的 $[-\frac{\pi}{2},0]$ 只有半个周期，不能作为这种长边约定的完整区间。

[数学推导] 长边约定不变时，$\theta_{le90}=((\theta+\pi/2)\bmod\pi)-\pi/2$。另一些定义用 $90^\circ$ 区间并交换宽高；它们与长边定义不同。正方形还存在额外对称性，不宜把某个数值角度视为唯一真值。

---

## 八、YOLOv9 / YOLOv10 / YOLO11 (2024)

### YOLOv9

- **GELAN（Generalized Efficient Layer Aggregation Network，通用高效层聚合网络）**：设计特征聚合与梯度路径
- **PGI（Programmable Gradient Information，可编程梯度信息）**：通过主分支、辅助可逆分支和多层辅助信息设计训练监督

[论文发现] 信息丢失是作者提出 PGI 的动机，推理时只保留主分支。[10，第 4–6 页] 这不是“所有深层网络必然丢失任务所需信息”或“信息瓶颈已被普遍解决”的数学定理。

### YOLOv10

- **NMS-Free**：通过一对一匹配训练，在推理时省去 NMS（Non-Maximum Suppression，非极大值抑制）这一候选框去重过程
- 双头训练：一对一匹配头 + 一对多匹配头。推理时只用一对一匹配头

[论文发现] 两个分支采用一致的匹配度量，利用一对多监督训练共享特征；详情见 [11] 第 4–5 页。不能把它缩减为“任何模型换成一对一标签就能保持原精度”。

### YOLO11

官方名称是 **YOLO11**，由 Ultralytics 于 2024 年发布。它延续该团队的多任务模型系列并调整特征提取结构；本节不把笼统的“更好骨干”当作机制解释。逐模块比较需要绑定发布版本，当前不列未经同条件核验的优劣结论。[官方模型说明](https://docs.ultralytics.com/models/yolo11/)

---

## 九、YOLO 系列演进总结

| 版本 | 年份 | Backbone | 关键技术 |
|------|------|----------|---------|
| v1 | 2016 | Custom CNN | 网格回归 |
| v2 | 2017 | Darknet-19 | Anchor + BN + Passthrough |
| v3 | 2018 | Darknet-53 | FPN + 多尺度 |
| v4 | 2020 | CSPDarknet53 | Mosaic + CIoU + Mish |
| v5 | 2020 起 | CSP 风格，依发布版本变化 | 模型缩放 + 自动 anchor |
| v8 | 2023 起 | C2f 等特征模块 | Anchor-Free + DFL + 解耦头 |
| v9 | 2024 | GELAN | PGI 辅助监督 |
| v10 | 2024 | 经效率设计的骨干与融合结构 | 一致双分配 + 无 NMS 推理 |
| YOLO11 | 2024 起 | Ultralytics 后续特征结构 | 需绑定版本比较 |

---

## 十、常见理论问题

### Q: YOLO 为什么叫 "You Only Look Once"？

**A**: 原论文强调用统一网络直接预测框与类别，区别于当时对大量候选区域分别提取特征的 R-CNN。Faster R-CNN 也能共享一次图像骨干特征提取，并在同一前向流程中完成候选框与区域预测；“单阶段”不等于只有它才能做一次网络前向。主要区别见 [R-CNN 串讲](rcnn.md)。

### Q: Anchor-Free vs Anchor-Based 的优缺点？

**A**:
- **Anchor-Free**：省去预设框形状，但仍需决定参考点、正样本范围、参数化与样本分配。
- **Anchor-Based**：提供形状先验，同时引入锚框尺度、比例和匹配规则。
- [分析判断] 二者都能检测小目标和不同长宽比目标。哪个更准确或稳定，需要控制骨干、分辨率、训练与匹配策略后比较，不能由名称直接推出。

### Q: 单阶段 vs 两阶段检测器的本质区别？

**A**: 单阶段直接在图像特征上预测目标；两阶段先生成候选区域，再对区域特征分类和细化框。速度与精度取决于具体模型和评价条件，结构分类本身不构成固定排名。

## 十一、原始出处与阅读边界

1. Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. “You Only Look Once: Unified, Real-Time Object Detection.” CVPR, 2016。[arXiv:1506.02640](https://arxiv.org/abs/1506.02640) 第 2–4 页，公式 (1)–(3)。
2. Joseph Redmon, Ali Farhadi. “YOLO9000: Better, Faster, Stronger.” CVPR, 2017。[arXiv:1612.08242](https://arxiv.org/abs/1612.08242) 第 2–5 页；框参数化在第 3 页。
3. Joseph Redmon, Ali Farhadi. “YOLOv3: An Incremental Improvement.” arXiv 预印本，2018。[arXiv:1804.02767](https://arxiv.org/abs/1804.02767) 第 1–3 页。
4. Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao. “YOLOv4: Optimal Speed and Accuracy of Object Detection.” arXiv 预印本，2020。[arXiv:2004.10934](https://arxiv.org/abs/2004.10934) 第 2–5 页。
5. Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren. “Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.” AAAI, 2020。[arXiv:1911.08287](https://arxiv.org/abs/1911.08287) 第 4 页公式 (8)–(11)。
6. Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao. “YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors.” CVPR, 2023（预印本 2022）。[arXiv:2207.02696](https://arxiv.org/abs/2207.02696) 第 3–5 页；辅助监督见第 5 页。
7. Chuyi Li, Lulu Li, Hongliang Jiang, Kaiheng Weng, Yifei Geng, Liang Li, Zaidan Ke, Qingyuan Li, Meng Cheng, Weiqiang Nie, Yiduo Li, Bo Zhang, Yufei Liang, Linyuan Zhou, Xiaoming Xu, Xiangxiang Chu, Xiaoming Wei, Xiaolin Wei. “YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications.” arXiv 技术报告，2022。[arXiv:2209.02976](https://arxiv.org/abs/2209.02976) 第 3–5 页。
8. Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang. “TOOD: Task-aligned One-stage Object Detection.” ICCV, 2021。[arXiv:2108.07755](https://arxiv.org/abs/2108.07755) 第 5 页公式 (9) 及样本分配段。
9. Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang. “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.” NeurIPS, 2020。[arXiv:2006.04388](https://arxiv.org/abs/2006.04388) 第 5 页公式 (5)–(6)。
10. Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao. “YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.” ECCV, 2024。[正式发表入口](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4462_ECCV_2024_paper.php)；本文页码按已核对的 [arXiv:2402.13616](https://arxiv.org/abs/2402.13616) 第 4–6 页。
11. Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. “YOLOv10: Real-Time End-to-End Object Detection.” NeurIPS, 2024。[正式发表入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html)；本文页码按已核对的 [arXiv:2405.14458](https://arxiv.org/abs/2405.14458) 第 4–5 页。

以上页码均指所链接 PDF 的阅读器页序。YOLOv5/v8/YOLO11 属于持续维护的模型系列，官方资料仅补充版本身份与结构事实；不把维护者宣传语替代为论文实验结论。本文还不是 YOLO 所有分支的完整综述。
