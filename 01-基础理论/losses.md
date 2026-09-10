# 常见损失函数详解：公式与适用场景

> 本文用于查阅常见损失的公式、梯度和适用条件。首次阅读先看分类与回归损失，再按任务查阅后续各节；完整的概率与似然串讲尚待补充。

文中 $N$ 表示参与平均的样本或预测项数量，$y$ 表示标签，带帽符号表示预测。**[数学推导]** 标注由公式得到的性质，**[论文发现]** 标注原文报告的设计，**[分析判断]** 标注选择与比较建议。

---

## 一、分类损失

### 1.1 交叉熵：对正确标签的预测概率取负对数

交叉熵（Cross Entropy，CE）在分类中衡量预测分布对标签的解释程度。以下从正确标签的预测概率写出损失。

#### 二元交叉熵

BCE（Binary Cross Entropy，二元交叉熵）用于二分类或多个独立的二分类输出。令 $\hat y_i\in(0,1)$ 为正类预测概率，$y_i\in\{0,1\}$。正确标签的预测概率为 $\hat y_i^{y_i}(1-\hat y_i)^{1-y_i}$；取负对数并平均，得到：

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^N [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

**适用场景**：二分类、多标签分类（每标签独立二分类）、Mask R-CNN 的 mask 预测。

#### 多类交叉熵

对互斥的 $K$ 个类别，令 $y_{i,k}$ 为目标分布、$\hat p_{i,k}$ 为预测概率。对各类别的负对数概率按目标分布加权，再对样本平均：

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \log \hat{p}_{i,k}$$

其中 $\hat{p}_{i,k} = \frac{e^{z_{i,k}}}{\sum_j e^{z_{i,j}}}$（Softmax）。当标签为 one-hot 时退化为 $-\frac{1}{N}\sum_i \log \hat{p}_{i,y_i}$。

**适用场景**：ImageNet 分类、Faster R-CNN 的 RoI 分类、所有互斥多分类。

#### BCE vs CE 的梯度对比

**[数学推导]** 对单样本损失 $\ell$ 求导，两者都得到“预测减目标”。注意，上面的平均损失 $\mathcal L$ 还需要乘以 $1/N$：

| 单样本损失 | 对未归一化分数（logit）的梯度 |
|------|-----------------|
| BCE+Sigmoid | $\hat{y}_i - y_i$ |
| CE+Softmax | $\hat{p}_k - \mathbb{1}[k=y_i]$ |

二分类中，令 $p=\sigma(z)$，则 $dp/dz=p(1-p)$。链式法则给出：

$$
\frac{d\ell}{dz}
=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)=p-y.
$$

多分类中，将 $p_k=e^{z_k}/\sum_j e^{z_j}$ 代入单样本交叉熵，利用 $\sum_k y_k=1$：

$$
\ell=-\sum_k y_kz_k+\log\sum_j e^{z_j},
\qquad
\frac{\partial\ell}{\partial z_k}
=-y_k+\frac{e^{z_k}}{\sum_j e^{z_j}}=p_k-y_k.
$$

梯度形式简洁，不等于任意计算方式都数值稳定。指数与对数的稳定计算另见 [卷积文档中的 Softmax 说明](convolution.md#softmax-stability)。

#### BCE vs CE 选择指南

| | BCE + Sigmoid | CE + Softmax |
|---|---|---|
| 标签关系 | 一个二分类；或多个分别预测的标签 | 一个输出位置对应一组互斥类别 |
| 输出和 | 无约束 | 和为 1 |
| 典型用途 | Mask 预测、Focal Loss | 分类任务、CLIP |

---

### 1.2 Focal Loss（焦点损失）

> **[论文发现]** Focal Loss 通过预测概率降低易分类样本的损失权重。[1，第 3–4 页]

**动机**：密集检测中，容易识别的背景候选可能远多于前景候选。交叉熵本身会给高置信度正确预测较小的损失，但大量背景项的总贡献仍可能较大。Focal Loss 进一步降低这类易分类项的权重。[1，第 3–4 页]

**公式**：

$$\mathcal{L}_{focal} = -\frac{1}{N} \sum_{i=1}^N \alpha_t (1-\hat{p}_t)^\gamma \log(\hat{p}_t)$$

- $y=1$ 时 $\hat p_t=\hat p$，$y=0$ 时 $\hat p_t=1-\hat p$（对正确类别的预测概率）
- $\alpha_t$：正负样本平衡权重（如正样本 0.25，负样本 0.75）
- $(1-\hat{p}_t)^\gamma$：**调制因子**（核心创新）

**调制因子的作用：以下取 $\gamma=2$**：

| $\hat{p}_t$ | 含义 | 调制因子 $(1-p_t)^2$ | 效果 |
|------------|------|---------------------|------|
| 0.1 | 预测错误，难样本 | ~0.81 | 正常学习，几乎不衰减 |
| 0.5 | 预测不确定 | 0.25 | 中等衰减 |
| 0.9 | 预测正确且确信，简单样本 | 0.01 | **大幅衰减**，几乎忽略 |

**适用场景**：单阶段检测器（RetinaNet）以及类别极度不平衡的分类任务（长尾分布）。

**[数学推导]** $\gamma=0$ 时调制因子为 1，退化为带类别权重的交叉熵。固定 $0<\hat p_t<1$ 时，增大 $\gamma$ 会减小该因子；实际取值需要结合训练效果选择。

---

### 1.3 标签平滑（Label Smoothing）

**[数学推导]** 下面采用把概率质量 $\epsilon$ 均匀分配给其他类别的定义，要求 $K>1$：

$$\tilde{y}_k = (1-\epsilon) \cdot \mathbb{1}[k=y] + \frac{\epsilon}{K-1} \cdot \mathbb{1}[k \neq y]$$

- ε=0.1 时，正确类目标 0.9，其余 K-1 类各 0.1/(K-1)

这些目标概率之和为 $(1-\epsilon)+(K-1)\epsilon/(K-1)=1$。在上一节的梯度式中将独热标签替换为平滑标签，就改变了优化所追求的目标概率。泛化与校准效果需在具体任务中验证。

---

## 二、回归损失

### 2.1 均方误差

MSE（Mean Squared Error，均方误差）对预测误差平方后取平均：

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2, \quad \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)$$

**[数学推导]** 单项平方误差对预测值求导为 $2(\hat y_i-y_i)$，平均后得到上式。误差绝对值越大，梯度绝对值越大，因此少量大误差项可能占据较大影响。

**[分析判断]** 适用于希望较强惩罚大误差的回归问题；若数据包含少量极端异常值，需要检查这些样本对目标函数的影响。

### 2.2 平均绝对误差

MAE（Mean Absolute Error，平均绝对误差）对预测误差取绝对值后平均：

$$\mathcal{L}_{MAE} = \frac{1}{N} \sum_i |y_i - \hat{y}_i|, \quad \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{N}\operatorname{sign}(\hat y_i-y_i)\quad(\hat y_i\ne y_i)$$

**[数学推导]** 绝对值函数在正半轴的导数为 1，负半轴为 −1；平均后除以 $N$。零误差处不可导，可以使用区间 $[-1/N,1/N]$ 中的次梯度。与平方误差相比，大误差不会继续增大对预测值的梯度绝对值。

**[分析判断]** 当希望减弱极端误差项的影响时，可与均方误差比较。图像是否更清晰、模型是否更鲁棒，还取决于数据和其他训练目标。

### 2.3 Smooth L1 Loss

下面给出带转折参数 $\beta>0$ 的分段定义：

$$\text{Smooth}_{L1}(x) = \begin{cases} 0.5 x^2 / \beta, & |x| < \beta \\ |x| - 0.5\beta, & |x| \geq \beta \end{cases}$$

**设计动机——结合 L1 和 L2 的优势**：

| 误差范围 | 行为 | 原因 |
|----------|------|------|
| 小误差（$\lvert x\rvert<\beta$） | 像 L2（平方） | 小误差需精确优化，L2 梯度随误差减小更平滑 |
| 大误差（$\lvert x\rvert\geq\beta$） | 像 L1（线性） | 大误差需鲁棒性，L1 恒定梯度不被异常值主导 |

**[数学推导]** 对分段定义求导，小误差区间得到 $x/\beta$，大误差区间得到 $\operatorname{sign}(x)$。在 $x=\pm\beta$ 两侧导数一致，因此转折处平滑。它限制的是损失对误差的导数，不能据此保证整个训练过程稳定。

**[分析判断]** 可以用于希望在小误差区间平滑优化、在大误差区间限制梯度幅值的回归任务。边界框采用何种坐标参数化和归一化方式，也会影响误差尺度。

---

## 三、边界框损失：重叠与几何关系

IoU（Intersection over Union，交并比）用两个区域的交集面积除以并集面积衡量重叠程度。以下 $A$ 为预测框、$B$ 为真实框，均假设具有正面积。

> 核心问题：L1/L2/Smooth L1 独立对待四个坐标分量，忽略了它们之间的关联，L1 Loss 小 ≠ IoU 高。

### 3.1 IoU Loss

$$\mathcal{L}_{IoU} = 1 - \frac{|A \cap B|}{|A \cup B|}$$

**[数学推导]** 当两框严格分离，且小幅坐标变化仍不能使其相交时，交集面积始终为 0，损失局部恒为 1。此时仅靠这一项无法提供使两框靠近的梯度。

### 3.2 广义交并比损失

GIoU（Generalized Intersection over Union，广义交并比）在交并比之外加入最小包围区域的惩罚。[2，第 3–4 页]

$$\mathcal{L}_{GIoU} = 1 - \left( \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|} \right)$$

**[数学推导]** $C$ 是包围两框的最小外接矩形，附加项衡量 $C$ 中未被两框覆盖的面积比例。它同时受位置和尺寸影响，应避免将其简单等同于中心距离。

当 $A$ 完全包含在 $B$ 内时，$C=B=A\cup B$，附加项为 0，GIoU 退化为 IoU。这不表示所有坐标梯度都为零：保持包含关系时，平移小框不改变重叠比例，但改变小框面积仍会改变 IoU。

### 3.3 完全交并比损失

CIoU（Complete Intersection over Union，完全交并比）进一步考虑中心距离和宽高比。[3，第 4–5 页]

$$\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

其中：

- $\rho$：中心点欧氏距离，$c$：外接矩形对角线
- $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$（长宽比惩罚）
- $\alpha = \frac{v}{(1-\text{IoU})+v}$（动态权重：IoU 高时长宽比权重更大）

### 3.4 IoU 系列对比

| 损失 | 在 $1-\mathrm{IoU}$ 之外加入什么 | 阅读时关注的问题 |
|---|---|---|
| IoU | 无 | 两框严格分离时缺少局部移动信号 |
| GIoU | 最小包围区域中未覆盖的比例 | 包含时附加项为零，不能理解为所有梯度都消失 |
| DIoU | 归一化中心距离 | 中心位置误差如何影响目标 |
| CIoU | 归一化中心距离与宽高比差异 | 位置、重叠和形状项如何共同作用 |

DIoU（Distance Intersection over Union，距离交并比）是加入归一化中心距离的形式；相关定义见 [3，第 3–5 页]。

---

## 四、对比学习损失

### 4.1 基于候选匹配的对比损失

下面给出批次内图像到文本的交叉熵形式。它也常称为 InfoNCE（Information Noise-Contrastive Estimation，信息噪声对比估计）形式：将正匹配与一组负匹配进行对比。此处直接从候选匹配概率理解它。

$$\mathcal{L}_{InfoNCE} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(v_i, t_j) / \tau)}$$

- $\text{sim}(v,t)=v \cdot t$（L2 归一化后为余弦相似度）
- 正样本：对角线上匹配的 $(v_i, t_i)$
- 负对：当前批次内未配对的图文组合
- $\tau>0$：温度参数，控制候选概率的集中程度

将相似度经 Softmax 转成匹配概率，再对正确匹配取负对数并平均，即得上式。双向目标与温度解释见 [CLIP 文档](../02-%E5%9F%BA%E7%A1%80%E8%A7%86%E8%A7%89%E4%BB%BB%E5%8A%A1/clip.md)。

### 4.2 三元组损失（Triplet Loss）

$$\mathcal{L}_{triplet} = \max(0, d(a, p) - d(a, n) + \text{margin})$$

- $a$：参考样本；$p$：与参考匹配的正样本；$n$：负样本
- margin：希望负样本距离比正样本距离至少多出的间隔

**[数学推导]** 当 $d(a,n)\geq d(a,p)+\text{margin}$ 时，损失为 0，否则惩罚间隔不足。每个三元组包含一个负样本，一个批次可以构造多个三元组；上节的匹配损失则在分母中比较多个候选项。

**[分析判断]** 两者都需要考虑负样本的选择、数量及错误负对，不能仅由公式断言某种方法不需要样本选择。

**适用场景**：人脸识别、度量学习、行人重识别。

---

## 五、分割损失

### 5.1 Dice Loss

$$\mathcal{L}_{Dice} = 1 - \frac{2|A \cap B| + \epsilon}{|A| + |B| + \epsilon}$$

上述集合形式便于说明重叠比例；用于梯度训练时，需要用连续预测概率定义可微的交集与区域大小，不能直接对阈值化后的集合求梯度。

**[分析判断]** 逐像素交叉熵与 Dice 类目标的聚合方式不同：前者累计像素预测误差，后者用区域大小归一化重叠。比较时应检查类别不平衡、空前景和归一化方式，而不能笼统认为某一项总对小目标更好。

**适用场景**：医学图像分割、语义分割。

---

## 六、生成对抗损失

GAN（Generative Adversarial Network，生成对抗网络）通过生成器与判别器的竞争学习数据分布。下式中，$G$ 是生成器，$D$ 是输出实数分数的判别器，$z$ 是用于生成样本的潜变量。

### 6.1 Hinge Loss

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_{data}}[\max(0, 1-D(x))] + \mathbb{E}_{z}[\max(0, 1+D(G(z)))]$$

$$\mathcal{L}_G = -\mathbb{E}_{z}[D(G(z))]$$

**[数学推导]** 对真实样本分数 $s=D(x)$，$\max(0,1-s)$ 在 $s<1$ 时导数为 −1，在 $s>1$ 时为 0；生成样本项同理具有零梯度区间。生成器目标对判别分数的导数为 −1，但传回生成器参数时仍需经过判别器的导数。因此不能把这一目标概括为“梯度不会饱和”或“必然更稳定”。

---

## 七、选择损失前需要明确什么

以下为 **[分析判断]**。选择损失时，先写清楚预测量、标签关系和误差含义，再比较候选目标。

| 问题 | 需要明确的条件 |
|---|---|
| 分类 | 一个位置只属于一类，还是可以同时具有多个标签？ |
| 回归 | 误差是否存在极端值？坐标或目标量是否经过归一化？ |
| 边界框 | 需要优化坐标偏差、区域重叠，还是中心与形状关系？ |
| 对比学习 | 正负匹配如何定义？未配对样本是否也可能语义相近？ |
| 分割 | 前景占比如何？如何处理没有前景的样本？ |
| 生成 | 模型预测的是数据、噪声还是其他量？目标权重如何设置？ |

混合多个损失时，应同时说明每项的归一化方式和权重。名称相同但采用求和或平均，可能给出不同的梯度尺度。

---

## 八、常见理论问题

### Q: Focal Loss 和加大负样本惩罚权重有什么区别？

**[数学推导]** 类别权重 $\alpha$ 对同一类别使用相同系数；Focal Loss 的 $(1-p_t)^\gamma$ 还随当前预测变化。高置信度正确预测的损失被进一步减小，但并非在所有有限概率下都严格等于零。

### Q: 为什么 DETR 用 L1+GIoU 两个回归损失？

**[论文发现]** DETR 联合使用归一化边界框坐标上的 L1 距离与 GIoU 损失。[4，第 4 页]

**[分析判断]** L1 直接约束各坐标分量的差异，GIoU 描述两个框的相对几何关系。两者关注的误差不同；不宜把它们简化为“绝对位置”和“没有位置信息”。

### Q: 噪声预测为什么常配合均方误差？

DDPM（Denoising Diffusion Probabilistic Models，去噪扩散概率模型）学习带噪样本的反向生成过程。**[论文发现]** 原论文比较了不同参数化与目标，并采用简化的噪声预测平方误差。[5，第 3–4 页]

**[分析判断]** 目标参数化与时间步权重应一起比较。不能用“原始数据的分布随时间步改变”解释其优势：从数据集中抽取的原始样本分布本身不随所选时间步变化。

## 论文出处

> [1] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. “Focal Loss for Dense Object Detection.” ICCV, 2017。焦点损失与调制因子见公开版第 3–4 页。[原文](https://arxiv.org/pdf/1708.02002)
>
> [2] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, Silvio Savarese. “Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression.” CVPR, 2019。定义与边界框损失见公开版第 3–4 页。[原文](https://arxiv.org/pdf/1902.09630)
>
> [3] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren. “Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.” AAAI, 2020。距离与宽高比项见公开版第 3–5 页。[原文](https://arxiv.org/pdf/1911.08287)
>
> [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko. “End-to-End Object Detection with Transformers.” ECCV, 2020。边界框损失见公开版第 4 页公式 (2)。[原文](https://arxiv.org/pdf/2005.12872)
>
> [5] Jonathan Ho, Ajay Jain, Pieter Abbeel. “Denoising Diffusion Probabilistic Models.” NeurIPS, 2020。反向过程参数化与简化目标见公开版第 3–4 页，公式 (11)–(14)。[原文](https://arxiv.org/pdf/2006.11239)
