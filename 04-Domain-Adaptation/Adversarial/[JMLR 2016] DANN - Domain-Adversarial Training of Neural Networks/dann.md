# DANN 精读与工业异常检测领域自适应调研

> **阅读草稿**：原文解读与应用拓展仍需核验；拓展设想不作为原论文结论。

> DANN（Domain-Adversarial Neural Network，领域对抗神经网络）  
> 归档原文：[DANN](paper.pdf)；[DSN](../%5BNIPS%202016%5D%20DSN%20-%20Domain%20Separation%20Networks/paper.pdf)

本文用 **[论文发现]** 标注所引论文直接报告的内容，用 **[数学推导]** 标注由明确前提推出的性质，用 **[业界共识]** 标注通用概念，用 **[分析判断]** 标注方法解释与应用设想。后续工作与应用设想会在相应段落中说明，不归因于原论文。

## 阅读建议

首次阅读从第 1–2 节的 [背景与正式设定](#paper-main)开始；第 0 节解释优化方向与任务分类的常见混淆。理解原方法后，再读第 3 节的 [异常检测拓展](#applications)，以及后续方法比较。

---

## 0. 概念对照：优化方向与任务设定

一种常见表述是：

> `-L_d` 力求最大化域分类损失，让模型混淆“特征来自源域还是目标域”，从而实现较好的 domain generalization。

这句话前半部分抓住了 DANN 的核心，但需要两处修正。

1. **[数学推导] `-L_d` 对不同参数没有同一个优化方向。**
   - 特征提取器最小化包含 `-L_d` 的目标，因此会**最大化**域分类损失，试图欺骗域分类器。
   - 域分类器最大化同一个鞍点目标，等价于**最小化**域分类损失，试图正确判断来源域。
   - 所以不能说“整个模型都在最大化 `L_d`”。竞争发生在特征提取器与域分类器之间。

2. **[业界共识] DANN 实现的是 Domain Adaptation（DA，领域自适应），不是 Domain Generalization（DG，领域泛化）。**
   - DANN 训练时显式使用无标签目标域样本，正式问题设定位于原文第 5 页。[1]
   - DG 的训练阶段看不到未来的目标域，只能依靠多源域、增强、因果不变性等手段面对未知域。

| 设置 | 训练时有源域数据 | 训练时有目标域数据 | 典型目标 |
|---|---:|---:|---|
| DA（Domain Adaptation，领域自适应） | 是 | 是，通常无标签或少标签 | 适配这个已知目标域 |
| DG（Domain Generalization，领域泛化） | 是 | 否 | 泛化到尚未见过的目标域 |

**[分析判断] 更准确的一句话是：**

> DANN 让域分类器尽量识别源域和目标域，同时通过反向梯度让特征提取器尽量破坏这种识别能力；二者达到对抗平衡后，特征应保留源域任务信息并减少域来源信息。由于训练使用了目标域数据，这属于无监督领域自适应，而不是领域泛化。

原文第 7 页写过“generalize well from one domain to another”，这里的 `generalize` 是普通英语动词，不是在把方法归类为 DG。[1]

---

<a id="paper-main"></a>

## 1. 理论与方法的时间线

- **2006**：Ben-David 等人把可由域分类器估计的分布差异用于领域自适应误差分析。[3]
- **2010**：该理论扩展为更完整的不同域学习理论，明确目标误差同时受源域误差、域差异和两域联合最优误差约束。[4]
- **2014-12**：Ajakan 等人在 NIPS Workshop 报告 Domain-Adversarial Neural Networks。[1, p.5]
- **2015-06**：Ganin 与 Lempitsky 在 ICML 发表 *Unsupervised Domain Adaptation by Backpropagation*，提出用反向传播实现领域对抗训练。[2]
- **2016-04**：JMLR 扩展版 DANN 正式发表，补充理论、浅层实验、图像分类、情感分析和行人重识别描述子学习。[1, pp.5, 26-30]
- **2016-12**：DSN（Domain Separation Network，领域分离网络）在 NIPS 发表，在共享表示之外显式建模源域和目标域的私有表示。[5]
- **2023**：IRAD 将共享/私有表示用于少量正常目标样本下的异常检测；GNL 系统研究分布偏移下的异常检测。[7, 8]
- **2025**：Two-fold Unsupervised Curse 工作指出“全量目标域直接对齐”会把目标异常也拉向源域正常；RoDA 则直接从 Memory Bank 和鲁棒最优传输切入工业异常检测。[9, 10]
- **2025-2026**：MVTec AD 2 引入真实照明变化等更困难工业场景，论文于 2026 年发表于 IJCV。[12]

**[业界共识]** 这条时间线说明 DANN 是“通用 UDA 基础模块”，并不是专为工业异常检测设计的方法。工业 AD 的关键新增困难是：源域通常只有正常样本，而无标签目标域可能同时包含正常和异常样本。

---

## 2. DANN 的正式问题设置

### 2.1 数据与三个网络组件

> UDA（Unsupervised Domain Adaptation，无监督领域自适应）：源域有任务标签，目标域样本可用于训练，但没有任务标签。

**[论文发现]** DANN 使用：

$$
S=\{(x_i,y_i)\}_{i=1}^{n},\qquad
T=\{x_i\}_{i=n+1}^{N}.
$$

- $S$：有标签源域样本。
- $T$：无任务标签的目标域样本。
- $G_f(x;\theta_f)$：特征提取器。
- $G_y(G_f(x);\theta_y)$：任务标签预测器，只在源域标签上训练。
- $G_d(G_f(x);\theta_d)$：域分类器，判断特征来自源域还是目标域。

数据定义见 DANN 第 5 页，通用深层网络记号见第 11 页。[1]

### 2.2 为什么“域不可区分”可能有用

> $\mathcal H$-divergence（$\mathcal H$ 散度）：用某一假设类中的分类器区分两个域的能力来刻画域差异。

**[论文发现]** 对源、目标样本训练二分类器，源域记为 0、目标域记为 1。若该分类器的泛化错误率为 $\epsilon$，PAD（Proxy A-distance，代理 A 距离）为：

$$
\widehat d_A=2(1-2\epsilon).
$$

公式来自 DANN 第 6 页 Equation (3)。[1]

- 当 $\epsilon=0$ 时，域可以被完美区分，$\widehat d_A=2$。
- 当平衡二分类的 $\epsilon=0.5$ 时，域不可区分，$\widehat d_A=0$。

**[论文发现]** DANN 第 7 页 Theorem 2 给出的有限样本上界为：

$$
\begin{aligned}
R_{D_T}(\eta) \leq {}& R_S(\eta)
+\sqrt{\frac{4}{n}\left(d\log\frac{2en}{d}+\log\frac{4}{\delta}\right)}
+\widehat d_{\mathcal H}(S,T) \\
&+4\sqrt{\frac{1}{n}\left(d\log\frac{2n}{d}+\log\frac{4}{\delta}\right)}
+\beta,
\end{aligned}
$$

其中：

- $R_{D_T}(\eta)$：目标域真实任务风险。
- $R_S(\eta)$：源域经验任务风险。
- $d$：假设类的 VC（Vapnik-Chervonenkis，瓦普尼克-切尔沃年基斯）维，用来度量假设类容量。
- $\widehat d_{\mathcal H}(S,T)$：经验域差异。
- 两个根号项：有限样本与模型容量产生的复杂度项。
- $\beta\geq\inf_{\eta^*\in\mathcal H}[R_{D_S}(\eta^*)+R_{D_T}(\eta^*)]$：两域联合最优分类器仍然必须承担的误差。

上界原始理论来自 Ben-David 等人的工作；这里采用 DANN 原文第 7 页的具体版本，避免不同论文对 $\mathcal H$ 散度和 $\mathcal H\Delta\mathcal H$ 散度的系数约定混用。[1, 3, 4]

**[数学推导]** 若暂时只看可由训练影响的主项，上界给出三个要求：

1. 源域任务误差要低。
2. 源、目标特征分布要难以区分。
3. 必须存在一个同时适合两域的任务函数，即 $\beta$ 不能过大。

DANN 主要直接优化前两项的替代目标，**并不直接优化或保证 $\beta$ 很小**。因此，“域完全混合”不是目标任务正确的充分条件。

### 2.3 DANN 的鞍点目标

设 $L_y$ 为源域任务损失，$L_d$ 为源域与目标域共同参与的域分类交叉熵。省略样本平均符号后，DANN 的核心目标是：

$$
E(\theta_f,\theta_y,\theta_d)
=L_y(\theta_f,\theta_y)-\lambda L_d(\theta_f,\theta_d),
$$

并求：

$$
(\widehat\theta_f,\widehat\theta_y)
=\arg\min_{\theta_f,\theta_y}E(\theta_f,\theta_y,\widehat\theta_d),
$$

$$
\widehat\theta_d
=\arg\max_{\theta_d}E(\widehat\theta_f,\widehat\theta_y,\theta_d).
$$

完整带样本平均的目标位于 DANN 第 11 页 Equations (10)-(12)；浅层网络的同一鞍点形式位于第 9 页 Equation (9)。[1]

### 2.4 `-L_d` 到底对谁做什么

| 参数 | 对 $E=L_y-\lambda L_d$ 的操作 | 等价效果 |
|---|---|---|
| 特征提取器 $\theta_f$ | 最小化 $E$ | 降低 $L_y$，同时增大 $L_d$ |
| 任务头 $\theta_y$ | 最小化 $E$ | 降低 $L_y$ |
| 域分类器 $\theta_d$ | 最大化 $E$ | 因为 $L_y$ 与它无关，所以等价于降低 $L_d$ |

**[数学推导]** 对应的随机梯度更新为：

$$
\theta_f\leftarrow\theta_f-\mu
\left(
\frac{\partial L_y}{\partial\theta_f}
-\lambda\frac{\partial L_d}{\partial\theta_f}
\right),
$$

$$
\theta_y\leftarrow\theta_y-\mu\frac{\partial L_y}{\partial\theta_y},
\qquad
\theta_d\leftarrow\theta_d-\mu\lambda\frac{\partial L_d}{\partial\theta_d}.
$$

推导只需分别对 $E$ 做梯度下降或上升；结果与 DANN 第 11 页 Equations (13)-(15) 一致。[1]

因此，“最大化 $L_d$”只对 $\theta_f$ 成立；对 $\theta_d$ 恰好相反。

### 2.5 GRL 如何把鞍点训练写成普通反向传播

> GRL（Gradient Reversal Layer，梯度反转层）：前向传播为恒等映射，反向传播把流向特征提取器的梯度乘以负数。

**[论文发现]** GRL 的伪函数定义为：

$$
R(x)=x,\qquad \frac{dR}{dx}=-I.
$$

公式位于 DANN 第 13 页 Equations (16)-(17)，结构图和文字解释位于第 12-13 页。[1]

**[数学推导]** 将经过 GRL 的联合目标记为两个正损失之和：

$$
L_{\text{GRL}}=L_y+\lambda L_d.
$$

域分类器位于 GRL 之后，更新 $\theta_d$ 时梯度不穿过 GRL，所以仍然下降 $L_d$。更新 $\theta_f$ 时，域损失梯度穿过 GRL 后变号：

$$
\frac{\partial L_{\text{GRL}}}{\partial\theta_f}
=\frac{\partial L_y}{\partial\theta_f}
-\lambda\frac{\partial L_d}{\partial\theta_f}.
$$

这正是上一节的特征更新。联合目标在数值上使用 $L_y+\lambda L_d$ 并不表示对抗符号消失；负号由 GRL 只施加到流向特征提取器的域损失梯度上。

### 2.6 “域分类准确率 50%”不能单独证明成功

**[数学推导]** 在源、目标样本平衡且域分类器接近最优时，域分类错误率接近 0.5 才对应较小的 PAD。

**[分析判断]** 训练中的域分类准确率接近 50% 还可能由以下原因造成：

- 域分类器容量不足或没有训练好。
- 学习率、$\lambda$ 或采样比例错误。
- 特征在快速变化，域分类器跟不上。
- 类别不平衡却仍用 50% 作为随机基线。

更可靠的诊断是：冻结训练后的特征提取器，重新训练一个容量足够且有独立验证集的域探针，再报告其准确率或 PAD。同时检查源域任务性能和目标域任务性能，不能只看域混淆。

还有一个更细的限制：对固定且不够强的域分类器单独最大化交叉熵，特征提取器可能只是让它“自信地预测错”，而非真正让分布相同。只有域分类器持续逼近其最优响应时，鞍点解释才成立。

### 2.7 `lambda` 调度不是装饰

**[论文发现]** DANN 的深层实验把训练进度记为 $p\in[0,1]$，使用：

$$
\lambda_p=\frac{2}{1+\exp(-10p)}-1.
$$

公式位于第 21 页。论文在第 21-22 页进一步说明：该变化系数只用于更新特征提取器，域分类器分支保持 $\lambda=1$，以免域分类器学得太慢。[1]

**[分析判断]** 训练初期特征尚未形成任务结构，过强域对抗容易先破坏任务信息。对 Memory Bank AD 做实验时，也应从较弱对齐逐渐增强，而不是一开始直接设大权重。

### 2.8 原论文已经说明了两件容易被忽略的事

1. **[论文发现] DANN 不是必然成功。** MNIST $\rightarrow$ SVHN 是原论文明确报告的失败案例，DANN 没有超过未适配模型，见第 25 页。[1]
2. **[论文发现] DANN 不局限于分类头。** 第 26-30 页将它用于行人重识别的 500 维描述子学习，最终仍以描述子距离进行匹配，并在八组跨数据集实验上改善识别表现。[1]

第二点与 Memory Bank AD 的“特征向量加距离”形式相近，说明对抗对齐可以作用在嵌入空间；但该实验仍有源域监督的身份对应和度量损失，不能直接当作 PatchCore 有效性的证据。

---

<a id="applications"></a>

## 3. DANN 能否接到 Memory Bank 工业异常检测

> AD（Anomaly Detection，异常检测）：学习正常模式，并对偏离正常模式的样本或区域给出较高异常分数。  
> IAD（Industrial Anomaly Detection，工业异常检测）：面向工业质检图像的异常检测。  
> Memory Bank（记忆库）：保存正常训练图像的 patch 特征，测试时以最近邻距离衡量异常程度。

### 3.1 先给结论

**[分析判断] DANN 可以作为 Memory Bank AD 的特征对齐模块，但不能原样插入 PatchCore。** 最主要的结构冲突不是模块连接方式，而是监督信号：

| DANN 原始设置 | PatchCore 类 Memory Bank AD |
|---|---|
| 源域有多类任务标签，$L_y$ 保持类别判别性 | 源训练集通常只有正常样本，没有可用的正常/异常分类头 |
| 目标域无任务标签，但默认任务标签空间与源域兼容 | 目标适配集可能混入异常，而源域没有对应异常模式 |
| 特征提取器可训练 | PatchCore 通常冻结 ImageNet 预训练 backbone |
| 训练后直接用任务头或描述子 | 特征变化后还要保证 Memory Bank 与测试特征处于同一空间 |

**[论文发现]** DANN 第 11 页还指出，为保留理论保证，域分类器产生的假设类应包含任务预测器的假设类，即 $\mathcal H_y\subseteq\mathcal H_d$。PatchCore 没有与原始 DANN 等价的监督任务头，这进一步说明二者并非直接同构。[1]

### 3.2 最大风险：目标异常污染对齐

**[数学推导]** 工业 AD 的源训练分布近似只有正常数据：

$$
P_S=P_S(X\mid Y=\text{normal}).
$$

若无标签目标适配集同时包含正常和异常，则：

$$
P_T=(1-\pi)P_T(X\mid Y=\text{normal})
+\pi P_T(X\mid Y=\text{anomaly}).
$$

普通 DANN 对齐的是整体边缘特征分布 $P_S(G_f(X))$ 与 $P_T(G_f(X))$，不知道第二项是异常。增大目标异常特征上的域分类损失，会推动它们也变得像源域正常特征，从而缩小异常与 Memory Bank 的距离。

这个推导说明：若 $\pi>0$，全量目标对齐存在固有的假阴性风险。2025 年 Two-fold Unsupervised Curse 论文用图示和正式问题定义明确讨论了同一问题，见第 1-4 页。[9]

### 3.3 第二个风险：PatchCore 没有防坍塌的任务损失

**[分析判断]** 原始 DANN 中 $L_y$ 阻止特征为了域混淆而丢掉任务语义。纯 PatchCore 没有训练任务头；若只训练 `backbone + GRL + domain head`，特征可以通过丢失大量信息来降低域可分性，最近邻异常检测所需的局部纹理差异也可能一起消失。

可行的约束至少要有一种：

- 冻结大部分 backbone，只训练小型 adapter、projection 或 BatchNorm 仿射参数。
- 对源域特征增加保持局部几何或保持预训练特征的约束。
- 引入适合一类学习的源域目标，而不是伪造一个恒为“正常”的分类头。
- 使用可控合成缺陷作为判别性约束，但必须验证合成模式没有替代真实缺陷语义。

这些属于待验证的设计，不是 DANN 原论文结论。

### 3.4 第三个风险：Memory Bank 会随特征空间一起失效

**[数学推导]** Memory Bank 最近邻异常分数可写为：

$$
s(x)=\max_p\min_{m\in\mathcal M}\lVert z_p(x)-m\rVert_2.
$$

RoDA 第 3 页 Equation (2) 给出了这一 PatchCore 式分数。[10]

若 $\mathcal M$ 由旧参数 $\theta_f^{(0)}$ 生成，而测试特征由适配后参数 $\theta_f^{(1)}$ 生成，距离比较发生在两个不同坐标系中。除非适配被严格约束为保持该空间，否则该距离没有原来的含义。

**[分析判断]** 正确流程应是：先完成适配并冻结特征提取器，再用同一版本的网络重建源域 Memory Bank；若有确认正常的目标样本，再决定是否将其加入 Bank。不要用适配前的旧 Bank 评估适配后的特征。

### 3.5 哪些工业场景更适合 DANN

| 目标侧条件 | 是否建议直接测试 DANN | 判断 |
|---|---:|---|
| 有一批人工确认正常的目标图像 | 是，作为基线 | 污染风险最低；先比较“直接加入目标 Bank”是否已足够 |
| 目标无标签，但异常极少 | 有条件 | 先筛选高置信正常 patch，再做加权或子集对齐 |
| 目标无标签且异常比例未知 | 否 | 全量 DANN 可能把异常拉向正常；优先鲁棒对齐或显式异常过滤 |
| 训练时完全没有目标数据 | 否 | 这是 DG，不是 DANN 的设置 |
| 偏移主要是光照、传感器、轻微背景变化 | 较适合 | 更接近“任务语义不变、域风格变化”的协变量偏移 |
| 偏移包含新零件结构、视角导致的可见面变化 | 高风险 | 共享任务函数假设更可能失效，$\beta$ 可能增大 |

**[论文发现]** DSN 本身也明确假设源、目标主要在低层图像统计上不同，而高层参数分布和标签空间相近，见 DSN 第 2 页。[5] 这个假设与照明、颜色、噪声较吻合，与大幅几何和产品结构变化并不等价。

### 3.6 用于区分机制的最小对照

**[分析判断]** 下列对照可分别检验目标域正常样本、特征对齐和异常污染的影响：

| 组别 | 训练/适配方式 | 目的 |
|---|---|---|
| A | Source-only PatchCore | 原始下界 |
| B | 将少量“确认正常”的目标特征直接加入或重建 Bank | 判断问题是否仅靠补充正常上下文即可解决 |
| C | Oracle-normal DANN，只用确认正常目标 patch | 验证“无污染时 DANN 是否有价值” |
| D | Vanilla DANN，使用全量无标签目标 patch | 显式测量异常污染造成的退化，不应默认它是最终方案 |
| E | Filtered/weighted DANN，只对齐高置信正常目标 patch | 验证异常筛选能否修复 DANN |
| F | RoDA 式鲁棒最优传输 | 与最贴近 Memory Bank 的近年方法比较 |

实验约束：

- backbone 先冻结，只调整轻量 adapter 或 BatchNorm 仿射参数，以限制特征漂移。
- 使用 patch-level 域头，并保留 global-level 域头作为消融；工业缺陷通常是局部的。
- 源、目标 patch 数量平衡采样，避免域分类器利用数量先验。
- 适配池与最终目标测试集分开；目标标签只用于最终评估，不用于调参。
- 每次特征参数变化后重建 Bank，并固定同样的 coreset 比例。
- 同时报源域与目标域性能，监测适配是否牺牲源域正常结构。

至少记录：

- 图像级 AUROC（Area Under the Receiver Operating Characteristic，ROC 曲线下面积）。
- 像素级 AUROC 与 AUPRO（Area Under the Per-Region Overlap，区域重叠曲线下面积）。
- 目标域正常样本的假阳性率。
- 目标正常 patch 到 Bank 的距离分布，以及目标异常 patch 到 Bank 的距离分布。
- 冻结特征后独立域探针的准确率或 PAD。

**停止条件**：若域探针更难区分了，但正常/异常距离间隔同步缩小，说明模型只完成了域混淆，没有保住异常判别性。

---

## 4. DSN 精读：它比 DANN 多解决了什么

> DSN（Domain Separation Network，领域分离网络）：将表示显式拆成跨域共享部分和每个域独有的私有部分，并用重构与差异约束防止拆分退化。

### 4.1 结构与总损失

**[论文发现]** DSN 包含：

- 共享编码器 $E_c$，源域与目标域共用。
- 源域私有编码器 $E_p^s$ 与目标域私有编码器 $E_p^t$。
- 共享解码器 $D$，用共享加私有表示重建输入。
- 任务头 $G$，只读取共享表示完成源域监督任务。

总损失为：

$$
L=L_{\text{task}}+\alpha L_{\text{recon}}
+\beta L_{\text{difference}}+\gamma L_{\text{similarity}}.
$$

公式位于 DSN 第 4 页 Equation (1)。[5]

共享与私有表示的差异损失为：

$$
L_{\text{difference}}
=\left\lVert {H_c^s}^{\!T}H_p^s\right\rVert_F^2
+\left\lVert {H_c^t}^{\!T}H_p^t\right\rVert_F^2,
$$

位于第 4 页 Equation (5)。相似性损失可以使用 DANN/GRL，也可以使用 MMD（Maximum Mean Discrepancy，最大均值差异），见第 4-5 页 Equations (6)-(7)。[5]

### 4.2 两个需要直接指出的理论问题

#### 软正交不等于统计独立

**[数学推导]** $\lVert H_c^T H_p\rVert_F^2$ 变小，只约束有限批次中共享维度与私有维度的线性内积接近零。零相关在一般非高斯分布下不推出统计独立。因此论文中“encourages independence”应理解为软代理目标，不能理解为已经严格分离了因果因素。

#### 论文所谓 scale-invariant MSE 的实际不变性

**[论文发现]** DSN 第 4 页 Equation (4) 定义：

$$
L_{\text{si-mse}}(x,\widehat x)
=\frac{1}{k}\lVert x-\widehat x\rVert_2^2
-\frac{1}{k^2}\left((x-\widehat x)^T\mathbf 1\right)^2.
$$

令残差 $r_i=x_i-\widehat x_i$，$\bar r=\frac1k\sum_i r_i$，则：

$$
L_{\text{si-mse}}
=\frac1k\sum_i r_i^2-\bar r^2
=\frac1k\sum_i(r_i-\bar r)^2.
$$

**[数学推导]** 这就是残差方差。给全部残差加常数 $c$ 时方差不变；把残差乘 $a$ 时损失变成 $a^2L$。所以直接用于原始像素时，它严格具有的是“全局加性残差偏移不变”，并非一般意义上的乘法尺度不变。

### 4.3 DSN 对工业 AD 的潜力与风险

**[分析判断]** DSN 的直觉比裸 DANN 更贴合“缺陷语义与成像域因素分开”的目标：共享分支保存产品结构，私有分支吸收照明、颜色和传感器特性，再用共享 patch 特征构建 Memory Bank。

但它引入三个新风险：

1. 目标私有分支可能把真正的缺陷也当成“目标域特有信息”吸收，导致共享表示中的异常信号减弱。
2. 原论文任务是全局分类与姿态估计，工业微小缺陷要求 patch-level 共享/私有分解，不能照搬全局向量。
3. **[论文发现]** DSN 实验使用少量有标签目标验证集选择超参数，并明确说无监督 DA 的通用验证仍是开放问题，见第 5-6 页。[5] 因此训练损失虽不使用目标任务标签，完整模型选择流程并非严格 target-label-free。

**[分析判断]** 实验优先级应当是：先做轻量 DANN/鲁棒对齐基线，确认“对齐确实有收益”后，再投入 DSN 式共享/私有 patch 表示。否则 DSN 的重构器和私有分支会让失败原因难以定位。

---

## 5. 近年 DA for AD 与工业域偏移工作

以下方法不全属于严格的 DA。把 DG、测试时适配和真正使用目标域训练数据的方法分开，是避免概念混淆的必要条件。

| 方法 | 设置 | 与 DANN/Memory Bank 的关系 | 证据与限制 |
|---|---|---|---|
| IRAD | 少量确认正常的目标样本 | 共享/私有编码器思路接近 DSN，之后在共享表示上训练 Isolation Forest | **[论文发现]** 需要少量正常目标样本；实验是数字和 Office-Home 全局语义异常，不是工业局部缺陷，见公开版第 1-4、6-8 页。[7] |
| GNL / ADShift | DG，训练时没有目标域；推理时做特征分布匹配 | 不是 DANN；说明没有目标数据时应走增强和分布不变正常性学习 | **[论文发现]** 在 MVTec corruption、PACS、MNIST-M 等偏移上验证，建立在 RD4AD 上；依赖训练增强，推理还需要随机源域正常样本，见第 1-6 页。[8] |
| Two-fold Unsupervised Curse | 源域只有正常、目标域无标签且含异常的 UDA | 先用聚类找目标主簇，再只将其与源正常对齐；可以替换成 GRL | **[论文发现]** 第 8 页 Table 7 中，筛选后 GRL 从 source-only 的 62.84% AUC 提升到 71.84%，但低于对比对齐的 75.54%；只验证语义 one-vs-all，全局特征，不是工业细粒度缺陷，限制见第 8 页。[9] |
| RoDA | 少量无标签目标适配数据，可含异常 | 直接建立在 PatchCore 式 Memory Bank 上，用离散 Sinkhorn 分配与目标增强减少异常错误匹配 | **[论文发现]** 只更新 BatchNorm 仿射参数，见第 3-5 页；实验采用 MVTec、RealIAD、MVTec 3D 的合成 corruption，尚不能替代真实产线验证，见第 5-8 页。[10] |
| AeBAD + MMR | 训练时正常、测试时有真实光照/视角偏移；主要是鲁棒 AD/DG 基准 | 提供真实工业域偏移数据；MMR 用遮挡多尺度重构提升稳健性，不是目标域驱动的 DANN | **[论文发现]** AeBAD-S/V 包含视角、光照、尺度和不对齐变化，见第 1-5 页。[11] |
| MVTec AD 2 | 数据集，不是适配方法 | 可验证照明条件变化下的 Memory Bank/DANN | **[论文发现]** 包含 8 个高分辨率场景和真实照明条件变化测试，见公开版第 1 页。[12] |

### 5.1 与 Memory Bank 路线最直接相关的两篇后续工作

**[分析判断]** 若目标是判断 DANN 是否值得做，优先读：[9] 和 [10]。

- [9] 直接回答“目标集含异常时为何不能全量 DANN”，还给出了筛选后 GRL 的消融结果。但它没有完成工业局部缺陷验证。
- [10] 直接以 Memory Bank 为源分布表示，显式处理异常 patch 的错误对齐。它在结构上比 IRAD 或 DSN 更接近当前主方法。

这两篇合起来给出的研究假设是：

> 领域对齐本身可能有效，但工业 AD 的首要问题不是选择 GRL、MMD 还是 OT，而是先保证参与对齐的目标特征大概率属于正常模式。

这里 OT（Optimal Transport，最优传输）指在带质量约束的代价矩阵上寻找源、目标样本的最小总代价匹配；Sinkhorn 方法通过熵正则高效近似求解。RoDA 的具体公式和离散化规则位于第 3-5 页。[10]

---

## 6. 结论

1. **[分析判断] 当前对 `-L_d` 的梯度直觉基本正确，但必须补上参数对象：特征提取器最大化域损失，域分类器最小化域损失。**
2. **[业界共识] DANN 是使用无标签目标域的 DA，不是看不到目标域的 DG。**
3. **[分析判断] 对 Memory Bank 工业 AD，DANN 最适合先在“目标适配样本已确认正常”的条件下做可行性实验。**
4. **[分析判断] 若无标签目标池可能含异常，Vanilla DANN 应只作为反例基线；主方案至少需要正常 patch 筛选、置信加权或 RoDA 式鲁棒匹配。**
5. **[分析判断] 在确认简单对齐有效前，不应先上完整 DSN。DSN 更强的表达能力同时带来“缺陷被私有分支吸收”的额外失败模式。**
6. **[分析判断] 第一轮实验的核心问题不是“域分类器能否降到 50%”，而是“降低域差异后，目标正常与目标异常到正常 Memory Bank 的距离间隔是否扩大”。**

---

## 论文引用

> [1] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky. “Domain-Adversarial Training of Neural Networks.” *Journal of Machine Learning Research*, 17(59):1-35, 2016.  
> 本文引用位置：UDA 定义 p.5；$\mathcal H$-divergence 与 PAD pp.6-7；目标和梯度 pp.9-13；调度 pp.21-22；失败案例 p.25；描述子学习 pp.26-30。

> [2] Yaroslav Ganin, Victor Lempitsky. “Unsupervised Domain Adaptation by Backpropagation.” *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:1180-1189, 2015.  
> 本文引用位置：方法时间线与 ICML 前身版本，pp.1180-1189。

> [3] Shai Ben-David, John Blitzer, Koby Crammer, Fernando Pereira. “Analysis of Representations for Domain Adaptation.” *Advances in Neural Information Processing Systems 19 (NIPS)*, pp.137-144, 2006.  
> 本文引用位置：域分类器估计域差异的理论起点，pp.137-144。

> [4] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, Jennifer Wortman Vaughan. “A Theory of Learning from Different Domains.” *Machine Learning*, 79:151-175, 2010.  
> 本文引用位置：领域自适应目标风险理论，尤其 pp.157-160；本文展示的具体有限样本公式采用 [1] p.7 的版本。

> [5] Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan. “Domain Separation Networks.” *Advances in Neural Information Processing Systems 29 (NIPS)*, pp.343-351, 2016.  
> 归档公开版引用位置：假设 p.2；结构与损失 pp.3-5；目标验证集和实验 pp.5-8。

> [6] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. “Towards Total Recall in Industrial Anomaly Detection.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.14318-14328, 2022.  
> 本文引用位置：Memory Bank 与最近邻工业异常检测背景，主要见论文 pp.14320-14323；本文展示的明确最近邻公式采用 [10] p.3 Equation (2)。

> [7] Ziyi Yang, Iman Soltani, Eric Darve. “Anomaly Detection with Domain Adaptation.” *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp.2958-2967, 2023. DOI: 10.1109/CVPRW59228.2023.00297.  
> 公开 arXiv 版本引用位置：问题设置和 IRAD 结构 pp.1-4；实验与限制 pp.6-8。

> [8] Tri Cao, Jiawen Zhu, Guansong Pang. “Anomaly Detection Under Distribution Shift.” *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp.6511-6523, 2023.  
> 公开版引用位置：问题和 GNL 概览 pp.1-3；训练与测试时分布匹配 pp.4-5；数据和实验 pp.6-8。

> [9] Nesryne Mejri, Enjie Ghorbel, Anis Kacem, Pavel Chernakov, Niki Foteinopoulou, Djamila Aouada. “When Unsupervised Domain Adaptation Meets One-class Anomaly Detection: Addressing the Two-fold Unsupervised Curse by Leveraging Anomaly Scarcity.” *arXiv:2502.21022v3*, 2025, pp.1-10.  
> 本文引用位置：问题定义 pp.1-4；对齐消融与工业细粒度限制 pp.7-8。

> [10] Jingyi Liao, Xun Xu, Yongyi Su, Rong-Cheng Tu, Yifan Liu, Dacheng Tao, Xulei Yang. “Robust Distribution Alignment for Industrial Anomaly Detection under Distribution Shift.” *arXiv:2503.14910v1*, 2025, pp.1-10.  
> 本文引用位置：Memory Bank 和问题定义 pp.1-3；OT、离散 Sinkhorn 与 BatchNorm 更新 pp.3-5；实验与限制 pp.5-8。

> [11] Zilong Zhang, Zhibin Zhao, Xingwu Zhang, Chuang Sun, Xuefeng Chen. “Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction.” *Computers in Industry*, 151:103990, 2023. DOI: 10.1016/j.compind.2023.103990.  
> 公开 arXiv 版本引用位置：AeBAD 的域偏移设定与数据组成 pp.1-5。

> [12] Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger. “The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection.” *International Journal of Computer Vision*, 134(4), 2026. DOI: 10.1007/s11263-026-02743-0.  
> 公开 arXiv:2503.21622v1 引用位置：数据规模、场景类型和照明条件变化，p.1。

## 论文主页

> JMLR DANN 论文主页：https://www.jmlr.org/papers/v17/15-239.html
