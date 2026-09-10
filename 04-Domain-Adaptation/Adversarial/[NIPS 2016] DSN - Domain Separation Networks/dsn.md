# Domain Separation Networks 精读与方法拓展

> **阅读草稿**：原文解读与应用拓展仍需核验；拓展设想不作为原论文结论。

> 正式名称：DSN（Domain Separation Network，领域分离网络）  
> 归档原文：[Domain Separation Networks](paper.pdf)
> 论文组成：主文 9 页，补充材料 6 页，共 15 页。

本文用 **[论文发现]** 标注所引论文直接报告的内容，用 **[数学推导]** 标注由明确前提推出的性质，用 **[业界共识]** 标注通用概念，用 **[分析判断]** 标注方法解释与应用设想。后续工作与应用设想会在相应段落中说明，不归因于原论文。

## 阅读建议

先读第 1–4 节的 [问题、结构与损失](#paper-main)，再看第 5–6 节的实验与方法比较。第 7–8 节的 [应用拓展](#applications)用于讨论设计可能性。

---

## 0. 一句话结论

**[分析判断]** DANN（Domain-Adversarial Neural Network，领域对抗神经网络）只要求网络学习“域不可区分”的共享表示；DSN 认为这还不够，因为域特有信息没有地方安放，可能继续污染共享表示。因此 DSN 同时学习：

- 一个跨域共享表示，负责源域任务并尽量去除域信息；
- 一个源域私有表示和一个目标域私有表示，负责吸收各自域特有的信息；
- 一个共享解码器，强迫共享与私有表示合起来仍能解释输入。

DSN 不是简单替代 DANN。原论文表现最好的版本恰好是 **DSN + DANN similarity loss**，即用 DANN 对齐共享空间，再用私有空间、重构损失和正交约束补足 DANN。[1, pp.3-8]

---

<a id="paper-main"></a>

## 1. 论文要解决的问题

> UDA（Unsupervised Domain Adaptation，无监督领域自适应）：源域具有任务标签，目标域数据参与训练但没有任务标签。

**[论文发现]** 论文考虑一个有标签源域和一个无标签目标域：

$$
X^s=\{(x_i^s,y_i^s)\}_{i=1}^{N_s},
\qquad
X^t=\{x_i^t\}_{i=1}^{N_t}.
$$

符号和问题定义位于原文第 3 页。[1]

论文的出发点是：仅要求源、目标共享表示相似，可能让与域相关的噪声继续混进共享表示。显式给域特有信息分配一个私有子空间，有机会让共享空间更纯净。[1, pp.1-3]

### 1.1 方法成立依赖的假设

**[论文发现]** DSN 假设源域和目标域主要在低层图像统计上不同，例如：

- 噪声；
- 分辨率；
- 光照；
- 颜色。

同时假设高层参数分布相近，并具有相同标签空间。论文任务还假设目标物体位于图像前景，见第 1-2 页。[1]

**[分析判断]** 这不是一个小前提。若两个域的物体类别、结构、姿态分布或标注函数本身差异很大，DSN 不能保证把差异放进私有空间后，剩余共享表示仍足以完成任务。

---

## 2. 网络结构

### 2.1 四个核心组件

**[论文发现]** DSN 包含：

1. 共享编码器 $E_c(x;\theta_c)$：源域、目标域共用，输出共享表示 $h_c$。
2. 私有编码器 $E_p^s(x;\theta_p^s)$ 与 $E_p^t(x;\theta_p^t)$：每个域一个，输出私有表示 $h_p^s$ 或 $h_p^t$。
3. 共享解码器 $D(h;\theta_d)$：读取共享表示与对应私有表示之和，重建输入。
4. 任务头 $G(h;\theta_g)$：只读取共享表示，在源域标签上训练。

结构见原文第 3 页 Figure 1。[1]

### 2.2 训练和推理的信息流

对域 $r\in\{s,t\}$，重建为：

$$
\widehat x^r=D\left(E_c(x^r)+E_p^r(x^r)\right).
$$

任务预测为：

$$
\widehat y=G(E_c(x)).
$$

两个公式位于原文第 3 页 Section 3.1 开头。[1]

**[分析判断]** 这里有三个重要设计：

- 任务头看不到私有表示，迫使完成任务所需的信息进入共享表示。
- 解码器同时需要共享和私有表示，避免私有分支完全无用。
- 两部分以加法融合，因此共享表示和私有表示必须具有相同维度；原文各实验网络也按这一方式配置，见补充材料第 13-15 页网络图。[1]

### 2.3 DSN 希望形成怎样的分工

| 表示 | 希望包含 | 希望排除 | 受到的主要损失 |
|---|---|---|---|
| 共享 $h_c$ | 类别、姿态、跨域稳定结构 | 背景、颜色、成像域差异 | task、similarity、difference、reconstruction |
| 源域私有 $h_p^s$ | 源域特有外观 | 跨域任务语义 | difference、reconstruction |
| 目标域私有 $h_p^t$ | 目标域特有外观 | 跨域任务语义 | difference、reconstruction |

**[分析判断]** “希望包含”不等于“理论保证包含”。DSN 没有可辨识性定理来唯一决定某个因素必须落在哪个子空间，最终分工由四项损失的相互制衡产生。

---

## 3. 四项损失逐项精读

### 3.1 总目标

**[论文发现]** DSN 最小化：

$$
L=L_{\text{task}}
+\alpha L_{\text{recon}}
+\beta L_{\text{difference}}
+\gamma L_{\text{similarity}}.
$$

公式位于原文第 4 页 Equation (1)。其中 $\alpha,\beta,\gamma$ 分别控制重构、共享/私有差异和跨域相似性的权重。[1]

### 3.2 任务损失：规定共享空间必须有用

**[论文发现]** 分类任务使用源域负对数似然：

$$
L_{\text{task}}
=-\sum_{i=1}^{N_s}y_i^s\cdot\log \widehat y_i^s,
\qquad
\widehat y_i^s=G(E_c(x_i^s)).
$$

公式位于第 4 页 Equation (2)。由于目标域无标签，该项只作用于源域。[1]

**[分析判断]** 任务损失的作用不只是训练分类器。它还规定了“共享”的语义：能够完成源域任务的信息必须留在 $E_c$ 中。没有这一项，共享空间可能只追求域相似，却丢掉任务判别性。

### 3.3 重构损失：防止表示丢失输入信息

**[论文发现]** 重构损失同时用于源域和目标域：

$$
L_{\text{recon}}
=\sum_{i=1}^{N_s}L_{\text{si-mse}}(x_i^s,\widehat x_i^s)
+\sum_{i=1}^{N_t}L_{\text{si-mse}}(x_i^t,\widehat x_i^t).
$$

论文采用所谓 scale-invariant MSE（scale-invariant Mean Squared Error，尺度不变均方误差）：

$$
L_{\text{si-mse}}(x,\widehat x)
=\frac1k\lVert x-\widehat x\rVert_2^2
-\frac1{k^2}\left((x-\widehat x)^T\mathbf 1_k\right)^2.
$$

公式位于第 4 页 Equations (3)-(4)。$k$ 是像素数。[1]

令残差 $r_i=x_i-\widehat x_i$，残差均值为 $\bar r$，则：

$$
\begin{aligned}
L_{\text{si-mse}}
&=\frac1k\sum_i r_i^2-\left(\frac1k\sum_i r_i\right)^2 \\
&=\frac1k\sum_i(r_i-\bar r)^2.
\end{aligned}
$$

**[数学推导]** 它实际等于残差方差：

- 将所有残差加同一个常数 $c$，损失不变。
- 将残差整体乘以 $a$，损失变成 $a^2L_{\text{si-mse}}$。

因此，将该损失直接用于原始像素时，严格成立的是“全局加性残差偏移不变”，并不是对任意乘法尺度变化都不变。这个结论由 Equation (4) 完整展开得到。

### 3.4 Difference loss：让共享与私有表示软正交

将一个 batch 内的共享表示按行组成 $H_c^s,H_c^t$，私有表示组成 $H_p^s,H_p^t$。**[论文发现]** 差异损失为：

$$
L_{\text{difference}}
=\left\lVert {H_c^s}^T H_p^s\right\rVert_F^2
+\left\lVert {H_c^t}^T H_p^t\right\rVert_F^2.
$$

公式位于第 4 页 Equation (5)。$\lVert\cdot\rVert_F$ 是 Frobenius 范数。[1]

**[数学推导]** 若该损失为零，则 batch 中共享维度与私有维度的线性内积为零。这只说明软正交或线性不相关趋势，不能推出一般分布下的统计独立。只有在额外的联合高斯等强假设下，零相关才足以推出独立；DSN 没有作出该保证。

### 3.5 Similarity loss：只对齐共享空间

DSN 把跨域对齐限制在 $h_c$，私有表示不需要对齐。论文实现了三种选择。

#### 选择一：DANN/GRL

> GRL（Gradient Reversal Layer，梯度反转层）：前向为恒等映射，反向把传给共享编码器的域分类梯度反号。

**[论文发现]** DSN 写出的域对数似然为：

$$
L_{\text{similarity}}^{\text{DANN}}
=\sum_{i=1}^{N_s+N_t}
\left[d_i\log\widehat d_i+(1-d_i)\log(1-\widehat d_i)\right].
$$

公式位于第 4-5 页 Equation (6)。域分类器最大化该对数似然，共享编码器通过 GRL 最小化它；若改用正的交叉熵记号，两个方向恰好相反。[1]

#### 选择二：MMD

> MMD（Maximum Mean Discrepancy，最大均值差异）：在核特征空间中比较两个样本分布均值的距离。

**[论文发现]** DSN 使用有偏的平方 MMD 估计：

$$
\begin{aligned}
L_{\text{similarity}}^{\text{MMD}}
={}&\frac1{N_s^2}\sum_{i,j}\kappa(h_{ci}^s,h_{cj}^s)
-\frac2{N_sN_t}\sum_{i,j}\kappa(h_{ci}^s,h_{cj}^t) \\
&+\frac1{N_t^2}\sum_{i,j}\kappa(h_{ci}^t,h_{cj}^t).
\end{aligned}
$$

公式位于第 5 页 Equation (7)。论文采用多个 RBF（Radial Basis Function，径向基函数）核的线性组合，以覆盖训练过程中不断变化的特征尺度。[1]

#### 选择三：CorReg

> CorReg（Correlation Regularization，相关矩阵正则）：匹配源、目标共享表示的二阶相关统计量。

**[原论文补充材料]** 作者又提出：

$$
L_{\text{similarity}}^{\text{CorReg}}
=\left\lVert {H_c^s}^T H_c^s-{H_c^t}^T H_c^t\right\rVert_F^2.
$$

公式位于补充材料第 11 页 Equation (8)。它可以看作将 CORAL（Correlation Alignment，相关对齐）改造成可端到端训练的正则项。[1]

**[分析判断]** DSN 的主体并不依赖某一种对齐算法。它提供“共享/私有分解”的外层结构，DANN、MMD 或 CorReg 负责让共享分布相似。

---

## 4. 四项损失如何互相制衡

| 缺少的损失 | 可能出现的退化解 |
|---|---|
| 无 $L_{\text{task}}$ | 共享表示域不变，但不包含完成任务所需的信息 |
| 无 $L_{\text{recon}}$ | 私有编码器可以被忽略；编码器也可以丢弃大量输入信息 |
| 无 $L_{\text{difference}}$ | 共享与私有表示重复编码同一信息，所谓分离只剩名称 |
| 无 $L_{\text{similarity}}$ | 共享表示仍可携带明显域信息，源任务头难以迁移到目标域 |

**[分析判断]** 四项损失不是分别把某类因素“精确送到”某个分支，而是在排除几类明显的退化方案。不同分解仍可能得到相近的总损失，所以共享/私有语义不是唯一可辨识的。

### 4.1 一个仍然存在的退化自由度

假设某个可重构因素同时出现在 $h_c$ 与 $h_p$ 中，只要二者经过旋转或非线性变换后 batch 内积较小，difference loss 仍可能很低；解码器也依然能够重建输入。

**[分析判断]** 因此，Figure 2 中“共享重构像前景、私有重构像背景”的可视化是支持性证据，不是分离正确性的证明。原文可视化位于第 7-8 页，更多场景位于补充材料第 11-12 页。[1]

---

## 5. 训练细节与实验结论

### 5.1 训练协议

**[论文发现]** 主要训练细节如下：

- 使用带 momentum 的 SGD（Stochastic Gradient Descent，随机梯度下降）。
- 每个 batch 取 32 个源域样本和 32 个目标域样本，共 64 个。
- 输入做均值中心化，并缩放到 $[-1,1]$。
- 为避免早期适配损失干扰主任务，额外 DA 损失在训练 10,000 step 后才开启。
- 补充材料给出的 DSN 初始学习率为 0.01，并搜索 $\alpha,\beta,\gamma$。

主文训练细节见第 7 页，超参数范围见补充材料第 13 页。[1]

**[原论文勘误性观察]** 学习率衰减存在文本不一致：主文第 7 页写每 20,000 step 乘 0.9，补充材料第 13 页写每 20,000 iteration 乘 0.95。这是两种不同的衰减设置，不能视为同一数值记载。

### 5.2 最重要的结果不是单个准确率，而是三组对照

**[论文发现]** Table 1 在四个分类迁移任务上比较了 DANN、MMD、DSN-MMD 与 DSN-DANN。DSN-DANN 分别达到 83.2、91.2、82.7、93.1，四项均为表中无监督方法最好结果，见第 6 页。[1]

**[分析判断]** 这组结果支持两个结论：

1. 共享/私有分解在这些实验中确实补充了单独的 DANN/MMD。
2. 对齐策略仍然重要，DSN-DANN 整体优于 DSN-MMD；DSN 不会自动消除不同 similarity loss 的差异。

**[论文发现]** Table 3 的消融显示：移除 difference loss，或把 scale-invariant MSE 换成普通 MSE，四个任务都下降，见第 8 页。[1]

**[论文发现]** 在 Synthetic Objects $\rightarrow$ LINEMOD 中，DSN-DANN 的分类准确率为 100%，平均姿态角误差为 $53.27^\circ$；DANN 为 99.90% 和 $56.58^\circ$，但 target-only 的角误差只有 $6.47^\circ$，见第 7 页 Table 2。[1]

**[分析判断]** 分类准确率接近饱和并不代表共享表示已经完全迁移，姿态误差仍揭示了很大的目标域差距。

### 5.3 “无监督”模型选择并不严格

**[论文发现]** 作者尝试 reverse validation，但发现它与目标测试准确率并不总一致，最后使用小规模有标签目标验证集为所有方法选择超参数，见第 5-6 页。[1]

**[分析判断]** DSN 的训练损失不使用目标任务标签，但论文完整实验流程使用了目标标签做模型选择。因此，论文报告的完整流程并非严格的 target-label-free（不使用目标域标签）流程。

---

## 6. DSN 与 DANN 的准确关系

| 维度 | DANN | DSN-DANN |
|---|---|---|
| 共享特征提取器 | 有 | 有 |
| 域分类器 + GRL | 有 | 有，只连接共享表示 |
| 域私有编码器 | 无 | 源域、目标域各一个 |
| 重构输入 | 无 | 有，共享解码器读取共享 + 私有表示 |
| 共享/私有正交约束 | 无 | 有 |
| 主要风险 | 为了域不变而抹掉任务信息 | 分解不唯一，任务或异常信息可能进入私有分支 |

**[分析判断]** DSN-DANN 可以写成：

> DANN 的共享空间对齐 + 两个域私有残差通道 + 重构约束 + 软正交约束。

因此，理解 DSN 的前提不是忘掉 DANN，而是明确 DANN 在 DSN 中只负责 $L_{\text{similarity}}$。

---

<a id="applications"></a>

## 7. 从论文结构自然得到的拓展

### 7.1 多源域 DSN

若有 $K$ 个源域，可以共享一个 $E_c$，并为每个域设置私有编码器 $E_p^k$。**[数学推导]** 原论文 Equation (5) 对源、目标两项求和，直接推广为：

$$
L_{\text{difference}}^{K}
=\sum_{k=1}^{K}left\lVert {H_c^k}^T H_p^k\right\rVert_F^2.
$$

这是把两域求和索引改成 $K$ 域求和，没有引入新的理论假设。

**[分析判断]（拓展设想）** 主要代价是私有分支数量随域数线性增长，并且推理时必须知道样本属于哪个域，或额外学习域路由器。

### 7.2 更强的共享/私有独立性约束

**[分析判断]（拓展设想）** Frobenius 内积只抑制线性相关，可以替换或补充为：

- HSIC（Hilbert-Schmidt Independence Criterion，Hilbert-Schmidt 独立性准则），检测核空间中的非线性依赖；
- 互信息上界最小化；
- 用一个预测器尝试从共享表示恢复私有表示，再对共享编码器做对抗训练。

这些约束可能减少非线性信息泄漏，但优化更难，也仍然不能自动规定“缺陷语义应该属于共享还是私有”。

### 7.3 从像素重构改为特征重构

**[分析判断]（拓展设想）** 对高分辨率工业图像，像素解码器成本高，而且容易把容量用于颜色与细节。可以冻结预训练 backbone，并让解码器重构其中层多尺度特征，而非 RGB 像素。

这保留了 DSN 的逻辑：共享和私有表示合起来必须解释输入；同时更接近工业异常检测依赖的 patch 表示。代价是重构目标继承预训练 backbone 的偏差。

### 7.4 轻量私有 adapter

**[分析判断]（拓展设想）** 不必为每个域复制完整 CNN 私有编码器。可以：

- 保留共享 backbone；
- 在若干层插入域私有 adapter；
- 或仅让域私有归一化参数表达颜色、亮度和传感器统计。

这种结构在域数量增加或私有分支参数预算受限时更紧凑，也减少私有分支吸收完整任务语义的能力。

### 7.5 新域与持续适配

**[分析判断]（拓展设想）** 面对新目标域，可冻结已有共享编码器，为新域初始化一个私有 adapter 和解码路径，再用重构、difference 与少量 similarity loss 适配。

需要额外防止：

- 更新共享编码器导致旧域遗忘；
- 域边界不清时选错私有分支；
- 新域样本含异常时，异常被当作域私有因素学习。

---

## 8. 向工业 Memory Bank AD 的拓展

> IAD（Industrial Anomaly Detection，工业异常检测）：面向工业质检图像的异常检测。  
> Memory Bank（记忆库）：保存正常训练图像的局部特征，以测试 patch 到正常特征最近邻的距离作为异常依据。

### 8.1 最自然的 DSN-PatchCore 组合

**[分析判断]（拓展设想）** 令预训练 backbone 在位置 $p$ 输出 patch 特征 $F_p(x)$，再由共享编码器产生 $h_{c,p}=E_c(F_p(x))$。只用源域正常图像的共享 patch 表示建立 Memory Bank：

$$
\mathcal M_c
=\left\{E_c(F_p(x_i^s))\mid x_i^s\text{ 为正常图像}\right\}.
$$

测试异常分数可以写成：

$$
s(x)=\max_p\min_{m\in\mathcal M_c}
\left\lVert E_c(F_p(x))-m\right\rVert_2.
$$

**[数学推导]** 该式由 PatchCore 最近邻分数中把原始 patch 表示 $F_p(x)$ 替换为 DSN 的共享表示 $E_c(F_p(x))$ 得到。PatchCore 原始 Memory Bank 和最近邻机制见 [3, pp.14320-14323]。

目标是让：

- $E_c$ 保存跨相机、跨光照都稳定的正常结构和缺陷线索；
- $E_p^s,E_p^t$ 吸收各域的颜色、亮度、传感器噪声；
- Memory Bank 只存在于共享空间。

### 8.2 这里比分类任务更困难

**[分析判断]** 原论文的 $L_{\text{task}}$ 有多类源标签，能够规定共享表示必须保留类别或姿态信息。工业一类 AD 的源训练集只有正常样本，不能用“全部预测 normal”的分类头替代，否则该任务损失几乎没有信息量。

至少需要另一个防止共享空间坍塌的目标，例如：

- 保持预训练 patch 特征的局部邻域结构；
- 正常样本自监督任务；
- 可控的合成异常判别；
- 多尺度特征重构。

这些都是新方法设计，DSN 原论文没有验证。

### 8.3 最大风险：缺陷被当成目标私有信息

**[分析判断]** 如果无标签目标适配集含异常，目标私有编码器可能把缺陷编码为“目标域独有因素”。任务头和 Memory Bank 都只读取共享表示，这会直接削弱异常信号并产生假阴性。

因此第一轮 DSN-IAD 实验应使用人工确认正常的目标适配集。若目标池可能被异常污染，应先筛选高置信正常 patch，不能把完整目标池直接送入重构和分离训练。

### 8.4 用于区分机制的验证顺序

1. Source-only PatchCore，得到原始域偏移退化程度。
2. DANN + 轻量 adapter，确认单纯共享对齐是否有效。
3. DSN + 特征重构，只训练共享/私有 adapter，Memory Bank 建在共享空间。
4. 比较有无 private encoder、有无 difference loss、有无 reconstruction loss。
5. 适配结束后冻结全部参数并重建 Memory Bank，不能继续使用旧特征空间中的 Bank。

重点不是只看域分类准确率，而是同时检查：

- 目标正常 patch 到 Bank 的距离是否下降；
- 目标异常 patch 到 Bank 的距离是否仍然足够大；
- 私有表示是否主要响应光照、颜色等域因素；
- 共享表示是否仍保留细小缺陷边界。

---

## 9. 最终评价

1. **[论文发现]** DSN 的真正创新是显式建模域私有表示，而不是提出新的单一分布距离。
2. **[分析判断]** DSN-DANN 仍然包含 DANN；DSN 是对“只学共享域不变表示”的结构扩展。
3. **[数学推导]** difference loss 只提供软正交，不等于一般意义的独立性。
4. **[分析判断]** reconstruction loss 防止明显的信息丢失，但不能保证共享/私有分解唯一或语义正确。
5. **[论文发现]** 论文实验支持 difference、reconstruction 与 DANN similarity 的组合，但超参数选择使用了目标标签验证集。
6. **[分析判断]（拓展设想）** 对工业 Memory Bank AD，最合理的用法是以共享 patch 表示建 Bank、私有轻量分支吸收成像差异；最大的风险是目标缺陷被私有分支吸收。

---

## 论文引用

> [1] Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan. “Domain Separation Networks.” *Advances in Neural Information Processing Systems 29 (NIPS)*, pp.343-351, 2016.  
> 归档公开版页码：方法假设 pp.1-2；结构与总损失 pp.3-5；实验设置和结果 pp.5-8；结论 p.9；CorReg、更多可视化、网络结构与超参数见 Supplementary pp.11-15。

> [2] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky. “Domain-Adversarial Training of Neural Networks.” *Journal of Machine Learning Research*, 17(59):1-35, 2016.  
> 本文引用位置：DANN/GRL 与 DSN 的关系；DANN 鞍点和 GRL 见 pp.9-13。

> [3] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. “Towards Total Recall in Industrial Anomaly Detection.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.14318-14328, 2022.  
> 本文引用位置：工业 Memory Bank、patch 特征与最近邻异常分数，主要见 pp.14320-14323。
