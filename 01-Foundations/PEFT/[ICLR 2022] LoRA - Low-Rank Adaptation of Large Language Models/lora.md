# LoRA 精读与领域自适应微调拓展

> **阅读草稿**：原文解读与应用拓展仍需核验；拓展设想不作为原论文结论。

> LoRA（Low-Rank Adaptation，低秩适配）：冻结预训练权重，只用两个低秩矩阵参数化每个被选中权重矩阵的更新量。  
> PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）：只训练模型中的少量新增参数或参数子集，以降低训练和多任务存储成本。  
> 归档原文：[LoRA: Low-Rank Adaptation of Large Language Models](paper.pdf)
> 归档版本：arXiv:2106.09685v2，2021-10-16，共 26 页；论文发表于 ICLR（International Conference on Learning Representations，国际学习表征会议）2022。

本文用 **[论文发现]** 标注所引论文直接报告的内容，用 **[数学推导]** 标注由明确前提推出的性质，用 **[业界共识]** 标注通用概念，用 **[分析判断]** 标注方法解释与应用设想。后续工作与应用设想会在相应段落中说明，不归因于原论文。

## 阅读建议

首次阅读先看第 1–4 节的 [问题与结构](#paper-main)，再看第 5–7 节的实验证据。第 0 节用于比较不同方法的作用，第 8–10 节是 [领域迁移与异常检测拓展](#applications)。

---

## 0. 一句话结论

**[分析判断] LoRA 不是一种领域对齐损失，也不是对预训练权重 $W_0$ 做低秩压缩。它只规定任务更新量 $\Delta W$ 采用低秩参数化：**

$$
\Delta W=BA.
$$

因此，LoRA 与 DANN、DSN 的关系是：

- DANN（Domain-Adversarial Neural Network，领域对抗神经网络）决定**训练目标**：让特征保留任务信息，同时难以被域分类器区分。
- DSN（Domain Separation Network，领域分离网络）决定**表示结构**：把共享信息和域私有信息拆开。
- LoRA 决定**允许哪些参数变化、以什么形式变化**：冻结基础模型，只让低秩更新参与优化。
- Memory Bank 决定**异常分数如何计算**：将测试 patch 与正常参考特征比较。

这四者不在同一个层级。把 LoRA 接入 DANN，只是把 DANN 原本对整个特征提取器的更新限制到 LoRA 参数中；**域对齐仍然来自 DANN 损失，不来自低秩本身。**

---

<a id="paper-main"></a>

## 1. LoRA 要解决的具体问题

### 1.1 全量微调的问题不是只有训练显存

> LLM（Large Language Model，大语言模型）：参数量很大的预训练语言模型，通常先进行通用预训练，再适配具体任务。
> GPT（Generative Pre-trained Transformer，生成式预训练 Transformer）：使用 Transformer 自回归建模文本的一系列预训练模型。

**[论文发现]** 论文以 GPT-3 175B 为主要动机。若每个下游任务都全量微调，就要为每个任务保存一套与基础模型同等规模的参数增量；同时支持多个任务时，存储和切换成本很高。论文还指出，Adapter 会增加串行网络深度，Prefix Tuning 会占用可用序列长度。[1, pp.1-4]

全量微调的条件语言建模目标写为：

$$
\max_{\Phi}\sum_{(x,y)\in\mathcal Z}\sum_{t=1}^{|y|}
\log P_{\Phi}(y_t\mid x,y_{<t}).
$$

这是原论文第 3 页 Equation (1)。预训练参数从 $\Phi_0$ 更新到 $\Phi_0+\Delta\Phi$；每个任务的 $|\Delta\Phi|$ 与 $|\Phi_0|$ 同量级。[1, p.3]

参数高效适配改为用更小的参数集合 $\Theta$ 编码更新量：

$$
\max_{\Theta}\sum_{(x,y)\in\mathcal Z}\sum_{t=1}^{|y|}
\log P_{\Phi_0+\Delta\Phi(\Theta)}(y_t\mid x,y_{<t}),
\qquad |\Theta|\ll|\Phi_0|.
$$

这是原论文第 3 页 Equation (2)。LoRA 的工作就是为 $\Delta\Phi(\Theta)$ 选择一种低秩表示。[1, p.3]

### 1.2 LoRA 没有消除基础模型

**[论文发现]** LoRA 节省的是每个任务的训练参数、优化器状态和任务增量参数，而不是让基础模型凭空消失。论文第 5 页脚注明确说明：使用 GPT-3 175B 仍需要约 350 GB 的基础模型；优势在于 100 个任务只需共享这一个基础模型，再各自保存约 35 MB 的 LoRA 权重，而不是保存 100 份 350 GB 模型。[1, p.5]

**[分析判断]** 因此要区分三种量：

| 量 | LoRA 是否显著降低 | 原因 |
|---|---:|---|
| 每个任务的可训练参数与优化器状态 | 是 | $W_0$ 冻结，只优化 $A,B$ |
| 每个任务的增量参数 | 是 | 只保存 $A,B$，可选再保存 bias |
| 基础模型本身的推理显存/存储 | 否 | 推理仍需要 $W_0$ 或合并后的 $W_0+BA$ |
| 未合并 LoRA 分支的额外计算 | 否 | 仍需计算 $Ax$ 和 $B(Ax)$ |
| 合并后的额外线性层深度 | 是，降为零 | $BA$ 可直接加回 $W_0$ |

---

## 2. 核心结构：低秩的是更新量，不是原权重

### 2.1 矩阵形状与前向传播

**[论文发现]** 对一个预训练权重矩阵：

$$
W_0\in\mathbb R^{d\times k},
$$

LoRA 冻结 $W_0$，把更新量写为：

$$
\Delta W=BA,
\qquad
B\in\mathbb R^{d\times r},
\qquad
A\in\mathbb R^{r\times k},
\qquad
r\ll\min(d,k).
$$

原论文第 4 页 Equation (3) 写出的主体前向传播是：

$$
h=W_0x+BAx.
$$

同一页紧接着说明实际还要乘缩放因子 $\alpha/r$。合并两处描述，实践中的表达式为：

$$
\boxed{
h=W_0x+\frac{\alpha}{r}BAx
}
$$

其中 $\alpha$ 是常数超参数，$r$ 是 LoRA rank。[1, p.4]

前向计算可以画成两条并行路径：

```text
                    ┌──────────────┐
x ─────────────────>│ frozen W_0   │──────┐
│                   └──────────────┘      │
│                                          ├──> h
│   ┌────────┐       ┌────────┐           │
└──>│ A: k→r │──────>│ B: r→d │── α/r ───┘
    └────────┘       └────────┘
       trainable        trainable
```

**[分析判断]** $A$ 先把 $k$ 维输入投影到 $r$ 维瓶颈，$B$ 再将这 $r$ 个适配方向映射到 $d$ 维输出。虽然瓶颈只有 $r$ 维，$B$ 仍可同时改变全部 $d$ 个输出坐标，所以“低秩”不等于“只改变少数输出通道”。

### 2.2 rank 约束为什么成立

**[数学推导]** 矩阵乘积的秩满足：

$$
\operatorname{rank}(BA)
\leq
\min\bigl(\operatorname{rank}(B),\operatorname{rank}(A)\bigr)
\leq r.
$$

因此 LoRA 直接保证：

$$
\operatorname{rank}(\Delta W)\leq r.
$$

但它不保证最终学到的更新恰好为 rank $r$；若某些方向退化或奇异值接近零，实际有效秩可以更低。

### 2.3 参数量如何减少

**[数学推导]** 直接训练一个 $d\times k$ 更新矩阵需要 $dk$ 个参数，LoRA 只需要：

$$
N_{\text{LoRA}}=dr+rk=r(d+k).
$$

相对参数比例为：

$$
\frac{N_{\text{LoRA}}}{N_{\text{full}}}
=\frac{r(d+k)}{dk}.
$$

若 $d=k$，则：

$$
\frac{N_{\text{LoRA}}}{N_{\text{full}}}=\frac{2r}{d}.
$$

例如 $d=768,r=4$ 时，一个方阵更新只训练原矩阵约 $1.04\%$ 的参数。这个比例是由矩阵形状直接推得，不是论文实验结论。

### 2.4 为什么初始化为“$A$ 随机、$B=0$”

**[论文发现]** 原论文用随机高斯初始化 $A$，将 $B$ 初始化为零，所以训练开始时：

$$
\Delta W=BA=0.
$$

初始模型输出因此与预训练模型完全一致。初始化说明位于原论文第 1 页 Figure 1 和第 4 页正文。[1, pp.1,4]

这不只是为了“初始扰动小”，还关系到梯度能否流动。令：

$$
s=\frac{\alpha}{r},
\qquad
z=Ax,
\qquad
\delta=\frac{\partial L}{\partial h}.
$$

前向为 $h=W_0x+sBz$。对 $B$ 的微分：

$$
dh=s\,dB\,z,
$$

于是：

$$
\begin{aligned}
dL
&=\delta^Tdh \\
&=s\delta^T dBz \\
&=\operatorname{tr}\left((s\delta z^T)^T dB\right),
\end{aligned}
$$

所以：

$$
\boxed{
\frac{\partial L}{\partial B}
=s\delta(Ax)^T
}
$$

对 $A$：

$$
dh=sB\,dA\,x,
$$

同理：

$$
\boxed{
\frac{\partial L}{\partial A}
=sB^T\delta x^T
}
$$

**[数学推导]** 在初始化时 $B=0$：

- $\partial L/\partial A=0$，所以第一个更新步中 $A$ 暂时不动；
- 只要随机 $A$ 使 $Ax\neq 0$，$\partial L/\partial B$ 通常不为零，$B$ 可以先开始学习；
- $B$ 离开零点后，$A$ 也会获得非零梯度。

若把 $A$ 和 $B$ 都初始化为零，则两个梯度同时为零，LoRA 分支会卡在零点。这就是只将其中一个因子置零的直接原因。

### 2.5 $\alpha/r$ 的作用

**[论文发现]** 论文用 $\alpha/r$ 缩放 LoRA 分支，目的是改变 $r$ 时减少重新搜索超参数的需要。作者在实验中把 $\alpha$ 设为最先尝试的 $r$，之后不再单独调节；论文称，在适当缩放初始化的前提下，对 Adam 而言调 $\alpha$ 与调学习率“roughly the same”。[1, p.4]

**[分析判断]** “roughly the same” 不能改写成“数学上等价”：

- $\alpha$ 直接缩放前向扰动和传入 $A,B$ 的梯度；
- 学习率控制优化器对梯度的更新步长；
- Adam 还有一、二阶矩状态、$\epsilon$、weight decay 等因素。

两者可能产生相近的尺度效应，但训练轨迹不必相同。

### 2.6 合并与取消合并

**[论文发现]** 推理前可计算：

$$
W_{\text{merged}}=W_0+\frac{\alpha}{r}BA.
$$

随后仍用一个普通线性层：

$$
h=W_{\text{merged}}x.
$$

因此网络深度和矩阵乘法次数与原模型相同。切换任务时，可以减去当前任务的 $BA$，再加上新任务的 $B'A'$。该设计和限制位于原论文第 4-5 页。[1, pp.4-5]

**[分析判断]** “无额外推理延迟”有一个前提：**已经合并权重**。若为了同一 batch 动态选择不同 adapter 而保持未合并状态，就仍需计算低秩分支。原论文第 5 页也将“同一 batch 混用不同任务的已合并 LoRA 不直接”列为限制。[1, p.5]

---

## 3. LoRA 与全量微调的准确关系

### 3.1 表达能力可以接近，优化过程并不相同

**[论文发现]** 作者称：若对所有权重矩阵都使用 LoRA、同时训练 bias，并令 $r$ 达到相应矩阵的满秩，就能“roughly recover”全量微调的表达能力。[1, p.4]

这个说法可以从矩阵分解解释。设任意目标更新 $\Delta W$ 的秩为 $q$，其紧致 SVD（Singular Value Decomposition，奇异值分解）为：

$$
\Delta W=U_q\Sigma_qV_q^T.
$$

令：

$$
B=U_q\Sigma_q^{1/2},
\qquad
A=\Sigma_q^{1/2}V_q^T,
$$

则：

$$
BA
=U_q\Sigma_q^{1/2}\Sigma_q^{1/2}V_q^T
=\Delta W.
$$

**[数学推导]** 因此，只要 $r\geq q$，LoRA 的参数化在表示层面可以覆盖这个更新矩阵。

**[分析判断]** 但“能表示同一个 $\Delta W$”不等于“训练会得到与全量微调相同的结果”：

- 全量微调直接优化 $W$；LoRA 优化双线性参数 $A,B$；
- 两者的损失几何、梯度尺度和隐式正则不同；
- 原论文自己使用的是“roughly”，没有给出优化轨迹等价定理。

### 3.2 $A,B$ 的分解不是唯一的

对任意可逆矩阵 $R\in\mathbb R^{r\times r}$：

$$
BA=(BR)(R^{-1}A).
$$

特别地，对任意非零常数 $c$：

$$
BA=(cB)(A/c).
$$

**[数学推导]** 所以 $A$ 和 $B$ 各自的数值、范数和坐标方向不是唯一可辨识的；真正决定网络输出的是乘积 $BA$。分析 adapter 时，$\Delta W$ 的秩、奇异值、行空间和列空间通常比单独观察某个因子的绝对数值更可靠。

### 3.3 “低内在维度”不等于“低内在秩”

**[论文发现]** LoRA 的动机来自已有观察：过参数化模型的任务解可能位于低内在维度的优化子空间。作者进一步**提出假设**：任务适配产生的矩阵更新 $\Delta W$ 也具有低“intrinsic rank”。这两句话位于第 2 页和第 4 页。[1, pp.2,4]

二者不是同一个数学对象：

| 概念 | 讨论对象 | 含义 |
|---|---|---|
| intrinsic dimension | 整个优化问题或可达解集合 | 完成任务需要多少独立参数方向 |
| matrix rank | 单个矩阵 $\Delta W$ | 行/列空间中有多少线性独立方向 |

**[分析判断]** “整个任务可在低维子空间优化”不会自动推出“每一层更新矩阵都低秩”。LoRA 把后者作为设计假设，并通过特定任务上的实验提供经验支持，而不是给出一般定理。

---

## 4. LoRA 在 Transformer 中加在哪里

### 4.1 原论文实际只研究了注意力权重

**[论文发现]** 一个 Transformer 自注意力模块有 $W_q,W_k,W_v,W_o$ 四个投影矩阵，前馈 MLP（Multi-Layer Perceptron，多层感知机）另有两个矩阵。论文为简化和节省参数，只在注意力投影中研究 LoRA，并冻结 MLP；多数实验把 LoRA 加在 $W_q$ 和 $W_v$ 上。[1, pp.5-6]

若模型有 $L$ 层、每层的 $W_q,W_v\in\mathbb R^{d\times d}$，则：

$$
N_{q,v}
=L\times 2\times(2dr)
=4Ldr.
$$

这与原论文第 6 页给出的通式一致。[1, p.6]

### 4.2 为什么不能把 $W_q,W_v$ 当成普遍最优答案

> MultiNLI（Multi-Genre Natural Language Inference，多体裁自然语言推理）：用多体裁文本评测自然语言推理能力的数据集，也常缩写为 MNLI。

**[论文发现]** 在 GPT-3 上固定约 18M 可训练参数时，Table 5 比较了不同放置位置：只调一个矩阵时使用 $r=8$，调两个时用 $r=4$，四个都调时用 $r=2$。$W_q+W_v$ 整体优于只把更高 rank 放在单个 $W_q$ 或 $W_k$ 上；四个矩阵都调在 MultiNLI 上更好，但 WikiSQL 与 $W_q+W_v$ 相同。[1, p.10]

这组消融支持：**固定参数预算时，覆盖更多矩阵可能比把 rank 全给一个矩阵更有效。** 它不证明 $W_q,W_v$ 对所有架构和任务都最优。

原论文在结论中也把“主要依赖启发式选择 LoRA 位置，能否更有原则地选择”列为未来问题。[1, pp.12-13]

### 4.3 对视觉模型的边界

> ViT（Vision Transformer，视觉 Transformer）：将图像切成 patch token，再使用 Transformer 编码视觉特征。
> BERT（Bidirectional Encoder Representations from Transformers，Transformer 双向编码表示）：使用双向 Transformer 编码上下文的预训练语言模型。RoBERTa 是 Robustly Optimized BERT Pretraining Approach（鲁棒优化的 BERT 预训练方案），DeBERTa 是 Decoding-enhanced BERT with disentangled attention（使用解耦注意力增强解码的 BERT）。

**[论文发现]** 原论文实验对象是 RoBERTa、DeBERTa、GPT-2 和 GPT-3，任务均为自然语言理解或生成，没有视觉、领域自适应或异常检测实验。[1, pp.5-8]

**[分析判断]（拓展设想）** 因此：

- 对 ViT/DINOv2 一类视觉 Transformer，可把 $q,v$ 作为起始基线，但必须重新做层位与 rank 消融。DINO 源于 self-DIstillation with NO labels（无标签自蒸馏），DINOv2 是其后续视觉表征模型。
- 对 CNN，原论文只说原则可用于任意 dense layer；把卷积核 reshape 后低秩分解、使用两个卷积参数化低秩路径，属于方法拓展，不是原论文验证结果。
- 对 PatchCore 一类提取多层 patch 特征的方法，LoRA 放在哪一层会直接改变 Memory Bank 的局部几何，不能只看分类精度选择位置。

---

## 5. 实验到底支持了什么

### 5.1 下游任务结果

> GLUE（General Language Understanding Evaluation，通用语言理解评测）：由多种自然语言理解任务组成的综合基准。
> E2E NLG Challenge（End-to-End Natural Language Generation Challenge，端到端自然语言生成挑战）：数据到文本生成基准；BLEU（Bilingual Evaluation Understudy，双语评估替补指标）用预测文本与参考文本的 n-gram 重合度评估生成质量。

**[论文发现]** 论文的主要结果可压缩为以下几组：

| 模型与任务 | 全量微调 | LoRA | 原文位置 |
|---|---:|---:|---|
| RoBERTa-base，GLUE 平均 | 125.0M 参数，86.4 | 0.3M 参数，87.2 | Table 2, p.6 |
| RoBERTa-large，GLUE 平均 | 355.0M 参数，88.9 | 0.8M 参数，89.0 | Table 2, p.6 |
| DeBERTa-XXL，GLUE 平均 | 1500.0M 参数，91.1 | 4.7M 参数，91.3 | Table 2, p.6 |
| GPT-2 Medium，E2E BLEU | 354.92M 参数，68.2 | 0.35M 参数，70.4 | Table 3, p.7 |
| GPT-3 175B，WikiSQL | 175255.8M 参数，73.8 | 4.7M 参数，73.4；37.7M 参数，74.0 | Table 4, p.8 |
| GPT-3 175B，MNLI-matched | 175255.8M 参数，89.5 | 4.7M 参数，91.7；37.7M 参数，91.6 | Table 4, p.8 |

**[分析判断]** 这些结果支持“在论文测试的语言模型和任务上，LoRA 能以很少的可训练参数达到有竞争力的性能”。不能把它扩大为“LoRA 总是优于全量微调”，原因包括：

- Table 2 中带 `*` 的若干基线来自既有论文，不是所有方法都在完全相同的训练协议下重跑；`†` 才标记了为 Adapter 公平比较而限制过的设置。[1, pp.6-7]
- GPT-3 成本很高，论文只报告每个任务的典型波动，没有对每个表项都做完整多随机种子统计。[1, p.8]
- 论文没有覆盖视觉、工业异常检测或强域偏移任务。

## 6. rank 实验与“低秩更新”证据

### 6.1 小 rank 在特定 GPT-3 任务上已经足够

**[论文发现]** Table 6 在 WikiSQL 和 MultiNLI 上比较 $r\in\{1,2,4,8,64\}$。当同时适配 $W_q,W_v$ 或四个注意力矩阵时，$r=1$ 已经具有竞争力；只适配 $W_q$ 时，更高 rank 更有帮助。[1, p.10]

但同一页脚注直接限制了这个结论：作者不认为小 rank 对所有任务都成立，并举例说，如果下游任务语言与预训练语言不同，接近全量重训的高 rank 可能更好。[1, p.10, footnote 6]

**[论文发现]** GPT-2 Medium 的补充实验也不是“$r=1$ 最优”：E2E 任务中验证损失在 $r=16$ 最低，BLEU 在 $r=4$ 最高；继续增大到 1024 没有带来稳定收益。结果位于第 26 页 Table 18。[1, p.26]

**[分析判断]** rank 是需要围绕“任务、模型、层位、数据量”共同选择的容量超参数，不应从 GPT-3 的两个任务直接复制一个固定值。

### 6.2 子空间相似度分析

**[论文发现]** 作者比较 $r=8$ 与 $r=64$ 学到的 $A$ 的右奇异子空间。设 $U_A^i$ 表示前 $i$ 个奇异方向，定义：

$$
\phi(A,B,i,j)
=\frac{\left\lVert {U_A^i}^{T}U_B^j\right\rVert_F^2}
{\min(i,j)}
\in[0,1].
$$

这是原论文第 11 页 Equation (4)；附录第 22-24 页说明它与 Grassmann 子空间的 Projection Metric 相关。[1, pp.11,22-24]

**[数学推导]** 若两个子空间完全正交，则交叉内积为零，$\phi=0$；若较小子空间完全包含在较大子空间中，则分子等于较小维度，$\phi=1$。

**[论文发现]** GPT-3 第 48 层的 Figure 3 显示：$r=8$ 与 $r=64$ 的最主要奇异方向明显重合，而其余方向重合较弱。不同随机种子的 Figure 4 也显示 $\Delta W_q$ 比 $\Delta W_v$ 共享更多方向。作者据此认为任务更新具有很低的有效方向。[1, pp.11-12]

**[分析判断]** 这是经验诊断，不是“最优 $\Delta W$ 必然低秩”的证明：它只考察 GPT-3、选定层、选定数据集和训练配置，而且 $A,B$ 的分解本身并不唯一。

### 6.3 低 rank 绝不等于小扰动

**[论文发现]** Table 7 将 $W_q$ 投影到 $\Delta W_q$ 的奇异子空间。在 GPT-3 第 48 层、$r=4$ 时：

$$
\left\lVert U^TW_qV\right\rVert_F=0.32,
\qquad
\left\lVert\Delta W_q\right\rVert_F=6.91.
$$

作者给出的放大比约为：

$$
\frac{6.91}{0.32}\approx21.5.
$$

数据位于原论文第 12 页 Table 7，附录第 25 页进一步讨论 amplification factor。[1, pp.12,25]

**[分析判断]** 这条结果对异常检测尤其重要：LoRA 参数很少、rank 很小，也可能沿少数方向产生很强的特征改动。**“参数高效”不能被当作“特征空间天然稳定”的保证。**

---

## 7. 原论文证明了什么、没有证明什么

| 命题 | 当前证据判断 |
|---|---|
| $\operatorname{rank}(BA)\le r$ | **[数学事实]** 由矩阵秩不等式严格成立 |
| 训练初始输出等于基础模型 | **[数学事实]** 在 $B=0$ 时严格成立 |
| 合并后不增加线性层深度 | **[数学事实]** 由 $W_{\text{merged}}=W_0+(\alpha/r)BA$ 成立 |
| LoRA 总能等价于全量微调 | **未证明**；满 rank 时表达能力可覆盖，但优化过程不同 |
| 任意下游任务的更新都低秩 | **未证明**；只在论文特定语言任务中有经验支持 |
| rank 越大性能越好 | **原实验反驳**；Table 6、18 都不是单调关系 |
| LoRA 自动学到域不变特征 | **未提出也未证明**；原文没有领域对齐目标 |
| LoRA 自动保护异常/正常距离 | **未证明**；第 12 页甚至显示少数方向可被强放大 |
| LoRA 原论文验证了视觉模型 | **没有**；原实验全部是语言模型 |
| LoRA 降低基础模型推理显存 | **通常不成立**；基础权重仍必须驻留或加载 |

---

<a id="applications"></a>

## 8. LoRA 如何成为 DA 的微调模块

> DA（Domain Adaptation，领域自适应）：训练时利用源域与目标域数据，使模型适应二者的分布差异。  
> UDA（Unsupervised Domain Adaptation，无监督领域自适应）：源域有任务标签，目标域参与训练但没有任务标签。  
> TTA（Test-Time Adaptation，测试时适配）：模型在测试阶段用当前目标数据更新少量参数或状态。

### 8.1 最重要的概念分工

**[分析判断]** LoRA 与 DA 可以组合，但不能互相替代：

| 组件 | 它回答的问题 | 它不回答的问题 |
|---|---|---|
| LoRA | 哪些权重允许变化；更新矩阵的最大 rank 是多少 | 什么叫“对齐”；目标数据是否正常；任务信息如何保留 |
| DANN/梯度反转 | 如何用域分类对抗推动共享特征域不可分 | 参数必须全量训练还是只训练 LoRA |
| DSN | 如何显式分开共享与私有表示 | 每个编码器内部采用全量参数还是 LoRA |
| Memory Bank | 如何用正常参考特征计算异常分数 | 特征空间如何适应目标域 |

### 8.2 DANN + LoRA：反向域梯度只更新低秩分支

> GRL（Gradient Reversal Layer，梯度反转层）：前向保持特征不变，反向将传给特征提取器的域分类梯度乘以负系数。

设基础特征提取器参数 $\theta_0$ 冻结，全部 LoRA 参数记为：

$$
\varphi=\{A_\ell,B_\ell\}_{\ell\in\mathcal S},
$$

其中 $\mathcal S$ 是选中的层。带 LoRA 的特征记为 $f_{\theta_0,\varphi}(x)$。使用正的域交叉熵：

$$
L_d
=-\mathbb E_{x^s}\log D(f_{\theta_0,\varphi}(x^s))
-\mathbb E_{x^t}\log\left(1-D(f_{\theta_0,\varphi}(x^t))\right).
$$

DANN 的两个更新方向为：

$$
\min_{\theta_d}L_d,
$$

$$
\min_{\varphi,\theta_y}
L_y-\lambda L_d.
$$

域分类器参数 $\theta_d$ 努力最小化域分类误差；任务头和 LoRA 特征分支通过 GRL 努力最大化同一个正交叉熵。这个正负号关系来自 DANN 的鞍点目标，见 DANN 原论文第 10-13 页。[2]

**[分析判断]（拓展设想）** 与全量 DANN 相比，此时反向的域梯度仍经过每层的：

$$
\frac{\partial L}{\partial B_\ell}
=\frac{\alpha_\ell}{r_\ell}\delta_\ell(A_\ell x_\ell)^T,
\qquad
\frac{\partial L}{\partial A_\ell}
=\frac{\alpha_\ell}{r_\ell}B_\ell^T\delta_\ell x_\ell^T,
$$

但 $W_{0,\ell}$ 不更新。LoRA 将 DANN 的特征更新限制在一组低秩权重增量中，降低训练状态和灾难性大范围改写的自由度。

**它仍然不保证：**

- 被混淆的是无关域因素，而不是异常判别信息；
- 目标异常不会被拉向源域正常特征；
- 小 rank 就一定保持 Memory Bank 的局部邻域结构；
- 域分类准确率接近 50% 就代表目标异常检测变好。

### 8.3 DSN + LoRA：共享 adapter 与私有 adapter

**[分析判断]（拓展设想）** DSN 式参数高效结构可以设计为：

- 冻结一个共享基础 backbone $\theta_0$；
- 共享 LoRA $\varphi_c$ 处理源域和目标域，并承受任务损失与域对抗损失；
- 源域私有 LoRA $\varphi_p^s$、目标域私有 LoRA $\varphi_p^t$ 只参与各自域的重构或私有表示；
- difference loss 仍约束共享与私有特征，而不是约束 $A,B$ 矩阵本身。

这个结构能降低完整三套编码器的参数开销，但它不是 LoRA 或 DSN 原论文实验过的组合。目标异常仍可能被目标私有 adapter 当作“目标域特有信息”吸收，所以异常污染问题没有因为参数变少而消失。

---

## 9. 已有 LoRA/PEFT 领域迁移工作的准确位置

以下只列与视觉领域迁移最接近的代表性工作，不把所有“换数据集微调”都叫作 DA。

> AD（Anomaly Detection，异常检测）：从以正常样本为主的参考分布中识别偏离正常模式的样本或局部区域。  
> VPT（Visual Prompt Tuning，视觉提示微调）：冻结视觉 Transformer 主干，仅学习少量提示 token。  
> CLIP（Contrastive Language-Image Pre-training，对比式语言-图像预训练）：通过图文对比学习得到可迁移视觉与文本表示的模型。  
> MFM（Medical Foundation Model，医学基础模型）：在大规模医学数据上预训练、再适配具体医学任务的基础模型。

| 工作 | 适配设置 | LoRA 在其中负责什么 | 不能据此推出什么 |
|---|---|---|---|
| ExPLoRA | 在新视觉域上继续自监督预训练，再做有监督下游微调 | 自监督阶段解冻 1-2 个 ViT block，其余层用 LoRA；下游阶段只用 LoRA 微调 | 它不是 DANN 式源/目标对抗对齐，也没有工业 AD 实验 [3, p.1] |
| PLUTO | 少量无标签目标样本的测试时域适配 | 预先建立 LoRA/Adapter/VPT 等模块库，目标侧选择并融合少数已有模块，不更新其权重 | 它说明 LoRA 可作为可组合域模块，不是“目标侧直接训练一套 LoRA” [4, p.1] |
| LoRA-TTT | CLIP 类视觉语言模型的测试时训练 | 只更新图像编码器中的 LoRA，并引入轻量重构目标以适应测试分布 | 它验证的是分类域偏移，不是 Memory Bank 工业缺陷；测试时目标也不同于 DANN [5, p.1] |
| MFM-DA | 少量无标签目标图像的医学分割 UDA | 先用适配后的扩散模型做源到目标风格转换，再用 channel-spatial alignment LoRA 对齐医学基础模型特征 | 收益来自整套生成与层级对齐框架，不能只归因于原始 LoRA [6, p.1] |

**[论文发现]（后续工作）** 最接近“LoRA + 工业 AD”的近期工作是 *Normality-Preserving Continual Industrial Anomaly Detection via Orthogonal LoRA Banks*。它冻结历史 LoRA bank，并让新类别 adapter 与历史子空间正交，用于持续工业异常检测；这是 continual learning（持续学习）问题，不是源域到目标域的 UDA，也没有解决无标签目标池中的异常污染。[7, p.1]

**[分析判断]** 这些后续工作的共同点不是“LoRA 自己完成 DA”，而是：

$$
\text{LoRA 参数化}
+
\text{目标域训练信号}
+
\text{防退化约束}
=
\text{具体适配方法}.
$$

缺少后两项时，只把 LoRA 插入 backbone 并不会产生任何领域适配。

---

## 10. 用于 Memory Bank 工业 AD 时

### 10.1 理论上一致的信息流

**[分析判断]（拓展设想）** 对当前以正常 patch Memory Bank 为核心的方法，更合理的顺序是：

```text
预训练 backbone θ0
        │ 冻结
        ▼
插入 LoRA φ
        │
        ├── 源域正常数据：保持结构/教师特征/一类目标
        └── 目标适配数据：DANN、统计匹配或鲁棒匹配
        │
        ▼
完成适配并冻结 θ0, φ
        │
        ▼
用同一 θ0, φ 重新提取正常 patch，重建 Memory Bank
        │
        ▼
目标测试图像使用同一 θ0, φ 与新 Bank 计算距离
```

适配后的 Bank 应定义为：

$$
\mathcal M_{\varphi}
=\left\{
f^{\text{patch}}_{\theta_0,\varphi}(x_i^s)
\right\}_{x_i^s\in\mathcal D_s^{\text{normal}}}.
$$

测试分数例如：

$$
s_{\varphi}(x)
=\max_p\min_{m\in\mathcal M_{\varphi}}
\left\lVert
f^{\text{patch}}_{\theta_0,\varphi}(x)_p-m
\right\rVert_2.
$$

以上两式是本文为实验流程写出的定义，不来自 LoRA 原论文。

### 10.2 旧 Bank 为什么必须重建

假设旧 Bank 由 $f_{\theta_0,0}$ 生成，而适配后测试特征由 $f_{\theta_0,\varphi}$ 生成。此时距离实际比较的是：

$$
\left\lVert
f_{\theta_0,\varphi}(x)-f_{\theta_0,0}(x_i^s)
\right\rVert_2.
$$

**[数学推导]** 两个向量来自不同参数定义的坐标映射。除非额外证明 LoRA 更新保持距离、角度和尺度，否则旧 Bank 的最近邻语义不再成立。LoRA 第 12 页的放大实验还说明，小 rank 也可能显著改变少数方向。因此每次改变 adapter 后，都应先冻结模型，再重建 Bank。[1, p.12]

### 10.3 一类异常检测缺少 DANN 原本的任务锚点

**[分析判断]** DANN 的源域分类损失 $L_y$ 会阻止特征为了域混淆而丢掉类别信息。工业 AD 的源训练集若全部标为“正常”，训练一个恒定单类别分类头不能提供同等约束；它没有类别边界可保护。

**[分析判断]（拓展设想）** LoRA-DANN 至少需要一个额外锚点，例如：

- 保持适配前后源正常特征的一致性；
- 保持源正常 patch 的局部邻域或相对距离；
- 使用自监督重构/增强一致性；
- 使用可控合成缺陷提供判别约束，但必须单独验证合成偏差。

一种最小的教师保持项可定义为：

$$
L_{\text{keep}}
=\sum_{\ell\in\mathcal F}
\mathbb E_{x^s}
\left\lVert
g_\ell(x^s;\theta_0,\varphi)
-\operatorname{sg}\bigl(g_\ell(x^s;\theta_0,0)\bigr)
\right\rVert_2^2,
$$

其中 $\mathcal F$ 是用于建 Bank 的特征层，$\operatorname{sg}$ 表示 stop-gradient。此式是本文提出的实验定义，作用是限制源正常特征漂移；它不保证目标异常分离，仍需实验验证。

### 10.4 目标异常污染仍是第一风险

**[数学推导]** 若无标签目标适配集为：

$$
P_T
=(1-\pi)P_T(X\mid Y=\text{normal})
+\pi P_T(X\mid Y=\text{anomaly}),
$$

普通 DANN-LoRA 仍对齐整体 $P_T(f(X))$。当 $\pi>0$ 时，异常样本产生的反向域梯度同样会更新 LoRA，把它们推向源域正常分布。低 rank 只限制更新形式，不识别哪些目标 patch 是异常。

因此第一轮实验应区分：

| 组别 | 目标适配数据 | 目的 |
|---|---|---|
| A | 不使用目标数据 | source-only 基线 |
| B | 人工确认正常的少量目标图像 | 测试无污染时 LoRA-DANN 是否有效 |
| C | 全量无标签目标池 | 测量异常污染造成的退化，不默认是最终方案 |
| D | 高置信正常 patch 或置信加权 | 验证过滤是否修复对齐 |

### 10.5 用于区分机制的 LoRA 消融

**[分析判断]（拓展设想）** 可用下列消融区分低秩约束、领域对齐与 Memory Bank 更新各自的作用：

1. 冻结 backbone，只在最后一个或两个特征 stage 插入 LoRA；ViT 先试 $q,v$，CNN 先试 projection/晚期卷积。
2. rank 先扫 $r\in\{1,2,4,8\}$，不要默认越大越好。
3. 固定每组的 Bank coreset 比例，任何 adapter 变化后都重建 Bank。
4. 同时记录域探针准确率、源正常特征漂移、目标正常到 Bank 的距离、目标异常到 Bank 的距离。
5. 只有当域可分性下降且正常/异常距离间隔扩大，才能认为对齐对 AD 有效。

**停止条件**：若域分类器更难区分源/目标，但目标异常也更接近正常 Bank，说明模型只学会了域混淆，没有保住异常判别性。

---

## 11. 常见误解直接纠正

### 误解一：LoRA 对 $W_0$ 做低秩分解

错误。LoRA 保留完整 $W_0$，低秩分解的是更新量 $\Delta W=BA$。[1, p.4]

### 误解二：$r$ 是中间特征本身的维度

不准确。$r$ 是单个权重更新矩阵的最大 rank，也是低秩支路的瓶颈宽度；主干特征仍可保持 $d$ 维。

### 误解三：rank 小，所以模型变化一定小

错误。rank 限制独立方向的数量，不限制这些方向的幅度；原论文第 12 页报告过约 21.5 倍的方向放大。[1, p.12]

### 误解四：LoRA 永远没有额外推理计算

错误。只有将 $BA$ 合并进 $W_0$ 后才没有额外线性分支；未合并并动态选择 adapter 时仍有额外计算。[1, pp.4-5]

### 误解五：用了 LoRA 就完成了 domain adaptation

错误。LoRA 没有域标签、分布距离、对抗分类器或目标域自监督目标。必须再提供 DANN、MMD（Maximum Mean Discrepancy，最大均值差异）、重构、伪标签、测试时训练等目标。

### 误解六：适配后可以继续使用旧 Memory Bank

通常错误。Bank 与测试特征必须由同一版本的特征映射产生，否则最近邻距离混合了两个坐标系统。

### 误解七：$A$ 或 $B$ 的单独范数能直接解释任务重要性

不可靠。因为 $BA=(cB)(A/c)$，两个因子存在尺度不唯一性；优先分析乘积 $\Delta W$ 和实际特征变化。

---

## 12. 结论

1. **[原论文 + 数学事实]** LoRA 冻结 $W_0$，用 $BA$ 参数化 $\Delta W$；参数量从 $dk$ 降为 $r(d+k)$，并可在推理前精确合并。
2. **[原论文证据]** 小 rank 在论文测试的语言任务中常已足够，但最优 rank 随模型、任务和放置位置变化；论文没有普遍低秩定理。
3. **[分析判断]** LoRA 是更新空间的约束，不是 DA 方法。DANN/DSN/测试时训练提供目标，LoRA 只承载这些目标产生的参数更新。
4. **[工业 AD 判断]** LoRA 很适合作为 DANN 的轻量 feature adapter，但它不解决目标异常污染，也不天然保护 Memory Bank 的正常/异常几何。
5. **[实验优先级]** 先在“目标适配样本确认正常”的条件下验证 LoRA-DANN，再测试过滤目标池；每次适配后重建 Bank，并以正常/异常距离间隔而不是域分类器准确率作为主要成功标准。

---

## 论文引用

> [1] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. “LoRA: Low-Rank Adaptation of Large Language Models.” *International Conference on Learning Representations (ICLR)*, 2022. arXiv:2106.09685v2, pp.1-26.  
> 本文引用位置：动机与初始化 pp.1-2；目标 Equations (1)-(2) p.3；低秩更新 Equation (3)、缩放、合并与表达能力 p.4；显存、存储、速度和限制 p.5；参数量与 GLUE 结果 p.6；GPT-2/GPT-3 结果 pp.7-8；放置和 rank 消融 p.10；子空间相似度 Equation (4) pp.11,22-24；更新方向与放大分析 pp.12,25-26。

> [2] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky. “Domain-Adversarial Training of Neural Networks.” *Journal of Machine Learning Research*, 17(59):1-35, 2016.  
> 本文引用位置：DANN 鞍点目标、域分类器与特征提取器的相反更新方向，pp.10-13。

> [3] Samar Khanna, Medhanie Irgau, David B. Lobell, Stefano Ermon. “ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts.” *International Conference on Machine Learning (ICML)*, 2025. arXiv:2406.10973, public version p.1.  
> 本文引用位置：在新域继续自监督预训练、解冻 1-2 个 ViT block 并在其余层使用 LoRA，以及后续 LoRA 微调，p.1。

> [4] Xiangyu Chang, Sk Miraj Ahmed, Srikanth V. Krishnamurthy, Basak Guler, Ananthram Swami, Samet Oymak, Amit K. Roy-Chowdhury. “Plug-and-Play Transformer Modules for Test-Time Adaptation.” arXiv:2401.04130, 2024, public version p.1.  
> 本文引用位置：模块库、少量无标签目标样本、稀疏选择与无权重更新融合，p.1。

> [5] Yuto Kojima, Jiarui Xu, Xueyan Zou, Xiaolong Wang. “LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models.” arXiv:2502.02069, 2025, public version p.1.  
> 本文引用位置：只更新视觉语言模型图像编码器的 LoRA、测试时重构目标与报告结果，p.1。

> [6] Jia-Xuan Jiang, Wenhui Lei, Yifeng Wu, Hongtao Wu, Furong Li, Yining Xie, Xiaofan Zhang, Zhong Wang. “MFM-DA: Instance-Aware Adaptor and Hierarchical Alignment for Efficient Domain Adaptation in Medical Foundation Models.” arXiv:2503.00802, 2025, public version p.1.  
> 本文引用位置：少样本 UDA 设置、扩散式域转换和 channel-spatial alignment LoRA，p.1。

> [7] Weibai Fang, Haijun Che, Feiyang Ren, Qiancheng Lao. “Normality-Preserving Continual Industrial Anomaly Detection via Orthogonal LoRA Banks.” arXiv:2606.02042v1, 2026, public version p.1.  
> 本文引用位置：冻结历史 LoRA bank、正交新 adapter 与持续工业异常检测设置，p.1。

## 论文主页

> ExPLoRA 论文主页：https://arxiv.org/abs/2406.10973  
> PLUTO 论文主页：https://arxiv.org/abs/2401.04130  
> LoRA-TTT 论文主页：https://arxiv.org/abs/2502.02069  
> MFM-DA 论文主页：https://arxiv.org/abs/2503.00802
