# CV-Learning

本仓库整理计算机视觉与机器学习的理论知识和论文，重点关注 **分布偏移下的工业异常检测**：训练只使用正常样本时，如何在产品、相机或光照条件变化后，识别并定位未知缺陷。

基础部分介绍理解这些方法所需的深度学习与视觉知识；领域自适应部分讨论如何利用源域知识适应目标域。本库只收录理论文档、论文原文和正文引用的论文图片。

## 从哪里开始

先看 [学习路径](learning-path.md)，了解已有内容与待编写主题，再阅读两篇总览：

- [工业异常检测：问题设定与方法总览](03-Anomaly-Detection/problem-setting-and-overview.md)：训练能使用什么数据，测试需要识别什么异常，各类方法依赖什么假设。
- [领域自适应：问题设定与方法总览](04-Domain-Adaptation/problem-setting-and-overview.md)：源域与目标域是什么关系，适配时能使用哪些数据，迁移会怎样影响异常识别。

基础知识不足时，可从 [卷积](01-Foundations/convolution.md)、[注意力](01-Foundations/attention.md)和 [损失函数](01-Foundations/losses.md)补起。学习路径中的链接区分已有正文、阅读草稿、仅有论文和待编写主题。

## 内容如何组织

分类目录用英文学术术语命名：缩写全大写，英文实词首字母大写、虚词小写，连字符保留；数字前缀保留。`01` / `02` 按核心研究问题划分，`03` / `04` 按方法族划分，空族保留。

| 目录 | 内容 |
| --- | --- |
| `01-Foundations/` | 深度学习基础 |
| `01-Foundations/Backbone/` | 骨干网络 |
| `01-Foundations/PEFT/` | 参数高效微调 |
| `02-Vision-Tasks/` | 基础视觉任务与多模态 |
| `02-Vision-Tasks/Classical-Vision/` | 传统视觉 |
| `02-Vision-Tasks/Object-Detection/` | 目标检测 |
| `02-Vision-Tasks/Segmentation/` | 图像分割 |
| `02-Vision-Tasks/Multimodal/` | 多模态 |
| `02-Vision-Tasks/Generative/` | 生成模型 |
| `03-Anomaly-Detection/` | 工业光学异常检测的问题设定、方法族与综述 |
| `03-Anomaly-Detection/OCC/` | 单类分类 |
| `03-Anomaly-Detection/Memory-Prototype/` | 记忆库与原型 |
| `03-Anomaly-Detection/Synthesis-Reconstruction/` | 合成与重建 |
| `03-Anomaly-Detection/Flow-Based/` | 流模型 |
| `03-Anomaly-Detection/Distillation/` | 蒸馏 |
| `03-Anomaly-Detection/VLM-AD/` | 视觉语言模型异常检测 |
| `03-Anomaly-Detection/Survey/` | 异常检测综述 |
| `04-Domain-Adaptation/` | 领域自适应；先介绍问题设定，再按方法族组织 |
| `04-Domain-Adaptation/Adversarial/` | 对抗 |
| `04-Domain-Adaptation/Alignment/` | 对齐 |
| `04-Domain-Adaptation/Discrepancy-Minimization/` | 差异最小化 |
| `04-Domain-Adaptation/Reconstruction-and-Disentanglement/` | 重建与解耦 |
| `04-Domain-Adaptation/Generative/` | 生成式翻译 |
| `04-Domain-Adaptation/Fusion-and-Mixing/` | 融合与混合 |
| `04-Domain-Adaptation/Mean-Teacher/` | 平均教师 |
| `04-Domain-Adaptation/Pseudo-Label/` | 伪标签与自训练 |
| `04-Domain-Adaptation/New-Problem-Settings/` | 新问题设定 |
| `04-Domain-Adaptation/VLM-Prior/` | 视觉语言模型先验（入口，不是 DA 方法） |
| `04-Domain-Adaptation/Survey/` | 综述 |
| `99-Inbox/` | 尚未归入上述领域的论文 |

`02` 与 `04` 下各有一个 `Generative/`，含义不同，按父目录区分。上表是分类目录的唯一清单，新增分类目录必须登记在这里；论文子目录用英文题名或已有英文文件夹名，不译成中文。

**方法族**指基本建模思路相近的一组方法，例如记忆库与原型方法、对抗对齐方法。**主题串讲**围绕一个方法族或基础主题，连贯介绍问题、假设、公式及方法之间的关系，是本库公开正文的主要形式。

一篇论文对应一个目录，原文与书目信息分别保存在 `paper.pdf` 和 `sources.md` 中。论文旁的单篇精读仍按阅读草稿对待；目录中只有 `to-be-filled.md` 时，表示该主题尚未补充内容。

## 维护说明

正文用中文，分类目录名用英文；目录名的语言不改变正文的术语难度。新增或修改内容都提交在 `master`。

维护规则集中在以下文档：

- [论文归档规范](agent/paper-unit.md)：论文目录布局、命名与 `sources.md` 字段。
- [阅读草稿规范](agent/deepdive.md)：单篇笔记的保存与核验要求。
- [Fork 配置](fork-profile.md)与 [配置应用规则](agent/apply-profile.md)：仅供需要调整自己副本结构的维护者使用。
- [协作准则](AGENTS.md)：内容范围、出处与术语要求。

## 理论文档的写法

正文从具体问题出发，依次说明模型假设、数学定义和信息流。论文发现、业界共识、数学推导与分析判断分别标注；方法、公式和实验数据给出原始出处及页码。非通用英文缩写首次出现时，给出英文全称、中文翻译和一句话定义。
