# Path glossary

分类目录的唯一对照表。中文分支生成器按本表转换路径；Markdown 文件名与论文目录名保持原样。

- **id**：稳定，kebab-case，不随目录大小写变。
- **English**：`master` 上的分类目录（Title-Case + 缩写全大写），带斜杠。
- **zh-dir**：`zh` 上的分类目录。生成器只认这一列。
- **Chinese**：给人看的标题，不是路径，不要 `mkdir`。

规则：缩写全大写；英文实词首字母大写、虚词小写、连字符保留；中文段只出现在 `zh-dir`。数字前缀保留。`.md` 文件名与论文目录名不译。

分类目录必须登记在本表中；生成器遇到未登记的分类会报错。表中已登记但磁盘上不存在的分类不会自动创建。增删分类后的同步步骤见 [维护与生成说明](generate-zh.md)。

## Top-level

| id | English | zh-dir | Chinese |
|---|---|---|---|
| foundations | `01-Foundations/` | `01-基础理论/` | 基础理论 |
| vision-tasks | `02-Vision-Tasks/` | `02-基础视觉任务/` | 基础视觉任务 |
| anomaly-detection | `03-Anomaly-Detection/` | `03-缺陷检测/` | 缺陷检测 |
| domain-adaptation | `04-Domain-Adaptation/` | `04-领域自适应/` | 领域自适应 |
| inbox | `99-Inbox/` | `99-待分类/` | 待分类 |
| learning-path | `learning-path.md` | — | 学习路径 |

## Foundations (`01-Foundations/`)

| id | English | zh-dir | Chinese |
|---|---|---|---|
| convolution | `convolution.md` | — | 卷积 |
| attention | `attention.md` | — | 注意力 |
| losses | `losses.md` | — | 损失 |
| probability | `probability.md` | — | 概率工具 |
| peft | `PEFT/` | `参数高效微调/` | 参数高效微调 |

## Vision tasks (`02-Vision-Tasks/`)

| id | English | zh-dir | Chinese |
|---|---|---|---|
| classical-vision | `Classical-Vision/` | `传统视觉/` | 传统视觉 |
| object-detection | `Object-Detection/` | `目标检测/` | 目标检测 |
| segmentation | `Segmentation/` | `图像分割/` | 图像分割 |
| multimodal | `Multimodal/` | `多模态/` | 多模态 |
| generative | `Generative/` | `生成式/` | 生成式 |
| clip | `clip.md` | — | CLIP |
| cross-attention | `cross-attention.md` | — | Cross-Attention |
| diffusion | `diffusion-models.md` | — | 扩散模型 |
| rcnn | `rcnn.md` | — | R-CNN 系列 |
| yolo | `yolo.md` | — | YOLO 系列 |
| faster-rcnn-to-detr | `DETR.md` | — | Faster R-CNN → DETR |
| classical-cv | `classical-cv.md` | — | 传统 CV 算法 |
| graph-and-analogy | `graph-and-analogy.md` | — | 图卷积与类比 |
| miscellaneous | `miscellaneous.md` | — | 其余技术 |

## Anomaly detection (`03-Anomaly-Detection/`)

| id | English | zh-dir | Chinese |
|---|---|---|---|
| ad-overview | `problem-setting-and-overview.md` | — | 问题设定与方法总览 |
| occ | `OCC/` | `OCC/` | OCC / 超球体 |
| memory-prototype | `Memory-Prototype/` | `记忆与原型/` | Memory / prototype |
| feature-synthesis | `Feature-Synthesis/` | `特征合成/` | 特征合成 + 判别 |
| distillation | `Distillation/` | `蒸馏/` | 蒸馏 |
| normalizing-flow | `Normalizing-Flow/` | `归一化流/` | 归一化流 |
| vae | `VAE/` | `VAE/` | VAE |
| vlm-ad | `VLM-AD/` | `VLM-AD/` | VLM-AD |
| ad-survey | `Survey/` | `综述/` | 异常检测综述 |
| to-be-filled | `to-be-filled.md` | — | 待填 |

## Domain adaptation (`04-Domain-Adaptation/`)

| id | English | zh-dir | Chinese |
|---|---|---|---|
| da-overview | `problem-setting-and-overview.md` | — | 问题设定与方法总览 |
| adversarial | `Adversarial/` | `对抗/` | 对抗 |
| alignment | `Alignment/` | `对齐/` | 对齐 |
| discrepancy-minimization | `Discrepancy-Minimization/` | `差异最小化/` | 差异最小化 |
| reconstruction-and-disentanglement | `Reconstruction-and-Disentanglement/` | `重建与解耦/` | 重建与解耦 |
| generative-da | `Generative/` | `生成式翻译/` | 生成式翻译 |
| fusion-and-mixing | `Fusion-and-Mixing/` | `融合与混合/` | 融合与混合 |
| mean-teacher | `Mean-Teacher/` | `Mean-Teacher/` | Mean Teacher |
| pseudo-label | `Pseudo-Label/` | `伪标签/` | 伪标签 / 自训练 |
| new-problem-settings | `New-Problem-Settings/` | `新设定/` | 新设定 |
| vlm-prior | `VLM-Prior/` | `VLM先验/` | VLM 先验（入口，不是 DA 方法） |
| survey | `Survey/` | `综述/` | 综述 |
| pointer | `pointer.md` | — | 入口 |

`generative`（02）与 `generative-da`（04）的 English 都是 `Generative/`，`zh-dir` 不同。映射键是相对其父分类的路径，不是裸文件夹名。

论文子目录用英文题名或已有英文文件夹名，不译成中文，也不为「缩短」改 PDF 文件名。
