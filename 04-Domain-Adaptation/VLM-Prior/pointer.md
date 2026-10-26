# 视觉语言预训练的跨主题入口

VLM（Vision-Language Model，视觉语言模型）联合处理图像与文本表示。DA（Domain Adaptation，领域自适应）利用源域知识与目标域数据处理域间差异。预训练表示与适配协议是两个层次。

**[分析判断]** GLIP、RegionCLIP、GLIPv2、Grounding DINO 的基础问题是区域–文本预训练、视觉定位与开放词汇检测。若将它们用于领域自适应，必须另外说明源域、目标域以及适配时可用的数据；仅使用这些模型不构成完整的适配协议。本目录保留已有论文位置，以下链接连接相关主题。

- 图文预训练：[CLIP 串讲](../../02-Vision-Task-Foundations/clip.md)；[CLIP 原文](<../../01-Foundations/Backbone/[ICML 2021] CLIP - Learning Transferable Visual Models From Natural Language Supervision/paper.pdf>)。
- 检测谱系：[DETR 系列](../../02-Vision-Task-Foundations/DETR.md)；用于理解 Grounding DINO 的检测结构。
- 异常识别：[视觉语言模型异常检测](../../03-Anomaly-Detection/VLM-AD/)；这里需要正常/缺陷评分与相应评价协议。
- 适配协议：[领域自适应总览](../problem-setting-and-overview.md)。
