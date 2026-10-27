# BLOOM：1760 亿参数多语模型走向开放

大模型长期被少数机构封闭开发，非英语语言也常被边缘化。BLOOM 以开放协作公开 **1760 亿参数**模型，覆盖 **46 种自然语言和 13 种编程语言**；微调版 BLOOMZ 又显著增强了 Zero-Shot 指令泛化。

## 核心方法

- **ROOTS 语料**：498 个数据集组成 1.61 TB 文本，覆盖 59 种语言。语言社区参与来源选择与过滤，逐来源去重和隐私处理，保留来源边界以便追溯。
- **架构**：采用 Causal Decoder-only Transformer。ALiBi 按 token 距离惩罚注意力分数，改善稳定性和下游表现；Embedding LayerNorm 用于避免训练发散。
- **Tokenizer**：250,680 词表的 Byte-level BPE 保留空白、换行和原始 Unicode 形式，避免英语专用切分伤害其他语言与代码。
- **训练**：384 张 A100 80GB GPU 运行约 3.5 个月，以三维并行和 ZeRO Stage 1 扩展；bfloat16 混合精度降低发散风险。

## 关键结果

- **多语能力**：Flores-101 的 One-Shot 评测中，BLOOM 在多个方向接近或超过监督模型 M2M-100；多语摘要也优于 OPT。
- **低资源短板**：Swahili 与 Yoruba 的训练 Token 各少于 50k，互译仍很差。“覆盖语言”不等于可靠掌握。
- **指令泛化**：原始 BLOOM 在不少 SuperGLUE 任务上接近随机；xP3 用 83 个数据集、46 种语言、16 类任务微调后，BLOOMZ 在未见多语任务上明显提升。
- **代码能力**：BLOOM 176B 的 HumanEval pass@1 为 15.52%、pass@100 为 55.45%，但不及 Codex。

训练耗电 433 MWh，能源排放约 25 吨 CO₂eq；计入制造和空闲后约 81 吨。BLOOM 公开权重，也公开治理、成本与失败模式。

> 本文参考自 [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)