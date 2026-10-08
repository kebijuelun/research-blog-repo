# 一个模型家族，两枚金牌：Nemotron 微调后横扫 IOI 与 IMO

一个通用底座模型能否被系统地改造成世界级领域专家？NVIDIA 用 Nemotron 3 给出了肯定答案：仅靠 SFT、RL 和反馈式推理这些标准方法，同一模型家族就在两条几乎正交的赛道上双双夺金——IOI 2026 拿下 **535.4/600 分**（金牌线 361.12，甚至超过人类最高分 498.27），IMO 2026 拿下 **30/42 分**（官方金牌线 29，由官方阅卷人评分）。

## 可复用的四步配方

两个项目共用同一套方法论，证明能力来自底座和可迁移的特化流程，而非针对某场比赛的奇技淫巧：

1. 从强大的 Nemotron 底座出发；
2. 整理领域题目与高质量推理轨迹；
3. 标准后训练（SFT，必要时加 RL）；
4. 配上"生成—评估—改进"推理循环。

## IOI 之路

- 整理 **22,000 道竞赛题**，训练 Nano-CC（300 亿参数，SFT+RL）和 Ultra-CC（5500 亿参数，仅 SFT）两个专家。
- IOI 2025 递进实验：Nano 从 130 分 → SFT 280 → RL 291 → 叠加 GenCorrect 后 **468 分**（金牌线 438.3）；Ultra-CC 直接拿到 502 分。
- 关键发现：**底座越强，需要的塑形越少**——Ultra 仅一个 epoch SFT 就全面超越完整后训练的 Nano。

## IMO 之路

- **SFT 语料**：414,890 条样本、15,818 道证明题，覆盖证明的生成、改进、验证、**元验证**四类任务——核心是让模型学会"判断力"。
- **RL 训练**：在 9,597 道位于能力边界的题目上进行。
- 推理时由 SFT、RL 双 checkpoint 加通用模型组成 generate-verify-refine 系统，**纯自然语言运行，无形式化证明器、无工具、无联网**，六道题中四道满分。

## 核心洞察

后训练与测试时计算是**乘法关系**：GenCorrect 放大微调增益，互补 checkpoint 优于暴力采样。金牌来自模型、数据、推理循环的共同设计——只卷训练或只卷推理都会浪费另一半增益。

所有模型、数据、代码均已开源（Nemotron Labs IMO collection、NeMo-Skills、Nemotron-3-Ultra-CC），配方可复现、可迁移，但超大算力门槛短期内仍是现实限制。

> 本文参考自 [One Model Family, Two Gold-Level Results: Fine-Tuning Nemotron for IOI and IMO](https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026)