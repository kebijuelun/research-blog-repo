# HybridAgent 精华总结：用潜空间思考，用文本行动

LLM Agent 的根本矛盾在于：外部交互必须用文本，而逐字推理又慢又贵；潜空间推理（Latent Reasoning）虽高效，却无法调用工具、难以审计。来自 Stanford、UIUC、Together AI 和 Meta 的 HybridAgent 让同一模型在一次生成中自主切换"潜空间思考"与"文本交互"两种模式：14B 版本在 11 个 benchmark 上平均超越最强 agentic baseline **17.4%**，同时文本 token 减少 **35.5%–54.7%**，推理提速 **1.8×–2.6×**，成本降低 **2.7×–3.7×**。

## 核心设计

- **HybridLink**：轻量残差适配器（$e_{t+1}=h_t+W_2\sigma(W_1 h_t)$，4B 模型仅约 1000 万参数），把最后一层 hidden state"翻译"回输入 embedding 空间，解决连续向量与离散 token 的表征失配。
- **模式切换**：模型生成 `<latent>` 进入无解码连续推理；当 latent 向量最近邻 token 恰为 `</latent>` 时退出。切换时机、latent 段长度完全由模型自主决定，无预设结构。
- **KV 即工作记忆**：文本、latent、工具观测统一产生 KV 条目跨轮复用；无损性定理保证缓存前向与全量前向逐位等价，省掉全部重算。
- **意外收获**：latent 向量天然贴近词表 embedding，可映射成人类可读的"伪文本"，兼具 latent 的效率与文本的可审计性。

## 训练方法

- **数据**：自动标注管线把纯文本 agentic 演示改写为 54K 混合轨迹——用外部 LLM 与模型自身预测熵判断每步归文本还是 latent（低熵放 latent，关键步留文本供审计），全程无人工标注。
- **HybridSFT**：先激活格式与 HybridLink，再做自我条件化（Self-Conditioned）训练——把模型自己预测的 latent 输入喂回去，闭合连续空间特有的曝光偏差。消融显示自我条件化在激活基础上平均再提升 **30.5%**。
- **HybridRL**：latent 转移是确定性的，策略似然只分解在采样文本位置（混合策略分解），配合精确 Rollout 回放，使 GRPO 式重要性比率在混合轨迹上良定义，且有梯度等价定理支撑。奖励只含正确性、格式、工具三项，不含任何模式使用的显式奖励。

## 关键结果

- Agentic 任务：14B 在 GAIA 52.4、WebWalker 49.5（ARPO 分别为 43.7 / 40.5），反超 GPT-5.6-Luna、Kimi-K3 等前沿大模型 Agent；纯 latent 方法（Coconut、SoftThinking）几乎不可用。
- 数学/知识任务：14B RL 版 AIME25 43.3、MATH500 88.2、GPQA-D 55.1，同规模最优；收益可下沉到 4B。
- 效率：文本 token 约减 2.7×，总生成步数最多减 54%，墙钟加速约 2.1×；每百万题成本 14B 从 470 美元降至 176 美元（估算把 latent step 也计为完整前向）。
- 可信性：因果干预证明 latent 语义内容被功能性使用（跨问题替换性能跌 44.9%–51.3%）；文本接口少 43%–49% token 但关键推理步密度反升 26%–28%，盲监视器 AUROC 与纯文本 Agent 相差不足 0.022。

一句话：HybridAgent 首次在同一生成过程中统一了"高效潜空间推理"与"文本 agentic 交互"，性能、效率、可审计性三者兼得，并为"latent 工作记忆作为一等公民"的新型 agent 架构指明方向。

> 本文参考自 [Hybrid Reasoning Agents](https://arxiv.org/abs/2610.18861)，代码开源：[github.com/togethercomputer/HybridAgent](https://github.com/togethercomputer/HybridAgent)