# DLite：为 Agent 推理卸下历史包袱的无历史投机解码

Agent 应用每一次执行都要高频调用大模型，推理速度直接决定产品体验；但主流加速手段投机解码在 Agentic 负载下严重"水土不服"——接受长度从 5~6 暴跌到 1~2，损失超过 80%。这篇来自上海交大等机构的工作提出 DLite：彻底不维护 draft KV cache 的"无历史"投机解码框架，在 Agentic 任务上接受长度达 2~6，比 SOTA 半自回归方法 DSpark 最高提升 70%，同时消除约占端到端延迟 10%~20% 的 draft 端 KV cache 维护开销。

## 问题：两个症状，一个病根

Agentic 负载是"长输入、短输出、高频多轮"：每轮带着任务、工具、历史轨迹组成的几万 token 输入，输出可能只是一行工具调用。作者观察到现有方法两头受挫：

- **接受长度断崖下跌**：小 draft 模型独立理解几万 token 历史力不从心，预测分布与 target 越偏越远（AgentLongBench、SWE-Bench 上集体跌到 1~2）；
- **KV cache 开销不可忽视**：draft 要回灌被接受 token、长期保存状态做 prefix 复用，SGLang 实测 DeepSeek-V4 的 MTP 模块上这些操作占端到端延迟 10%~20%。

作者一针见血：两者同根同源——**draft 模型被迫独立编码和维护一份越来越长的历史**。此前的架构之争（自回归 vs 并行 vs 半自回归）都默认 draft 要"自己记住历史"，DLite 是第一个掀掉这条公理的工作。

## 方法：把"历史建模"变成"局部预测"

**无历史并行 Backbone。**DLite 直接复用 target 模型多个中间层的 hidden states 作为压缩上下文（加可学习 depth embedding 标注来源层），长上下文建模完全委托给 target，draft 只看当前 anchor token 加 $K$ 个投机位置，位置编号按局部窗口重新计数——预测问题规模与序列累积长度彻底脱钩。

**轻量串行 Head 只补缺失依赖。**并行 backbone 抹掉了投机块内的自回归依赖，而块外历史已压缩在 hidden states 里，缺的恰好只是块内前缀。DLite 用低秩 RNN 式模块递归压缩块内已生成 token，**直接在 hidden space 做修正、词表投影只做一次**，省掉 DSpark 式昂贵的中间全词表投影，对高并发吞吐是实打实的贡献。

**两阶段训练。**直接对 DLite 做 KL 蒸馏效果很差（异构条件下分布难对齐）。解法是"先搭桥再过河"：先训一个结构同构但保留历史 KV cache 的教师模型，用前向 KL（位置指数衰减权重）蒸馏给 DLite 作为平滑起点；再丢掉教师，用交叉熵 + 全变分距离（$\lambda_{\mathrm{CE}}=0.1$、$\lambda_{\mathrm{TV}}=1$）直接对齐 target。

## 结果：通用不丢分，长文和 Agent 大幅领先

- **通用负载**（GSM8K/MATH/HumanEval 等）：DLite 几乎全面持平或最优，Qwen3-8B 上 GSM8K 接受长度 6.48——去掉历史 KV cache 不牺牲草稿质量；
- **长上下文**（LongBench v2）：四类任务端到端加速全面领先，长对话历史上 3.88×~3.96×（DSpark 约 3.0×~3.16×）；
- **Agentic 负载**：优势最大。SWE-Bench 上 Qwen3-4B 接受长度 3.15（DSpark 仅 1.89），加速 2.68×；8B 上 3.19 对 1.94，提升约 64%。DFlash 在 AgentLongBench 甚至出现 0.95×——加速方法反而变慢；
- **高并发吞吐**：并发 1~64 下全面领先 DSpark；即便接受长度接近的任务（如 MT-Bench）吞吐仍更高，印证轻量 head 本身更快。

**消融要点**：KV cache 从滑窗到完全去除，充分训练后接受长度几乎一致——历史 KV 只在训练早期有用；target hidden states 取 14 层即饱和；sequence-wise 拼接优于 feature-wise；串行 head 秩 512（$D/8$）后收益收敛，说明 draft 所需信息可在低秩空间解算。

**一句话总结**：DLite 把 draft 模型从"小号复读机"变成"target 表示的本地解码头"，用零历史成本换回了 Agentic 场景下的加速能力。

> 本文参考自 [DLite: History-Free Speculative Decoding for Agentic AI](https://arxiv.org/abs/2610.23787v1)