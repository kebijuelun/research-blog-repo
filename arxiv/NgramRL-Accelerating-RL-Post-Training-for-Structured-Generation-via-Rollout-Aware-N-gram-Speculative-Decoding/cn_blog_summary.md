# NgramRL：用历史答案加速 RL 解码

RL 后训练的瓶颈往往不是参数更新，而是反复生成 rollout；NgramRL 把同一 prompt 的历史回答变成无需训练的草稿库，让目标模型并行验收。稳定阶段每次目标前向可提交 **3.60–7.77 个 Token**，Spider 上实测 rollout 吞吐最高提升 **1.38 倍**。

## 为什么旧答案能当草稿？

GRPO 每轮都要针对同一批 prompt 采样多组回答，rollout 可占单步训练时间的 70% 以上。直接把投机解码搬进 RL 却有三个障碍：GPU 显存紧张、策略持续变化、单次上下文中可复用内容有限。

NgramRL 抓住了被忽略的规律：**同一 prompt 会跨 epoch 反复出现，旧 rollout 因而成为免费且能自我更新的 drafter**。这在 SQL、代码和工具调用中尤其明显。相邻 epoch 的 Spider 测量显示，旧回答可覆盖新回答的 **48%–96%**，重复片段长达 **7–23 个 Token**。

## 缓存如何跟上策略？

系统按 prompt 独立维护 CPU 侧 n-gram 缓存，流程只有三步：**查旧答案—并行验收—写回新答案**。

- **起草**：从历史 rollout 提取 n-gram，以前缀查找续写，并采用最长匹配减少歧义；不增加模型参数，也不占用常驻 GPU 的草稿状态。
- **更新**：只保留最近 $W$ 步证据，既限制内存，也避免旧回答与当前策略脱节。
- **验证**：目标模型按标准投机采样规则接受或修正候选，因此缓存只改变提议速度，**最终采样分布不变**，而非保证两次运行逐 Token 相同。

笔者认为，关键不是单纯扩大缓存，而是 per-prompt 索引隔离题目语义、滑动窗口追踪策略漂移。消融中，全局 n-gram 缓存的平台期指标为 2.41，per-prompt 则达到 3.05；仅保留 4 次历史，也能获得完整历史接受收益的 **79%–91%**。

## 异步预取如何消除 GPU 等待？

CPU 缓存虽省显存，却让同步、查表和调度暴露在关键路径上，使 GPU 每轮约 **29%** 的时间空闲。NgramRL 在验证尚未结束时，就为可能的接受长度与 bonus token 组合提前备稿，即 fan-out prefetch（扇出预取）。

真实结果返回后，GPU 以 $O(1)$ Key Probe 取草稿；未命中仅让对应请求退回顺序查找，最坏退化为同步基线，不影响正确性。Spider 上，候选预算为 16 时，Qwen3-8B 实测加速 **1.38 倍**，接近 1.40 倍重叠上界；GSM8K 与 CodeContests 的实测收益则为 **1.05–1.06 倍**。

## 结论：把训练重复变成解码并行

- 稳定阶段，NgramRL 相对 SuffixDecoding 提升 **1.20–1.99 倍**，相对 PLD 最高达 **7.0 倍**。
- $n_{\max}=6$、草稿长度 $k=16$ 位于收益拐点；更长历史并不总是更好。
- 方法已集成到 verl 的 vLLM rollout worker，无需修改 Reward Model、优化器或策略架构。

它最适合 prompt 会复访、输出结构可复用的 RL 场景；冷启动、输出熵和软硬件配比仍决定实际收益，端到端数字不能直接外推到任意平台。

> 本文参考自 [NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding](http://arxiv.org/abs/2610.32881v1)