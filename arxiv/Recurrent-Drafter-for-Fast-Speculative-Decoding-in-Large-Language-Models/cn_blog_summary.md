# ReDrafter：苹果用 RNN 草稿模型刷新投机解码加速纪录

大模型逐 token 生成时，每一步都要把全部权重从显存搬一遍，推理速度被内存带宽死死卡住——投机解码用小模型先"打草稿"、大模型并行验证来破局，但收益上限取决于草稿猜得准不准。Apple 这篇 ICLR 2025 论文提出 **ReDrafter**（Recurrent Drafter），在 H100 上把 Vicuna 推理加速至 **2.8 倍**，在 Apple Silicon（M2 Ultra）上也拿到最高 **2.28 倍**，全面超越 Medusa 和 EAGLE，且输出与原 LLM 严格一致。

## 三个核心设计

- **RNN 草稿模型**：一个挂在 LLM 最后一层隐藏状态上的单层 RNN，输入为隐藏状态 $h$ 与历史 token embedding。所有预测位置**共享同一套参数**——对比 Medusa 每个预测头独立参数、互不通信的做法，ReDrafter 参数量不随预测距离增长，还天然建模了 token 间的时序依赖。
- **Beam Search**：推理时同时维护多条候选序列。束越宽，"LLM 会接受的最长前缀"落在候选集里的概率越大。本质是把投机解码从"省内存带宽"推向"用冗余算力换延迟"，束宽束长成为可按硬件调节的旋钮。
- **Dynamic Tree Attention**：候选序列大量共享前缀，合并后候选集塌陷成前缀树，配合修改后的注意力掩码送去验证。树在运行时动态构建，且利用"所有候选等长"的性质用**纯张量运算**完成去重——核心实现只有五行代码，GPU 天然友好，稳定砍掉 30%–60% 的待验证 token。
- **知识蒸馏**：不用 ground truth 当标签，而是用 LLM 自己的采样结果做监督；反向传播只穿过草稿模型，LLM 完全冻结。相比直接拟合 ground truth，加速比和 Tokens/Step 提升约 **10%**。

## 实验结果

- **PyTorch / H100**（Vicuna 7B/13B/33B，贪心解码）：MT-Bench 上 ReDrafter 在 7B 和 13B 取得 **2.80x** 加速、Tokens/Step 达 4.20/4.21，双双第一；33B 上 Tokens/Step 仍最高（3.87）。temperature=1 拒绝采样下优势进一步扩大至 **3.50x / 3.51x / 3.27x**。
- **MLX / Apple Silicon**（batch size=1）：M2 Ultra 上 7B/13B/33B 分别加速 **1.52x / 1.94x / 2.28x**——模型越大、越 memory-bound，收益越大。
- **最优束宽随算力移动**：H100 最优束宽 50 以上，M2 Ultra 是 3，M1 Max 只有 1；端侧 BW=4 时甚至出现负加速（0.75x）。
- **部署指引**：要低延迟就"宽束 + 小 batch"，要总吞吐就"窄束 + 大 batch"（BW=2、BSZ=80 时吞吐达 1636 TPS×BSZ）。

## 小结

ReDrafter 的公式是：RNN 参数共享草稿 + beam search 换接受率 + 张量化树注意力去重 + 冻结 LLM 的局部蒸馏。其中动态树注意力与蒸馏目标不依赖 ReDrafter 本身，可迁移到其他带 beam search 的投机解码方案。局限在于 33B 大模型上验证开销开始侵蚀收益，端侧大模型场景还需量化等手段配合。

> 本文参考自 [Recurrent Drafter for Fast Speculative Decoding in Large Language Models](https://arxiv.org/abs/2403.09919)