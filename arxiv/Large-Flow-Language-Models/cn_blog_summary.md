# Lafa：分块异步 Flow 提速 2.54 倍

自回归（AR）大模型每次前向只能提交一个 Token，串行生成成为推理吞吐的瓶颈。Lafa（Large Flow Language Models）的解法是：**保留块间自回归，在块内学习连续异步 Flow，并用模型自身保留的 AR 预测器验证 Flow 草稿**。最终用少于 30B 训练 Token 将模型扩展到 8B，Lafa 7B 在质量相近时达到 AR SFT 基线的 **2.54 倍端到端吞吐**（100.2 vs 39.5 TPS），平均准确率还从 73.9% 微升至 74.5%。

## 核心设计

- **块间因果、块内并行**：已提交的块始终是干净前缀，可像普通 AR 一样复用 KV Cache；只有当前块处于连续噪声状态，块内用双向注意力联合去噪。
- **Tokenwise Time**：每个 Token 有自己的"进度条" $\tau^i(t)$，靠前位置更早去噪。时间表不是手工设定，而是从各位置去噪 Loss 中在线学出——Loss 高的困难区间自动推进更慢。这种从左到右的节奏是训练自然涌现的。
- **从预训练 AR 改造，而非重训**：
  - 用恒等初始化的 adaLN 注入时间条件，不破坏预训练计算；
  - 注意力掩码让前缀看不到噪声块，缓存友好成为结构性质；
  - 共享词表 embedding 但固定 L2 范数，只学方向、不改长度；
  - 一个联合 Loss 同时训练 AR 与 Denoiser（AR 交叉熵 + 去噪交叉熵 + embedding MSE），所有块并行训练、不重复计算前缀。

## 三种解码方式

- **Flow Decoding**：多步 Euler 积分换质量，7B、$B=4$、32 步达 75.2%，超过 AR SFT 的 73.9%。
- **Flow Map Decoding**：把多步蒸馏成少步跳跃，7B 两步学生达 63.3%，接近四步教师的 64.3%；但一步生成（49.0%）距多步仍远。
- **Self-Speculative Decoding**：一步 Flow 起草整块，AR 从左到右验证，只提交最长一致前缀——**输出严格等于 greedy AR**，加速不靠误差容忍。验证与下一块重起草还融合进同一次前向。

## 关键实验结论

- **联合训练反而增强 AR**：Lafa 单 Token 解码在 0.5B/7B/8B 上比匹配 AR SFT 分别提升 2.9、2.5、6.3 个百分点；8B 的 63.1% 与 Efficient-DLM 的 63.5% 接近，但训练 Token 至少少 $18\times$。
- **并行度（TPF）**：7B 从 $B=4$ 到 $B=16$，每前向提交 Token 数从 2.6 增至 4.9；$B=16$ 平均准确率 58.0%，与 AR SFT 的 58.4% 基本持平。
- **真实吞吐**（单张 H100，batch size 1）：$B=4$ 约 2.02 倍且质量更高（76.5%），$B=16$ 达 2.54 倍；DeepSeek-MTP 与 MARS 最高仅约 1.4 倍。
- **消融**：块双向注意力比全因果高 5.3 个百分点；异步时间再加 1.8；Self-Conditioning 带来 5.8 个百分点且推理时无需额外前向；Prefix Guidance 再加 3.8。

## 总结

连续 Flow 不必替代 AR：让"Flow 起草 + AR 验证"在同一模型中协作，反而形成更实用的质量—速度前沿。下一步关键是改进 Flow Map 对 Self-Conditioning 与 Prefix Guidance 的蒸馏，缩小少步与多步生成的质量差距。

> 本文参考自 [Large Flow Language Models](http://arxiv.org/abs/2610.28559v1)