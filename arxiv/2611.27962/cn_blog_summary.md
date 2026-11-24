# Model Carving：训练时雕刻拓扑，推理省34%算力

Transformer 的拓扑通常在训练前固定，Model Carving 让模型在同一次训练中联合学习权重与结构，把“贡献配不上算力”的子层删除，并将省下的训练算力换成更多 tokens。实验最高覆盖 Standard 2.7B 和 Looped 1.3B：每 token 推理 FLOPs 最多减少 **34%**，loss 优势最高相当于 full-depth 基线多消耗约 **1.7 倍**训练算力。

## 核心方法：按学习曲线决定删什么

- **可关闭的子层**：每个 Attention 或 MLP 应用通过标量门写入残差流，RMS 归一化让门只控制相对幅度；门初始化为 0.1。DenseFormer 风格的可学习跨层连接则让保留层复用早期表示。
- **算力压力不是固定稀疏惩罚**：删除子层后，剩余训练阶段省下的 FLOPs 可换取更多 tokens；压力就是这部分额外训练预计带来的 loss 改善。训练早期剩余时间长、压力较大，最后 20% learning-rate cooldown 阶段压力清零。
- **在线估计学习速度**：用旧参数快照和当前参数在同一 batch 上计算 loss，抵消数据难度差异，再以幂律外推训练终点的学习速度；每次运行仅增加 101 至 204 次前向计算，且计入训练预算。
- **真正得到静态小模型**：对数惩罚持续推动低贡献门下降，裁剪使其精确归零；零门子层被编译出计算图。最终无需 router、mask 或动态早退，所有 tokens 共用一张静态图。

## 关键结果：质量、推理成本与生成收益同时改善

- 九组主实验在**相同训练 FLOPs**下全部取得更低 bpb，相同训练 tokens 下也保持优势；所有预先减层的 shallower-depth 基线都更差，说明收益并非只是“小模型多训 tokens”。
- Standard 模型最终参数量为初始模型的 66% 至 86%。相同 batch 下解码最高加速 **1.45 倍**，各自最大 batch 下最高 **1.61 倍**；Standard 2.7B 的 KV cache 降至基线的 69%。
- 等推理预算下，Standard 2.7B carved 可采样 24 次，full-depth 只能采样 16 次，GSM8K、MATH-500 等五项任务全部转为正收益，其中 GSM8K 提升 **0.091**、MATH-500 提升 **0.052**。
- 在对齐预算的剪枝对照中，Standard 1.3B carved 的 CORE-22 领先 1.9 至 3.7 分；蒸馏学生也全部落后，但教师训练与前向成本未计入，结论需限定在这一预算口径内。
- Compute-optimal 分析显示优势随预算扩大：达到同等 loss 所需的基线算力倍数，Looped 从 1.16 增至 1.32，Standard 从 1.10 增至 1.25。

## 学到了什么，边界在哪里

Standard 模型倾向在总深度约前三分之一的连续区段删除子层，而非简单删深层；保留下来的子层更难替代，knockout 后的中位 loss 增幅相对 full-depth 从 140M 的 2.4 倍升至 2.7B 的 5.5 倍。跨 seed、指令微调和低比特量化实验也提供了稳健性证据。

当前方法从随机初始化开始训练，尚未证明可直接改造已有预训练模型；压力预测依赖恒定 learning-rate 假设，最大验证规模仍为 Standard 2.7B、Looped 1.3B。

> 本文参考自 [Model Carving: Training Transformers That Optimize Their Own Topology](http://arxiv.org/abs/2611.27962v1)