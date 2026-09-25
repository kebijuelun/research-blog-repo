# 何时、何地信任 Teacher：UECR-GRPO 精华解读

数学推理的 RL 训练长期面临"信号两难"：Verifier 能判定回答对错，却说不出哪一步对错（信用广播到所有 token，且组内奖励全同时信号消失）；Teacher 能提供逐 token 稠密反馈，但"稠密偏好 ≠ 过程正确性"。论文的诊断显示：在 Verifier 失效的组中，84.3%–98.4% 的组 Teacher 仍能提供区分度；但在"一对一错"的回答对上，Teacher 排序与正确性冲突的比例高达 39.1%–50.8%，与盲审过程质量评判的一致率仅 56.8%——Teacher 有信息量，但不可全信。

UECR-GRPO 的关键洞见是：Teacher 证据应在**归一化之前**（而非像 ATOD、Distilled RL 那样在之后）进入更新，且在**轨迹级**与**token 级**分开处理。方法由两个模块构成：

- **PUU（路径效用统一）**：利用自回归分解，token 级 log 比率之和恰等于精确的路径对数密度比。在统一的 KL 正则目标下（存在唯一 Gibbs 最优解），将 Verifier 奖励与长度归一化的 Teacher 改进量合并为统一效用 $R^U = R^{\mathrm{task}} + \alpha R^T$，再做组内归一化。这样 Teacher 证据能在裁剪前改变回答排序，verifier-degenerate 组的信号不再归零。
- **ECR（熵校准信用重分配）**：诊断发现 82.4% 的大 Teacher gap 落在高熵区间（Teacher 自己也不确定），且直接用局部信号会悄悄改变任务信用总量。ECR 用带符号的 Teacher gap 定方向（tanh 有界）、用 Teacher 全词表熵做置信度衰减、再做响应内零和投影——理论上证明权重算术均值恒等于 1、符号保持不变，即只重分配 Verifier 信用而不增减总量。

实验覆盖 Qwen3-1.7B 与 4B 两个尺度、五个数学基准（AIME 2024/2025、AMC 2023、HMMT 2025 Feb/Nov，Avg@12）：1.7B 档平均 17.21%，超过最强基线 ATOD-aligned 0.89 个点（五项中四项领先）；4B 档平均 65.09%，超过 Distilled RL 0.56 个点。消融表明：联合归一化（PUU）比分别归一化提升约 2 个点，完整的 ECR（gap + 熵 + 投影）再进一步，二者缺一不可。离线审计还证实：PUU 救活了 56%–90% 的"废组"；$\alpha = 1$ 远在排序安全区内；ECR 把负信用精准集中到了盲审判定的错误步骤上。

结论：Teacher 是"策略相对的参考信号"而非正确性标签——当它能在 Verifier 沉默处区分轨迹时（何时）、在它低熵且方向明确的前缀处（何地）才值得信任。

> 本文参考自 [When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment](http://arxiv.org/abs/2609.28385v1)