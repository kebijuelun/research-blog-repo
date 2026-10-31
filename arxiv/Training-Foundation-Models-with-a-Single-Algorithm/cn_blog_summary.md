# Unified Training：一套 Loss 贯穿训练

基础模型训练常被 Pretraining、Midtraining、SFT 和 RL 拆成不同目标与系统，使数据复用、遗忘和初始化效应难以联合研究。Unified Training 把合适的 RL 更新也写成最大似然，让不变的学习规则作用于时变数据混合。受控推理中，最佳 Unified 的 Base pass@1 约 **39%**，Staged 为 **24.54%**；真实数据接续 Coding RL 后，5% Unified 的 HumanEval+ pass@1 达 **24.96%**，高于 Staged 的 **22.94%**。

## 核心方法：混合数据，而非拼接算法

所有阶段都可看作拟合某个目标答案分布：

$$
\mathcal{L}(\theta;q_t)=-\mathbb{E}_{(x,y)\sim q_t}[\log\pi_\theta(y\mid x)],\qquad q_t=\sum_s\lambda_{s,t}(\rho_{s,t}\circ\mu_{s,t}).
$$

- **MaxRL 补齐 RL**：只保留模型采样中的成功答案，再对其做最大似然，等价于优化成功概率的对数。它像让学生模仿自己答对的解法，教材随策略更新。
- **Unified 统一全程**：静态语料、SFT 示范与 RL 成功样本共享 Loss、数据格式和 Trainer；Staged 只是某时刻单一数据源权重为 1 的特例。
- **异步但不混淆边界**：生成器可用旧模型快照，Trainer 同时消费静态与生成数据；Off-policy 数据不会因此自动变成 On-policy。

## 实验：看完整轨迹，而非阶段终点

- **受控推理**：在 124M 模型上，Unified 在 Base、Extended 和 12 人泛化上的最佳 pass@1 约为 **39%、24%、13%**，对应 Staged 为 **24.54%、16.91%、5.58%**。但 Staged 在部分中等算力区间仍更强，Unified 并非全预算获胜。
- **能力保留**：Unified 各混合档位的 Pretraining Loss 为 **2.537–2.540**，仅 Pretraining 参考为 2.537；最佳保留型 Staged 则升至 **4.135**，五项通用基准均分由 **42.37%** 降至 **38.49%**。
- **初始化价值**：Midtraining 结束时的即时分数不能预测后续 RL 表现；跨阶段复用相同数据，比单纯增加 Staged Epoch 更关键。

## 真实数据：保留与准确率存在交换

在 0.76B 模型上，论文只验证数学 RL 期间混入 Midtraining，并未测试 Unified Pretraining。更多混合通常降低保留集 Loss、提高多次采样覆盖率，却可能损害单次准确率：0% 与 5% Unified 的 Coding pass@1 分别为 **20.29%** 和 **15.93%**，而保留集 Loss 变化由 **+0.0737** 降至 **−0.1474**。混合时机同样重要；平均比例相同，半程退火的 Coding pass@1 达 **18.85%**，恒定混合仅 **16.51%**。

## 结论

- 论文的贡献不是发明新 RL 算法，而是用共同的最大似然语言串联完整训练流程。
- 数据的内容、比例、时机和复用成为核心变量，单阶段最优不等于整段训练最优。
- Unified 更突出地改善高算力推理、泛化、能力保留与后续学习，但并不在所有预算和指标上占优。
- 目前证据限于 124M 与 0.76B 模型；跨尺寸、多任务的可预测数据分配规律仍待建立。

> 本文参考自 [Training Foundation Models with a Single Algorithm](http://arxiv.org/abs/2610.40564v1)