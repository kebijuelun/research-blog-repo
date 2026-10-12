# Medusa：给 LLM 多长几个"头"，推理提速 2.8 倍且质量无损

大模型推理慢的根因不是算力不足，而是自回归解码每生成一个 token 都要把全部参数从显存搬进缓存，计算单元大量时间在"等数据"（memory-bandwidth-bound）。ICML 2024 的 Medusa 另辟蹊径：不引入独立 draft model，直接在原模型最后隐状态上挂几个轻量解码头，并行预测多个后续 token。**Medusa-1 在不损失生成质量的前提下加速超 2.2 倍，Medusa-2 进一步达到 2.3–2.8 倍**，且对 serving 系统几乎零侵入。

## 核心方法

- **Medusa Heads**：在原模型隐状态上追加 K 个解码头，每个头只有一层带残差的 FFN，负责预测第 t+k+1 个 token。初始化时复用原 LM head 权重，训练从"已会预测"的起点开始，收敛快；与 backbone 共享表示，天然避免分布偏移。
- **Tree Attention**：各头取 top-k 预测的笛卡尔积构成候选树，配合树状 attention mask 一次前向并行验证全部候选，不扩大 batch size。自顶向下建树，结构静态、实现简单。
- **两档训练**：Medusa-1 冻结 backbone 只训 heads，配 4-bit 量化单卡 A100 训 Vicuna-7B 仅需约 5 小时；Medusa-2 用组合 loss + 差分学习率（heads 4 倍）+ 两阶段 warmup 联合训练，保住质量再提速。经验上 5 个头封顶，3-4 个即够用。

## 三个关键扩展

- **Typical Acceptance**：放弃严格分布一致的拒绝采样，只要候选 token 概率超过"硬阈值 + 熵相关软阈值"即接受。温度越高接受越长，与拒绝采样恰好相反，更贴合实际使用。
- **Self-Distillation**：没有训练数据时，用公开种子 prompt 让目标模型自生成数据；backbone 用 LoRA 微调，"关掉 adapter 就是教师模型"，一份权重完成知识蒸馏，显存几乎不增。
- **稀疏树优化**：用校准集统计各 head 各 top 预测准确率，贪心加入期望接受长度贡献最大的节点。**64 节点稀疏树即超过 256 节点稠密树**。

## 实验结果

- MT-Bench 上：Vicuna-7B Medusa-1/2 分别加速 **2.18 / 2.83 倍**，Vicuna-13B 为 **2.33 / 2.83 倍**；coding 类最高 **3.29 倍**，信息抽取类达 **3.62 倍**。
- 自蒸馏场景（Vicuna-33B、Zephyr-7B）同样 2.3–2.9 倍，质量分与原模型基本持平（最大负差仅 -0.14），且全面优于调优后的 Speculative Decoding 基线。
- 消融显示每项设计都有增量：多头并行 ~1.5x → tree attention ~1.9x → 稀疏树 ~2.2x → Medusa-2 训练 ~2.8x。
- 附录 roofline 分析证明：Medusa 把每步处理 token 数从 1 提到 q，直接将 decoding 算子从带宽受限推向算力受限（Llama-33B 上 FLOP/s 提升 44 倍）；最优候选数约 64，大 batch 场景收益会缩水。

一句话总结：让模型自己长几个头，省去 draft model 的一切麻烦——Medusa 已被 TensorRT、HuggingFace TGI 等主流推理库跟进，正成为 LLM 推理优化的标配组件。

> 本文参考自 [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)