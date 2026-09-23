# RULER：用实例感知评分细则做奖励，让 8B 模型的 SVG 生成比肩 DeepSeek-V3

开放式 SVG 生成有一个本质困境：一条指令对应无数个合法渲染结果，不存在绝对视觉 ground truth，导致评估测不准、RL 训不好。RULER（蚂蚁集团、港科大（广州）、牛津等团队提出）的思路是：先解决"怎么评"，再把验证过的评分机制改造成 RL 奖励。

**先证"评得准"。** 作者用 900 个渲染样本和 10 位标注员验证：VLM 按多轴 rubric 打分与人类偏好的 Spearman 秩相关达 ρ=0.7929（Aesthetic 仅 0.6051、CLIP 仅 0.5518），成对排序一致性 γ=0.7574（对手约 0.53-0.55）——显式多维度拆解比不透明标量更能跟踪人类感知。

**方法核心：实例感知 rubric 奖励。** 每条指令由前沿模型（Claude-Opus-4.6）生成专属六项评分细则，横跨语义保真、视觉质量、渲染风格三条轴线（固定权重 5,5,5,4,5,5）。生成 rubric 时模型先"想象"一份理想 SVG 作质量锚点，但 RL 打分时完全不看它，因此全程无需配对 ground truth。策略模型采样多条 rollout 渲染后，由冻结的 Judge VLM（Qwen3-VL-8B）逐项打分，加权平均得到稠密奖励，驱动 GRPO 优化。数据侧经两步过滤得到 30,737 条 prompt-rubric 对，平均每条约 $0.0655。

**关键结论数字。**

- 主结果：Qwen3-8B 基座训练后，MMSVG-Illustration/Icon 上 Rubric 分数从 0.432/0.395 提升到 **0.693/0.683**，超过所有专用 SVG 模型，追平甚至反超 DeepSeek-V3；人类盲测对各基线胜率 53.3%~96.5%。
- 奖励设计对比：CLIP+Aesthetic+HPS 标量组合奖励导致典型 reward hacking（Icon 上 Rubric 崩到 0.262，序列长度暴涨至 6.3k token）；通用 rubric 有效但粒度粗；实例感知 rubric 最强且均衡。
- 消融：三轴缺一不可，去掉视觉质量跌幅最大；更严格的 Rubric-S 变体反而诱发 "text-hint hacking"——互补的轴至少和打分严格性一样重要。
- 鲁棒性：在 4B 基座上仍大幅有效；换用 GPT-5.5 作 rubric 生成器，性能基本持平，方法不绑定特定生成器。

**局限：** 奖励质量继承外部模型的偏置；RL 每步需渲染并查询 judge，成本高；三轴分解未必覆盖所有艺术意图。

> 本文参考自 [RULER: Instance-aware Rubric Rewards for SVG Generation](https://arxiv.org/abs/2609.25270)