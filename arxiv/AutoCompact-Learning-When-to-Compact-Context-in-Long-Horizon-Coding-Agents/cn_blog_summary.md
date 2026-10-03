# AutoCompact：让 Coding Agent 自己学会"何时压缩上下文"

长程 Coding Agent 的真正瓶颈不是"上下文会爆"，而是模型不懂**何时压缩、保留什么、压缩后如何继续**。AutoCompact 把这三个决策直接训练进策略，在 SWE-bench Verified 上 pass rate 达 **39.6%**（较基线 +9.2%），SWE-PolyBench Verified 达 **24.5%**（+5.0%），且在 $0.10–$4.00 所有推理预算档位全面领先。

## 问题：压缩不是防溢出

现有方法分两派，各有短板：

- **长度触发型**（CompactionRL、Claude Code 自动压缩）：时机绑定在"长度"而非任务进度，过时信息堆积，还可能在阶段中间丢掉关键证据。
- **主动型**：rubric 引导（SelfCompact）没有训练信号；离线插入压缩调用做 SFT（SWE-Compressor）保留了原有后续动作，模型学不到"压缩后该怎么行动"。

## 方法：把压缩变成可学习动作

- 给 Agent 增加 `compact()` 动作，可在上下文上限**之前**主动调用，用 `# Auto Context Summary` 工作状态摘要替换历史。
- **第一阶段 SFT**：基座模型几乎从不主动压缩，纯 prompt 教不会。于是用 Judge（GPT-5.5-Codex）在 rollout 每一步**执行前**在线纠正三类错误——时机、摘要内容、续接动作，环境执行纠正后的输出。在 379 个任务上收集 1,052 条轨迹微调。
- **第二阶段 RL**：在 SWE-Gym 上用 GRPO，奖励仅为最终 patch 是否通过测试的二值信号，压缩决策、摘要、续接与普通编码共享同一 advantage——压缩无需任何辅助目标。

## 结果与发现

- AutoCompact（39.6%）大幅领先所有基线；长度触发压缩甚至有害（Fixed Compaction 比 Base 低 1.6%）。
- 在线 Judge 纠正优于离线插入：AutoCompact-SFT 比 SWE-Compressor 高 1.2%；RL 再带来最大一跳（+7.4%）。
- **关键消融**：跳过同一 checkpoint 的 `compact()` 调用，$0.10 预算下 pass rate 掉 **19.9%**——压缩动作本身就在"赚钱"，预算越紧越值钱。
- **行为证据**：RL 后主动压缩率从 44.3% 升至 58.5%，摘要遗漏关键信息比例从 3.1% 降到 0.2%。仅靠任务成功信号，模型就学会了保住关键信息。
- **摘要自洽性**：RL 还学出 SFT 学不会的能力——记录的状态真正约束下一步行动（SFT 版摘要记着未解决的语法错误却提议提交，最终失败；RL 版先验证再提交，通过测试）。

## 启示

对 Agent 产品最大的提醒：别等上下文快爆才压缩——让模型在任务阶段转换处主动整理工作状态，收益比想象中大得多。将学到的主动压缩与现有长度触发兜底机制结合，是直接可做的改进方向。

> 本文参考自 [AutoCompact: Learning When to Compact Context in Long-Horizon Coding Agents](http://arxiv.org/abs/2610.02163v1)