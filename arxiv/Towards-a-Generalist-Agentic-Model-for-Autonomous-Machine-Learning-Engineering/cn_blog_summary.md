# MLExpert：让 35B 模型自主做机器学习实验，奖牌率翻近一倍

机器学习工程（MLE）最难的不是写代码，而是在数小时的实验循环中自主决定下一步改什么、何时回滚、提交哪个候选——这正是 AI for AI 走向递归自我改进的关键一环。MLExpert 将真实 Kaggle 任务、模型主导的交互环境、分层 Agentic SFT 与 Windowed Interaction RL（WIRL）连成训练闭环，把一个 35B MoE 模型（Qwen3.6-35B-A3B）在 MLE-Bench Lite 上的 Any Medal 从 37.9% 提升到 72.7%（+34.8 个百分点，追平 GLM-5.3、Kimi-K3、GPT-5.5），并在 NatureBench Lite 上将 Surpass-SOTA 从 0.0% 提升到 50.0%。

## 环境：真实任务 + 模型主导的实验世界

- **Kaggle-MLE Factory**：多 Agent 流水线把历史 Kaggle 比赛自动加工成"公开工作区 + 隐藏测试 + 统一评分"的可执行任务包，从 11,649 条竞赛元数据出发，最终人工审查保留 154 个任务，覆盖视觉、文本、音频、表格等模态。训练信号来自真实执行和真实评分。
- **Interaction Harness**：ReAct-style 循环，但分支选择权交给模型。每次修改形成 solution node，连成 experiment tree；模型可比较节点、回滚或换路线，并通过 `invalidate_node` 剔除虚高候选、用 `update_answer` 维护提交。实验记忆与提交责任都显式交给模型。

## 训练一：分层 Agentic SFT

- **两阶段迁移**：先用通用编码与工具轨迹学会操作，再用 636 条教师轨迹（MLE 由 GLM-5.2 生成，科学实验由 GLM-5.2 与 Qwen3.8-Max 生成）学会算法设计与迭代优化；训练数据共 5,539 条记录、约 3 亿 tokens。
- **三类互补示范**：开局方案、长程优化、分析 sub-agent 轨迹，避免模型只会"把第一版跑起来"。
- **结果引导筛选**：以提交峰值（阈值 η=0.4）为锚点截断轨迹，保留峰值前的探索与失败恢复，切掉持续退化的尾部（δ=0.05）。

## 训练二：WIRL——按执行窗口在线学习

- **Execution window**：两次可评分实验结果之间的决策划为一个窗口，长任务未结束就已产生训练数据，不必等整条轨迹。
- **窗口奖励**：按奖牌阈值给绝对质量分（执行错误 -1.0，金牌 2.0+），再叠加与上次提交相比的趋势项 T（±0.2），直接作为共享权重，不用 group-relative 归一化。
- **异步收集 + token 级 policy correction**：长轨迹跨越多次参数更新时，按 token 的 log-prob 比值（区间 [0.2, 4.0]）加权校正，配 PPO-style 裁剪与 KL、熵正则。

## 结果：每个阶段都有增益，迁移超出 Kaggle

- **MLE-Bench Lite**：Base 37.9% → +General SFT 47.0% → +MLE SFT 62.1% → +WIRL 72.7%；Gold 从 22.7% 升至 50.0%。换用通用 Claude Code harness 仍有 69.7%，领先同规模最优 22.7 个百分点，说明提升来自模型而非专用脚手架。
- **RL 消融**：GRPO（56.1%）和 PPO-style（60.6%）反而低于 MLE-SFT；去趋势奖励或去策略校正的 WIRL 居中，完整 WIRL 最高。异步窗口还把 learner 等待比例从 GRPO 的 87.6% 降到 39.3%。
- **NatureBench Lite**：Surpass-SOTA 0.0% → 50.0%，与 GLM-5.3 持平。
- **AIME scaffold 优化**：让模型优化调用固定 solver 的 agent scaffold，MLExpert 留出集平均准确率 43.7%（Base 23.3%）；SFT 版仅用 3.51 次内部调用即达 38.1%，比 Kimi-K3 少用约 59% 调用。
- **MLS-Bench-Lite**：12 个跨领域算法组件优化任务中 10 胜 1 平 1 负，平均奖励从 0.32 升至 0.43。

核心启示：长程实验优化不必依赖人工编排的 workflow，它本身是一种可训练的模型能力。当然，并非所有指标都领先（GPT-5.6 Sol 的 77.3%、GLM-5.3 的 AIME 48.1% 仍更高），更长任务与统一评测条件下的稳定泛化仍待验证。

> 本文参考自 [MLExpert: Towards a Generalist Agentic Model for Autonomous Machine Learning Engineering](http://arxiv.org/abs/2610.22344v1)