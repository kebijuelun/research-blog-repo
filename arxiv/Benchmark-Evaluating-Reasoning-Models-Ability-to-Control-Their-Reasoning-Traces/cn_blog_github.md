# 论文精读：CoT-Control 基准揭示——推理模型其实很难“有意识地控制自己的思维链”

## 一句话先看结论
这篇论文的核心发现很“反直觉”： **现在的大多数推理模型，在控制最终输出方面很强，但在控制 Chain-of-Thought（CoT）方面反而很弱** 。  
从安全视角看，这反而是阶段性的好消息——因为 CoT 越难被模型“伪装”，CoT 监控（monitoring）就越可靠。

---

## 背景：为什么要研究 CoT controllability？
过去一段时间，很多安全工作把 CoT 视为观察模型意图的重要窗口：模型是否在“想坏事”，常常会在推理链里留下痕迹。  
但这里有一个隐含前提：模型不会轻易把这些痕迹藏起来。

这篇论文提出的关键风险点是：如果模型具备较高的 **CoT controllability** （可控性），它可能在被监控时伪装成“很安全”，部署后再绕过监控。

---

## 论文做了什么：提出 CoT-Control 评测套件
作者构建了一个系统化评测集（CoT-Control / CoT Controllability Evaluation Suite），总计 14,076 个测试实例，来自：

- QA：MMLU-Pro、GPQA、HLE
- Agent 任务：SWE-Bench Verified、BFCL

核心设计是：在做题的同时，给模型额外的 CoT 约束，观察其能否“边推理边遵守”。

三大约束类型（每类 Easy/Medium/Hard）：

1. 信息抑制（Information Suppression）
2. 信息添加（Information Addition）
3. 风格改写（Stylistic Modification）

---

## 方法亮点：不是只看“是否遵守”，还看“是否隐蔽”
作者的指标有三层：

- **Controllability Score** ：是否成功遵守 CoT 约束
- **Meta Discussion Rate** ：是否在 CoT 里暴露“我正在遵守约束”
- **无 Meta 的 Controllability** ：更接近真实“隐蔽绕监控”能力

也就是说，模型如果一边遵守约束，一边说“我现在要避免这个词”，其实很容易被监控器抓到，并不属于真正危险的“隐蔽控制”。

---

## 主结果：当前推理模型 CoT 可控性普遍偏低
![主结果总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/cot_controllability_final_by_model.png)

> 图解：横轴是不同模型，纵轴是 CoT controllability（越低越安全）。多数模型低于 10%，说明“控制 CoT”远难于“控制输出文本”。

论文中一个非常典型的对比是：同一个模型在输出空间可控，但在 CoT 空间不可控。  
例如摘要给出的数字：Claude Sonnet 4.5 在 CoT 控制约 2.7%，但控制最终输出可达约 61.9%。

---

## CoT vs 输出控制：差距非常大
![三类 baseline 对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/cot_controllability_3_baselines_vs_reasoning_space.png)

> 图解：黄色（CoT 控制）普遍显著低于其他基线（控制输出、控制外显 reasoning）。说明“让模型改最终答案措辞”不等于“让模型稳定改内部推理痕迹”。

这点对安全评估很关键：只测 instruction-following 远远不够，必须专门测 CoT 空间。

---

## 影响因子 1：模型越大，CoT controllability 趋势上升
![同家族参数规模对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/gpt_oss_vs_qwen3_reasoning_space.png)

> 图解：在同家族 open-weight 模型中，参数规模增大通常伴随 CoT controllability 上升（风险上升）。

这意味着“单纯放大参数”可能会提升规避监控的潜力。

---

## 影响因子 2：推理 effort 越高，CoT controllability 反而下降
![推理 effort 对 controllability 的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/reasoning_efforts_compliance.png)

> 图解：从 low 到 high/extra high，CoT controllability 下降。可理解为：推理越长、越展开，越难全程维持“受控伪装”。

这与“加大 test-time compute 提升能力”并不冲突：能力提升的同时，监控可见面也在变大。

---

## 影响因子 3：RL 推理训练会进一步降低 CoT controllability
![RL 训练阶段变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/rl_stages_compliance.png)

> 图解：多数开源模型在 reasoning 训练后 CoT controllability 显著下降。说明结果导向 RL 可能让模型更关注“做对题”，而不是“精细操控推理文本形式”。

作者还观察到：不同训练域影响不同，Math/Code 域下降往往更快。

---

## 影响因子 4：任务更难、推理更长，CoT 更难控制
![难度与 CoT necessity 分析](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/cot_necessity_compliance.png)

> 图解：当题目更依赖真实推理时，CoT controllability 更低。换句话说，越“必须认真思考”的题，越难伪装推理轨迹。

并且论文给出逻辑回归分析：推理长度上升与 controllability 下降显著相关。

$$
\mathrm{SE}=\sqrt{\frac{p(1-p)}{n}}
$$

> 图解：这是文中用于比例指标标准误估计的核心公式（独立伯努利近似）。

---

## 压力测试：能不能把 CoT controllability “逼”出来？

### 1) 情境感知（被告知“你正在被监控/评测”）
![situational awareness 结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/awareness_compliance_comparison.png)

> 图解：部分模型在 awareness 提示下 controllability 小幅上升（最高几个百分点）。这提示“评测感知”确实可能改变行为，但幅度有限。

### 2) 自动红队 + GEPA 式 prompt 优化
![GEPA 迁移测试](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/new_three_mode_pressure_comparison.png)

> 图解：训练阶段看似有增益，但迁移到未见约束类型时效果较差，说明更像“过拟合某几条约束”，而不是通用 CoT 控制能力。

### 3) 重复尝试（pass@k）
![pass@k 结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Benchmark-Evaluating-Reasoning-Models-Ability-to-Control-Their-Reasoning-Traces/figures/pass_at_k_compliance.png)

> 图解：k 增大时成功率上升，说明“多次重试”会抬高风险上界，部署时要重视首错拦截和重试策略治理。

---

## 论文的安全含义：谨慎乐观，但不能放松
作者的态度可以概括为：

- 当前模型总体 CoT controllability 低，短期看是好消息；
- 但该能力与模型规模、情境提示等因素有关，不是“永久低风险”；
- 前沿实验室应把 CoT controllability 纳入常规 pre-deployment 评测，并与 monitorability 一起评估。

理想安全区间是： **高 monitorability + 低 controllability** 。

---

## 复现与评测细节中值得注意的点
- 统计诊断里给了 EPV、LR test、McFadden pseudo-$R^2$，样本量整体充足。
- 评测不只看 QA，还包含 agent/coding 场景，覆盖面较实用。
- 对个别难以自动判分的任务（如 ignore-question）使用 LLM judge，并做人类一致性校验。

---

> 本文参考自 [Benchmark: Evaluating Reasoning Models’ Ability to Control Their Reasoning Traces](https://arxiv.org/abs/2603.05706)