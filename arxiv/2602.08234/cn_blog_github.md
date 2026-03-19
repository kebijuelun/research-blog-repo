# SkillRL：让 LLM Agent 从“记流水账”进化到“长技能树”

## 背景：这篇论文在解决什么问题？

现在很多 LLM Agent（例如 Web 导航、检索问答、交互式任务）都有一个共性痛点： **每一轮任务都像“失忆重开”** 。  
虽然已有 memory-based 方法会保存历史轨迹，但原始轨迹通常又长又噪，包含大量试错与回退动作，最终导致两件事：

- 信息密度低，Token 开销大。
- 模型学不到可复用的高层策略，只是在“抄作业”。

这篇论文提出了 **SkillRL** 。其核心思想是： **把经验从“轨迹记忆”升级为“技能抽象”** ，并让技能库在 RL 训练中持续进化，而不是一次性静态写死。

---

## 方法总览：SkillRL 的三段式闭环

![SkillRL Framework](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.08234/asset/method.png)

> 图解：整体流程是“收集轨迹 → 蒸馏技能 → 冷启动 SFT 学会用技能 → RL 训练中递归更新技能库”。横向是训练阶段推进，纵向是知识形态从 raw trajectory 到 structured skill 的抽象提升。

SkillRL 包含三个关键模块：

1. **Experience-based Skill Distillation** ：将成功/失败轨迹都蒸馏为技能。
2. **Hierarchical Skill Library（SkillBank）** ：构建“通用技能 + 任务专用技能”的分层库。
3. **Recursive Skill Evolution** ：在 RL 过程中依据验证失败样本增量扩展技能。

---

## 为什么它比“存轨迹”更有效？

![Intro Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.08234/asset/fig1.png)

> 图解：左图对比了方法差异。传统方法（灰色虚线）偏向存原始轨迹；SkillRL 转向结构化技能抽取。右图给出 ALFWorld 曲线，SkillRL 收敛更快、最终成功率更高。

作者的关键判断是： **经验迁移的本质是抽象，不是复制** 。  
人类专家不会记住每一步点击，而是提炼“何时该用什么策略”的技能规则。SkillRL 的技能条目是结构化的，通常包含：

- skill name（技能名）
- principle（策略原则）
- when_to_apply（适用条件）

---

## 方法细节拆解（含核心公式）

### 1) 经验蒸馏：成功学“该做什么”，失败学“不要怎么做”

先用基础策略 $\pi_{\text{base}}$ 在环境 $\mathcal{E}$ 中采样轨迹，保留成功集 $\mathcal{T}^+$ 和失败集 $\mathcal{T}^-$。

- 成功轨迹提炼正向策略：

$$
s^+ = \mathcal{M}_T(\tau^+, d)
$$

- 失败轨迹提炼反事实教训：

$$
s^- = \mathcal{M}_T(\tau^-, d)
$$

这里教师模型 $\mathcal{M}_T$ 不只是摘要，还会定位失败点、错误原因、正确替代动作与可迁移原则。论文报告该蒸馏可带来约 **10–20×** 的 Token 压缩。

### 2) SkillBank：分层组织 + 检索增强决策

SkillBank 由两层组成：

- **General Skills** $\mathcal{S}_g$：跨任务通用策略。
- **Task-Specific Skills** $\mathcal{S}_k$：任务类型 $k$ 的专项启发。

推理时给定任务描述 $d$，总是注入 $\mathcal{S}_g$，并从 $\mathcal{S}_k$ 中做相似度检索：

$$
\mathcal{S}_{\text{ret}} = \text{TopK}\left(\{s \in \mathcal{S}_k : \text{sim}(e_d, e_s) > \delta\}, K\right)
$$

策略条件化为：

$$
a_t \sim \pi_\theta(a_t \mid o_{\le t}, d, \mathcal{S}_g, \mathcal{S}_{\text{ret}})
$$

### 3) 递归进化：技能库与策略一起长大

- 先做 **Cold-start SFT** ，让模型先学会“如何使用技能”，避免直接 RL 时不会调用技能：

$$
\theta_{\text{sft}} = \arg\min_\theta \mathcal{L}_{\text{CE}}(\mathcal{D}_{\text{SFT}}; \theta)
$$

- 再做 GRPO 训练，采用组内归一化优势：

$$
A_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^{G})}{\text{std}(\{R_j\}_{j=1}^{G})}
$$

目标函数（PPO-style clipping + KL 正则）：

$$
\mathcal{J}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\left(\rho_i A_i,\ \text{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\right)-\beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right]
$$

每轮验证后，若某类任务成功率低于阈值，就分析失败轨迹并生成新技能加入技能库，从而形成“失败驱动增量学习”。

---

## 实验设置与基线

- 环境：ALFWorld、WebShop、7 个 Search-augmented QA 数据集。
- Base Model：Qwen2.5-7B-Instruct。
- Teacher：OpenAI o3。
- RL：GRPO。
- 关键超参：$K=6$（检索数）、失败更新阈值 $\delta=0.4$、RL 学习率 $1\times10^{-6}$。

---

## 主结果：不是小幅提升，而是一档跃迁

### ALFWorld + WebShop

| 方法 | ALFWorld(All) | WebShop Succ. |
|---|---:|---:|
| GRPO | 77.6 | 66.1 |
| Mem0+GRPO | 54.7 | 37.5 |
| SimpleMem+GRPO | 62.5 | 46.9 |
| SkillRL | 89.9 | 72.7 |

结论非常直接：

- 相比 GRPO，ALFWorld 绝对提升 **12.3%** 。
- 相比强 memory+RL 组合（Mem0+GRPO），ALFWorld 领先约 **35.2%** 。
- 甚至超过 GPT-4o / Gemini-2.5-Pro 在该任务上的分数。

### Search-Augmented QA（7 数据集平均）

| 方法 | Avg. |
|---|---:|
| Search-R1 | 38.5 |
| EvolveR | 43.1 |
| SkillRL | 47.1 |

在多跳任务（尤其 Bamboogle）上提升明显，说明分层技能对复杂推理链有效。

---

## 消融实验：每个模块都“真有用”

| 变体 | ALFWorld | WebShop |
|---|---:|---:|
| SkillRL | 89.9 | 72.7 |
| w/o Hierarchical Structure | 76.8 | 61.4 |
| w/o Skill Library（Raw Trajectories） | 61.7 | 50.2 |
| w/o Cold-start SFT | 65.2 | 46.5 |
| w/o Dynamic Evolution | 84.4 | 70.3 |

关键观察：

- 去掉分层结构后，性能显著下降。
- 用 raw trajectory 替代 skill library 时下降最大，验证了“抽象优于记忆”。
- 不做冷启动 SFT 会明显崩溃。
- 动态进化还能继续抬升上限。

---

## 训练动态与效率证据

![Skill Growth](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.08234/asset/skill_mixed_chart.png)

> 图解：技能库从 55 增长到约 100，增长主力是 task-specific skills，说明策略在训练中确实学到更多任务化“招式”。

![Prompt Length](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.08234/asset/prompt_length_comparison.png)

> 图解：与 raw memory 检索相比，SkillRL 的 prompt token 更短且更稳定，平均约减少 10.3% 上下文开销。

![Success Curve](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.08234/asset/success_rate_curve.png)

> 图解：带递归进化的版本更快到达高成功率（约 60 steps 超过 80%），且最终上限更高。

![Case Study](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.08234/asset/case.png)

> 图解：案例显示 agent 会组合调用 General Skills 与 Task-Specific Skills，而非机械复读历史轨迹，体现“策略级检索”而非“文本级回忆”。

---

## 附录中值得关注的复现信息

- **训练预算** ：8×H100 80GB，总墙钟约 30 小时/实验。
- **冷启动 SFT 数据规模** ：ALFWorld 7500、WebShop 2400。
- **验证驱动更新频率** ：每 5 steps 可触发一次技能进化。
- **提示词工程** ：附录给出完整模板，覆盖在线执行、失败技能发现、冷启动轨迹合成三类模板。
- **技能可解释性** ：附录展示了 ALFWorld/WebShop 的技能样例和错误 taxonomy（失败原因 → 规避策略）。

---

## 结论

SkillRL 的价值不只在“分数更高”，还在于它把 Agent 学习范式从 **Trajectory Memory** 推向 **Skill Abstraction + Co-evolution** ：

- 用结构化技能降低噪声与 Token 成本。
- 用冷启动让模型真正学会使用技能。
- 用失败驱动递归更新，让系统持续适应新场景。

如果你在做 Agent 训练，这篇文章最可迁移的工程启发是：  
**别把 memory 当数据库，要把 memory 变成可执行策略接口。**

> 本文参考自 [SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning](https://arxiv.org/abs/2602.08234)