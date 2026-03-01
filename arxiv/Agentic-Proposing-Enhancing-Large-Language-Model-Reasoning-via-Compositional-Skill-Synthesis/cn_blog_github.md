# Agentic Proposing：把“出题”变成可训练智能体，重塑 LLM 推理数据供给

## 先说结论：这篇论文到底做了什么？

这篇工作不再把数据合成看成“一次性写题目”的文本生成任务，而是把它重构为一个可交互、可反思、可验证的决策过程：由一个专门的 Proposer Agent 动态组合“原子技能”，在 **Draft → Check → Refine → Finalize** 闭环里生成高难、可解、可验的问题。

作者的核心主张很直接：当前推理模型的瓶颈不主要是参数规模，而是 **高密度高质量训练信号** 。  
在这个前提下，他们给出一个很强的实证结果：30B Solver 仅用 11k 合成轨迹，AIME 2025 达到 91.6%。

---

## 背景问题：为什么“合成数据”这么难？

现有方法通常卡在一个两难：

- 保守模板 + 强结构约束：题目更稳定，但难度上不去。
- 放开约束追求高难：容易出现逻辑矛盾、不可解或不可验证样本。

论文认为根因是：问题生成本质上是 **组合式逻辑工程** ，不是单次采样。  
因此需要一个能“边想边做边纠错”的 agent，而不是静态 prompt。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agentic-Proposing-Enhancing-Large-Language-Model-Reasoning-via-Compositional-Skill-Synthesis/figures/head_cropped.png)

> 图解：左图是传统 open-loop 单轮生成，缺少中途校验；右图是闭环 agentic proposing，会在生成过程中调用工具与验证环节，保证题目可解性与难度标定同时成立。

---

## 方法总览：Agentic Proposing 三阶段

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agentic-Proposing-Enhancing-Large-Language-Model-Reasoning-via-Compositional-Skill-Synthesis/figures/proposer_pipeline_cropped.png)

> 图解：整条流水线分三段：先建技能库，再做 Agentic SFT 学会反思与工具调用，最后用 MGPO 强化学习优化“终局质量 + 中间行为”。

### 1) Stage 1：Skill Acquisition（技能获取与格式化）

作者先从混合语料中抽取候选技能，经过阈值过滤保留高质量技能，形成自治技能库 $\mathcal{K}_{self}$。  
技能不是一句话描述，而是结构化元组：

- $k=\langle \iota,\mu,\delta,\tau\rangle$
- $\iota$：意图（为什么用）
- $\mu$：方法（怎么做）
- $\delta$：难度影响
- $\tau$：工具提示（何时调用 exec/edit 等）

核心过滤分布：

$$
p_{skill}(k)=\frac{\mathbb{I}[r(k)\ge \tau_r]\cdot \pi_{teacher}(k\mid \mathcal{D}_{corpus})}{\sum_{k'}\mathbb{I}[r(k')\ge \tau_r]\cdot \pi_{teacher}(k'\mid \mathcal{D}_{corpus})}
$$

### 2) Stage 2：Agentic SFT（行为克隆）

不是只学“最后输出题目”，而是学习完整轨迹：反思、调用工具、裁剪技能、再提交。  
只保留通过 verifier 的轨迹进入 SFT 数据，目标是得到可用于 RL 初始化的 $\pi_{ref}$。

### 3) Stage 3：MGPO（多粒度策略优化）

这是论文算法亮点。作者把奖励拆成两层：

- 终局奖励：题目是否有效、是否有难度增益。
- 过程奖励：中间行为是否正确（如反思合理、工具调用有效）。

终局奖励定义：

$$
r_T = V(q)\cdot\left(R_{base}+\mathbb{I}[\rho(q)>0]\cdot \lambda(1-\rho(q))\right)
$$

其中：

- $V(q)\in\{0,1\}$：是否可解且逻辑一致
- $\rho(q)$：外部 prober 估计的 Pass@$k$
- 只有可解题才给难度加分，避免“不可解高难伪样本”

---

## 关键建模：为什么要用 POMDP？

作者把合成任务建成 POMDP，因为“题目是否真的可解”是隐变量，不能仅靠对话表面判断。  
观测定义为：

$$
o_t=\langle \mathcal{K}_t,h_t,\sigma_t \rangle
$$

- $\mathcal{K}_t$：当前激活技能子集
- $h_t$：历史上下文与工具输出
- $\sigma_t$：认知阶段标记（draft/check/refine）

动作空间三类：

- $\tau^{think}$：内部反思
- $\tau^{exec},\tau^{edit}$：工具执行与技能裁剪
- $\tau^{submit}$：提交最终题目

这一设计的价值在于：模型可在中途主动发现冲突并回退修正，而不是把错误带到终局。

---

## MGPO 再拆一层：多粒度 Credit Assignment

标准 GRPO 只看轨迹级稀疏奖励，长链条合成任务会很难学。  
MGPO 额外加了阶段级优势：

- 轨迹级：$A_E(\tau_i)$
- 阶段级：$A_S(a_t)$

融合后：

$$
A^{fused}_{i,t}=A_E(\tau_i)+\omega\cdot A_S(a_t)
$$

并通过非对称门控稳定训练（负权重衰减更强），最终做加权似然优化。  
直观理解：不仅奖励“最终题好”，还奖励“中间每一步做对”。

---

## 实验设置与评测协议

- 预算严格固定：10k–11k 轨迹
- 任务覆盖：竞赛数学、代码、科学推理
- 评测口径：
  - 数学：Mean@64
  - LiveCodeBench：Best-of-5
  - 通识科学：Mean@1
- 下游 solver 训练统一用 GRPO，避免“算法差异”干扰数据质量比较。

---

## 主结果：数据质量优势非常明显

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agentic-Proposing-Enhancing-Large-Language-Model-Reasoning-via-Compositional-Skill-Synthesis/figures/overall_cropped.png)

> 图解：在固定 10k 预算下，Agentic-Proposer-4B 生成数据训练出的 4B solver，整体达到 38.3%，比 zero-shot 基线 34.2% 提升 +4.1。

### 4B 数学任务（固定 10k 轨迹）

- AIME24：53.6（+3.8）
- AIME25：51.2（+4.5）
- HMMT：36.5（+5.5）
- AMO-Bench：11.8（+2.5）
- Overall：38.3（+4.1）

很多经典数据合成基线（如 MetaMath、WizardMath）在该设定下甚至低于 zero-shot，说明“样本多”不等于“有效信号密”。

### 30B 扩展结果（11k 混合轨迹）

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agentic-Proposing-Enhancing-Large-Language-Model-Reasoning-via-Compositional-Skill-Synthesis/figures/acc_cropped.png)

> 图解：横向比较模型规模与 AIME25 精度，作者方法在 30B 上达到 91.6%，在开源阵营中表现非常强，同时逼近/对标头部闭源模型。

- AIME25：91.6（+6.6 vs zero-shot）
- HMMT：77.6（+6.2）
- LCB v5：73.4（+5.3）
- LCB v6：71.2（+5.2）
- Overall：81.5（+5.9）

### 跨域泛化（4B）

在 OlympicArena / MMLU-Redux / MMLU-Pro / GPQA / SuperGPQA 上总体 +5.3，  
其中 SuperGPQA +7.3、GPQA +6.3，说明该方法不是“只会刷数学榜”。

---

## 消融实验：每个模块都不是装饰

### 1) Proposer 专业化 vs 纯提示工程

- GPT-5.2 raw：32.8
- + Skill Library：34.6
- + Agentic Workflow：36.4
- Agentic-Proposer-4B：38.3

结论：结构化工作流对大模型也有帮助，但专门训练过的小模型 Proposer 仍更强。

### 2) Agentic 闭环的增益

- One-shot：31.5
- + tool-use：33.8
- + reflection：33.4
- Full pipeline：38.3（+6.8）

### 3) MGPO 对比标准 GRPO

- GRPO：31.8
- MGPO 去掉阶段优势：35.1
- Full MGPO：38.3（+6.5）

### 4) 动态技能裁剪（ $\tau^{edit}$ ）

- 无裁剪：有效率 68.7%
- 有裁剪：82.3%

作者报告约 14.5% 轨迹会触发主动裁剪，显著降低逻辑冲突。

---

## 我认为最有价值的三个启发

### 1) 把“出题”从生成问题变成决策问题

POMDP + 工具 + 反思，让合成过程可控、可纠错、可学习。

### 2) 训练目标从“结果正确”升级为“过程可归因”

MGPO 的多粒度优势分配，解决了长链条稀疏奖励问题。

### 3) 数据工程范式升级：从静态语料走向自进化生态

技能库、课程采样、验证委员会、二次审计，这些机制共同组成可扩展闭环。

---

## 复现时需要重点关注的工程细节

- 先把 verifier 做扎实，再追求难度；无效样本会污染 RL 信号。
- 下游 solver 训练只喂 `{question, answer}`，不要混入 proposer 内部轨迹。
- 课程分布用 $p(c)\propto 1/(m_c+\epsilon)$，避免模型躲进“低难安全区”。
- MGPO 超参里 $\omega$ 很关键，论文给出的最佳点在 0.5 附近。
- 工具调用必须进入训练分布，不然部署时反思-执行链条会断裂。

---

## 总结

这篇论文真正推进的，不只是一个新数据集或新 Prompt，而是一个 **可训练的“数据生成智能体”框架** 。  
它把高难推理数据生产，从“模板扩写”升级为“技能组合 + 闭环验证 + 强化学习”的系统工程。  
从结果看，这条路线已经证明：小规模高质量合成信号，确实可以替代大量低密度数据堆砌。

> 本文参考自 [Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis](https://arxiv.org/abs/2602.03279)