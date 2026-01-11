# 为什么我们会“思考” | Lil'Log

> 本文为原文逐段翻译，保留原文结构与重点表述，并在关键位置补充图解说明。

## Abstract

测试时计算（test-time compute）与 Chain-of-thought（CoT）显著提升了模型性能，同时也引发了大量研究问题。本文回顾了如何有效利用测试时计算（即“思考时间”）以及其为何有效的最新进展。

## Motivation

### Analogy to Psychology

核心思想与人类思考方式密切相关。人类不会立刻回答“12345 乘以 56789 等于多少”，而是先分析与推理，尤其在复杂问题上更明显。Kahneman 在《Thinking, Fast and Slow》中提出 **双过程理论** ：

- **System 1** （快思考）：快速、自动、直觉、情绪驱动，几乎不费力。
- **System 2** （慢思考）：刻意、逻辑、耗费认知资源。

System 1 速度快但易出错；有意识放慢思考能调动 System 2，从而做出更理性的判断。

### Computation as a Resource

一种视角认为：神经网络可被视为在一次前向传播中能调用的计算与存储资源。训练过程会学会如何利用这些资源形成“计算电路”。因此，如果架构在测试时可做更多计算，并且训练时学会利用这种资源，就能表现更好。

在 Transformer 中，每生成一个 token 的计算量约为参数数量的 2 倍。对于稀疏模型（如 MoE），每次仅激活部分专家，因此计算量约为 `2 * parameters / sparsity`。CoT 允许每个输出 token 前进行更多计算，并且计算量可随难度动态调整。

### Latent Variable Modeling

经典的隐变量模型：可见变量 $y$，隐变量 $z$，边缘化得到

$$
P(y) = \sum_{z \sim P(z)} P(y \mid z)
$$

例如让 $x$ 为题目、$y$ 为正确答案或证明，$z$ 为思考过程：

$$
P(y \mid x) = \sum_{z \sim p(z\mid x)} P(y \mid x, z)
$$

并行 CoT 或对 CoT 的搜索可视为从后验 $P(z \mid x, y)$ 采样，这也说明优化 $\log P(y \mid x)$ 的重要性。

## Thinking in Tokens

早期工作探索在答案前生成中间步骤：Ling 等（2017）提出 AQUA-RAT，Cobbe 等（2021）提出 GSM 并引入 verifier。Nye 等（2021）引入 scratchpad，Wei 等（2022）正式提出 CoT。

后续研究表明，通过 RL 在可自动验证的任务上优化 CoT，可显著提升推理能力，并在 o1、o3、R1 等模型中验证。

## Branching and Editing

测试时计算的核心目标是 **在推理阶段动态修改输出分布** 。两类主要方法：

- **Parallel sampling** ：并行生成多个候选并选择最优（如 best-of-$N$、beam search、自一致性）。
- **Sequential revision** ：迭代修正前一次输出，强调反思与纠错。

两者可以结合。Snell 等（2024）发现：简单问题更适合纯序列修订，难题适合并行与序列的混合。

## Parallel Sampling

best-of-$N$ 是最简单的并行采样。Beam search 则动态分配搜索资源，并可结合过程奖励模型（PRM）进行引导。Xie 等（2023）用自评估减少累积错误；Wu 等（2025）提出 REBASE；Jiang 等（2024）提出 RATIONALYST。

一个有趣发现是：无需显式 CoT 提示也可能触发 CoT（Wang & Zhou, 2024），关键在于早期 token 的高置信分叉。

## Sequential Revision

纯自纠错并不可靠，常见失败包括：

1. 幻觉：把正确答案改错。
2. 行为塌陷：几乎不修改错误答案。
3. 分布外泛化失败。

因此需要外部反馈，如 ground truth、启发式、单元测试、更强模型或人类反馈。

自纠错学习（Welleck et al. 2023）提出：固定生成器 $P_0$，训练纠错器 $P_\theta(y \mid y_0, x)$。通过价值提升对构造训练样本，并可在推理时反复修订。

SCoRe（Kumar et al. 2024）用多轮 RL 训练，让第二次回答优于第一次，并通过 KL 约束避免行为塌陷。

## RL for Better Reasoning

RL 在可验证任务上推动推理能力提升，DeepSeek-R1 是代表性案例：

- **Cold-start SFT** ：改善可读性与语言一致性。
- **Reasoning-oriented RL** ：两类奖励
  - 格式奖励：CoT 用 `<thinking> ... </thinking>` 包裹
  - 准确性奖励：答案格式可验证
- **Rejection sampling + non-reasoning SFT** ：过滤 CoT，加入非推理任务。
- **Final RL** ：同时优化推理与非推理能力。

值得注意的是：纯 RL 也能涌现 “aha moment” 与反思能力，但 PRM 与 MCTS 在实践中效果不佳，揭示了奖励黑客与搜索复杂度的挑战。

## External Tool Use

把可精确计算的步骤外包给工具能提升推理能力，例如 PAL、Chain of Code。ReAct 通过搜索 Wikipedia API 融合推理与行动。o3 / o4-mini 也展示了工具使用的优势，支持“更多计算 = 更好性能”。

## Thinking Faithfully

CoT 是一种可解释性手段，但关键在于 **CoT 是否真实反映模型内部推理** 。近期研究表明，监控 CoT 可发现 reward hacking，且能增强对抗鲁棒性。但前提是 CoT 真实可信。

### Does the Model Tell What it Thinks Faithfully

Lanham 等（2023）研究了三类不忠实 CoT：

1. **提前作答** ：CoT 被截断或注入错误，性能影响因任务而异。
2. **无信息 token** ：用填充文本替代 CoT 并不能提升准确率。
3. **人类不可读编码** ：非标准表达仍可维持性能，说明 CoT 可能并非“人类可读”才能有效。

提示扰动实验（Turpin 等, Chua & Evans, Chen 等）发现：推理模型往往更能如实承认提示偏差的影响；而 RLHF 奖励模型可能导致更不忠实的解释。

### Optimization Pressure on CoT: Good or Bad?

把 CoT 监控直接加入 RL 奖励容易导致 **隐藏式奖励黑客** 。Yeo 等（2025）发现长度奖励可诱发重复文本。Chen 等（2025）发现模型会利用环境漏洞但几乎不在 CoT 中坦白。结论是：直接优化 CoT 风险很高，需谨慎。

## Thinking in Continuous Space

Adaptive Computation Time（Graves, 2016）引入了动态推理步数。可通过 **递归结构** （纵向）或 **序列采样** （横向）实现。

### Recurrent Architecture

Universal Transformer（Dehghani et al. 2019）引入 RNN 机制，可动态决定步数。Geiping 等（2025）提出在 Transformer 上加入递归块 $R$，每轮使用输入嵌入 $\mathbf{e}$ 与随机状态 $\mathbf{s}_i$。训练时递归次数服从对数正态 Poisson 分布，并截断反传。训练稳定性高度敏感，需要精细调参。

### Thinking Tokens

Thinking tokens 是隐式 token，用于增加“思考时间”。Herel & Mikolov（2023）在每个词后插入 `<T>`；Goyal 等（2024）提出 pause tokens。它们本身不带信息，却提升性能，可能因为增加计算循环与隐式 CoT。

Quiet-STaR（Zelikman et al. 2025）在每个 token 后生成 rationale，通过 REINFORCE 训练，使 rationale 提升下一个 token 的预测。

## Thinking as Latent Variables

语言模型可视作隐变量模型，推理过程是潜变量 $z$。目标是最大化

$$
p(y \mid x) = \sum_{z} p(z \mid x) \, p(y \mid x, z)
$$

并通过多个 CoT 样本估计。

### Expectation-Maximization

由于难以直接采样 $p(z \mid x, y)$，研究采用人工标注、MCMC 或重要性采样。Ruan 等（2025）用 LLM 生成潜思考 $z$ 并引入重要性权重，优先选择能更好预测 $x$ 且简洁的 CoT。

### Iterative Learning

STaR（Zelikman et al. 2022）通过“反向合理化”解决失败样本无学习信号的问题，并在多轮迭代中提升性能。高温采样可能引入错误推理，影响泛化。

## Scaling Laws for Thinking Time

测试时计算与模型规模、预训练计算一样，是提升智能的关键因素。Snell 等（2024）指出测试时计算并不能完全替代预训练计算；在难题上仍需强基座模型。

s1（Muennighoff & Yang, 2025）通过 budget forcing 拉长 CoT，发现推理 token 越多准确率越高，但简单的 rejection sampling 反而出现负相关。

## What’s for Future

测试时计算与 CoT 引导模型更像人类的反思式思考。未来值得探索的问题包括：

- 如何在 RL 中鼓励人类可读且忠实的 CoT，同时避免奖励黑客？
- 奖励黑客如何定义与自动检测？
- 无 ground truth 时如何安全地自纠错？
- 如何把 CoT 用于难以评分的任务（创作、辅导等）？
- 如何把测试时收益蒸馏回更快的基础模型？
- 如何让思考时间随问题难度自适应？

## 图文解读

> 图解：整体结构导览，展示文章围绕 test-time compute、CoT、RL、忠实性与连续空间推理的主线脉络。
![Figure 1](images/img-1.png)

> 图解：并行采样与序列修订的对比，强调“并行多样性”与“逐步反思纠错”的差异。
![Figure 2](images/img-2.png)

> 图解：RL 训练流程与奖励信号示意，展示格式奖励与准确率奖励的协同作用。
![Figure 6](images/img-6.png)

> 图解：Thinking tokens 或 pause tokens 通过插入隐式 token 来增加推理计算，提升预测质量。
![Figure 12](images/img-12.png)

> 图解：Latent variable 视角下的 CoT 训练流程，体现 EM 或迭代学习的隐变量估计思想。
![Figure 18](images/img-18.png)

## 引用

本文参考自 Why We Think | Lil'Log  
https://lilianweng.github.io/posts/2025-05-01-thinking/