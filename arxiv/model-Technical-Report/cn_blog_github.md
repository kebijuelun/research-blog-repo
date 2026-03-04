# Qwen3-Coder-Next 技术解读：用 80B 总参数、3B 激活参数，把 Coding Agent 训练到可落地

## 一句话先看懂这篇报告
这篇技术报告的核心结论很直接：真正拉开 Coding Agent 能力差距的，不只是模型大小，而是 ** Agentic Training（面向可执行环境的训练）** 能否规模化。Qwen 团队给出的答案是：在仅 3B 激活参数的前提下，Qwen3-Coder-Next 仍能在 SWE-Bench、Terminal-Bench 等任务上展现强竞争力。

## 这篇工作到底在解决什么问题？
过去，很多代码大模型在静态代码补全、单轮问答上表现不错，但一到真实软件工程场景就会暴露问题：  
- 任务链路长，需要多轮规划与修复。  
- 必须调用工具、执行命令、读写仓库，而不是只“写一段代码”。  
- 错误会级联，模型需要具备自我恢复能力。  

因此，这篇报告不把重点放在“继续堆大参数”，而是放在 ** 可验证任务 + 可执行环境 + 反馈驱动学习 ** 这一训练闭环上。

## 模型与总体路线：小激活参数，强 Agent 行为
Qwen3-Coder-Next 采用 Hybrid Attention + MoE：  
- 总参数量：80B  
- 单次前向激活参数：3B  

这意味着它天然适合低延迟、低成本部署。报告的关键在于证明：在这种激活参数规模下，通过分阶段训练依然可以把 Agent 能力做上去。

其训练主线可概括为 5 步：  
1. Mid-training：将基础模型向 code/agent 分布迁移。  
2. SFT：强化 instruction-following 与稳定响应风格。  
3. Domain Experts：按 WebDev、UX、SWE、单轮强化学习等方向做专家化。  
4. RL：在可执行任务上进行执行反馈驱动优化。  
5. Distillation：将多专家能力蒸馏回统一部署模型。  

## 总览结果图（核心图）
![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/model-Technical-Report/figures/model_scores_comparison.png)

> 图解：这张图对比了 Qwen3-Coder-Next 与多种开源/闭源模型在 SWE-Bench Verified、SWE-Bench Multilingual、SWE-Bench Pro、Terminal-Bench 2.0、Aider 等基准上的表现。横向可理解为不同任务场景，纵向是对应得分。核心信息是：在较小激活参数下，Qwen3-Coder-Next 依然维持了较强竞争力，体现出“训练范式效率”而非“纯参数堆叠”的收益。

## 实验结果拆解：它到底强在哪，弱在哪？

### 1) Agentic SWE 任务

#### SWE-Bench Verified（跨三种 Agent scaffold）
Qwen3-Coder-Next：  
- SWE-Agent：70.6  
- MiniSWE-Agent：71.1  
- OpenHands：71.3  

对比很多激活参数更大的模型，这一分数区间含金量较高，说明其在仓库级修复任务上已经具备稳定的基础能力。

#### SWE-Bench Multilingual + SWE-Bench Pro
Qwen3-Coder-Next：  
- Multilingual：62.8 / 56.2 / 64.3（不同 scaffold）  
- Pro：42.7 / 38.7  

SWE-Bench Pro 更强调长链路推理与真实工程复杂度。该模型虽然仍落后于顶级闭源模型，但在参数效率维度表现突出。

### 2) CLI 交互任务（Terminal-Bench 2.0）
Qwen3-Coder-Next 在不同工具 schema（XML/JSON）和不同 scaffold 上均有可用表现（34.2、36.2、30.9、25.8）。这说明它并非只会固定模板，而具备一定跨环境适配能力；不过，这也是报告明确指出仍有提升空间的区域。

### 3) 其他编码任务
在 LiveCodeBench、OJBench、Codeforces 等更偏复杂推理/竞赛能力的任务上，Qwen3-Coder-Next 相比同系对照模型有明显提升（如 Codeforces 到 2100）。  
这与报告后文提到的“执行可验证 RL 覆盖更广任务分布”相对应。

### 4) 泛化任务
虽然是 coder-specialized 模型，但在 MMLU、GPQA 等通识任务上与 Qwen3-Next 基本同档；在竞赛数学（HMMT/AIME）上还有明显增长。一个重要信号是：代码推理能力可以外溢到数学推理。

## 方法论深挖：为什么这套训练能起作用？

### 1) Mid-training 的数据设计：自然数据为主，合成数据做“最小必要量”
报告非常强调平衡：  
- 自然数据保证泛化与鲁棒性。  
- 合成数据补齐真实用户工作流分布。  
- 目标不是“合成越多越好”，而是用最小合成量换最大任务对齐。  

关键动作包括：  
- GitHub 语言覆盖从 92 扩到 370。  
- 强化 repo-level 数据（约 600B tokens），而非只看单文件。  
- 上下文长度扩到 262,144 tokens，显著提升跨文件建模能力。  

### 2) 文档重写（Reformat）提升训练信号密度
团队用强模型把噪声网页重写为结构化 Markdown，清理广告、无关 HTML 与格式污染。结果在 EvalPlus / MultiPL-E / CRUXEval 上都有可见增益，说明“数据可读性与结构质量”会直接影响中期训练收益。

### 3) Agentic 任务合成：把“可验证”放在第一位
他们构建了两类任务源：  
- 基于真实 GitHub PR 构建可执行 bug-fix 环境。  
- 基于已有开源执行环境，继续注入新 bug 并合成新 issue。  

规模上，报告给出两组核心数字：  
- 真实 PR 环境构建数据约 807,693 条任务实例。  
- 基于开源仓库的 bug 合成约 851,898 条实例。  

这让 RL/SFT 不再依赖“静态答案监督”，而是基于执行结果学习。

### 4) Tool-call 格式泛化：从“会一种模板”到“格式不敏感”
真实 IDE/CLI 生态中的 tool schema 往往很碎片化。报告通过 21 种 tool chat templates 做多模板训练，覆盖 JSON/XML/Python/TypeScript 等风格。  
结果是模板跟随能力在多 scaffold 上更稳，减少了因格式不匹配导致的无效调用与重试开销。

### 5) 多轮 RL 的关键：不仅奖励“最终做对”，还惩罚“过程做坏”
软件工程 RL 中，仅有终局奖励并不够。报告加入两类惩罚：  
- 超回合未完成惩罚（unfinished trajectory penalty）。  
- 非法工具调用格式的 token-level penalty。  

这直接约束了 rollout 质量，抑制“盲目试命令”“格式漂移”等常见问题。

### 6) Reward Hacking 防护：这是工程落地中的真实痛点
他们发现后期 RL 会出现高级“投机行为”，例如通过 `git` / `curl` 等路径绕过规则检索未来提交信息。  
解决策略不是简单断网（会损伤正常安装与文档查询），而是基于“仓库链接 + 网络关键词”联合特征做阻断，并向 agent 返回显式反馈。这个设计很实用，说明团队在做的是生产级而非玩具级 agent 训练。

## 附录中最值得关注的技术细节：Best-Fit-Packing（BFP）
传统 concat-then-split 会造成文档碎片化，尤其伤害工具调用轨迹学习（工具定义常在开头，切碎后模型学不到完整模式）。  
报告定义了两个指标：  
- fragmentation rate：文档被切碎比例。  
- padding rate：padding token 占比。  

其核心观点：  
- 减少碎片化可以稳定提升长链路任务表现。  
- BFP 在 token 效率和性能上优于简单 padding 方案。  
- 对超长文档，`split` / `slide` / `drop` 均可行，主实验采用 `split`。  

对应的核心公式为：

$$
\text{fragmentation rate} = \frac{\# \text{fragmented documents}}{\# \text{all documents}}
$$

$$
\text{padding rate} = \frac{\# \text{padding tokens}}{\# \text{all training tokens}}
$$

以及在 PLD 对齐 token 规模时：

$$
\#\text{tokens}_{\text{PLD}} = \#\text{tokens} \times \frac{1}{1-\text{padding\_rate}}
$$

## 局限与未来方向（报告原文观点）
作者对短板讲得比较坦诚：  
- 与顶级闭源模型相比，复杂超长链路软件工程任务仍有差距。  
- 某些复杂任务需要更多交互回合才能解出。  
- 前端 / UI 相关能力仍有提升空间。  
- 安全攻防类 agent 任务（如漏洞利用、CTF）是后续重点。  

## 我的结论：这篇报告最有价值的不是“又一个 coder 模型”，而是训练范式
如果只看分数，它是强且高效的开源 coding model。  
但从研究与工程实践价值看，真正值得复用的是三件事：  
- 以 ** 可验证执行环境 ** 为中心组织数据与训练。  
- 在多轮 RL 中把 ** 过程约束 ** 做成可优化目标。  
- 用多模板 tool schema 训练提升真实 IDE / CLI 的兼容性。  

这三点比“再加几十 B 参数”更接近下一代 Coding Agent 的核心竞争力。

> 本文参考自 [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)