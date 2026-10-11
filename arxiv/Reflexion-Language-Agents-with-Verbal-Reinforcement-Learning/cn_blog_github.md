# Reflexion：让大模型用"写检讨"的方式自我进化

大语言模型（LLM）已经可以作为 Agent 与环境交互——玩游戏、写代码、调 API——但它们有一个致命短板： **不会从失败中学习** 。传统强化学习（RL）需要海量样本和昂贵的梯度更新，对动辄千亿参数的大模型几乎不可行。这篇 NeurIPS 2023 的论文提出了 **Reflexion** ：不动模型权重，而是让模型在失败后用自己的语言"写一份反思"，存进记忆里，下一次尝试时带着这份"教训"重来。效果相当惊人：在 HumanEval 代码基准上，Reflexion 把 pass@1 准确率推到了 **91%** ，直接超过了 GPT-4 单跑的 80%；在 AlfWorld 决策任务上比强基线绝对提升 **22%** ，在 HotPotQA 推理任务上提升 **20%** 。

## 一、提出问题：LLM Agent 为什么学不会"吃一堑长一智"

先看看当时 LLM Agent 的处境。ReAct、SayCan、Toolformer、HuggingGPT 等工作已经证明，LLM 可以通过生成文本和"动作"来与外部环境交互，完成自主决策。但这些方法有一个共同的局限： **教学手段只有 in-context examples** 。

原因很简单：模型太大，传统的基于梯度下降的强化学习在计算和时间成本上都不可承受。于是 Agent 在某条轨迹上失败之后，下次遇到同样的局面还是会犯同样的错——它没有"记忆"，更没有"教训"。

人类是怎么做的？我们写代码报错时，会回头想想"刚才哪里写错了"，然后带着这个认知再试一次。这种"反思—修正—重试"的循环，本质上就是一种 few-shot 的学习过程。Reflexion 要解决的，就是 **如何不更新权重、纯用语言信号，让 LLM Agent 拥有这种从试错中快速学习的能力** 。

难点有两个：一是 credit assignment（信用分配）问题——长轨迹失败后，模型要能说清到底是哪一步走错了；二是要生成 **可执行的改进建议** ，而不是泛泛的"下次努力"。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Reflexion-Language-Agents-with-Verbal-Reinforcement-Learning/figures/reflexion_tasks.png)

> 图解：Reflexion 在三大类任务上的工作方式。左（决策）：AlfWorld 家务任务中，Agent 第一次失败后反思"应该先找台灯再找杯子"，第二次径直完成任务；中（推理）：HotPotQA 问答中，Agent 反思"我搞错了搜索顺序"，改正后答对；右（编程）：代码未通过自生成单测后，Agent 反思实现逻辑并修复。三个场景共用同一套"试错—反思—记忆—重试"范式。

## 二、分析问题：与前人工作的差异在哪

在 Reflexion 之前，"让模型自我改进"并非全新概念，但各有短板：

- **Self-Refine** 等自我优化框架只做单次生成的打磨，不涉及多步决策，也没有持久记忆；
- **Beam search / 随机搜索类方法** （如 Xie et al.）借助自我评估做更高效的决策搜索，但经验不沉淀；
- 编程方向的 **AlphaCode、Self-Debugging、CodeRL** 依赖隐藏的 ground truth 测试用例，按规则这会使 pass@1 资格失效； **CodeT** 虽不访问隐藏测试，却没有自我学习环节。

下表概括了 Reflexion 与代表性工作的能力对比（✓/✗）：

| 决策与推理方向 | 自我优化 | 无隐藏约束 | 多步决策 | 二值奖励可用 | 有记忆 |
| --- | --- | --- | --- | --- | --- |
| Self-Refine | ✓ | ✗ | ✗ | ✗ | ✗ |
| Beam search | ✓ | ✓ | ✓ | ✓ | ✗ |
| **Reflexion** | ✓ | ✓ | ✓ | ✓ | ✓ |

| 编程方向 | 测试执行 | 调试执行 | 自生成测试 | 多语言 | 自我反思 |
| --- | --- | --- | --- | --- | --- |
| AlphaCode | ✓ | ✗ | ✗ | ✓ | ✗ |
| CodeT | ✓ | ✗ | ✓ | ✗ | ✗ |
| Self-debugging | ✓ | ✓ | ✗ | ✗ | ✗ |
| CodeRL | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Reflexion** | ✓ | ✓ | ✓ | ✓ | ✓ |

可以看到，Reflexion 的独特定位是： **把稀疏的环境奖励"放大"为自然语言的经验总结，并以持久记忆的形式跨 trial 沉淀** 。这正是此前所有方法缺失的一环。相比传统 RL 的 policy/value-based 学习，它还有四个优势：轻量（无需微调 LLM）、反馈更细腻（可指明具体动作怎么改）、记忆可读可解释、对未来 trial 给出显式行动提示。代价则是依赖 LLM 自身的自我评估能力，且没有形式化的成功保证。

明确了差异点之后，下一个问题是：这套"语言强化"具体是怎么搭起来的？

## 三、解决问题：Reflexion 框架详解

### 3.1 总体框架：三个模型 + 两类记忆

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Reflexion-Language-Agents-with-Verbal-Reinforcement-Learning/figures/reflexion_rl.png)

> 图解：Reflexion 的架构示意。Actor（ $M_a$ ）根据状态和记忆生成轨迹 $\tau_t$ ；Evaluator（ $M_e$ ）对轨迹打分得到标量奖励 $r_t$ ；Self-Reflection（ $M_{sr}$ ）把 $\{\tau_t, r_t\}$ 转化为自然语言反思 $sr_t$ ，追加进记忆 mem ；Actor 下一轮带着 mem 重新决策，直到 Evaluator 判定通过或达到最大 trial 数。

Reflexion 由三个模块化的模型组成：

- **Actor（ $M_a$ ）** ：核心策略模型，基于 LLM，根据环境观测生成文本和动作。类比传统 policy-based RL，它从当前策略 $\pi_{\theta}$ 中采样动作 $a_t$ 并接收观测 $o_t$ 。论文探索了 Chain of Thought（CoT）和 ReAct 两种 Actor 实现。
- **Evaluator（ $M_e$ ）** ：对 Actor 产出的轨迹打奖励分。由于语义空间上定义好的 reward 函数很难，论文按任务定制：推理任务用 exact match（EM，精确匹配）判分；决策任务用手写启发式函数；也实验过用另一个 LLM 实例当 Evaluator。
- **Self-Reflection（ $M_{sr}$ ）** ：同样由 LLM 担任，是整套框架的灵魂。它接收稀疏奖励（比如 success/fail）、当前轨迹和持久记忆，输出具体、细腻的口头反馈。

笔者认为这里最聪明的设计在于 **把"梯度"语义化** 了。传统 RL 中，标量奖励经过反向传播去更新权重，credit assignment 是个老大难问题；Reflexion 则让 LLM 直接"读"轨迹，用语言指出"你第 $i$ 步的 $a_i$ 导致后面全错了，下次应该选 $a_i'$ "——这比标量奖励携带的信息量大几个数量级，而且天然可解释。

### 3.2 记忆机制：短期记忆 + 长期记忆

Reflexion 借用了人类的记忆结构：

- **短期记忆** = 当前 trial 的轨迹历史，提供细粒度的即时上下文；
- **长期记忆** = Self-Reflection 模型输出的经验总结，跨 trial 持久保存。

Actor 决策时同时条件化在这两类记忆上：既记得"刚才发生了什么"，也记得"之前几次失败的教训是什么"。这正是 Reflexion Agent 区别于其他 LLM 决策方法的关键优势。

### 3.3 迭代优化过程

整个 Reflexion 流程可以形式化为一个迭代优化循环：

1. 第一轮：Actor 与环境交互产生轨迹 $\tau_0$ ，Evaluator 打分 $r_0 = M_e(\tau_0)$ ；
2. Self-Reflection 模型分析 $\{\tau_0, r_0\}$ ，把标量奖励"放大"为语言形式的经验总结 $sr_0$ ，存入 mem ；
3. 后续每轮 $t$ ：Actor 带着 mem 重新生成 $\tau_t$ → 评估得 $r_t$ → 生成 $sr_t$ 追加进 mem ；
4. 循环直到 Evaluator 判定 $\tau_t$ 正确，或达到最大 trial 数。

形式上，策略参数可以写作 $\theta = \{M_a, mem\}$ ——注意这里被"优化"的不是神经网络权重，而是 **记忆内容本身** 。实践中为了适配 LLM 的上下文长度限制，mem 通常截断为最近 $\Omega$ 条经验（一般取 1~3 条）。

反思信号从哪里来？论文给了三种实现： **环境自带的二值反馈** 、 **针对常见失败模式的预定义启发式** 、以及 **自我评估** （决策任务里用 LLM 做二分类，编程任务里用模型自己写的单元测试）。无论哪种，最终都会被放大成自然语言经验存入长期记忆。

框架搭好了，能不能打，要看实验。作者在三类任务上做了验证：顺序决策、推理、编程。

## 四、实验验证

### 4.1 顺序决策：AlfWorld

AlfWorld 是一套基于 TextWorld 的文本交互环境，Agent 要在虚拟家庭中完成多步任务（比如在抽屉里找锅铲、把番茄放冰箱冰镇）。实验沿用 ReAct 的设置，跑 134 个环境、六类任务，LLM 用 GPT-3。

由于环境只给"任务是否完成"的信号，作者设计了两种自我评估手段：一是用 LLM 做自然语言二分类；二是一条简单启发式—— **同一动作得到相同回应超过 3 轮，或当前环境动作数超过 30** ，就判定需要反思重来。基线组触发该条件时直接重置环境、不带任何教训重开；Reflexion 组则先生成反思、更新记忆再重开。记忆截断为最近 3 条反思。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Reflexion-Language-Agents-with-Verbal-Reinforcement-Learning/figures/alfworld_success.png)

> 图解：AlfWorld 134 个任务的累计完成比例随 trial 数的变化。横轴为 trial 次数，纵轴为已解决任务的累计比例。ReAct + Reflexion（配合启发式或 GPT 分类自评估）在 12 轮内持续爬升，最终完成 130/134 个任务；而纯 ReAct 在第 6~7 轮后完全停滞，绝对差距约 22%。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Reflexion-Language-Agents-with-Verbal-Reinforcement-Learning/figures/alfworld_failure.png)

> 图解：按失败原因对 AlfWorld 轨迹的分类统计。纯 ReAct 的轨迹中，"幻觉"（以为拿到了物品其实没有）和"低效规划"占比很高，幻觉率收敛在 22% 左右无法恢复；Reflexion 几乎消除了这类错误。

错误分析很有代表性：基线最常见的失败是 **物品幻觉** ——Agent 以为自己手里拿着东西，其实并没有，然后在一条长轨迹里越走越错，无法回溯。Reflexion 通过把长失败轨迹蒸馏成"自我提示"几乎根除了这个问题。长期记忆在两种情形下特别有用：一是长轨迹早期的错误可以被快速定位并给出新计划；二是当需要翻找的容器太多时，Agent 能利用跨 trial 的经验系统性地搜完整个房间。

### 4.2 推理：HotPotQA

HotPotQA 是包含 11.3 万问答对的维基百科多跳推理数据集。实验分两种设定：

- **纯推理能力测试** ：Reflexion + CoT，其中 CoT (GT) 变体直接提供 ground truth 上下文 $C_{gt}$ ，隔离出"只考推理"的场景，即 $Q, C_{gt} \rightarrow A$ ；
- **综合问答能力测试** ：Reflexion + ReAct，Agent 自己调 Wikipedia API 检索上下文再作答。

trial 之间用 exact match 做二值判分，记忆大小为 3 条经验；失败任务允许重试，直到连续 3 次失败为止。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Reflexion-Language-Agents-with-Verbal-Reinforcement-Learning/figures/hotpotqa_success.png)

> 图解：100 道 HotPotQA 问题上，Reflexion ReAct 与 Reflexion CoT 随 trial 数的准确率变化。两种 Reflexion 变体都随轮次持续提升，且显著超过所有基线；纯 ReAct 和纯 CoT 在多轮中毫无改进——temperature 0.7 下，第一轮做错的题后面也做不对。

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Reflexion-Language-Agents-with-Verbal-Reinforcement-Learning/figures/hotpotqa_cot_gt.png)

> 图解：CoT (GT) 设定下（直接给出 ground truth 上下文、只考推理）的结果。即使有标准上下文，基线仍有 39% 的题答不对；加上 Reflexion 后，Agent 在不看标准答案的情况下自我纠错，准确率再提升 14%。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Reflexion-Language-Agents-with-Verbal-Reinforcement-Learning/figures/hotpotqa_ablation.png)

> 图解：消融实验——以 CoT (GT) 为基线，依次加入 episodic memory（EPM，即把上一轮轨迹原文塞进上下文）和完整的 Reflexion 反思步骤。结果显示：仅加轨迹记忆有一定提升，而"用语言写成的第一人称反思"在记忆之上再带来约 8% 的绝对提升，说明 **关键不是"记得"，而是"想明白了"** 。

这个消融实验支撑了论文的一个重要论点：单纯的 refinement（带着旧答案重做）不如 self-reflection 引导的 refinement——蒸馏过的语言经验比原始轨迹更有信息量。

### 4.3 编程：HumanEval / MBPP / LeetcodeHardGym

编程任务给了 Reflexion 一个更"接地气"的自我评估工具： **自生成单元测试** 。具体做法是：用 CoT 提示让模型生成带自然语言描述的多样化测试，用抽象语法树（AST）解析过滤掉语法非法的，再从中采样最多 6 条组成测试套件 $T = \{t_0, t_1, \dots, t_n\}$ 。代码通过全部自测即可提交，否则触发反思重试——整个流程不接触任何隐藏测试用例，因此完全满足 pass@1 的报告资格。记忆上限设为 1 条经验。

实验覆盖 Python 和 Rust（用 MultiPL-E 编译器把 HumanEval/MBPP 子集翻译成 Rust），并引入新基准 **LeetcodeHardGym** ：40 道 GPT-4 预训练截止日期（2022 年 10 月 8 日）之后发布的 Leetcode hard 题，支持 19 种语言。

| 基准 + 语言 | 此前 SOTA pass@1 | 当前 SOTA pass@1 | **Reflexion pass@1** |
| --- | --- | --- | --- |
| HumanEval (Python) | 65.8（CodeT + GPT-3.5） | 80.1（GPT-4） | **91.0** |
| HumanEval (Rust) | -- | 60.0（GPT-4） | **68.0** |
| MBPP (Python) | 67.7（CodeT + Codex） | **80.1（GPT-4）** | 77.1 |
| MBPP (Rust) | -- | 70.9（GPT-4） | **75.4** |
| Leetcode Hard (Python) | -- | 7.5（GPT-4） | **15.0** |

唯一的例外是 MBPP Python（77.1 vs GPT-4 的 80.1）。作者很诚实地做了归因分析：

| 基准 + 语言 | Base | Reflexion | TP | FN | FP | TN |
| --- | --- | --- | --- | --- | --- | --- |
| HumanEval (PY) | 0.80 | **0.91** | 0.99 | 0.40 | 0.01 | 0.60 |
| MBPP (PY) | **0.80** | 0.77 | 0.84 | 0.59 | 0.16 | 0.41 |
| HumanEval (RS) | 0.60 | **0.68** | 0.87 | 0.37 | 0.13 | 0.63 |
| MBPP (RS) | 0.71 | **0.75** | 0.84 | 0.51 | 0.16 | 0.49 |

其中 TP 表示"测试通过且答案正确"，FP 表示"测试通过但答案错误"（假阳性），FN 表示"测试失败但答案正确"（假阴性），TN 表示"测试失败且答案错误"。关键在 FP 列：MBPP Python 的假阳性率高达 **16.3%** ，而 HumanEval Python 只有 **1.4%** 。也就是说，MBPP 上模型自写的测试套件质量较差，错误实现经常"骗过"测试被提前提交。作者还指出一个有意思的不对称性： **假阴性比假阳性好** ——测试误杀正确解时，Agent 还有机会通过反思识别出是测试本身写错了、保留原实现；而假阳性会让 Agent 直接交上一份错答案。

进一步的消融实验在 HumanEval Rust 最难的 50 题上进行：

| 方案 | 测试生成 | 自我反思 | pass@1 |
| --- | --- | --- | --- |
| Base model（GPT-4 裸跑） | ✗ | ✗ | 0.60 |
| 去掉测试生成 | ✗ | ✓ | 0.52 |
| 去掉自我反思 | ✓ | ✗ | 0.60 |
| **完整 Reflexion** | ✓ | ✓ | **0.68** |

两个结论值得细品：

- 没有测试反馈的反思是盲目的（0.52，比裸跑还差）——Agent 无法判断当前实现对不对，只能被迫改满所有轮次，反而把好代码改坏；
- 没有反思的试错是无效的（0.60，与裸跑持平）——测试和编译能报出语法、逻辑错误，但实现修复并不能落实这些提示。这说明近期一些"盲目重试式"调试方法在难任务上是行不通的， **从错误定位到实现改进之间，必须用反思这座桥来连接** 。

### 4.4 附录亮点：模型能力与其他发现

附录中的补充实验揭示了一个重要事实： **自我纠错是大模型的涌现能力** 。在 HumanEval Python 上换用较弱的 starchat-beta，Reflexion 与基线几乎无差异（均为 0.26）；而在 HotPotQA 上换用不同模型的结果如下：

| 模型 | 基线准确率 | Reflexion 准确率 |
| --- | --- | --- |
| CoT (GT) + text-davinci-003 | 0.60 | **0.77** |
| CoT (GT) + gpt-3.5-turbo | 0.57 | **0.71** |
| CoT (GT) + gpt-4 | 0.68 | **0.80** |
| ReAct + text-davinci-003 | 0.30 | **0.55** |
| ReAct + gpt-3.5-turbo | 0.26 | **0.38** |
| ReAct + gpt-4 | 0.39 | **0.51** |

模型越强，反思带来的增益越大——这与作者的判断一致：随着 LLM 能力进步，这个范式只会越来越好。

附录还记录了一个"失败案例"：在 WebShop（电商导购任务）上，Reflexion 跑了 4 轮毫无起色，甚至写不出有用的反思。作者的解释是，WebShop 需要处理自然语言搜索的高度歧义，要求极具多样性和探索性的行为，而这正是 Reflexion 这类"沿既有经验优化"的方法难以跳出局部最优的场景。

## 五、结论与展望

回顾全文，这篇工作的核心要点：

- **新范式** ：Reflexion 把策略参数化为"记忆 + LLM"，用语言反思替代梯度更新，实现免微调的强化学习；
- **三模块设计** ：Actor 生成、Evaluator 打分、Self-Reflection 把标量奖励放大为可执行的语言经验，存入长期记忆；
- **全面验证** ：AlfWorld +22%、HotPotQA +20%、HumanEval 达 91% pass@1 超 GPT-4，并发布 LeetcodeHardGym 新基准；
- **关键消融** ：单纯重试无效、盲目反思有害，"测试反馈 + 语言反思"缺一不可；自我纠错是大模型的涌现能力；
- **明确边界** ：需要高多样性探索的任务（如 WebShop）上会因局部最优而失效。

展望未来，作者认为自然语言形式的 value learning、off-policy 探索等传统 RL 技术都有机会被搬进这个语言框架里；记忆模块也可以从滑动窗口升级为向量数据库等更高级的结构。从今天的视角回看，Reflexion 几乎定义了后来所有 Agent 自我改进工作的基本范式——"把经验写成文字"这个朴素的想法，至今仍是 Agent 记忆系统设计的核心思想之一。

> 本文参考自 [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)