# 有品味的 Agent：如何度量并训练长程任务中的「决策品味」？

> 论文：The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks（Microsoft & CityU 等，2026 年 9 月）

LLM Agent 正在处理越来越长的任务：从 ML 研究的全流程（idea 到 paper），到跨越多个版本演化大型软件项目，甚至在部署中自我改进 scaffold。在这类 **长程任务（long-horizon tasks）** 中，Agent 要做大量「影响不止当前一步」的决策——测哪个假设、基于哪个实现继续、下一步跑哪个实验。

问题在于： **一个错误的决策在当下往往看起来很合理** ，它的代价要到 Agent 花掉大半预算之后才会显现。这篇文章把「做好长程决策的能力」称为 Agent 的 **品味（taste）** ，并回答了两个问题：

1. **能否在不依赖专家标注的情况下，自动度量一个 Agent 的品味？**
2. **品味能否被训练？**

答案是都能。作者构建了 **Taste-Bench** ——一个包含 502 道「品味题」的 benchmark，并证明通过蒸馏可以把这种判断力灌进模型权重，最终提升端到端任务成功率。

## 一、问题的提出：端到端榜单测不出「决策质量」

已有的 Agent benchmark（AgentBench、SWE-bench、SWE-bench Pro、MLAgentBench、RE-Bench 等）只报告「任务最终有没有完成」，完全不衡量过程中每个决策的质量。但直接度量决策质量有两个天然困难：

- **结果延迟可见** ：长程决策的好坏在决策点不可见，好选择和坏选择在当下看起来同样合理；
- **需要深厚的领域专业知识** ：用人工标注又贵又难扩展到新领域。

这篇文章的核心观察非常优雅： **轨迹的后半段本身就是前半段决策的「事后证据」（hindsight evidence）** 。现实中 Agent 系统经常对同一个任务做多次尝试，这些尝试往往在中途分岔走向不同方向。一旦分岔，每条分支的记录结果就标识出了哪个方向更好——轨迹本身提供了一组 **已标注的对比** 。作者把这种分岔点称为 **决策岔口（decision fork）** 。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/decision_fork_example.png)

> 图解：一个来自 ML 研究轨迹的决策岔口实例。在岔口处，被评测模型需要在两个候选方向中做选择；后续的 loss 曲线才会揭示候选 A 明显更优——但模型在做选择时看不到这些后续信息。这正是「品味」要被检验的瞬间。

度量方式随之水到渠成： **把轨迹冻结在岔口处，隐藏岔口之后的所有工作，让模型在两个方向中选一个** 。选中被事后证据支持的方向，就算有品味。

## 二、形式化：用「事后之明」给判断力打标签

作者把品味视为一种 **长程判断（long-horizon judgment）** ：更优方向的优势在决策时不可见，只在后续工作中显现。一个干净的实现和一个仓促的实现在写的时候能通过同样的测试，前者的优势只在「之后的每次改动都更容易」时才体现出来。

直接用单条轨迹的结果判断决策好坏是不行的，因为除了判断本身，执行质量和环境也会影响结果。作者的解法是寻找 **除判断之外一切都相同的跨轨迹对比** ：多次尝试共享一个等价前缀，然后在某点分岔——由于分岔前的前缀等价、且采样随机性把判断分配到了不同分支，分支结果的差异就可以主要归因于岔口处的判断。

形式化地，每个决策岔口定义一道品味题。设任务 $q$ 的多次尝试共享轨迹前缀 $h_t = (o_0, a_0, \ldots, o_t)$（$o$ 为 observation，$a$ 为 action），在时刻 $t$ 分岔为两个候选方向 $c_1$ 和 $c_2$。每条分支 $i$ 跑到结束，产生证据 $E_i$（测试结果或研究分数），结果度量 $U$ 把证据映射为标量。品味好的模型应在没有任何证据可用时，就选中期望结果更高的候选 $\mathbb{E}[U \mid h_t, c_i]$。于是用实现的结果在相同条件下估计这些期望，把岔口打上 **受支持候选（supported candidate）** 的标签：

$$
y = \arg\max_{i \in \{1,2\}} U(E_i)
$$

题目本身则是 $x = (q, h_t, c_1, c_2)$，岔口时刻 $t$ 之后的一切对被评测模型隐藏。模型 $\pi$ 输入 $x$ 输出一个选择 $\pi(x)$，其品味估计 $\widehat{T}(\pi)$ 就是 $\pi(x) = y$ 的问题占比。

## 三、Taste-Bench：502 道题是怎么炼出来的

把原始轨迹变成题目，意味着要从非结构化记录中恢复任务 $q$、前缀 $h_t$、候选方向和标签 $y$。作者用了 **两种互补的构造方式** ，覆盖两个领域（软件工程 + ML 研究），形成 2×2 的设计。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/tastebench_construction.png)

> 图解：Taste-Bench 的构建与过滤流水线。左半部分展示两种构造：parallel 构造对比同一任务两条独立尝试的分支；detour 构造对比单条轨迹中「被放弃的方向」与「后来的自救方向」。右半部分是过滤阶段，剔除两类坏题——只看候选措辞就能猜对答案的 trivial 题，以及标签与完整记录不一致的 undecidable 题。

### 3.1 两种岔口挖掘方式

**Parallel trajectories（平行轨迹）** ：同一任务的多次独立尝试在相同岔口分岔，且尝试走向结束时 Agent 根本没意识到自己选错了方向。构造时把一对结果相反的尝试在分岔点对齐：岔口前的等价部分成为前缀，两个方向写成简短中性的候选描述，哪条分支通过测试/达成实验目标，哪边就是标签。这种构造捕捉的是 **Agent 从未察觉的错误判断** 。

**Detour trajectories（弯路轨迹）** ：Agent 在单条轨迹内自我纠错——先走上一个方向，撞墙，放弃，换方向最终完成任务。记录本身就标记了错误（因为 Agent 放弃了该方向）。一个 generator 模型读完整轨迹和结果，依次定位三个事件：走入该方向的步骤、终结该方向的可观测失败、以及换方向完成任务的步骤。岔口放在「走入被放弃方向之前的那一步」，并检查前缀不泄露失败和修复方案。这种构造捕捉的是 **Agent 走了又纠正的错误** ，考察被评测模型能否比当事 Agent 更早识别失败。

### 3.2 挖掘、过滤与组成

- **数据来源** ：工程池包含 GPT-5.4/GPT-5.5 在 517 个 SWE-bench Pro 任务上产生的 2,677 条带评分 rollout；研究池来自 METR 公开的 MALT transcript，包含 47 个 RE-Bench / HCAST 研究任务上的 1,132 条运行记录（Claude 3.5/3.7 Sonnet、Sonnet 4、Opus 4、DeepSeek V3 等 Agent）。
- **生成器** ：GPT-5.6 Sol（high reasoning effort）按 rubric 读轨迹并提出候选岔口，证据必须逐字引用、顺序机械可验，证据不干净就必须拒绝。
- **双重过滤** （评委团为 Kimi K2.5、GPT-4.1、Llama 4 Maverick、Mistral Large 3，不含生成器）：
  - **Trivial 过滤** ：只给两个候选（不给轨迹），若所有评委都能答对，说明题目不考轨迹判断，丢弃；
  - **Undecidable 过滤** ：给评委完整记录（任务 + 轨迹 + 结果），只有所有评委都认同标签，题目才入选。

最终每道发布的题都满足： **至少一个评委在隐藏轨迹时答错** （保证不简单），且 **所有评委在看到完整记录时认同标签** （保证标签可靠）。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/filtering_funnel.png)

> 图解：四个 cell 的过滤漏斗。生成器共提出 4,657 个候选岔口，1,809 个通过 rubric，729 个躲过 trivial 过滤，最终 502 道题进入发布版本——总通过率仅 10.8%。严格的筛选是这套「免人工标注」流水线可信的关键。

最终 502 道题按 2×2 组织：engineering 390 题、research 112 题。各 cell 题量为：parallel engineering 124、parallel research 48、detour engineering 266、detour research 64。

**人工复核** ：作者抽样 100 道题做 human review，两位评审先在岔口处独立选方向，再看后续记录与结果的摘要判断哪个决策更好。在 172 个明确 A/B 判断中，170 个与挖掘标签一致， **一致率达 98.8%** ；两位评审之间的 Cohen's $\kappa = 0.973$。这说明自动挖掘的标签质量足够硬。

### 3.3 评测协议：位置偏置的应对

二选一题目存在 **位置偏置（position bias）** ：仅交换两个候选的顺序就会改变很多模型的答案。因此每道题评测两次——一次按固定种子顺序、一次完全反向。 **只有两种顺序都答对，这道题才算对** ；随顺序翻转的答案不算品味。这样随机猜测的准确率是 25%（每次 1/2，乘积），而永远偏好同一位置的模型得 0%。论文同时报告两种顺序上的 mean accuracy。

## 四、实验：前沿模型的「品味」及格了吗？

作者在相同接口和 token 预算下评测了 **14 个当代模型** ，覆盖 Claude、GPT、Grok、DeepSeek、GLM、MiniMax、Mistral 等主要前沿家族。主指标是上述双顺序 accuracy，headline 数字为 Average（research 与 engineering 子集准确率的 1:1 均值）。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/model_results_system.png)

> 图解：14 个模型在 Taste-Bench 上的表现。三列分别为 Average、research 子集、engineering 子集，每列独立从高到低排序。实心柱为「两种顺序都答对」的 accuracy，虚线延伸部分是两种顺序的 mean accuracy，须线为 95% item-bootstrap 区间。注意随机猜测基线是 25%。

### Finding 1：前沿模型的品味相当有限

最好的模型是 **GPT-5.6 Sol（59.7%）** ，GPT-5.5 紧随其后（59.5%），其余模型散布在更低的位置。要知道每道题都只是二选一，而随机猜测都有 25%——即便最强模型也无法在决策时可靠地识别更优方向。四个 cell 中，detour 岔口在两个领域都比 parallel 岔口更难，且这个差距超过两个领域之间的差距：parallel engineering（对比结果清晰分离的两条分支）最容易（模型均值 58.1%），detour engineering（要求比当事 Agent 更早发现错误）最难（35.9%）。

### Finding 2：错误集中在「时间地平线」长的岔口上

直觉上，决定性证据出现得越晚，岔口越难。作者给每个岔口标注了 **时间地平线（time horizon）** ——站在岔口处的观察者需要看多远才能明确判定受支持候选，分四级：

- **in prefix** ：前缀中已有排除某候选的决定性事实；
- **inferable** ：无单一决定性事实，但前缀中的线索综合起来足以判定；
- **next step** ：岔口后第一个 observation 即可定夺；
- **more work** ：需要完成一次局部验证或更大量的后续工作。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/horizon_effort.png)

> 图解：左图是按时间地平线分级的 accuracy（深线为 14 个模型的均值及 95% t 区间，浅色线为各模型），右图是三档 reasoning effort 设置下同一对比。可以清楚看到左图中 accuracy 随地平线变长而单调下滑，而右图中三档设置在每一级上几乎完全重叠。

14 个模型的均值从 in-prefix 级的 62.3% 一路跌到 more-work 级的 21.0%—— **已经低于随机猜测的 25%** 。这说明答这些题真的需要「预测后续工作」，而不是靠前缀表面线索。

### Finding 3：加大推理预算救不了品味

准确率下滑会不会只是「想得不够久」？作者对 GPT-5.6 Sol 和 GPT-5.6 Luna 在三档 reasoning effort 下重跑全部题目（6 个条件、6,024 条响应），其余协议不变。结果： **从最低到最高 effort，Sol 的准确率变化 -0.2 个点，Luna 变化 +2.2 个点** ，在每个时间地平线上各档设置都重叠。

有意思的是，两个模型在开启 reasoning 的每档设置下，都在 more-work 级产生了最多的 reasoning token——也就是准确率最低的那一级。模型其实 **认得出哪些岔口难** ，也在那里想得最久，但决定性证据只存在于后续工作中，多想无益。

### 与端到端 benchmark 的关系：品味不是端到端能力的复述

品味度量要有意义，前提是不能只是端到端能力的换一种说法。作者把各模型的 Average 与其 SWE-bench Verified 公开分数（Vals AI 榜单，统一 harness）对比：

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/taste_vs_swebench.png)

> 图解：Taste-Bench Average 对 SWE-bench Verified 分数的散点图，每个点是一个模型，虚线为最小二乘拟合，红点偏离拟合最远，灰色带标出 SWE-bench Verified 最高的四个模型。两者只有部分相关。

- Average 与 SWE-bench Verified 的 Pearson 相关为 $r = +0.63$（$R^2 = 0.39$）；
- 但在 engineering 子集上相关仅 $r = +0.37$——尽管该子集正是从 SWE-bench Pro 任务挖出来的，本应是最接近的对比；
- SWE-bench Verified 前四名彼此相差不到 4.0 分，而它们的 Taste-Bench Average 相差 10.7 分。典型例子：DeepSeek V4 Flash 在 SWE-bench Verified 排第 5、Taste-Bench 排第 10；GPT-5.5 则反过来，第 8 vs 第 2。

## 五、品味可以蒸馏：从「测」到「训」

既然每道题天然带有「两个候选 + 决定标签的结果」这种监督信号，能不能反过来 **从轨迹中训练品味** ？作者给出了肯定的答案。Base model 为 Qwen3.6-27B，训练只更新 LoRA adapter。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/distillation_advisor.png)

> 图解：蒸馏与建议注入总览。左：teacher 看过受支持候选（demonstration），把它的完整推理蒸馏给只看题目的 student；右：评测时 student 在每个岔口的判断被写成一条 advice 注入任务上下文，由一个固定的 executor 独立完成任务。

### 5.1 蒸馏配方：蒸推理，而不是拟合标签

- **任务不相交折（task-disjoint folds）** ：390 道工程题按任务切成两折，student 在一折上训练、另一折上评测，训练中永远见不到评测题的来源任务。
- **为什么不能用标签直接微调** ：每题只有一个二元标签、没有中间推理，直接拟合只会记住答案而学不会判断。
- **Privileged teacher 自蒸馏** ：teacher 和 student 是同一个冻结 base model，只是上下文不同。teacher 上下文里多了受支持候选的简短 demonstration，因此能在自由生成的推理中可靠选中正确方向；训练时把 teacher 生成的 token 在 student 上下文（只有任务、前缀、打乱顺序的两个候选）下对齐。损失是 student 到 teacher 的 token 级 **forward KL** （沿用了 SDPO 的自蒸馏目标），覆盖推理 token 和最终选择，且在 **teacher 采样** 的续写上计算——因为两个上下文的差异恰好体现在 teacher 因 demonstration 而改变预测的那些 token 上，student 自己的样本几乎不会产生这些 token。
- **泄漏控制** ：答案是封闭二选一，teacher 序列止于最终答案，输出中没有任何片段是 demonstration 文本的复制，从结构上杜绝表面泄漏。
- **Calibration** ：蒸馏后再用 student 自己的自由推理 trace，仅在答案位置用受支持标签做 cross-entropy 校准，只调整最终选择、不动推理分布。

训练成本相当友好：LoRA rank 16、79.7M 可训练参数，单张 A100 80GB 每折约 2 小时。

### 5.2 迁移到未见任务

训练折上单顺序正确率从 48.6% 升到 92.9%，但这分不清是学会判断还是背了答案，关键看 held-out 折：

- 双顺序 accuracy：base model 30.0% → **student 47.9%** （提升 17.9 个百分点）；
- 两种顺序的 mean accuracy：42.7% → 62.4%。

也就是说，在 **训练中从未见过的任务** 上，判断力依然迁移了。把 student 放进 14 个模型的工程子集排名，它超过 Claude Opus 5、追平 GLM-5.2，仅次于 GPT-5.6 Sol 和 GPT-5.6 Terra——一个 27B 的 LoRA student 在「品味」上挤进了前沿梯队。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2609.25804/figures/distillation_results.png)

> 图解：蒸馏结果。左图：在 390 道 held-out 工程题上，student 的双顺序 accuracy 显著高于 base model，虚线是 GPT-5.6 Sol 在同批题上的 56.9%。右图：在 held-out SWE-bench Pro 任务上，注入 student advice 后 executor 成功率逼近「全部 advice 正确」的上限（虚线条）。

### 5.3 端到端验证：advice 真的能提升任务成功率

最后一步是把品味变成实打实的任务收益。对 41 个 held-out SWE-bench Pro 任务（共 98 个岔口），把每个岔口处的判断写成一条 **advice** （包含岔口情形、应避开的候选、应选择的候选），放进任务上下文，由固定的 Qwen3.6-27B executor（SWE-agent）独立完成。三种设置对比：

| 设置 | Executor 成功率 |
| --- | --- |
| 无 advice | 14.6% |
| 全部正确的 advice（上限） | 39.0%（+24.4） |
| Student 给的 advice | **33.7%（+19.1）** |

Student 在 98 个岔口中选对了 77 个，其 advice 拿到了上限收益的绝大部分。这些任务在训练中完全未见，说明更好的判断力真实地转化为训练分布之外的任务成功。

**Finding 4：品味可以被蒸馏，更好的判断带来任务成功率的实际提升。**

## 六、与相关工作的位置

- **长程 Agent 评测** （AgentBench、SWE-bench 系、MLAgentBench、RE-Bench）测「能不能完成长任务」，Taste-Bench 测「任务内部选没选对方向」；
- **事前判断研究方向** （预测两个 idea 哪个更好、执行前偏好哪个方案等）判断的是孤立的想法或方案，Taste-Bench 判断的是 **已执行轨迹内部的岔口** ，每道题都携带 Agent 的处境且由记录结果打标；
- **过程评估与步级判断** （process supervision、Agent-as-a-Judge、Who&When 等）用人工标签或 rollout 结果给步骤打分、或把失败归因到决定性步骤，Taste-Bench 则让被测模型在看到后续结果 **之前** 做选择；
- **蒸馏与上下文内化** （STaR、LEAP、SDPO 等）：本文配方沿用 SDPO，但用受支持候选的 demonstration 替代环境反馈，并从 teacher 采样目标，把 in-context 判断转移进权重。

## 七、总结

这篇文章的贡献可以概括为三点：

1. **概念与度量** ：把 Agent 的「品味」形式化为在决策岔口选择更优方向的能力，并证明它可以仅凭现有轨迹的 hindsight 自动度量，无需专家标注；
2. **Benchmark** ：发布 Taste-Bench（502 题，覆盖软件工程与 ML 研究，2×2 设计），发现前沿模型普遍品味有限（最佳仅 59.7%）、错误集中在长时间地平线的岔口、且加大推理预算无济于事；
3. **可训练性** ：品味可以被蒸馏——27B LoRA student 在未见任务上判断力提升 17.9 个百分点，注入其 advice 后 executor 在 held-out SWE-bench Pro 上成功率从 14.6% 升至 33.7%。

对做 Agent 的人来说，这篇工作的启发是双重的：一方面，「会做事」和「会选路」是两种能力，端到端榜单只反映了前者的一部分；另一方面，Agent 系统每天产生的海量轨迹本身就是一座未开采的监督金矿——每一次失败的重跑、每一次中途的改道，都是一条免费的「品味」标注。

数据集与代码均已开源：[Hugging Face 数据集](https://huggingface.co/datasets/wenbopan/taste-bench)、[GitHub 代码库](https://github.com/wbopan/tastebench)。

> 本文参考自 [The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks](https://arxiv.org/abs/2609.25804)