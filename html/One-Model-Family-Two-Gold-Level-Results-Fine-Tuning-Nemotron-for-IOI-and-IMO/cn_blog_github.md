# 一个模型家族，两枚金牌：Nemotron 微调后在 IOI 与 IMO 双双夺金

AI 在单一竞赛上拿牌已经不新鲜了，但同时在 **代码竞赛（IOI）和数学证明（IMO）** 这两条完全不同的赛道上都达到金牌水平，指向的是更本质的东西：一个通用底座模型究竟能不能被系统地改造成世界级领域专家。NVIDIA 这篇文章给出的答案是肯定的——以 Nemotron 3 为底座，团队只用 SFT、RL 和反馈式推理这些标准方法，就在 IOI 2026 上拿到 **535.4/600 分**（金牌线 361.12，甚至超过人类最高分 498.27），在 IMO 2026 上拿到 **30/42 分**（官方金牌线 29），且 IMO 证明由官方阅卷人评分。

![Nemotron 在 IOI 2026 与 IMO 2026 的成绩及金牌线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/One-Model-Family-Two-Gold-Level-Results-Fine-Tuning-Nemotron-for-IOI-and-IMO/images/LqM9etFUGTjJT116M7Hq8.png)

> 图解：左侧是 IOI 2026 的 535.4/600 分，超过金牌线 361.12 和人类最高分 498.27；右侧是 IMO 2026 的 30/42 分，超过金牌线 29。IOI 成绩来自同等时间、网络和提交限制下的真实参赛运行，不属于官方排名；IMO 提交证明由官方阅卷人评分。

## 两份成绩单：同一个底座，两种专家

先看最硬的数据，两场比赛用到的其实是同一套方法论的不同实例：

| 竞赛 | Nemotron 专家版本 | 成绩 |
| --- | --- | --- |
| IOI 2026 | Nemotron-3-Ultra-CC + SFT + GenCorrect | 535.4/600，高于金牌线 361.12 和人类最高分 498.27 |
| IMO 2026 | Nemotron 3 Ultra 通用版 + SFT + RL 三 checkpoint 组成的 generate-verify-refine 系统 | 30/42，高于官方金牌线 29 |

IOI 考的是算法与代码：在严格的时间和提交次数限制下，程序必须通过隐藏测试。IMO 考的是用自然语言写出严密的数学证明。两者需要的技能几乎正交，这正是这组结果的意义所在—— **同时做到两件事，说明能力来自底座模型和一套可复用的特化方法，而不是针对某一场比赛的奇技淫巧**。

## 可复用的特化配方：四步走

这篇文章反复强调的一点是："易微调"不应该只意味着 checkpoint 能被训练，而应该意味着一个有能力的底座模型可以用一套清晰、可复用的配方适配到高要求领域。两个项目共用同一套四步配方：

1. 从一个强大的 Nemotron 底座模型出发；
2. 整理领域特定题目和高质量推理轨迹；
3. 应用标准的后训练方法——SFT，必要时加 RL；
4. 给专家模型配上一个"生成—评估—改进"的推理循环。

> 博主点评：这个配方的聪明之处在于"克制"。不需要为每个挑战重训一个新底座模型，全部用标准件组装。这意味着这套方法是 **可复现、可迁移** 的，这也是 NVIDIA 把模型、数据、代码全部开源的底气。

## IOI 之路：从通用代码能力到竞赛金牌

明确了配方之后，第一个问题是：具体到竞赛编程上，数据和训练量要多大？

团队的答案是：整理 **22,000 道竞赛题目**，生成合成推理轨迹，训练两个专家模型：

- **Nemotron-3-Nano-CC**：总参数 300 亿，激活参数 30 亿，SFT + RL 全上；
- **Nemotron-3-Ultra-CC**：总参数 5500 亿，激活参数 550 亿，只做 SFT。

IOI 2025 上的递进实验把"特化的价值"展示得非常直观。Nano 的成绩轨迹是：

- 后训练前：130 分
- SFT 后：280 分
- RL 后：291 分
- 叠加 GenCorrect（迭代式的生成—评估—改进策略）后：468 分，越过 438.3 的金牌线

Ultra-CC 用同样的测试时策略直接拿到 502 分。

![Nemotron Nano 在 IOI 2025 上经过 SFT、RL 和测试时计算的分数变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/One-Model-Family-Two-Gold-Level-Results-Fine-Tuning-Nemotron-for-IOI-and-IMO/images/GptKxXbYlSsm2ksIJqZzE.png)

> 图解：Nano 从底座的 130 分，经三轮 SFT 达到 262、276、280 分，再经 RL 达到 291 分；五轮反馈式推理将分数依次推至 361、403、419、443、468，最终超过 438.3 的金牌线。横轴是后训练与测试时计算阶段，纵轴是 IOI 得分（满分 600）。

这里有一个值得注意的 **规模—训练量权衡** 发现：对小模型 Nano 来说，SFT 贡献了主要增益，RL 只带来较小但稳定的提升；而对更强的 Ultra 模型， **仅仅一个 epoch 的 SFT 就足以在 IOI、ICPC 和 LiveCodeBench Pro 三个榜单上全面超过完整后训练的 Nano**。换句话说，底座越强，需要的"塑形"越少。这个发现直接指导了 IOI 2026 参赛系统的构建——基于 Ultra 的 Ultra-CC 最终拿到 535.4 分。

> 博主点评：这其实是"scaling law 迁移到后训练阶段"的又一个证据——大模型的能力上限更高，少量的领域数据就能激活；小模型则需要更重的后训练和更多的测试时计算来"代偿"。

## IMO 之路：教模型证明、检查、再修改

解决了代码竞赛，下一个更硬的问题是数学证明——答案不再是可执行测试的代码，而是自然语言的严密论证。

IMO 项目沿用同一思路，从 Nemotron 3 Ultra 出发，分别训练一个 SFT 专家和一个 RL 专家：

- **SFT 语料**：包含 414,890 条质量筛选后的样本，覆盖 15,818 道互不相同的证明题。关键设计在于数据不只是教"最终答案"，而是覆盖 **证明生成、改进、验证、元验证（meta-verification）** 四类任务——模型既学会构造论证，也学会发现漏洞、回应批评、判断一个证明是否完整。
- **RL 训练**：在 9,597 道证明题上进行，这些题目特意选在模型能力边界附近（capability frontier）。

> 博主点评：把"验证"和"元验证"直接写进 SFT 数据是这套系统的点睛之笔。比赛系统的瓶颈往往不是生成能力，而是 **判断力** ——模型能不能分辨哪个候选证明真的成立。让生成者和评判者同源同体，后面的推理循环才有可靠的信号。

开发实验中，两个后训练 checkpoint 都超过了通用版本：SFT checkpoint 在首轮搜索中最强，RL checkpoint 则取得了最好的单 checkpoint 总体成绩。两者优势互补，所以最终系统 **同时使用 SFT、RL 两个专家加上通用模型**。

推理时的工作流是这样的：对每道题，多个模型生成候选证明、打分、产出批评意见、改进最有希望的尝试；最后一个独立的高算力阶段负责选出最终提交。整个系统 **纯自然语言运行，没有形式化证明器、没有外部工具、没有联网**，最终拿到 30/42 分，六道题中四道满分，越过官方金牌线。

![IMO 2026 推理过程中内部验证、外部评审与人工评分随时间的变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/One-Model-Family-Two-Gold-Level-Results-Fine-Tuning-Nemotron-for-IOI-and-IMO/images/Wo0kpoA9qTLPaxWv48DnL.png)

> 图解：横轴是分钟数（对数刻度），绿色实线是内部验证评分，蓝色虚线是事后外部评审，黑点是官方人工累计评分。比赛在 270 分钟结束，正式结果是 30/42。右侧灰色区域是赛后继续运行：内部评分约 35.2、外部评审 35、非官方人工重评 33，这些不能计入正式比赛成绩。

## 关键洞察：微调与测试时计算是乘法关系

这两项成绩放在一起，揭示了一个比单个竞赛结果更重要的规律： **后训练与测试时计算（test-time compute）必须协同设计**。

NVIDIA 此前关于 IOI 2025 的博客已经展示过，测试时计算可以把开源权重模型推到金牌水平。这次的新结果补上了另一半： **更好的特化意味着推理系统有更好的候选、更好的批评者、更好的改进者**。

- 在 IOI 上，GenCorrect 把微调带来的增益在多轮反馈中 **放大** 成更大的提升；
- 在 IMO 上，使用互补的 SFT + RL 双 checkpoint，比单纯从一个 checkpoint 多采样 **更有价值**。

金牌既不是微调单独造出来的，也不是暴力采样堆出来的——它们来自模型、数据、推理循环三者的共同设计（co-design）。这个区分对实际构建 AI 系统的人非常重要：只卷训练或只卷推理，都会浪费掉另一半的增益。

## 全部开源：模型、数据与配方

NVIDIA 希望这些结果的价值不止于比赛本身，因此把几乎全部资产都放了出来：

- **Nemotron Labs IMO 2026 collection**：SFT 和 RL checkpoint、两份训练数据集，以及新的 **Nemotron-IMO-Bench**（200 道奥赛级题目基准）；
- **IMO 论文**：描述训练方法和 generate-verify-refine 系统；
- **NeMo-Skills 仓库**：包含 IMO 推理 pipeline、prompts、提交的证明原文，以及可复现的 quickstart；
- **Nemotron-3-Ultra-CC 模型**：已在 Hugging Face 上线（`NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4`），IOI 论文提供训练配方和 GenCorrect 方法论，IOI 的评测与推理 pipeline 同样在 NeMo-Skills 中。

## 总结

- **一个底座，两枚金牌**：Nemotron 3 同一模型家族，经标准后训练后在 IOI 2026（535.4/600）和 IMO 2026（30/42）双双达到金牌水平。
- **可复用配方**：强底座 → 领域数据与推理轨迹 → SFT/RL → 生成—评估—改进推理循环，四步通用。
- **规模决定训练量**：Ultra 模型只需一个 epoch SFT 即超越完整后训练的 Nano；Nano 则更依赖 RL 和测试时计算代偿。
- **数据要教"判断力"**：IMO 的 SFT 语料覆盖证明的生成、改进、验证、元验证，是系统能自我纠错的关键。
- **训练 × 推理是乘法**：GenCorrect 放大微调增益，互补 checkpoint 优于暴力采样，co-design 才是夺金的真正原因。

展望来看，这套"通用底座 + 透明推理工作流"的组合拳，完全可以迁移到代码和数学之外的领域（比如科学发现、工程设计）；局限也很明显——超大规模的训练和推理算力门槛，短期内仍不是每个团队都能负担的。但至少，配方和所有原料都已经开源了。

> 本文参考自 [One Model Family, Two Gold-Level Results: Fine-Tuning Nemotron for IOI and IMO](https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026)