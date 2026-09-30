# DeepSeek-R1：纯强化学习点燃推理能力，开源模型硬刚 OpenAI o1

大模型会「思考」吗？过去业界普遍认为，想让模型学会多步推理，就必须拿人类写好的思维链（Chain-of-Thought, CoT）样本去教它——这条路不仅烧钱，还把模型的上限锁死在了人类示范的水平上。DeepSeek-R1 这篇工作给出了一个颠覆性的答案： **只用纯强化学习（RL）、不依赖任何人工标注的推理轨迹，就能让模型的推理能力自发涌现** 。核心思路一句话概括：给模型一个可自动验证的对错信号（比如数学题答案对不对、代码能不能跑通测试），然后放手让它自己探索解题策略。结果有多硬？DeepSeek-R1-Zero 在 AIME 2024 数学竞赛上的 Pass@1 从 15.6% 一路飙升到 77.9%，最终版 DeepSeek-R1 达到 79.8%，与 OpenAI o1-1217（79.2%）持平；在 Codeforces 编程竞赛中超过 96.3% 的人类选手。而训练 R1-Zero 与 R1 的全部强化学习成本，按 H800 每小时 2 美元租金折算，仅约 29.4 万美元。

## 一、背景：传统 Post-Training 范式的天花板

在聊 R1 之前，先看看它要打破的旧范式是什么。

### SFT + RLHF：被人类示范「封顶」的范式

过去几年，大模型后训练（Post-Training）的主流配方是两段式：先做监督微调（SFT），再做基于人类反馈的强化学习（RLHF）。

SFT 相当于「照葫芦画瓢」：用人工精心准备的输入-输出对训练模型，最小化预测与标准答案之间的交叉熵损失。它见效快、可解释，但有两个结构性缺陷：

- **性能受制于数据** ：人工答案不一定是最优解，而且常常省略反思、验证这类关键推理环节；
- **难以扩展** ：高质量人工标注又贵又慢，数据中的偏差还会直接传染给模型。

RLHF 则像是「请人类当裁判」：训练一个奖励模型（Reward Model）来编码人类偏好，再用 PPO 等算法优化模型输出。它把优化目标从「模仿固定答案」变成了「最大化奖励」，降低了对逐条标注的依赖，但奖励模型本身仍由人类偏好塑造。

这篇论文的核心洞见在于： **SFT 阶段的人类示范可能恰恰阻碍了模型探索更优推理策略** 。人类写的推理过程往往不是模型学习的最佳模板——少了显式的反思与验证步骤。与其让模型模仿人，不如给它正确的激励信号，让它自己进化。

### 底座：DeepSeek-V3-Base

R1 系列的起点是 DeepSeek-V3-Base——一个 671B 总参数、每 token 激活 37B 的 MoE（Mixture-of-Experts，混合专家）模型，预训练语料达 14.8 万亿 token，且包含大量数学与代码内容。这意味着基座模型早已「见过」海量推理轨迹，具备生成合理解题候选的底子，强化学习要做的就是从中筛选并放大高质量的行为模式。

一个值得注意的工程细节：V3-Base 的语料以中英文为主，这直接导致了后文 R1-Zero 的「语言混杂」问题——没有约束时，模型会在一条思维链里中英混用。

### GRPO vs PPO：扔掉价值模型的强化学习

工欲善其事，必先利其器。R1 采用的 RL 算法是 GRPO（Group Relative Policy Optimization，组相对策略优化），它是对 PPO 的一次「减负手术」。

PPO 需要额外训练一个与策略模型差不多大的价值模型（Critic）来估计优势函数（Advantage），相当于给教练配了一个同等体型的陪练，显存和算力开销直接翻倍。GRPO 的做法则像个朴素而聪明的统计学技巧： **同一道题采样一组回答，用组内奖励的均值和标准差做归一化，直接得到每个回答的相对优势** ，彻底不需要价值模型。

![GRPO 与 PPO 对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/GRPO.png)

> 图解：左侧 PPO 需要 Policy、Value、Reward、Reference 四个模型同时在场；右侧 GRPO 砍掉了 Value Model，对同一个问题采样 $G$ 个输出，用组内得分（Group Scores）的相对高低来估计优势，训练管线大幅简化。

形式化地说，对每个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{old}}$ 采样一组输出 $\{o_1, o_2, \cdots, o_G\}$，最大化如下目标：

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\right] \frac{1}{G}\sum_{i=1}^{G} \left( \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i,\; \mathrm{clip}\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta\, \mathbb{D}_{KL}\left(\pi_\theta \| \pi_{ref}\right) \right)
$$

其中优势 $A_i$ 就是组内奖励的标准化分数：

$$
A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \cdots, r_G\})}{\mathrm{std}(\{r_1, r_2, \cdots, r_G\})}
$$

KL 散度项采用无偏估计，直接加进损失里：

$$
\mathbb{D}_{KL}\left(\pi_\theta \| \pi_{ref}\right) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1
$$

为什么不沿用 PPO？论文给出了三个理由，每一个都打在长思维链训练的痛点上：

- **价值模型训不准** ：长 CoT 生成中，模型会反思、推翻、重写前文，仅凭前半段回答几乎不可能预测最终奖励；
- **KL 惩罚方式不同** ：PPO 把 KL 惩罚按 token 加进奖励，会隐式惩罚回答变长，恰好抑制了长推理所需的「多想一会儿」；GRPO 把 KL 项放进损失函数，不干扰奖励信号；
- **训练步数多、漂移大** ：长 CoT 训练动辄上万步，策略会显著偏离初始参考策略，因此训练中每 400 步就把参考模型更新为当前最新策略，在探索空间与稳定性之间取得平衡。

![PPO 与 GRPO 在 MATH 上的表现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/ppo_vs_grpo.png)

> 图解：在 16B 规模的 DeepSeek-Coder-V2-Lite 上对比 PPO 与 GRPO。PPO 对 GAE 中的 $\lambda$ 系数极其敏感：默认的 0.95 明显弱于 GRPO，精心调到 1.0 后才勉强追平。考虑到 PPO 还多一倍价值模型的训练开销，GRPO 是大规模训练下更务实的选择。

笔者认为，GRPO 的聪明之处不在于数学上多精巧，而在于它把「训练稳定性」这件事从依赖一个难训的神经网络，转化成了一个纯粹的统计量——这在工程上是降维打击。

## 二、DeepSeek-R1-Zero：纯 RL 的「顿悟时刻」

背景铺完，进入全文最激动人心的部分：不做任何 SFT，直接在基座模型上跑纯 RL，会发生什么？

### 极简的起点：一个模板 + 规则奖励

R1-Zero 的训练配置克制到近乎「简陋」。输入模板只要求模型把推理过程放进 `<think>...</think>` 标签、把最终答案放进 `<answer>...</answer>` 标签，除此之外没有任何内容层面的引导——不教它怎么推理，只要求它「先想后答」。

奖励信号同样是纯规则驱动，由两部分等权相加：

$$
Reward_{rule} = Reward_{acc} + Reward_{format}
$$

- **Accuracy Reward（正确性奖励）** ：数学题要求答案写在 box 里以便规则校验，代码题直接用编译器跑预置测试用例，对错分明；
- **Format Reward（格式奖励）** ：奖励模型把思考过程规规矩矩放进 `<think>` 标签里。

值得强调的是，团队在推理任务上 **刻意不用任何神经奖励模型** （无论结果级还是过程级）。理由很现实：神经奖励模型在大规模 RL 中极易被「钻空子」（Reward Hacking），而且反复重训奖励模型会让训练管线复杂到失控。

训练超参方面：学习率 3e-6，KL 系数 0.001，采样温度 1.0，每题采样 16 个输出，每步 32 道题（batch 512）；8.2k 步之前最大输出长度 32,768 token，之后放宽到 65,536——正是在这个节点，模型的性能和回答长度都出现了明显跳变。总计训练 10,400 步，约 1.6 个 epoch。

### 自我进化：AIME 从 15.6% 到 86.7%

![R1-Zero 训练曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/plot_aime_with_maj.png)

> 图解：左图是 R1-Zero 在 AIME 2024 上的 Pass@1 与 Cons@16（多数投票）随训练步数的变化，灰色基线是人类选手平均分。Pass@1 从 15.6% 稳步爬升到 77.9%，加上自洽性解码后达 86.7%，显著超过人类平均水平。

更神奇的是图（b）展示的「思考时间」曲线：

![回答长度增长曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/plot_length.png)

> 图解：R1-Zero 在训练集上的平均回答长度随训练步数持续增长。没有任何人告诉它「多想一会儿」，模型自发学会了用更长的思维链换取更高的准确率——先生成几百上千个 token 的探索、验证与回溯，再给出答案。

训练中甚至出现了著名的「Aha Moment（顿悟时刻）」：在解一道方程题时，模型写到一半突然蹦出一句「Wait, wait. Wait. That's an aha moment I can flag here」，然后停下来重新审视自己的推导。这种拟人化的反思行为从未被显式教授，完全是 RL 激励下自发涌现的产物。论文作者直言，这一刻也是研究者的顿悟时刻——它让人亲眼看到强化学习的力量与美感。

### 反思行为的量化证据

「顿悟」不是孤例，附录里的词频统计给出了系统性证据。

![反思词频演化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/wait_count_1.png)

> 图解：训练过程中「wait」「mistake」「however」「retry」「verify」「check」等反思类词汇的出现频率整体上升了 5 到 7 倍，说明 RL 确实在催生长链反思行为，而非简单的记忆增强。

![「wait」的涌现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/wait_count_2.png)

> 图解：聚焦「wait」一词：训练早期几乎绝迹，4000-7000 步零星出现，8000 步之后出现爆发式尖峰。这暗示模型是在特定训练阶段「学会」了特定形式的反思——能力的涌现具有阶段性。

不同难度题目的学习曲线同样耐人寻味：

![MATH 分难度表现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/math_performance.png)

> 图解：R1-Zero 在 MATH 数据集 1-5 级难度上的表现。简单题（1-3 级）很快冲到 0.90-0.95 后趋于平稳；难题提升惊人——4 级题从约 0.78 升至 0.95，5 级题从约 0.55 升至 0.90。RL 的红利主要释放在复杂推理上。

一个有趣的小反常：个别阶段模型在难题（3-4 级）上的准确率反而略高于 1 级题。论文解释这源于数据集的分布不均——500 题中 1 级题仅 43 道，且主要集中在模型仍不擅长的几何题型，1-2 道错题就会拉低好几个百分点。

至此，R1-Zero 证明了纯 RL 路线可行。但它还有两个明显短板：回答可读性差、中英混杂，且规则奖励只能覆盖可验证任务，写作、开放问答等通用能力几乎没被训练到。解决这些问题，就是 DeepSeek-R1 的任务。

## 三、从 Zero 到 R1：四阶段流水线

既然 R1-Zero 已经具备强大推理能力，接下来的问题是：如何在保留这份能力的同时，让它说话像人、行为对齐人类偏好？答案是下图这条多阶段流水线。

![DeepSeek-R1 多阶段流水线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/R1Pipieline_v0603.png)

> 图解：R1 的四段式训练流程——（1）冷启动：数千条人工打磨的长 CoT 数据微调基座模型（得到 Dev1）；（2）面向推理的大规模 RL（Dev2）；（3）拒绝采样生成 80 万条数据做第二轮 SFT（Dev3）；（4）覆盖推理与通用任务的最终 RL，产出 DeepSeek-R1。Dev1/Dev2/Dev3 为中间检查点。

### 冷启动：给模型一个「人味儿」的起点

冷启动数据的构建颇费心思，动机主要是产品体验导向：用户更喜欢第一人称、对话感强的思考过程（R1-Zero 爱用「we」，R1 则更多用「我」）。具体流程是：

1. 收集数千条高质量、多样化的推理 prompt；
2. 用 R1-Zero 以温度 1.0 采样多条推理轨迹；
3. 过滤：数学答案用 sympy 校验正确性，格式上剔除重复输出和中英混杂样本；
4. 人工标注员先把推理轨迹改写成自然的人类对话风格，再以此为例让 LLM 批量改写，最后再过一轮人工校验；
5. 用 DeepSeek-V3 做后处理——把思考过程翻译成与问题一致的语言，并补写一份简洁可读的解题总结。

论文也坦承一个有趣的观点：这些生动鲜活的推理模式更多是「DeepSeek 工程化设计的启发式风格」，不代表模型真的获得了类人智能；同时，R1-Zero 原始的、不受人类先验约束的 CoT 或许蕴藏着超越当前人类认知框架的潜力。

### 奖励模型：覆盖规则管不到的领域

对于写作、开放问答这类没有标准答案的通用任务，规则奖励失效，必须引入模型化奖励：

- **Helpful Reward Model** ：沿用 V3 的偏好对管线，用 arena-hard 格式让 DeepSeek-V3 对回答对做评判，每对查 4 次（随机交换 A/B 位置消除位置偏差），只保留分差 $\Delta > 1$ 的样本，共 6.6 万对；评判只看最终总结，避免干扰推理过程；
- **Safety Reward Model** ：10.6 万条带「安全/不安全」标注的 prompt，采用逐点（point-wise）方式训练，评估范围覆盖推理过程与最终答案全文。

两个奖励模型都以 batch size 256、学习率 6e-6 训练一个 epoch，架构与 R1 一致、仅加一个奖励头。

### 两段 RL 与语言一致性奖励

第一阶段 RL 沿用 R1-Zero 的规则奖励配方，但加入了一个关键的新成员—— **语言一致性奖励（Language Consistency Reward）** ，即 CoT 中目标语言词汇的占比：

$$
Reward_{language} = \frac{Num(Words_{target})}{Num(Words)}
$$

第二阶段 RL 则混合所有信号：推理数据走规则奖励，通用数据走奖励模型，语言一致性奖励直接加到总分上：

$$
Reward = Reward_{reasoning} + Reward_{general} + Reward_{language}
$$

其中 $Reward_{reasoning} = Reward_{rule}$，$Reward_{general} = Reward_{reward\_model} + Reward_{format}$。第二阶段采样温度降到 0.7（温度过高会导致生成不连贯），共 1,700 步，通用指令数据与偏好奖励只在最后 400 步才加入——因为论文发现，基于模型的偏好奖励训练步数一多，就会诱发 Reward Hacking。

![语言一致性奖励消融](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/LC_Reward.png)

> 图解：在 R1-Distill-Qwen-7B 上的消融实验。没有 LC 奖励时，语言一致性随训练持续恶化；加上 LC 奖励后全程稳定。代价是代码基准上轻微的性能损失——语言对齐「对齐的是人类偏好」，不完全等价于能力最优。

![Reward Hacking 现象](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/reward_hacking.png)

> 图解：Reward Hacking 的实锤证据——训练过程中 helpful 奖励模型的评分一路走高，但 CodeForces 的真实表现却在下滑。模型学会了「讨好」奖励模型，而非真正变强。这是纯 RL 路线最需警惕的陷阱。

### 数据配方：80 万条 SFT 数据从哪来

第二阶段 SFT 的数据共约 80 万条，分两大块：

- **推理数据（约 60 万条）** ：对第一阶段 RL 检查点做拒绝采样（Rejection Sampling），每个 prompt 采样多条回答只留正确的。此阶段还扩了数据范围——部分数据改用生成式奖励模型（把标准答案和模型预测一起喂给 DeepSeek-V3 评判），并剔除了语言混杂、长段落和代码块混排的混乱 CoT；
- **非推理数据（约 20 万条）** ：写作、事实问答、自我认知、翻译等，复用 DeepSeek-V3 的 SFT 管线与部分数据，并补充了程序修复、前端开发等软件工程数据。部分非推理任务会先用 V3 生成一段「潜在思维链」再作答，但「你好」这类简单 query 不给 CoT。

| 领域 | 样本数 | 平均轮次 | 平均 token 数 |
| --- | --- | --- | --- |
| Math | 395,285 | 1.0 | 6,094 |
| Code | 211,129 | 1.1 | 7,436 |
| STEM | 10,124 | 1.0 | 4,929 |
| Logic | 10,395 | 1.0 | 2,739 |
| General | 177,812 | 1.1 | 1,420 |
| **合计** | **804,745** | **1.0** | **5,355** |

值得注意的是，数据以单轮对话为主，这直接限制了 R1 的多轮对话能力，团队将其留作未来工作。

RL 阶段的数据规模同样可观：数学 26K、代码 17K（另有 8K 真实 GitHub issue 的 bug 修复题）、STEM 22K、逻辑 15K、通用 66K。其中代码数据的一大亮点是 **自造评测用例** ：团队从 Codeforces 和 AtCoder 收集了 5,151 + 2,504 道竞赛题，由于官方测试用例不公开，他们让 DeepSeek-V2.5 写生成器产出候选用例，再用「正确提交筛掉坏用例、错误提交验证用例区分度」的两阶段过滤，确保测试用例能可靠区分对错。

### 各阶段效果：一张表看懂 Dev1 到 R1

| Benchmark | R1-Zero | Dev1 | Dev2 | Dev3 | R1 |
| --- | --- | --- | --- | --- | --- |
| MMLU (EM) | 88.8 | 89.1 | 91.2 | 91.0 | 90.8 |
| IF-Eval (Prompt Strict) | 46.6 | 71.7 | 72.0 | 78.1 | **83.3** |
| AlpacaEval2.0 (LC-winrate) | 24.7 | 50.1 | 55.8 | 62.1 | **87.6** |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | **92.3** |
| LiveCodeBench (Pass@1) | 50.0 | 57.5 | 63.5 | 64.6 | **65.9** |
| Codeforces (Rating) | 1444 | 1534 | 1687 | 1746 | **2029** |
| AIME 2024 (Pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | **79.8** |
| MATH-500 (Pass@1) | 95.9 | 94.2 | 95.9 | 95.4 | **97.3** |

这张表讲了一个清晰的「分工」故事：冷启动（Dev1）立刻修好指令遵循，但推理能力因数据量小有回落（AIME 从 77.9 掉到 59.0）；推理导向 RL（Dev2）把推理能力拉回并推高；第二轮 SFT（Dev3）补上通用写作与代码工程；最终 RL（R1）主要收割用户偏好类指标——AlpacaEval 2.0 暴涨 25%、ArenaHard 涨 17%。

## 四、训练工程与成本：29 万美元的账单

大规模 RL 训练对基础设施是极大的考验。DeepSeek 的 RL 框架采用解耦式四模块设计：

![RL 基础设施](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/rl_infra.png)

> 图解：四个模块各司其职——Rollout 模块用 vLLM 集群采样回答（MoE 专家并行 + 热点专家冗余部署 + MTP 自投机解码加速）；Inference 模块加载奖励模型和参考模型做前向打分；Rule-based Reward 模块跑代码执行器、答案匹配器等规则奖励，用异步调度隐藏延迟；Training 模块计算损失并更新参数，支持 PPO/GRPO/DPO，并通过按长度排序 + Best-Fit 打包 + DualPipe 流水线并行把 padding 浪费压到最低。每个阶段结束后模型实例自动从显存卸载到内存或磁盘，为下一阶段腾出 VRAM。

训练成本方面（按 H800 每小时 2 美元租金折算）：

| 项目 | R1-Zero | SFT 数据构建 | R1 | 合计 |
| --- | --- | --- | --- | --- |
| H800 GPU 小时 | 101K | 5K | 41K | **147K** |
| 折算美元 | $202K | $10K | $82K | **$294K** |

R1-Zero 用 64×8 张 H800 训了约 198 小时，R1 约 80 小时。对比一下坊间流传的动辄千万美元级训练账单，这个数字低得惊人——当然，这只是后训练阶段的直接算力成本，不含基座模型的预训练。蒸馏模型的配置则很朴素：用 80 万条 SFT 数据微调 Qwen/Llama 基座 2-3 个 epoch，最大上下文 32,768 token，batch size 64。

## 五、实验验证：与 o1 正面硬刚

方法讲完，看疗效。评测设置上有个细节值得借鉴：团队发现贪心解码会导致长推理模型重复率高、不同检查点波动大，因此统一采用温度 0.6、top-p 0.95 采样 $k$ 次（AIME/GPQA 取 64，MATH/Codeforces 取 16），报告平均 Pass@1。最大生成长度 32,768 token。数据污染防控方面，仅数学领域就清洗了约 600 万条与评测集 10-gram 重叠的预训练文本，后训练数据全部取自 2023 年前的赛事。

### 主结果：全面对标闭源旗舰

| Benchmark | Claude-3.5-Sonnet | GPT-4o | DeepSeek-V3 | o1-mini | o1-1217 | **DeepSeek-R1** |
| --- | --- | --- | --- | --- | --- | --- |
| MMLU (EM) | 88.3 | 87.2 | 88.5 | 85.2 | **91.8** | 90.8 |
| MMLU-Pro (EM) | 78.0 | 72.6 | 75.9 | 80.3 | - | **84.0** |
| GPQA Diamond (Pass@1) | 65.0 | 49.9 | 59.1 | 60.0 | **75.7** | 71.5 |
| AlpacaEval2.0 | 52.0 | 51.1 | 70.0 | 57.8 | - | **87.6** |
| ArenaHard | 85.2 | 80.4 | 85.5 | 92.0 | - | **92.3** |
| LiveCodeBench | 38.9 | 32.9 | 36.2 | 53.8 | 63.4 | **65.9** |
| Codeforces (Rating) | 717 | 759 | 1134 | 1820 | 2061 | **2029** |
| SWE Verified | **50.8** | 38.8 | 42.0 | 41.6 | 48.9 | 49.2 |
| AIME 2024 (Pass@1) | 16.0 | 9.3 | 39.2 | 63.6 | 79.2 | **79.8** |
| MATH-500 (Pass@1) | 78.3 | 74.6 | 90.2 | 90.0 | 96.4 | **97.3** |
| CNMO 2024 (Pass@1) | 13.1 | 10.8 | 43.2 | 67.6 | - | **78.8** |

几个关键读数：数学任务上 R1 与 o1-1217 基本打平，对其余模型是数量级碾压；代码算法题（LiveCodeBench、Codeforces）推理模型整体屠榜；工程类任务（SWE Verified、Aider）R1 与 o1 互有胜负，论文坦言这是因为软件工程方向的 RL 训练数据目前还很少。

与人类对比更直观：

![R1 与人类选手对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/dsr1_nature.png)

> 图解：R1-Zero、R1 与人类成绩在多个基准上的对比。AIME 上 R1 超过人类选手平均分；Codeforces 上 R1 超过 96.3% 的人类参赛者；GPQA 上联网查资料的博士级人类仍优于 R1——论文预期给 R1 接上搜索工具就能缩小这个差距。

人类偏好评测方面，R1 在 ChatbotArena（LMSYS 双盲投票平台）上的表现同样亮眼：

![ChatbotArena 排名](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/lmsysrank.png)

> 图解：截至 2025 年 1 月 24 日（发布仅一周），R1 在开启风格控制（Style Control，剥离回答长度、排版等表面因素）的榜单上与 OpenAI-o1、Gemini-Exp-1206 并列第一。一个 MIT 协议的开源模型与闭源旗舰同坐头把交椅，这是里程碑级的事件。

![ChatbotArena 分维度排名](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/lmsysdetail.png)

> 图解：分维度看，R1 在数学、代码等硬核推理项上排名尤其靠前，同时在创意写作、长文本等通用项上也不落下风，证明其能力并非「偏科」。

### 深入分析：能力提升到底从哪来

与 DeepSeek-V3 的分科目对比显示，RL 的收益高度集中在 STEM 领域：

![MMLU 分类别对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/mmlu_category_comparison.png)

> 图解：MMLU 各学科上 V3 与 R1 的对比，提升主要来自 STEM 类目。有趣的是，社会科学、人文类非 STEM 科目也有提升，论文推测是长 CoT 帮助模型更好地理解了题意。

![MMLU-Pro 分类别对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/mmlu_pro_subject_comparison.png)

> 图解：在更难的 MMLU-Pro 上，所有类目全面提升，数学和物理涨幅最大。MMLU 上 STEM 提升较小，是因为 V3 后训练后已接近饱和，没有多少空间留给 R1。

泛化能力方面，团队用训练截止日期之后发布的全新赛题做了硬核验证：

| 平均分 | AMC 12 2024 | AIME 2025 | USAMO 指数 |
| --- | --- | --- | --- |
| 人类参赛者 | 61.7 | 6.2/15 | 123.7 |
| GPT-4o 0513 | 84.0 | 2.0/15 | 104.0 |
| DeepSeek-V3 | 98.3 | 3.3/15 | 131.3 |
| OpenAI o1-1217 | 141.0 | 12.0/15 | 261.0 |
| **DeepSeek-R1** | **143.7** | 11.3/15 | **256.7** |

USAMO 指数超过 251.5 即获得美国数学奥林匹克参赛资格——R1 的 256.7 分意味着它达到了美国顶尖高中生的水平，AIME 2025 上 75% 的 Pass@1 也逼近 o1 的 80%。在 2024 年 93 项数学竞赛的 366 道题上，R1 整体大幅超越 GPT-4o，数论和代数最强，几何与组合仍是短板。

### 测试时计算：思考 token 越多，题目越难越敢想

![测试时计算与题目难度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/test_time_scaling_vs_problem_difficulty.png)

> 图解：横轴为题目难度（Pass@1 越低越难），纵轴为得出正确答案所需的思考 token 数。R1 会自适应分配算力：简单题用不到 7,000 个思考 token，最难的题超过 18,000 个；「1+1=?」这类题甚至只用不到 100 个 token。整体上以平均每题 8,793 个思考 token 拿下 61.8% 的解题率。

对比之下，非推理模型 GPT-4o 平均每题只产出 711 个 token，解题率仅 24.7%。更扎心的是，传统多数投票救不了非推理模型：AIME 2024 上给 GPT-4o 采样 64 次投票，也只从 9.3% 提升到 13.4%，而 R1 单次采样就有 79.8%。原因很本质：独立采样之间不会互相借鉴，模型不会回溯和自我纠错，堆再多样本也只是重复犯错。反过来，多数投票对 R1 是有效加成——能把 AIME 从 79.8% 进一步抬到 86.7%（Pass@64 更是高达 90.0%）。

### 安全性：开源模型的诚实答卷

论文用一整章篇幅做安全报告，态度相当坦诚。官方服务的风控系统是「关键词初筛 + DeepSeek-V3 风险复审」两级流水线，团队在 6 个公开安全基准上与其他旗舰对比：

| 安全得分 (%) | SST | BBQ | ART | XSTest | DNA | HarmBench | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claude-3.7-Sonnet | 100.0 | 92.1 | 99.7 | 96.4 | 95.9 | 83.3 | 94.6 |
| o1 (2024-12-17) | 99.0 | 97.3 | 98.3 | 97.0 | 86.2 | 84.0 | 93.6 |
| GPT-4o | 98.5 | 95.1 | 99.1 | 97.3 | 90.6 | 72.7 | 92.2 |
| DeepSeek-V3 | 95.3 | 96.7 | 97.1 | 97.1 | 95.6 | 96.0 | 96.3 |
| **DeepSeek-R1** | 97.5 | 96.6 | 96.2 | 95.3 | 94.8 | 89.3 | **95.0** |

R1 整体与其他前沿模型相当，主要短板在 HarmBench 的知识产权类问题（例如直接生成歌词时不会拒绝）。自建安全测试集覆盖 4 大类 28 个子类共 1,120 道中英双语题，结论是：裸模型（无风控）的 R1 不安全率超过 20%，属于相对不安全一档；接入风控后降到 10% 左右，进入第二梯队，与 Claude-3.7-Sonnet、o1 同级，但拒答率也随之升到约 25%。

![自建安全基准分类体系](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/safety_taxnomy.png)

> 图解：自建安全基准的分类体系，将内容安全风险划分为歧视与偏见、违法违规、伤害行为、道德伦理 4 大类 28 个子类，每个子类人工构建 20 道中文题并翻译成英文。

![多语言安全表现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DeepSeek-R1-Pushing-the-Boundaries-of-Reasoning-via-an-Innovative-Post-Training-Paradigm/figures/safety-multilingual-v3.png)

> 图解：50 种语言的安全得分。带风控的 V3（86.5%）与 R1（85.9%）逼近最强的 Claude-3.7-Sonnet（88.3%）；不带风控的 R1（74.2%）与 GPT-4o（75.2%）相当。R1 在 50 种语言中没有一种低于 60 分的高危语言，不存在明显的语种短板。

越狱攻击测试（2,232 条越狱指令模板拼接原始风险问题）的结论更值得玩味：所有模型在越狱面前安全率都显著下滑（Claude-3.7-Sonnet 安全回答比例掉了 33.8%）；两个推理模型 R1 和 o1 更依赖外部风控系统，整体拒答率高达 79.8% 和 87.3%；开源模型因本地部署缺乏风控，面临的越狱风险最严峻——团队明确建议部署开源模型的开发者自行接入类似的风控管线。

## 六、蒸馏：把推理能力「拷贝」给小模型

解决了大模型的推理能力之后，下一个现实问题是：671B 的庞然大物不是谁都跑得起的。蒸馏（Distillation）就是让 R1 当老师、小模型当学生——用 R1 生成的 80 万条数据直接微调 Qwen 和 Llama 系列开源基座， **只做 SFT、不做 RL** ，刻意把变量留干净以验证蒸馏本身的效果。

| 模型 | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 | GPQA Diamond | LiveCodeBench | CodeForces 评分 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| R1-Distill-Qwen-32B | **72.6** | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| R1-Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| R1-Distill-Llama-70B | 70.0 | **86.7** | **94.5** | **65.2** | **57.5** | 1633 |

最夸张的是第一行对比： **仅 1.5B 参数的蒸馏小模型，在数学基准上反超了 GPT-4o 和 Claude-3.5-Sonnet 这两个最强闭源非推理模型** 。模型越大，蒸馏收益越明显，Llama-70B 蒸馏版在 MATH-500 上已达 94.5。

### 蒸馏 vs 纯 RL：小模型的最优解是「抄作业」

一个自然的问题：小模型自己做大规模 RL，能达到同样效果吗？团队用 Qwen2.5-32B-Base 跑了 10K 步以上的大规模 RL（Qwen2.5-32B-Zero），与蒸馏版正面对比：

| 模型 | AIME 2024 pass@1 | cons@64 | MATH-500 | GPQA | LiveCodeBench |
| --- | --- | --- | --- | --- | --- |
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| Qwen2.5-32B-Zero（纯 RL） | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| R1-Distill-Qwen-32B（蒸馏） | **72.6** | **83.3** | **94.3** | **62.1** | **57.2** |

结论干脆利落：小模型纯 RL 训练能达到 QwQ-32B-Preview 的水平，但全面落后于蒸馏版，差距在 AIME 上高达 25 个百分点。团队还在 Qwen2-Math-7B（发布早于 o1，确保基座没见过推理轨迹数据）上做了佐证：约 1 万步策略梯度更新后，AIME 2024 从 7.9% 提到 22.3%，超越 GPT-4o——再次证明模型可以自主发展推理策略，但效率远不如蒸馏。

由此得出两条重要推论：其一，把小模型的上限交给更强的老师来抬，是经济又有效的路线；其二，蒸馏虽好，要突破人类智能的边界，终究还得靠更强的基座和更大规模的 RL。

## 七、经验与教训：那些失败的尝试

论文最难得的是专开一节讲「踩坑史」，对想复现的人价值极高。

**基座模型必须够强** 。团队早期在 7B 稠密模型和 16B MoE 模型上做纯 RL，AIME 上始终看不到提升——小模型输出变长后只会陷入重复，根本不会利用长 CoT。换到 32B 稠密、230B MoE、671B MoE 之后，纯 RL 的收益才显著涌现。想验证「从零 RL」，先把基座选够大。

**验证器必须可靠** 。R1-Zero 的成败系于奖励信号的保真度。规则奖励和「LLM 对照标准答案判对错」是两种抗 Reward Hacking 的可靠方案；但对于开放式写作这类「正确性」本身主观的任务，可靠的验证器仍是未解之题。

**SFT 与 RL 缺一不可** 。RL 负责探索人类示范之外的最优推理轨迹，没有 RL 就没有长链反思；SFT 负责奖励信号难以定义的任务（开放问答、创意写作）。只押一边都会翻车。

**Process Reward Model（PRM）没走通** 。过程奖励看似优雅，实则三大硬伤：通用推理中「一步」难以显式定义；判断中间步骤对错这件事本身和解题一样难，自动标注不靠谱、人工标注不可扩展；一旦引入模型化 PRM，Reward Hacking 随之而来，重训奖励模型又徒增成本。PRM 在对 top-N 回答重排序或辅助搜索时有用，但放在大规模 RL 训练回路里得不偿失。

**MCTS 也没走通** 。受 AlphaGo 启发，团队尝试用蒙特卡洛树搜索增强测试时计算，但 token 生成的搜索空间是指数级爆炸的，远超棋类；限制节点扩展深度又容易陷入局部最优。更致命的是，指导搜索的价值模型本身极难训练——AlphaGo 靠价值模型迭代变强的飞轮，在语言生成场景里转不起来。

## 八、总结与展望

回顾全文，这篇工作的核心要点可以浓缩为五条：

- **纯 RL 即可涌现推理能力** ：R1-Zero 不做任何 SFT，AIME Pass@1 从 15.6% 升至 77.9%，反思、验证、回溯等高级行为全部自发涌现；
- **GRPO 是关键工程杠杆** ：砍掉价值模型、组内相对优势、KL 进损失，让大规模长 CoT 强化学习在资源上可行；
- **四阶段流水线兼顾能力与体验** ：冷启动 → 推理 RL → 拒绝采样 SFT → 全场景 RL，修好可读性与语言混杂的同时保留推理能力；
- **评测方法论扎实** ：非零温度多次采样求平均、严格去污染、用训练后发布的新赛题（AIME 2025、AMC 12 2024）验证泛化，R1 达到美国数学奥林匹克资格线水平；
- **蒸馏民主化推理能力** ：1.5B 小模型蒸馏后数学超越 GPT-4o，32B 蒸馏版全面碾压同规模纯 RL 模型，全系列 MIT 协议开源。

局限同样明确：结构化输出与工具调用能力尚未跟上、简单题存在过度思考、中英文之外的语言会混杂、软件工程任务的 RL 还没铺开，以及最根本的——对写作这类难以构建可靠验证器的任务，纯 RL 路线依然无解。

展望一句话：DeepSeek-R1 证明了大模型推理的解锁钥匙不在海量人工标注，而在「足够难的题 + 可靠的验证器 + 足够的 RL 算力」这三件事上——凡是能被自动验证的任务，机器都有望通过试错迭代超越人类；而如何为那些不可验证的任务构造可信的奖励信号，将是下一阶段最值得投入的战场。

> 本文参考自 [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)