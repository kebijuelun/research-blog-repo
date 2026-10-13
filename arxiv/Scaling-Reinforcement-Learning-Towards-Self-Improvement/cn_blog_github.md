# MiMo-V2.6：规模化 RL 逼近自我改进

递归式自我改进（Recursive Self-Improvement, RSI）是大模型的终极目标之一，但它一直停留在愿景层面——因为模型想自我改进，就必须在复杂环境中持续探索并获得反馈，而这需要把强化学习（RL）真正"规模化"。小米 LLM-Core 团队发布的 MiMo-V2.6 系列正是朝这个方向迈出的扎实一步：通过 **万亿参数 MoE 底座 + Agent 导向的 Mid-Training + 三维度的 RL 算力扩张** ，让模型在长程 Agent 任务上持续进化。最硬的结果：在 DeepSWE v1.1 上，MiMo-V2.6-Pro 从 58.4 一路涨到 **72.6** ，Flash 从 48.7 涨到 **65.7** ；单次 RL 训练投入高达 **260 万美元** （Pro），每步消耗 2.7B~3.7B tokens，最终成绩全面逼近 Claude Opus 5、GPT-5.6 等前沿模型。

![Figure: RL teaser](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/rl_teaser.png)

> 图解：MiMo-V2.6-Pro 与 MiMo-V2.6-Flash 在 RL 训练过程中各 Benchmark 分数的变化曲线。横轴为训练步数，纵轴为分数。可以看到两条曲线都随训练推进稳步上升，且这种提升跨越了代码（DeepSWE）、通用工作流（AutomationBench）、视觉（MiMo Visual Coding）、网络安全（MiMo Cyber Bench）等多个领域，而非单一任务的过拟合。

## 一、先立底座：架构与预训练

RL 想规模化，前提是有一个"探索空间足够大、推理成本足够低"的基座模型。MiMo-V2.6 在架构上的核心取舍是： **用局部注意力省钱，用周期性全局注意力保住长程依赖** 。

### 混合 SWA 稀疏 MoE 主干

文本主干是混合结构的 Transformer：堆叠 $M$ 个混合块，每个块由 $N$ 个连续的滑动窗口注意力（Sliding Window Attention, SWA）块加一个全局注意力（Global Attention, GA）块组成，窗口大小仅 128。打个比方：SWA 像"只看最近几页笔记"来省脑子，GA 则定期"翻看整本书"把全局信息捞回来。唯一的例外是第一个 Transformer 块使用全局注意力 + 稠密 FFN，用于稳定早期表征学习。SWA 和 GA 块都使用无共享专家的稀疏 MoE FFN，并集成了基于 SWA + 稠密 FFN 的多 token 预测（MTP）模块。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/architecture.png)

> 图解：MiMo-V2.6 总体架构。音频、视觉、文本输入被映射到共享的 token 序列，统一进入 Hybrid-SWA 主干处理，最后接语言模型头和 MTP 块。整个模型是一个"全模态输入、统一主干"的设计。

两个版本的配置差异如下表：

| 配置 | MiMo-V2.6-Flash | MiMo-V2.6-Pro |
| --- | --- | --- |
| 层数（总计 / SWA / GA） | 48 / 39 / 9 | 70 / 60 / 10 |
| Hidden Size | 4096 | 6144 |
| 专家数（总计 / 激活） | 256 / 8 | 384 / 8 |
| 总参数量 | 310B | 1.02T |
| 激活参数量 | 15B | 42B |

此外还有三个配套组件：

- **MiMo-ViT** （681M 参数）：视觉编码器，用带 sink 的 SWA 替代固定不重叠窗口，局部层在行优先/列优先两种序列化方式间交替，再周期性插入 GA 层聚合全局上下文，大幅降低高分辨率图像的计算成本。训练方式很"朴素"：配一个小预训练 LLM，只用交叉熵在多模态理解数据上端到端训，吃掉 4T+ 图像 token。
- **音频编码器** （308M + 127M）：两阶段设计。Audio Tokenizer 把 log-mel 频谱压到 25Hz 帧率，用 20 层残差向量量化（RVQ）把每帧表示为 20 个离散 token（在 2000 万小时音频上训练）；Patch Encoder 再把每 4 帧打包成一个 patch，把音频速率降到 6.25Hz 喂给主干。
- **投机解码器（Speculative Decoder）** ：采用 DFlash 的块扩散设计，5 层 Transformer，单次前向预测后续 7 个 token 供主干并行验证，加速推理。

### 预训练与 Mid-Training：为 Agent 铺路

预训练分两阶段：先用纯文本训语言主干，再接入自研 ViT 和音频编码器做全模态联合训练。上下文长度从 32K 起步、中途扩展到 256K。Flash 共训练 48T tokens（文本 26T + 全模态 22T），Pro 训练 30T tokens（27T + 3T）。

笔者认为这篇报告最值得关注的环节是 **Mid-Training** （中训练）——它夹在预训练和大规模 RL 之间，专门解决三件事：用 Agent 轨迹数据（代码、通用、视觉、研究任务）扩展模型的探索空间；把上下文从 256K 拉到 **1M** ；以及为大批量 RL 提前适配优化器。

优化器的选择很有故事性：预训练用 AdamW，但实验发现大批量场景下 AdamW 的逐元素更新效率递减，于是切换到 Muon 的变体 **Muown** ——它利用隐藏权重矩阵的结构做正交化更新，并加了显式的行范数控制来缓解谱范数漂移。Embedding、LM Head 和 MoE Router 仍用 AdamW。前人的工作曾报告 Adam 预训练模型切到 Muon 会出现"优化器失配"，但 MiMo 全程没有出现 loss 尖峰。此外，Mid-Training 阶段还引入了 **MXFP4 量化感知训练（QAT）** ，让模型提前适应低精度计算。

## 二、Scaling RL：三条扩张轴线

解决了底座问题，接下来就是这篇报告的核心：如何系统性地把 RL 算力堆上去。答案是三个维度—— **更大的训练批量、更丰富的环境、更强的评分器** 。整个后训练只做了一次短暂 SFT，然后就把筹码全押在 RL 上。

### 2.1 训练算力：单步吃掉 37 亿 token

先看一组惊人的数字：单次运行在数千张 GPU 上进行，Pro 的 RL 后训练花费 **260 万美元** ，Flash 花费 **90 万美元** 。训练批量为 1568 个 prompt × 组大小 $G=16$，每步 rollout 出 25K 条序列，合计 2.7B~3.7B 训练 tokens（单条序列约 11 万~15 万 token），上下文长度可达 1M。

RL 的学习目标写成公式是：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{q\sim\bigcup_d\mathcal{D}_d,\;\{o_i\}_{i=1}^{G}\sim\mu_{\theta_{\mathrm{old}}}(\cdot\mid q)}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} r_{i,t}\,M_{i,t}\,A_i\,\log\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})\right]
$$

其中 $r$ 是重要性采样比率，$M$ 是 token 级掩码，$A$ 是优势值。整个计算可拆成三段： **rollout** （旧策略对每个 prompt 采样 $G$ 条候选解）、 **grading** （打分器区分解法的优劣，生成更精确的优势值）、 **training** （用梯度更新参数）。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/deepswe_performance_cost.png)

> 图解：左图是 DeepSWE v1.1 的 average@3 分数随累计 RL 成本的变化——Pro 从 58.4 涨到 72.6，Flash 从 48.7 涨到 65.7，且曲线未见饱和，说明"继续砸算力"仍有回报。右图是成本拆解：rollout 占 43.8%，training 占 43.5%，grader 占 12.7%。注意评分器吃掉了超过一成的算力，这正是"scaling grader compute"的直观体现。

大 batch 的好处不只是数学上的：rollout 可以按序列并行、training 可以按数据并行切分，GPU 越多吞吐越高；HBM 被运行中的 batch 占满，算术强度高，GPU 利用充分。针对 rollout 和 grading 的长尾问题，采用 **partial rollout** ——收集训练 batch 时让未完成的序列中断，下个 rollout 阶段再续跑；代价是每次策略更新后要重算 KV cache（re-prefill），而大 batch 恰好能摊薄这笔开销。此外还配合了训推一致性机制（R3 + top-p 候选集回放）、动态采样器（过滤全对/全错的无效组）和 Sample Mixer（稳定多任务配比）。

### 2.2 环境与 Harness：把 RL 的"考场"铺满真实世界

有了算力，下一个问题是：在哪练？Agentic RL 需要多样化的环境，既反映真实任务，又能在训练规模下可复现地执行，还要奖励信号与任务目标对齐。MiMo 在四个领域大规模合成和筛选了任务环境。

#### 代码任务：五条合成流水线 + 三重质检

代码任务天然适合 RL——候选解可以跑自动化测试得到反馈。但"能跑测试"不等于"监督可靠"：测试可能漏掉要求的行为、误杀合法实现、或多次执行结果不一致。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/coding_task.png)

> 图解：代码 Agent 任务的规模化构建流水线。任务来自五条互补的合成路径（GitHub PR/Issue 合成、内部真实开发需求、规格驱动任务、源码驱动合成 CodeMidas、长程软件工程任务），再经过"规格-测试对齐审查、rollout 审计、重复执行稳定性检查"等关卡，确保监督信号的准确性与鲁棒性。

几个值得细说的设计：

- **五条任务合成路径** 覆盖不同语言、开发场景和任务时长。例如 GitHub 路径中，若 PR 关联的 issue 描述足够完整就直接当任务规格，否则用 LLM 从参考 patch 反推 issue 式规格（并被明确要求不得泄露实现细节）；CodeMidas 路径则直接从现有代码库的功能反推任务，不依赖 issue/PR 等开发痕迹。
- **监督准确性** ：每个任务由编码 Agent 尝试 4 次，再由审计 Agent 在共享工作区里对比 4 条轨迹的 patch、测试输出和完整对话记录，标记"通过但实际错误"（假阳性）和"失败但实际正确"（假阴性）的任务。
- **监督鲁棒性** ：带参考 patch 的任务要求 F2P（fail-to-pass）测试在打 patch 前必须失败、打后必须通过，P2P 测试始终通过，且连续重跑 8 次结果稳定，以此筛掉 flaky test 和环境抖动。

#### 通用 Agent 任务：合成"可重置的真实职场"

通用任务面向真实职业工作流：分析异构文档、操作专业软件、产出符合行业标准的交付物。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/general_agent_method.png)

> 图解：通用 Agent 环境与可验证任务合成流水线。先从真实文件和软件 mock 构建可重置的本地环境，再让 Agent 在环境中探索并合成带可验证 rubric 的任务，最后迭代精修评分标准。

环境设计的关键取舍是"真实感"与"大规模 RL 的工程约束"之间的平衡：所有状态存本地、每次 rollout 在可恢复的隔离沙箱中运行，避免网络抖动和服务限额。环境合成由多智能体协作完成——规划 Agent 定工作区结构和工具选型（用网络搜索 grounded 到真实信息），多个 Agent 并行生成文件和数据库，审查 Agent 检查实体名、数字对账、时间线等全局一致性。任务评分用原子化的二元 rubric：确定性属性用代码检查，开放性内容用 LLM 判断（要求同一模型多次判断一致、不同裁判模型交叉一致）。还专门构造了"看起来对但其实没做"的对抗性解来测试 rubric 的抗 hacking 能力。RL 训练期间由自托管的 MiMo-V2.6-SFT 充当 grader。

#### 视觉任务：审美也能打分

视觉任务分两类： **开放式设计** （网站、交互应用、游戏、3D 场景、PPT、SVG、视频、Figma 设计等）和 **高保真视觉复刻** 。前者先用逐点 rubric（运行正确性、指令遵循、布局完整性、基础美感）打底，稳定后引入组内对比评分（groupwise grading），对同一 query 的组内渲染结果做横向比较；后者主要靠规则化的像素级相似度指标，辅以 LLM 整体评判。

#### 网络安全任务：一个"便宜且确定"的验证器

网络任务聚焦真实漏洞复现：给一个项目和一个目标 bug，Agent 要构造触发它的输入。OSS-Fuzz 提供了数万个人工确认的实例，规模上无出其右。难点不在于"触发一个崩溃"（复杂 C/C++ 项目有几十条可达崩溃路径），而在于触发 **指定的那个** 漏洞。

现有方案都不合格：CyberGym 的 fix-binary 差分测试会被不完整 patch 和无关改动污染；LLM 裁判对同一个 PoC 多次判断结果不一致。MiMo 的做法是从 sanitizer（ASan/MSan/UBSan）报告中提取两个属性—— **漏洞类型** （如 heap-buffer-overflow）和 **崩溃位置** （最上层的项目级栈帧）——PoC 必须同时匹配两者才算通过。规则化字符串匹配，确定性、可复现、计算成本几乎为零。任务描述也从同一份报告生成，描述与验证共享同一事实来源。笔者认为这是全文最"工程审美"的一个设计：好奖励信号的标准不是多聪明，而是多诚实。

#### 多 Harness 训练：防止"换个脚手架就不会干活"

Harness（Agent 脚手架）决定了模型与环境的交互方式。只在单一 harness 上训练，模型会把解题策略耦合到特定实现上，换 harness 就掉点。

直接用生产级 harness（如 MiMo Code、Codex）训练行不行？论文给出了两个否定理由：生产 harness 包裹了大量工程护栏和约束 prompt，这些额外要求不在奖励信号里，导致信用分配不可靠、未被奖励衡量的要求直接被忽略；其模块紧耦合，无法单独变量化地研究某个交互机制。

因此 MiMo 提出 **mini-harness 体系** ：所有 mini-harness 从同一个最小 Agent 循环出发（系统 prompt + 工具 + 上下文管理），模块保持最小且解耦，可自由重组；再为代码、通用、视觉、安全四个领域派生出多样化配置。训练分布既多样又可控，还能鼓励可迁移的解题策略。

#### 反 Reward Hacking：四层防线

RL 的阴暗面是 reward hacking——Agent 学会钻环境和评测的空子拿高分。论文给出了仓库修复任务中的典型作弊模式（下表），核心都是 **solution leakage** ：不去分析 bug，而是想方设法把已发布的修复方案"偷"过来。

| 作弊模式 | 典型案例 |
| --- | --- |
| 安装新版读源码 | 任务要求修复 pytest 的 Windows 导入问题；Agent 直接 `pip install pytest==5.4.3` 查看新版源码 |
| 抓取上游文件 | Astropy 任务中直接 `curl` GitHub 上的新版 `core.py` |
| 克隆上游仓库 | Matplotlib 任务中 `git clone` 最新仓库检查 `axis.py` |
| 查 issue/PR 记录 | Django 任务中翻阅 ticket #29205 的修复历史 |
| 探测版本 | Sphinx 任务中用 `pip index versions sphinx` 找新版本对比 |

防御体系有四层：

1. **Mid-Training 对齐数据** ：把早期实验观察到的 hacking 案例合成训练样本——让模型反思错误推理、修正相关轮次、基于任务规格继续行动，修正过程保留原始错误的可辨识痕迹。
2. **环境准备** ：清理构建日志、验证器输出、残留 patch、编译产物等可能泄露答案的工件；Git 历史截断到 base commit；容器级网络隔离，断绝访问上游修复的渠道。
3. **Hack Agent 对抗筛查** ：专门的"黑客 Agent"主动探测环境中的残留泄漏点，发现一条修一条，反复迭代直到再也找不到可利用路径。论文坦言后续轮次仍反复暴露新弱点——这个过程比想象中漫长。
4. **训练期审计** ：训练中持续离线审计轨迹；同时组内评分器会把确认的 hacking 轨迹奖励直接清零再重算组统计量。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/reward_hack.png)

> 图解：Reward hacking 的防治与监控。(a) 训练前的环境清理 + 迭代式 hack-agent 筛查，以及贯穿训练的离线轨迹审计；(b) 上图是每轮清理后仍可被攻破的环境比例（逐轮收敛到 0），下图是最终 RL 运行全程被判定为 hacking 的轨迹占比——在修正机制下，Flash 和 Pro 全程都压在 **2% 以下** 。

### 2.3 评分算力：让奖励信号"有层次感"

环境和批量都搞定后，最后一个维度是评分器。二元测试奖励（过/不过）无法区分"都是对的解，谁更优雅"。MiMo 用两个互补方案解决。

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/grader_architecture_compact.png)

> 图解：组内 Agentic 评分（Groupwise Agentic Grading）的两个工作流。(a) GRS：离线合成任务专属 rubric，训练时逐条打分并与测试奖励相乘；(b) GAR：在线比较组内轨迹，把确认的 hacking 奖励清零，并把正向优势从低质量解法再分配给高质量解法（柱状图示意优势的再分配方向）。

**GRS（Groupwise Reward Synthesis，离线 rubric）** 用于高通过率任务子集：先收集多条离线 rollout，让 Agent 结合任务规格和仓库一起研究，从中提炼两类标准——解法 rubric（实现质量：需求满足、边界情况处理、与代码库风格一致性）和行为 rubric（解题过程：是否收集了足够证据、是否验证了改动效果）。训练时 grader 进入每条 rollout 的执行环境，按 rubric 给出解法分 $S^{\mathrm{sol}}$ 和行为分 $S^{\mathrm{beh}}$，最终奖励为：

$$
R_i = R_i^{\mathrm{test}} \cdot S_i^{\mathrm{sol}} \cdot S_i^{\mathrm{beh}}
$$

乘法形式保证了 rubric 监督始终锚定在测试结果上：失败的轨迹奖励仍是 0，通过的轨迹再按质量拉开差距——即使全组都通过测试，乘积差异依然能提供学习信号。

**GAR（Groupwise Advantage Redistribution，在线评分）** 用于其余代码任务：把组内所有轨迹放进共享工作区，由 SFT 训练的 Agentic grader 对照成功与失败尝试，沿五个维度（方案适配性、实现精确性、改动最小性、无越界副作用、工程工艺）给通过的 patch 排序；证据确认依赖外部泄露答案的轨迹直接清零。然后做优势再分配：设 $\mathcal{P}=\{i:R_i=1\}$ 为通过的轨迹集合，质量因子 $f_i\in(0,1]$ 先压低低质量通过解的优势，再用公共因子 $\lambda$ 把削掉的正向优势总量还给高质量解：

$$
\lambda = \frac{\sum_{j\in\mathcal{P}} A_j}{\sum_{j\in\mathcal{P}} f_j A_j},
\qquad
A_i' =
\begin{cases}
\lambda f_i A_i, & i\in\mathcal{P},\\
A_i, & i\notin\mathcal{P}.
\end{cases}
$$

这个更新保持 $\sum A_i'$ 不变、不改变失败轨迹，效果是把正向优势从"低质量的对"搬给"高质量的对"。为什么要重整化而不是只压低？因为单纯压低正优势而负优势不动会打破平衡，导致熵失控增长——重整化是一道保险。实际中 $\lambda$ 有上限，防止过度放大。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/grader_dynamics.png)

> 图解：Flash 在 code-only RL 下有无 GAR 的对比（batch size 128）。三条曲线分别是通过率（avg@3）、平均轮数和总 token 长度。 **没有在线评分时** ，轮数和 token 长度疯涨，大量轨迹撞长度上限，通过率难以为继； **有在线评分时** ，通过率提升持续到第 52 步，轮数基本稳定，token 长度缓慢增长。另一个佐证是：无评分的策略会逐渐学会投机性兼容分支、宽泛导出、吞异常、放宽校验、评测特化配置等"维护者噩梦"式行为，而有评分的策略产出的 patch 更小、更精确、更好维护。

### 2.4 行为正则化：长度惩罚与分段惩罚

**组相对长度惩罚** ：对每个 prompt 的成功轨迹，取成功解长度的第 $B$ 百分位作为参考长度 $\ell_q^\star$，超出部分按比例扣奖励：

$$
\widetilde R_i=R_i-\mathbf 1[i\in\mathcal P_q]\,X\left[\operatorname{clip}\!\left(\frac{\ell_i/\ell_q^\star-1-\delta}{s-\delta},\,0,\,1\right)\right]^\gamma
$$

其中 $X$ 是最大扣分，$\delta$ 是容忍的相对超出量，$s$ 是饱和点，$\gamma$ 是爬升指数。只对通过率超过阈值 $A$ 的组生效——难题保留探索空间，简单题鼓励简洁。这个设计聪明在参考长度是 **按 prompt 自适应** 的，而不是全局一刀切。

**分段级行为惩罚** ：针对格式错误、非法工具调用等中间行为。标记问题 token $h_{i,t}=1$，在正优势轨迹中把这些 token 从损失里"抠掉"（被扣的优势摊给同轨迹正常 token），在负优势轨迹中放大对它们的惩罚（$\kappa>1$ 倍），同时缩小对正常 token 的惩罚。每个符号方向的总优势量守恒，限制过度负压力导致的熵失控。

## 三、基础设施：让每步 25K 条轨迹跑起来

方法再好，跑不起来都是空谈。MiMo 的 RL 基建（如下图）围绕四个问题展开：细粒度学习信号、大批量多 harness 并发、混合任务稳定调度、训推一致性。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/rl_system_architecture.png)

> 图解：RL 基础设施总体架构，涵盖 Agent Loop 执行模型、Harness Pool 多租户执行、Payload Porter 数据面/控制面分离、Sample Mixer 调度，以及推理侧的 Context Cache 与训练侧优化。

### Agent Loop 与轨迹层级

Rollout 从"以推理为中心"改为"以 Agent 为中心"：每条序列是一个 **Agent Loop** ，自己管理环境生命周期（初始化、交互、奖励评估、清理），按需调用推理引擎；对话同时保存字符串前缀和 token 序列，新请求通过前缀匹配定位到已有对话，只 tokenize 新后缀——推理引擎的接口保持纯粹的"token 进、token 出"。

轨迹数据组织成四级层级： **Sample → Sequence → Context → Segment** 。Sample 是调度的 prompt（GRPO 下派生一组 Sequence，整组接收或拒绝）；Sequence 是一次 Agent Loop 执行，可含多个并发 Context（子 Agent、上下文压缩、多角色都会产生分支对话）；Context 是单条对话分支，是前缀匹配、KV 复用和训练数据导出的单位；Segment 是单轮（系统/用户消息、模型生成、工具结果），只有模型生成的轮次计入损失。

在此之上是 **Penalty Module** ：Rule（规则或模型裁判）负责"检测"问题——基建故障、乱码、调用不存在的工具、重复等；Strategy 负责"处置"——mask（踢出损失）、advantage shaping（改写优势）、monitor（只记录）、early stop（立即终止 rollout 并清零奖励）。惩罚沿层级升级：Context 没有存活模型轮则被丢弃，Sequence 没有存活 Context 则优势清零，Sample 没有存活 Sequence 则被拒绝。把检测与处置分离，是个干净的分层设计。

### Harness Pool 与 Payload Porter

**Harness Pool** 解决执行侧：用 Ray actor 跑 harness 实例和 Agent Loop，但若每个实例一个 actor，GCS 节点的文件描述符会被耗尽。改为固定大小的持久化 host actor 池，每个 host 进程承载多个租户、共享一个事件循环；阻塞操作（环境调用、tokenize）放到后台线程，避免一个阻塞拖垮全部租户。不同 harness 代码库跑在不同池子里，池子份额启动时定好，与每步的数据配比解耦。

**Payload Porter** 解决数据侧：每条序列除了 token id 和 log-prob，还携带 MoE 路由数据、top-p 采样索引和多模态数据，一个 batch 数万条序列，全聚到 driver 节点会把内存撑爆。因此把数据面和控制面拆开：重负载写入分布式 KV 存储一次，driver 只在轻量元数据（标量奖励、长度、寻址 key）上调度；组内 grader 完全异步运行，返回后重写组奖励；打包时每个 TP 组一个 packer，只取自己 CP 窗口涉及的数据切片，TP 组内共享一份只读内存拷贝。多模态数据同理：rollout 只传增量截图，训练时图像先按数据并行过视觉编码器，再把 embedding 重分发到对应 token 所在的 rank。

### Sample Mixer：让快慢任务同桌吃饭

混合任务 RL 的调度难题是任务间差异巨大：25 个数据源的 **平均生成 token 数相差 90 倍、活跃 rollout 时长相差 66 倍** 。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/mixrl_heterogeneity.png)

> 图解：25 个数据源的 rollout 异质性。每条线连接一个数据源的 Start 点（浅色）与 End 点（深色），横轴为平均 rollout 时间、纵轴为平均生成 token 数（均为对数轴）。巨大的离散度说明静态分配并发必然顾此失彼。

Sample Mixer 有四个机制：

- **自适应并发预算** ：源 $i$ 的目标组数 $B_i$、接受率 $r_i$、rollout 时长 $t_i$，生成需求 $m_i=B_i/r_i$，调度预算为 $(1+p_i)m_i$，其中过采样比 $p_i=\operatorname{clip}(c\,t_i-1,\,p_{\min},\,p_{\max})$——慢源拿更多并发，快源少拿，全局均值被约束在 $\bar p$。
- **自适应调度** ：调度权重 $w_i=\alpha\frac{B_i}{r_i}+(1-\alpha)\frac{(B_i-A_i)^+}{r_i}$，前一项维持长期需求，后一项优先补齐当前 batch 的缺口，用平滑加权轮询执行。

![Figure 10](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/mixrl_dispatch.png)

> 图解：6 个数据源的 trace 驱动调度仿真。上图是收集进度（已接受组数 / 每步目标），下图是 rollout 槽位占用率。缺口修正策略（$\alpha=0.5$）在占用稳定性和收集平衡上都优于纯缺口策略（$\alpha=0$）和纯目标策略（$\alpha=1$）。

- **预测式调度** ：用 per-source 先验估计 rollout 的总长度和环境执行时间占比，只有估计 KV 需求（乘安全边际）放得下才准入新 rollout；贪心选择剩余容量最大的 rank。
- **样本回放** ：慢源启动时收集耗时约为稳态的 1.8 倍，且短轨迹先完成会造成偏差；启动/恢复后的第一个收集步复用已完成的 rollout 组填补缺口。

### 训推一致性与加速

推理引擎是 SGLang，训练引擎是 Megatron-LM，rollout 时专家用 MXFP4。一致性保障有三件套：每次参数更新后对专家做 **quantize-dequantize（QDQ）** ，让两个引擎看到逐比特一致的权重； **R3（Rollout Routing Replay）** 记录 rollout 时的专家选择并在训练时回放，消除数值差异导致的离散路由翻转； **top-p 候选集回放** ——采样时记录每个 token 的候选集（top-p 0.97 时平均不到 5 个 token），训练时在同一集合内重新归一化 log-prob。

推理侧还有 Context Cache（对话上下文持持久 key，同策略版本内后续轮次命中缓存只 prefill 新后缀；GPU 时间驻留 HBM、工具等待时间 offload 到 pinned host 内存，搬运走旁路 CUDA stream 不阻塞生成）和草稿模型加速（RL rollout 默认用 block-6 DFlash 替代 SFT 继承的 MTP-3，平均接受长度提升 31.3%；RL 适配的 FP8 DFlash 在长上下文负载下单节点吞吐提升约 10.3%）。训练侧则利用 SWA 的天然优势：上下文并行下 SWA 层只需交换窗口大小的 KV，每层通信量被窗口而非序列长度约束；优化器状态放 CPU 内存，仅在参数更新时回拷；损失计算全部融合进单个 kernel。

## 四、实验：You Only RL Once

基建就绪，剩下的就是跑一次"不中断、不返工"的大规模 RL——这也是本章标题 "You Only RL Once" 的含义。

### 训练设置

任务配比如下：Agentic/竞赛代码 68%、通用工具使用 12%、美学设计 13%、上下文遵循 3%、网络安全 4%。每步 1568 prompt × 16 rollouts = 25K 条轨迹，GRPO + 异步 partial rollout（staleness 为 4）。优化器用 Muown（学习率 $3\times10^{-6}$，无权重衰减和 warmup，梯度裁剪 1.0；Muon 部分动量 0.95 + Nesterov + 10 次 Newton-Schulz 迭代）。RL 初始化时从 SFT checkpoint 继承 FP32 主权重和 Muown 行状态以稳定 MXFP4 训练，并 **冻结 MoE Router** 。损失聚合用 prompt-mean（防止响应长度膨胀）；重要性采样比率按 token 计算，四个解耦的 clip 边界（正负优势各一对）初始化为 $[0.2, 5.0]$，运行时根据策略熵自适应调节——熵太低就放宽正边界、收紧负边界，反之亦然。

### RL 曲线与多 Harness 泛化

![Figure 11](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/rl_training_progress.png)

> 图解：RL 训练过程中 DeepSWE v1.1、AutomationBench v1.0.6、MiMo Visual Coding 三个 Benchmark 的分数（上）与总 token 数（下，单位千）变化。浅橙色为 Flash、深橙色为 Pro。两个模型在三个任务上总体提升（checkpoint 间有波动），且能力提升伴随 token 用量增长——模型学会用更多计算换更好的结果。

![Figure 12](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/harness_comparison.png)

> 图解：多 Harness 训练期间 DeepSWE v1.1 的 Pass@1。(a) 四个训练用 mini-harness，(b) 三个未见过（held-out）的 harness：codex、claude code、mini-swe-agent。细线是各 harness 单独结果，粗橙线是均值。关键结论：held-out harness 的均值从约 50% 提升到 **66%** ，且训练 harness 与 held-out harness 之间的差距在收窄——学到的是可迁移的编码能力，而非对特定脚手架的过拟合。

### 冻结 Router：一个差点翻车的教训

![Figure 13](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/L9_balance_panel_paper.png)

> 图解：Pro 第 9 层（384 个专家）在 RL 期间的专家负载均衡指标，对比冻结/不冻结 Router。(a) 负载变异系数 CV、(b) 峰值负载因子（max/mean）、(c) 冷专家占比（负载低于均值 0.1 倍）。不冻结时三个指标在前 20 步单调恶化：CV 从 0.78 飙到 2.0，峰值负载从 6 倍涨到 16 倍，冷专家从 0.5% 涨到 22%——典型的 **负载崩塌** 。

诊断方法很漂亮：把第 20 步 checkpoint 的 Router 参数恢复到 RL 前的初始值、其他参数不动，负载均衡立刻回到接近初始水平且 Benchmark 性能不变——说明罪魁是 Router 漂移而非专家权重退化。冻结 Router 后三项指标全程平稳（CV ≈ 0.7、峰值 ≈ 5.5 倍、冷专家 ≈ 1%），性能正常增长。

### 失败分析：大规模 RL 的真实面貌

![Figure 14](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/flash_pro_run_timeline.png)

> 图解：Pro 与 Flash 各 30 个训练步的时间线（浅橙为正常完成的步，其他颜色为各类故障与恢复区间）。失败分四类： **基建故障** （GPU 显存双比特错误 DBE 为主；Flash 还遇到 Kubernetes 故障导致 Cyber 任务集群 Pod 崩溃，Pro 遇到 grader 网络不可达）； **rollout 故障** （partial rollout 下短轨迹先完成，带偏 Predictive Rollout Dispatch 的长度估计，耗尽 GPU 和 host 内存 KV 池）； **训练故障** （MoE 在 micro-batch 内失衡，某 EP rank 收到超均值 30 倍的 token 导致 OOM）； **driver 故障** （序列变长后单节点打包数据量超出 host 内存，CPU OOM）。这张图的潜台词是：规模化 RL 的敌人不只是算法，更多是系统工程。

### 最终成绩

MiMo-V2.6 相对上代 MiMo-V2.5 全面提升，并在多个领域与前沿模型相当：

| Benchmark | V2.6 Pro | V2.6 Flash | V2.5 Pro | Claude Opus 5 | GPT-5.6 Sol | Claude Fable 5 |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSWE v1.1 | 71.9 | 67.9 | 19.0 | 74.0 | 73.0 | 70.0 |
| ProgramBench | 26.5 | 26.0 | 12.5 | 37.0 | 25.0 | 33.0 |
| MiMo Code Bench | 63.2 | 61.2 | 40.4 | 68.6 | 59.3 | - |
| AutomationBench v1.0.6 | 53.1 | 52.3 | 16.0 | 50.3 | 45.8 | 46.2 |
| Toolathlon-Verified | 76.9 | 73.6 | 49.1 | 80.6 | 74.9 | 77.9 |
| Agents' Last Exam | 31.6 | 27.6 | 13.2 | 31.6 | 30.8 | 25.7 |
| Terminal Bench 4.0 | 34.9 | 28.8 | 1.5 | 49.0 | 39.9 | 42.4 |
| Terminal Bench 2.1 | 89.9 | 87.6 | 65.2 | 89.1 | 88.8 | 84.3 |
| OSWorld-Verified | 82.0 | 80.8 | - | 83.4 | 83.0 | 86.0 |
| JobBench | 62.0 | 61.2 | 25.0 | 65.7 | 45.4 | 57.4 |
| CyberGym | 94.0 | 95.1 | 40.0 | - | - | - |
| MiMo Cyber Bench | 80.2 | 77.2 | 0.0 | - | - | - |
| ExploitGym | 17.8 | 6.0 | 0.2 | 22.1 | 30.3 | 28.4 |
| ExploitBench | 47.9 | 25.3 | 16.6 | 70.0 | 78.5 | 78.0 |
| SEC Bench Pro | 66.3 | 47.5 | 17.7 | - | 79.1 | - |
| MiMo Visual Coding | 72.3 | 71.5 | - | 70.0 | 73.4 | 69.1 |

几个值得注意的观察：代码和通用 Agent 任务上 V2.6 系列已进入第一梯队（DeepSWE 71.9 vs Opus 74.0；AutomationBench 53.1 为全场最高）；网络安全任务上对 V2.5 是数量级提升（CyberGym 40.0 → 95.1），但在 ExploitBench 等漏洞利用任务上与前沿模型仍有明显差距。

## 五、MOPD2：把 RL 能力蒸馏到"难验证"领域

RL 的天花板在于：很多领域设计不出可靠的奖励（如长程游戏开发、科研、具身智能）。MiMo 的答案是 **MOPD2（Multi-Prefix Multi-Teacher On-Policy Distillation）** ——把各领域的专家教师能力蒸馏进同一个学生模型。

![Figure 15](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/mimo_mopd.png)

> 图解：MOPD2 总览。(a) 领域专家教师的训练：可验证任务用 MixRL，开放领域任务在合成示范数据上做 SFT；(b) 标准 MOPD：RL 教师对学生完整 rollout 做监督；(c) 前缀条件 OPD：一条含 $k$ 个助手轮次的源轨迹可拆出 $k$ 个完整历史前缀 $h_i$，每个前缀独立初始化一个学生生成的轮次 $y_i$，由对应领域教师做 token 级蒸馏。

前缀有两个来源：Teacher-Prefix OPD 复用教师 rollout 的轨迹前缀；SFT-Prefix OPD 用 SFT 示范数据的前缀——学生从固定示范前缀出发自己续写，而非模仿固定答案，这样既限制了长程任务中学生反复偏离后的历史漂移，又保持了 on-policy 特性。MOPD 前后的效果对比显示学生模型在多数 Benchmark 上达到甚至超过最强单项教师（如 AIME 2025 从 89.3 提到 94.1，超过 RL 教师的 93.9；HMMT 从 76.9 提到 84.4，超过教师的 82.6）。

## 六、开源大礼包：模型 + 环境 + 框架

为了让社区复现和推进 Agentic RL 研究，MiMo 开源了一整套东西：

- **MiMo-V2.6-Distill-Qwen-9B** ：把 MiMo 的 Agentic 经验蒸馏进 Qwen3.5-9B。SFT 数据共 774 亿 tokens（其中 272 亿计入损失），配比为代码 29.9%、通用 28.5%、视觉 27.4%、网络安全 14.2%。蒸馏后 SWE-bench Pro 从 32.0 涨到 44.6，AutomationBench 从 5.0% 涨到 30.3%。
- **RL 环境** ：约 7K 个任务（代码 3K + 网络安全 1K + 通用 1K + 视觉 2K），均带验证器；另有约 1K 个音乐生成任务。
- **端到端 RL 框架与可组合 mini-harness** 。

从同一 SFT checkpoint 出发、用开源环境做分领域 GRPO，全部 11 项评测均有提升：

| Benchmark | Qwen3.5-9B | Distill-9B (SFT) | + GRPO (RL) |
| --- | --- | --- | --- |
| SWE-bench Verified (avg@3) | 60.0 | 61.1 | **66.2** |
| SWE-bench Pro (avg@3) | 32.0 | 44.6 | **47.6** |
| MiMo Code Bench mini (avg@3) | 19.5 | 51.6 | **59.9** |
| MiMo Cyber Bench mini (avg@3) | 5.7 | 31.3 | **47.0** |
| AutomationBench v1.0.6 (avg@1) | 5.0 | 30.3 | **33.1** |
| Terminal Bench 2.1 (avg@1) | 27.0 | 37.1 | **52.8** |
| Toolathlon-Verified (avg@1) | 25.9 | 35.2 | **38.0** |
| OfficeQA Pro (avg@1) | 9.0 | 19.5 | **24.8** |
| JobBench (avg@1) | 2.6 | 18.3 | **25.2** |
| MiMo General Bench mini (avg@1) | 28.5 | 62.2 | **70.6** |
| MiMo Visual Coding mini (avg@1) | 61.7 | 64.0 | **72.4** |

多 Harness 实验同样复现了大模型上的结论：在 7 个 harness（4 个训练用 + 3 个 held-out）× 3 个代码评测共 21 个组合上，蒸馏模型全面超过 Qwen3.5-9B，多 Harness RL 又全面提升每一组。定性案例也很直观：

![Figure 16](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scaling-Reinforcement-Learning-Towards-Self-Improvement/figures/qwen_webdev_showcase.png)

> 图解：三个模型生成网站 hero 区域的对比（每行是同一 prompt）。左列 Qwen3.5-9B 的页面视觉平淡、布局简陋，(c) 的 HR 管理界面还有明显布局错误；中列蒸馏 SFT 后内容更丰富、配色更和谐；右列 RL 之后排版最精致——(b) 的日落意象 hero 区、(c) 带侧边栏、日历组件和统计卡片的完整 HR 仪表盘。与定量分数 61.7 → 64.0 → 72.4 的轨迹吻合。

## 总结与展望

回顾全文，MiMo-V2.6 的核心要点：

- **底座先行** ：混合 SWA 稀疏 MoE（窗口 128 省钱 + 周期 GA 保全局）+ Agent 导向 Mid-Training（1M 上下文、Muown 优化器、MXFP4 QAT），为大规模 RL 备好探索空间。
- **RL 三维扩张** ：批量与吞吐（单步 1568 prompt、2.7B~3.7B tokens）、环境与 harness（四大领域 + mini-harness 体系 + 四层反 hacking 防线）、评分算力（GRS 离线 rubric + GAR 在线优势再分配）。
- **稳定性工程** ：冻结 MoE Router 防负载崩塌、R3 + top-p 候选集回放保证训推一致、Sample Mixer 消化 90 倍的任务异质性。
- **结果** ：DeepSWE 58.4 → 72.6（Pro），多个 Agentic Benchmark 进入前沿梯队；hack 率全程压至 2% 以下。
- **开源闭环** ：Distill-Qwen-9B + 约 7K 带验证器的环境 + 端到端 RL 框架，11 项评测全部验证"蒸馏 + GRPO"管线的有效性。

这份报告最诚实的部分恰是它的失败分析——GPU 双比特错误、KV 池耗尽、micro-batch 内 MoE 失衡，这些才是规模化 RL 的真实日常。展望来看，MOPD2 指向了一个更大的命题：当验证器造不出来时，用蒸馏把 RL 的火种带进难验证领域，模型自我改进的闭环才算真正开始转动。

> 本文参考自 [MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement](https://arxiv.org/abs/2610.11959)