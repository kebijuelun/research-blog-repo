# SayCan：让大模型在机器人上「量力而行」

大语言模型（LLM）肚子里装满了世界知识，但它有一个致命短板： **没有身体，不懂物理世界**。你问它「我把可乐洒了，怎么办」，它会回答「你可以用吸尘器吸干净」——说得头头是道，可对一台既没有吸尘器、也不会用吸尘器的厨房机器人来说，这等于废话。Google Robotics 与 Everyday Robots 的这篇工作提出了 **SayCan**：让 LLM 负责「说」（Say，判断哪个技能对完成任务有用），让预训练技能的价值函数负责「能」（Can，判断这个技能在当前状态下能不能做成），两者相乘选出下一步动作。在真实厨房的 101 条指令评测中，PaLM-SayCan 取得了 **84% 的规划成功率** 和 **74% 的执行成功率**，执行表现相比未做可供性接地的基线几乎翻倍。

![SayCan 核心动机](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/intro.png)

> 图解：左半边展示未接地的 LLM 的窘境——面对「我洒了饮料，能帮忙吗」，LLM 给出「用吸尘器」「叫清洁工」等对这台机器人毫无意义的回答；右半边是 SayCan 的回答：「1. 找到海绵，2. 捡起海绵，3. 拿给你，4. 完成」，并且机器人真的在厨房里执行了这套动作。差别就在于后者把语言模型「接地」到了机器人真实的技能与场景之中。

## 提出问题：LLM 的知识如何变成机器人的行动？

LLM 从海量网页文本中学到了大量关于日常任务的知识——怎么清理污渍、怎么准备午餐。直觉上，这些知识对机器人价值巨大。但问题是，LLM 从未与物理世界交互过，也没观察过自己输出的后果，因此它给出的计划在具体的机器人、具体的环境下往往不可行，甚至不安全。

有人会说：用 Prompt Engineering 让 LLM 把高层指令拆成子任务不就行了？文章指出这不够——拆分必须同时考虑 **机器人会什么** 和 **环境现在是什么状态**。同样是「给我拿个苹果」，视野里没有苹果、和机器人手里已经拿着苹果，最优的下一步技能完全不同。

于是本文要解决的问题可以精确表述为： **如何把 LLM 中封存的海量语义知识提取出来，用于驱动具身智能体完成高层的、时间跨度长的自然语言指令？**

## 预备知识：两个关键概念

在讲方法之前，先补齐两块拼图。

**Large Language Models**。语言模型对文本序列 $W = \{w_0, w_1, \dots, w_n\}$ 建模概率 $p(W)$，通常用链式法则分解为：

$$
p(W) = \prod_{j=0}^{n} p(w_j \mid w_{<j})
$$

本文主要使用 540B 参数的 PaLM 模型，并在消融实验中对比了 PaLM 8B/62B 与 137B 的 FLAN。

**价值函数与强化学习**。在马尔可夫决策过程（MDP）$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ 中，时序差分（TD）方法学习状态-动作价值函数 $Q^\pi(s, a)$，优化目标是：

$$
L_{TD}(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \left[ R(s,a) + \gamma \, \mathbb{E}_{a^* \sim \pi} Q_\theta^\pi(s', a^*) - Q_\theta^\pi(s,a) \right]
$$

这里有一个非常巧妙的洞察：在不打折、稀疏奖励的设定下（任务成功得 1 分，失败得 0 分），RL 训出来的价值函数 **天然就是一个可供性函数（Affordance Function）**——它的数值正好表示「这个技能在当前状态下成功的概率」。这就是 SayCan 把 RL 和 LLM 缝在一起的理论支点。

## 核心方法：Say 决定「想做什么」，Can 决定「能做什么」

### 问题定义与概率分解

系统输入一条用户指令 $i$（可以很长、很抽象、很模糊），以及一个技能集合 $\Pi$。每个技能 $\pi$ 配有三样东西：一个底层策略、一段简短的语言描述 $\ell_\pi$（如「find a sponge」）、以及一个可供性函数 $p(c_\pi \mid s, \ell_\pi)$——表示在状态 $s$ 下执行该技能成功的概率。直白地说，这个概率回答的是：「如果我让机器人做 $\ell_\pi$，它能做成吗？」

LLM 提供的是另一个概率：$p(\ell_\pi \mid i)$，即某个技能描述是指令 $i$ 的合理下一步的可能性。而我们真正想要的是「技能既正确、又能做成功」的联合概率 $p(c_i \mid i, s, \ell_\pi)$。假设「选对了但执行失败」对任务推进的贡献为零，就可以做如下分解：

$$
p(c_i \mid i, s, \ell_\pi) \propto \underbrace{p(c_\pi \mid s, \ell_\pi)}_{\text{Can：世界接地}} \cdot \underbrace{p(\ell_\pi \mid i)}_{\text{Say：任务接地}}
$$

笔者认为，这个乘法分解是全文最优雅的一笔：LLM 的概率负责 **任务接地（task-grounding）**，价值函数的概率负责 **世界接地（world-grounding）**，两个各自不完美的信号相乘，得到的是「既有用又可行」的决策。

### 从自由生成到约束打分

直接让 LLM 自由生成计划有两个毛病：输出可能包含机器人根本不会的动作，格式也难以解析。SayCan 的做法是不做生成，改做 **打分（scoring）**：把规划组织成一段用户与机器人的对话（「How would you bring me a coke can?」→「I would: 1. find a coke can, 2. ……」），然后用 Prompt 里的示例把 LLM 约束在这个句式里，再逐一查询每个候选技能描述 $\ell_\pi$ 的似然，取：

$$
\ell_\pi = \operatorname*{arg\,max}_{\ell_\pi \in \ell_\Pi} \; p(\ell_\pi \mid i)
$$

这种做法还有个副产品： **可解释性**。模型不仅给出选择，还给出了所有候选的分数，人类可以直接看到机器人在「考虑」什么。

![LLM 打分过程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/llm.png)

> 图解：LLM 一侧的工作方式。把 Prompt 示例 + 高层指令喂给打分接口，得到每个候选技能的 LLM 概率；选出技能后把它追加到对话里，再次查询，迭代规划下一步，直到输出「done」。

### SayCan 主循环：两个概率相乘，选最大者执行

把 Say 和 Can 合起来，完整的决策规则是：

$$
\pi = \operatorname*{arg\,max}_{\pi \in \Pi} \; p(c_\pi \mid s, \ell_\pi) \cdot p(\ell_\pi \mid i)
$$

每一步对所有技能并行计算两个概率、相乘、取 argmax，执行选中的技能、更新状态，然后把该技能的语言描述追加进 Prompt，循环往复，直到 LLM 选择终止符「done」。伪代码如下：

```
给定: 高层指令 i, 初始状态 s_0, 技能集合 Π 及其语言描述
n = 0
循环（直到上一步选择了 "done"）:
    对每个技能 π ∈ Π:
        p_LLM       = p(ℓ_π | i, 已选技能序列)    # LLM 打分
        p_affordance = p(c_π | s_n, ℓ_π)          # 价值函数打分
        p_combined   = p_affordance × p_LLM
    π_n = argmax p_combined
    在环境中执行 π_n, 更新状态 s_{n+1}
    n = n + 1
```

![SayCan 整体流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/vfs_llm_all.png)

> 图解：SayCan 的完整决策链路。给定高层指令后，LLM 输出「每个技能对指令有用吗」的概率分布，价值函数模块基于当前观测输出「每个技能现在能做成吗」的概率分布，两者相乘后最高分技能被执行；执行完把技能追加回对话，再次查询，直到终止。

![价值函数空间可视化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/vfs.png)
![场景中有红牛和苹果时的 pick 价值](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/vf_values_two.png)
![空场景中的 pick 价值](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/vf_values_none.png)

> 图解：价值函数模块根据当前观测为所有技能打分，构成一个「可供性空间」。中间图里红牛罐和苹果都在视野中，因此「pick up the red bull can」和「pick up the apple」得分很高；右图机器人面对空地，所有 pick 类技能得分都很低。这正是「世界接地」的直观体现——同一个指令，场景不同，该做什么就不同。

## 系统实现：技能、策略和价值函数从哪来？

方法讲清楚了，下一个问题是：这些底层技能和价值函数在真实机器人上怎么训练出来？

### 语言条件的策略与价值函数

实现上，每个技能需要一个策略、一个价值函数和一段语言描述。为了摊薄训练几百个技能的成本，作者训练的是 **多任务的、以语言为条件** 的策略和价值函数，而不是每个技能单独训一个。语言描述通过一个预训练的句子编码器（Universal Sentence Encoder）转成 embedding，作为策略和价值函数的输入；训练时该编码器参数冻结。值得注意的设计是： **生成技能 embedding 的语言模型和做规划的 LLM 可以是不同的模型**——一个管「细粒度表达技能」，一个管「跨技能的高层规划」，各取所长。

- **BC 策略**：基于 BC-Z 方法，用图像输入的行为克隆训练。数据来自 10 台机器人 11 个月采集的 68000 条遥操作演示，外加 276000 条自主执行回合经成功过滤后补充的 12000 条正样本；人工标注员还会剔除不安全、扰动无关物体或不可行的回合。连续动作用 MSE 损失，离散动作（如终止）用交叉熵，16 块 TPUv3 训练约 27 小时。
- **RL 价值函数**：基于 MT-Opt 在仿真中训练，用 RetinaGAN 做 sim-to-real 迁移，再用在线数据持续提升。奖励是稀疏的 0/1（由 3 名标注员中至少 2 人判定技能是否完成），因此 Q 函数用 log loss 更新，输出经 sigmoid 限制在 $[0,1]$，可直接解释为成功概率。训练动用 16 块 TPUv3 约 100 小时，外加 6000 个 CPU worker 分别采集回合和计算目标 Q 值。

![BC 策略网络结构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/bc_architecture.png)

> 图解：BC 策略架构。语言指令经句子编码器 embedding 后，通过 FiLM 条件注入 ResNet-18 视觉主干，最后经全连接层输出各动作分量（机械臂位置、姿态、夹爪、终止动作）。

![RL 价值函数网络结构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/rl_architecture.png)

> 图解：RL 价值函数架构。相机图像先过 7 层卷积；语言 embedding、机器人动作与非图像状态拼接后过全连接层，空间平铺后加到卷积特征图上，再过 11 层卷积，sigmoid 门控输出 $[0,1]$ 内的 Q 值。为支持异步控制，模型还接收「上一个动作还剩多少没执行完」的信息。

动作空间包含末端执行器 6 自由度位姿、夹爪开合、移动底盘的 x-y 位置和朝向增量，以及 terminate 动作。技能库共设计了 551 个技能、横跨 7 个技能族和 17 个物体，实际评测选用其中组合性好、当前成功率高的子集。

### 可供性函数的工程校准

理论上的 $p(c_\pi \mid s, \ell_\pi)$ 落地时需要按技能族分别校准：

- **Pick**：对 RL 价值 $v$ 做归一化裁剪，$p = \mathrm{clamp}\left(\frac{v - v_{min}}{v_{max} - v_{min}}, 0, 1\right)$，取 $v_{max} = 0.5$、$v_{min} = 0.2$；
- **Go to**：按到目标位置的距离 $d$ 线性衰减，$p = \mathrm{clamp}\left(\frac{d_{max} - d}{d_{max} - d_{min}}, 0, 1\right)$；
- **Place**：恒为 1.0，因为「place 只在 pick 之后才可能」这层逻辑语言侧已经能搞定；
- **Terminate**：固定给一个较小的 0.1，保证没有可行技能时规划能收尾。

此外还有一个实用规则：已完成的技能（奖励已拿到）会被压制，避免机器人「走到已经在的桌子前」这种无效动作。这些细节看着琐碎，却是系统真正跑起来的关键。

## 实验验证：101 条指令、两个厨房

解决了「怎么造」，接下来看「好不好用」。

### 实验设置

实验在两个环境进行：一个真实办公厨房，以及一个复刻该厨房的模拟环境（同时也是技能的训练环境）。场景含 5 个有语义的位置（近操作台、远操作台、桌子、垃圾桶、用户位置）和 15 个常见物体。机器人是 Everyday Robots 的移动操作平台（7 自由度机械臂 + 二指夹爪），默认 LLM 为 PaLM 540B，即 PaLM-SayCan。

![实验环境与机器人](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/mk_setup.png)
![15 个实验物体](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/object_setup3.jpg)
![移动操作机器人](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/robot_setup.png)

> 图解：左为厨房实验环境，中为 15 个常用物体（可乐、七喜、苹果、海绵、薯片等），右为 Everyday Robots 移动操作机器人。策略仅从 RGB 图像观测进行控制。

101 条评测指令被分为 7 个指令族，分别考验系统的不同能力维度：

| 指令族 | 数量 | 考察点 | 示例 |
| --- | --- | --- | --- |
| NL Single Primitive | 15 | 单技能的自然语言表达 | Let go of the coke can |
| NL Nouns | 15 | 抽象名词/同义词 | Bring me a fruit |
| NL Verbs | 15 | 抽象动词 | Restock the rice chips on the far counter |
| Structured Language | 15 | 结构化指令（对照组） | Move the rice chips to the far counter |
| Embodiment | 11 | 不同机器人/环境初始状态 | Put the coke on the counter（从任务中途开始） |
| Crowd-Sourced | 15 | 众包的真实随意表达 | My favorite drink is redbull, bring one |
| Long-Horizon | 15 | 长程多步推理 | I spilled my coke, throw it away and bring me something to clean |

评测有两个指标： **规划成功率**（选出的技能序列是否正确，3 名标注员中 2 人同意即算对，与实际能否执行无关）和 **执行成功率**（整套系统是否真的把任务做成了）。

### 主结果

| 指令族 | 数量 | 模拟厨房 Plan | 模拟厨房 Exec | 真实厨房 Plan | 真实厨房 Exec |
| --- | --- | --- | --- | --- | --- |
| NL Single | 15 | 100% | 100% | 93% | 87% |
| NL Nouns | 15 | 67% | 47% | 60% | 40% |
| NL Verbs | 15 | 100% | 93% | 93% | 73% |
| Structured | 15 | 93% | 87% | 93% | 47% |
| Embodiment | 11 | 64% | 55% | 64% | 55% |
| Crowd Sourced | 15 | 87% | 87% | 73% | 60% |
| Long-Horizon | 15 | 73% | 47% | 73% | 47% |
| **总计** | **101** | **84%** | **74%** | **81%** | **60%** |

几个值得注意的点：名词类比动词类难（候选名词有 15 个物体 + 5 个位置，动词只有 6 个）；结构化指令的规划率（93%）反而低于自然语言动词类（100%），说明 LLM 解析自然语言几乎没有损耗；Embodiment 族失分主要来自可供性函数误判；Long-Horizon 最难，失败多为 LLM 提前终止（只拿了一件东西就说 done）。整体错误中 **65% 来自 LLM，35% 来自可供性函数**。

下面两组长程任务的实拍很能说明系统的推理能力：

![长程任务一：运动后恢复](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/overhead_workout.png)

> 图解：「我刚健身完，能给我拿点恢复用的饮料和零食吗？」系统需要理解「恢复」暗示健康——于是拿了水和苹果而非汽水和薯片；还要记住「已经拿了饮料，接下来该拿零食再终止」。

![长程任务二：清理与擦拭](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/overhead_three_sponge.png)

> 图解：「我留了可乐、苹果和水在外面，能把它们扔掉再拿块海绵擦桌子吗？」系统必须追踪「它们」指代哪些物体、海绵该拿到哪里。

![决策过程可视化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/spill_coke_rc5.png)

> 图解：PaLM-SayCan 决策过程的可视化。指令提到可乐（「我把可乐洒了」），但系统理解到关键物体是「清洁工具」，最终综合得分最高的是拿海绵——蓝色（LLM 打分）与橙色（可供性打分）相乘后，正确技能排在最上方。这种实时的、一目了然的决策过程，是 SayCan 可解释性的直接体现。

### 消融：砍掉 Can 或砍掉 Say 会怎样？

| 方法 | 说明 | 规划/执行成功率 |
| --- | --- | --- |
| PaLM-SayCan | 完整系统 | 84% Plan / 74% Exec |
| No VF | 去掉价值函数，只按 LLM 分数选技能 | 67% Plan |
| Generative | LLM 自由生成后投影到最近技能（USE embedding） | 74% Plan |
| BC NL | 整句指令直接喂给语言条件策略 | 0% Exec |
| BC USE | 指令经 USE embedding 投影到最近技能 | 9% Exec（仅单技能任务 60%） |

两组消融各自说明一件事： **没有 affordance**，LLM 会选「正确但此刻做不了」的技能（No VF 掉到 67%）； **没有 LLM**，传统语言条件策略面对抽象指令全军覆没（BC NL 为 0%）。Say 和 Can 缺一不可，这正是标题「Do As I Can, Not As I Say」的含义。

### 换个更强的 LLM，机器人立刻变强

| 指令族 | 数量 | PaLM-SayCan Plan | PaLM-SayCan Exec | FLAN-SayCan Plan | FLAN-SayCan Exec |
| --- | --- | --- | --- | --- | --- |
| NL Single | 15 | 100% | 100% | 67% | 67% |
| NL Nouns | 15 | 67% | 47% | 60% | 53% |
| NL Verbs | 15 | 100% | 93% | 80% | 67% |
| Structured | 15 | 93% | 87% | 100% | 87% |
| Embodiment | 11 | 64% | 55% | 64% | 55% |
| Crowd Sourced | 15 | 87% | 87% | 73% | 67% |
| Long-Horizon | 15 | 73% | 47% | 47% | 33% |
| **总计** | **101** | **84%** | **74%** | **70%** | **61%** |

仅把 LLM 从 FLAN 137B 换成 PaLM 540B，执行成功率就从 61% 升到 74%，错误率近乎减半。附录中还对 PaLM 8B/62B/540B 与 FLAN 做了生成式对比：总分分别为 38%、72%、74%、43%，大模型明显更好（62B 与 540B 差距不大，但 Chain-of-Thought 等结果表明更难任务上差距会拉开）。

笔者认为这是全文最有想象力的结论： **机器人第一次搭上了 LLM Scaling 的便车**——语言模型变好，机器人不改动一行代码就跟着变强，两个领域从此可以协同 scale。

### Prompt 工程消融：示例数量的影响

作者还建了一个「语言模拟器」来快速调 Prompt：给定指令和正确技能序列，自动生成一致的可供性分数与干扰项，验证 SayCan 能否恢复正确解。结果显示示例数量对规划成功率影响很大：

| Prompt 示例数 | 要求正确终止 | 不要求终止 |
| --- | --- | --- |
| 0 | 10% | 52% |
| 1 | 64% | 74% |
| 2 | 68% | 76% |
| 4 | 82% | 84% |
| 8 | 80% | 80% |
| 完整 Prompt（17 个） | 88% | 88% |

零示例时 LLM 几乎不会主动说「done」（10%）；4 个示例后基本够用；最终版本用 17 个示例。经验之谈还包括：步骤显式编号、每步单独一行会显著提升效果；示例中反复出现某类物体会带偏打分；对话式（How would you / I would）比命令式更自然、效果更好。

## 新能力案例：加技能、Chain-of-Thought、多语言

主实验之外，论文还展示了三个「即插即用」的新能力。

**添加新技能：抽屉操作**。只需把新技能加进 LLM 的候选集、配上价值函数、在 Prompt 里补几个示例，SayCan 就能做「把可乐和百事补货进抽屉」这类任务。21 条查询下规划成功率 100%，执行成功率 33%（瓶颈在串联的底层操作策略，如抽屉开得不够大），且不影响其他指令的表现。

![抽屉任务序列](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/drawer_seq.png)

> 图解：机器人从抽屉取米饼的完整序列。由于机器人只有一条手臂，它必须规划一长串动作：先把米饼从抽屉取出放到台面，关上抽屉后再重新拿起——这种「单手约束」下的长链条规划完全由 SayCan 自动完成。

**Chain-of-Thought 推理**。原版 SayCan 继承了大模型不擅长否定的毛病（如「给我拿个不是苹果的零食」）。引入 CoT prompting 后，在 Prompt 中增加一段「Explanation」，先让 LLM 生成一段解释，再把解释并入上下文做打分，即可处理否定和需要推理的任务。例如「给我带一杯无咖啡因的果味饮料」，模型先推理「用户要果味且无咖啡因的饮料，我选青柠汽水」，再输出技能序列。

**多语言查询**。虽然系统完全没为多语言设计，但 PaLM 本身在多语料上训练过，中文、法语、西班牙语指令的规划成功率几乎没有下降（12 条多语查询中仅 1 条法语长句失败）。

此外，文章还指出 SayCan 目前是开环的——只在当前决策步通过价值函数获得环境反馈；后续工作 Inner Monologue 正是基于 SayCan 的可扩展语言接口，引入成功检测器、场景描述甚至人类反馈实现闭环规划。

## 开源版本

作者在 say-can.github.io 开源了一个 Colab 实现：桌面环境 + UR5 机械臂 + 随机生成的彩色方块和碗，底层策略用 CLIPort，可供性用 ViLD 开放词汇检测器代替价值函数，LLM 用 GPT-3，输出形如「pick up the `object` and place it in `location`」的代码化步骤。

![开源桌面环境](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Do-As-I-Can-Not-As-I-Say-Grounding-Language-in-Robotic-Affordances/figures/open_source.png)

> 图解：开源版 SayCan 的桌面环境。任务「把所有方块放进同色碗」被拆解为「捡起蓝色方块放进蓝色碗」等一系列步骤，由 CLIPort 策略执行。

## 相关工作一瞥

- **语言接地**：以往工作或为 LLM 增加环境输入/动作输出，或靠 Prompt 工程接地，但都没有交互经验；SayCan 用预训练价值函数接地，零样本即可获得长程行为能力。
- **语言条件行为学习**：大量工作用模仿学习或 RL 学语言条件策略，但多聚焦低层指令；SayCan 的焦点是榨取 LLM 中的高层知识来完成长程任务。
- **任务与运动规划**：经典方法依赖显式符号与约束；SayCan 用 LLM 的语义知识同时承担「理解指令」和「理解如何执行」，配合通用的学习技能库，可以扩展到真实世界。

## 总结与展望

- **核心思想**：LLM 打分（Say，任务接地）× 价值函数打分（Can，世界接地），相乘取 argmax，得到「既想做又能做」的下一步技能。
- **理论支点**：稀疏 0/1 奖励下 RL 训出的价值函数天然就是可供性函数，直接给出技能成功概率。
- **关键数字**：101 条真实厨房指令，PaLM-SayCan 规划 84%、执行 74%，执行表现较未接地基线近乎翻倍；错误 65% 来自 LLM、35% 来自可供性。
- **最激动人心的性质**：机器人性能随底层 LLM 变强而免费提升（FLAN→PaLM 执行率 61%→74%），语言模型与机器人首次实现协同 scale。
- **工程启示**：新技能即插即用，CoT、多语言能力开箱即得；但系统瓶颈在底层技能库的广度与鲁棒性，且对「报告高价值却执行失败」的情况缺乏反应能力。

一句话展望：SayCan 把「语言理解」与「机器人控制」解耦又对接的因子化范式，为后来的 Inner Monologue、RT-2 等具身智能工作铺了路；而它留下的开放问题——真实机器人经验能否反过来改进 LLM 的事实性与常识推理、自然语言是不是编程机器人的正确「本体」——至今仍是具身智能最值得追的方向。

> 本文参考自 [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)