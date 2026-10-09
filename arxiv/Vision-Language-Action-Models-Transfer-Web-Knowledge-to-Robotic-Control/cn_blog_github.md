# RT-2：让机器人"说话"的模型也能"动手"——视觉-语言-动作模型如何将互联网知识迁移到机器人控制

机器人领域一直有个尴尬的现实：大模型在网页数据上学到了丰富的语义知识和推理能力，但机器人学到的只是"抓这个、放那里"的低层技能，两者之间隔着一道鸿沟。Google DeepMind 的这篇 RT-2 论文提出了一个简单到令人惊讶的方案：把机器人动作表示成文本 token，让视觉-语言模型（VLM）像说话一样"说出"动作，从而把互联网规模的预训练知识直接注入端到端的低层机器人控制。效果有多硬？在约 6000 次真实机器人评测中，RT-2 在未见过的物体、背景、环境上的平均成功率达到 **62%** ，相比之前最强的 RT-1（32%）和 MOO（35%）接近 **翻倍** ；在测试符号理解、推理、人物识别的"涌现能力"评测中，最佳模型 RT-2-PaLI-X 的平均成功率（60%）是次优基线的 **3 倍以上** 。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/rt2_teaser.png)

> 图解：RT-2 的整体思路。机器人动作被表示成"另一门语言"，转化为文本 token 后与互联网规模的视觉-语言数据一起训练；推理时，模型输出的文本 token 再被反词元化（de-tokenize）回机器人动作，实现闭环控制。这样 VLM 的骨干网络和预训练成果被完整继承，其泛化性、语义理解和推理能力随之迁移到机器人控制中。

## 一、背景：机器人为什么需要"互联网知识"？

大语言模型（LLM）和视觉-语言模型（VLM）已经在开放词汇识别、视觉问答、复杂推理上展现出惊人的能力。这些能力对要在真实世界执行多样任务的通用机器人来说，显然价值巨大。

但问题是，机器人如何获得这些能力？蛮力路线是收集数百万条机器人交互数据——可最强的 VLM 是在数十亿 token 和图像上训练的，短期内机器人数据根本追不上这个量级。另一条路是直接套用现成大模型，但大模型推理的是语义、标签和文本，机器人需要的是笛卡尔空间末端执行器指令这样的低层动作，两者"语言不通"。

此前的工作（如 SayCan、PaLM-E 的规划用法）大多只让大模型负责 **高层规划** ：充当一个"状态机"，把指令解析成一个个技能原语（抓取、放置），再交给独立的低层控制器执行。低层控制器本身完全享受不到互联网规模模型的语义知识。

于是这篇论文提出了核心问题： **能否把大规模预训练 VLM 直接集成到低层机器人控制中，既提升泛化能力，又激发涌现的语义推理能力？**

RT-2 的答案既简单又有效：直接把为视觉问答设计的 VLM 训练成能输出低层机器人动作的模型。做法是把动作 token 化成文本 token，构造出"多模态句子"——输入是相机观察加任务指令，输出是对应的动作。与 CLIPort 这类把 VLM 嵌入策略结构的工作、或 Gato 这类从零设计新架构的工作不同，RT-2 **不引入任何新参数** ，直接在已经烧掉大量算力预训练好的 VLM 上做文章。作者把这类模型命名为 Vision-Language-Action（VLA）模型，并基于 RT-1 的数据协议和类似的机器人数据集，实例化了 RT-2（Robotics Transformer 2）。

> 博主点评：这个思路的聪明之处在于"借力"——与其在机器人数据上从零堆算力，不如站在已经训练好的 VLM 肩膀上，让动作成为模型词表里的"新方言"。这也解释了为什么 RT-2 不需要任何新的结构组件。

## 二、方法：RT-2 是如何炼成的？

解决了"要不要借 VLM 的力"之后，下一个问题是"具体怎么借"。方法部分可以拆成三步：选底座、改动作、保能力。

### 2.1 预训练 VLM 底座

RT-2 基于两类 VLM 构建：

- **RT-2-PaLI-X** ：基于 PaLI-X，视觉侧用 ViT-22B 处理图像（可接受 $n$ 张图像序列，每张图像产生 $n \times k$ 个 token），随后经由投影层送入 32B 参数、50 层的 encoder-decoder 骨干（类似 UL2），以自回归方式生成输出 token。
- **RT-2-PaLM-E** ：基于 PaLM-E-12B，这是一个 decoder-only LLM，用 ViT-4B 把图像投影到语言 embedding 空间。连续变量与文本输入的拼接让 PaLM-E 成为完全多模态的模型。

这些模型原本接收图像、输出自然语言 token，能做图像构成推断、物体关系问答等任务。模型规模从几十亿到 550 亿参数不等。

### 2.2 机器人动作微调：把动作变成"词"

要让 VLM 控制机器人，它必须学会输出动作。RT-2 采用最直接的路线： **把动作表示为模型输出中的 token，与语言 token 一视同仁** 。

动作编码沿用 RT-1 的离散化方案。动作空间包含：

- 末端执行器 6-DoF 的位置与旋转位移；
- 夹爪开合程度；
- 一个特殊的离散"终止"指令，由策略在任务成功完成时触发。

除终止指令外，每个连续维度被均匀离散成 256 个 bin，因此一个机器人动作可以表示为 8 个整数（各离散 bin 的序号）。微调目标就是把动作向量拼接成一个字符串：

$$
\text{terminate} \quad \Delta \text{pos}_x \quad \Delta \text{pos}_y \quad \Delta \text{pos}_z \quad \Delta \text{rot}_x \quad \Delta \text{rot}_y \quad \Delta \text{rot}_z \quad \text{gripper\_extension}
$$

一个具体的动作字符串实例是：`1 128 91 241 5 101 127`。

接下来要把这 256 个离散 bin 关联到模型 **现有** 词表中的 token。两个底座模型的处理方式不同：

- PaLI-X 的分词器中，1000 以内的整数各自有独立 token，直接把动作 bin 对应到相应整数 token 即可；
- PaLM-E 没有这种便利的数字表示，于是直接 **覆写** 词表中使用频率最低的 256 个 token 作为动作词表。作者指出，这种"覆写已有 token"的做法本质上是一种 symbol tuning，在先前的 VLM 研究中已被证明有效。

机器人数据随之被改造成 VQA 格式：输入是相机图像加文本任务描述（`Q: what action should the robot take to [task instruction]? A:`），输出就是代表动作的数字/token 字符串。

### 2.3 两个关键训练细节

**Co-Fine-Tuning（协同微调）** 。这是整个配方里最关键的技术细节：不只在机器人数据上微调，而是把机器人数据与原始网页数据（视觉问答、图像描述等） **混合** 在一起微调。实验会证明，这让策略在微调期间同时接触网页数据中的抽象视觉概念和低层机器人动作，泛化性显著更好，本质上是防止模型"遗忘"预训练知识。具体配比上，RT-2-PaLI-X 通过提高机器人数据的采样权重，使机器人数据约占每个训练 batch 的 50%；RT-2-PaLM-E 则约为 66%。

**输出约束（Output Constraint）** 。RT-2 与标准 VLM 的一个重要区别是：它在机器人任务上必须输出合法的动作 token 才能被执行。因此解码时施加约束——当 prompt 是机器人动作任务时，只在合法动作 token 中采样；而在标准视觉-语言任务上，模型仍可输出全部自然语言 token。

### 2.4 实时推理：55B 模型怎么跑在机器人上？

现代 VLM 动辄数百亿参数，本文最大的模型达 55B，直接跑在机器人自带 GPU 或桌面机上不现实。据作者所知，RT-2 是迄今为止用于 **直接闭环机器人控制** 的最大模型（超出此前一个数量级），因此需要一套新的部署方案：把模型部署在 **多 TPU 云服务** 上，机器人通过网络查询该服务。这样既达到了可用的控制频率，还能让多台机器人共享同一云服务。具体频率：55B 的 RT-2-PaLI-X 可达 1–3 Hz，5B 版本约 5 Hz。

## 三、实验：RT-2 到底行不行？

方法讲完了，接下来是验证环节。实验围绕四个问题展开，共进行了约 6000 次真实机器人评测：

1. RT-2 在已见任务上表现如何，更重要的是，对新物体、新背景、新环境的泛化如何？
2. 能否观察和测量 RT-2 的涌现能力？
3. 泛化能力随参数量和设计决策如何变化？
4. RT-2 能否像 VLM 一样展现思维链（Chain-of-Thought）推理？

**模型与数据** 。训练用两个实例：RT-2-PaLI-X（5B 和 55B）与 RT-2-PaLM-E（12B）。网页数据来自 PaLI-X 和 PaLM-E 的原始预训练混合（主体是 WebLI，约 100 亿图文对、覆盖 109 种语言，按跨模态相似度取前 10% 得到 10 亿训练样本）。机器人演示数据来自 RT-1：13 台机器人历时 17 个月在办公室厨房环境收集，每条轨迹标注了描述任务的自然语言指令（动词描述技能如"pick"，名词描述物体如"7up can"）。

**基线** 。所有基线使用完全相同的机器人数据：

- **RT-1** ：35M 参数的 Transformer 策略，检验 VLM 预训练是否真的重要；
- **VC-1** 与 **R3M** ：SOTA 预训练视觉表征 + RT-1 骨干，检验"只用表征"的路线；
- **MOO** ：用 VLM 生成语义地图通道再送入 RT-1 骨干，检验"VLM 只做感知增强"的路线。

### 3.1 泛化能力：见过的不难，没见过的才见真章

已见任务沿用 RT-1 的 200+ 指令评测套件（抓取 36 项、撞倒 35 项、竖立放置 35 项、移动 48 项、开关抽屉 18 项、抽屉取放 36 项）。注意这些"分布内"评测仍会变化物体摆放、时间、机器人位置，并非完全静态。

泛化评测分为 **未见物体** 、 **未见背景** 、 **未见环境** 三类，每类再分 easy / hard（如未见物体的 hard 是更难抓取的玩具类物品；未见环境的 hard 是视觉差异很大的办公桌面，easy 是厨房水槽），共 280+ 任务。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/generalization_evals_dm.png)

> 图解：泛化评测的场景示例。从左到右覆盖未见物体、未见背景、未见环境三个维度，并按分布偏移大小区分 easy（偏移较小）与 hard（偏移较大）情形。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/rt2_overall_wide_dm.png)

> 图解：RT-2 两种实例与各基线在已见任务及三类泛化评测上的整体成功率对比。RT-2 在已见任务上与 RT-1 相当，但在所有泛化维度上大幅领先。

具体数字如下表（成功率，%）：

| 模型 | 已见任务 | 未见物体 Easy/Hard | 未见背景 Easy/Hard | 未见环境 Easy/Hard | 未见平均 |
|---|---|---|---|---|---|
| R3M | 45 | 32 / 14 | 13 / 9 | 0 / 2 | 12 |
| VC-1 | 63 | 34 / 10 | 13 / 3 | 0 / 0 | 10 |
| RT-1 | **92** | 31 / 43 | 71 / 9 | 26 / 14 | 32 |
| MOO | 75 | 58 / 48 | 38 / 41 | 19 / 3 | 35 |
| RT-2-PaLI-X-55B | **91** | **70** / **62** | **96** / **48** | **63** / **35** | **62** |
| RT-2-PaLM-E-12B | **93** | **84** / **76** | **75** / **71** | **36** / **33** | **62** |

结论很清晰：已见任务上 RT-2 与 RT-1 基本持平，但泛化场景差距巨大——两种 RT-2 实例平均比 RT-1 和 MOO 好约 **2 倍** ，比 R3M、VC-1 好约 **6 倍** 。这印证了 VLA 模型的核心优势：从互联网规模预训练数据中迁移更具泛化性的视觉与语义概念。另外 PaLM-E 版在更难场景下表现更好、在简单场景下稍弱，两者平均打平。

**开源 Language-Table 基准** 。为了用开源基线和环境提供额外对照，作者还在 Language-Table 仿真环境中协同微调了一个较小的 PaLI-3B 模型（动作离散为 `X Y` 格式，表示末端执行器 2D 笛卡尔目标点增量，取值 -10 到 +10），5 Hz 推理：

| 模型 | Language-Table 成功率 |
|---|---|
| BC-Zero | 72 ± 3 |
| RT-1 | 74 ± 13 |
| LAVA | 77 ± 4 |
| **RT-2-PaLI-3B** | **90 ± 10** |

同一 checkpoint 在真实世界也展现了分布外行为（新的推物体任务、指向环境中未见过的物体），说明 VLM 预训练 + 大模型的表达力在不同机器人、不同（仿真）环境中同样有效。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/LangTable-R2-Narrow-DM.png)

> 图解：Language-Table 真实世界环境中的分布外行为示例。与仿真评测使用完全相同的 RT-2-PaLI-3B checkpoint，模型展示了新颖的推物体技能，并能指向该环境中从未出现过的目标物体。

### 3.2 涌现能力：机器人数据里没教过的，它也会

泛化只是第一层。更有意思的问题是：网页知识迁移能否带来机器人数据中 **从未演示过** 的新能力？作者称之为"涌现能力"——不指望它产生新的物理 **动作** （动作技能仍受限于机器人数据分布），但期望语义、视觉概念（关系、名词）能有效迁移。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/RT2-capabilities-dm.png)

> 图解：RT-2 在需要推理、符号理解和人物识别的真实场景中的泛化示例。这些交互全部未在机器人数据中出现过，例如"把草莓放进正确的碗"（需要理解草莓应与同类水果放在一起）、"拿起快掉下桌子的袋子"（需要物理直觉在两个袋子中识别摇摇欲坠的那个）。

定量评测将涌现能力分为三类，用 A/B 测试框架（四个模型在完全相同的条件下依次评测）与最强的两个基线 RT-1、VC-1 对比：

- **符号理解（Symbol Understanding）** ：考察是否迁移了机器人数据中完全不存在的语义知识，如 `move apple to 3`、`push coke can on top of heart`；
- **推理（Reasoning）** ：要求视觉推理（`move the apple to cup with same color`）、数学（`move X near the sum of two plus one`）、多语言理解（西班牙语 `mueve la manzana al vaso verde`）；
- **人物识别（Human Recognition）** ：如 `move the coke can to the person with glasses`、把可乐罐移向 Taylor Swift 等名人照片。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/quant_eval_collage_dm.png)

> 图解：涌现能力定量评测的部分场景总览，覆盖三大类：(a) 推理、(b) 符号理解、(c) 人物识别。

![Figure 6a](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/rt2_emergent_dm.png)

> 图解：RT-2 与 RT-1、VC-1 在各类涌现技能评测上的成功率对比，RT-2 在所有类别上显著领先。

具体数字（成功率，%）：

| 模型 | 符号理解平均 | 推理平均 | 人物识别平均 | 总平均 |
|---|---|---|---|---|
| VC-1 | 11 | 10 | 13 | 11 |
| RT-1 | 16 | 16 | 20 | 17 |
| RT-2-PaLI-X-55B | **82** | **46** | **53** | **60** |
| RT-2-PaLM-E-12B | 36 | 43 | 43 | 40 |

RT-2-PaLI-X 的总平均成功率是次优基线 RT-1 的 **3 倍以上** 。有个耐人寻味的细节：更大的 PaLI-X 版在符号理解、推理、人物识别上整体更好，但更小的 PaLM-E 版在数学推理上反而占优（35 vs 25）。作者将其归因于 PaLM-E 预训练混合数据使其数学计算能力强于以视觉为主的 PaLI-X——预训练数据配方的影响直接渗透到了下游机器人行为上。

### 3.3 消融：参数量与训练方式各贡献多少？

这部分使用尺寸灵活的 RT-2-PaLI-X，对比 5B 与 55B 两种规模、三种训练方式（从头训练、仅机器人数据微调、协同微调），专注泛化指标：

![Figure 6b](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/rt2_ablations_dm.png)

> 图解：RT-2-PaLI-X 在参数量与训练策略上的消融结果。三条规律一目了然：从头训练几乎不可用；协同微调稳定优于纯微调；模型越大泛化越好。

| 训练方式 | 规模 | 未见物体 Easy/Hard | 未见背景 Easy/Hard | 未见环境 Easy/Hard | 平均 |
|---|---|---|---|---|---|
| 从头训练 | 5B | 0 / 10 | 46 / 0 | 0 / 0 | 9 |
| 仅微调 | 5B | 24 / 38 | 79 / 50 | 36 / 23 | 42 |
| 协同微调 | 5B | 60 / 38 | 67 / 29 | 44 / 24 | 44 |
| 仅微调 | 55B | 60 / 62 | 75 / 38 | 57 / 19 | 52 |
| 协同微调 | 55B | 70 / 62 | 96 / 48 | 63 / 35 | 63 |

三点结论：

1. **从头训练大模型效果极差** ，5B 模型平均成功率仅 9%，作者因此直接跳过了 55B 从头训练的评测；
2. **协同微调优于纯微调** （55B：63 vs 52），因为保留原始数据让模型不遗忘 VLM 训练中学到的概念；
3. **模型越大，泛化越好** （协同微调下 55B 的 63 对 5B 的 44）。

> 博主点评：这张表是全文信息密度最高的实验之一——它同时回答了"VLM 预训练有没有用"（从头训练 9% vs 微调 42%+）和"怎么保住预训练知识"（协同微调 +11 个点）两个问题，为整个方法路线提供了闭环证据。

### 3.4 思维链推理：让机器人先"想"再"动"

受 LLM 思维链 prompting 启发，作者对 RT-2-PaLM-E 做了仅数百步梯度更新的微调，让模型学会把语言与动作结合。做法是增强数据：加入一个"Plan"步骤，先用自然语言描述即将执行动作的目的，再输出动作 token，例如：

```
Instruction: I'm hungry. Plan: pick rxbar chocolate. Action: 1 128 124 136 121 158 111 255.
```

这种数据增强相当于在 VQA 数据（视觉推理）与操作数据（生成动作）之间架了一座桥。定性结果显示，带思维链的 RT-2 能响应更复杂的指令——因为它先有一个用自然语言规划动作的位置。比如"找一个能当锤子用的东西"（计划：选石头），或者"给疲惫的人挑一种饮料"（计划：选能量饮料）。这为"LLM/VLM 做规划 + 低层策略执行"融合进单一 VLA 模型提供了初步证据。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/RT2CoT_Narrow.png)

> 图解：带思维链推理的 RT-2 执行过程。模型同时生成自然语言 Plan 和动作 token，先推理"该做什么、为什么"，再输出具体控制指令。

![Figure 11](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/rt2_filmstrip.png)

> 图解：更多 RT-2 思维链推理的 rollout 示例，展示模型如何在多阶段语义推理后给出动作。

## 四、局限性：RT-2 还不是万能的

作者很坦诚地列出了几个边界：

- **没有新动作** 。网页预训练提升的是语义与视觉概念的泛化，机器人并不会因此获得新的运动技能——物理技能仍受限于机器人数据中见过的技能分布。例如按特定部位抓握（如抓把手）、机器人数据中没见过的新动作（用毛巾擦拭、使用工具）、灵巧精密动作（叠毛巾）、需要多层间接推理的扩展推理，目前都表现不佳。未来方向包括用人类视频等新数据范式来获取新技能。
- **推理成本高** 。虽然实现了大模型的实时控制，但算力开销巨大；在需要高频控制的场景中，实时推理可能成为主要瓶颈。量化与蒸馏是值得探索的方向。
- **可用底座稀缺** 。目前能用来构建 VLA 的开放 VLM 很少，作者希望更多开源模型出现、闭源模型开放微调 API（开放微调 API 就足以构建 VLA 模型）。

![Figure 10](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control/figures/LangTable-R2-fail.png)

> 图解：Language-Table 真实环境中的典型失败案例——无法泛化到 **未见过的物体动力学** 。模型能正确理解指令并移动到正确物体，但控制不了这些物体的动力学特性：笔直接从桌上滚走（左图），而香蕉的质心远离机器人接触点（右图）。推物体的动力学向来以难以预测和控制著称，作者推测需要在更多样的环境与物体上扩展数据才能改善。

## 五、总结

- **核心思路** ：把机器人动作 token 化成文本，让预训练 VLM 直接输出动作，构建端到端的 Vision-Language-Action 模型，不加任何新参数。
- **关键配方** ：协同微调（机器人数据 + 原始网页数据）防止遗忘，是泛化能力的重要来源；解码时约束输出词表保证动作合法。
- **最硬结果** ：约 6000 次真实评测中，未见场景平均成功率 62%，约为 RT-1/MOO 的 2 倍；涌现能力评测 60%，是次优基线的 3 倍以上。
- **消融结论** ：VLM 预训练、协同微调、更大模型规模三者缺一不可，分别对应"知识从哪来""怎么保住知识""知识有多少"。
- **涌现亮点** ：符号理解、多语言指令、数学与语义推理、人物识别均可迁移；加入 Plan 步骤后还能进行"选石头当锤子"式的多阶段思维链推理。

一句话展望：RT-2 证明了机器人学习可以直接"搭"VLM 进步的便车——动作只是模型的又一门语言；但要真正走向通用机器人，还需解决新技能获取（而非仅新语义）与高频实时推理这两道坎。

> 本文参考自 [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)