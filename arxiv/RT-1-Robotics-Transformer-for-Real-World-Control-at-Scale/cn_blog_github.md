# RT-1：一个模型执行 700+ 指令

让机器人听懂一句话并完成操作，难点不只是“听懂”，而是面对新物体、新背景和新指令组合时仍然做得对。真实机器人数据采集昂贵，这种泛化能力直接决定机器人能否走出为单一任务定制的实验室流程。RT-1 的思路是：用 13 台机器人收集约 13 万条多任务示范，再以语言条件化的 EfficientNet、TokenLearner 和 Transformer 将图像与指令直接映射为离散动作。最终，它在已见指令上取得 97% 成功率、在未见指令上取得 76% 成功率，并完成了超过 3000 次真实世界评估。

![RT-1 高层架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/rt1_teaser_model.png)

> 图解：RT-1 接收相机图像和自然语言指令，经过 FiLM 条件化的 EfficientNet、TokenLearner 与 Transformer，输出机械臂、移动底盘和夹爪的离散动作。模型共有 35M 参数，却能以 3 Hz 进行闭环控制。

![RT-1 数据与能力概览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/rt1_teaser_tasks.png)

> 图解：从左到右依次概括了 RT-1 的五类能力：在 700 多项任务、约 13 万条示范上训练；泛化到新任务；抵抗背景、干扰物与场景变化；执行长时序任务；吸收仿真和其他机器人平台的数据。

## 为什么机器人需要“通用底座”

传统 End-to-End 机器人学习通常围绕一个或少数任务采集数据，再训练对应策略。这很像早期监督学习：每换一个任务，就重新准备一套专用数据和模型。

这种做法在机器人上代价尤其高。示范往往依赖人工遥操作，或者需要复杂的自主采集系统；而现实环境不断变化，只记住训练场景的策略很容易失效。

RT-1 借鉴视觉和 NLP 中大模型“先吸收广泛经验，再迁移到具体任务”的思路，关注两个问题：

- 能否用一个多任务模型学会大量机器人指令？
- 数据规模和多样性增加后，模型能否 Zero-shot 泛化到新任务、新物体和新环境？

此前，Gato 也探索了通用 Transformer 策略，但真实机器人任务主要集中于积木堆叠，未系统评估新任务泛化；BC-Z 等指令跟随方法则在新任务上的表现有限。RT-1 的差异不仅是换用 Transformer，更是把**大规模真实数据、语言早期融合和实时推理**放进同一套系统。

### 把控制写成序列建模问题

在时刻 $t$，策略接收语言指令 $i$ 和截至当前的图像历史 $\{x_j\}_{j=0}^t$，输出动作分布：

$$
a_t \sim \pi\left(\cdot \mid i, \{x_j\}_{j=0}^{t}\right)
$$

机器人执行动作后获得下一帧图像，如此循环，直到策略发出终止指令或达到步数上限。一次完整交互称为一个 episode，最终用二元奖励 $r\in\{0,1\}$ 表示指令是否完成。

训练采用 Behavioral Cloning（行为克隆）：数据集中只保留成功示范，模型最大化示范动作的条件似然，等价于最小化负对数似然：

$$
\mathcal{L}=-\sum_{n,t}\log \pi\left(a_t^{(n)}\mid i^{(n)},\{x_j^{(n)}\}_{j=0}^{t}\right)
$$

Transformer 在这里承担的是“从视觉—语言 Token 序列到动作 Token 序列”的映射。真正困难的是：怎样既保留足够信息，又把序列压短到可以实时控制。

## 数据底座：13 台机器人、17 个月、744 条指令

要研究泛化，首先要把“任务多”落实为可执行、可评估的数据。RT-1 使用 Everyday Robots 的移动操作机器人：它有一只 7 自由度机械臂、两指夹爪和可移动底盘。

![RT-1 机器人、环境与物体](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/RT-1_Robot_Setup.png)

> 图解：(a) 是用于大规模采集的 Robot Classroom；(b)、(c) 是两间真实办公室厨房 Kitchen1 和 Kitchen2，它们在光照、背景和厨房几何结构上不同；(d) 是移动操作机器人及 $300\times300$ RGB 输入；(e) 是扩展技能多样性的常用物体；(f) 是主要用于扩展抓取物体多样性的更大物体集合。

数据采集持续 17 个月，由 13 台机器人完成，最终得到约 13 万条人工示范。每条 episode 都配有一段事后标注的自然语言指令，通常由动词和一个或多个名词组成，例如“拿起冰茶罐”或“把可乐罐移到绿色薯片袋附近”。

| 技能 | 指令数 | 含义 | 示例指令 |
|---|---:|---|---|
| Pick Object | 130 | 将物体从表面拿起 | pick iced tea can |
| Move Object Near Object | 337 | 将第一个物体移到第二个物体附近 | move pepsi can near rxbar blueberry |
| Place Object Upright | 8 | 将细长物体竖直放置 | place water bottle upright |
| Knock Object Over | 8 | 将细长物体碰倒 | knock redbull can over |
| Open Drawer | 3 | 打开任意橱柜抽屉 | open the top drawer |
| Close Drawer | 3 | 关闭任意橱柜抽屉 | close the middle drawer |
| Place Object into Receptacle | 84 | 将物体放入容器 | place brown chip bag into white bowl |
| Pick Object from Receptacle and Place on Counter | 162 | 从容器取出物体并放到台面 | pick green jalapeno chip bag from paper bowl and place on counter |
| 长指令与限制实验的附加技能 | 9 | 面向真实长时序场景的技能 | open the large glass jar of pistachios；pull napkin out of dispenser；grab scooper |
| **合计** | **744** |  |  |

采集过程本身也被系统化：每个 episode 开始时，机器人自主驶向工位；软件采样要演示的指令和场景随机化方式，再通知操作员。操作员通过两个 VR 遥控器遥操作，遥控器的三维平移和旋转被映射为机械臂末端位移，摇杆则控制底盘转向和行进距离，从而保证示范与策略使用一致的动作空间。

![任务数、数据量和已见任务成功率随时间的变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/data_tasks_success.png)

> 图解：横轴是从 2021 年第一季度到 2022 年第四季度的采集进程；左纵轴表示已见任务成功率与数据量百分比，右纵轴以对数刻度表示任务数。浅蓝柱为任务数，绿线为已见任务成功率，黑线为数据量。随着任务和数据持续增加，已见任务成功率总体上升，最终任务数达到 741（该图统计口径与正文最终表格的 744 略有差异）。

数据解决“学什么”，接下来的问题是模型怎样在不超出实时控制预算的前提下吸收这些经验。

## RT-1 架构：先融合，再压缩，最后决策

![RT-1 完整模型架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/rt1_full_model.png)

> 图解：自然语言指令先由 Universal Sentence Encoder 编码，再通过 FiLM 层条件化预训练 EfficientNet；视觉—语言特征经 TokenLearner 压缩后，送入 decoder-only Transformer，最终输出离散动作 Token。

### 语言条件化视觉：让网络先看“与任务有关的东西”

RT-1 输入连续 6 帧图像，每帧分辨率为 $300\times300$。ImageNet 预训练的 EfficientNet-B3 将每帧编码为 $9\times9\times512$ 的空间特征图，再展开为 81 个视觉 Token。

与先独立切分图像 Patch、最后再融合语言的做法不同，RT-1 在视觉编码早期就注入指令。指令先由 Universal Sentence Encoder 转为预训练文本向量，再通过 FiLM（Feature-wise Linear Modulation，特征级线性调制）调整卷积特征：

$$
\operatorname{FiLM}(F,i)=\gamma(i)\odot F+\beta(i)
$$

其中 $F$ 是视觉特征，$\gamma(i)$ 和 $\beta(i)$ 由指令嵌入生成。直觉上，语言像一枚“注意力滤镜”：指令要求开抽屉时，视觉网络可以更早突出抽屉，而不是平均保留所有物体信息。

直接在预训练网络中插入 FiLM 可能破坏原有激活，因此作者将生成仿射变换的全连接层零初始化，使 FiLM 初始近似恒等映射。这样既保留 ImageNet 预训练能力，又能逐步学习任务相关调制。包含 FiLM 的 EfficientNet-B3 共有 16M 参数、26 层 MBConv 与 FiLM 模块。

笔者认为，这个设计的价值不只是“多用了语言信息”，而是让语言在视觉信息被压缩之前就参与筛选；后面的抗干扰实验也支持这一判断。

### TokenLearner：把 81 个 Token 压成 8 个

如果直接对 6 帧、每帧 81 个 Token 做 Self-Attention，序列长度会达到 486，实时推理压力较大。TokenLearner 用可学习的逐元素注意力，软选择并组合信息，将每帧 81 个 Token 压缩为 8 个。

6 帧一共留下 $6\times8=48$ 个 Token，加入位置编码后送入 Transformer。与固定裁剪或平均池化相比，TokenLearner 能根据内容决定保留哪些区域和特征组合。

### Transformer 与离散动作：一次输出 11 个控制量

主干是 8 层 decoder-only Transformer，共有 19M 参数；加上视觉—语言编码器，完整模型约 35M 参数。它输出 11 个动作维度：

- 机械臂 7 维：$x,y,z$ 平移，roll、pitch、yaw 旋转，以及夹爪开合；
- 底盘 3 维：$x,y$ 移动与 yaw 转向；
- 模式 1 维：在控制机械臂、控制底盘、结束 episode 三种模式间切换。

连续动作维度在各自取值范围内被均匀划分为 256 个 bin，动作预测因而变成分类问题，训练使用标准 Categorical Cross-Entropy 和 causal masking。

为什么不用常见的高斯连续动作？单峰高斯难以表达“多种正确操作方式”的多模态分布，而逐维离散分类更灵活。后文消融显示，换成连续动作后，已见任务成功率从 97% 降至 68%，干扰物场景从 83% 降至 37%，说明这一选择对多任务示范尤其重要。

### 实时推理：3 Hz 不只是模型跑得快

人类完成论文中的单步操作通常需要 2–4 秒，作者据此将控制目标设为至少 3 Hz。考虑相机和通信延迟后，网络推理预算需低于 100 ms。

RT-1 采用两项优化：

1. TokenLearner 减少注意力序列长度，带来 2.4 倍加速；
2. 相邻滑动窗口存在重叠，已计算的图像 Token 可以缓存复用，带来 1.7 倍加速。

模型不按自回归方式逐个生成动作维度，因为实验发现这会显著减速，却没有带来整体收益。为减少控制抖动，系统还使用固定等待机制：从采集状态到执行动作等待 280 ms，即各组件观测到的最大总延迟。

模型设计完成后，真正关键的是用同一套真实数据、同一套真实机器人协议验证它到底比已有策略好在哪里。

## 实验一：已见任务接近饱和，未见任务仍有差距

所有架构都在作者收集的同一数据集上训练，因此主实验比较的是模型，而不是谁的数据更好。Gato 被缩至 37M 参数，与 35M 的 RT-1 接近；原始 1.2B Gato 在机器人上的单次推理需要 1.9 秒，无法满足控制频率。BC-Z 使用 ResNet 前馈策略和连续动作，另有参数规模更接近 RT-1 的 BC-Z XL。

评估覆盖四类场景：

| 评估类型 | 协议与难点 |
|---|---|
| 已见任务 | 正文主实验覆盖 200 多条训练指令，同时随机改变物体位置、时间和机器人位置 |
| 未见任务 | 正文描述 21 条新指令；附录标准化协议列出 53 条，物体与技能分别见过，但组合未见过 |
| 抗干扰物 | 正文进行 30 项真实任务，增加干扰物数量，困难设置还会遮挡目标 |
| 抗背景变化 | 正文进行 22 项真实任务，更换台面纹理、光照和厨房 |

![干扰物、背景与真实场景泛化评估](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/robustness2.png)

> 图解：第一行是干扰物评估，从 0–5 个干扰物，到 9 个干扰物，再到目标被部分遮挡；第二行是背景评估，从原环境到花纹桌布，再到新厨房；第三行是真实厨房中的 L1、L2、L3 泛化，逐步叠加新布局、新干扰物、新物体或新位置。

### 主结果：优势最大的是在分布外场景

| 模型 | 已见任务 | 未见任务 | 干扰物 | 背景变化 |
|---|---:|---:|---:|---:|
| Gato | 65 | 52 | 43 | 35 |
| BC-Z | 72 | 19 | 47 | 41 |
| BC-Z XL | 56 | 43 | 23 | 35 |
| **RT-1** | **97** | **76** | **83** | **59** |

> 表中数值均为成功率百分比。RT-1 相比次优模型，在未见任务上高 24 个百分点，在干扰物和背景变化上分别高 36 和 18 个百分点。

![RT-1 与基线的总体成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/main_baselines_flip.png)

> 图解：横轴依次是已见任务、未见任务、干扰物和背景变化，纵轴是成功率。RT-1 在四组中均为最高；已见任务的差距已经明显，但未见任务和困难干扰场景更能体现架构差异。

RT-1 的 97% 并不意味着测试场景与训练完全相同：物体摆放、机器人位置和一天中的时间仍会变化。另一方面，76% 的未见指令成功率也说明，从新组合到可靠执行仍有约四分之一的失败空间。

![RT-1 在不同指令下的真实执行轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/trajectories-bigger.png)

> 图解：每行展示一条真实机器人轨迹，覆盖不同技能、环境和物体。图像序列体现策略如何根据指令连续完成接近、抓取、移动或放置，而不是只对单帧做识别。

附录进一步给出了三类分布外轨迹，便于看清失败与成功发生在怎样的场景中。

![不同背景下的 RT-1 轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/backgrounds_trajectories.png)

> 图解：这些轨迹改变桌面纹理、光照、柜台材质和厨房背景。最困难设置直接进入从未训练过的厨房，检验视觉变化是否会破坏操作策略。

![L1、L2、L3 真实指令泛化轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/generalization_trajectories.png)

> 图解：L1 引入新厨房布局与光照，L2 再加入未见干扰物，L3 进一步加入新物体或水槽旁等新位置。三种等级把多种分布偏移组合起来，比单变量测试更接近实际部署。

![不同干扰程度下的 RT-1 轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/distractors_trajectories.png)

> 图解：训练数据通常只有 2–4 个干扰物，而评估逐渐增加杂物密度；困难场景中目标还会被遮挡。轨迹用于观察策略是否仍能锁定任务相关物体并完成交互。

### 真实厨房：多种变化同时出现时才算真正泛化

作者还设计了补给抽屉、整理碰倒的调料瓶、准备橙子和纸巾、寻找太阳镜与玩具等真实任务，并按分布偏移程度划分 L1、L2、L3。

| 模型 | 全部场景 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| Gato | 30 | 63 | 25 | 0 |
| BC-Z | 45 | 38 | 50 | **50** |
| BC-Z XL | 55 | 63 | **75** | 38 |
| **RT-1** | **70** | **88** | **75** | **50** |

![真实厨房三级泛化结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/kanishka_data.png)

> 图解：该图与上表对应，比较 Gato、BC-Z、BC-Z XL 和 RT-1 在 L1、L2、L3 及整体场景中的成功率。RT-1 整体为 70%，但 L2 与 BC-Z XL 持平、L3 与 BC-Z 持平，并非每个子项都独占优势。

单步泛化之后，下一个问题是：RT-1 能否吸收与原始真实示范差异更大的数据，而不是只在自己的数据分布内变强？

## 实验二：仿真与其他机器人的经验也能被吸收

### 加入仿真：新物体技能从 23% 提升到 87%

作者先用 real-to-sim 方法将真实策略迁移到仿真，再通过多任务 RL 收集新物体抓取经验，最终获得 51.8 万条成功仿真轨迹，并与真实示范混合训练。所有评估仍在真实世界完成，只测试 pick 和 move-to 技能。

| 模型 | 训练数据 | 真实物体＋已见技能 | 仅仿真见过的物体＋已见技能 | 仅仿真见过的物体＋未见技能 |
|---|---|---:|---:|---:|
| RT-1 | 仅真实数据 | 92 | 23 | 7 |
| RT-1 | 真实＋仿真 | 90（-2） | **87（+64）** | **33（+26）** |

![加入仿真数据后的性能变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/sim_delta.png)

> 图解：加入仿真数据后，真实物体上的已见技能仅从 92% 微降至 90%；只在仿真中出现过的物体，其已见技能真实成功率从 23% 跃升至 87%，未见技能组合也从 7% 提升到 33%。

这里的“未见技能”指物体只在仿真抓取任务中出现，而测试指令可能是此前从未与该物体组合过的“把物体移到苹果附近”。因此，提升不仅是记住仿真动作，还包含跨域物体知识与真实技能组合的迁移。

### 加入 Kuka 数据：跨本体经验带来近两倍成功率

第二个实验混合 Everyday Robots 数据与 Kuka IIWA 的 QT-Opt 箱内抓取数据。后者包含 20.9 万个 episode，由 RL 智能体采集，在机器人外观、动作空间、环境动力学和动作分布上都与人工遥操作数据不同。

![跨机器人数据混合训练与评估](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/kuka_data_figure.png)

> 图解：上方是 Everyday Robots 的 RT-1 数据，下方是 Kuka 箱内抓取数据；两者共同训练一个 RT-1，再在 Everyday Robots 上分别进行箱内抓取和原 Classroom 任务评估。

为统一数据，作者将 Kuka 的 4 自由度动作转换到 RT-1 动作空间：roll 和 pitch 置零，保留 yaw，并把二值夹爪命令映射为连续开合量。由于 Kuka 数据没有物体名，所有指令统一标为“pick anything”。两类数据按 RT-1 数据与 Kuka 数据 $2:1$ 混合。

| 训练数据 | Classroom 评估 | 箱内抓取评估 |
|---|---:|---:|
| Kuka＋Everyday Robots | 90（-2） | **39（+17）** |
| 仅 Everyday Robots | 92 | 22 |
| 仅 Kuka | 0 | 0 |

> 箱内抓取评估使用 Everyday Robots，包含 72 次抓取试验；为使场景接近 Kuka 数据，作者还调整了料箱、线缆和夹爪外观。

![混合 Kuka 数据后的性能变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/kuka_delta.png)

> 图解：混合训练使原 Classroom 成功率仅下降 2 个百分点，而箱内抓取从 22% 提升至 39%。只用 Kuka 数据直接在 Everyday Robots 上评估则为 0%，说明收益来自跨本体数据混合，而非简单的直接迁移。

模型能吸收异构经验之后，还要验证这些单步能力能否串联成真正的长任务。

## 实验三：接入 SayCan，最长执行 50 步

SayCan 用大型语言模型把高层指令拆成低层技能，再用 affordance（可执行性）价值函数判断当前机器人能否完成各步骤。RT-1 在其中负责底层移动与操作。

例如“给我拿两种不同汽水”可以拆成：找到可乐、拿起、送来、放下，再对百事可乐重复，最后结束。步骤越多，成功率越容易因误差累积而下降，因此底层单步可靠性十分关键。

评估包含 15 条长时序指令，平均每条需要 9.6 步、涉及 2.4 种操作技能，例如：

- 清空桌上的所有物品；
- 拿一罐青柠汽水和一袋薯片；
- 扔掉可乐，再拿清洁用品；
- 准备水果、汽水和薯片作为午餐。

| 方法 | Kitchen1 规划 | Kitchen1 执行 | Kitchen2 规划 | Kitchen2 执行 |
|---|---:|---:|---:|---:|
| 原始 SayCan | 73 | 47 | — | — |
| SayCan＋Gato | 87 | 33 | 87 | 0 |
| SayCan＋BC-Z | 87 | 53 | 87 | 13 |
| **SayCan＋RT-1** | **87** | **67** | **87** | **67** |

> 原始 SayCan 使用了略不同的 prompt，因此规划成功率与其他方法不完全可比；Kitchen2 是新厨房，原始 SayCan 未报告该设置。

RT-1 在两间厨房中的执行成功率都为 67%，没有在其他基线上出现的大幅跨场景下滑。补充视频还展示了最长 50 步的任务执行；这是一种能力演示，不应与上述 15 条指令的平均统计混为一谈。

## 实验四：数据多样性比单纯堆数量更重要

真实机器人数据昂贵，因此“需要多少数据”和“需要怎样的数据”比单纯扩大模型更实际。作者分别构造两类子集：

- **减少数量**：保留全部任务，但将每条任务示范上限分别截为 200、100、50 条，得到总数据量的 51%、37%、22%；
- **减少多样性**：删除数据最少的任务，只保留 75% 的任务，却仍保留 97% 的总数据量。

| 任务保留比例 | 数据保留比例 | 已见任务 | 综合泛化 | 未见任务 | 干扰物 | 背景 |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 100% | 97 | 73 | 76 | 83 | 59 |
| 100% | 51% | 71 | 50 | 52 | 39 | 59 |
| 100% | 37% | 55 | 46 | 57 | 35 | 47 |
| 100% | 22% | 59 | 29 | 14 | 31 | 41 |
| 75% | 97% | 86 | 54 | 67 | 42 | 53 |

![数据规模与多样性消融](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/data_ablation_simple.png)

> 图解：横轴为保留的数据量比例，纵轴为成功率；蓝点表示已见任务，灰点表示综合泛化。绿色箭头表示减少数据数量，紫色箭头表示减少任务多样性；右侧近乎半色的点表示只训练 75% 任务。只删掉 25% 的任务、保留 97% 数据，就会让综合泛化从 73% 降至 54%，损失接近把数据总量砍到约一半。

结论是：数据减少时，泛化通常比已见任务下降更快；而覆盖任务变窄时，即使总数据几乎不变，泛化也会显著受损。对机器人数据采集而言，“增加不同任务和场景”往往比“在已有任务上重复采集”更值得投入。

## 消融实验：哪些设计真正贡献了性能

下表同时报告总体成功率和相对完整 RT-1 的变化。干扰物进一步分为 Easy、Medium、Hard，推理时间只计算网络本身，不包含所有模型共有的相机与通信延迟。

| 模型 | 已见 | 未见 | 干扰物全部 | Easy | Medium | Hard | 背景 | 推理时间（ms） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gato | 65（-32） | 52（-24） | 43（-40） | 71 | 44 | 29 | 35（-24） | 129 |
| BC-Z | 72（-25） | 19（-57） | 47（-36） | 100 | 67 | 7 | 41（-18） | 5.3 |
| BC-Z XL | 56（-41） | 43（-33） | 23（-60） | 57 | 33 | 0 | 35（-24） | 5.9 |
| **RT-1** | **97** | **76** | **83** | **100** | **100** | **64** | **59** | **15** |
| 缩小模型 | 89（-8） | 62（-14） | 77（-6） | 100 | 100 | 50 | 53（-6） | 13.5 |
| 去掉 ImageNet 预训练 | 84（-13） | 43（-33） | 60（-23） | 100 | 67 | 36 | 41（-18） | 15 |
| 改用连续动作 | 68（-29） | 43（-33） | 37（-46） | 71 | 67 | 0 | 35（-24） | 16 |
| 改用自回归动作 | 85（-12） | 71（-5） | 67（-16） | 100 | 78 | 43 | 65（+6） | 36 |
| 去掉图像历史 | 82（-15） | 62（-14） | 50（-33） | 71 | 89 | 14 | 59（+0） | 15 |
| 去掉 Transformer | 86（-13） | 62（-14） | 67（-16） | 100 | 100 | 29 | 59（+0） | 26 |

![模型消融相对 RT-1 的成功率变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/model_ablation_with_baselines.png)

> 图解：横轴列出基线与各项消融，纵轴是相对完整 RT-1 的成功率差值，越负表示下降越多。蓝、红、黄、绿分别对应已见任务、未见任务、干扰物和背景。连续动作替换与去除预训练的下降最明显；自回归动作只在背景项提高 6 个百分点，却拖慢推理并损害其他指标。

可以得到四个具体判断：

1. **ImageNet 预训练主要帮助泛化**：去掉后未见任务下降 33 个百分点，背景成功率下降 18 个百分点。
2. **离散动作是多模态示范的关键**：连续高斯动作让所有主要指标大幅下降，困难干扰场景甚至降至 0%。
3. **历史信息主要帮助处理遮挡与干扰**：只输入单帧后，干扰物成功率从 83% 降至 50%，背景指标则不变。
4. **自回归动作不值得其实时代价**：推理从 15 ms 增至 36 ms，整体性能反而下降；RT-1 因而并行输出各动作维度。

RT-1 的 15 ms 推理远快于同量级 Gato 的 129 ms，但仍慢于 5.3 ms 的 BC-Z。也就是说，它并非所有维度上的绝对最优，而是在性能、泛化和实时性之间取得了更适合该系统的平衡。

## 附录中的两个复现要点

### 用 real-to-sim 做模型选择，而不是直接替代真实评估

多任务机器人策略很难靠少量真实 rollout 选 checkpoint。作者将仿真环境扩展到 551 项任务，为每项定义场景与机器人位姿随机化、成功判据，再用 RetinaGAN 把仿真图像转换成更接近真实的视觉外观。

![原始仿真、RetinaGAN 转换与真实图像](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/env_examples_v2.png)

> 图解：左列是原始仿真图像，中列是经过 RetinaGAN 的仿真图像，右列是真实世界图像。转换主要缩小纹理、光照和场景外观差距，使真实数据训练的策略能够在仿真中完成整段 rollout。

这里不要求仿真成功率等于真实成功率，只要求模型排序具有相关性：真实表现更好的策略，在仿真中也应相对更好。这是一个务实的评估思路——仿真用于低成本筛选，最终结论仍由真实机器人试验给出。

### 注意力为何能抑制干扰

![RT-1 不同层与注意力头的可视化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RT-1-Robotics-Transformer-for-Real-World-Control-at-Scale/figures/viz_attention_map.png)

> 图解：每行对应一个层和注意力头，横向展示 rollout 过程，高亮区域表示注意力集中位置。Layer 2 Head 6 在抓取任务中关注墨西哥辣椒薯片袋、百事罐等可交互物体，Layer 4 Head 2 在开抽屉任务中关注抽屉区域。

不同层和头会关注不同区域，但最常见的是夹爪与目标物体的交互位置。结合 FiLM 的早期语言条件化，可以形成一种解释：语言先帮助视觉提取任务相关特征，TokenLearner 和注意力瓶颈再进一步压缩无关背景与干扰物。这是作者基于可视化给出的机制解释，而不是严格的因果证明。

## 总结：RT-1 留下了什么

- **一个模型学习多任务**：35M 参数、约 13 万条真实示范，覆盖 744 条语言指令，已见任务成功率达到 97%。
- **泛化优于同数据基线**：未见任务达到 76%，在干扰物和背景变化上分别达到 83% 和 59%，但仍未完全解决分布外可靠性。
- **架构围绕实时控制共同设计**：FiLM 早期融合语言，TokenLearner 压缩序列，离散动作表达多模态分布，最终以 3 Hz 闭环运行。
- **数据广度比数据堆叠更关键**：只减少任务覆盖、几乎不减少数据量，也会显著伤害泛化；仿真和跨机器人数据则可以被同一模型吸收。
- **长任务依赖高可靠单步技能**：接入 SayCan 后在两间厨房均取得 67% 执行成功率，并展示最长 50 步操作。

RT-1 仍受行为克隆上限、已见概念组合范围和操作灵巧性的限制，模型卡也明确指出它不适合与人直接交互；未来需要更丰富的环境与动作数据、更强的记忆和反应速度，以及让非专家也能低成本扩展机器人技能的数据采集方式。

> 本文参考自 [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)