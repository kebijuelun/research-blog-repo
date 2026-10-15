# Player-1：合成经验练出游戏智能体

让 AI 学会打游戏，最难的从来不是"看懂画面"，而是 **持续、精准、并发的键鼠操控** ——而真实游戏里又很难拿到可规模化、可验证的奖励信号来做强化学习。Player-1 给出的答案是：既然商业游戏里的奖励不好写，那就 **围绕游戏共通的操控需求（瞄准、战斗、驾驶、导航、UI 交互）自建 16 个合成 2D/3D 环境**，在里面用可程序化验证的奖励做 RL，再让学到的操控能力迁移到真实游戏上。效果相当硬：在 Minecraft 的 MineDojo-300 基准上，Player-1-9B 相比自己的 SFT 初始化提升了 **27.8 个百分点**（10.54% → 38.33%），而最强外部基线 GameWAM 只有 12.63%；在完全没见过 RL 训练的 ViZDoom 上，Player-1-35B-A3B 比最强基线高出 **83.7%**。

## 为什么游戏是 Computer-Use Agent 的下一站

Computer-use agent（CUA）已经能看截图、动键鼠，在办公软件和浏览器里完成多步任务。但游戏把难度直接拉满：画面复杂且瞬息万变，操作要求 **精确、持续、并发** ——一边转视角一边跑一边开火，这种"一心三用"是普通 GUI 任务很少涉及的。

现有游戏智能体的学习管线有两条老路，各有硬伤：

- **模仿学习（吃人类演示）** ：Game-TARS、NitroGen 这类工作靠大规模游戏录像训练，但演示只覆盖了开放任务中一小部分状态，模型遇到没见过的局面就容易懵。
- **交互学习（做 RL）** ：SIMA 2、Odysseus 等引入在线强化学习，但在商业游戏里写"程序化奖励"极难规模化——每个游戏的内部状态接口都不一样，逐游戏接入的成本太高。

这就引出了本文的核心问题： **合成经验能否把游戏操控能力推到演示之外，并且泛化到 RL 训练集以外的游戏？**

要回答它，得先迈过三道坎：第一，来自不同来源的游戏录像操作习惯、采样频率各不相同，需要统一到一个共享动作空间；第二，超越演示需要策略在自己走到的状态上获得反馈，但商业游戏的奖励难以程序化；第三，来自不同环境的经验，其操控需求、动力学、奖励尺度都不同，要揉进一个共享策略里还不能互相干扰。

## 核心思路：SFT 打底，合成环境喂经验

Player-1 的整体框架可以用一句话概括： **先用精选的人类演示把 CUA 模型"扶上马"，再在自建合成环境里用可验证奖励的 RL"送一程"** ，中间由一个统一的游戏操控动作空间连接。

![Player-1 整体框架](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/overview.png)

> 图解：Player-1 分为两个阶段。左侧是监督初始化：从多个公开数据集和自录数据中整理出游戏演示轨迹，把录制的键鼠操作映射到共享动作空间，并用 VLM 标注任务指令和视觉推理，对 CUA 模型做 SFT。右侧是交互学习：Player-1-Gym 提供 16 个围绕共通操控需求设计的合成 2D/3D 环境，带程序化可验证奖励和观测/操控动力学随机化，用 PPO 做多环境 RL。两个阶段共用同一个动作空间，因此 RL 学到的操控能力能无缝迁移到训练集之外的 Minecraft、ViZDoom、浏览器游戏乃至商业游戏。

在展开细节之前，先看 Player-1 和代表性游戏智能体管线的定位差异：

| 方法 | 学习重点 | 监督信号 | RL 环境 | 评测范围 | 是否开放 |
| --- | --- | --- | --- | --- | --- |
| Game-TARS | 大规模预训练 | 推理 + 动作 | --- | 留出游戏 | ✗ |
| SIMA 2 | 具身助手 | 推理 + 动作 | 游戏 + 世界模型 | 留出游戏 | ✗ |
| NitroGen | System-1 通用策略 | 动作 | --- | 游戏 + 适配 | ✓ |
| Odysseus | 长程 RL | 推理 + 动作 | 5 个马里奥关卡 | 马里奥游戏 | ✓ |
| **Player-1** | **超越演示** | 推理 + 动作 | **16 个合成环境** | **RL 留出游戏** | ✓ |

> 表格解读：Player-1 的独特位置在于——它不靠逐游戏接奖励，而是把 RL 全部放在按"操控需求"组织的合成环境里，然后直接在 **RL 从未见过的游戏** 上评测迁移效果。这是它与"在每个目标游戏里做在线训练"路线的根本分野。

笔者认为这个思路的聪明之处在于：它把"游戏智能"拆成了两层—— **具体游戏规则** （由 SFT 和推理能力兜底）与 **通用操控能力** （瞄准、走位、协调输入，由合成环境 RL 专攻）。后者恰恰是跨游戏可迁移、又最容易在可控环境里写出奖励的部分。

## 方法：Player-1 的三块拼图

### 拼图一：为游戏扩展 Computer-Use 动作空间

普通 CUA 的动作空间是为 GUI 设计的：点击用归一化绝对坐标、按键都是点按。这在游戏里不够用——第一人称视角转相机时屏幕上根本没有指针，"按住 W 往前跑"也不是一次点按能表达的。Player-1 在保留原有动作的基础上做了三个扩展。

在第 $t$ 个回合，策略接收任务指令 $q$、当前截图 $I_t$ 和交互历史 $C_H(h_{<t})$，输出推理 $z_t$ 和动作 $\hat a_t$：

$$
(z_t,\hat a_t)\sim\pi_\theta(\cdot\mid q,I_t,C_H(h_{<t}))
$$

**（1）相对指针运动。** GUI 点击是绝对坐标（$x, y \in [0, 1000]$），游戏转视角需要的是位移量：

$$
\mathcal A_{\mathrm{pointer}} = \{\operatorname{absolute}(x,y)\} \cup \{\operatorname{relative}(\Delta x,\Delta y)\}
$$

其中 $\Delta x, \Delta y \in [-1000, 1000]$，按屏幕宽高归一化，正值分别表示向右、向下。实际视角转动幅度还取决于游戏的相机增益——这一点后面随机化时还会用到。

**（2）持续输入（hold）。** 点按（tap/click）之外，增加持续一个完整控制步长 $\Delta t$ 的按住动作：

$$
\mathcal A_{\mathrm{press}}(b) = \{\operatorname{tap/click}(b)\} \cup \{\operatorname{hold}(b;\Delta t)\}
$$

连续多步重复 hold，就实现了"一直按着 W 往前跑"这类持续移动。

**（3）组合动作（composite actions）。** 这是笔者认为最有游戏味儿的扩展：在一个控制步内 **并发执行** 两个或以上兼容的键鼠输入：

$$
\mathcal A_{\mathrm{step}} = \{\operatorname{wait}\} \cup \mathcal A_{\mathrm{keyboard}} \cup \mathcal A_{\mathrm{mouse}} \;\cup\; \{\mathbf{input}_1 \parallel \cdots \parallel \mathbf{input}_m\}, \quad m \ge 2
$$

比如"按住 W + 点按空格"是跑跳，"移动指针 + 按住右键 + 点击左键"是边瞄准边开火。人类玩家的操作本质上就是大量这种并发输入的组合，动作空间不支持并发，上限就被锁死了。

执行层面的语义也有明确规定：相对位移按 $(\Delta x, \Delta y) \mapsto (\Delta x W / 1000, \Delta y H / 1000)$ 映射到像素；同一个键在同一步里不能既 hold 又 tap；相机增益为 $g$（弧度/像素）时，请求的偏航/俯仰增量为 $|\delta\psi| = gW|\Delta x|/1000$、$|\delta\phi| = gH|\Delta y|/1000$。

### 拼图二：6,044 条精选演示做 SFT 初始化

动作空间定义了"能按什么"，但策略还得学会"什么时候按"。Player-1 从公开数据集（D2E、Gaming 500h、Open-P2P、VPT）和自录数据中精选了 **6,044 条游戏轨迹、295,097 个交互回合** ，覆盖 172 个游戏、2,544 个录制会话，折合采样控制时长约 16.4 小时。

![SFT 演示数据的来源与分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_data_sources.png)

> 图解：左图（a）展示各数据来源的轨迹规模与采样控制时长，右图（b）展示 172 个游戏的分布。可以看到数据来源多样，单个游戏的占比都不至于让数据集"一边倒"。

数据工程上有几个值得细说的点：

- **时间对齐** ：连续游戏录像的源时间戳被切分为 150–250 ms 的连续区间，再投影到统一的 200 ms 目标网格上；短于 100 ms 的按键记为 tap，更长的记为 hold，鼠标按下不超过 300 ms 记为 click。
- **空间对齐（鼠标校准）** ：不同录制源的鼠标输入尺度各异，作者利用同一会话中自由移动的 GUI 指针做回归，估计校准系数 $k_i = r_i^\top p_i \,/\, r_i^\top r_i$（$r_i$ 为原生输入，$p_i$ 为指针位移），把相对输入统一到像素等价位移。越界投影直接丢弃而非截断，保证动作质量。
- **指令与推理标注** ：用 Gemini 3.8 Flash 为每个游戏建立规则、角色、输入的可复用档案，再结合全轨迹 storyboard 和动作时间线生成任务指令；逐回合的视觉推理则基于指令、当前与历史画面、以及（关键的）已固定的动作目标来标注，避免"事后编理由"。

![演示数据的动作分布与标注样例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/control_v05_action_distribution.png)

> 图解：（a）按操控类型、每步输入个数、指针运动模式统计的动作分布——每步两个以上并发输入即构成组合动作，可以看出组合动作在数据中占了相当比例；（b）一条标注样例，录制的动作被配上了任务指令和逐回合视觉推理。

SFT 的样本构造为 $(q, I_t, C_H(h_{<t}), z_t, a_t)$，沿用 Game-TARS 的 **2+2 历史设置** ：最近 2 个"视觉+动作"回合 + 更早 2 个"仅动作"回合（历史截图替换为 `<image omitted>`，历史推理替换为 `<thinking omitted>`），在上下文长度和时序记忆之间取平衡。

另一个细节是 **动作平衡采样（action-balanced sampling）** ：人类录像里键盘输入和鼠标移动占绝对大头，左键点击这类关键动作占比不足 1%。作者把每个回合按动作类型分进若干有重叠的池（键盘、鼠标移动、点击、按住、键鼠组合等），每个操控池权重为 1、剩余"其他"池权重 0.1，按权重抽池、池内均匀回放，让小众但关键的动作获得足够曝光。

### 拼图三：Player-1-Gym，16 个合成环境喂 RL

SFT 让模型"会玩"，但要"玩得更好"就必须让它在自己的行为后果上学习——这就是合成环境的用武之地。

![Player-1-Gym 环境套件](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_environment_suite.png)

> 图解：（a）五大任务族共享的操控需求——瞄准、战斗、驾驶、导航、UI 交互；（b）16 个 2D/3D 环境的可视化，同一操控需求通过不同机制、视角和空间布局反复呈现；（c）同一环境施加外观、相机、场景随机化后的效果对比。

**环境构建。** 作者基于自研的共享 2D/3D 引擎（Three.js 浏览器运行时）、可复用素材和场景模板，实现了 16 个环境：瞄准 2 个、战斗 5 个、驾驶 4 个、导航 4 个、UI 交互 1 个。所有环境统一输出截图观测、执行同一套游戏操控动作空间，且仿真按固定步长推进、与推理延迟解耦——做 RL 的朋友会明白"固定步长仿真"对吞吐有多重要。

**可验证奖励。** 每个环境定义一个随游戏状态变化的累计分数 $S_e(s)$，逐步奖励就是分数增量：

$$
r_{e,t} = S_e(s_{t+1}) - S_e(s_t)
$$

奖励计算用的是特权状态（privileged state，即引擎内部数据），但策略只能看到截图和任务文本—— **判分靠上帝视角，决策靠肉眼** ，完全不需要学习型 judge。各类环境的计分规则一览：

| 2D 环境 | 任务与操控 | 计分项 |
| --- | --- | --- |
| 2D aiming | 追踪并点击移动目标 | 命中 $+0.1 \times$ 累计精度 |
| 2D shooting | 移动、瞄准、射击 | 击杀数 |
| 2D collect dodge | 收集金币并躲避障碍 | 金币数 |
| 2D lane driving | 把握变道时机 | 金币数 |
| 2D circuit driving | 沿赛道转向行驶 | 过检查点数 |
| 2D platforming | 跑、跳、冲刺 | 金币 + 跨沟/避刺 + 到达旗帜 |
| 2D UI crafting | 查配方、合成、装备 | 查阅配方 + 材料奖励 $+3 \times$ 合成 $+2 \times$ 装备 |

| 3D 环境 | 任务与操控 | 计分项 |
| --- | --- | --- |
| 3D aiming | 追踪并命中目标 | 命中数 |
| 3D combat arena | 协调移动、瞄准、开火 | 击杀数 |
| 3D road driving | 过弯与检查点 | 检查点数 |
| 3D lane driving | 变道收集与避让 | 金币数 |
| 3D spear duel | 追踪、蓄力、释放、闪避 | 长矛命中 $-0.5 \times$ 受伤 |
| 3D sword duel | 接近、瞄准、挥砍 | 命中 + 击杀 $-0.5 \times$ 受伤 |
| 3D melee arena | 协调朝向、距离与走位 | 命中 + 击杀 $-0.5 \times$ 受伤 |
| 3D navigation | 依次抵达旗帜 | 到达目标数 |
| 3D chase | 绕障碍追击 | 追达目标数 |

**环境随机化。** 为了让经验足够多样，作者在四个互补维度上联合随机化：

1. **外观（Appearance）** ：实体色相在 $\{0, 60, \dots, 300\}^\circ$ 离散采样，饱和度、明度、色温、环境光/主光强度等连续扰动，保持碰撞几何与功能角色不变；
2. **相机（Camera）** ：2D 正交缩放 $U(0.80, 1.00)$，3D 的 FOV 偏移 $U(-8, 8)^\circ$、相机架高度/跟随距离/俯仰偏移、初始视角偏航俯仰等；
3. **场景（Scene）** ：布局、出生点、任务特定配置（如赛道几何可镜像、配方与干扰项变化）；
4. **操控（Control）** ：相机转动增益 $L$、移动 $M$、驾驶 $D$、转向 $T$、跳跃冲量 $J$ 等输入响应乘子，独立采样自三角分布 $c_{e,k} \sim \operatorname{Tri}(a_{e,k}, 1, b_{e,k})$，整个 episode 内保持不变。

特别值得一提的是 **操控随机化** ：同样的鼠标位移在不同 episode 里会转出不同视角幅度，逼策略学会"看效果调力度"而不是背手感——这正是迁移到手感和灵敏度各异的商业游戏的关键。下面以 2D aiming 和 3D chase 两个环境为例展示随机化效果：

| 环境 | 原始 | 外观 | 相机 | 场景 | 联合随机化 |
| --- | --- | --- | --- | --- | --- |
| 2D aiming | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_2d_aim-original.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_2d_aim-appearance.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_2d_aim-camera.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_2d_aim-scene.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_2d_aim-all-1.png) |
| 3D chase | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_3d_navigate_chase-original.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_3d_navigate_chase-appearance.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_3d_navigate_chase-camera.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_3d_navigate_chase-scene.png) | ![](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/p1_gym_perturbations/base_3d_navigate_chase-all-1.png) |

> 图解：每一列只扰动一个维度（外观/相机/场景），最后一列是三者联合。可以看出外观变化不改变几何布局，相机变化改变视角取景，场景变化则直接改变任务布局与难度。操控响应的随机化无法体现在静态截图上，故未展示。

**多环境 PPO 训练。** 16 个环境的奖励尺度差异很大（金币数是几十分，决斗类带负惩罚），直接混训会让大奖励环境主导梯度。为此作者对广义优势估计（GAE）做 **分环境优势归一化**：

$$
\widetilde A_{i,j} = \frac{A_{i,j} - \mu_e}{\sqrt{\sigma_e^2 + \epsilon}}, \quad
\mu_e = \frac{\sum_{(i,j) \in \mathcal S_e} A_{i,j}}{N_e}, \quad
\sigma_e^2 = \frac{\sum_{(i,j) \in \mathcal S_e} (A_{i,j} - \mu_e)^2}{N_e - 1}
$$

其中 $i$ 索引回合、$j$ 索引响应 token，$\mathcal S_e$ 是环境 $e$ 在 rollout batch 中的响应 token 集合。此外奖励还带两个 shaping 项：

$$
\widetilde r_{e,t} = r_{e,t} - p^{\mathrm{invalid}}_{e,t} - p^{\mathrm{length}}_{e,t}, \qquad
G_{e,t} = \sum_{u=t}^{T} \widetilde r_{e,u}
$$

$p^{\mathrm{invalid}}$ 惩罚非法动作（执行时 fallback 为 wait），$p^{\mathrm{length}} = \min\{1, 0.005 \max(0, n^{\mathrm{reason}}_t - 64)\}$ 抑制过长推理。取 $\gamma = \lambda = 1$，return-to-go 在完整轨迹上累计，让后期结果反向指导早期的操控决策。PPO 损失先在回合内对 token 取平均、再跨回合取平均，保证每个环境按其包含的回合数等比例贡献。

**训练开销** ：SFT 用 32 张 H200 跑约 24 小时；200 步 RL 用 64 张 H200 约 36 小时；浏览器 rollout 由 32 个 worker、每个管理 16 个隔离 Playwright 上下文支撑，共 512 个并发环境。

## 实验：合成经验到底管不管用

### 训练与评测设置

两个模型分别从 Qwen3.5-9B 和 Qwen3.5-35B-A3B 初始化，先做 2 epoch SFT，再在 16 环境套件上做 200 步 PPO（每步 256 个 episode，均匀分配到各环境，每个 episode 至多 50 回合）。评测分四档：

| 评测面板 | 任务/地图 | 试次 | 决策预算 | 指标 |
| --- | --- | --- | --- | --- |
| Player-1-Gym（域内） | 16 个游戏 | 128 / 游戏 | 50 次决策 | 平均 episode return |
| MineDojo-300 | 300 个任务 | 8 / 任务 | 1,000 次决策 | 成功率；任务覆盖数 |
| ViZDoom-10 | 10 张地图 | 50 / 地图 | 150–2,100 次决策 | 缩放 return |
| GameWorld | 170 任务 / 34 游戏 | 3 / 任务 | 400 次决策 | 成功率；进度 |

后三个基准的环境 **全部在 Player-1-Gym 之外** ，其中 MineDojo-300 含 200 个战斗任务和 100 个采集任务，"任务覆盖"统计的是在 8 次试次中至少成功 1/2/4 次的任务数。基线包括游戏专用模型（JARVIS-VLA、NitroGen、Open-P2P、GameWAM）和 CUA 模型（UI-TARS-1.5-7B、Qwen3.5 基座），各基线保留其原生预处理与推理流程。

顺带一提，论文估算了评测成本：完整协议每个模型最多消耗 314 万个决策回合，若按前沿闭源 API 定价折合超过 10 万美元——这也解释了为什么评测聚焦开源/可本地部署的模型。

### MineDojo-300：RL 把成功率拉高 27.8 个百分点

| 方法 | 战斗 (%) | 采集 (%) | 总体 (%) | [95% CI] | ≥1/8 | ≥2/8 | ≥4/8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JARVIS-VLA | 7.56 | 5.00 | 6.71 | [5.38, 8.08] | 95 | 41 | 5 |
| UI-TARS-1.5-7B | 0.56 | 0.00 | 0.38 | [0.13, 0.67] | 8 | 1 | 0 |
| NitroGen | 0.00 | 0.00 | 0.00 | [-] | 0 | 0 | 0 |
| Open-P2P-1.2B | 0.44 | 0.00 | 0.29 | [0.08, 0.50] | 7 | 0 | 0 |
| GameWAM | 14.00 | 9.88 | 12.63 | [10.79, 14.54] | 155 | 79 | 16 |
| Qwen3.5-9B | 11.75 | 4.13 | 9.21 | [7.50, 11.00] | 106 | 59 | 13 |
| Qwen3.5-35B-A3B | 8.69 | 6.00 | 7.79 | [6.25, 9.46] | 99 | 50 | 8 |
| Player-1-9B-SFT | 12.50 | 6.63 | 10.54 | [8.67, 12.50] | 120 | 63 | 20 |
| Player-1-35B-A3B-SFT | 30.31 | 9.50 | 23.38 | [20.63, 26.21] | 172 | 133 | 76 |
| **Player-1-9B** | 52.31 | 10.38 | 38.33 | [35.13, 41.58] | 209 | 173 | 136 |
| **Player-1-35B-A3B** | **55.56** | **12.00** | **41.04** | [37.83, 44.29] | **224** | **182** | **142** |

> 表格解读：几个要点——① SFT 对 9B 的提升很有限（9.21% → 10.54%），演示数据本身并没有教会模型多少；② RL 带来了压倒性的增益，9B 提升 27.79 个百分点、35B-A3B 提升 17.7 个百分点；③ 收益主要来自战斗类任务（12.50% → 52.31%），这正对应合成环境里大量瞄准/战斗训练；④ 覆盖度上，"8 次中至少成功 4 次"的任务从 20 个涨到 136 个，说明提升不只是"运气变好"，而是 **可重复的稳定能力** 。

### ViZDoom-10 与 GameWorld：泛化到训练集之外

| 方法 | 精准射击 | 目标导航 | 移动战斗 | 生存 | 总体 |
| --- | --- | --- | --- | --- | --- |
| JARVIS-VLA | -0.6270 | -0.1459 | 0.1340 | 0.1565 | -0.1458 |
| UI-TARS-1.5-7B | -0.6648 | -0.1064 | 0.0587 | 0.1444 | -0.1743 |
| NitroGen | -0.3877 | -0.0732 | 0.0740 | 0.1471 | -0.0793 |
| Open-P2P-1.2B | -0.5385 | -0.0174 | 0.6493 | 0.1525 | 0.0602 |
| GameWAM | 0.1175 | -0.1285 | **1.1860** | 0.1424 | 0.3938 |
| Player-1-9B-SFT | 0.4403 | 0.1839 | 0.5293 | 0.5423 | 0.4361 |
| Player-1-35B-A3B-SFT | 0.7376 | 0.3019 | 0.5380 | 0.6065 | 0.5644 |
| **Player-1-9B** | **1.0772** | **0.5974** | 0.3033 | 0.7927 | 0.6922 |
| **Player-1-35B-A3B** | 1.0423 | 0.2628 | 0.5960 | **0.8979** | **0.7236** |

| 方法 | GameWorld 成功率 (%) | GameWorld 进度 (%) |
| --- | --- | --- |
| JARVIS-VLA | 6.47 | 17.61 |
| UI-TARS-1.5-7B | 5.49 | 19.71 |
| NitroGen | 8.82 | 23.41 |
| Open-P2P-1.2B | 2.35 | 10.37 |
| GameWAM | 5.88 | 16.91 |
| Player-1-9B-SFT | 13.14 | 31.43 |
| Player-1-35B-A3B-SFT | 15.49 | 34.60 |
| **Player-1-9B** | 15.88 | 34.45 |
| **Player-1-35B-A3B** | **17.25** | **38.79** |

> 表格解读：ViZDoom 是第一人称射击，GameWorld 是 34 个风格迥异的浏览器游戏，两者都与合成训练环境外观、机制完全不同。Player-1-35B-A3B 在 ViZDoom 总体缩放 return 上达 0.7236，比最强外部基线 GameWAM（0.3938）高 83.7%；值得注意的是 GameWAM 在"移动战斗"单项上仍领先（它本身就在 Doom 上训练过），这恰好反衬出 Player-1 的均衡性。GameWorld 上的提升幅度相对温和（进度 34.60% → 38.79%），但也保持了正向迁移。

### 案例：在从未见过的商业游戏里开车

![Player-1-9B 在 Watch Dogs 2 中的表现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/qualitative_watchdogs2_storyboard_4frames.png)

> 图解：Player-1-9B 在 **SFT 和 RL 都未见过** 的商业游戏《看门狗 2》中的一段连续行为：徒步导航 → 接近车辆 → 上车 → 城市驾驶。序列展示了导航中移动与相机的协调控制，以及上车、驾驶的指定按键操作。作者把 Player-1 以"截图观测 + 键鼠输出"的循环接入游戏，无需任何游戏内接口。

这个案例虽然只有定性展示，但说服力不弱：商业 3A 游戏的画面风格、相机手感、交互逻辑与合成环境天差地别，模型依然能完成"看到车—走过去—上车—开走"的连贯决策链，说明学到的确实是可迁移的操控先验。

## 深入分析：能力是怎么长出来的

### 学习动态：RL 增益早期最猛，SFT 初始化不可或缺

![学习动态与监督初始化的作用](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/learning_dynamics_with_initialization.png)

> 图解：（a, b）9B 与 35B-A3B 在 200 个 RL 步上的平均 episode return 与游戏得分（淡色为原始值，粗线为 EMA 平滑）：两条曲线都持续上升，早期增益最大、后期趋缓，35B-A3B 全程略高于 9B。（c）9B 上 Base→RL 与 SFT→RL 的对比：SFT→RL 从 17.67% 升到 51.67% 的 Minecraft 成功率，而 Base→RL 仅从 12.67% 爬到 16.67%——100 步时两者差距高达 **35 个百分点** 。

SFT 初始化的价值在这里体现得淋漓尽致：它提供的不只是起始分数，更是"能产出合法动作、能被奖励信号有效塑形"的策略起点。直接从基座做 RL，模型连有效探索都做不到。

![分环境的学习曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/per_environment_learning.png)

> 图解：16 个训练环境的逐环境 episode return 曲线。不同环境的学习进度、回报尺度和两模型相对表现差异很大：2D aiming 这类简单任务快速上升后进入平台期，而 2D circuit driving 的增益要到训练后期才显现。这也印证了分环境优势归一化的必要性。

### 环境 Scaling：环境越多，迁移越好

一个自然的问题是：在固定训练预算下，是把 episode 花在少数环境上练精，还是铺满 16 个环境练广？作者固定 25,600 个 episode（100 RL 步 × 每步 256 个），从同一 SFT checkpoint 出发，分别用 2、4、8、16 个环境训练：

| 环境数 | 环境子集 | 每环境每步 episode 数 |
| --- | --- | --- |
| 2 | 2D shooting；3D navigation | 128 |
| 4 | 上述 2 个 + 2D circuit driving；3D aiming | 64 |
| 8 | 上述 4 个 + 2D UI crafting；2D platforming；3D sword duel；3D road driving | 32 |
| 16 | 完整套件 | 16 |

![环境数量 scaling](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/environment_scaling.png)

> 图解：横轴为 RL 步数，纵轴为 Minecraft 留出任务上的成功率变化（0 为 SFT 参照）。在 **总 episode 预算完全相同** 的条件下，环境覆盖越广，迁移到域外游戏的效果越好——哪怕每个环境分到的交互量只有原来的 1/8。这说明跨环境共享的是可迁移的操控能力，而不是对单一环境的过拟合。

### 行为演化：组合动作自己"长"出来了

![RL 过程中动作组合的演化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Player-1-Learning-to-Play-from-Synthetic-Experience/figures/analysis_agent_behavior.png)

> 图解：9B 在套件内验证轨迹中，组合动作（一步内并发多个键鼠输入）的回合占比从 RL 开始时的 64.7% 升至 50 步后的 85.4%、100 步后的 93.2%。柱状图展示各 checkpoint 最高频动作的占比（K-H：按键 hold；M-M：鼠标移动；M-LC：鼠标左键点击）。

笔者认为这是全文最有趣的发现之一： **奖励里没有任何一项鼓励组合动作** ，但随着任务驱动 RL 的推进，"边移动边转视角边开火"这类协调输入的使用率自发攀升至九成以上。换句话说，组合动作不是被教会的技巧，而是游戏操控需求的 **自然解** ——动作空间给了表达能力，奖励给了选择压力，行为复杂度就自己涌现了。

## 总结与展望

- **问题** ：游戏智能体要么困于演示数据覆盖不足，要么困于商业游戏奖励难以程序化，交互学习难以规模化。
- **方案** ：Player-1 以"共享游戏操控动作空间（相对指针 + 持续输入 + 组合动作）"为轴，串联 6,044 条精选演示的 SFT 初始化与 16 个合成环境的多环境 PPO（RLVR）。
- **关键机制** ：可程序化验证的分数增量奖励、四维联合随机化（外观/相机/场景/操控）、分环境优势归一化、跨回合 return-to-go 信用分配。
- **核心结果** ：MineDojo-300 上 9B 提升 27.8 个百分点至 38.33%（最强基线 12.63%）；ViZDoom-10 上 35B-A3B 超最强基线 83.7%；增益泛化到训练套件之外的浏览器游戏与商业游戏。
- **规律发现** ：环境覆盖越广迁移越好；组合动作使用率在 RL 中自发从 64.7% 升至 93.2%。

这项工作把"合成经验"确立为学习可迁移游戏操控的有效资源——不必在每个目标游戏里做在线训练。当然，局限也清晰可见：采集类（Harvest）任务提升远小于战斗类，长程规划与资源管理并不是当前合成环境重点覆盖的能力维度；如何在合成环境中纳入更长视野的任务结构，恐怕是这条路线下一个必须回答的问题。

> 本文参考自 [Player-1: Learning to Play from Synthetic Experience](https://arxiv.org/abs/2610.20584)