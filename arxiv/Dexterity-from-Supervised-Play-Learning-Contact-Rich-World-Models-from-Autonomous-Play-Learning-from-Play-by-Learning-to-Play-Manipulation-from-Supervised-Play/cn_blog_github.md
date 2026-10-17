# Dexterity from Supervised Play：监督玩耍练就灵巧操作

让多指手自主学会转动物体，最难的往往不是再设计一个控制器，而是获得足够多、足够安全、足够多样的接触数据；这类数据正是灵巧操作从人工演示走向自主学习的瓶颈。DSP（Dexterity from Supervised Play）的思路是：先用几分钟无目标的人类遥操作教会机器人“怎么玩”，再让它自主玩耍数小时，用收集的数据训练 action-conditioned world model，并由同一个 play policy 为模型预测控制提供候选动作。在仿真任意姿态立方体重定向中，DSP 只用约 70.1 万帧就取得 91.4% 的 SR@0.6 rad 和 72.7% 的 SR@0.4 rad，而 hindsight BC 分别只有 53.9% 和 25.8%。达到相近 SR@0.6 rad 的 PPO-PM 则需要 3.44 亿帧，交互帧数约为 DSP 的 490 倍。

![DSP 总流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/teaser.png)

> 图解：DSP 包含五个环节。首先用无目标遥操作训练 play policy；随后由它自主收集交互数据；数据用于训练 Diffusion Forcing world model；推理时，play policy 提出候选动作，world model 预测结果，MPC 按目标距离排序并执行；最后还可以在世界模型中 self-play，把规划器蒸馏成 goal-conditioned closed-loop policy。

## 问题：灵巧操作缺的不是目标，而是有效交互

灵巧操作需要持续接触、滑动、重新抓取和失误恢复。传统路线通常有两笔很高的成本：人工遥操作的成本随采集时长线性增长；强化学习虽然能自己生成交互，却往往依赖任务奖励、仿真器和数亿次探索。

“让机器人自己玩”看起来是第三条路，但 play 很难写成规则。什么叫有用但不过激的探索？怎样既持续接触物体，又避免损坏硬件？手工启发式规则和安全边界不仅难设计，也很难跨任务复用。

论文的核心假设是：play 可以被建模为马尔可夫决策过程（MDP）中的、以状态为条件的 goal-agnostic 概率策略。它不需要目标标签，也不依赖任务奖励，因此可以直接用 Behavior Cloning（BC，行为克隆）从人类演示中学出来。

这个定位与几类已有工作有明显区别：

- **与 MuZero、self-play RL 相比**，后者证明了交互与规划的力量，但通常依赖明确的规则、奖励或仿真器；DSP 面向难以设计奖励和安全探索的 contact-rich manipulation。
- **与 curiosity 相比**，预测误差或 Random Network Distillation 能鼓励新奇状态，却不保证机器人持续、安全地与物体接触；DSP 直接从人类 play 中学习交互结构。
- **与 Play-LMP 等 learning-from-play 方法相比**，已有方法多把 play 当作固定的模仿数据集；DSP 把少量人类 play 变成可自主运行的数据收集器。
- **与灵巧操作 RL 相比**，DSP 不靠任务专用演示和奖励，也不要求先把任务做成高保真仿真，再从仿真迁移到现实。

换句话说，人类不再负责演示“每个任务怎么做”，而是只负责展示“怎样持续进行有意义的交互”。解决了 play 的来源之后，下一个问题就是：如何把这些无目标交互变成可复用的物理知识和目标导向控制？

## 方法：把“会玩”变成数据、模型与控制器

DSP 的主链路有四个阶段，另有一个可选的蒸馏阶段：Supervised Play → Autonomous Play → World Modeling → MPC，以及可选的 Policy Distillation。

![DSP 方法流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/method.png)

> 图解：(a) 从无目标人类演示学习 $\pi_{\mathrm{play}}$；(b) 策略与环境闭环交互，记录 $(o_t,a_t,o_{t+1})$；(c) world model 根据历史观测和动作预测后续观测；(d) 推理时并行提出多组动作，分别预测结果并计算目标奖励，选出最优动作执行，再从新观测重新规划。

### Supervised Play：只教交互先验，不教任务答案

设 $o_t$ 为机器人和物体的观测，$a_t$ 为记录的关节命令。作者在无目标人类遥操作数据集 $\mathcal{D}_{\mathrm{human}}$ 上，通过 BC 训练 Diffusion Transformer play policy。策略根据历史观测，一次采样长度为 $H$ 的 action chunk：

$$
\hat{a}_{t:t+H-1} \sim \pi_{\mathrm{play}}(\cdot \mid o_{\leq t})
$$

这里最关键的是训练数据不指定最终目标。演示只需要包含持续接触、移动物体、切换抓取等行为，不需要展示“把立方体转到某个指定姿态”的完整解法。

这样设计的好处是，人类较容易提供演示，策略也不会被绑定到单一目标。安全行为同样不作为显式规则注入，而是隐含在演示中；不过，这意味着它的安全性仍取决于数据覆盖，而不能被理解为形式化保证。

### Autonomous Play：把几分钟演示扩展成小时级数据

训练好的 play policy 被直接部署到机器人上，自主滚动并收集 $\mathcal{D}_{\mathrm{play}}$。每条数据记录感觉观测、本体感觉和动作造成的转移：

$$
(o_t,a_t,o_{t+1})
$$

人类监督此时不再随采集时长同步增加，采集还可以并行到多台机器人和多个环境。笔者认为，这一设计真正改变了人类演示的角色：演示不是终点，而是启动自主经验积累的“火种”。

### World Modeling：学习动作后果，而不是背下轨迹

仅有 play 数据还不能主动达成新目标。DSP 因而训练 action-conditioned world model，根据最近 $C$ 步观测和输入动作序列，预测未来 $K$ 步观测：

$$
\hat{o}_{t+1:t+K}=\mathcal{M}_{\mathrm{wm}}(o_{t-C+1:t},a_{t:t+K-1})
$$

其通用训练目标是对各预测时刻的误差加权求和：

$$
\mathcal{L}(\phi)=\mathbb{E}_{(o,a)\sim\mathcal{D}_{\mathrm{play}}}\left[\sum_{k=1}^{K}\lambda_k\ell(\hat{o}_{t+k},o_{t+k})\right]
$$

其中，$\phi$ 是模型参数，$\lambda_k$ 是不同预测步的权重，$\ell$ 是观测预测误差。模型学习的是“执行这组动作后会发生什么”，而不是“这条轨迹对应哪个任务”。

具体训练采用 Diffusion Forcing：给不同时间步的观测分配独立噪声等级，再进行因果去噪。这样，模型可以条件于干净的已观测历史，同时逐步生成未来观测，并支持候选动作下的 autoregressive rollout。

### Play-guided MPC：让同一个 play policy 充当动作先验

MPC（Model Predictive Control，模型预测控制）可以想象成“先在脑中试走多条路，再执行当前最好的一段”。DSP 的特色是，候选路不是随机撒出来的，而是由 play policy 提供，因此搜索集中在已演示过的可行接触行为附近。

在时刻 $t$，系统采样 $N$ 个候选 action chunk，并分别预测未来 $K\leq H$ 步：

$$
\hat{a}^{(i)}_{t:t+H-1}\sim\pi_{\mathrm{play}}(\cdot\mid o_{\leq t}),\qquad i=1,\ldots,N
$$

$$
\hat{o}^{(i)}_{t+1:t+K}=\mathcal{M}_{\mathrm{wm}}(o_{t-C+1:t},\hat{a}^{(i)}_{t:t+K-1})
$$

给定目标观测 $o_g$，每个候选取预测区间内最好的目标奖励，再选出最优候选：

$$
r_i=\max_{1\leq k\leq K}r(\hat{o}^{(i)}_{t+k},o_g),\qquad i^{\star}=\arg\max_{1\leq i\leq N}r_i
$$

系统只执行最优 chunk 的前 $E$ 个动作（$1\leq E\leq K$），获得新观测后重新规划。目标只进入奖励函数，play policy 和 world model 本身始终保持 goal-agnostic；更换目标姿态时，不需要重新训练这两个模型。

![play-guided planning 涌现的灵巧技能](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/atomic_skills.png)

> 图解：五列分别是 yaw rotation、edge flip、corner flip、fine adjustment 和 composite action；每列时间自上而下推进，黄色箭头表示物体运动。它们不是逐一演示、逐一命名的任务技能，而是 MPC 从共享的 play 动作库中，按预测结果组合出来的行为。

这个设计的聪明之处在于分工：play policy 负责“提出像样子的动作”，world model 负责“判断动作后果”，MPC 负责“按当前目标做选择”。三者各自可复用，也避免让单一策略同时承担探索、物理预测和目标控制。

### Policy Distillation：把在线搜索摊销为一次策略前向

MPC 每次重规划都要评估大量候选，推理成本较高。DSP 的可选方案是在已学习的 world model 中 self-play：从已记录的观测历史和采样目标出发运行 MPC，用模型预测推进历史，并收集 $(h,o_g,\hat{a}^{\star})$。其中，$h$ 是当前历史，$\hat{a}^{\star}$ 是规划器选中的动作前缀。

随后用 BC 训练 goal-conditioned policy $\pi_\theta$：

$$
\mathcal{L}_{\mathrm{distill}}(\theta)=-\mathbb{E}_{(h,o_g,\hat{a}^{\star})\sim\mathcal{D}_{\mathrm{distill}}}\left[\log\pi_\theta(\hat{a}^{\star}\mid h,o_g)\right]
$$

部署时，蒸馏策略直接从真实观测历史和目标预测动作前缀，再用新观测闭环更新。它不是完全取代 MPC，而是用一部分精度换取更低的在线计算量。

方法已经闭环，接下来需要回答三个实证问题：它能否跨任务工作？play-guided planning 是否真的比直接模仿更高效？真实硬件上的数据与延迟是否可接受？

## 实验设置：四项任务与统一的重定向指标

论文将实验分为四项任务：

- **Task A：Push-T**。在固定、紧张的 10K 帧遥操作预算下，把平面 T 形块推到目标位姿，衡量成功率和最佳 mask overlap。
- **Task B：仿真重定向**。多指手把立方体从初始姿态转到目标姿态，且初始误差大于 $90^{\circ}$。
- **Task C：真实重定向**。检验同一框架能否在硬件上完成任意目标重定向，并保持接近仿真的表现。
- **Task D：视觉重定向**。play policy 改看 RGB 图像，world model 在学习到的 64 维 latent 表示中评估候选动作，检验机制能否脱离显式物体位姿。

重定向误差定义为当前旋转 $R_t$ 与目标旋转 $R_g$ 在 $\mathrm{SO(3)}$（三维旋转空间）上的距离：

$$
e_t=d_{\mathrm{SO(3)}}(R_t,R_g),\qquad e_0>90^{\circ}
$$

$e_0>90^{\circ}$ 排除了短距离、容易碰巧成功的试验。常用成功率阈值为 0.4、0.6、0.8 rad，分别约为 $22.9^{\circ}$、$34.4^{\circ}$、$45.8^{\circ}$。

仿真使用 Isaac Lab。Task B 的每个控制器都在相同的 128 个 seeded trials 上评测，每次 2,500 个控制帧；方块掉落后会重生并继续试验。机器人观测手部状态和立方体位姿，控制 22 个关节位置。规划器每次采样 1,024 个 play proposals，预测 50 步、执行 25 步。Task D 使用 64 次试验和 3,600 步时域，因此它不是与 Task B 严格对齐的 RGB-versus-state 对比。

![仿真与真实硬件平台](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/system.png)

> 图解：左侧为 Push-T 与多指手掌内重定向仿真；右侧真实平台包含安装在 Tianji Marvin M6 机械臂上的 Sharpa Hand、用于遥操作的 Manus Glove，以及 ZED 2i Camera。真实手部同样控制 22 个关节，立方体上的标记用于六自由度位姿跟踪，真实试验时间预算为 120 秒。

基线也按问题设计得很清楚：Push-T 比较 BC、hindsight BC、goal-conditioned BC 和同等数据量 RL；仿真重定向比较直接执行 play policy、hindsight BC、15 分钟任务演示训练的 goal-conditioned BC、同帧数 PPO-TM，以及性能对齐的 PPO-PM；真实实验则比较关节随机游走、平均角速度对齐的 $\mathrm{SO(3)}$ 随机游走和 play policy。

## 实验结果：规划把无目标探索变成目标达成

### Push-T：优势随精度要求提高而扩大

先看一个直观的规划例子。

![Push-T 中的候选动作与预测轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/mpc_proposals_seed3000.png)

> 图解：左图橙色轨迹是 play policy 从当前状态提出的候选 action chunk；右图蓝色轨迹是 world model 预测的 T 形块运动，黄色轨迹是 MPC 选中的方案。规划不是让 play policy 直接对准目标，而是从多种可能后果中挑出更能接近目标的一条。

定量结果如下。SR@0.90 与 SR@0.95 对应不同的覆盖阈值，后者更严格；Coverage 表示覆盖度。

| 控制器 | SR@0.90（%） | SR@0.95（%） | Coverage |
|---|---:|---:|---:|
| Behavior Cloning | 64 | 28 | 0.751 |
| Hindsight BC | 74 | 36 | 0.764 |
| Goal-conditioned BC | 60 | 34 | 0.733 |
| RL, matched data | 0 | 0 | 0.155 |
| DSP | 98 | 74 | 0.953 |

DSP 相比最强的直接策略基线 hindsight BC，在 SR@0.90 上领先 24 个百分点，在 SR@0.95 上领先 38 个百分点，平均 Coverage 也从 0.764 提升到 0.953。差距在严格阈值下继续扩大，说明 model-based selection 改善的不只是“推到附近”，更是精确放置。相同交互预算下的 RL 成功率为 0，表明这一数据规模不足以支撑任务专用奖励学习。

### 仿真重定向：少量交互换来高目标到达率

![仿真重定向成功率与交互帧数对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/sim_reorientation_combined.png)

> 图解：左图纵轴为成功率，横轴依次为 SR@0.4、0.6、0.8 rad；DSP 在三档分别达到 72.7%、91.4%、96.1%。右图纵轴为 SR@0.6 rad，横轴为对数刻度的训练帧数；DSP 用约 701K 帧达到 91.4%，PPO-PM 用 344M 帧达到 86.7%，PPO-unlimited 达到 96.9%。

直接执行 play policy 在三档阈值上只有 15.6%、36.7%、54.7%；hindsight BC 提升到 25.8%、53.9%、71.9%，仍明显低于 DSP。goal-conditioned BC 为 21.9%、34.4%、50.0%，说明仅靠目标条件模仿并不能替代在线规划。

这组对比揭示了 play 数据与目标控制之间的缺口：数据中确实有有用动作，但事后贴目标标签，不等于会在当前状态下组合这些动作。MPC 能显式比较多个反事实未来，因此更容易把局部动作组织成朝向指定姿态的长过程。

与 PPO 的关系则不是“全面碾压”。同为 701K 帧时，PPO-TM 在三档只有 0%、0%、1.6%；但要达到 86.7% 的 SR@0.6 rad，PPO-PM 需要 344M 帧。与此同时，PPO-unlimited 的 96.9% 说明，在可仿真、交互几乎不限且奖励明确的任务上，任务专用 RL 仍能取得很强的精细收敛能力。DSP 的突出优势是交互效率，以及不依赖任务仿真和 sim-to-real transfer。

### 真实重定向：有效来自接触结构，而非单纯“动得快”

![真实与仿真重定向结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/draft/real_reorientation.png)

> 图解：横轴为三档成功阈值，纵轴为成功率。关节随机游走始终为 0；角速度对齐的 $\mathrm{SO(3)}$ 随机游走为 6.3%、11.4%、17.5%；play policy 为 20.9%、38.3%、54.1%；真实 DSP 达到 68.6%、90.0%、97.1%，斜线柱所示仿真 DSP 为 72.7%、91.4%、96.1%。

$\mathrm{SO(3)}$ 随机游走已经对齐了 play policy 的平均角速度，却仍大幅落后。这排除了“只要转得快，数据就有用”的解释，说明关键更可能是学到的持续接触、重抓和协调运动结构。

加入 world-model search 后，真实 DSP 相比单独 play policy 分别提高 47.7、51.7、43.0 个百分点。真实与仿真结果在每个阈值上的绝对差都不超过 4.1 个百分点，表明闭环重规划能够在一定程度上修正模型误差和执行误差。

![真实 Sharpa 手上的任意姿态重定向过程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/dis_to_goal.png)

> 图解：两组真实试验中，图像序列下方的曲线横轴为时间（秒），纵轴为到目标的角度误差（度）。误差中途会明显上升，但最终仍下降并接近目标；右侧 $\mathrm{SO(3)}$ 轨迹以紫色点、红色星和黄绿色星分别标出起点、目标和终点，展示了完整旋转路径。目标导向并不意味着误差每一步都单调下降，有时需要先绕开当前抓取姿态。

### 视觉重定向：机制可以迁移到 RGB，但表格口径需谨慎

![RGB world model 的三秒预测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/wm_pred.png)

> 图解：上排是 Ground Truth，下排是 World Model Prediction，三列分别对应 1、2、3 秒。预测图像保留了多指手与彩色方块的大致位姿变化，说明模型能够在像素空间生成合理的未来观测；控制实验中，候选评估则使用学习到的 64 维 latent 表示。

论文正文对视觉结果的叙述是：RGB policy 单独达到 34.4%、53.1%、76.6%；加入 world-model MPC 后达到 56.2%、78.1%、89.1%，分别提高 21.8、25.0、12.5 个百分点。原表数据如下，行标签按原文保留。

| 控制器（原表标签） | SR@0.4 rad | SR@0.6 rad | SR@0.8 rad |
|---|---:|---:|---:|
| RGB-space DSP | 34.4% | 53.1% | 76.6% |
| State-space DSP | 56.2% | 78.1% | 89.1% |

需要指出，原表第二行标为“State-space DSP”，正文却把同一组数字解释为 RGB policy 加入 MPC 后的结果，两处口径并不一致。因此，这里不把它当作严格的状态空间对照结论。更稳妥的读法是：实验支持 play prior 与 world-model selection 能迁移到视觉表示，但 Task D 的 64 次试验、3,600 步时域与 Task B 不同，不能据此做协议对齐的 state-versus-RGB 排名。

## 消融与效率：为什么 Diffusion、更多数据和蒸馏重要

### 好的 play policy 要覆盖转移，不只是覆盖状态

| Transformer Policy | 平均旋转速度（deg/s） | $\mathrm{SO(3)}$ 状态覆盖（%） | 转移覆盖（%） |
|---|---:|---:|---:|
| Deterministic | 57.8 | 87.6 | 55.4 |
| Diffusion | 124.9 | 98.8 | 84.2 |

确定性与 Diffusion policy 的 held-out command error 很接近，分别为 0.192 和 0.181，但闭环数据差异很大。Diffusion 的旋转速度翻倍以上，状态覆盖提高 11.2 个百分点，转移覆盖更从 55.4% 提升到 84.2%。

这意味着确定性策略可能以较窄的运动模式到达许多姿态，呈现出类似 mode averaging 的行为。对 DSP 而言，转移多样性尤其重要：它既让 world model 学到更充分的动作条件，也给 MPC 提供多条到达目标的路径。只看开环模仿误差，会漏掉这两个关键性质。

### 更多自主 play 仍能改善预测，但收益逐渐减小

![真实机器人自主 play 数据规模与世界模型误差](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Dexterity-from-Supervised-Play-Learning-Contact-Rich-World-Models-from-Autonomous-Play-Learning-from-Play-by-Learning-to-Play-Manipulation-from-Supervised-Play/figures/real_wm_scaling.png)

> 图解：横轴为自主 play 数据量（小时，对数刻度），纵轴为 50 步旋转 MAE（度）。数据从 0.54、1.03、2.05、3.30 增至 5.11 小时时，误差依次为 $34.96^{\circ}$、$33.13^{\circ}$、$30.34^{\circ}$、$29.10^{\circ}$、$28.40^{\circ}$，整体单调下降。

在模型和 held-out evaluation 固定的情况下，0.54 到 5.11 小时的数据扩展带来 18.8% 的 MAE 降低；最终模型也比 $38.60^{\circ}$ 的 persistence baseline 低 26.4%。主要收益出现在前两小时，之后曲线变平但仍未停止改善，说明自主采集确实能持续补充物理知识，只是在当前规模上出现边际收益递减。

### 实时性来自批量评估与动作块，蒸馏进一步降延迟

真实系统的感知和规划通过局域网运行在同一张 RTX 5090 上。各阶段延迟如下，单位为毫秒。

| 阶段 | Median（ms） | p95（ms） |
|---|---:|---:|
| Visual Cube Pose Tracking | 19.4 | 29.4 |
| Play policy（1,024 个 proposal chunks） | 60.5 | 64.3 |
| World model（$1,024\times25$ 步 rollout） | 44.2 | 50.3 |
| MPC scoring and selection | 23.4 | 26.4 |
| Replan, end to end | 133.6 | 141.1 |

视觉跟踪能跟上 30 fps 相机，单次端到端重规划中位数约 134 ms。系统并没有让机械手停下来等待完整搜索，而是把批量候选评估与 action chunking 结合起来：执行当前动作前缀时并行计算下一段计划，最终闭环运行达到 15–18 Hz。

蒸馏则进一步压缩在线成本。作者在 54K 个已记录状态上，各配 24 个按角度距离均匀采样的目标，完全在 world model 内生成约 1.3M 个规划标签，不需要新交互，也不需要物理引擎。蒸馏策略取得 40.6%、72.7%、86.7% 的 SR@0.4/0.6/0.8 rad，其中 0.8 rad 档已接近 MPC 超过 90% 的表现；单次运行仅需 21.8 ms，按原文口径约比 MPC 重规划快 7 倍。

## 讨论：DSP 与 BC、RL、好奇心的边界

与 BC 相比，DSP 最重要的变化不是多加一个规划器，而是改变监督的用途。goal-agnostic 不等于无结构：维持接触、安全探索和抓取切换本身就包含物理知识。人类提供这些行为结构，机器人再通过自主交互学习动作后果，因此一份演示可以支持远超其原始时长的数据生产。

与任务专用 RL 相比，DSP 把“获得物理知识”和“指定任务目标”分开。前者由自主 play 和 world model 完成，后者只在规划奖励中出现，所以同一套交互先验和动力学可以服务不同目标，无需逐目标重训。不过，PPO-unlimited 的结果也提醒我们：当任务可仿真、奖励明确且交互充足时，RL 仍是很强的方案。

与 curiosity 或随机探索相比，DSP 不要求“有用探索”从通用新奇性目标中自行涌现。人类直觉负责给出接触行为的形式，自主经验负责补齐后果模型。这为具身智能提供了一种更务实的分工，而不是把安全、有效的接触完全交给随机搜索。

由此还可以自然连接到 behavior foundation models：一个更大的预训练行为模型可以提供通用 play repertoire，再用特定机器人和环境中的自主经验落地其动作后果。当然，阅读当前结果仍需保留两点谨慎：仿真中的掉落方块会重生，成功率更接近“试验中曾到达目标”，不等于全程稳定保持；论文结论也明确说明，真实硬件汇总仍需冻结协议和不确定性统计，部分最终定量结果与 curiosity、匹配架构消融仍待补齐。

## 结论与展望

- DSP 用几分钟无目标人类 play 启动自主数据采集，把人类监督从“任务答案”转变为“交互先验”。
- 同一个 play policy 既负责收集数据，又为 MPC 提供动作候选；world model 预测后果，目标只在规划奖励中出现。
- 在仿真重定向中，DSP 以约 701K 帧达到 91.4% 的 SR@0.6 rad 和 72.7% 的 SR@0.4 rad；性能对齐的 PPO-PM 需要约 490 倍帧数。
- 真实 DSP 达到 68.6%、90.0%、97.1% 的三档成功率，整体接近仿真；Push-T 与 RGB 实验进一步展示了跨任务、跨表示的潜力。
- Diffusion 带来的转移覆盖、自主数据增长带来的预测改善，以及 21.8 ms 的蒸馏推理，共同说明这套框架的效果来自多个可独立优化的模块。

未来若能用 behavior foundation model 提供更通用的 play policy，并补齐冻结评测、多种子不确定性与稳定保持指标，supervised play 才有机会从一套灵巧操作流程扩展为可复用的具身学习基础设施。

> 本文参考自 [Dexterity from Supervised Play](http://arxiv.org/abs/2610.22340v1)