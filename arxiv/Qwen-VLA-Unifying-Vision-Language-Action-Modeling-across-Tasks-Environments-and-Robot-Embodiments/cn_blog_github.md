# Qwen-VLA 深度解读：一个模型统一机器人操作、导航与轨迹预测

## 论文核心结论

Qwen-VLA 试图回答一个关键问题：机器人操作、视觉语言导航、第一视角人类动作建模和轨迹预测看起来输出形式完全不同，能否用一个统一的 Vision-Language-Action 模型来建模？

答案是可以。它把这些任务都抽象为：给定视觉观察、语言指令和机器人本体描述，预测未来一段动作或轨迹。

最值得关注的结果是，Qwen-VLA-Instruct 作为一个 generalist policy，在多个任务族上同时取得了很强表现：

| 任务 | 结果 |
|---|---:|
| LIBERO | 97.9% |
| Simpler-WidowX | 73.7% |
| RoboTwin Easy / Hard | 86.1% / 87.2% |
| R2R Val-Unseen OSR | 69.0 |
| RxR Val-Unseen SR | 59.6 |
| ALOHA 真实机器人 OOD 平均成功率 | 76.9% |
| DOMINO 动态操作 Zero-shot SR | 26.6% |

这篇论文的重点不只是“指标高”，而是提出了一套可扩展的 VLA 统一建模范式：用强 VLM 负责感知、语言理解和空间推理，用基于 DiT 的 flow matching action decoder 负责连续动作生成，再通过 embodiment-aware prompt 将不同机器人平台、控制频率、动作维度和预测 horizon 接入同一个模型。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Qwen-VLA-Unifying-Vision-Language-Action-Modeling-across-Tasks-Environments-and-Robot-Embodiments/figures/qwenvla_overview_final.png)

> 图解：这张图展示了 Qwen-VLA 的整体思路。模型接收图像、语言指令和机器人本体描述，既能输出文本回答，也能输出连续动作轨迹。左侧强调多源数据混训，包括机器人操作、导航、人类第一视角演示和通用视觉语言数据；右侧强调同一模型可以覆盖不同任务和不同机器人 embodiment。

## 背景：为什么需要统一 VLA？

当前 Embodied AI 的一个核心问题是模型碎片化。操作机器人有操作专用模型，导航有导航专用模型，双臂机器人、单臂机器人、移动机器人、灵巧手又往往各自训练一套策略。这会带来三个问题：

1. 任务之间难迁移：操作模型学到的空间 grounding 很难直接迁移到导航，导航中的长程规划能力也难以迁移到操作。
2. 机器人之间难共享：不同平台的 action space 不同，比如末端位姿、关节角、夹爪状态、waypoint、手部姿态，很难直接合并训练。
3. 数据规模难做大：如果每个任务和平台都单独建模，数据利用率会很低，无法像 VLM 那样通过大规模混合预训练获得通用能力。

Qwen-VLA 的核心洞察是：这些任务表面形式不同，但底层结构相似。它们都需要模型根据视觉观察 $o_t$、语言指令 $x$、本体描述 $e$ 和可选任务标识 $z$，预测未来一段动作或轨迹 $y_{t:t+H-1}$：

$$
p_\theta(y_{t:t+H-1} \mid o_t, x, e, z)
$$

这里的 $y$ 可以是机械臂动作、导航 waypoint、人类手腕轨迹、手部姿态、车辆未来轨迹等。只要把它们都看成未来一段连续序列，就可以用同一个 action decoder 来学习。

## 模型架构：VLM 负责理解，DiT 负责动作

Qwen-VLA 的架构可以拆成两个部分：

- Vision-language backbone：基于 Qwen3.5-4B 多模态模型，负责图像理解、语言指令理解、目标 grounding 和空间关系推理。
- Action expert：一个 DiT-style flow matching decoder，负责生成连续动作序列。

### Vision-Language Backbone

Qwen3.5 本身是 natively multimodal backbone，视觉 token 会和文本 token 一起进入 Transformer。这个设计对 VLA 很重要，因为机器人任务往往不是简单识别物体，而是需要理解“红色杯子左边的碗”“按颜色顺序堆叠”“把饮料集中到一起”这类语言绑定空间关系的指令。

论文特别强调，VLA 不只是 action prediction，前面的 perception、referential grounding 和 instruction following 能力也很关键。因此，Qwen-VLA 没有从零训练感知模型，而是直接站在强 VLM 的基础上扩展动作生成。

### DiT-Based Action Expert

动作模块使用 DiT-style flow matching policy。它会把 VLM hidden states 和带噪声的 action chunk 拼接成一个序列，通过 self-attention 处理，并用 AdaLN 做 timestep conditioning。

训练时，模型不是直接回归动作，而是学习一个 velocity field。给定真实动作 $\mathbf{Y}_0$ 和高斯噪声 $\mathbf{Y}_1$，构造插值：

$$
\mathbf{Y}_{\tau} = (1-\tau)\mathbf{Y}_0 + \tau \mathbf{Y}_1
$$

模型学习预测从噪声到真实动作的方向：

$$
v_\theta(\mathbf{Y}_{\tau}, \tau \mid o_{1:t}, x, e, z) \approx \mathbf{Y}_1 - \mathbf{Y}_0
$$

核心 action loss 是：

$$
\mathcal{L}_{act}
=
\mathbb{E}_{\tau,\mathbf{Y}_0,\mathbf{Y}_1}
\left[
\frac{1}{c}
\sum_{k=0}^{c-1}
\ell_k
\right]
$$

其中 $c$ 是当前 embodiment 的有效 action channel 数。论文采用 per-channel、per-step 的 mask loss，避免 zero-padding 的无效维度主导梯度。

推理时，模型从噪声开始，通过少量 Euler integration step 生成动作 chunk，因此延迟较低，适合实时控制。

## 关键设计一：Embodiment-Aware Prompt

不同机器人平台最麻烦的地方在于 action semantics 不统一。例如：

| 平台类型 | 可能的动作格式 |
|---|---|
| 单臂机械臂 | $\Delta$ end-effector pose + gripper |
| 双臂机器人 | left/right arm actions + gripper |
| Humanoid | joint positions + hand states |
| VLN 移动机器人 | $(\Delta x, \Delta y, \Delta \theta)$ waypoint |
| 人类第一视角数据 | wrist SE(3) + hand eigengrasp |

Qwen-VLA 没有为每个机器人单独加一个 head，而是把平台信息写进 prompt。例如：

```text
The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile base].
The control frequency is {FPS} Hz.
Please predict the next {chunk_size} control actions to execute the following task: {instruction}.
```

这个设计的好处是，模型架构完全不变。机器人平台、控制频率、动作 horizon 和控制约定都通过文本描述告诉模型。换句话说，embodiment prompt 是唯一的平台特定接口。

这也是 Qwen-VLA 能做 multi-embodiment co-training 的关键：模型可以共享视觉 grounding 和空间推理能力，同时根据 prompt 切换不同动作语义。

## 关键设计二：统一 Action and Trajectory Representation

论文没有强行把所有控制信号转成同一种物理语义，而是统一成同一个 tensor interface。

每个训练样本的目标动作表示为：

$$
\mathbf{Y} \in \mathbb{R}^{H \times K}
$$

其中 $H$ 是预测 horizon，$K$ 是统一的最大 channel 数。对于当前任务实际需要的 $c$ 个动作维度，放在前 $c$ 个 channel，剩余维度 zero-padding。同时使用 mask $\mathbf{M}$ 标记有效位置：

$$
M_{h,k}=1
$$

当且仅当第 $h$ 个时间步、第 $k$ 个 channel 是有效动作信号。

这个设计很朴素，但非常实用。它避免了为每个 embodiment 设计独立输出头，也避免了把不同机器人动作强行映射到一个不自然的物理空间。

## 训练流程：先学动作先验，再接视觉 Grounding

Qwen-VLA 的训练 recipe 是全篇最值得细读的部分。作者认为，VLM backbone 和 action decoder 的初始状态非常不对称：VLM 已经预训练好了，DiT action decoder 是随机初始化。如果一开始就多模态联合训练，decoder 要同时学习动作分布、语言条件、视觉 grounding 和 flow matching 动力学，容易低效且不稳定。

于是论文提出了四阶段训练：

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Qwen-VLA-Unifying-Vision-Language-Action-Modeling-across-Tasks-Environments-and-Robot-Embodiments/figures/qwen35vla_arc.png)

> 图解：这张图展示了 Qwen-VLA 的四阶段训练流程。Stage I 只用文本到动作训练 DiT，让 decoder 先学会语言条件下的动作先验；Stage II 再引入图像做 continued pretraining；Stage III 用 SFT 对多任务和真实机器人分支进行对齐；Stage IV 用 RL 直接优化闭环任务成功率。

### Stage I：Text-to-Action DiT Pretraining

T2A 阶段冻结 VLM，只训练 DiT，并且刻意不输入图像。模型只根据语言指令和 embodiment prompt 重建动作轨迹。

这背后的思路很有意思：语言指令是高度压缩的任务描述，而动作轨迹是高维、长序列、连续的控制信号。因此，T2A 本质上是在训练一个 language-conditioned action decompressor。

例如，“pick up the red cup”只有几个 token，但实际动作可能包含上百个关节位置、末端姿态和夹爪状态。T2A 让 DiT 先学会：不同语言对应动作空间中的哪些区域，同一个任务在不同 embodiment 下应该对应怎样的 motor program。

### Stage II：Continued Pretraining

CPT 阶段解冻 VLM 和 DiT，在异构数据混合上继续预训练。这个阶段的目标是把 T2A 学到的动作先验 grounding 到真实视觉观察中。

也就是说，T2A 学的是“语言大概对应怎样的动作”，CPT 学的是“在这张图里，这个语言目标对应哪个对象、哪个位置，以及怎样执行”。

### Stage III：Supervised Fine-Tuning

SFT 从 CPT checkpoint 分出两条路线：

- Multi-task SFT：混合 VQA、spatial grounding、manipulation、navigation 等任务。
- Real-robot SFT：用 ALOHA 等真实机器人遥操作数据做部署对齐。

SFT 的目标同时包含 vision-language next-token loss 和 flow matching action loss。论文中设置 VL loss 权重为 0.1，action loss 权重为 1.0，说明 fine-tuning 阶段主要优化动作生成，同时保留视觉语言能力。

### Stage IV：Reinforcement Learning

SFT 是 imitation learning，优化的是演示轨迹 likelihood；但机器人真正关心的是闭环执行是否成功。于是，Qwen-VLA-Instruct 在 SFT 后继续用 PPO 做 RL。

PPO actor loss 为：

$$
\mathcal{L}^{actor}(\theta)
=
-
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right]
$$

其中：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

总损失为：

$$
\mathcal{L}(\theta)
=
\mathcal{L}^{actor}(\theta)
+
c_v \mathcal{L}^{value}(\theta)
$$

比较特殊的是，flow matching policy 不是 autoregressive softmax policy，没有天然的 log-probability。论文通过在 Euler denoising step 中加入可控噪声，把每一步转成显式 Gaussian transition，从而可以计算 PPO 所需的 log-probability。

RL 只在 SimplerEnv 中收集 rollout，但结果显示它对其他 benchmark 没有造成明显遗忘，甚至对 OOD 和 DOMINO 动态操作也有轻微正迁移。

## 预训练数据：核心是异构混合

Qwen-VLA 的预训练数据非常杂，但不是随便混合。它覆盖了机器人操作、人类第一视角、导航、合成仿真和通用视觉语言数据。

| 数据源 | 占比 |
|---|---:|
| Robot Manipulation Trajectories | 74.2% |
| Human Egocentric Trajectories | 6.0% |
| Navigation Trajectories | 7.5% |
| Synthetic Simulation Trajectories | 3.7% |
| General Vision-Language Data | 3.4% |
| Spatial Grounding 2D | 2.5% |
| Autonomous Driving VQA | 2.4% |
| Fine-Grained Embodied Action Caption | 0.2% |

### 机器人操作数据

机器人操作轨迹占比最高，约为 74.2%。公开数据包括 RobotSet、Galaxea、AgiBot World、RoboCOIN、RoboMIND、DROID、BridgeData V2、RH20T、RT-1、BC-Z 等，还包括 InternData-A1 和 GR00T-X-Embodiment-Sim 等仿真轨迹。

这些数据覆盖 tabletop manipulation、mobile manipulation、bimanual manipulation、dexterous hand control 等多种形态，总量超过 10,000 小时。

动作归一化采用每个数据集自己的 quantile statistics。对数据集 $k$ 的动作维度 $d$，用第 1 和第 99 百分位 $q^k_{01}$、$q^k_{99}$ 做线性映射：

$$
\tilde{a}_d
=
2 \cdot
\frac{a_d - q^k_{01}}
{q^k_{99} - q^k_{01}}
- 1
$$

然后裁剪到 $[-1,1]$。这样可以减轻不同机器人 action scale 的差异。

### 人类第一视角数据

Egocentric human data 占 6.0%，来自 Ego4D、EPIC-KITCHENS、EgoDex、EgoVerse、Xperience 等数据集。它的价值在于规模大、场景真实，且物体和操作语义丰富。

对每只手，模型预测 wrist motion 和 hand articulation：

- Wrist motion：未来 wrist coordinate frame 相对初始帧的 SE(3) 变化，用 translation + axis-angle 表示，共 6 维。
- Hand articulation：对 45 维 MANO hand pose 做 PCA，保留前 10 个 principal components，也就是 eigengrasps。

双手合计每个时间步 32 维动作。

### 合成仿真数据

论文构建了两类 synthetic simulation data：

1. Vision-language-action data：输入图像和语言，输出动作。
2. Language-action data：只输入语言，输出动作，主要服务 T2A。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Qwen-VLA-Unifying-Vision-Language-Action-Modeling-across-Tasks-Environments-and-Robot-Embodiments/figures/sim_demo.png)

> 图解：这张图展示了 RoboInF 生成的仿真任务。上半部分是短 horizon 的任务，例如把两个绿色订书机并排放好；下半部分是长 horizon 的组合任务，例如把饮料聚到一起，并把清洁海绵单独放置。图中每一列对应执行过程中的关键状态，体现了从高层指令到多阶段动作片段的分解。

Vision-conditioned synthetic data 包含 20 个 tabletop scenes，每个 scene 有 10 种物体初始布局，形成 200 个基础配置。任务数量为 450 个，每个任务生成 300 条成功轨迹，并随机化光照、相机、背景、桌面材质、机器人初始状态、物体位置和控制器参数。最终得到 359,848 条成功轨迹，包括 subtask segments。

Language-action data 更大。论文定义了 6 类单臂操作模板，包括 pick-and-place、pushing、pulling、rotation、viewpoint rotation、object swapping，并在 Franka Panda、UR10e、UR5e、Kinova Gen3、TM12、xArm7 六种机器人上生成约 7.2M 条轨迹，总计超过 14,000 小时。

### 导航数据

Navigation data 占 7.5%，包括 instruction following、object searching、target tracking。移动机器人被建模为 3-DoF，即平面位移和 heading angle。输出是 waypoint 序列，形式类似：

$$
(\Delta x, \Delta y, \Delta \theta)
$$

这部分数据对 VLA 的价值不只是“让机器人会导航”，还让模型学习长程 instruction following、历史记忆、目标搜索和空间路径决策。

### 视觉语言数据

视觉语言数据包括 general VL、2D spatial grounding、autonomous driving VQA 和 fine-grained embodied action caption。

其中，fine-grained embodied action caption 很有意思。普通机器人数据可能只有“pick up, rotate, and place the ceramic bowl”这样的粗标签，但实际动作可能包含具体接触点、旋转方向、轨迹和放置位置。论文用 Qwen3.6-plus 加人工审核构建了约 48,000 个细粒度 video-caption pairs，用来减少语言和动作之间的歧义。

## 实验一：仿真机器人操作

Qwen-VLA 在四个操作 benchmark 上评估：

| Benchmark | 特点 |
|---|---|
| LIBERO | 单臂 tabletop，多 split |
| Simpler-WidowX | WidowX real-to-sim |
| RoboCasa-GR1 | 双臂 humanoid kitchen tasks |
| RoboTwin 2.0 | 双臂任务，Easy / Hard |

主要结果如下：

| 方法 | 类型 | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
|---|---|---:|---:|---:|---:|---:|
| $\pi_0$ | Specialist | 94.4 | - | - | 65.9 | 58.4 |
| StarVLA-OFT | Specialist | 96.6 | 48.8 | 64.6 | 50.4 | - |
| GR00T N1.6 | Specialist | 97.2 | 49.9 | 63.2 | 47.6 | - |
| $\pi_{0.5}$ | Specialist | 97.6 | 37.0 | 46.9 | 82.7 | 76.8 |
| ABot-M0 | Specialist | 98.6 | 58.3 | - | 86.0 | 85.0 |
| Qwen-VLA-Base | Generalist | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 |
| Qwen-VLA-Instruct | Generalist | 97.9 | 56.7 | 73.7 | 86.1 | 87.2 |

这里最关键的点是：Qwen-VLA-Instruct 是单个 generalist model，而很多 baseline 是 benchmark-specific specialist。它在 Simpler-WidowX、RoboTwin-Easy、RoboTwin-Hard 上都超过了已列出的 specialist baselines，在 LIBERO 和 RoboCasa-GR1 上也非常接近最强模型。

这说明 multi-task、multi-embodiment co-training 并没有显著牺牲单任务性能，反而在不少场景中带来了更强泛化。

## 实验二：真实世界 ALOHA 机器人

真实机器人评估使用 ALOHA 双臂平台，包括两个 6-DoF 机械臂、parallel-jaw gripper、两个 wrist camera 和一个 first-person-view camera。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Qwen-VLA-Unifying-Vision-Language-Action-Modeling-across-Tasks-Environments-and-Robot-Embodiments/figures/task_overview.png)

> 图解：这张图概览了 ALOHA 真实机器人任务。任务覆盖 pick and place、table cleaning、bowl stacking、towel folding、fine-grained manipulation 等类型，既有短 horizon 的基础操作，也有长 horizon 和精细接触任务。它主要用于验证大规模预训练能否迁移到真实双臂机器人。

论文比较了两个同架构模型：

- Qwen-VLA-aloha w/o pretrain：从零训练。
- Qwen-VLA-aloha w/ pretrain：从 Qwen-VLA-Base fine-tune。

### In-Domain 结果

| 模型 | Pick and Place | Table Cleaning | Bowl Stacking | Bowl Pick & Place | Towel Folding | Fine-grained | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| GR00T N1.6 | 30.8 | 38.5 | 53.8 | 19.2 | 19.2 | 10.3 | 28.6 |
| $\pi_{0.5}$ | 73.1 | 84.6 | 88.5 | 69.2 | 80.8 | 33.3 | 71.6 |
| Qwen-VLA w/o pretrain | 30.8 | 53.8 | 61.5 | 64.1 | 50.0 | 30.8 | 48.5 |
| Qwen-VLA w/ pretrain | 96.2 | 92.3 | 98.7 | 87.2 | 65.4 | 61.5 | 83.6 |

预训练带来的提升非常明显，平均成功率从 48.5% 提升到 83.6%。这说明提升不是因为架构本身，而是来自大规模 VLA 预训练中学到的可迁移动作和视觉语言表示。

### OOD 结果

| 模型 | Color | Instance | Position | Background | Instruction | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| GR00T N1.6 | 46.2 | 38.5 | 3.8 | 19.2 | 19.2 | 25.4 |
| $\pi_{0.5}$ | 57.7 | 61.5 | 19.2 | 26.9 | 42.3 | 41.5 |
| Qwen-VLA w/o pretrain | 42.3 | 30.8 | 34.6 | 30.8 | 42.3 | 36.2 |
| Qwen-VLA w/ pretrain | 88.5 | 76.9 | 53.8 | 80.8 | 84.6 | 76.9 |

OOD 平均成功率达到 76.9%，比 $\pi_{0.5}$ 高 35.4 个百分点。尤其是 background 和 instruction generalization，分别达到 80.8% 和 84.6%。这说明 VLM backbone 的视觉鲁棒性和语言理解能力，在真实机器人泛化中确实发挥了作用。

## 实验三：视觉语言导航

导航实验在 VLN-CE 的 R2R 和 RxR Val-Unseen split 上进行。

| 方法 | R2R NE↓ | R2R OS↑ | R2R SR↑ | R2R SPL↑ | RxR NE↓ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NaVid | 5.7 | 49.2 | 41.9 | 36.5 | 5.7 | 45.7 | 38.2 | - |
| Uni-NaVid | 5.6 | 53.3 | 47.0 | 42.7 | 6.2 | 48.7 | 40.9 | - |
| NaVILA | 5.2 | 62.5 | 54.0 | 49.0 | 6.8 | 49.3 | 44.0 | 58.8 |
| StreamVLN | 5.0 | 64.2 | 56.9 | 51.9 | 6.2 | 52.9 | 46.0 | 61.9 |
| Qwen-VLA-Base | 5.2 | 61.7 | 53.8 | 49.4 | 6.4 | 55.1 | 45.8 | 56.2 |
| Qwen-VLA-Instruct | 5.1 | 69.0 | 57.5 | 51.2 | 5.8 | 59.6 | 47.8 | 57.1 |

Qwen-VLA-Instruct 在 R2R 的 OS 和 SR 上最好，在 RxR 的 SR 和 SPL 上最好。这个结果说明统一 VLA 不只对操作任务有用，也能覆盖 waypoint-style 的导航决策。

## 实验四：静态 OOD 操作

论文构建了 SimplerEnv-OOD，训练只用 Bridge split 的简单 pick-and-place，测试时要求模型完成未见过的空间关系和任务类型。

| 方法 | MoveAway | MoveRight | PlaceNear | PlaceRight | PutFront | StackYellow | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| $\pi_{0.5}$ | 26.1 | 0.0 | 0.0 | 32.1 | 13.0 | 4.2 | 12.6 |
| Qwen-VLA-Base | 31.3 | 31.6 | 16.7 | 47.1 | 6.3 | 18.8 | 25.3 |
| Qwen-VLA-Instruct | 43.8 | 33.3 | 39.6 | 47.9 | 4.2 | 22.9 | 32.0 |

Qwen-VLA-Instruct 平均成功率为 32.0%，明显超过 $\pi_{0.5}$ 的 12.6%。尤其在 MoveRight 和 PlaceNear 上，$\pi_{0.5}$ 完全失败，而 Qwen-VLA 能做到 33.3% 和 39.6%。这说明它不是只记住 pick-and-place，而是有一定空间关系泛化能力。

## 实验五：动态 OOD 操作 DOMINO

DOMINO 是动态物体操作 benchmark，要求模型在物体运动和约束不确定的情况下执行操作。Qwen-VLA 在没有动态操作 fine-tuning 的情况下进行 zero-shot 测试。

| 方法 | 训练设置 | SR ↑ | MS ↑ |
|---|---|---:|---:|
| PUMA | Dynamic fine-tuned | 17.2 | 35.0 |
| LingBot-VA | Zero-shot | 24.1 | 36.1 |
| Qwen-VLA-Base | Zero-shot | 21.1 | 37.4 |
| Qwen-VLA-Instruct | Zero-shot | 26.6 | 39.5 |

Qwen-VLA-Instruct 不仅超过 zero-shot baselines，也超过了部分专门针对动态操作 fine-tuned 的方法。作者认为原因有两个：

1. Flow matching decoder 生成的是连贯 action chunk，执行更果断，适合动态窗口很窄的任务。
2. 大规模混合预训练让模型学到了更通用的 spatial-to-kinematic prior，而不是只记忆静态操作套路。

## 消融实验：T2A 为什么重要？

T2A 是 Qwen-VLA 训练流程的核心。论文在 Simpler-WidowX 上做了多个消融。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Qwen-VLA-Unifying-Vision-Language-Action-Modeling-across-Tasks-Environments-and-Robot-Embodiments/figures/vl_abl.png)

> 图解：这张图包含两个消融方向。左侧比较 action-only 和 VL+VLA co-training，可以看到加入视觉语言数据后，在 RoboCasa-GR1 和 RoboTwin-2.0 这类更依赖细粒度识别和组合指令解析的任务上明显提升。右侧比较随机初始化 DiT 和 pretrained DiT，预训练 DiT 收敛更快、峰值更高，说明动作 decoder 的预训练不是简单 warm-up，而是带来了可迁移动作先验。

### T2A 数据组成

纯真实数据 T2A 得到 51.04%，纯合成数据得到 64.06%，但最佳组合是约 20% synthetic + 80% real，达到 71.09%。

这个结果很合理：合成数据覆盖广，可以提供丰富的语言-动作模式；真实数据更贴近物理动态，可以修正过于理想化的轨迹。

### Full-Sequence 优于 Chunk Prediction

T2A 中 full-sequence prediction 一直优于 chunk prediction。原因是 full trajectory 能让 decoder 学到完整任务的起点、终点、阶段结构和时间一致性，而 chunk 只提供局部片段，容易丢失长程语义。

### T2A 不应该输入图像

在 T2A 加入图像反而下降。论文给出的解释是：T2A 的目的就是让 decoder 学语言到动作的压缩映射。如果加入图像，模型可能依赖视觉相关性走捷径，削弱语言动作先验。

### Flow-Matching Timestep 分布

T2A 最适合 Sigmoid-Normal timestep distribution，而 CPT/SFT 使用 Beta distribution 更好。直观理解是，T2A 没有视觉条件，intermediate noise level 的学习信号更重要；到了 CPT/SFT，VLM hidden states 已经提供强条件，Beta 分布更高效。

### T2A 训练步数

T2A 在 2,000 steps 达到最好，继续训练到 40,000 steps 反而下降。这说明 T2A 主要学习结构性 action prior，不需要长时间拟合固定语料，过度训练会降低后续 CPT 的可塑性。

## 消融实验：VL 数据会不会干扰动作学习？

论文比较了 action-only 和 action + VL data co-training。结果显示，在 LIBERO 和 Simpler-WidowX 上二者差不多；在 RoboCasa-GR1 和 RoboTwin-2.0 上，加入 VL 数据分别带来 +4.9 和 +4.6 个百分点提升。

这说明 VL 数据不但没有干扰动作学习，反而在复杂场景中帮助模型保持物体识别、空间 grounding 和组合语言理解能力。对 VLA 来说，动作能力和视觉语言能力不是相互独立的两个模块，而是共享 backbone 的耦合能力。

## 消融实验：多 Embodiment 投影怎么做？

不同机器人 action dimension 不同，论文比较了三种投影方式：

| 设计 | 思路 | 参数量 |
|---|---|---|
| Multi-MLP | 每个 embodiment 一套 encoder / decoder | $2h\sum_i d_i$ |
| Concatenation | 所有 action 拼成大向量，每个平台占固定片段 | $2h\sum_i d_i$ |
| Zero-Padding | 统一 pad 到最大维度，共享 MLP | $2h d_{max}$ |

结果如下：

| 训练方式 | Bridge | RoboCasa |
|---|---:|---:|
| Bridge Only | 62.8 | - |
| RoboCasa Only | - | 53.4 |
| Multi-MLP | 63.3 | 52.1 |
| Concatenation | 63.0 | 52.8 |
| Zero-Padding | 63.0 | 53.2 |

三种方式性能差异很小，但 Zero-Padding 参数最少，因此最终采用 Zero-Padding。这也说明，只要 shared latent space 建立起来，具体 projection head 不是性能瓶颈。

## 消融实验：RL Post-Training 有多大作用？

| 阶段 | Simpler | RoboCasa | RoboTwin-E | RoboTwin-H | LIBERO | SimplerOOD | DOMINO SR | DOMINO MS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPT | 64.3 | 40.4 | 64.3 | 66.4 | 90.8 | 25.3 | 21.1 | 37.4 |
| + SFT | 70.8 | 56.0 | 86.3 | 87.1 | 97.8 | 31.6 | 25.7 | 39.1 |
| + RL | 73.7 | 56.7 | 86.1 | 87.2 | 97.9 | 32.0 | 26.6 | 39.5 |

SFT 带来最大幅度提升，RL 在此基础上继续提高。最直接的提升出现在 SimplerEnv，因为 RL rollout 就在这个环境中收集。但更重要的是，RL 没有造成明显 catastrophic forgetting，RoboCasa、LIBERO、SimplerOOD 和 DOMINO 都有轻微提升或基本保持。

这说明用 sparse binary success reward 优化闭环成功率，确实可以补上 imitation learning 的不足。

## 消融实验：需不需要显式状态输入？

很多机器人策略会输入 proprioceptive state，例如关节角。Qwen-VLA 比较了三种方式：

| 状态输入方式 | RoboTwin-Easy | RoboTwin-Hard |
|---|---:|---:|
| No State | 88.7 | 87.4 |
| State in VLM Prompt | 89.3 | 88.7 |
| State in DiT | 89.4 | 88.3 |

显式状态只带来很小提升。作者认为原因有两个：多视角图像已经能提供机器人当前构型信息；同时 action decoder 预测的是 relative action displacement，对绝对状态依赖没有那么强。

因此，默认框架不加入 proprioceptive state，继续保持 embodiment prompt 作为唯一平台特定输入。这对跨机器人泛化更简洁。

## 方法亮点总结

Qwen-VLA 最重要的贡献可以概括成四点。

第一，它把 manipulation、navigation、egocentric action modeling、trajectory prediction 统一成 action-and-trajectory prediction。这个统一不是口号，而是落实到了同一个 conditional prediction objective 和同一个 DiT action decoder。

第二，它用 embodiment-aware prompt 处理机器人差异。不同机器人、不同控制频率、不同 action horizon、不同控制语义，都通过文本 prompt 进入模型，而不是依赖额外架构分支。

第三，它提出了合理的 staged training recipe。T2A 先建立语言动作先验，CPT 做视觉 grounding，SFT 做任务对齐，RL 优化闭环成功率。这个流程针对 VLM 已预训练、动作 decoder 随机初始化的不对称性，设计得比较务实。

第四，它证明通用 VLA 不一定输给专用策略。在多个 benchmark 上，Qwen-VLA-Instruct 作为单一 generalist model 达到或超过了一批 specialist policy。

## 局限与未来方向

论文也承认 Qwen-VLA 仍有明显限制。

首先，embodied action data 的规模和多样性仍然远小于视觉语言数据。长尾物体、复杂接触、罕见环境、失败恢复等能力还不够充分。

其次，多任务联合训练存在 objective balancing 问题。动作学习、导航、视觉语言理解之间可能相互拉扯，需要更好的数据 curriculum、loss balancing 和模块化设计。

第三，当前评测仍偏短 horizon 和 benchmark-driven。真实世界长时间部署中的失败恢复、长期记忆、状态跟踪和自我纠错，还没有得到充分解决。

未来更值得期待的方向包括：更大规模真实交互数据、sim-to-real 自动数据生成、第三人称与第一人称人类视频联合学习、episodic memory、world modeling、force / tactile feedback，以及更大规模的 RL。

## 结语

Qwen-VLA 的意义在于，它把 VLM 的感知与语言推理能力真正接到了可执行动作上。相比只预测视觉未来状态的 world model，Qwen-VLA 更强调把多模态理解落到机器人可以执行的 continuous control 上。

从这篇论文可以看到，通用机器人策略的关键不只是“更大的模型”，还要同时解决数据异构、动作表示、embodiment 差异、训练阶段不稳定和闭环成功率优化。Qwen-VLA 给出了一条比较完整的路线：强 VLM backbone + flow matching action decoder + embodiment-aware prompt + 大规模混合预训练 + SFT/RL post-training。

> 本文参考自 [Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments](https://arxiv.org/abs/2605.30280)