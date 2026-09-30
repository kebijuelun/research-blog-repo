# Holo-M：首个面向人形机器人的离散 VLA 模型

VLA（Vision-Language-Action）模型已经在机械臂上验证了「把动作变成语言模型词表里的 Token」这条路线，但人形机器人全身近百个自由度的动作空间，让离散化这条路线迟迟走不通。地平线机器人（Horizon Robotics）提出的 Holo-M，是首个面向人形机器人 Loco-Manipulation（移动操作）的离散 VLA 模型：它把全身动作拆成四个身体部位，分别用专用 Tokenizer 编码进语言模型词表，再用分组离散扩散（Grouped Discrete Diffusion）并行解码。在 SIMPLE 人形机器人基准上，Holo-M 在 Generalist 评测中拿下 143/180 的成功率（第二名 Ψ0 为 114/180），Specialist 评测 163/180（Ψ0 为 154/180），同时解码步数比自回归方案减少 6.5 倍。

## 为什么人形机器人用不上「离散动作 Token」？

要理解 Holo-M 的贡献，先得看清 VLA 模型内部的一个关键设计分歧：动作怎么表示。

一条路线是 **连续动作生成** ：在 VLM 主干外面挂一个 Diffusion 或 Flow-Matching 的 Action Expert，输出连续动作向量。机械臂上的 π0、人形机器人上的 Ψ0 都走这条路。

另一条路线是 **离散动作 Token** ：把动作量化成和文字同一个词表里的 Token，动作生成就变成语言模型本来就会的「下一个 Token 预测」。RT-2、OpenVLA 走这条路，但基本只服务于机械臂。

为什么离散路线没扩展到人形机器人？论文指出了三个层层叠加的障碍：

- **Tokenization 难** ：人形全身动作空间（腿、躯干、手臂、灵巧手）维度高且异构，用一个扁平词表要么太粗、要么大到塞不进上下文。
- **训练难** ：没有任何单一数据集能覆盖人形全部自由度，必须混合遥操作机器人数据、第一人称人类视频、仿真数据这些差异极大的来源。
- **实时推理难** ：动作 Token 序列比机械臂长得多，逐 Token 自回归解码的延迟满足不了全身控制的实时要求。

笔者认为，这三个障碍其实互为因果：正因为动作空间大，序列才长；正因为序列长，解码才慢；正因为没有单一数据集，才需要异构数据混合训练。Holo-M 的设计正是对这三点逐一击破。

## 核心思路：把全身动作「分而治之」

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Humanoid-Loco-Manipulation-With-Discrete-VLA-Model/figures/fig1_overview.png)

> 图解：Holo-M 总体框架。左侧，统一动作 Tokenizer 在遥操作、人类第一人称、仿真三类异构数据上预训练，把人形动作空间分解为 EEF（末端执行器）、Body（身体）、Hand（手）、Kinematics（运动学）四个部位 Tokenizer，共享一个离散词表。右侧推理时，VLM 主干把图像、语言指令和本体状态编码进同一词表，通过分组离散扩散解码动作 Token——组内并行、组间自回归——再反 Tokenize 成全身控制器指令驱动机器人。

Holo-M 的总体思路可以概括为一个比喻： **不给整个人形机器人配一本词典，而是给每个身体部位各配一本小词典，再规定说话顺序** 。具体来说：

- 用 **统一动作 Tokenizer** 解决 Tokenization 和训练问题：动作空间按身体部位分解，不同数据源只监督它覆盖的部位。
- 用 **分组离散扩散** 解决实时推理问题：组内 Token 并行去掩码，组间保持自回归依赖。
- 动作 Token 直接扩展进 Qwen3-VL 的词表，与语言 Token 由同一个序列模型生成——这就避开了连续 Action Expert 方案必须做的「知识隔离」（Knowledge Insulation，防止外挂专家污染预训练主干语义表示的问题）。

下面我们逐块拆解。

## 方法一：FAST-RVQ 统一动作 Tokenizer

### 四组分解的规范动作空间

Holo-M 定义了一个固定顺序的规范动作表示，每个动作 Chunk 包含 30 个时间步（30 Hz，即 1 秒）：

$$
A_t = \left[ A_t^{\mathrm{eef}} \,\middle\vert\, A_t^{\mathrm{body}} \,\middle\vert\, A_t^{\mathrm{hand}} \,\middle\vert\, A_t^{\mathrm{kinematics}} \right] \in \mathbb{R}^{30 \times 96}
$$

四组的维度和 Token 数如下：

| 分组 | 表示内容 | 维度 | Token 数 |
| --- | --- | --- | --- |
| End Effector | 手腕与指尖位姿（相机坐标系） | 48 | 100 |
| Body | 身体关节目标 | 29 | 62 |
| Hand | 手部关节目标 | 14 | 32 |
| Kinematics | 底盘速度、高度、偏航角速度 | 5 | 14 |
| **完整动作** | 全部四组 | 96 | 208 |

这个分解的聪明之处有两点。第一，EEF 组采用任务空间（手腕、指尖位姿）表示，这是人类视频和机器人数据天然共有的「公共语言」——人类视频里没有机器人关节角，但手腕和指尖位置是可以提取的。第二，缺失的组不参与 Loss，而不是用人造目标填充，这样每个数据源只需监督自己覆盖的部位。

### FAST-RVQ：DCT 低频 + RVQ 残差

具体的 Tokenization 方案叫 FAST-RVQ，可以理解为「先抓主旋律，再分层补细节」：

1. 对每个动作组 $X_g \in \mathbb{R}^{T \times D_g}$ ，沿时间轴做 DCT（离散余弦变换），保留前 $N=2$ 个低频系数——这相当于动作轨迹的「主旋律」。
2. 对剩余的高频残差用 RVQ（残差向量量化）分层编码，共 $Q=4$ 级，每级码本用 $k$-means 拟合，不需要训练单独的神经 Tokenizer。

形式化地，低频重建与残差为：

$$
\widehat{X}_{g,N} = \operatorname{IDCT}\!\left( P_N \operatorname{DCT}(X_g) \right), \quad R_{g,N} = X_g - \widehat{X}_{g,N}
$$

其中 $P_N$ 是只保留前 $N$ 个系数的对角投影矩阵。每组的 Token 长度为：

$$
L_g = N D_g + Q = 2 D_g + 4
$$

代入四组维度，得到 $(L_{\mathrm{eef}}, L_{\mathrm{body}}, L_{\mathrm{hand}}, L_{\mathrm{kin}}) = (100, 62, 32, 14)$ ，总计 208 个动作 Token（不含标记组边界的结构 Token）。

和原始 FAST 方案（DCT + BPE）相比，FAST-RVQ 的关键改动是产出 **定长** Token 序列——BPE 的变长序列适合自回归，但不利于掩码并行解码，而定长组边界正是后面分组扩散的前提。这套方案在验证集上的重建误差仅 0.0068。

## 方法二：动作 Token 进词表，主干直接用 Qwen3-VL

Holo-M 的骨干网络是 Qwen3-VL-2B，词表直接扩展加入状态 Token、动作 Token 和结构 Token。条件上下文的固定顺序为「图像 → 任务 → 设定 → 状态」：

$$
c_t = \left[ v_t \,\middle\Vert\, q^{\mathrm{task}} \,\middle\Vert\, q^{\mathrm{setup}} \,\middle\Vert\, q_t^{\mathrm{state}} \right]
$$

其中 setup 字段标明本体和数据来源（如 "G1 humanoid robot from Humanoid Everyday"），state 字段把关节角裁剪到 $[-\pi, \pi]$ 、归一化后均匀量化成 256 个状态 Bin Token。实际 Prompt 结构如下：

```text
[image embeddings]
Task: [language instruction]
<setup_start>[embodiment and data source]<setup_end>
<state_start>[quantized state tokens]<state_end>
```

动作序列同样带结构标记：

```text
Action: <action_output><action_start>
  <end_effector_start>[100 tokens]<end_effector_end>
  <body_start>[62 tokens]<body_end>
  <hand_start>[32 tokens]<hand_end>
  <kinematics_start>[14 tokens]<kinematics_end>
<action_end>
```

这里值得对比的是知识隔离问题：连续 Action Expert 方案（如 Ψ0）必须小心翼翼地把外挂专家的梯度与预训练 VLM 主干隔离，否则会破坏主干的语义表示。Holo-M 没有这个边界——动作和语言是同一个模型在同一个词表上生成的，动作能力直接受益于主干的预训练能力。笔者认为这是离散路线最被低估的架构优势。

## 方法三：分组离散扩散解码

208 个 Token 逐一生成，意味着每个动作 Chunk 要 208 次串行前向计算——这对 30 Hz 的全身控制是不可接受的。Holo-M 的解法是 **分组离散扩散** ：组内并行去掩码，组间保持因果依赖。

### 分组掩码训练

训练时沿用 EEF → body → hand → kinematics 的固定顺序，每组构成一个扩散块。组内 Token 双向交互（可以并行重建），组间保持因果（当前组能看到条件上下文和前面所有组，但看不到后面的组）。

对每个有效组独立采样掩码概率 $p_g$ ，把对应位置替换为 `[MASK]`，模型基于上下文、前序组和组内可见 Token 预测被掩码位置：

$$
\mathcal{L}_g = -\frac{1}{p_g N_g} \sum_{j \in \mathcal{M}_g} \log p_\theta \left( y_{t,j}^g \mid c_t, y_t^{<g}, \tilde{y}_t^g \right)
$$

对于只覆盖部分动作组的数据源，用组有效性指示 $v_g \in \{0, 1\}$ 把缺失组排除在 Loss 之外：

$$
\mathcal{L}_{\mathrm{diff}} = \frac{\sum_{g \in \mathcal{G}} w_g v_g \mathcal{L}_g}{\sum_{g \in \mathcal{G}} w_g v_g}
$$

注意这个目标函数直接作用在后训练的 VLM 上， **没有引入任何独立的扩散模型或动作头** 。

### 基于置信度的去掩码

推理时，四组按规范顺序依次解码，组内所有位置初始为 `[MASK]`。每次迭代模型预测所有被掩码位置，按预测概率给出置信度，高置信度位置先揭码，低置信度位置留到下一轮。一组完成后，其 Token 追加进因果上下文供下一组使用，条件前缀和已完成组都做缓存。

效果上，每组 8 次去掩码迭代时，四个组只需 32 次串行解码步骤，对比自回归的 208 步， **串行深度减少 6.5 倍** 。这个设计比全并行扩散（整个 Chunk 一次去掩码）更保守，但保住了组间依赖——比如先定手腕轨迹、再定手指和底盘，符合操作的物理因果。

## 训练流水线：四阶段渐进

![Training Pipeline](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Humanoid-Loco-Manipulation-With-Discrete-VLA-Model/figures/training_pipeline.png)

> 图解：Holo-M 的四阶段训练流程。Stage I 在 EgoDex（人类第一人称操作）和 Humanoid Everyday（人形机器人数据）上做跨本体自回归预训练；Stage II 在 SIMPLE 人形训练集上后训练，对齐目标本体；Stage III 用分组离散扩散目标微调，把逐 Token 生成替换为组内并行去掩码；Stage IV 可选地在单任务数据上微调得到 Specialist 策略。全程使用同一 Qwen3-VL-2B 主干和规范动作词表，视觉编码器冻结，不引入任何独立动作模型。

几个值得注意的细节：

- **Stage I 的混合训练** 是四组分解设计真正发光的地方。比如一个 EgoDex 样本只有 $v_{\mathrm{eef}} = 1$ ，其余组为 0，只有 EEF Token 参与 Loss。对比之下，Ψ0 需要把人类和机器人数据分阶段处理、用不同的动作解码器，而 Holo-M 用一个共享表示在同一阶段混合训练。
- **Stage II 的自回归目标** 对每个有效 Token 等权重：

$$
\mathcal{L}_{\mathrm{AR}} = -\frac{\sum_{g \in \mathcal{G}} v_g \sum_{j=1}^{L_g} \log p_{\theta} \left( y_{t,j}^{g} \mid c_t, y_t^{<g}, y_{t,<j}^{g} \right)}{\sum_{g \in \mathcal{G}} v_g L_g}
$$

- Stage III 的产物是 Holo-M Generalist 策略，Stage IV 的产物是各任务的 Specialist 策略。

## 实验：SIMPLE 基准上全面领先

### 实验设置

训练数据沿用 Ψ0 的配置，三个来源：

| 数据源 | 领域 | 轨迹规模 | 任务数 | 模态 | 用途 |
| --- | --- | --- | --- | --- | --- |
| EgoDex | 人类 | 829 小时 / 33.8 万条 | 194 | RGB、手部姿态 | AR 预训练 |
| Humanoid Everyday | 人形 | 31 小时 / 1.03 万条 | 260 | RGB、动作 | AR 预训练 |
| SIMPLE | 人形 | 100 万条 | 24 | RGB、状态、动作 | 后训练 |

评测在 SIMPLE 基准的 6 个代表性任务上进行（覆盖刚体操作、非抓握交互、铰链物体、双臂协调、移动操作），每个任务在 Level 0/1/2 三档域随机化下各跑 10 次 Rollout。选择 SIMPLE 的原因是该基准报告过仿真中的策略排名与真实世界表现有良好相关性。

### Generalist 评测：143/180，领先第二名 29 次成功

Generalist 定义为在全部 24 个 SIMPLE 任务上联合后训练的单一 Checkpoint，不做任何任务级微调：

| 方法 | XMovePick | BendPick | Handover | Mobile P&P | Grasp | XMoveBendPick | 总计 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ψ0 | 9/10/9 | 4/2/4 | 10/10/9 | 6/6/3 | 8/7/6 | 4/4/3 | 114/180 |
| π0.5 | 0/0/3 | 0/0/0 | 6/1/6 | 0/0/0 | 6/4/2 | 0/0/0 | 28/180 |
| DreamZero | 0/0/0 | 0/0/0 | 7/7/6 | 0/0/0 | 7/5/6 | 0/0/0 | 38/180 |
| ACT | 0/0/0 | 10/9/10 | 0/0/0 | 6/8/9 | 5/6/8 | 0/5/3 | 79/180 |
| Holo-M AR (消融) | 9/5/8 | 10/8/8 | 9/6/8 | 5/6/7 | 8/9/8 | 7/8/9 | 138/180 |
| **Holo-M** | 9/7/3 | 9/9/8 | 7/7/10 | 8/9/6 | 8/9/8 | 9/8/9 | **143/180** |

两组数字值得玩味。第一，Holo-M AR（自回归消融版）就已经超过所有基线，说明主要收益来自 **Tokenizer 设计和联合训练方案** ，而不是解码器。第二，分组离散扩散版不但快了 6.5 倍，成功率还略高（143 vs 138）——组内并行解码并不需要拿任务表现换速度。

### Specialist 评测：163/180，长时程任务优势最大

Specialist 定义为从 Generalist Checkpoint 出发、在单任务数据上分别微调。Holo-M 达到 163/180，超过最强基线 Ψ0 的 154/180（其余基线如 GR00T N1.6、π0.5、EgoVLA、DP 等均在 150 以下）。

差距最大的任务是 Mobile P&P——六个任务中时程最长的移动操作任务，需要机器人拿着物体在桌子之间导航并放置。Holo-M 在 30 次 Rollout 中成功 25 次（8/9/8），Ψ0 只有 18 次（7/5/6）， **相对提升接近 40%** 。这暗示离散 Token 方案在长时程、复合动作上的优势：移动和操作在同一词表里联合建模，而不是像 Action Expert 方案那样在模块间传递信息。

### 数据 Scaling：预训练数据带来 2.6 倍提升

逐步增加预训练数据的消融实验（Holo-M AR）：

| 预训练数据 | Open-loop MAE ↓ | Closed-loop 成功率 ↑ |
| --- | --- | --- |
| 无预训练 | 0.022 | 53/180 |
| 仅 Humanoid Everyday | 0.020 | 119/180 |
| EgoDex + HE | 0.019 | 138/180 |

仅靠 Humanoid Everyday 预训练就把闭环成功率翻倍（53 → 119），再加上人类第一人称数据 EgoDex 进一步提升到 138。这验证了统一 Tokenizer 的核心假设：机器人数据建立全身动作基础后，更丰富的人类视频数据还能继续「打磨」闭环鲁棒性——而人类视频远比遥操作机器人数据容易获取，这是可扩展性的关键。

### 真机部署：RTX 5090 上实时运行

部署平台是 Unitree G1-comp（竞赛版），配两个 Dex3 灵巧手和头顶 RealSense 相机。策略输入 640×360、30 Hz 图像和 31 个上肢关节角，输出 1 秒动作 Chunk，在单张 RTX 5090 上通过 WebSocket 运行。整套真机栈复用 Ψ0 的客户端、协议和全身控制器，只替换策略服务器，保证差异只来自策略本身。

![Fixed Schedule](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Humanoid-Loco-Manipulation-With-Discrete-VLA-Model/figures/fixed_schedule.png)

> 图解：固定调度控制循环（以 8 步去掩码为例）。每 500 ms（15 个 Tick）采集一次观测，模型在固定的 Hold 预算内完成推理、输出 30 步（1 秒）动作 Chunk；推理期间机器人继续执行上一个 Chunk。新 Chunk 丢弃落在 Hold 窗口内的前导预测步，从 Hold 结束后接管，执行剩余 15 步。若推理超时，机器人回到之前姿态以保证安全。

实测推理延迟：2 步去掩码 175.8 ms，4 步 275.7 ms，8 步 476.8 ms，全部落在对应的 Hold 预算内。真机成功率：Tabletop Grasp 10/10，Move Pick 8/10（各 10 次 Rollout）。

## 总结与展望

- **首个离散人形 VLA** ：Holo-M 把动作 Token 直接放进 Qwen3-VL 词表，避免了连续 Action Expert 的知识隔离问题，动作生成直接受益于主干能力。
- **四组分解 Tokenizer** ：EEF/body/hand/kinematics 独立 Tokenize（FAST-RVQ：DCT 低频 + RVQ 残差），让人类视频、遥操作、仿真数据能各取所长地混合训练。
- **分组离散扩散解码** ：组内并行、组间自回归，串行解码深度减少 6.5 倍，成功率不降反升。
- **全面领先的实验结果** ：SIMPLE 基准 Generalist 143/180、Specialist 163/180，预训练数据贡献 2.6 倍提升，真机部署实时可达。
- **可扩展的叙事** ：动作 Token 与语言同词表，意味着未来可以直接在序列中插入推理 Token（子任务分解、失败恢复），长时程能力随主干推理能力自然扩展。

局限与方向：Holo-M v1 依赖解耦式全身控制器（WBC），未来计划把反 Tokenization 重映射到 HoloMotion、SONIC 等其他控制器；更大规模的第一人称人类视频也是明确的下一步。

> 本文参考自 [Humanoid Loco-Manipulation With Discrete VLA Model](http://arxiv.org/abs/2609.35709v1)