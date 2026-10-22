# VPP2 论文解读：预测得更好，才能行动得更好

机器人领域最近很流行一类做法：把视频生成大模型改造成机器人策略，即 World Action Model（WAM，世界动作模型）。但这类方法有个尴尬的通病——在训练过的任务上表现很好，一到开放式新场景就"胡思乱想"，预测出错误的未来运动，进而执行错误动作。这篇 VPP2（Video Prediction Policy 2）给出的答案很直接：**先把视频预测本身的泛化能力做扎实，再谈动作生成**。核心手段是为操作视频配上详细文本标注、做"事件级"的视频预训练，再蒸馏成单步预测器并外挂动作专家。最硬的结果是：14B 的 VPP2 在开放式任务的指令跟随成功率上反超 64B 的 Cosmos3 达 11 个百分点；在真实 ALOHA 双臂机器人上零样本成功率 58.5%，比最强基线 $\pi_{0.5}$ 高出 18.5 个百分点。

## 提出问题：WAM 为什么在开放场景翻车？

先建立一点直觉。一个典型的机器人策略做的事情是：输入一句简单指令和当前画面，输出一小段动作序列。当数据规模越来越大、越来越杂，"指令 + 画面 → 动作"这个映射会变得越来越不确定——同一个画面下，合理的下一步动作可能有很多种。模型学不到真正的因果，只能学到一些短视的虚假相关性。

WAM 的想法本来很美：视频基础模型见过互联网规模的物理世界，如果能先"想象"出未来画面，再从画面中反推动作（逆动力学），泛化能力应该水到渠成。但作者的评测发现，现有 WAM 在未见过的开放场景中经常预测出错误运动，导致执行出错。在那些考察泛化能力的 benchmark 上，即使标准测试接近满分的策略，一旦遇到扰动任务配置或新的技能组合，成功率就大幅跳水。

作者把问题归因到两点：

- **底座视频模型不为操作而生**：通用视频模型优化的是内容创作和美学质量，而非精细的物理动态，因此常常无法忠实执行操作指令。
- **动作学习会"污染"视频模型**：天真地把动作组件或动作训练目标加进预训练视频模型，会显著损害其原有的泛化能力。

![VPP2 核心动机](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig1_0926_scientific_colors_checked.png)

> 图解：上方是典型机器人策略的学习路径——简单指令加观测直接映射到短动作块，随着数据规模扩大，这个映射越来越"多模态、不确定"，模型容易学到短视的虚假相关。下方是 VPP2 的路径——先用详细文本标注做"下一事件视频预测"（next event video prediction），建立从语义到轨迹的一致映射，再经后训练和蒸馏生成短动作块。

这个观察很犀利：WAM 的问题不在"动作模块怎么设计"，而在于"预测的目标本身太模糊"。解决了这个诊断问题，接下来自然要问——怎么把预测目标变得明确？答案藏在数据里。

## 数据处理：把"模糊的未来"变成"确定的事件"

VPP2 的第一板斧是一套精心设计的数据流水线，核心思想一句话概括：**降低未来预测的不确定性，让模型学到从条件信息到子任务轨迹的一致映射**。

### 视频来源

作者汇集了大规模、多样化的操作视频，用于对开源视频基础模型 Wan2.1 做继续预训练。数据分三大类：机器人操作数据、人类活动数据、通用视频数据（后者经过类似 LVP 的过滤流水线，只保留与操作相关的片段）。

| 数据类型 | 本体类型 | 数据来源 |
| --- | --- | --- |
| 机器人 | 单臂 | OXE、RoboMIND、DROID、RH20T、MolmoAct |
| 机器人 | 双臂 | AgiBot World-Beta、RoboCOIN、RDT、GM-100、ABC-130K、自采 ALOHA 数据 |
| 机器人 | 移动与人形 | InternData-A1、Galaxea Open-World、自采灵巧手数据 |
| 人类 | 人手 | 开源：EgoDex、Ego4D、EPIC-KITCHENS、Something-Something V2、VITRA-1M；自采第一视角数据 |
| 通用视频 | 人手 | OpenVid、Panda-70M |

### 视频过滤、切分与标注

原始轨迹先做清洗：剔除相机故障、画面跳变等损坏片段，然后把剩下的轨迹切成 1–12 秒、语义连贯的短 clip。切分的候选边界靠启发式线索来找——比如人手或末端执行器速度的局部极小值、夹爪或手指开合的瞬间；之后再用视觉语言模型（VLM）做精修，合并过碎的 clip、微调起止点。

切分完成后，为每个 clip 生成详细 caption，要求覆盖四要素：任务描述、明确的活动末端执行器、无歧义的目标物体、相机视角与运动。

![数据处理流水线示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig2_2.png)

> 图解：数据处理流水线的一个实例。左侧输入多视角原始视频，切分成完整子任务片段后，配上详细 caption——包括任务细节、哪只手（哪个末端执行器）在动、目标物体是谁、相机视角如何变化。左下角还展示了"T 形"多视角拼接方式：主视角在左，两个辅助视角纵向堆叠在右。

为什么这么做？博主认为这里的洞察值得划重点：如果 caption 只说"拿起杯子"，而画面里有两个杯子、两只手，那么"指令 → 未来轨迹"的映射本质上还是一对多的，模型的预测不确定性一点没减少。**详细标注的本质是把条件信息补全到足以唯一确定未来演化的程度**。这与视频生成领域"prompt 越细、生成越可控"的经验一脉相承。

### 统一动作空间与坐标系

不同机器人数据集的坐标系和相机视角往往不一致，导致视觉上相似的运动对应完全不同的动作数值——比如"向左移动"在一个数据集里是 $x$ 轴正方向，在另一个里却是 $y$ 轴负方向。这让模型很难学到每个动作维度的语义。

作者聚焦第一视角双臂数据集（ALOHA 风格、人形风格、人类第一视角），它们占数据总量 80% 以上，且共享类似的双臂结构和相机视角。对这些数据做两级对齐：

1. **工作空间对齐**：施加世界坐标系变换，让不同机器人的末端工作空间可比；
2. **末端执行器对齐**：施加局部变换，统一末端坐标原点和轴向。

形式上，对数据集 $d$，设原始末端位姿为 $T_d = {}^{W_d}T_{E_d} \in SE(3)$，其中 $W_d$ 和 $E_d$ 分别是该数据集的世界系和末端系。定义工作空间对齐 $A_d = {}^{\bar{W}}T_{W_d}$ 和末端对齐 $B_d = {}^{E_d}T_{\bar{E}}$（$\bar{W}$、$\bar{E}$ 为规范坐标系），对齐后的末端位姿为：

$$
\widetilde{T}_d = A_d \, T_d \, B_d
$$

左乘对齐世界系，右乘对齐末端原点和轴向——这个"夹心"变换干净地分离了两级对齐。

![坐标系对齐示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig3_2.png)

> 图解：(a) 中不同颜色包围盒展示了不同数据集对齐后的工作空间，基本重合；(b) 展示对齐后的末端执行器坐标系，各数据集的轴向定义归于统一。

### 统一 T 形多视角输入

机器人数据集通常带多个相机视角。作者把它们拼成 T 形复合图像：主视角放左边，两个辅助视角纵向堆在右边；不足两个辅助视角时用黑图占位。这样不同数据集的输入格式完全统一，一个模型就能吃所有数据。

## 核心方法：三阶段训练流水线

数据备好之后，VPP2 的训练分为三个阶段，整体目标是：先让视频模型在操作领域既泛化又实时，再让动作专家在视频模型的 KV cache 条件下学动作。VPP2 基于预训练的 Wan2.1-I2V-14B，通过混合 Transformer 架构（MoT，Mixture-of-Transformers）引入动作专家。

![VPP2 三阶段训练流水线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig4_3.png)

> 图解：三阶段流水线全貌。Stage 1 做大规模事件级视频预训练，得到可泛化的操作视频模型；Stage 2 后训练并蒸馏模型，使其单次前向（约 0.1 秒）预测跨 8 秒的长时视频块；Stage 3 训练动作专家，以单步视频 latent 为条件生成 2 秒动作块。

### Stage 1：事件级视频继续预训练

这一阶段的关键词是"事件级"——每个训练样本覆盖一个**完整**的操作子任务，让子任务描述和"从初始观测到子任务完成"的完整视觉演化严格对齐。

给定演示轨迹 $\tau = (o_0, o_1, \ldots, o_T)$，在整个事件上均匀采样 $N$ 帧未来画面：

$$
\mathbf{y}_{\mathrm{event}} = \left( o_0,\; o_{\lfloor T/N \rfloor},\; o_{\lfloor 2T/N \rfloor},\; \ldots,\; o_T \right)
$$

其中 $T$ 是末帧索引，$N$ 不含初始条件帧 $o_0$。记 $\mathbf{x}_1 = \mathcal{E}(\mathbf{y}_{\mathrm{event}})$ 为对应的视频 latent（$\mathcal{E}$ 是视频编码器），构造插值 latent：

$$
\mathbf{x}_s = (1-s)\,\boldsymbol{\epsilon} + s\,\mathbf{x}_1, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad s \in [0,1]
$$

$s$ 是 flow 时间，$s=0$ 对应纯噪声，$s=1$ 对应数据。模型用 Flow Matching 目标优化：

$$
\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{(\mathbf{x}_1, c),\, \boldsymbol{\epsilon},\, s} \left[ \left\| v_\theta(\mathbf{x}_s, s, c) - (\mathbf{x}_1 - \boldsymbol{\epsilon}) \right\|_2^2 \right]
$$

$v_\theta$ 预测流速，条件 $c$ 包含子任务指令和初始观测 $o_0$。

实现细节上：预测 49 帧、分辨率 $416 \times 240$，对 Wan2.1-I2V-14B 继续预训练 30,000 步，batch size 1024，学习率 1e-5。实验部分我们会看到，这个 14B 模型的操作指令跟随能力甚至超过了 64B 的 Cosmos3。

### Stage 2：块级后训练 + 一致性蒸馏

事件级预训练解决了泛化，但引入了新麻烦：不同时长的事件被采样成同样多的帧，预测帧之间的时间间隔不一致，给下游动作对齐添乱。于是 Stage 2 把模型后训练为预测**固定时长**的未来块，为动作学习建立统一的时间尺度——人手操作预测 2 秒，机器人操作预测 8 秒。

设预测时域 $H$ 秒、数据集帧率 $f$ fps，采样帧间隔 $\delta = fH/N$（以数据集帧计）。从第 $t$ 帧开始的块，目标视频为：

$$
\mathbf{y}_{\mathrm{chunk}}^{(t)} = \left( o_t,\; o_{t+\lfloor \delta \rfloor},\; o_{t+\lfloor 2\delta \rfloor},\; \ldots,\; o_{t+\lfloor N\delta \rfloor} \right)
$$

末帧索引封顶在 $T$，仍用上面的 Flow Matching 目标训练，$\mathbf{x}_1 = \mathcal{E}(\mathbf{y}_{\mathrm{chunk}}^{(t)})$。

接着做**一致性蒸馏**（Consistency Distillation），把多步采样的视频模型压成单步生成器。做法是强迫相邻 flow 时间的终端预测保持一致：

$$
\mathcal{L}_{\mathrm{CD}} = \mathbb{E} \left[ \lambda(s, s') \left\| F_\theta(\mathbf{x}_s, s, c) - \operatorname{stopgrad}\left( F_{\bar{\theta}}(\widehat{\mathbf{x}}_{s'}, s', c) \right) \right\|_2^2 \right]
$$

其中 $0 \le s < s' \le 1$，$\widehat{\mathbf{x}}_{s'}$ 由冻结教师模型从 $(\mathbf{x}_s, s)$ 积分到 $s'$ 得到，$\bar{\theta}$ 是学生参数的 EMA 版本，并满足边界条件 $F_\theta(\mathbf{x}, 1, c) = \mathbf{x}$。为突出单步生成，以 0.5 概率直接取 $s'=1$，否则从其余 flow 时间采样。推理时视频 latent 一次前向得到：$F_\theta(\boldsymbol{\epsilon}, 0, c)$。

博主点评：这个"先事件级后块级"的顺序很讲究。事件级训练建立了语义到完整轨迹的干净映射（泛化的来源），块级后训练只调整时间尺度而不破坏这个映射，蒸馏再解决速度问题——每一步只解决一个问题，互不干扰。

### Stage 3：动作专家预训练

视频模型就绪后，在处理好的双臂数据集上训练动作专家。动作专家是一个 0.9B 参数的扩散 Transformer（DiT），采用标准 MoT 架构，学习以预测未来为条件的**隐式逆动力学模型**——即从"现在 + 想要的未来"反推"该怎么动"。

由于动作专家是全新初始化的，训练早期冻结视频 DiT 的底座参数，只用 LoRA 适配视频骨干。这个策略保护了预训练视频表征不被动作梯度带偏——这正是针对引言中"动作学习损害视频模型泛化"这一诊断开的药方。

**推理延迟**：后训练阶段把视频输入降到 $17 \times 416 \times 240$，配合单步蒸馏，bfloat16 精度加 torch.compile 下视频模型约 0.12 秒；动作专家以视频 latent 为条件做 5 步去噪，约 0.1 秒。**每个动作块总延迟约 0.22 秒**，满足实时控制要求。

### VLM 高层规划

面对长程任务和模糊的人类指令，VPP2 采用层级架构：VLM 负责子任务规划（高层推理、语义理解和记忆），VPP2 专心做"明确指令 → 轨迹"的映射。VLM 可以本地部署，也可以调前沿模型 API。此外 VLM 规划还会生成与训练 caption 模板一致的详细子任务描述——即所谓的 prompt enhancement，视频生成领域的前作已证明详细 caption 对生成质量至关重要。

## 实验：预测质量和动作能力双丰收

实验回答三个问题：视频预测的运动质量比 SOTA 视频基础模型如何？零样本操作能力有多广？领域后训练后表现如何？

### 视频预测质量

作者在开放式场景（办公室环境、机器人工作间实拍照片 + 自由选取的操作指令）上测试预训练 VPP2 的泛化性。

![人类手部操作视频预测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig6.png)

> 图解：VPP2 对人手操作的视频预测结果。在大规模多样化操作数据上训练后，模型能泛化到开放场景的人手操作，预测出的未来轨迹忠实跟随指令。

![机器人操作视频预测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig7.png)

> 图解：VPP2 在机器人操作上的视频预测。模型支持超过 20 种机器人本体，并能泛化到未见的本体和任务；同时支持 1–3 个相机视角的联合预测（主视角在左、辅助视角在右的 T 形布局）。

定量对比选了四个视频基础模型加一个消融变体：基座 Wan2.1-I2V-14B-480p、同样改造 Wan 做操作的 LVP、在机器人数据上充分训练的 Cosmos3-Nano-16B、Cosmos 3 家族最强的 Cosmos3-Super-Image2Video-64B，以及消融 VPP2-Fixed-Step（预测固定时长块而非完整事件，预测未来与描述全事件的指令可能错位）。人手和机器人操作各随机收集 50 个图像-指令对，用 GPT 评估器打分：

| 方法 | 人手 | 机器人 |
| --- | --- | --- |
| Wan2.1-14B | 0.32 | 0.04 |
| LVP | 0.62 | 0.46 |
| Cosmos3-16B | 0.44 | 0.32 |
| Cosmos3-64B | 0.70 | 0.78 |
| VPP2-Fixed-Step | 0.48 | 0.42 |
| **VPP2** | **0.80** | **0.90** |

指令跟随成功率上，VPP2 全面领先，机器人侧比 64B 的 Cosmos3 高 12 个百分点（综合两个场景高 11 个百分点），比自家基座 Wan2.1-14B 的 0.04 更是天壤之别。消融变体 VPP2-Fixed-Step 明显落后于完整版，直接验证了"事件级预测"的价值。

![人评胜率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig8_2_0921.png)

> 图解：人类评估者两两对比各模型预测结果后，VPP2 对各基线的胜率条形图。VPP2 对所有基线的胜率均超过 50%，在需要空间理解的复杂场景优势最明显。

![指令跟随定性对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig5_2.png)

> 图解：VPP2、Wan-14B、Cosmos3-64B 在复杂指令上的定性对比。例如涉及空间关系的指令（"把物体放到某个相对位置"），Wan-14B 和 Cosmos3-64B 会出现错误运动或改错物体，VPP2 能正确执行。

作者的分析是：详细 caption 训练鼓励了更准确的"指令 → 轨迹"映射。博主补充一句：这也解释了为什么 14B 能打赢 64B——泛化差距主要不是容量问题，而是训练信号质量问题。

### 真实 ALOHA 零样本操作

视频预测过关后，看真机。VPP2 直接部署在训练时见过的 ALOHA 本体上，**不做任何任务特定微调**。公平起见，基线 $\pi_{0.5}$ 和 Fast-WAM（Wan-14B 版本）在 VPP2 训练所用到的全部 ALOHA 数据上做微调。10 类零样本任务的结果：

![真实 ALOHA 零样本成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/fig10.png)

> 图解：10 个随机选取的任务类别上，VPP2、$\pi_{0.5}$、Fast-WAM 的零样本成功率对比柱状图。VPP2 平均成功率 58.5%，大幅领先 $\pi_{0.5}$ 的 40.0% 和 Fast-WAM 的 20.5%，并在 10 类中的 9 类拿到最佳。

值得注意的是，基线还吃到了针对性微调的红利，VPP2 是裸奔上场——这个 18.5 个百分点的差距含金量很高。

### LIBERO 系列：后训练后的泛化能力

所有模型只在四个标准 LIBERO 套件上训练，然后在三种互补设定下评测：LIBERO-ID（分布内标准性能）、LIBERO-Pro（物体位置扰动 Position 和任务描述扰动 Task）、LIBERO-OOD（沿空间、物体、目标三个维度做组合泛化，重组成未见过的任务配置）。

| 方法 | LIBERO-ID 总 | LIBERO-Pro Position | LIBERO-Pro Task | LIBERO-Pro 总 | LIBERO-OOD Spatial | LIBERO-OOD Object | LIBERO-OOD Goal | LIBERO-OOD 总 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\pi_0$ | 94.2 | 0.5 | 0.0 | 0.3 | 0.7 | 0.3 | 4.3 | 1.7 |
| $\pi_{0.5}$ | 96.9 | 20.8 | 1.3 | 11.0 | 36.7 | 2.3 | 41.7 | 26.8 |
| MolmoAct | 86.6 | 1.5 | 1.5 | 1.5 | -- | -- | -- | -- |
| X-VLA | 98.1 | 0.8 | 6.8 | 3.8 | -- | -- | -- | -- |
| AtomVLA | 97.0 | 7.3 | 5.3 | 6.3 | -- | -- | -- | -- |
| Cosmos-Policy | 98.5 | -- | -- | -- | 30.7 | 0.3 | 0.7 | 10.5 |
| Fast-WAM | 97.6 | -- | -- | -- | 12.7 | 0.0 | 15.7 | 9.4 |
| DiT4DiT | 98.6 | -- | -- | -- | 9.0 | 0.0 | 10.3 | 6.4 |
| Temporal Ratio | 94.0 | -- | -- | -- | 58.6 | 40.0 | 80.3 | 59.4 |
| **VPP2** | **98.8** | **42.3** | **47.8** | **45.0** | **59.3** | **44.5** | **87.7** | **63.9** |

这张表的信息量很大。多数基线在 LIBERO-ID 上都超过 94%，说明分布内性能早已饱和；但一到扰动环境就原形毕露：LIBERO-Pro 上 VLA 基线最高只有 11.0%，而 VPP2 达到 45.0%，其中 Task 扰动一项差距最悬殊（47.8% 对 6.8%）——这说明 VPP2 真的在**听指令**，而不是回放背下来的轨迹。LIBERO-OOD 上，视频类策略 Cosmos-Policy、Fast-WAM、DiT4DiT 最高仅 10.5%，VPP2 拿到 63.9%，还超过了专门针对该泛化缺口设计的 Temporal Ratio（59.4%）。

### RoboDojo：42 个双臂任务的综合擂台

RoboDojo 是仿真与真机统一的通用操作策略 benchmark，仿真部分含 42 个双臂任务，覆盖五个能力维度：泛化、记忆、精细操作、长程执行、开放词汇指令跟随。为支持依赖记忆的任务，作者把 episode 首帧和按 1 秒间隔采样的 8 个历史帧加入模型输入；由于输入结构变了，RoboDojo 后训练从 Stage 2 蒸馏前的 checkpoint 初始化，先单训视频模型 10,000 步，再联合训练视频模型与动作专家（视频噪声固定最大，推理时视频预测仍只需一次前向）。

| 指标 | $\pi_0$ | $\pi_{0.5}$ | X-VLA | Fast-WAM | OpenWAM-$\alpha$ | Xiaomi-Robotics-1 | GPT-6-Astra | **VPP2** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 平均分 | 3.48 | 11.44 | 10.13 | 3.48 | 17.18 | 20.07 | 28.97 | **39.26** |
| 成功率（%） | 1.53 | 6.93 | 6.52 | 2.03 | 11.92 | 13.93 | 22.48 | **32.26** |

VPP2 平均分 39.26、成功率 32.26%，双双 SOTA（基线取自 2026 年 9 月的官方排行榜），超过前沿基础模型 GPT-6-Astra（28.97 / 22.48%）。

### VLM 子任务规划的加成

以上 VPP2 用的都是整任务指令、无子任务规划。作者进一步在 RoboDojo 五个任务组（物体分类、积木堆叠、积木交换、麻将、井字棋）上测试 VLM 子任务规划：训练时把演示切段并配上子任务指令，测试时 VLM 依据视觉观测和执行反馈挑选下一个子任务交给 VPP2 执行。

![子任务规划效果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/subtask_planning.png)

> 图解：有无 VLM 子任务规划的成功率对比（每任务 50 次试验）。带规划的配置平均成功率 57.6%，基线仅 27.6%；提升最大的是语言条件积木堆叠（10% → 52%）和井字棋（0% → 38%）。

## 附录细节：caption 如何消灭四类不确定性

附录展开了视频标注流水线针对的四类不确定性来源，值得复现者细读：

1. **详细任务描述**：描述末端执行器如何完成任务，包括操作顺序和方式；
2. **明确的末端执行器标识**：活动的末端执行器必须在首帧可见，否则裁掉片段开头；多末端时逐一描述各自运动；
3. **无歧义的目标物体指定**：画面中有多个相同或相似物体时，用独特属性或空间关系锁定目标，且目标必须在首帧可见；
4. **相机视角描述**：用光流剔除视角变化过大的第一视角 clip；对中等相机运动的 clip，由 VLM 描述视角变化并写入 caption。

附录还展示了蒸馏前后单步视频预测的对比：

![蒸馏前后单步预测对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Video-Prediction-Policy-2-Predict-Better-Act-Better/figures/video_comparison_8groups_updated.png)

> 图解：开放任务上，蒸馏前后的**单步**视频预测对比。蒸馏后用一次采样即可获得与蒸馏前相当的高质量预测，证明一致性蒸馏没有明显牺牲预测质量。

此外附录给出逐套件的完整数据。LIBERO-ID 四个套件上 VPP2 分别为 Spatial 99.2 / Object 99.8 / Goal 98.2 / Long 98.0（平均 98.8）；LIBERO-Pro 按套件×扰动细分，VPP2 在 8 列中 7 列最优（如 Spatial-T 58.0、Object-S 58.0），仅 Goal-S 一项略低于 $\pi_{0.5}$（37.0 对 38.0）；LIBERO-OOD 三轴上 VPP2 全部最优；RoboDojo 分维度成功率上，VPP2 在泛化（标准 46.44）、精细（35.17）、长程（37.00）、记忆（55.56）四个维度领先，开放词汇维度（4.08）仍弱于 GPT-6-Astra（31.00），是最明显的短板。

## 相关工作速览

- **视频基础模型**：Latent diffusion 和大规模 DiT 推动了视频生成；大规模双向视频模型提供强通用先验，连自回归生成器也拿双向模型做初始化或教师。这支持了 VPP2"先学可泛化视频模型、再适配高效预测"的路线。
- **World Action Model**：早期方法先生成未来帧再用逆动力学推动作，延迟高；近期方法以视频特征为条件或联合学习预测与动作。但已有证据表明动作学习会损害预训练能力，VPP2 的应对是事件级预训练 + 固定时域后训练 + 单步蒸馏 + 冻结骨干的 MoT 动作专家。
- **可泛化机器人策略**：VLA 模型把视觉语言知识迁移到控制，但通用 VL 能力不必然转化为控制能力；层级规划、强化学习等各有进展，而对任务扰动和新指令的鲁棒性仍是难点——这正是 VPP2 用"指令对齐的未来预测"主攻的方向。

## 总结

- **问题诊断**：WAM 泛化差的根因有二——基座视频模型不为操作优化，且天真引入动作学习会损害视频模型的泛化。
- **核心思想**：降低未来预测的不确定性是泛化的关键；用详细 caption + 事件级预测建立"语义 → 轨迹"的一致映射。
- **方法流水线**：事件级继续预训练 → 固定时域块级后训练 → 一致性蒸馏成单步规划器 → MoT 动作专家（冻结视频骨干 + LoRA），每个动作块延迟约 0.22 秒。
- **视频预测**：14B 的 VPP2 指令跟随成功率 0.80/0.90（人手/机器人），反超 64B Cosmos3 约 11 个百分点；消融证明事件级预测功不可没。
- **真机与 benchmark**：真实 ALOHA 零样本 58.5%（超 $\pi_{0.5}$ 18.5 个百分点）；LIBERO-Pro 45.0%、LIBERO-OOD 63.9%、RoboDojo 32.26% 成功率，均为参评方法最高。
- **展望**：论文传递的信号很明确——未来 WAM 研究应少卷动作建模的结构花样，多把"视频预测的泛化质量"当作一等公民；开放词汇指令跟随和超大视角变化场景仍是待补的短板。

> 本文参考自 [Video Prediction Policy 2: Predict Better, Act Better](https://arxiv.org/abs/2610.10270v2)