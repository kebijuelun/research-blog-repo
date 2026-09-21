# Cosmos 3 深度解读：NVIDIA 的全模态世界模型，一个模型统一理解、生成与行动

在 Physical AI（物理人工智能）的语境下，一个机器人或自动驾驶系统往往需要同时调动一整套模型：用 VLM 做场景理解、用视频生成模型做未来模拟、用 VLA / 世界-动作模型做动作预测。这种"拼装式"架构不仅繁琐，而且各模块之间表示不通、误差逐级累积。NVIDIA 最新发布的技术报告 **Cosmos 3** 给出了一个相当激进的答案：用一个统一的 **全模态（Omnimodal）世界模型**，在同一个 Mixture-of-Transformers 架构里同时处理并生成 **语言、图像、视频、音频和动作** 五类模态，把 VLM、文生图/文生视频模型、世界模拟器、世界-动作模型全部"收编"进一个框架。

这篇报告的信息量非常大（正文 + 附录数百个基准与消融），本文按照"提出问题 → 架构设计 → 数据与训练 → 基础设施 → 实验验证 → 消融分析"的逻辑，带大家完整梳理一遍。

## 一、提出问题：为什么要把"理解"和"生成"统一起来？

Physical AI 智能体需要在真实世界中感知、推理并行动，但直接在真实世界训练智能体 **缓慢、昂贵且危险**。因此业界共识是构建"模拟世界"作为训练场，而智能体需要两个本质耦合的能力：

- **理解（Understanding）**：从部分观测中推断语义、状态与动态；
- **生成（Generation）**：预测并模拟未来，推演世界如何演化、智能体该如何行动。

以往的工作把这两根支柱割裂开来：判别式模型（VLM）负责感知推理，生成式模型（视频生成、前向动力学模型）负责模拟，动作模型（VLA、WAM）负责控制。论文的核心论点是：这种分离是 **根本性受限的** ——理解需要对世界未来演化和动作后果的推理，而生成依赖于对世界和智能体行为的紧凑结构化表示。两者的统一对 Physical AI 来说不是锦上添花，而是必需品。

以一个家用机器人"饭后清理餐桌"为例：在现有范式下，它需要拼接一个 VLM（定位餐具并规划）、一个 VLA/WAM（生成动作序列）、一个前向动力学模型（模拟和评估未来状态）。而 Cosmos 3 试图让 **同一个模型** 原生地完成这一切。

![Cosmos 3 总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/introduction/tikz_cosmos3_overview.png)

> 图解：Cosmos 3 作为 Physical AI 的通用骨干网络。通过对语言、图像、视频、音频、动作做联合的"理解 + 生成"建模，它把视觉-语言模型、图像生成模型、音视频生成模型、策略/世界-动作模型、前向动力学模型、逆动力学模型全部统一进同一个网络架构。根据输入-输出配置的不同，同一个模型可以在 VLM、文生图、文生视频、图生视频、视频续写、音视频联合生成、世界-动作模型等角色之间无缝切换。

更进一步，Cosmos 3 把自己定位成 Physical AI 的"训练起点"，从三个层面缓解数据与环境瓶颈： **合成数据生成**（后训练为更好的 T2I/I2V 合成器）、 **任务特化**（无需改架构即可后训练为机器人策略）、以及长远的 **训练环境生成**。

![Cosmos 3 平台定位](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/introduction/tikz_cosmos_platform.png)

> 图解：Cosmos 3 作为 Physical AI 智能体训练的起点。中训（mid-training）得到的基础模型可以在不改架构的前提下后训练成不同应用：更好的合成数据生成器（T2I / I2V）、更强的机器人策略（DROID 上的世界-动作模型），未来还可生成复杂的闭环训练环境。

## 二、模型家族与开源资源

Cosmos 3 包含三个规模档，全部基于双塔 MoT 架构、从预训练 VLM 初始化：

| 变体 | 总参数量 | 骨干 | LLM 层数 | Hidden | 注意力头 | KV 头 | FFN 维度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cosmos3-Edge | 4B | 自研 2B dense（从零训练） | 28 | 2048 | 16 | 8 | 9216 |
| Cosmos3-Nano | 16B | Qwen3-VL 8B 改造 | 36 | 4096 | 32 | 8 | 12288 |
| Cosmos3-Super | 64B | Qwen3-VL 32B 改造 | 64 | 5120 | 64 | 8 | 25600 |

本次报告开源了 Nano 与 Super 两个档位（Edge 将随后续版本发布），并以 **OpenMDW-1.1 许可证** 开放了代码（github.com/nvidia/cosmos）、模型权重（HuggingFace `nvidia/Cosmos3-Super`、`Cosmos3-Nano`、`Cosmos3-Super-Text2Image`、`Cosmos3-Super-Image2Video`、`Cosmos3-Nano-Policy-DROID`）、五个合成数据集（SDG 系列）以及评测基准 Cosmos-HUE。

截至报告撰写时，后训练版本的 Cosmos 3 在 Artificial Analysis 榜单上位列 **开源文生图、图生视频双第一**，并在 RoboArena 上成为 **最强机器人策略模型**。

## 三、模型架构：一个序列，两个世界

### 3.1 模态编码器：统一表示空间的入口

Cosmos 3 为每种模态配备专用编码器，把输入嵌入统一表示空间，并为每个非语言模态加上一个可学习的模态嵌入向量，让共享参数能区分模态：

- **视觉理解**：使用 ViT 编码器（patch 大小 16×16，后接两层 MLP 合并 2×2 token），沿用 Qwen3-VL 的 DeepStack 多层特征聚合与文本时间戳交错方案，与骨干网络 **联合训练**；
- **视觉生成**：使用 Wan2.2-TI2V-5B 的视频 VAE（时间压缩 4 倍、空间压缩 32×32），训练中 **冻结**；
- **音频**：采用 ETTA 的音频 VAE，48 kHz 立体声、hop size 1920，每秒音频对应 25 个 token，同样冻结；
- **动作**：这是 Cosmos 3 的关键创新点，下文单独展开。

**统一动作表示** 值得一提。不同本体（自动驾驶、相机运动、机器人、第一人称人体运动）的原生控制空间完全不同——关节轨迹、方向盘指令、身体姿态、相机变换等。Cosmos 3 把它们映射为共享几何结构的紧凑表示：动作最多包含三个成分—— **ego 位姿**（主观测坐标系）、 **末端执行器位姿** 和 **抓取状态**。位姿用相对变换的"伪动作"表示：

$$
\Delta \mathbf{T}_t = \mathbf{T}_{t-1}^{-1} \mathbf{T}_t
$$

旋转采用 6D 连续表示（z 轴沿手指/夹爪方向，x 轴向右），抓取状态则直接编码当前操控状态（如夹爪开合值、指尖位置）。不同本体通过 **域感知的输入/输出投影层** 接入共享的 MoT 骨干：

$$
\mathbf{z} = \mathbf{W}_{\mathrm{in}}^{(k)} \mathbf{x} + \mathbf{b}_{\mathrm{in}}^{(k)}, \qquad \mathbf{x} = \mathbf{W}_{\mathrm{out}}^{(k)} \mathbf{z} + \mathbf{b}_{\mathrm{out}}^{(k)}
$$

其中 $k$ 是域标识，投影矩阵按域独立、骨干共享，推理时用 SVD 把预测的 6D 旋转还原为 $\mathrm{SO}(3)$ 旋转矩阵。

![统一动作表示](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/model_architecture/action/tikz_action_representation.png)

> 图解：统一动作表示。异构本体控制被映射为由共享几何成分构成的紧凑动作向量：ego 与末端执行器运动用相对位姿伪动作（3D 平移 + 6D 旋转）编码，抓取状态直接编码当前操控状态；域感知投影层处理不同长度的动作向量，同时保留共享语义空间。

### 3.2 Token 编排：AR 子序列 + 扩散子序列

所有任务都被统一成一种交错多模态序列格式，由两个子序列拼接而成：

- **AR 子序列**：负责推理与理解，包含语言 token 和 ViT 编码的图像/视频 token，路由到 Reasoner 参数；
- **扩散（DM）子序列**：负责生成，包含 VAE 编码的视觉 token、音频 token、动作 token，路由到 Generator 参数。

排列规则固定为：AR 在前、DM 在后；DM 内部先放干净的条件 token、再放带噪的目标 token；条件与目标内部按视觉 → 音频 → 动作排序。基于这一格式，各种任务只是"谁干净、谁带噪"的组合问题：

- **语言生成**：只有 AR 子序列，模型就是标准 VLM；
- **文生图（T2I）**：$\mathbf{S}_{\mathrm{T2I}} = [\mathbf{S}_{\mathrm{AR}}, \tilde{v}_1]$，其中 AR 前缀 $\mathbf{S}_{\mathrm{AR}} = [l_1, \ldots, l_n, \langle\text{EOS}\rangle, \langle\text{BOG}\rangle]$；
- **文生视频（+音频）**：$\mathbf{S}_{\mathrm{T2V+Audio}} = [\mathbf{S}_{\mathrm{AR}}, \tilde{v}_{1:N}, \tilde{s}]$；
- **图生视频 / 视频续写**：$\mathbf{S}_{\mathrm{V2V}} = [\mathbf{S}_{\mathrm{AR}}, v_{1:P}, \tilde{v}_{P+1:N}]$，$P=1$ 即 I2V；
- **视频迁移（Transfer）**：$\mathbf{S}_{\mathrm{Transfer}} = [\mathbf{S}_{\mathrm{AR}}, v^{\mathrm{ctrl}}_{1:N}, \tilde{v}_{1:N}]$，控制视频（边缘、深度等）作为干净条件；
- **动作**：三种模式——前向动力学（FD，给动作预测未来视觉）、逆动力学（ID，从视觉转移反推动作）、策略模式（联合预测动作与视频）。

![动作生成模式](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/model_architecture/action/tikz_action_modes.png)

> 图解：动作序列配置。动作 token 位于相邻视频 token 之间（$a_t$ 连接 $v_{t-1}$ 到 $v_t$）。前向动力学模式下视觉 token 带噪、动作 token 干净；逆动力学模式相反；策略模式下两者都带噪、联合去噪。

### 3.3 Mixture-of-Transformers：双塔 + 双流联合注意力

这是整个架构的核心。每个 Transformer decoder 层包含 **两套参数**：Reasoner 塔处理 AR 子序列，Generator 塔处理扩散子序列，两者都从预训练 VLM 权重共同初始化，从而让模型在继承语言与视觉推理能力的同时学会高保真生成。

两塔通过 **双流联合注意力（Dual-Stream Joint Attention）** 交互，但信息流是严格不对称的：

- AR token 只对 AR 子序列做 **因果自注意力**，完整保留 VLM 的自回归特性：

$$
\mathbf{O}_{\mathrm{AR}} = \mathrm{Attn}_{\mathrm{causal}}\big(\mathbf{Q}_{\mathrm{AR}}, \mathbf{K}_{\mathrm{AR}}, \mathbf{V}_{\mathrm{AR}}\big)
$$

- DM token 做 **全双向注意力**，其 key/value 来自 AR 与 DM 的拼接，让每个扩散 token 都能自由读取文本 prompt 和所有条件/扩散 token：

$$
\mathbf{O}_{\mathrm{DM}} = \mathrm{Attn}_{\mathrm{full}}\big(\mathbf{Q}_{\mathrm{DM}}, [\mathbf{K}_{\mathrm{AR}}; \mathbf{K}_{\mathrm{DM}}], [\mathbf{V}_{\mathrm{AR}}; \mathbf{V}_{\mathrm{DM}}]\big)
$$

注意 AR token 永远不会被 DM token 影响，保证了条件通路的因果完整性。

![MoT 架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/model_architecture/tikz_mot_architecture.png)

> 图解：Cosmos 3 的 MoT 架构。左图：单个 Transformer 处理由 AR（蓝色）和 DM（橙色）子序列组成的统一序列，AR 携带离散文本 token 与 ViT 视觉 token，以 `<EOS>` 和 `<BOG>` 结尾；DM 携带各编码器输出的连续 token。每个 Transformer 块内，两路 token 使用各自独立的 LayerNorm 与 MLP，仅在共享自注意力算子处交汇。右图：注意力掩码——AR 为因果掩码，DM 为全注意力。

### 3.4 多模态位置编码：绝对时间轴上的 3D MRoPE

当视频、音频、动作以不同帧率/采样率同时生成时，token 必须对齐到同一根 **物理时间轴**。Cosmos 3 在 Qwen3-VL 的 3D MRoPE 基础上做了两处关键扩展：

**位置索引分配**。AR 子序列完全沿用 Qwen3-VL 的 3D MRoPE（语言 token $t=h=w$ 单调递增，退化为 1D RoPE）。扩散 token 中：视频 token 的三轴 $(t, h, w)$ 分别随时间帧与空间网格变化；图像视为单帧视频；音频与动作 token 只有时间坐标（$h=w=0$），时间索引分别随音频 hop 和动作采样步推进。

**AR 与 DM 之间的时间间隔**。实践中发现，如果扩散 token 直接从最后一个 AR token 的时间偏移开始，会导致视频首帧过饱和和棋盘格伪影（Super 模型上尤其明显）。原因是最后一个语言 token 与第一帧视觉 token 占据了相邻时间位置、时间嵌入几乎相同。解法很朴素：在 AR 与扩散子序列之间插入 **固定的 15000 时间间隔**，形成位置空间上的缓冲区，无需任何架构改动或额外嵌入。

**绝对时间调制（FPS Modulation）**。定义"每秒时间步数" TPS：视频为帧率除以时间压缩比 4，音频为 $48000/1920 \approx 25$，动作即采样频率。取基准 $\mathrm{TPS}_{\mathrm{base}} = 24/4 = 6$，当 token 需要推进一个时间单位时，实际增量为：

$$
\delta t = \frac{\mathrm{TPS}_{\mathrm{base}}}{\mathrm{TPS}}
$$

这样，无论数据是 60 FPS 还是 24 FPS，相同物理时长的内容都占据相同的位置范围。

![3D MRoPE 坐标分配](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/model_architecture/tikz_mrope_coordinate_assignment.png)

> 图解：3D MRoPE 坐标分配示意。左：打包序列中语言、视频（两帧、各 2×2 空间网格）、音频、动作 token 各自获得 $(t, h, w)$ 三元组；语言 token $t=h=w$，视频 token 三轴变化，音频/动作只有时间轴；模态偏移 $k$ 分隔文本与视觉的时间区间。右：FPS 调制将帧索引映射为缩放后的时间位置，使 16/24/30 FPS 下相同的真实时长占据相同的位置区间（24 FPS 为基准帧率）。

## 四、数据体系：理解侧与生成侧的双轨课程

Cosmos 3 的两条通路虽然共享骨干与 token 表示，但数据诉求完全不同：Reasoner 吃"图文/视频-文本配对"的监督数据，Generator 吃"重建式目标"的大规模多模态语料。两者都采用多阶段课程，数据配比随训练阶段演化。

### 4.1 Reasoner 数据：22M 预训练 + 2.2M SFT

**预训练** 以 Nemotron Nano 2 数据合集中挑选的 19.7M 样本为主，外加 2.3M 针对数学、视频、空间定位、指令跟随的增强样本。数据要过两道关卡：

- **语义去重**：把媒体表示与指令-回答文本表示拼接成联合嵌入（图像/文本用 Qwen3-VL-Embedding-8B，视频用 PE-Core-G14-448），先 K-means 聚类、再在簇内做余弦相似度近重去除（阈值 0.95），共剔除 4.23% 的数据；
- **AI 裁判质量过滤**：用 Gemma-4-31B-it 从忠实性、完整性、正确性三个维度打 1–5 分，样本必须在 **三个维度同时** 过阈值才保留。预训练用阈值 2（保留率 78%），SFT 用阈值 5（保留率 46%）——作者特意对比过，阈值过高会不成比例地删掉 grounding 与 caption 类数据，破坏能力分布。

最终预训练混合中 OCR 占 42.9%（9.44M）、2D grounding 占 16.5%、视觉 QA 占 11.3%。

**SFT 阶段** 的 2.2M 样本中视频-文本占 50%，聚焦三大 Physical AI 域：自动驾驶（人工标注 + 自动标注的动作 CoT 超 1.1M 条、Nexar 行车记录仪时序事件定位、MADS 3D 车辆定位）、机器人（动作 CoT——用 Qwen3-VL-72B 生成定位推理、Molmo-7B 做指代定位、MolmoAct/DROID 提供运动规划目标；83K BEHAVIOR-1K 长程规划样本；398K 手术机器人 VQA 对话）、智慧基建（仓库空间智能、5.6M 人工标注行人框、交通异常推理）。此外还有通用空间理解（2D/3D grounding、模拟器 grounding 的空间推理）、时序理解（密集时序 caption、743K 事件三元组、FoundationMotion 运动问答）、物理合理性判断（Cosmos HUE 人评数据 + VideoPhy-2）以及结构化时空场景上采样（即 prompt upsampler 的训练数据）。

![Reasoner 数据构成](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/data/reasoner/tikz_reasoner_pretraining_category_mix.png)

> 图解：Cosmos 3 Reasoner 预训练阶段按能力类别的数据构成环形图。OCR 是最大成分（42.9%），其次为 2D grounding（16.5%）、视觉问答（11.3%）、图像推理（7.5%），辅以文本 QA、caption、视频 QA 等，强调图文对齐、阅读与空间定位能力，同时保留轻量视频成分为后续 SFT 的时序任务做准备。

### 4.2 Generator 图像/视频数据与结构化 JSON Caption

预训练从 **78 亿原始图片、30 亿原始视频** 出发，经过去重、分类、过滤后得到 **7.67 亿图片 + 3.477 亿视频片段**。流水线包括：TransNetV2 场景切分与黑边去除；Qwen3-VL-Embedding-8B / Cosmos-Embed1-448p 嵌入 + cuML KMeans（2 万簇）簇内去重；47 个层级类别的语义标注；美学分/真实感分/DOVER/VTSS 质量分与约 100 个二元伪影标签的过滤。

中训练数据强调质量与域覆盖：图像侧 15.6M 样本（60% 真实 + 36% 合成 + 4% 文字渲染）；视频侧 74.7M 片段（46% 高质量预训练池精选 + 43.9% 机器人/驾驶/人类活动等域数据 + 10.1% 困难案例数据）。另有 3M 视频迁移数据（边缘/模糊/深度/分割/驾驶场景地图控制）。

**结构化 Caption 是本文一个很重要的工程决策**。作者放弃了自由文本 caption，改用预定义语义字段的 JSON 对象：图像 schema 覆盖主体、背景、光照、美学、镜头语言、文字元素等，并独创 **象限扫描** 机制——把图像切成四个象限加中心区分别描述再合并，显著提升多主体、复杂布局场景的细节召回；视频 schema 在此基础上增加动作、状态变化、相机运动、分段时间线、转场、音频描述等时序字段。Caption 模型是自训的 Qwen3-VL-8B LoRA（图像用象限扫描、视频用 8 FPS 抽帧 + 时序二次标注）。在自建的 caption 质量基准（断言级 precision/recall）上，结构化方案在保持高精度的同时显著提升了召回率。

### 4.3 音频数据：让声音与画面"因果对齐"

原始网络视频的音频对训练很不友好：旁白描述画面而非由画面产生，后期配乐会掩盖物理声音。为此：

- **预训练**：1.389 亿条带音轨片段全部保留，6250 万条 30 秒以下短片用 Qwen3-Omni-Captioner 生成音频描述，追求规模与多样性；
- **中训练**：筛选出 1880 万条高精度音视频对（1280 万非语音 + 600 万语音同步）。流水线相当讲究：SAM-Audio 做语音/残余声源分离，SyncNet 做唇形同步打分（`has_face` 且置信度 ≥ 3.0 才算语音同步），FireRedASR2S 估计语音/音乐占比，Qwen3-VL 判断是否"画面中有乐器"（保护乐器演奏视频的音乐不被误删），语音池用 Qwen3-ASR 转写并由 GPT-OSS-120B 合并转写与音频描述。核心原则一句话：**只保留与可见人脸同步的语音，去掉画面外的语音，去掉压过物理声音的非器乐 BGM**。

### 4.4 动作数据：8.4M episodes、61.3K 小时的四大支柱

动作中训练覆盖四大 Physical AI 支柱：第一人称运动（4.13 万小时，占 67.4%，170 万条双手操作 episodes，每帧带头部相机位姿和每只手 21 个 3D 关键点）、自动驾驶（1 万小时，Hyperion 平台驾驶日志挖掘）、机器人（5400 小时，AgiBot/Franka/Google Robot/WidowX/UMI/UR 共 51.67 万 episodes，成功与失败片段都保留）、相机运动（4600 小时，用 ViPE + DepthAnything3 从预训练视频反推相机位姿，190 万条片段）。所有数据转换为统一动作 token 化格式，按维度归一化到约 $[-1, 1]$；多视角拼接成画布并把相机布局写进 JSON 元数据。

![多视角动作打包](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/data/action/multiview_droid_16164052.jpg)

> 图解：多视角动作 prompt 打包示例（DROID）。多个相机视角被拼接成单一画布——上排为腕部相机，下排为左右两个第三人称视角——视角布局元数据直接写进结构化 JSON prompt 的 `cinematography.framing` 字段，让模型把每个像素区域与对应相机流关联起来。

### 4.5 合成数据：五个 SDG 数据集补齐长尾

预训练语料的概念分布天然长尾，机器人、驾驶、仓库等关键 Physical AI 域覆盖不足。Cosmos 3 基于 Isaac Sim / Omniverse 构建了五个合成数据生成（SDG）数据集，全部开源：

| 数据集 | 片段数 | 分辨率/FPS | 特点 |
| --- | --- | --- | --- |
| SDG-PhyxSim | 76,489 次仿真 | 1080p/30 | 10 类刚体交互场景（多米诺、台球、保龄球、wrecking ball 等），4 机位同步，带逐帧分割、无损深度、逐对象物理量（线速度、角速度、质心位移、累计旋转） |
| SDG-RobotSim | 386,270 | 多样 | 移动机器人、四足、人形、双臂、灵巧手；运动 44.2% / 操作 28.5% / 碰撞 27.3% |
| SDG-DriveSim | 264,000 | 4K/24 | 约 1467 小时，7 类长尾场景（cut-in 32.9%、行人 21.1%、变道 12.9%、紧急车辆等），LLM 场景 Agent 从自然语言生成可运行 USD 世界 |
| SDG-SynHuman | 236,937 | 1080p/30 | 5841 小时数字人视频，4050 个人物资产、8184 个动画、14 种相机运动，带逐帧深度与相机参数 |
| SDG-Warehouse | 约 123K | 1080p/30 | 412 小时工业安全事件（叉车-人险情、火灾疏散、叉车撞货架、拣箱），带深度/分割/边缘等密集标注 |

![SDG-PhyxSim 样例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/assets/sdg_phyxsim/wrecking_ball_5524_f046_rgb.jpg)

> 图解：SDG-PhyxSim 的 wrecking_ball 场景在撞击瞬间的 RGB 帧（Corner 机位）。数据集还为同一帧提供质心位移、累计旋转、线速度、角速度的"物理着色"视频——物体颜色编码瞬时物理状态（红=X、绿=Y、蓝=Z，饱和度随幅值增大），让模型直接看到物理量。

![SDG-DriveSim 场景](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/assets/sdg_drivesim/frame_collision.jpg)

> 图解：SDG-DriveSim 代表性场景之一（车辆碰撞）。该数据集专门针对真实车队数据难以覆盖的长尾安全关键场景：紧急车辆交互、绕过停放障碍物、相邻车道 cut-in、恶劣天气能见度、非常规轨迹行人横穿等，每个场景最多可扩展为 10 个关于时间、天气、路面材质、车辆行人资产的确定性排列。

为了验证合成数据与预训练语料确实互补，作者在 Cosmos-Embed1 视频嵌入空间做了 PCA→UMAP 可视化与距离量化：所有 SDG 源都形成独立紧凑的簇，与预训练分布仅窄窄重叠（如 SDG-Warehouse 的局部质心覆盖数仅 1920，而预训练自参考为 19934），确认 SDG 占据预训练分布未覆盖的长尾区域。

## 五、训练方案：先 Reasoner，后 Generator，渐进式课程

整体分两大阶段：先训练 Reasoner，再用其权重初始化 Generator 的两塔（Reasoner 塔冻结、Generator 塔学习生成）。

### 5.1 Reasoner 训练

- **预训练**：语言模型 + ViT + 多模态投影层从头联合训练（作者发现单独的对齐阶段没有必要），next-token 预测目标，两个 epoch，序列上限 16K token，采用平方根归一化的逐 token 损失加权以平衡长短序列；AdamW，LLM/投影层峰值学习率 $5\times10^{-5}$、ViT 为 $5\times10^{-6}$，余弦衰减。
- **SFT**：8200 步、全局 batch 512，重要性感知采样（按数据集价值分配预算），并以 1:4 的比例混入高质量预训练子集防止通用能力退化，外加 80 万条指令跟随数据稳定对话能力。

### 5.2 Generator 预训练

训练目标是所有模态统一的 **rectified flow matching**：对干净目标 $x_0$ 构造带噪样本 $x_\sigma = \sigma \epsilon + (1-\sigma) x_0$，训练去噪器 $v_\theta(x_\sigma, \sigma, c)$ 预测恒定速度 $v^* = \epsilon - x_0$，条件 token 从损失中掩掉。噪声水平 $\sigma$ 按模态独立采样（图像/音频/动作用 logit-normal，视频用 mode sampling），并经 shift 重参数化 $\sigma = s\bar{t}/(1 + (s-1)\bar{t})$ 偏向高噪声区。

关键设计：

- **多分辨率训练**：256p/480p/720p 三档 × 5 种宽高比 × 可变帧数（256p/480p 最多 400 帧、720p 最多 300 帧），图像/视频 256p/480p/720p 的 batch 配比为 1:1:2:1，shift 值分别为 1/3/5；固定 74000 token 的打包预算消除 padding；
- **四种训练模式**：T2I、T2V、I2V、V2V 仅以条件帧数 $T_{\mathrm{cond}}$ 区分（0 / 0 / 1 / 2），采样比 20%/56%/16%/8%；
- **FPS 调制**：时间坐标按真实物理时间分配，同时把时长与 FPS 写进 JSON caption 做文本条件；
- **优化**：只更新生成侧参数，Reasoner 塔冻结；FusedAdamW（lr $10^{-4}$），10% text dropout 支持 CFG。

规模：Cosmos3-Nano 在 **1024 张 GB200** 上训练了 **31.05T token**；Cosmos3-Super 在 **2048 张 GB200** 上训练了 **17.86T token**。

![多分辨率序列打包与数据配比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/model_architecture/tikz_multiresolution_sequence_packing.png)

> 图解：多分辨率训练与序列打包。三个分辨率档在最大帧预算、可用源素材、rectified-flow shift 值上各不相同；来自不同档的变长序列被打包进固定的 74000 token 上下文窗口，无 padding 地最大化 GPU 利用率。预训练采用图像-视频联合训练，视频采样占 80%、图像占 20%，视频批内再按 T2V/I2V/V2V 三种条件模式均匀采样。

### 5.3 Generator 中训练（Mid-Training）

中训练是"通用生成模型 → Physical AI 世界模型"的桥梁，有两个目标： **域特化**（加入高价值 Physical AI 域数据）与 **多模态整合**（引入动作与迁移两种新监督）。配比为：图像 T2I 10%、视频（T2V/I2V/V2V）32%、视频+音频 8%、动作（FD/ID/策略）25%、通用迁移（边缘/模糊/深度/分割）20%、驾驶迁移（场景地图）5%。shift 值提高到 3/5/10 以改善动态与高分辨率伪影；动作损失乘以 10 倍权重补偿归一化动作向量较小的 MSE。此阶段 Nano/Super 分别训练 2.4T / 1.9T token。

![Generator 数据课程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/data/tikz_data_curriculum.png)

> 图解：Generator 数据课程表。每行是一种训练模式、每列是一个训练阶段，彩色单元格标注该阶段的样本数，灰色表示该模式未激活。动作与视频迁移数据在中训练首次引入；中训练产出的 Cosmos3-Nano/Super 基础模型随后分别后训练为 Cosmos3-Super-Text2Image、Cosmos3-Super-Image2Video 与 Cosmos3-Nano-Policy-DROID，架构完全不变。

### 5.4 三个后训练样板

- **Cosmos3-Super-Text2Image**：两阶段 SFT——第一阶段 2 万步（45% 真实图像 + 40% 合成 + 15% 文字渲染），第二阶段用 47 万超高质量图文对精修 2000 步；UniGenBench 总分 91.36 登顶；
- **Cosmos3-Super-Image2Video**：480p、189 帧（约 8 秒@24fps），1 万步、约 500 亿 token；混入 20% T2I token 保持语义对齐，并用 agentic 工作流识别模型弱点定向检索样本；
- **Cosmos3-Nano-Policy-DROID**：从中训 checkpoint 继续训练，动作编码器/解码 MLP/动作嵌入重新初始化并给 5 倍学习率；输入当前本体状态 + 三视角画面（腕部 360×640 在上、两个 180×320 外视角在下拼成 540×640 画布），预测 32 步未来关节位置动作（15Hz）并辅助生成 RGB 视频帧；推理只用 4 步扩散 + CFG 并行 + 跳过视频解码，可部署在 2 张 RTX Pro 6000 上。

## 六、基础设施：为"理解+生成"一体化训练打造的全栈平台

![基础设施总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/infrastructure/tikz_infra_overview.png)

> 图解：Cosmos 3 基础设施四大支柱。数据基础设施把原始多模态流加工成 WebDataset 格式的训练分片；训练基础设施在 NVIDIA GPU 集群上以高效并行化、数据加载与检查点机制消费这些分片；产出的 checkpoint 兵分两路：服务基础设施负责低延迟推理部署，基准基础设施对同一 checkpoint 做标准化评测与回归追踪。

### 6.1 数据基础设施 SILA

SILA（Scalable Infrastructure for Large-scale data processing and Annotation）是专为数十亿级多模态数据加工建的平台：以 Lance 列式表为统一数据层（每个样本一行，caption/标签/质量分都是可追加的类型化列，取代了旧架构里表-per-流水线 + CDC 同步的复杂 join）；以 Lance 片段（fragment）为单位做租约式分布式协调与故障恢复；用 Ray 分阶段流水线执行并按背压限速；模型推理通过 vLLM 节点本地端点完成；支持 Lepton/Slurm 上的机会式集群利用；还暴露了 agent 友好的操作接口，让长时程编排 agent 自动监控、重启失败阶段。效果： **作业启动延迟从 30–60 分钟降到约 5 分钟，整体吞吐提升 10 倍**，峰值期单阶段日处理数十亿行标注。向量检索侧直接用 LanceDB 在同一存储层建 IVF_PQ 索引（4096 维、64K IVF 分区），省掉了独立向量库的同步开销。

### 6.2 训练基础设施的四个杀手锏

1. **Joint Data Loader**：多模态样本 token 数相差两个数量级以上（一条 720p 视频顶几十条短文本），传统"固定样本数"批构造会造成严重负载不均甚至 NCCL 超时。方案是 token 预算打包 + 跨流复用的联合加载器 + **rank 同步流选择**（全局种子选择器让所有 rank 每步处理同一模态/分辨率桶，吞吐 +54%）+ **前看打包**（把超预算样本暂存 look-aside 缓冲区、继续扫描更小样本填缝，有效序列长度 +8%）；
2. **双流扁平注意力**：把 Reasoner 因果注意力与 Generator 跨通路双向注意力各实现为一次变长 SDPA kernel 调用，key/value 流按样本粒度交错为 $[R_0, G_0, R_1, G_1, \ldots]$，比 FlexAttention 基线端到端吞吐 +22%（Hopper 上用 FlashAttention-3，Blackwell 上用 NATTEN）；
3. **选择性激活重计算（SAC）**：按"FLOPs/显存比"排序优先物化注意力输出，Nano 吞吐 +13% 且数值不变；另加 `torch.compile`（fullgraph + dynamic）带来 +41%；
4. **异步检查点**：专用 Gloo 进程组后台写对象存储，保存计划记忆化再省 60% 开销，相对同步检查点端到端省时 4%（Nano）/ 9%（Super）。

另一个有趣的优化是 Wan2.2 VAE 视频分词器：分块编码（256p/480p/720p 分别每次编码 68/24/12 帧）+ AOTInductor 把 45 个静态形状图（3 分辨率 × 5 宽高比 × 3 调用模式）分片到不同 rank 并行编译，预热时间从约 15 分钟降到 1 分钟以内。

稳态吞吐方面（GB200）：Nano 每 GPU 520 TFLOPS、MFU 0.23，每小时 507 次迭代、1623 万视频 token/GPU·时；Super 每 GPU 673 TFLOPS、MFU 0.30，185 次迭代、591 万视频 token/GPU·时。

### 6.3 服务与评测

Reasoner 接入 vLLM 与 TensorRT-LLM（Nano/Super 直接复用 Qwen3-VL 支持，Edge 按 vLLM 贡献规范做了可上游化的集成）；Generator 接入 vLLM-Omni，支持 Cache-DiT、Ulysses 上下文并行、CFG 并行、HSDP、CPU offload、VAE 分块并行与 FP8 量化。PyTorch 参考实现上的优化包括：transformer 块级 torch.compile + CUDA Graph（T2I 提速 30–60%）、CFG 并行（每步近乎减半延迟）、Reasoner 塔输出缓存（条件固定，只需算一次）、以及复用训练打包机制的推理 batching（256p T2V 吞吐最高 +55%）。评测侧用统一的基准系统管理生成与打分解耦的作业流，Reasoner 评测走 VLMEvalKit + vLLM。

![服务延迟](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/infrastructure/serving/cosmos3_serving_latency_combined_1x3.png)

> 图解：Cosmos 3 服务性能。(a) Cosmos3-Nano 720p T2V 单卡延迟在 H100 NVL 与 B200 上的对比；(b) Nano 720p T2I 单卡延迟；(c) B200 上 720p T2V 从 1 卡到 8 卡的延迟扩展性（Nano 与 Super）。越低越好。

## 七、实验验证：几乎全线 SOTA

### 7.1 总览

先看一张总表——Cosmos 3 与各路专用模型在理解（Reasoning）与生成（Generation）能力上的对比（$^\ast$ 为后训练变体，$^\dagger$ 为闭源模型）：

| 模型 | 通用理解 | 机器人 | 智慧基建 | 驾驶 | 文生图 | 文生视频 | 图生视频 | 机器人 FD | 机器人策略 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cosmos3-Super | 73.7 | 57.8 | **62.6** | **79.3** | **91.36**$^\ast$ | **80.0** | **82.8** | **26.0**$^\ast$ | — |
| Cosmos3-Nano | 69.6 | 55.1 | 61.0 | 76.0 | 84.61 | 79.4 | 82.7 | 25.5$^\ast$ | **39.7**$^\ast$ |
| Gemini 3.1 Pro$^\dagger$ | **77.5** | **58.2** | 58.6 | 47.2 | — | — | — | — | — |
| Qwen3-VL-32B | 72.8 | 52.6 | 56.1 | 40.7 | — | — | — | — | — |
| Veo-3.1$^\dagger$ | — | — | — | — | — | 79.1 | 82.6 | — | — |
| Wan2.2-A14B | — | — | — | — | — | 78.0 | 81.3 | — | — |
| Ctrl-World | — | — | — | — | — | — | — | 23.0 | — |
| $\pi_{0.5}$ | — | — | — | — | — | — | — | — | 28.1 |

通用理解上 Cosmos3-Super（73.7）略逊于 Gemini 3.1 Pro（77.5），但在智慧基建与驾驶两个 Physical AI 域反超所有对手，驾驶域（79.3）更是大幅领先 Gemini 的 47.2。

### 7.2 Reasoner：48 个基准

Reasoner 在 48 个基准上评测，归为通用（19 个）、机器人（17 个）、智慧基建（VANTAGE-Bench + TAR）、驾驶（LingoQA + 两个内部安全关键分类基准）四类。结论：通用能力上优于 Cosmos-Reason2（受益于多 20% 的预训练数据多样性），与开源模型相当但仍落后 Gemini 3.1 Pro；在机器人、智慧基建、驾驶域则超越包括 RynnBrain、MiMo-Embodied、Gemma-4 在内的所有开源与闭源对手，仅机器人域与 Gemini 3.1 Pro 有小差距。值得一提的是 VideoPhy2（物理合理性）上 Cosmos3-Super 得 47.4，远高于 Gemini 3.1 Pro 的 28.7。

### 7.3 图像生成：开源第一

Cosmos3-Super-Text2Image 在 UniGenBench（600 原始 + 570 新增 Physical AI prompt）上总分 **91.36，超过 Gemini 3 Pro Image 的 90.69，位列所有参评模型第一**；英文长文本渲染 CVTG-500L 的 GNED 达 80.88 也是最高。在 Artificial Analysis 文生图竞技场（2026-05-28）中，它排名 **开源权重模型第一、全部模型第四**。

![T2I 样例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/results/sft_t2i/sft_t2i_demo.jpg)

> 图解：Cosmos3-Super-Text2Image 的生成样例。图像兼具物理合理性与照片级真实感：物体几何连贯、物体与环境交互一致。所有图像均由单次上采样 JSON prompt 生成（shift=3.0、guidance=4.0、50 步扩散），展示了其作为机器人、自动驾驶等场景"真实世界图像模拟器"的潜力。

![T2I 竞技场榜单](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/results/sft_t2i/2026-05-28-Cosmos3-Super-Text2Image-Leaderboard.jpg)

> 图解：Artificial Analysis 文生图榜单（2026-05-28）。Cosmos3-Super-Text2Image 在开源权重模型中排名第一，含闭源模型在内总排名第四。提交版本使用了 agentic 上采样流程（见消融章节）。

### 7.4 视频生成：PAIBench-G / RBench / Physics-IQ 三线告捷

- **PAIBench-G**（1044 个图文 prompt 对、六大 Physical AI 域，Overall = 0.5×Quality + 0.5×Domain，与人类 ELO 的 Pearson 相关 0.918）：Cosmos3-Super 在 T2V（80.0）与 I2V（82.8）双赛道总分第一， **超过闭源的 Veo-3.1（79.1 / 82.6）**；
- **RBench**（650 个具身机器人案例，与人类判断 Spearman 相关 0.96）：Cosmos3-Nano 58.4% 与 Seedance-1.5-Pro 并列第二，为开源最佳之一；
- **Physics-IQ**（396 个真实物理场景，专测物理遵循）：Cosmos3-Super I2V 直接得分 43.8（加 WMReward+BoN 后 48.9）、V2V 直接得分 59.7（BoN 后 63.4）， **两种模式、有无重排序四个设置全部 SOTA**，超过 Sora2 与 Magi-1。

| 模型 | PAIBench-G T2V | PAIBench-G I2V | RBench I2V |
| --- | --- | --- | --- |
| Cosmos3-Super | **80.0** | **82.8** | 58.1% |
| Cosmos3-Nano | 79.4 | 82.7 | 58.4% |
| Wan2.2-A14B | 78.0 | 81.3 | 50.7% |
| Veo-3.1（闭源） | 79.1 | 82.6 | 56.3% |
| Wan 2.6（闭源） | 78.6 | 81.9 | **60.7%** |

### 7.5 人类评测：Cosmos-HUE 与 Human World Bench

自动指标对长尾物理/具身失败系统性不敏感（同一批 prompt 上，T2V 模型在 PAIBench-G 总分只差约 4 分，而人评可拉开约 10 分），因此作者设计了 **Cosmos-HUE**：用三层 VLM 流水线（Domain Strategist → Scene Parser → Auditor，均基于 GPT 5.2）为每个 prompt 自动生成最多 20 个原子化二元问题（Yes/No/Unclear，Unclear 按 No 处理），覆盖语义对齐、物理定律、几何推理、视觉完整性四个维度；每个 (视频, 问题) 由两名标注员独立打分、分歧交 QC 仲裁。单条视频得分为：

$$
\mathrm{HUE}(v) = \frac{\sum_{q \in \mathcal{Q}_v} \mathbf{1}[\mathrm{ans}(v,q) = \text{Yes}]}{|\mathcal{Q}_v|} \times 100\%
$$

结果：HUE T2V 上 Veo-3.1（91.3）领先， **Cosmos3-Super（89.3）为开源最佳** 且在 12 个分轴中的 9 个领跑开源，并在 AV（87.7）与 Physics（91.5）两个域击败包括闭源在内的所有生成模型；HUE I2V 上 Cosmos3-Super（89.6）与 Veo-3.1（89.7）仅差 0.1 分。真实视频 GT 参考分为 93.6（T2V）/ 94.4（I2V），头部模型距真实视频约 2.3 分。

**Human World Bench（HWB）** 专测任务指令下的第一人称人体操作生成：Cosmos3-Super 得 **71.9，全场第一**，比 Veo-3.1（67.8）高 4.1 分，比最强非 Cosmos 开源基线 Wan2.2-A14B（60.7）高 11.2 分。

### 7.6 音频生成：语义对齐最强，保真度尚有差距

作者构建了 Cosmos-SoundBench（144 个非语音 prompt，源自 FoleyBench），用多模型 MLLM 裁判流水线计算语义音视频分 $\mathrm{SAV} = 0.6\,\mathrm{SA} + 0.3\,\mathrm{AVAlign} + 0.1\,\mathrm{VisualSupport}$，再与感知音质 PQ 合成 $\mathrm{AVQ} = 0.5\,\mathrm{SAV} + 0.5\,\mathrm{AQ}$。结果：Seedance-1.5-Pro 凭 PQ 优势拿到最高 AVQ（7.64），但 **Cosmos3-Nano 包揽 SAV / SA / AVAlign 三项第一，Cosmos3-Super 拿下 Visual Support 第一（9.18）** ——说明 Cosmos 3 的中训练在"声音与画面事件的语义绑定"上最强，剩余差距集中在底层音频保真度。

![音视频对齐](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/results/audio/av_align_no_reels.png)

> 图解：音频-视频事件对齐示例。Cosmos3-Nano 生成视频的选定帧与生成音频的频谱图配对：彩色帧标注锤子敲击时刻，其时间标记与频谱中的尖锐瞬态重合；灰色帧为两次敲击间的非接触时刻，没有可比声学瞬态。这是"声学瞬态与视觉撞击时刻对齐"的定性证据。

### 7.7 视频迁移：统一骨干取代 ControlNet 全家桶

在 PAIBench-C（600 片段 × 模糊/边缘/分割/深度四种控制）上，Cosmos3-Nano 在感知质量（DOVER 10.39）与分割 mIoU（0.72）领先，Cosmos3-Super 在边缘 F1（0.50）与深度 si-RMSE（0.58）领先，双双超越使用逐模态 ControlNet 分支的 Cosmos-Transfer2.5。结论很硬： **单一统一骨干原生支持四种空间控制，不再需要专用适配器**。驾驶场景迁移（AVBench-C，486 个单视角片段 + 世界场景地图控制）上同样全面打平或超越基线，人评视频质量 2.86/2.82 vs 2.59。

![驾驶场景迁移](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/results/transfer/avtransfer02.jpg)

> 图解：驾驶场景视频迁移结果。上行为 720p 控制视频——编码了 HD 地图元素（车道线、道路标线、杆状物、红绿灯及其状态）与按粗类别着色的 3D 立方体（卡车、车辆、行人等），可表达含立交桥的复杂道路拓扑；下行为 Cosmos3-Nano 据此生成的逼真帧。

### 7.8 动作：一个模型通吃 FD / ID / 策略

作者系统对比了两种初始化：PT-init（预训练 checkpoint，没见过动作数据）与 MT-init（中训练 checkpoint，见过跨域动作数据）：

| 模型 | AV 逆动力学 RRE°↓ | AV RTE(m)↓ | 相机 FD RRE°↓ | 相机 FD ATE(m)↓ | 第一人称 FD PSNR↑ | 机器人 FD PSNR↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Cosmos3-Super (MT-init) | 0.232 | **0.014** | **0.142** | **0.99** | **16.19** | **26.04** |
| Cosmos3-Nano (MT-init) | **0.211** | **0.014** | 0.147 | 1.24 | 16.12 | 25.52 |
| Cosmos3-Nano (PT-init) | 0.249 | 0.017 | 0.172 | 1.61 | 15.22 | 23.24 |
| VGGT | 0.596 | 0.768 | — | — | — | — |
| Lingbot-World | — | — | 0.299 | 2.88 | — | — |
| LOME | — | — | — | — | 9.36 | — |
| Ctrl-World | — | — | — | — | — | 22.99 |

三个观察：其一，PT-init 已经能打平甚至超过领域专用基线（说明通用世界模型预训练本身就蕴含动作先验）；其二，MT-init 一致优于 PT-init（统一动作中训练提供了可迁移的动作域先验）；其三，Cosmos 3 在相机运动 FD 上把 Lingbot-World 的 ATE 从 2.88m 压到 0.99m，在机器人 FD 上 26.04 dB vs Ctrl-World 的 22.99 dB。

![相机前向动力学](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/results/action/camera_fd_figure.png)

> 图解：相机前向动力学对比。给定复杂真实轨迹，Cosmos3-Nano (MT-init) 在生成视频中忠实复现指定相机运动。每个示例第一行为序列起始附近帧、第二行为末尾附近帧，向下箭头表示时间推进，旁边文字标注指令化的相机运动。

**机器人策略** 是重头戏。Cosmos3-Nano-Policy-DROID 在三大基准全部登顶：

- **RoboLab-120**（120 个语言条件任务 × 3 种指令粒度）：Specific 指令下成功率 **39.7%**，大幅超过 $\pi_{0.5}$ 的 28.1% 与 DreamZero 的 25.2%，在所有指令粒度与难度等级上领先；且优于直接从预训练初始化的版本（30.2%），证明动作中训练数据的价值；
- **RoboArena**（真实世界、众包双盲 A/B 对比）：截至 2026-05-30 排名第一；
- **MolmoSpaces**：All Combined 设定下 39.0% oracle 成功率，排名第一（2026-06-20），且与 RoboLab/RoboArena 提交的是完全相同的模型与超参，未做基准特调。

![RoboArena 榜单](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/results/action/roboarena_leaderboard_final.jpg)

> 图解：RoboArena 真实世界基准排行榜（2026-05-30），Cosmos3-Nano-Policy-DROID 位列第一。该基准允许任何拥有 DROID 平台的人在任意环境与任务上做双盲 A/B 对比，最终分数由成对偏好聚合而成。

![真实机器人部署](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Cosmos-3-Omnimodal-World-Models-for-Physical-AI/figures/results/action/snapshots_20260520_02_put_the_screwdriver_on_the_top_shelf_view_1_zoom_tlr10_spaced.jpg)

> 图解：Cosmos3-Nano-Policy-DROID 的真实机器人部署快照（任务："把螺丝刀放到顶层架子上"）。策略能可靠完成简单抓取放置与需要多步顺序执行的长程任务，对未见过的物体与任务有强泛化，且容忍失败重试与执行过程中的人为干预。

**新本体快速适配**：在从未见过的 LIBERO-10 环境上，MT-init 500 步即达 24.6% 成功率（PT-init 为 0%），2000 步时达 **97.4%**（PT-init 为 95.2%），中训练先验显著加速了收敛。

## 八、消融实验：每个设计决策都有据可查

附录中的消融是这份报告最有营养的部分之一，挑重点：

**Reasoner 对 Generator 的增益**。用 Cosmos3-Nano Reasoner 替换 Qwen3-VL-8B 作为理解塔（Generator 均从零训练，90K 步、256 GPU），T2V 的 Domain 分从 73.7 升至 75.7，增益集中在物理域：Robot +4.8、AV +2.3、Industry +1.0——说明在 Physical AI 数据上训练过的 Reasoner 能为 Generator 提供更好的嵌入。

**FPS 控制方式**。对比无控制、纯文本控制、纯 MRoPE FPS 调制、两者结合四种设置，以视频质量 VQ 与运动保真度 MF 的乘积为综合分：

$$
\mathrm{MF} = \big(1 - |\mathrm{DD} - \mathrm{DD}_{\mathrm{ref}}|\big)(1 - \mathrm{MC}), \qquad \mathrm{MC} = \mathbb{E}_p\!\left[\frac{\mathrm{var}_p}{\mathrm{var}_p + \mathrm{mean}_p^2}\right]
$$

其中 DD 是动态程度、MC 衡量同一 prompt 下控制的稳定性。两者结合最优（综合分 9.81 vs 基线 8.51），且 VQ 在四组间波动不到 0.2——增益全部来自时序行为而非画质。

**音频是否伤害视频质量？** 不会。联合视频-音频预训练相比纯视频训练，T2V 总分 79.1 vs 78.6、I2V 82.2 vs 81.7，视频指标不降反升。

**FD/ID/策略三模式能否共享结构？** PushT 实验：单模式各训 2K 步 vs 联合训 6K 步（每模式优化步数相同）。联合模型 ID MSE 从 $1.11\times10^{-3}$ 降到 $3.09\times10^{-4}$（降 72%），策略覆盖率 74.1%→77.3%，代价仅是 FD PSNR 从 27.13 微降至 26.22。

**视频-动作一致性**。Cosmos3-Nano-Policy-DROID 只在 DROID 上训练，在未见过的 RoboLab 环境中，把预测的动作 chunk 放进仿真器执行，再与模型联合预测的视频算 PSNR：左第三人称视角 23.19 dB、腕部视角 17.33 dB——预测动作与预测视频高度一致，这是"世界模型"属性成立的关键证据。

**SDG 合成数据消融**。每个 SDG 源单独微调都提升其对应域（PhyxSim 给 Industry +0.85、SynHuman 给 Robot +1.23 且总分最高 79.79），但也都有一致的短板：**Human 域在所有变体上无一例外地下降**——即使是人造人数据集 SynHuman 也没能挽回，说明当前模拟器对真人外观、运动与行为的微妙还原仍不足以跨越 sim-to-real 鸿沟。混合全部五源（SDG-All）则 9 项指标中 8 项为正，数据多样性抑制了单源偏差，因此最终模型在中训练阶段把 SDG 与真实高质量视频混合使用。

**动作域协同矩阵**。跨域共同训练呈正迁移：AV 数据让相机 FD PSNR +0.86；机器人域之间广泛互益（WidowX-250 从 Google Robot 获得 +1.39 FD PSNR）；人类第一人称数据也给 AgiBot 带来稳定增益（预热后适配，FD PSNR 后期 +1.3~1.6）。

**Cosmos3-Edge 的 LLM 训练**。Edge 的 2B 骨干从零训练：15T token 基础预训练（AdamW、WSD 调度、8K 序列）→ 90B token 长上下文扩展（128K 序列、RoPE base 1e8）→ 26M 样本 SFT。文本基准上 HMMT25 Feb 76.3 vs Qwen3.5-2B 的 22.9，GPQA 56.4 vs 51.6，数学与科学推理大幅领先。

**Agentic 上采样**。Cosmos3-Super-Text2Image 在 Artificial Analysis 的提交使用了一个 agentic 闭环：LLM（GPT-5.5）迭代上采样 prompt → VLM 批评家（Gemini3.1-Pro）按物理、解剖、光照、文字等维度打分（1–10）并列出缺陷 → LLM 重写正负 prompt，最多 2 轮、分数 ≥9 且无严重问题则早停。这套"结构化 JSON prompt + 迭代精修"正是 Cosmos 3 JSON 接口带来的独特玩法。

## 九、总结

Cosmos 3 的核心贡献可以归纳为一句话：**把"理解世界"和"模拟世界"放进同一个可扩展的序列建模框架，并第一次把"动作"提升为与语言、视觉、音频平级的一等模态**。其技术支柱包括：模态专用编码器 + 统一 token 编排（AR 子序列做理解、扩散子序列做生成）、双塔 MoT 与双流联合注意力（理解不被生成污染，生成充分条件于理解）、面向物理时间轴的 3D MRoPE 扩展、统一动作表示与域感知投影、以及结构化 JSON caption 与 prompt 上采样构成的"控制语言"。外围还有一整套与之匹配的工程体系：24M 级 Reasoner 数据课程、数百亿候选的 Generator 数据流水线、五个开源 SDG 合成数据集、SILA 数据平台、定制注意力 kernel 与分布式训练栈、vLLM-Omni 服务集成。

实验上，Cosmos 3 在理解侧（驾驶、智慧基建域）与生成侧（T2I、T2V、I2V、物理遵循、人评、机器人策略）几乎全线达到或超越专用模型与闭源系统，同时保持了完全开源（OpenMDW-1.1）。当然它也有明确的边界：通用理解仍落后 Gemini 3.1 Pro，音频保真度不及头部闭源模型，合成数据对 Human 域的 sim-to-real 鸿沟仍未跨越。作者对 Cosmos 3 的长远定位是成为连接合成世界与真实世界的桥梁——更好的合成数据、更好的特化起点、以及未来更好的闭环训练环境。

> 本文参考自 [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800)