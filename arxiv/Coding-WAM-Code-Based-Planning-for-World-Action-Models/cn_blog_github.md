# Coding-WAM：代码规划让真机得分提高 25.1 点

机器人既要“想清楚下一步做什么”，也要“预见动作会带来什么后果”，但现有的代码规划与世界模型往往各自只擅长其中一半。这个断层直接限制了长时序、多步骤操作中的语义理解、技能复用和稳定执行。Coding-WAM（论文中的模型名为 **AW-0**）把代码、未来视频和机器人动作放进同一个联合去噪过程，让程序计划同时引导世界预测与连续控制。它在 8 个真机任务上取得 **71.1%** 的平均得分，比最强基线高 **25.1 个百分点**；在 RoboTwin 2.0 的 Clean 设置中达到 **80.8%** 成功率，比 WLA 高 **16.0 个百分点**。

![Coding-WAM 总览：预训练数据、三专家模型、技能标注和混合执行](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/teaser_v6_1003.png)

> 图解：左侧是约 3,000 小时的 YAM 机器人与第一视角人类操作数据；中间的 AW-0 由 Code、Video、Action 三个专家组成；下方流水线把关键帧转成对齐的程序并沉淀技能库；右侧展示平行夹爪与灵巧手任务。生成的代码默认只负责规划，也可以通过有效性检查后进入 Hybrid Execution，与动作专家的 Action Chunk 交替驱动机器人。

## 为什么世界模型还需要“会写代码”？

先把两类方法的分工看清楚。

**World Action Model（WAM，世界动作模型）**同时预测未来视觉状态与动作。它像一名凭经验演练的机器人：在真正动手前，先想象物体如何移动、接触如何发生，再据此生成电机控制信号。DreamZero、Cosmos Policy 等工作已经证明了这种“视觉前瞻”的价值。

但视频和 Action Chunk 并不会自动回答：长任务应该拆成哪些子步骤？当前该调用哪个技能？技能需要什么对象与参数？因此，WAM 擅长物理后果，却缺少显式的组合式技能结构。

**Agentic Robotics** 则更像拿着流程单的调度员。Code-as-Policies、ASPIRE、CaP-X 等系统通过程序选择工具、绑定对象、组合技能并重新规划，结构清楚，也方便复用。然而，代码之外的执行器未必理解真实物理动态；系统能力还受底层 Primitive API 的覆盖范围限制。

AW-0 的关键选择是：**不把代码放在策略外部当唯一控制器，而把代码变成端到端模型内部的高层规划接口。**默认模式下，程序并不直接执行，而是作为条件影响视频与动作生成；只有 Hybrid 模式才让通过检查的程序真正接管部分阶段。

这一区别很重要。它既保留了学习型 Action Expert 对精细接触和灵巧动作的适应性，也让显式技能、对象参数和控制逻辑参与长时序推理。

## 总体框架：三个专家，共享一次生成过程

AW-0 使用 Mixture-of-Transformers（MoT）架构。可以把它理解为三位共用一张工作台的专家：Code Expert 写当前子任务，Video Expert 想象接下来会发生什么，Action Expert 决定具体怎么动。三者保留各自的网络参数，却在每层 Transformer 中通过 Masked Joint Attention 交换信息。

![AW-0 的 Mixture-of-Transformers 架构与联合去噪流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/method_v6_0926.png)

> 图解：上方从左到右是代码、视频、动作三个生成流；共享注意力按 $C\rightarrow V\rightarrow A$ 的块状因果顺序传递信息。左下方表示一次联合去噪同时更新代码 Token、视频 Latent 和动作；右下方表示代码训练时随机遮挡，采样时按置信度逐步揭示。右侧执行区区分 Pure Action Chunk 与 Hybrid 两条路径。

### 三个专家各自负责什么？

- **Code Expert**：由 SDAR-1.7B 初始化语言部分，接入冻结的 Qwen3-VL ViT。当前多视角图像被编码成 Visual Token，与指令、历史记忆和程序 Token 一起输入，采用 LLaDA 风格的 Masked-Token Diffusion 生成技能调用与控制逻辑。
- **Video Expert**：从 Cosmos-Predict2.5-2B 初始化，在 VAE Latent 空间预测未来帧。它读取代码流，因此“想象未来”不再只靠视觉惯性，也受到当前语义计划的约束。
- **Action Expert**：一个从头训练的 Action DiT，读取代码与视频表示，并结合机器人状态生成连续 Action Chunk。真实机器人使用统一的 64 维状态—动作空间，预测跨度为 32 步。

笔者认为，最聪明的不是简单增加一个代码模块，而是注意力的方向。若采用 $V\rightarrow C\rightarrow A$，代码只能根据已经形成的视频做解释；采用 $C\rightarrow V\rightarrow A$ 后，代码先规定“要做什么”，再影响“未来会怎样”和“手该怎么动”。后面的消融实验也验证了这一点。

## 联合建模：离散程序与连续动作如何一起训练？

代码是离散 Token，视频与动作是连续变量，三者的噪声机制并不相同。AW-0 将它们写成同一个条件生成问题。

定义程序为 $C_0=(c_1,\ldots,c_{N_c})$，未来视频 Latent 为 $Z_0$，长度为 $H$ 的动作序列为 $A_0=(a_t,\ldots,a_{t+H-1})$。模型条件包括当前多视角观察 $\mathcal{O}_t$、语言指令 $L$、机器人状态 $S_t$ 和此前程序切片的压缩记忆 $M_t$：

$$
\mathcal{C}_t=\{\mathcal{O}_t,L,S_t,M_t\},\qquad
p_\theta(C_0,Z_0,A_0\mid\mathcal{C}_t)
$$

给定遮挡后的代码 $\widetilde C_r$、带噪视频 $Z_{\sigma_v}$ 和带噪动作 $A_{\sigma_a}$，统一模型输出代码 Logits 与两个连续速度场：

$$
(\ell_\theta^c,v_\theta^v,v_\theta^a)
=F_\theta(\widetilde C_r,Z_{\sigma_v},A_{\sigma_a},\sigma_v,\sigma_a;\mathcal{C}_t)
$$

这里 $r$ 是代码遮挡比例，$\sigma_v$ 与 $\sigma_a$ 分别是视频和动作的噪声水平。训练时三种噪声独立采样；生成时三流在每个去噪步骤中共同演化，而不是先完整写完代码、再串行生成视频和动作。

### 连续流：视频与动作的 Flow Matching

对 $m\in\{v,a\}$，令 $X_0^v=Z_0$、$X_0^a=A_0$，带噪输入写为：

$$
X_{\sigma_m}^m=(1-\sigma_m)X_0^m+\sigma_m\epsilon_m,\qquad
\epsilon_m\sim\mathcal{N}(0,I)
$$

噪声水平经过 Shift 变换：

$$
\sigma_m=\frac{\gamma_m u_m}{1+(\gamma_m-1)u_m}
$$

视频和动作均取 $\gamma_m=5$。视频原始采样变量由 $u_v=0.999b$、$b\sim\mathrm{Beta}(7,1)$ 给出，动作使用 $u_a\sim U(0,1)$。两个连续专家回归速度目标 $\epsilon_m-X_0^m$。

### 离散流：代码的 Absorbing-Mask Corruption

代码采用不同的破坏方式：以 $r\sim U(0,1)$ 为遮挡比例，独立地将程序 Token 替换为 Mask：

$$
\widetilde c_i=
\begin{cases}
[\mathrm{MASK}], & b_i=1,\\
c_i, & b_i=0,
\end{cases}
\qquad b_i\sim\mathrm{Bernoulli}(r)
$$

只有程序切片被遮挡并计算 Cross-Entropy；观察帧、Visual Prefix 和语言条件保持干净。推理时，默认画布包含 128 个 Mask Token，模型在 4 个联合去噪步骤中逐步提交高置信度预测，已经揭示的位置保持不变。

### 为什么要重采样视频噪声？

预训练视频模型偏向高噪声区域，但 Action Expert 需要覆盖更广的视频表示，才能获得稳定的预测引导。AW-0 因而将原始分布与均匀分布各按 0.5 混合：

$$
q(u)=\alpha p_{\mathrm{pre}}(u)+(1-\alpha)p_{\mathrm{ref}}(u),\qquad
p_{\mathrm{ref}}=U(0,1),\qquad \alpha=0.5
$$

再用重要性权重区分“学习视频本身”和“为代码、动作提供预测引导”：

$$
w_v(u)=\frac{p_{\mathrm{pre}}(u)}{q(u)},\qquad
w_g(u)=\frac{p_{\mathrm{ref}}(u)}{q(u)}
$$

$$
\mathcal{L}=\mathbb{E}_{u_v\sim q}\left[
\lambda_v\bar w_v\ell_v+
\bar w_g(\lambda_a\ell_a+\lambda_c\ell_c)
\right]
$$

其中 $\ell_v,\ell_a$ 是 Flow Matching Loss，$\ell_c$ 是代码的 Masked Cross-Entropy；横线表示截断并归一化到单位 Batch 均值的权重。实际损失权重 $(\lambda_v,\lambda_c,\lambda_a)=(1,0.5,1)$，重要性权重截断为 4.0。

这套设计解决了一个容易忽略的问题：**视频预测质量高，不等于它的中间表示在所有噪声阶段都适合指导动作。**重采样让动作与代码看到更均衡的视频状态，同时避免破坏原视频生成分布。

## 数据从哪来：把操作视频变成可复用程序

统一架构还缺少一类监督：每段机器人或人类视频究竟对应什么程序？人工为近 300 万个子任务切片逐段写代码并不现实，因此作者设计了三阶段自动标注流水线。

![自动程序标注与技能发现流水线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/code_label_pipeline_v3_1003.png)

> 图解：顶部是技能发现阶段，Coding Agent 根据 Primitive API、已有技能和子任务帧写程序，Coordinator 提炼跨任务重复模式；中部是冻结后的 50 项技能库；底部是语料标注阶段，系统先查询 Subtask-to-Code Cache，接触与运动证据匹配就复用，未命中才重新调用模型并写回缓存。

### 第一步：发现技能，而不是堆积任务脚本

作者先从每个任务抽取 1 条 Episode。Gemini 3.1 Pro 根据声明式 API 和有序帧生成子任务级程序；Claude Code 配合 Claude Opus 4.8 担任 Coordinator，审计重复模式并提炼通用技能。

候选技能必须满足几个条件：参数不绑定单个 Episode、至少跨 2 个任务重复出现、抽象粒度是一个动作技能而非完整任务配方，并且不与现有技能重复。最终冻结的技能库包含 YAM 的 23 项技能和第一视角人类数据的 27 项技能，共 50 项。

例如，折叠布料可以封装为：

```python
def fold_edge_over(robot, item_name, left_delta, right_delta):
    robot.grasp(item_name, 'left')
    robot.grasp(item_name, 'right')
    robot.servo_ee_delta('left', left_delta)
    robot.servo_ee_delta('right', right_delta)
    robot.release('left')
    robot.release('right')
```

调用时只需写：

```python
fold_edge_over(robot, 'blanket', [0.15, 0.0, 0.0], [0.15, 0.0, 0.0])
```

技能把反复出现的“双臂捏边—搬运—释放”压缩成一个命名接口；对象与位移仍是参数，因此不是某条示范的硬编码回放。

### 第二步：冻结词表，建立可验证缓存

技能库冻结后，系统重新标注抽样 Episode，使所有缓存都使用最终词表。缓存键对应任务与子任务，同一子任务可以保存多种合法实现，对象名也会规范化，避免把颜色等实例属性误当成通用模板。

声明式 API 的意义在于：离线示范无法真正执行候选程序，但 API 可以声明自己会产生什么接触事件。YAM 上，程序隐含的夹爪开合序列必须与传感器记录一致；人类第一视角数据没有同样可靠的接触传感器，则比较双手运动轨迹轮廓，只有距离低于阈值才复用，循环模板也不直接复用。

### 第三步：标注全语料，并对齐训练切片

缓存命中时复用程序，未命中时生成新标签并写回。全语料共有 **2,075,373 个子任务切片**，因此“验证后复用”比逐切片调用模型更可扩展，同时保留帧与代码的时间对齐。

训练样本由当前观察、指令和活跃的程序切片组成。历史切片被压缩为 $K$ 个可学习 Memory Slot，作为 Prefix Token 放在提示前；训练时以概率 $p$ 用模型自己的预测替换历史标注，缓解训练与推理的历史分布差异。每个 Episode 开始时记忆清空，之后随重规划逐步累积。

## 复现细节：数据、表示与训练配置

解决了模型和标注之后，下一个问题是：这些模块怎样落到统一的数据表示与可执行训练配置中？

### 预训练语料

| 数据 | YAM 机器人 | 第一视角人类 | 合计 |
|---|---:|---:|---:|
| Episodes | 73,101 | 175,541 | 248,642 |
| 时长（小时） | 1,919.30 | 993.91 | 2,913.22 |
| 任务数 | 279 | 327 | 606 |
| 子任务切片 | 986,314 | 1,089,059 | 2,075,373 |
| 每 Episode 平均切片数 | 13.5 | 6.2 | 8.3 |

YAM 数据以 30 Hz 记录，包含左腕、右腕和顶部 RGB 视角；人类数据只有头戴 RGB 视角，训练时复制到三个输入槽，手部跟踪结果被重定向为每手 22 个灵巧手关节目标。

### 统一 64 维状态—动作空间

每侧包含 9 维末端位姿、22 维手部关节和 1 维平行夹爪宽度，因此双侧共有：

$$
2\times(9+22+1)=64
$$

其中 9 维位姿由 3 维位置与 6D Rotation 表示组成。状态全部使用绝对值；Action Chunk 的末端位姿使用相对 Chunk 首帧的增量，夹爪或手部关节使用绝对目标。某个 Embodiment 不用的维度置零且不归一化，其余维度按第 1 和第 99 百分位数归一化。

这一表示让平行夹爪和 22-DoF 灵巧手共享模型结构，也为跨机器人、人类示范的联合预训练提供统一接口。

### 模型结构

| 配置 | Code | Video | Action |
|---|---|---|---|
| Backbone | SDAR-1.7B | Cosmos-Predict2.5-2B | Action DiT |
| Residual Width | 2,048 | 2,048 | 1,024 |
| FFN Width | 6,144 | 8,192 | 4,096 |
| Attention | Bidirectional | Bidirectional | Bidirectional |
| Transformer 层数 | 28 | 28 | 28 |
| Query / KV Heads | 16 / 16 | 16 / 16 | 16 / 16 |
| Head / Attention Width | 128 / 2,048 | 128 / 2,048 | 128 / 2,048 |

三者共享 Qwen3-VL-2B ViT Tower，视觉深度与宽度分别为 24 和 1,024；Spatial Patch 为 $16\times16$，Merge 为 $2\times2$。每个视角产生 70 个 Visual Token，三个视角共 210 个，映射宽度为 2,048。图像分辨率为 $224\times320$，每段视频采样 9 帧、Stride 为 4，但只有首帧作为观察条件。

真实机器人状态和动作宽度均为 64，RoboTwin 均为 14，Action Horizon 为 32。代码训练预算为指令与程序合计 512 Token，210 个 Visual Prefix 另行计算。

### 初始化与训练

Video DiT 使用 Cosmos-Predict2.5-2B 初始化，原 VAE 换成冻结的 Wan2.2 VAE，具有 48 个 Latent Channel，视频输入与输出投影随之重建。Code Expert 的 Token Embedding、LayerNorm、FFN 和 Vocabulary Head 从 SDAR-1.7B 初始化并冻结，只训练 MoT Attention Projection 与 Modulation 参数；Qwen3-VL Vision Tower 冻结，Merger 可训练。Action DiT 从头训练。

| 项目 | 设置 | 项目 | 设置 |
|---|---|---|---|
| Video / Action Loss | Flow Matching | Code Loss | Masked CE |
| Video Mixture $\alpha$ | 0.5 | Sigma Shifts $(v,a)$ | $(5,5)$ |
| Loss Weights $(v,c,a)$ | $(1,0.5,1)$ | Importance Clip | 4.0 |
| Optimizer | AdamW | Adam 系数 | $(0.9,0.95)$ |
| Peak Learning Rate | $10^{-4}$ | LR Schedule | Cosine |
| Linear Warmup | 5% | Weight Decay | 0 |
| Gradient Clip | 1.0 | Precision / ZeRO | bf16 / Stage 1 |
| Code / Video Dropout | 0.15 / 0 | Seed | 42 |
| GPU | 48 张 NVIDIA H100 | 每 GPU Batch | 6 |
| Gradient Accumulation | 9 | Effective Batch | 2,592 |
| Training Budget | 1 Epoch | Joint Denoising Steps | 4 |
| Video / Action Solver | UniPC / Euler | Code 顺序 | Confidence |
| Code Canvas | 128 Token | Early-Reveal $\rho$ / Warmup | 1 / 0 |

预训练还会在 15% 的 Batch 中把指令和程序输入替换为 Mask，但保留 Visual Prefix。下游 YAM 与 Tianji–Sharpa 任务均使用每任务 100 条示范、微调 20 Epoch；RoboTwin 使用 16 张 GPU、Effective Batch 96、学习率 $2\times10^{-4}$，训练 100,000 个优化步骤。

## 实验一：RoboTwin 2.0，默认强在 Clean，Hybrid 强在 Random

实验比较四类基线：强调语义推理的 VLA $\pi_{0.5}$、使用大规模人类第一视角数据预训练的 EgoScale、联合视频与动作的 DreamZero，以及联合文本子任务、子目标图像与动作的 WLA。

RoboTwin 2.0 包含 50 个任务，每任务用 50 条 Clean Demonstration 训练，共 2,500 条；每任务每设置评估 50 次。Clean 使用固定光照与干净背景，Random 随机化纹理、光照、桌面高度与干扰物。

| 方法 | Clean 成功率（%） | Random 成功率（%） | 平均（%） |
|---|---:|---:|---:|
| $\pi_{0.5}$ | 54.9 | 27.7 | 41.3 |
| DreamZero | 33.6 | 17.6 | 25.6 |
| EgoScale | 26.2 | 2.9 | 14.5 |
| WLA | 64.8 | 3.6 | 34.2 |
| **AW-0 默认模式** | **80.8** | 20.6 | **50.7** |
| **AW-0 Hybrid** | 63.1 | **29.2** | 46.2 |

![RoboTwin 2.0 各方法在 Clean 与 Random 设置下的成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/results/robotwin_bars_2.png)

> 图解：横轴按方法分组，纵轴为成功率，百分比越高越好；每组两根柱分别对应 Clean 与 Random。默认 AW-0 在 Clean 上最高，Hybrid 在 Random 上以 29.2% 超过 $\pi_{0.5}$ 的 27.7%，但它在 Clean 上明显低于默认模式。

默认 AW-0 的平均成功率最高，优势主要来自 Clean：80.8% 比 WLA 高 16.0 个百分点。到了 Random，默认模式降至 20.6%，说明视觉分布偏移仍会伤害像素级世界预测，代码并不能消除这一问题。

Hybrid 则呈现互补性：Random 从 20.6% 提升到 29.2%，保留自身 Clean 成功率的约 46%；默认模式只保留约 25%。代价是 Clean 从 80.8% 降到 63.1%，低了 17.7 个百分点。

这里需要谨慎解读：仿真 Hybrid 使用另行收集的 CaP 程序示范，而默认模式使用 RoboTwin 原生运动规划示范，因此比较的是两套完整配置，**并没有单独隔离“执行方式”这一个变量**。

### Hybrid 到底用了多少代码？

Hybrid 每次重规划都同时得到代码切片 $c$ 和 Action Chunk $a_{1:H}$。代码依次通过 Stage、Dependency、Syntax、Whitelist 与 Vacuity 等检查后才执行；否则使用同一 forward pass 的动作序列。系统不会自动修复原始代码，也不会预测“代码是否比动作更好”，有效性就是主要准入标准。

| 设置 | Episode 集合 | 代码执行步占比 | 直接动作步占比 | 每 Episode 步数 |
|---|---|---:|---:|---:|
| Clean | 全部 | 8.5% | 91.5% | 26,720 |
| Clean | 成功 | 14.5% | 85.5% | 11,549 |
| Clean | 失败 | 6.3% | 93.7% | 52,687 |
| Random | 全部 | 5.6% | 94.4% | 41,716 |
| Random | 成功 | 35.0% | 65.0% | 8,400 |
| Random | 失败 | 3.7% | 96.3% | 55,483 |

在成功的 Random Episode 中，代码承担了 35.0% 的仿真步数，说明提升确实与代码执行有关；但全部 Episode 的占比被大量运行到步数上限的失败样本稀释。Clean 下准入代码反而可能替换掉本来更强的动作，并继承 Molmo、SAM 3、Contact-GraspNet 等底层 Primitive 的失败模式。

![RoboTwin 中代码执行与直接动作交替完成积木排序](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/blocks_ranking_rgb_progress.png)

> 图解：这是 `blocks_ranking_rgb` 的成功 Episode，帧按执行进度排列，颜色区分 Code Execution 与 Direct Action。规则物体和明确的 RGB 顺序适合程序化 Pick-and-Place，三个搬运循环主要由代码完成，阶段交界处由动作专家接管。

![RoboTwin 中代码定位瓶子、动作专家补救抓取的混合过程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/pick_dual_bottles_progress.png)

> 图解：这是 `pick_dual_bottles` 的成功 Episode。代码能定位并接近瓶子，却因形状不规则未能稳固抓取；直接动作接手完成抓取后，代码再次接管后续阶段，显示两条通道的互补关系。

## 实验二：8 个真机任务，平行夹爪与灵巧手都受益

真机实验使用两种平台：YAM 是固定底座双臂系统，每臂 6-DoF 并配平行夹爪；Tianji–Sharpa 使用人形躯干、双臂和每手 22-DoF 的 Sharpa 灵巧手。两者都输入顶部或头部视角加左右腕部视角，共三路 RGB。

![YAM 双臂平行夹爪真机平台](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/hardware_setup/hardware_setup_yam.png)

> 图解：YAM 工作站由两条 6-DoF 机械臂和线性夹爪组成，上方 RealSense D405 观察桌面，两只腕部相机提供局部视角；图中底部相机不参与策略输入。

![Tianji–Sharpa 双臂灵巧手真机平台](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/hardware_setup/hardware_setup_tianji_sharpa.png)

> 图解：Tianji–Sharpa 工作站包含 Tianji 躯干、双臂和两只 Sharpa Wave 灵巧手，头部 SHW5G 相机与双腕相机共同提供三路观察。手指动作由 Manus 手套采集并重定向为每手 22 维关节目标。

每个任务使用 100 条示范后训练，采用 3 个训练 Seed，每个 Seed 评估 16 次，即每任务 $3\times16$ 次 Rollout，并随机化初始物体位姿。下表不是单纯成功率，而是按任务子步骤累计的 Rubric Score。

| 方法 | 摆水果 | 装背包 | 换灯泡 | 叠毛巾 | 用夹子摆水果 | 发扑克 | 翻记分牌 | 解魔方 | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $\pi_{0.5}$ | 70.8 | 49.7 | 63.8 | 81.7 | 7.3 | 13.1 | 26.3 | 36.1 | 43.6 |
| DreamZero | 53.3 | 36.7 | 67.4 | 79.8 | 22.6 | 37.1 | 28.4 | 42.5 | 46.0 |
| EgoScale | 37.5 | 27.3 | 60.2 | 53.3 | 17.6 | 25.6 | 40.8 | 24.9 | 35.9 |
| WLA | 39.6 | 29.6 | 56.0 | 74.6 | 14.2 | 22.9 | 29.7 | 52.3 | 39.9 |
| **AW-0** | **91.7** | **64.6** | **75.0** | **84.4** | **54.2** | **58.3** | **68.8** | **72.1** | **71.1** |

AW-0 在全部 8 项任务上均为最高分，平均 71.1%，比最强基线 DreamZero 的 46.0% 高 25.1 个百分点。两类提升尤其清楚：

- **语言跟随与语义推理**：摆水果要求按指令选择对象，翻记分牌要求计数，AW-0 分别达到 91.7% 和 68.8%。
- **接触丰富与精细技能**：换灯泡、发扑克、解魔方需要预测接触和协调多关节动作，代码提供技能结构，视频提供物理动态，两者共同改善控制。

8 个任务的评估内容可以概括为：

| 平台 | 任务 | 主要评分点 |
|---|---|---|
| YAM | Arrange Fruits | 移除未指定水果、取回指定水果、终局匹配且无掉落 |
| YAM | Pack Backpack | 双手接力拉拉链、开包、装物、重新闭合 |
| YAM | Replace Lightbulb | 先测试灯是否正常；坏灯需拆下、更换并点亮，好灯应停止操作 |
| YAM | Fold and Stack Towels | 折平 3 条毛巾、堆成一叠、颜色顺序符合指令 |
| Tianji–Sharpa | Arrange Fruits with Tongs | 取夹子、按指令摆放水果、无掉落、归还夹子 |
| Tianji–Sharpa | Deal Poker | 双手交接牌堆、拇指分出单牌、抓取并插入牌槽 |
| Tianji–Sharpa | Flip Scoreboard | 固定底板、准确翻动 $k$ 次、显示目标分数后按按钮 |
| Tianji–Sharpa | Solve Rubik's Cube | 拿起魔方、将被随机转动的层反向转 90°、复原并放回 |

![换灯泡任务的关键帧与对齐代码标签](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/keyframes_codegt/replace_lightbulb.png)

> 图解：关键帧按时间展示测试开关、固定灯座、拆卸与更换灯泡等子步骤，程序块与对应帧区间对齐。该任务还包含“灯泡正常”的条件分支：若按下开关后灯亮，正确策略是不再更换并回到初始姿态。

![发扑克任务的关键帧与对齐代码标签](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/keyframes_codegt/deal_poker.png)

> 图解：图中展示双手交接牌堆、右手拇指顶出单牌、捏住并插入牌槽的过程。每个色块对应一个程序切片，体现代码如何把接触密集的长任务拆成可监督的子步骤。

![解魔方任务的关键帧与对齐代码标签](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/keyframes_codegt/solve_rubiks_cube.png)

> 图解：每次 Rollout 随机选择一个面并随机顺、逆时针打乱，可见颜色和摆放角度也随机化。策略需要拿起魔方，将被转动的层完整转回 90°，而不是背诵固定层或固定转法。

真机 Hybrid 只在 YAM 上测试，因为当前缺少面向灵巧手的完整 Code-as-Policy 系统。在 Fold and Stack Towels 上，同一后训练模型、同一评估条件下，Hybrid 得分 **90.6%**，高于纯 Action Chunk 的 **84.4%**；这项比较没有仿真 Hybrid 的数据采集差异。

![真机叠毛巾任务中的代码执行与直接动作交替](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/real_hybrid_progress.png)

> 图解：单个 Episode 从上排延续到下排，绿色与紫色分别表示代码执行和直接动作。灵巧折叠主要依赖 Action Expert，接近与堆叠等结构化运动更多由代码执行，后者直接提供堆叠顺序和运动方向。

## 消融实验：收益来自代码，还是来自更大的模型？

总体结果之后，更关键的是拆开设计变量。作者围绕注意力顺序、视觉特征、代码内容、生成方式和推理成本逐项消融，所有变体沿用真机后训练与评估协议。

### 代码必须引导视频，而不只是读取视频

![不同代码与视频注意力顺序的真机任务得分](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/results/ablation_attention_bars.png)

> 图解：横轴为 8 个真机任务，纵轴为平均 Rubric Score，柱形依次比较 $C\rightarrow V\rightarrow A$、$V\rightarrow C\rightarrow A$ 与无代码的 $V\rightarrow A$。$C\rightarrow V\rightarrow A$ 在每个任务都最高；无代码方案的平均分为 32.9%，完整模型为 71.1%。

以 Pack Backpack 为例，三种顺序分别得到 64.6%、29.1% 和 16.7%；Arrange Fruits 分别为 91.7%、31.3% 和 27.1%。这不仅说明“有代码更好”，也说明代码的位置决定了它能否塑造预测中的物理未来。

### 语义规划需要 ViT，而不是复用 VAE 特征

| 视觉条件 | 摆水果 | 装背包 | 换灯泡 | 叠毛巾 | 夹子摆水果 | 发扑克 | 翻记分牌 | 解魔方 | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT（AW-0）** | **91.7** | **64.6** | **75.0** | **84.4** | **54.2** | **58.3** | **68.8** | **72.1** | **71.1** |
| VAE | 83.3 | 39.6 | 66.7 | 61.3 | 41.7 | 47.9 | 50.0 | 53.9 | 55.6 |

VAE 特征面向像素重建，未必保留“该选哪个物体、当前处于什么任务状态”所需的语义区分。ViT 在 8 项任务上全部更好，Pack Backpack 和解魔方尤其明显。

### 注释、技能与程序结构缺一不可

![完整代码、去注释、去技能和自然语言子目标的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/results/ablation_label_bars.png)

> 图解：横轴为 8 个任务，纵轴为 Rubric Score，四种柱形分别是完整代码、无注释代码、无技能代码和自然语言 Subgoal。完整代码在全部任务上超过无注释与无技能版本，并在 7/8 个任务上超过 Subgoal；唯一例外是用夹子摆水果，Subgoal 为 58.3%，完整代码为 54.2%。

| 规划表示 | 8 任务平均分（%） | 相对完整代码 |
|---|---:|---:|
| **完整代码** | **71.1** | — |
| 无注释代码 | 62.8 | -8.3 |
| 无技能代码 | 63.3 | -7.8 |
| 自然语言 Subgoal | 63.6 | -7.5 |

结果说明，代码收益不是“换了一种文本格式”这么简单。注释连接任务意图与程序结构，技能调用提供可复用抽象，显式参数和顺序则让计划更贴近执行。Deal Poker 中，去掉技能后从 58.3% 降至 42.0%，尤其能说明技能本身的价值。

### 联合去噪更快，也比 AR 更稳

| 任务 | Joint Denoise（AW-0） | Autoregressive（AR） |
|---|---:|---:|
| Arrange Fruits | **91.7** | 81.3 |
| Pack Backpack | **64.6** | 62.5 |
| Replace Lightbulb | 75.0 | **79.2** |
| Fold Towels | **84.4** | 53.2 |
| Fruits w/ Tong | **54.2** | 20.8 |
| Deal Poker | **58.3** | 44.1 |
| Flip Scoreboard | **68.8** | 39.3 |
| Solve Cube | **72.1** | 55.8 |
| **平均** | **71.1** | **54.5** |

![联合去噪、无代码与自回归生成的真机得分及推理延迟](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/results/joint_denoise_speed_bars.png)

> 图解：横轴为无代码、Joint Denoise 和 AR 三种方案，左纵轴与柱形对应 8 任务平均 Rubric Score，右纵轴与红线对应 NVIDIA RTX 5090 上的端到端延迟。三者分数为 32.9%、71.1%、54.5%，延迟分别为 0.19、0.23、0.55 秒。

联合去噪在 7/8 个任务上超过 AR，平均高 16.6 个百分点；速度为 0.23 秒对 0.55 秒，达到 **2.39 倍加速**。加入代码相比无代码只增加 0.04 秒，即 21.1% 延迟，却显著提升全部任务。

AR 训练使用 Teacher Forcing，推理时却要以自己生成的前缀为条件，误差容易沿 Token 累积并影响后续视频。联合去噪减少串行依赖，并让代码在整个生成过程中持续提供语义条件。延迟统计包含视觉编码、联合去噪、视频解码和动作后处理。

## 附录证据：程序真的看懂了场景吗？

基准得分证明“有效”，代码质量分析则进一步回答“学到了什么”。作者在预训练验证集上直接评估 Code Expert，而不是只观察下游控制结果。

| 指标 | 数值 | 含义 |
|---|---:|---|
| Teacher-Forced Loss | 0.535 | 给定正确前缀时的代码损失 |
| Teacher-Forced Accuracy | 0.897 | Token 级准确率 89.7% |
| Grounding Gap | 1.774 | 打乱视觉观察后代码损失的绝对增量 |
| Parse Rate | 0.875 | 自由生成程序的可解析率 87.5% |
| Entity F1 | 0.534 | 字符串实体与参考程序的重合度 |
| Primitive F1 | 0.365 | 机器人 API 名称与参考程序的重合度 |
| Skill-Call Rate | 0.292 | 29.2% 的生成程序包含技能式 Helper Call |

打乱图像使 Loss 增加 1.774，说明代码确实依赖视觉上下文，而非只背任务指令。但 87.5% 的可解析率与明显更低的 Entity、Primitive F1 也提醒：**语法像程序，不等于语义和参数都正确。**

![YAM 验证集上的预训练代码生成案例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/pretrained_code_generation_yam.png)

> 图解：任务是用布清洁键盘，生成程序包含布的交接与循环 `scrub_back_and_forth`，保留了对象、手臂分配和循环结构；但位移参数多出两个元素，说明正确技能计划仍可能带有阻止直接执行的参数错误。

![第一视角人类验证集上的预训练代码生成案例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/pretrained_code_generation_human.png)

> 图解：在“Squeeze toothpaste”任务中，模型为当前窗口生成拧开瓶盖的准备子任务，包括固定牙膏管、反复旋转和移开瓶盖，调用、参数与循环结构与该窗口标注一致。

后训练后的在线案例也呈现同样的两面性。Pack Backpack 中，模型生成的抓取技能、拉链头对象、右臂分配和数值参数均与参考程序一致；解魔方中，`hold(robot, 'rubiks cube', 'right')` 的技能与对象正确，但注释换行缺少注释符，使原始输出仍无法直接执行。

![Pack Backpack 真机部署中的在线代码生成](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/posttrained_code_generation_yam.png)

> 图解：左侧为重规划时的观察，随后是模型在线生成的程序与执行帧；参考程序来自同任务、同子任务的另一条示范。两者在 `approach_best_grasp`、目标对象、手臂和参数上对应。

![Solve Rubik's Cube 真机部署中的在线代码生成](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/posttrained_code_generation_sharpa.png)

> 图解：生成程序选择了正确的保持技能、魔方对象和右手，但注释格式存在错误。这个案例直观地区分了“计划语义正确”与“原始代码必然可执行”。

### 代码也改善了未来视频的对齐

![Arrange Fruits 中三种规划方式的未来帧预测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/video_quality_comparison.png)

> 图解：三行依次比较 AW-0、自然语言 Subgoal 和无代码模型，列为右腕视角的未来帧。面对“左碗最后只留芒果”的指令，AW-0 将交互对准芒果，Subgoal 在苹果与芒果之间含糊，无代码模型则指向苹果；三次 Rollout 的初始观察相似但不完全相同。

![Deal Poker 中三种规划方式的插牌未来帧预测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/video_quality_comparison_deal_poker.png)

> 图解：三组未来帧聚焦扑克牌对准并插入窄槽的最后阶段。AW-0 的相对运动更连贯，Subgoal 与无代码版本在手和物体附近出现更多模糊与形变；输入是同一 Episode 邻近时刻的相似而非完全相同观察。

这些定性结果支持一条完整因果链：显式技能和对象参数让 Video Expert 的未来更符合任务，Action Expert 再读取这些预测表示，从而获得更可靠的执行引导。不过，它们仍是少量定性案例，不能替代大规模视频质量指标。

## 失败案例与边界：代码不是万能保险

![六类真机失败案例及局部放大](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Coding-WAM-Code-Based-Planning-for-World-Action-Models/figures/appendix/failure_analysis.png)

> 图解：每行按时间展示一个失败 Episode，最右侧放大末帧红框区域。六类问题依次为夹到篮沿、拉背包时包体倾倒、灯泡与灯口未精确对准、夹爪闭合戳到梨、交接扑克时滑落，以及翻记分牌时指尖失去接触。

这些失败主要集中在几何与接触层面：

1. **Container Grasp**：接近水果时夹爪越过篮沿，闭合后夹住篮壁，需要估计周围几何并检查 clearance。
2. **Object Dragging**：拉拉链时缺少对柔性包体的动态支撑，需要更频繁的视觉反馈与双手协调。
3. **Imprecise Alignment**：灯泡到了灯口上方却未满足位置和角度对齐，说明“到达区域”离“可旋入位姿”仍有距离。
4. **Tool Collision**：夹子尖端在接近梨时保持闭合，工具开合状态没有与轨迹协调。
5. **Slipping Off**：交接扑克时，接手尚未稳定支撑，给手已经释放。
6. **Contact Loss**：翻牌需要持续接触，指尖短暂碰到边缘后滑脱，分数并未改变。

由此可以归纳三项边界。首先，自动标注质量与冻结技能库覆盖范围限制规划上限，未来需要用执行反馈修正参数、扩充技能。其次，Hybrid 的仲裁器只检查有效性，不比较代码与动作的预期成功率，导致 Clean 场景也可能退化。最后，系统尚不能从在线交互中持续学习，真机 Hybrid 也尚未覆盖灵巧手。

## 总结：代码把规划、预测与执行连成闭环

- AW-0 将离散代码、连续视频和连续动作放入同一个 MoT 联合去噪过程，代码不再是策略外部的孤立控制器。
- $C\rightarrow V\rightarrow A$ 的注意力顺序让技能计划先引导未来预测，再引导动作；这一结构在全部 8 个真机任务上优于无代码版本。
- 自动标注流水线用接触与运动证据验证缓存复用，从 2,913.22 小时数据中提炼出 50 项跨任务技能。
- 真机平均得分达到 71.1%，超过最强基线 25.1 个百分点；联合去噪相对 AR 还实现 2.39 倍推理加速。
- 默认 Action Chunk 与可执行代码具有互补性：前者强在 Clean 与精细接触，后者提升 Random 鲁棒性，但现有有效性仲裁仍有明显权衡。

下一步最值得探索的是让执行反馈反过来评估计划可靠性、修正技能参数并扩充技能库，使当前“一次预训练加任务微调”的统一架构真正走向持续 Agentic Learning。

> 本文参考自 [Coding-WAM: Code-Based Planning for World Action Models](https://arxiv.org/abs/2610.30136)