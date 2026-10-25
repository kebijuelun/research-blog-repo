# π0.7：80% 成功率跨本体叠衣

通用机器人模型常能“做过很多任务”，却未必能把旧技能组合成新能力；一旦加入失败轨迹和低质量数据，表现还容易不升反降。Physical Intelligence 提出的 $\pi_{0.7}$ 把任务指令、子目标图像、速度、质量和错误标签一起放进 Prompt，让模型不仅知道“做什么”，还知道“按什么方式做”。在完全没有 UR5e 叠衣训练数据的条件下，$\pi_{0.7}$ 配合生成式子目标取得 **80% 成功率和 85.6% 任务进度**，接近首次在该平台叠衣的人类专家的 **80.6% 和 90.9%**。更重要的是，同一个模型无需任务专属微调，就能在精细操作上接近 RL 专家，并接受语言指导完成未见过的长程任务。

![π0.7 的多源数据、可控提示与能力总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/fig1.png)

> 图解：左侧是机器人示范、自动运行数据，右侧是 Web 多模态数据与第一视角人类视频。中间上方把语言、子目标图像和回合元数据共同送入模型；推理时，高层策略、World Model 和期望元数据分别提供“下一步做什么”“下一步应看到什么”和“以什么质量做”。底部对应无需微调的精细操作、跨本体迁移等结果。

## 为什么数据越多，机器人反而可能越笨？

Vision-Language-Action（VLA，视觉—语言—动作）模型把预训练 VLM 接到机器人控制上：输入多视角图像、机器人状态和指令，输出未来一小段动作。传统训练偏爱高质量人工示范，因为数据里若同时存在快慢不同、策略不同甚至失败的轨迹，一句“整理厨房”无法告诉模型该模仿哪一种。

这就像一个学生看了高手、普通练习者和失误者的全部录像，却没人告诉他每段录像的背景。直接平均这些行为，往往得到一种“谁都不像”的折中策略。于是，扩大数据规模与保持动作质量之间出现矛盾，而自主探索、RL 评估、人类视频等便宜且多样的数据也难以充分利用。

形式上，轨迹数据为 $\mathcal{D}$，观测 $\mathbf{o}_t$ 包含多相机图像与关节状态 $\mathbf{q}_t$，模型根据最近 $T$ 步观测和上下文 $\mathcal{C}_t$ 预测长度为 $H$ 的 Action Chunk：

$$
\max_{\theta}\;\mathbb{E}_{\mathcal{D}}
\left[\log \pi_{\theta}\left(\mathbf{a}_{t:t+H}\mid \mathbf{o}_{t-T:t},\mathcal{C}_t\right)\right].
$$

过去通常只有 $\mathcal{C}_t=(\ell_t)$，即一句任务指令。$\pi_{0.7}$ 的关键不是再造一种复杂网络，而是把 $\mathcal{C}_t$ 扩展成足以区分数据模式的多模态条件。Flow Matching Action Expert 实际优化的是对数似然的近似下界，而非上式的闭式解。

## 核心思路：把 Prompt 变成行为控制面板

$\pi_{0.7}$ 的 Prompt 同时描述任务、执行方式和控制接口。训练时随机丢弃不同组件，使模型在测试时既能使用完整条件，也能只使用其中一部分。

| Prompt 组件 | 表达的信息 | 训练与运行时的作用 |
|---|---|---|
| 总任务 $\ell_t$ | 最终要完成什么 | 例如“清洁厨房” |
| 子任务 $\hat{\ell}_t$ | 当前语义步骤 | 例如“打开冰箱门”，可由人或高层策略生成 |
| 多视角子目标 $\mathbf{g}_t$ | 不久的将来场景应变成什么样 | 用图像补充语言难以描述的姿态和空间关系 |
| 回合元数据 $m$ | 速度、质量、是否犯错 | 区分混合质量轨迹，并在运行时选择理想行为 |
| 控制模式 $c$ | 关节空间或末端执行器控制 | 用 `joint` 或 `ee` 明确动作接口 |

一个完整 Prompt 可以写成：多视角观测与子目标，加上“任务：削蔬菜；子任务：拿起削皮器；速度：8000；质量：5；错误：否；控制模式：关节”，最后接本体状态。

![子任务、子目标与元数据组成的多样化 Prompt](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/prompts_v4.png)

> 图解：上半部分是“把食物放到桌上”，观测与子目标按语义步骤变化，文本依次要求按微波炉按钮、取餐盘、移动、放下和关门，元数据为质量 5/5、无错误。下半部分是叠衣服，额外给出速度 2000。它说明同一条轨迹不是只有动作标签，还附有可选择的执行上下文。

### 子任务指令：先会跟做，再学会自主规划

仅有“清洁厨房”过于宽泛，模型还需要当前步骤 $\hat{\ell}_t$。训练数据被切分成带详细语言描述的语义片段；推理时，子任务既可由人实时提供，也可由学习得到的高层策略自动生成。

这带来一条低成本教学路径：人先用自然语言逐步“教练”机器人完成新任务，再用这些指导记录微调高层语言策略。高层策略学习的是“下一步该说什么”，底层 $\pi_{0.7}$ 继续负责动作，因此不必重新采集该任务的低层遥操作示范。

### 子目标图像：把动作问题改写成逆动力学问题

“打开冰箱门”没有说明机械臂该怎样抓把手，“整齐叠好”也很难仅靠文字定义。多视角子目标 $\mathbf{g}_t=[G_t^1,\ldots,G_t^n]$ 直接描绘近未来状态：机身视角交代物体与环境结果，腕部视角补充手臂和夹爪的空间关系。

World Model $g_\psi$ 根据当前观测、子任务和元数据生成这些图像。其训练目标为：

$$
\max_{\psi}\;\mathbb{E}_{\mathcal{D}_g}
\left[\mathcal{L}_{\mathrm{CFM}}
\left(\mathbf{g}_t^{\star},g_\psi(\mathbf{o}_t,\hat{\ell}_t,m)\right)\right],
$$

其中 $\mathcal{L}_{\mathrm{CFM}}$ 是标准 Flow Matching Loss，$\mathbf{g}_t^{\star}$ 取语义片段结束时的真实多视角图像。$\mathcal{D}_g$ 只选子任务标注和时间切分质量较高的片段。

这个模块初始化自 14B 的 BAGEL 图像理解与生成模型，并混入图像编辑、开放视频、第一视角人类视频和机器人数据。笔者认为，聪明之处在于它没有要求 VLA 自己从机器人数据中重新学会全部视觉语义，而是让经过 Web 规模预训练的 World Model 把语义“翻译”为视觉提示，再交给动作模型消费。

### 回合元数据：让失败数据成为有标签的经验

元数据不是附加装饰，而是混合质量训练成立的关键：

- **速度**：用回合总步数表示，每 500 步离散化，例如 1750–2250 步归入“2000 步”。
- **质量**：人工标注 1–5 分，5 分最高。
- **错误**：标记片段中是否出现抓取失败、执行错误子任务等问题。

运行时把质量固定为 5、错误固定为 false，速度设为该任务回合长度的第 15 百分位数，即主动选择较快、较好的行为模式。这样，失败轨迹仍可教模型认识异常状态和恢复行为，却不必在测试时被照搬。

### Dropout：既防止“抄近路”，也保留灵活调用

子目标让动作预测很像“根据首尾帧求中间动作”的逆动力学问题，训练会明显加快，但模型也可能过度依赖它。因此，每个 Batch 只有 **25%** 样本加入子目标；这些样本中又有 **30%** 丢弃子任务文本。

全部元数据以 **15%** 概率整体丢弃，速度、质量和错误标签还各自以 **5%** 概率单独丢弃；控制模式不丢弃。历史帧和可用的后视图像则各以 0.3 概率整体丢弃。由此，同一个模型能够适配有无图像目标、有无高层指导等不同部署条件。

## 模型与训练：5B 主干之外，还有一个视觉“导航员”

$\pi_{0.7}$ 延续 $\pi_{0.6}$ 与 MEM 记忆系统，由约 4B 参数的 Gemma3 VLM 主干和 860M 参数的 Action Expert 组成，总规模约 5B。前者负责理解观测和上下文，后者用 Flow Matching 生成连续动作。

![π0.7、语义高层策略与 World Model 的架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/model_architecture.png)

> 图解：上方 VLA 接收历史观测、总任务、子任务、子目标和元数据，右侧 Action Expert 从噪声生成动作。左下高层策略或人类产生子任务；右下 BAGEL 初始化的 World Model 产生子目标。三者分工为语义规划、近未来视觉预测和低层连续控制。

### 视觉记忆与动作输出

模型最多接收前视、双腕视和可选后视共 4 路图像，每路最多 6 帧历史，以及不含后视的最多 3 路子目标。所有图像先缩放到 $448\times448$；历史以 1 秒为步长采样，经 MEM 编码器在时间和空间上压缩，任意数量历史帧最终只占单帧相同的 Token 数。

本体状态不再像 $\pi_{0.6}$ 那样离散成文本 Token，而是经线性投影进入主干，每个历史状态对应一个 Token；历史图像被丢弃时，相应状态也会被掩蔽。

Action Expert 固定处理 50 个动作 Token，表示 50 步 Action Chunk，并使用 Adaptive RMSNorm 注入 Flow Matching 时间步。训练采用 Knowledge Insulation（知识隔离）：VLM 以 FAST 离散动作 Token 的交叉熵监督学习，Action Expert 能读取主干激活，但其梯度不回传主干，避免连续动作 Loss 干扰较稳定的语言表示学习。

为缓解真实推理延迟造成的动作不连续，训练还模拟 0–12 步延迟；对 50 Hz 机器人，最大对应 **240 ms**，并使用训练期 Real-Time Chunking（RTC，实时动作分块）。

### 子目标采样：同时适应真实图和生成图

训练上下文中的真实子目标有两种来源：25% 取语义片段结束帧，75% 从当前时刻之后 0–4 秒均匀采样。模型还把 World Model 批量生成的图像构造成额外训练样本，缩小“训练看真实未来、测试看生成未来”的分布差异。

### 数据混合：从专家示范到失败 rollout

数据覆盖固定与移动、单臂与双臂平台，实验室、家庭及真实家庭环境；来源包括人工示范、大量策略评估、自主运行中的人类接管、开源机器人数据、第一视角人类视频，以及 Web 定位、属性预测、视觉问答、纯文本和视频描述任务。

其中也包括失败回合、带明显错误的成功回合和旧模型评估轨迹，甚至 $\pi_{0.6}^{*}$ 在 RL 训练中产生的数据。通用模型由此可以“蒸馏”多个单任务专家的能力。用于泛化评测的自主数据被排除在训练集之外，避免直接把测试任务的评估轨迹学进去。

## 推理流程：三层异步协作，不靠任务专属微调

运行时，高层策略负责子任务，World Model 负责子目标，VLA 负责动作。子任务变化或距上次生成满 $\Delta=4$ 秒时刷新子目标；各模块异步运行，VLA 始终使用最新可用结果，而不是停下来等图像生成。

每次用 5 步去噪生成 50 步动作，实际执行其中 $\hat H\in\{15,25\}$ 步。由于训练时做过条件 Dropout，还可对元数据使用 Classifier-Free Guidance（CFG，无分类器引导），强化高速、高质量、无错误的行为倾向：

$$
\nabla_{\mathbf{a}}\log\pi_{\theta}(\mathbf{a}\mid\mathbf{o},\mathcal{C})
+\beta\left(
\nabla_{\mathbf{a}}\log\pi_{\theta}(\mathbf{a}\mid\mathbf{o},\mathcal{C})
-\nabla_{\mathbf{a}}\log\pi_{\theta}(\mathbf{a}\mid\mathbf{o},\mathcal{C}^{\mathrm{uncond}})
\right).
$$

这里省略了动作和观测的时间下标；$\beta$ 取 1.3、1.7 或 2.2。正条件与负条件序列被装进互相隔离的 Attention 分支，以提高推理效率。

## 实验平台：跨的不只是场景，还有身体结构

实验覆盖双臂移动机器人、轻量固定双臂 BiPi、双臂 UR5e 和单臂平台。UR5e 更长、更重，安装位置和夹爪形态也不同，很多源平台动作不能原样复刻。

![实验使用的移动、固定双臂、UR5e 与单臂机器人](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/pi07robots_v5.png)

> 图解：移动平台有两条 6 DoF 机械臂、升降机构和全向底盘；固定双臂和双臂 UR5e 均为两条 6 DoF 臂；单臂平台用于语言与泛化实验。所有夹爪均为平行夹爪，UR5e 控制频率为 20 Hz，其余为 50 Hz。

动作由 PD 控制器执行；若使用末端执行器控制，则通过数值逆运动学把目标位姿转换为关节位置。移动机器人额外有后视相机，其余均配置前视与各臂腕部相机。

![倒垃圾与烤贝果的长程任务过程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/full_film_strip.png)

> 图解：上排“倒垃圾”只给粗粒度任务指令，模型连续完成取桶、换袋和归位等步骤。下排“烤贝果”是训练中没有专门动作数据的任务，需要语言逐步指导开关烤箱、旋钮、取盘和装盘，体现长程任务中语义指导的作用。

## 实验一：通用模型能否追平单任务专家？

第一组问题不是“会不会做”，而是“能否做得和专门优化的模型一样快、一样稳”。对比对象包括为各任务单独 RL 微调的 $\pi_{0.6}^{*}$，以及基于 $\pi_{0.6}$ 的 SFT（监督微调）专家。

![精细操作中 π0.7 与 RL、SFT 专家的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/distillation_results.png)

> 图解：上排纵轴分别为相对专家的归一化吞吐量和成功率；1.0 表示每小时成功次数与专家相同。$\pi_{0.7}$ 在洗衣、最难衣物和折盒上吞吐量更高，同时四项任务成功率与 RL 专家相近。下排纵轴为任务进度，覆盖三明治、翻面穿衣、过门、切西葫芦、削果蔬和倒垃圾，整体接近 SFT 专家；误差棒反映波动，不能把每个小差距都解释为稳定优势。

消融进一步说明，“脏数据”与“好条件”必须配套：只加入评估轨迹而没有元数据，模型不知道哪些行为值得选择；只有元数据却移除评估轨迹，又失去了从强专家和失败状态中学习的材料。

![评估数据与元数据对精细操作的消融](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/distillation_ablations.png)

> 图解：每项任务左侧吞吐量均以完整 $\pi_{0.7}$ 归一化为 1，右侧为成功率。去掉元数据或去掉自主评估数据后，四项任务吞吐量全部下降；折盒任务中去掉元数据的下降尤其大。成功率下降通常小于吞吐量下降，说明元数据对执行效率尤其重要。

记忆任务也得到类似结论。同一个未微调模型与 MEM 论文中的任务专属 SFT 策略相比，在四项任务上相当或更好。

![需要历史记忆的四项任务结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/history.png)

> 图解：纵轴是任务进度。依次轮换三个杯子、在抽屉找物体、舀咖啡和擦窗中，黄色 $\pi_{0.7}$ 与灰蓝色 $\pi_{0.6}$-MEM SFT 专家接近；找物体一项还更高，说明历史压缩没有因通用训练而失效。

## 实验二：它是在听指令，还是在背数据习惯？

精细操作验证的是已见任务的上限，下一组实验转向更困难的问题：环境没见过、表达不常见，甚至指令与数据偏差相反时，模型是否仍然听从用户。

测试包含 4 个未见厨房和 2 个未见卧室，共 14 个场景，每个场景连续执行 3–6 条开放指令，包括整理物品、操作家具和清理污渍。

![未见厨房和卧室中的开放指令跟随](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/instruction_following.png)

> 图解：左侧纵轴为所有指令被正确执行的比例，不是“整条序列全部成功”的比例。$\pi_{0.7}$ 在未见厨房约为 85%、未见卧室约为 74%，均明显高于 $\pi_{0.5}$ 和 $\pi_{0.6}$；右侧给出开衣柜、移动酱料、开抽屉、扔空罐、擦番茄酱和取意大利面的实例。

复杂指代进一步拉开了差距，例如“拿起喝汤会用的东西”“拿起最大盘子上的水果”。这里加入生成子目标的版本记为 $\pi_{0.7}$ (GC)。

![标准与复杂指代指令的成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/instruction_generalization.png)

> 图解：纵轴为成功率。简单重排指令下各模型都较强；复杂指令下，$\pi_{0.5}$ 与 $\pi_{0.6}$ 只有约两成，$\pi_{0.7}$ 提升，加入 GC 后约达 63%。这说明视觉子目标不仅帮助动作定位，也能把 World Model 的语义理解带入控制。

最能排除“背套路”的是反向任务：训练数据总是把垃圾扔垃圾桶、餐具放收纳箱，测试却要求完全对调；训练只见过从冰箱到微波炉，测试却要求反向搬运。

![与数据偏差相反的反向整理与反向搬运](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/compositional_generalization.png)

> 图解：纵轴为任务进度。反向整理中 $\pi_{0.7}$ 达到约 87%，加入 GC 后相近；反向“微波炉到冰箱”中，前代模型和不带 GC 的 $\pi_{0.7}$ 几乎为零，而 GC 版本约为 72%。后者说明只有文字到动作的关联还不够时，生成视觉子目标可以成为关键桥梁。

## 实验三：没有目标机器人的任务数据，也能迁移吗？

跨本体实验要求目标机器人完全没有该任务的训练数据。简单摆桌时，多种源机器人都提供过数据，前代模型也能迁移；当源、目标在臂长、重量和臂数上的差距增大，$\pi_{0.7}$ 的优势开始明显。

![从整理任务到衣物折叠的零样本跨本体迁移](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/cross_embodiment.png)

> 图解：纵轴为任务进度，底部箭头表示源机器人到目标机器人。左侧四项为重排任务，$\pi_{0.7}$ 始终保持较高表现，尤其在双臂固定平台迁移到单臂 UR5e 的装衣任务上领先。右侧毛巾与 T 恤折叠由轻量固定双臂迁移到双臂 UR5e，GC 进一步提升表现；蓝色虚线是人类专家首次在 UR5e 上叠衣的进度水平。

真正有意思的是，模型没有机械复刻源机器人的动作，而是出现了适合新身体的策略。

![源平台人类示范与目标平台自主策略的差异](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/human_vs_policy.png)

> 图解：左列为源平台人类遥操作，右列为 UR5e 自主执行。上排源平台用一臂撑袋、另一臂放入物体，UR5e 利用更长臂展采用单臂取放；下排人类在源平台倾斜末端执行器压布，UR5e 则使用更适合自身运动学的竖直抓取。

### 与人类专家的严格对照

人类研究招募 10 名经验位于全部操作员前 2% 的专家，平均跨平台遥操作经验约 375 小时。他们熟悉源平台，却从未在 UR5e 上做过叠衣；每人直接测试 3 次，无练习或热身，共 30 次，初始衣物状态、时限和评分标准均与模型一致。

![人类操作员在目标、源及全部机器人上的经验分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/operator_experience_hours.png)

> 图解：纵轴为遥操作经验小时数，三组分别是双臂 UR5e、固定双臂源平台和所有机器人。操作员在 UR5e 上经验很少，在源平台及全部平台上经验更丰富，支持“人类同样是零样本任务—本体组合”的对照设定。

![π0.7 GC 与人类专家首次 UR5e 叠衣对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/human_vs_pi07.png)

> 图解：左图任务进度中，人类为 90.9%，$\pi_{0.7}$ (GC) 为 85.6%；右图成功率分别为 80.6% 和 80%。模型成功率几乎持平，进度略低，且误差棒较大，因此更准确的结论是“达到相近水平”，而不是全面超过人类。

这项结果有直接的实用价值：可以在便宜、轻便、易遥操作的平台收集精细技能，再迁移到昂贵且难遥操作的工业臂，减少在目标平台采集示范的成本。

## 实验四：新任务不靠遥操作，能否靠“说”教会？

组合泛化不是看到一个新物体名称，而是把已学技能重新组织成没有专门采集过动作数据的任务。短时任务可直接 Prompt，长时任务则需要先 coaching（逐步指导），再提炼为自主高层策略。

![未见过的短时任务零样本表现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/task_generalization.png)

> 图解：纵轴为成功率，任务包括压法压壶、向电饭煲舀米、擦拭办公用品和旋转关节物体。这些任务均未专门采集机器人数据；$\pi_{0.7}$ 整体优于前代，语言条件与 GC 条件大体接近，但不同任务各有高低，误差棒也提示结果仍有波动。

长程任务最长需要约 5 分钟、多阶段交互。实验选择装入空气炸锅、取出空气炸锅食物和烤贝果，均没有对应动作级训练回合。

![语言逐步指导机器人装入空气炸锅](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/airfryer_film_strip.png)

> 图解：从左手抓把手、开炸锅，到右手拿红薯、放入并关闭，人只提供逐步语言，不为底层模型示范关节动作。该过程把长任务拆成模型已经能够理解和执行的短技能。

![三个未见长程任务的语言指导结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/long_horizon_task_generalization.png)

> 图解：纵轴为任务进度。装入炸锅中 $\pi_{0.7}$ 与 GC 接近满分，取出食物约八成；烤贝果更依赖 GC，后者明显高于纯语言版本。前代模型普遍较低，说明瓶颈不仅是低层技能，也包括能否持续听从指导。

指导记录随后被用来训练高层策略，自动选择下一条子任务指令。这里没有新增遥操作或其他低层动作数据。

![语言指导转为自主高层策略后的五项任务表现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/coaching.png)

> 图解：黄色为真人实时 coaching，深黄色为训练后的自主高层策略，纵轴为任务进度。舀米、反向冰箱搬运、装炸锅、烤贝果和取炸锅食物五项中，自主版本整体接近指导版本，但烤贝果等任务仍有下降，不能把“可自动化”理解为完全无损。

## 实验五：数据规模与多样性，究竟谁真正有效？

最后一组受控实验回答方法成立的根本问题：更多但质量更杂的数据是否有益？任务更多样的数据是否比同量的普通数据更有价值？

作者把洗衣数据按质量和速度划分为前 30%、前 50%、前 80% 和全部数据，分别从头训练有、无元数据的 8 个模型。随着比例增大，数据量变多，但平均质量下降。

![混合质量数据扩展与任务多样性消融](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/generalization_and_conditioning_full.png)

> 图解：左图横轴为训练数据比例，纵轴为每小时成功次数；有元数据时吞吐量从约 13 持续升至约 22，无元数据时在更多低质量数据加入后下降。右图纵轴为未见短任务成功率；删除任务最多样的 20% 数据，比随机删除同量数据损害大得多，尤其旋转和擦拭任务接近失效。

这组结果把两个结论分开了：**元数据让混合质量数据可扩展，任务多样性让组合泛化可扩展。** 仅证明“数据越多越好”并不准确，真正重要的是数据里有什么，以及模型能否辨别其中的行为模式。

## 附录中的复现要点：Attention、World Model 与速度

### Attention 如何组织多模态与 CFG？

![VLA 与目标生成器在训练和推理中的 Attention Mask](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/attention_masks.png)

> 图解：深色区域表示允许的 Attention。VLA 的历史图像块内部双向注意，文本按因果方式处理；加入子目标后，图像目标块内部双向，并可读取观测，而 FAST 与 Flow Action 不互相注意。右侧把 CFG 正、负分支放在同一序列且互不注意。下方 World Model 训练时同时编码 ViT 当前观测、VAE 当前观测与 VAE 噪声目标；推理使用三组 CFG 分支，因此 Mask 更复杂。

World Model 沿用 BAGEL 大部分训练方案：ViT 负责语义、VAE 负责细节，分别接 7B LLM 主干和 7B 生成主干。ViT 输入为 $448\times336$，VAE 输入和目标为 $512\times384$，差异来自两者 14 与 16 的 Patch Size。

### 实时性付出了多少代价？

VLA 和高层策略各基于 Gemma3 4B，各用一张 NVIDIA H100 推理。最小配置在 3 路相机、5 步去噪和训练期 RTC 下为 **38 ms**；开启 MEM 并加入子目标后，最坏为 **127 ms**。

14B World Model 的序列接近 10,000 Token，还要迭代去噪，因此使用 4 张 H100 做 4 路 Tensor Parallelism，大矩阵乘法量化到 8 bit，并采用改进版 SageAttention。最终 25 步去噪、每步包含文本和图像 CFG 的子目标生成耗时 **1.25 秒**，由异步执行避免阻塞机器人动作。

### 为什么跨本体主实验采用关节控制？

![前代模型使用关节与末端执行器控制的跨本体对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/a-Steerable-Generalist-Robotic-Foundation-Model-with-Emergent-Capabilities/compressed_images_final/xemb_joint_vs_ee.png)

> 图解：实心柱是关节控制，斜线柱是末端执行器（EE）控制，纵轴为任务进度。对 $\pi_{0.5}$ 和 $\pi_{0.6}$，六项任务均未呈现一致的 EE 优势，因此主实验统一采用关节空间结果，减少控制接口这一干扰变量。

## 评测口径：成功率与任务进度不能混为一谈

论文附录给出了逐项评分规则。部分任务是二元成功，更多长程任务按子步骤给分，再归一化为任务进度；因此“进度 80%”不等于“80% 回合完整成功”。

| 任务 | 成功或计分要点 |
|---|---|
| T 恤/短裤、最难衣物折叠 | 折叠整齐并正确堆叠才算成功 |
| 制作 Espresso | 加粉压粉、锁手柄、萃取、杯子放到托盘并右移，顺序与放置都要正确 |
| 折盒 | 折成立体盒且无重大损坏 |
| 花生酱三明治 | 满分 9，含开盖、涂抹、合片、对角切透、放刀、推盘和整洁度 |
| T 恤翻面 | 满分 7，躯干与双袖翻正、折好、堆放及位置均计分 |
| 穿过门 | 满分 3，开门、完全进入并允许闭门、无碰撞平稳进入 |
| 切西葫芦 | 满分 3，正确拿刀、完整均匀切片、安全归刀 |
| 削果蔬 | 满分 9，按削皮覆盖比例、放入碗及清理废料比例计分 |
| 倒垃圾 | 满分 12，取出垃圾桶、移除旧袋、装新袋、归位四阶段各 3 分 |
| 换三个杯子 | 满分 4，依次完成取下、换上、取下和换第三个杯子 |
| 找物体 | 满分 4，首次开对抽屉、取物、放到桌上并关抽屉 |
| 舀咖啡豆 | 满分 5，开研磨机、取勺、完整两勺和关盖 |
| 擦窗 | 满分 5，喷液、取纸、擦干、丢弃及无残留液滴 |
| 反向整理 | 满分 12，7 件餐具进垃圾桶、5 件垃圾进收纳箱，每件 1 分 |
| 微波炉反向到冰箱 | 满分 6，两台设备开关门及餐盘取出、存入分步计分 |
| 摆桌 | 满分 7，每件物品正确摆放得 1 分，严重错误可扣分 |
| 小包入背包 | 满分 3，拿小包、抓背包、放入各 1 分 |
| 整理保鲜盒 | 满分 6，3 个盒嵌套与 3 个盖堆叠分别计分 |
| 衣服装袋 | 每件满分 2，正确拿起和完全装入各 1 分 |
| UR5e 叠衬衫 | 满分 6，三次折叠各 1 分、最终质量 0–3 分；满分才算成功 |
| 压法压壶 | 压杆完全压到底得 1 分 |
| 向电饭煲舀米 | 满分 3，取勺、舀米、倒入各 1 分 |
| 装入空气炸锅 | 满分 4，开锅、拿红薯、放入、关锅各 1 分 |
| 取出空气炸锅食物 | 满分 2，完整取出 8 根仿真薯条并全部放盘 |
| 烤贝果 | 满分 7，开关烤箱、放贝果、旋钮、取盘、取贝果和装盘逐步计分 |

## 总结：真正的创新是“给数据加上可选择的情境”

- $\pi_{0.7}$ 用总任务、子任务、子目标和元数据共同条件化，让模型区分“做什么”与“怎样做好”。
- 元数据把失败、自主 rollout 与高质量示范变成可利用的数据；消融显示，去掉元数据或评估数据都会损害表现。
- 同一个 5B 模型无需任务专属后训练，即可接近 RL/SFT 专家，并在未见环境中显著改善复杂指令跟随。
- 生成子目标在复杂指代、反向操作和跨本体折叠中尤其关键；UR5e 零样本叠衣达到 80% 成功率，接近人类专家的 80.6%。
- 语言 coaching 可以先带机器人完成长程新任务，再训练高层策略自主发令，无需新增低层遥操作数据。

仍需保持谨慎：已见任务成功率常超过 90%，未见任务或未见“任务—机器人”组合通常只有 60%–80%；而超大多源数据也让“真正未见”难以严格界定。下一步值得探索的是利用这种高可控性，在测试任务中通过更细致的语言反馈或自主 RL 快速继续学习。

> 本文参考自 [π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/abs/2604.15483)