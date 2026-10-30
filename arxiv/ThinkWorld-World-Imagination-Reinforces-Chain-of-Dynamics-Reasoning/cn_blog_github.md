# ThinkWorld：世界想象带来93.3%真机成功率

机器人要走出训练场，最大的障碍之一不是“没见过某个物体”，而是无法把动作与它引起的世界变化联系起来；这限制了 VLA（Vision-Language-Action，视觉—语言—动作）模型在新任务、新环境中的泛化。人类视频包含远多于机器人示教的行为经验，却没有可直接执行的动作标签，因此一直难以同时兼顾数据规模与动作监督。ThinkWorld 的核心思路是：先把人和机器人的动作写成共同的 Chain-of-Dynamics（CoD，动态链），再用“想象的一秒后未来”与真实未来是否一致来强化这条推理链。它在 120 次真机试验中取得 93.3% 的 Micro 成功率，最强基线 MolmoAct2 为 60.0%；在仅见标准 LIBERO、直接测试扰动版 LIBERO-PRO 时，平均成功率也达到 55.6%。

![ThinkWorld 从人类与机器人经验学习可迁移的动态推理](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/thinkworld_teaser_blue_v2_final.png)

> 图解：ThinkWorld 不只从机器人示教中学习控制，还从人类第一视角视频学习动作引起的动态变化。CoD 将不同 embodiment（本体）的运动映射到共享表示，再由未来想象提供反馈，最终支持真机泛化与按照人类视频组合原子技能。

## 为什么机器人需要理解“动作之后会发生什么”

通用机器人策略通常把视觉观测、语言指令映射为可执行动作。随着机器人示教规模扩大，VLA 已能完成许多操作，但其行为边界仍受动作标注轨迹覆盖范围制约。采集这些轨迹需要真实硬件、特定机器人接口和大量人工操作，成本远高于收集普通人类视频。

人类视频提供了另一条扩展路径，却带来一个两难问题：

- **Latent Action 路线**：从相邻帧中推断隐动作，不需要动作标签，容易扩展；但隐变量缺少约束，可能学到外观变化或相机运动，而不是可控行为。
- **显式监督路线**：估计手部位姿、物体交互或重定向轨迹，动作接地更强；但依赖准确的位姿估计和人机对应关系，标注成本高。

世界模型与视觉预见方法则提供了第三种信号：无论行为来自人还是机器人，视频都记录了它造成的后果。已有方法常把预测未来像素、特征或轨迹作为规划输出，而 ThinkWorld 问的是另一个问题——**能否把“预测后果是否正确”变成动作推理本身的训练奖励？**

这也让它区别于两类相关工作。Reasoning-augmented VLA 往往加入文本计划、空间轨迹或连续 latent，却未必显式检验这些中间表示对应什么世界变化；Dreamer 一类方法则在想象 rollout 中优化策略。ThinkWorld 的未来预测只在训练时用于评价 CoD，不用于部署时规划，也不直接充当控制输出。

明确了监督信号之后，下一个问题是：人和机器人的身体结构差异很大，怎样让它们共享同一种动作推理语言？

## CoD：把不同本体的动作写成同一套“动态语言”

可以把 CoD 理解为动作领域的“中间草稿”：它不直接描述关节该转到多少度，而是先描述末端执行器如何运动、世界将如何随之变化。严格来说，CoD 是一串离散运动 Token，表示连接当前状态与未来状态的中间动态。

给定语言指令 $l$、当前观测 $o_t$、历史图像 $\mathbf{h}_t$，以及机器人特有的本体状态 $s_t$，模型最终仍要输出连续动作序列 $\mathbf{a}_t$。不同之处在于，VLM 内部先形成 CoD 表示 $\mathbf{c}_t$，再利用这些由动态学习塑造的表示帮助控制。

### 用动作标注初始化共享词表

对于机器人动作片段或带标注的人手腕运动片段 $\mathbf{m}_t$，作者先统一为归一化的末端执行器增量表示，再使用 FAST 风格的 motion tokenizer $\mathcal{T}(\cdot)$ 离散化：

$$
\mathbf{c}_t^\star = \mathcal{T}(\mathbf{m}_t)
$$

其中 $\mathbf{c}_t^\star$ 是监督目标。VLM 根据指令、历史和当前观测，自回归生成 $\mathbf{c}_t$。这里的共享并不意味着人手与机器人具有相同运动学结构，而是让两者的一秒运动片段进入同一个离散词表。

附录给出了更具体的定义：每个时间步，单手是 6D 向量，双手是 12D 向量，包含平移与旋转向量，不含夹爪状态；平移和旋转分别用 0.3448 m、0.9119 rad 归一化，词表大小为 2,048。机器人位姿位于机器人基座坐标系，人手势位于以操作者为中心的坐标系，其中 $+z$ 与重力方向对齐，$+x$ 随操作者当前 yaw 朝向变化。

**笔者认为，这一步的价值不在于找到一种“人手等于机械臂”的伪对应，而在于先给推理空间一个物理锚点。**后续即使视频没有动作标签，模型也不是从完全自由的 latent 开始猜测动作。

## 世界想象：用真实未来检验动作草稿

有了动作语言，还不能保证每个 CoD 真理解其后果。ThinkWorld 因此加入一个训练专用的“内部世界模型”：根据当前场景与 CoD 想象一秒后的视觉特征，再拿真实未来进行核对。

![ThinkWorld 架构与两阶段训练流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/thinkworld_main_arch_full_blue.png)

> 图解：阶段（a）的 SFT 同时学习两件事：将机器人动作和已标注人手腕运动转成共享 CoD，以及根据 CoD 预测未来视觉特征。阶段（b）的 RL 对同一上下文采样多个 CoD，用想象未来与观测未来的对齐程度打分，使无动作标注的人类和机器人视频也能优化动态推理。连续控制仍由 Action Expert 完成。

### 为什么预测 DINO 特征，而不是生成 RGB 视频

作者在 CoD 后追加可学习的 future queries $\mathbf{q}_{t+\Delta}$。VLM 对视觉上下文、teacher-forced CoD 和这些 queries 编码，再由轻量解码器 $\mathcal{D}_{\mathrm{future}}$ 输出未来表示：

$$
\begin{aligned}
\tilde{\mathbf{q}}_{t+\Delta}
&= \operatorname{VLM}(l,\mathbf{h}_t,o_t,\mathbf{c}_t^\star,\mathbf{q}_{t+\Delta}),\\
\mathbf{z}_{t+\Delta}
&= \mathcal{D}_{\mathrm{future}}(\tilde{\mathbf{q}}_{t+\Delta})
\end{aligned}
$$

真实目标来自冻结的 DINOv3 编码器：

$$
\mathbf{z}_{t+\Delta}^\star
= \mathcal{E}_{\mathrm{future}}(o_{t+\Delta})
$$

未来帧只用于构造监督目标，不会作为 VLM 输入。选择 DINO 特征而非像素，是为了保留密集的空间与物体结构，同时减少对纹理、光照等外观细节的要求；模型需要答对“场景结构怎样变”，无需生成照片级画面。

实现上，DINOv3 ViT-H/16+ 接收 $224\times224$ 的主相机图像，产生 $14\times14$ 个、每个 1,280 维的 patch 特征。64 个 future-query Token 的 2,048 维隐藏状态，经两层 MLP 与六层 cross-attention decoder，变成 196 个预测 patch。目标时间 $\Delta=1\,\mathrm{s}$；若数据中最接近的帧偏离目标超过 50 ms，该样本不计入 future loss。

### SFT：同时学习动作与未来

监督微调目标由 CoD 的 cross-entropy 和未来特征距离组成：

$$
\mathcal{L}_{\mathrm{SFT}}
= \mathcal{L}_{\mathrm{CoD}}
+ \lambda_{\mathrm{future}}\mathcal{L}_{\mathrm{future}}
$$

其中：

$$
\mathcal{L}_{\mathrm{CoD}}
= \operatorname{CE}\!\left(p_\theta(\mathbf{c}_t\mid l,\mathbf{h}_t,o_t),\mathbf{c}_t^\star\right)
$$

未来损失对有效 patch 集合 $\mathcal{V}$ 同时使用余弦距离与 Smooth L1：

$$
\operatorname{dist}(\mathbf{z},\mathbf{z}^\star)
= \frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}
\left[1-\cos(\mathbf{z}_p,\mathbf{z}_p^\star)
+\lambda_{\mathrm{SL1}}\operatorname{SmoothL1}_{\beta}(\mathbf{z}_p,\mathbf{z}_p^\star)\right]
$$

两张特征图先做 $\ell_2$ 归一化，附录设置为 $\beta=0.01$、$\lambda_{\mathrm{SL1}}=0.25\times21.792413$。训练配置中的总目标等价于 CoD/VQA CE 加 $3\mathcal{L}_{\mathrm{future}}$。

一个容易忽略的细节是：**Action Expert 并不直接读取生成的 CoD Token 或 future queries。**它接收输入 Token 在各 Qwen 层产生的 key/value 特征，CoD 与未来预测通过改变这些 VLM 表示来间接改善控制。因此，CoD 是训练中的推理接口，不是部署时额外串接的一段动作指令。

## World-Imagination RL：没有动作标签，也能给推理打分

仅依靠 teacher-forced CoD，模型学会的是“复现给定动作并预测后果”，未必会主动寻找从当前状态到目标未来的中间动态。强化学习阶段改为从同一上下文采样多个候选 CoD，比较哪一个所预测的未来更接近真实结果。

### 奖励既看全局，也看真正运动的区域

每个候选 CoD 经冻结的 future projector 得到 $\mathbf{z}_{t+\Delta}$，基础奖励为：

$$
r_{\mathrm{world}}
= \operatorname{sim}(\mathbf{z}_{t+\Delta},\mathbf{z}_{t+\Delta}^\star)
$$

若直接平均整图相似度，大面积静止背景会淹没真正重要的运动。作者因此将全局平均与运动加权平均各占一半。令 $s_p=\cos(\mathbf{z}_p,\mathbf{z}_p^\star)$，则：

$$
\operatorname{sim}(\mathbf{z},\mathbf{z}^\star)
= \frac{1}{2}\left[
\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}s_p
+ \frac{\sum_{p\in\mathcal{V}}w_ps_p}{\sum_{p\in\mathcal{V}}w_p}
\right]
$$

权重 $w_p$ 只由当前帧和真实未来帧的 DINO 特征变化决定：不高于变化中位数的 patch 权重为 0.2，到第 90 百分位线性增至 1，超出范围则截断；近乎静止的样本改用两项相同的全局平均。这样既不放弃全局一致性，又避免背景主导奖励，而且全程不需要动作标注。

### 用 GRPO 更新 VLM，用回放守住动作锚点

每个上下文采样 $K=4$ 个 CoD，组内奖励标准化后再在 rollout batch 内标准化，候选的优势值共享给它的所有 Token。优化目标为：

$$
\mathcal{L}_{\mathrm{RL}}
= \lambda_{\mathrm{GRPO}}\mathcal{L}_{\mathrm{GRPO}}(r_{\mathrm{world}})
+ \lambda_{\mathrm{replay}}\mathcal{L}_{\mathrm{replay}}
$$

GRPO 的 Token 级 importance ratio 截断在 $[0.8,1.28]$，KL 系数为 0.04；$\lambda_{\mathrm{GRPO}}=0.999$，$\lambda_{\mathrm{replay}}=0.001$。格式错误的 CoD 奖励设为组内最低候选相似度再减 0.05。Replay 是另一批标准 CoD 上的 teacher-forced CE，用于避免 RL 破坏 SFT 已建立的运动接地。

RL 只更新 VLM，Action Expert 与 future projector 均冻结，也不优化 $\mathcal{L}_{\mathrm{future}}$。这意味着被动视频可以参与推理学习，却不需要机器人在线执行候选动作。部署时连 CoD 生成和未来预测也一并移除，保留下的是已经学到动作中心表示的 VLM 与控制器。

## 实验一：真机分布外泛化，差距在环境变化后拉开

方法是否有效，首先要看它面对训练时没见过的物体、颜色与轨迹能否仍然成功。真机实验使用双臂 YAM 系统，包含一个顶部全局相机和两个腕部 RealSense D405 相机，策略只使用 RGB。

三个任务分别为颜色匹配、挂杯子和物体分类。每项任务先收集多种物体配置下的示教，再在 ID（in-distribution，分布内）与 OOD（out-of-distribution，分布外）条件下测试，每个设置进行 5 次试验。OOD 会改变颜色、物体类别、环境，部分条件还要求训练轨迹中未出现的运动路径。

![三项真机任务的 ID 与 OOD 成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/thinkworld_real_sr.png)

> 图解：三组柱形图分别对应颜色匹配、挂杯子和分类任务，并比较 ID 与 OOD 成功率。下方示意新颜色、新物体、新环境或新分类轨迹。各方法在 ID 下差距相对有限，进入 OOD 后基线明显下滑，而 ThinkWorld 在三项任务上均保持最高成功率。

下表保留完整真机结果。颜色匹配有 6 个 ID 和 6 个 OOD 设置，其余任务各有 3 个 ID 和 3 个 OOD 设置；Macro SR 是六个任务—条件成功率的等权平均，Micro SR 汇总全部 120 次试验。

| 模型 | 颜色匹配 ID | 颜色匹配 OOD | 挂杯子 ID | 挂杯子 OOD | 分类 ID | 分类 OOD | Macro SR | Micro SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Diffusion Policy | 18/30（60.0%） | 11/30（36.7%） | 9/15（60.0%） | 1/15（6.7%） | 4/15（26.7%） | 1/15（6.7%） | 32.8% | 36.7% |
| MolmoAct2 | 24/30（80.0%） | 8/30（26.7%） | 11/15（73.3%） | 14/15（93.3%） | 12/15（80.0%） | 3/15（20.0%） | 62.2% | 60.0% |
| LaWAM | 16/30（53.3%） | 8/30（26.7%） | 9/15（60.0%） | 12/15（80.0%） | 4/15（26.7%） | 4/15（26.7%） | 45.6% | 44.2% |
| **ThinkWorld** | **30/30（100.0%）** | **24/30（80.0%）** | **15/15（100.0%）** | **15/15（100.0%）** | **15/15（100.0%）** | **13/15（86.7%）** | **94.4%** | **93.3%** |

ThinkWorld 在 60 次 ID 试验中全部成功，60 次 OOD 试验中成功 52 次。相比 MolmoAct2，Micro SR 提高 33.3 个百分点。这个结果支持论文的核心判断：模型学到的是与任务相关的动态，而不是只记住外观或某条固定轨迹；不过它仍是单一训练 seed 下的物理试验，不能据此推断跨训练种子的方差。

## 实验二：只教机器人原子技能，让人类视频负责“排列组合”

分布外泛化考察的是“换个场景还能不能做”，技能组合则进一步考察“没教过完整流程，能不能照人类示范组织动作”。

烤箱任务中，机器人数据只包含四个独立技能：打开（open）、关闭（close）、把左侧物体放入（put-left）、把右侧物体放入（put-right）。人类视频则展示完整序列。模型必须把人类示范中的组成关系迁移到机器人，同时保持顺序并完成技能间切换。

![从人类示范组合机器人的原子技能](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/skill_composition.png)

> 图解：顶部是单独示教的机器人原子技能，中部的人类视频给出完整任务顺序，例如 open → put-left → close；底部是 ThinkWorld 在一次 rollout 中执行完整组合。机器人训练数据从未出现这条完整轨迹。

附录还报告了抽屉上的定性组合，包括上层抽屉的 open → put-left → close、open → put-right → close，以及下层抽屉的 open → put-left → close。实验使用 100 条机器人原子动作 episode 与 90 条人类完整任务视频共同训练；人类样本只优化 VLM 的 future-prediction loss，不计算 Action DiT loss。

这里的意义不是让机器人直接复刻人的关节轨迹，而是让共享 CoD 空间承载“先做什么、再做什么”的动态知识，再由机器人自己的连续动作专家执行。

## 实验三：LIBERO 高分之外，更要看扰动后的 LIBERO-PRO

真机结果之后，作者用可控仿真扰动拆分泛化来源。ThinkWorld-LIBERO 只在标准 LIBERO 的 Object、Goal、Spatial、LIBERO-10 训练集上微调，之后直接测试 LIBERO-PRO，不接触其物体属性、位置、指令语义、任务目标与环境扰动。

下表数值均为成功率（%）。LIBERO-PRO 每个轴向分数是 Goal、Spatial、LIBERO-10、Object 四套任务的算术平均；VLA-JEPA 的 48.5 为作者复现结果，其余基线来自官方仓库。Params. 列沿用原文参数量数值。

| 模型 | Params. | LIBERO Spatial | Object | Goal | Long | LIBERO Avg. | PRO Object | Position | Semantic | Task | Environment | PRO Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA | 7.5 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 93.0 | 0.0 | 97.3 | 0.0 | 68.0 | 51.7 |
| $\pi_0$ | 3.2 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | 90.5 | 0.0 | 90.5 | 0.0 | 38.8 | 44.0 |
| $\pi_{0.5}$ | 3.4 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | 96.0 | 20.8 | 95.8 | 0.8 | 52.8 | 53.2 |
| MolmoAct | 8.1 | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 | 76.0 | 1.5 | 85.8 | 1.5 | -- | 41.2 |
| VLA-JEPA | 2.8 | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 | 80.4 | 3.5 | 97.2 | 20.8 | 41.0 | 48.5 |
| **ThinkWorld** | **2.9** | 98.2 | 98.8 | 97.8 | 94.2 | **97.3** | 73.3 | 18.2 | 93.9 | **31.9** | 60.9 | **55.6** |

ThinkWorld 在 LIBERO 上取得 97.3%，在 LIBERO-PRO 上取得 55.6%，两者均为表中最高平均值。尤其在 Task 与 Environment 扰动下分别达到 31.9% 和 60.9%，比同样含 latent world modeling 的 VLA-JEPA 高 11.1 和 19.9 个百分点。

但 Position 与 Object 并非全面领先：OpenVLA 的 Position 为 97.3%，$\pi_{0.5}$ 的 PRO Object 为 96.0%，都明显高于 ThinkWorld。换句话说，证据支持的是**整体扰动泛化更强**，而不是每个维度都占优。

## 分析一：CoD 真的跨本体组织成了“运动空间”吗

如果 CoD 只是记住数据来源，人类与机器人的共享词表就没有太大意义。作者提取每个样本最后一个 `</think>` Token 的 VLM hidden state，用它概括生成的 CoD，再做 t-SNE 可视化。颜色表示四种二维末端运动方向，点的形状表示数据集。

![人类与机器人 CoD 特征的 t-SNE 可视化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/cod_tsne.png)

> 图解：每个点是一个样本，颜色编码运动方向，标记形状编码数据来源。相似运动方向形成跨数据集聚类，而同一来源的数据会按运动方向分开，说明特征更偏向动作结构，而非“这是人还是机器人”的身份信息。

这是一种描述性表示分析，不能单独证明跨本体迁移的因果关系；但它与后面的技能组合结果相互印证，说明共享空间确实围绕运动而非外观组织。

## 分析二：人类视频的收益不只是“数据变多了”

作者固定机器人数据量，将人类第一视角视频占总训练数据的比例提高到 10%、25%、50%。除 robot-only 基线外，还设置一个额外 robot-only 对照，用更多机器人数据匹配 50% 人类数据版本的训练规模和迭代数。各版本经过 SFT 后都在 LIBERO 上做本体后训练，再测试 LIBERO-PRO。

![人类视频比例对 LIBERO-PRO 成功率的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/scaling_graph.png)

> 图解：横轴区分机器人数据与人类第一视角视频的数据配置，纵轴是 LIBERO-PRO 成功率（%）。随着人类视频比例增加，整体成功率单调上升；robot-only 基线为 45.4%，50% 人类数据版本为 47.8%，并比训练规模匹配的 robot-only 对照高约 1.2 个百分点。

增益幅度并不巨大，且这些配置是用于分析数据贡献的实验变体，不等同于最终模型配置。但匹配规模的对照排除了“只是多训练了一些”的简单解释，支持异质人类经验本身带来泛化收益。

## 分析三：未来预测究竟有没有听懂 CoD

接下来需要验证，CoD 对未来预测的贡献是否集中在真正发生运动的区域。作者比较正常 CoD、打乱 CoD、空 CoD 和关闭 CoD 时的未来 DINO 特征。

![不同 CoD 条件下的未来特征 PCA 可视化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/thinkworld_fig_pca_main_tight.png)

> 图解：各 PCA 面板用相同颜色映射展示预测或真实 DINO 特征的空间结构。正常生成的 CoD 更接近未来真实状态的结构；打乱、置空或移除 CoD 后，预测更多保留当前状态，或转向不正确的变化，说明 CoD 提供了动作导致的状态转移信息。

定量结果进一步把全部 patch 与变化最大的前 25% patch 分开。$\Delta$ 均相对 CoD on 计算，余弦相似度越高越好。

| 条件 | 全部 patch Cos | $\Delta$ | 前 25% 动态 patch Cos | $\Delta$ |
|---|---:|---:|---:|---:|
| Shuffle | 0.71 | -0.024 | 0.55 | -0.071 |
| Empty | 0.72 | -0.020 | 0.54 | -0.079 |
| CoD off | 0.72 | -0.018 | 0.55 | -0.071 |
| **CoD on** | **0.74** | — | **0.62** | — |

全部 patch 上只有约 0.02—0.03 的差距，动态区域则扩大到约 0.07—0.08。这正是 motion-enhanced reward 想要强调的现象：背景让全局指标看起来都不错，真正的差别发生在动作影响的局部。

注意力图提供了另一条观察视角。

![CoD Token 对当前观测的注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/figures/images/attn_main_2col.png)

> 图解：热图表示 CoD Token 对输入观测各区域的关注程度。无论人类还是机器人样本，注意力都更多落在手、机械臂和被操作物体等动态区域，而非静态背景。附录还显示 future-prediction Token 的注意力分布更广，符合 CoD 聚焦局部动作、未来预测整合全场景的分工。

需要注意，PCA 与注意力都来自 Qwen 第 24—27 层，属于描述性分析，作者并未把它们当作因果证据。

## 消融：性能不是某一个模块单独带来的

从 Qwen3-VL 加 MolmoAct2 Action Expert 的基线出发，作者依次累积历史帧、未来预测、CoD 与 world-imagination RL。

| 累积模型变体 | Object | Position | Semantic | Task | Environment | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| Qwen baseline | 59.5 | 1.9 | 84.4 | 17.2 | 37.8 | 40.1 |
| + History frames | 61.3 | 4.9 | 84.6 | 22.8 | 29.8 | 40.7 |
| + Future prediction | 70.0 | 11.9 | 89.4 | 26.3 | 54.4 | 50.4 |
| + CoD | 74.5 | 16.9 | 94.4 | 29.7 | 56.4 | 54.4 |
| + RL | **73.3** | **18.2** | 93.9 | **31.9** | **60.9** | **55.6** |

历史帧只将平均值从 40.1% 提到 40.7%；未来预测带来最大单次跃升，达到 50.4%；CoD 再提高到 54.4%；RL 最终达到 55.6%，主要改善 Position、Task 和 Environment。Object、Semantic 在最后一步略有回落，因此更准确的结论是各组件整体互补，而非每加入一个模块就在所有维度单调增长。

## 复现细节：模型、数据与训练流程

前面的结果说明了“是否有效”，附录则回答“怎样把它训出来”。

### 模型与连续动作输出

- **VLM**：Qwen3-VL-2B-Instruct；SFT 与 RL 使用 BF16。
- **Action Expert**：从 MolmoAct2 初始化，包含 36 个 non-causal Transformer block，隐藏宽度 768，以 FP32 存储和优化。
- **动作张量**：最大为 $30\times32$，二维 validity mask 保留各数据集原生 horizon 与动作通道。
- **Flow Matching**：对干净动作 $a$ 和高斯噪声 $\epsilon$，采样 $z\sim\operatorname{Beta}(1,1.5)$，令 $t=0.001+0.999z$，构造 $x_t=(1-t)\epsilon+ta$，回归 $a-\epsilon$；每条样本使用 8 个独立加噪 flow 样本，推理采用 10 步 Euler 积分。

输入顺序为主视角历史、当前主视角和其余当前视角。历史最多 16 帧，以 2 Hz 采样，从旧到新排列，不重复当前帧。本体状态按训练集统计归一化、截断到 $[-1,1]$，量化为 256 个离散 bin 后序列化；SFT 时以 0.10 概率丢弃状态条件，以 0.20 概率丢弃腕部视角。

图像保持原始宽高比，当前图像像素面积限制为 4,096—200,704，历史帧为 4,096—50,176，多模态序列最长 2,048 Token，超长样本直接移除。训练增强包括固定 scale 为 0.95 的 RandomResizedCrop、±5° 旋转、颜色抖动，以及概率 0.2 的高斯模糊；同一样本各视角与时间步共享一次增强参数。RL 关闭增强，保证四个候选接收完全相同的媒体输入。

### 训练数据：4,276.1 小时的人机混合经验

| 来源 | Episodes | Frames | Hours | Hz | 视角 | 原生动作表示 |
|---|---:|---:|---:|---:|---|---|
| YAM | 32.2K | 76.0M | 704.1 | 30 | 顶部 + 双腕 | $30\times14$D 绝对关节 |
| DROID | 74.6K | 17.8M | 328.9 | 15 | 外部 + 腕部 | $15\times8$D 绝对关节 |
| RoboCOIN | 112.0K | 82.3M | 755.2 | 30/50 | 单视角或三视角 | $30/50\times(12/14)$D 绝对 EEF |
| Bridge | 53.2K | 1.9M | 105.2 | 5 | 主视角 | $5\times7$D delta EEF |
| RT-1 | 87.2K | 3.8M | 350.6 | 3 | 主视角 | $3\times7$D delta EEF |
| BC-Z | 39.4K | 5.5M | 152.0 | 10 | 主视角 | $10\times7$D delta EEF |
| EgoVerse | 55.6K | 124.8M | 1,155.4 | 30 | 第一视角 | 一秒 6D/12D 运动 Token |
| EgoDex | 314.8K | 78.3M | 724.8 | 30 | 第一视角 | 一秒 6D/12D 运动 Token |
| **合计** | **769.1K** | **390.3M** | **4,276.1** | — | 混合 | 数据集原生格式 |

这些数字是 episode-disjoint 验证集划分前的源数据规模，论文没有公开采样权重。Action SFT 排除 RoboCOIN 的 50 Hz 轨迹，仅保留 30 Hz。另有 Pixmo-Cap、Molmo2-ER、Molmo2 Cap-LLaVA、Molmo2 Video-Track、RoboPoint、RoboVQA 等辅助视觉语言数据，用于降低灾难性遗忘；它们只提供语言目标，不贡献连续动作或未来特征损失。

### 三阶段训练配置

| 设置 | VLM SFT | Action SFT | RL |
|---|---|---|---|
| 初始化 | Qwen3-VL-2B | VLM SFT | Action SFT |
| 可训练模块 | Language model + future projector | Action Expert | Language model |
| 目标 | CoD/VQA CE + $3\mathcal{L}_{\mathrm{future}}$ | Masked flow matching | GRPO/GDPO + 弱 SFT replay |
| 数据 | 机器人 + 人类运动，辅助 VQA | 机器人轨迹 | 机器人 + 人类运动 |
| 迭代数 | 150K | 50K（5K 当前视角 + 45K 历史） | 2K |
| 学习率 | VLM $10^{-5}$；projector $10^{-4}$ | $10^{-4}\rightarrow5\times10^{-5}$ | $10^{-6}$ |
| 学习率计划 | 5% warmup，之后恒定 | Warmup + cosine cooldown | 50 步 warmup，之后恒定 |
| 全局 Batch | 128 | 128 | 128 |
| Seed | 42 | 43 / 44 | 45 |
| 代表资源 | 128 张 H100 | 先 64、后 128 张 H100 | 64 张 H100 |

所有阶段使用 fused AdamW，$(\beta_1,\beta_2)=(0.9,0.95)$，$\epsilon=10^{-8}$，weight decay 为 0.1，梯度范数截断为 1.0，采用 DeepSpeed ZeRO stage 1 且不 offload optimizer。下游 LIBERO 微调为 50K 步 masked flow matching；真机任务同样微调 50K 步，对 VLM 使用 LoRA，并全量更新 Action DiT decoder。

这套资源规模也说明，ThinkWorld 的“可利用无标签视频”降低的是**动作标注成本**，并不等同于低算力训练。

## 补充评估：零样本、感知扰动与推理成本

### SimplerEnv 与闭环未来预测

模型没有针对 SimplerEnv 做额外本体微调，但 SFT 与 RL 已见过 Bridge、RT-1，因此这里的 zero-shot 指“无 benchmark 专属适配”，不是从未接触相关机器人设置。

| 分组 | 任务 | 成功率 |
|---|---|---:|
| Google-VM | — | 49.31% |
| Google-VA | Pick Coke can | 56.00% |
| Google-VA | Drawer | 33.33% |
| Google-VA | Move-near | 55.83% |
| Google-VA | Put-in-drawer | 2.12% |
| **Google-VA** | **Micro avg.** | **46.54%** |
| WidowX VM | — | 41.67% |
| **Overall** | **Micro** | **46.51%** |

![SimplerEnv 闭环执行中的一秒未来预测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ThinkWorld-World-Imagination-Reinforces-Chain-of-Dynamics-Reasoning/suppl_figures/images/simplerenv_closedloop_qual.png)

> 图解：每列将前一重规划时刻预测的一秒后 DINO 特征 PCA 图，与目标时刻实际到达的观测配对。上排是真实后续观测，下排是对应预测；空间布局与任务进展大体一致，说明策略在自己产生的闭环分布偏移下，未来表示仍保留了有用结构。初始帧没有前置预测。

### LIBERO-Plus：平均最高，但相机和噪声仍是短板

LIBERO-Plus 进一步覆盖相机、机器人初始状态、语言、光照、背景、传感器噪声与物体布局七类扰动，同样使用只在标准 LIBERO 上微调的模型直接评估。

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| $\pi_0$-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| DreamVLA | 65.0 | 40.8 | 63.5 | 85.7 | 82.6 | 84.9 | 74.0 | 69.9 |
| **ThinkWorld** | 46.9 | **70.5** | 73.5 | **90.7** | 90.0 | 62.7 | **79.6** | **71.7** |

平均成功率 71.7%，比 DreamVLA 高 1.8 个百分点；机器人初始状态扰动为 70.5%，下一名为 40.8%，物体布局为 79.6%，下一名为 74.2%。但 Camera 仅 46.9%、Noise 仅 62.7%，均不是最佳，说明动态推理并不能自动解决所有感知鲁棒性问题。

### 部署时没有额外推理链开销

延迟测试在同一张 NVIDIA H100 80GB 上进行，batch 为 1，关闭 CUDA graph，以 eager mode 预热 5 次后连续计时 50 次。各模型使用自己的原生部署输入，因此这是实用策略延迟对比，不是完全相同输入下的架构消融。

| Policy | p50（ms） | p95（ms） | 50 次总 Wall（s） | 峰值显存（GiB） |
|---|---:|---:|---:|---:|
| MolmoAct2 | 335.07 | 467.96 | 18.44 | 12.11 |
| MolmoAct2-Think | 879.37 | 1014.47 | 46.76 | 12.12 |
| **ThinkWorld** | **324.84** | **333.09** | **16.31** | **7.07** |

ThinkWorld 的 p50 与 p95 在三者中最低，峰值显存也最小。关键原因是 CoD 和未来预测只参与训练，部署策略不执行这段推理过程。这提供了一种很实用的分工：**训练时让模型“多想一步”，运行时不必为这段思考付出逐次延迟。**

## 总结：把动作监督扩展成“后果监督”

- **共享动作接口**：CoD 将人手腕运动与机器人末端运动离散到同一空间，为跨本体推理提供物理锚点。
- **未来作为奖励**：想象未来与真实未来的 DINO 特征对齐，使无动作标签视频也能直接优化动作推理。
- **关注真实变化**：全局与运动加权相似度各占一半，减少静止背景对奖励的稀释。
- **泛化得到多维验证**：真机 Micro SR 为 93.3%，LIBERO-PRO 平均为 55.6%，并展示人类视频指导的原子技能组合。
- **训练与部署解耦**：Action Expert 间接利用被 CoD 塑造的 VLM 表示，部署不生成 CoD 或未来，因此保留较低延迟。

当前结果仍受单训练 seed、特定硬件与输入配置限制，且在相机、噪声等感知扰动上并不领先；下一步值得研究的是，如何把后果监督与更强的感知鲁棒性、跨训练种子验证及更低成本训练结合起来。

> 本文参考自 [ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning](http://arxiv.org/abs/2610.37666v1)