# H-JEPA：分层世界模型，让视觉规划「先谋后动」

让机器人看着画面规划一个长程任务——比如绕过迷宫走到远处、抓起方块放到指定位置——现有的 JEPA 世界模型只在 **一个潜空间、一个时间尺度** 上预测和搜索，误差随 rollout 步数累积，动作搜索空间也爆炸。这篇来自 NYU、AMI（Yann LeCun 团队）和 Brown 的工作提出 **H-JEPA**：把多个 action-conditioned JEPA 叠成层级，每层在 **自己学到的潜空间** 里以更长的时间跨度做预测，规划时自上而下逐层细化子目标。最硬的结果：在 Visual AntMaze 上，三层层级把规划成功率从 18% 提升到 73%，**同时规划算力更低**；在真实机器人视频数据集 DROID 上，两层 H-JEPA 的离线规划保真度比共享潜空间的层级基线 HWM 再高出 5 个百分点。

## 为什么需要分层：单一潜空间的两个天花板

先交代背景。JEPA（Joint-Embedding Predictive Architecture）类世界模型的思路是：不重建像素、不依赖奖励，直接在 **潜空间** 里预测未来状态，然后用梯度下降在这个空间里搜索动作序列。DINO-WM、V-JEPA 2、PLDM、LeWM 都是这条路线，但它们全部在单一潜空间、单一时间尺度上工作。

作者指出这有两个结构性缺陷：

- **时间尺度单一**：长程任务意味着要 rollout 很多细粒度步数，预测误差逐步累积，候选动作的搜索空间也随之膨胀；
- **表示职责冲突**：同一个潜空间既要支撑底层动力学预测，又要充当「离目标还有多远」的代价度量。一个保留所有快变细节（比如机器人的关节姿态）的潜向量可能把动力学建模得很好，但当目标是抽象的（「到达某个位置」而非「摆出某个姿势」）时，它给出的 goal-matching 代价反而很差。

人类大脑恰好不是这样组织的：皮层区域构成内在时间尺度的层级，越高层的表征变化越慢。H-JEPA 把这个直觉工程化——**高层走大步、管战略，低层走小步、管执行**。

## H-JEPA 方法：一叠各自为战的 JEPA

### 层级构建：stride 与 window 两个旋钮

第一层（Level 1）直接编码观测：$z^{(1)}_t = E^{(1)}(o_t)$，动作块编码为 $a^{(1)}_t = A^{(1)}(a_t)$。往上的每一层 $\ell > 1$ 由两个时间超参数定义：

- **stride $s_\ell$**：下采样倍率，上层时间 $t$ 对应下层时间 $t \cdot s_\ell$；
- **window $w_\ell$**：每个上层状态汇总的下层步数。

第 $\ell$ 层的状态与动作嵌入为：

$$
z^{(\ell)}_t = E^{(\ell)}\left(z^{(\ell-1)}_{t \cdot s_\ell \, : \, t \cdot s_\ell + w_\ell}\right), \qquad
a^{(\ell)}_t = A^{(\ell)}\left(a^{(\ell-1)}_{t \cdot s_\ell + w_\ell - 1 \, : \, (t+1) \cdot s_\ell + w_\ell - 1}\right)
$$

一个值得注意的设计：状态编码器池化 $w_\ell$ 步，而动作编码器聚合的是 $s_\ell$ 步、与 $w_\ell$ 无关——这样 $a^{(\ell)}_t$ 始终覆盖从 $z^{(\ell)}_t$ 到下一个上层状态的 **完整转移**。实验中所有模型取 $w_\ell = 1$、$s_\ell = 2$，即上层状态编码器是逐点的（pointwise），每往上一层，单步预测跨越的环境步数翻倍。

### 每层目标：teacher-forced 预测 + SIGReg 防坍塌

每一层都是一个独立的 JEPA：给定 $c_\ell$ 个历史潜状态和动作，预测器 $F^{(\ell)}$ 做 teacher-forced 单步预测：

$$
\mathcal{L}^{(\ell)}_{\mathrm{pred}} = \frac{1}{c_\ell} \sum_{\tau=1}^{c_\ell} \left\| F^{(\ell)}(z^{(\ell)}_{<\tau}, a^{(\ell)}_{<\tau}) - z^{(\ell)}_{\tau+1} \right\|_2^2
$$

再配上 LeJEPA 的 SIGReg 正则（一种 sketched normality 检验，把嵌入分布推向各向同性高斯以防坍塌），每层目标为：

$$
\mathcal{L}^{(\ell)} = \mathcal{L}^{(\ell)}_{\mathrm{pred}} + \lambda_\ell \, \mathrm{SIGReg}\left(Z^{(\ell)}\right)
$$

关键在于 **所有层端到端联合训练**：上层的预测误差梯度会穿过下层的编码器一路反传。笔者认为这正是整篇文章的「发动机」——正因为编码器和预测器联合优化，每一层才会被迫只保留 **在自己时间尺度上可预测** 的特征，把预测不了的细节主动丢掉。后面的「选择性抽象」现象全部源于此。

### 自上而下规划：子目标接力

规划时，当前观测 $o_0$ 和目标观测 $o_g$ 被逐层编码成初始状态与目标状态 $\{z^{(\ell)}_0\}$、$\{g^{(\ell)}\}$。顶层规划器优化宏动作序列，让最终预测状态逼近目标：

$$
a^{(L),*}_{0:H_L-1} = \arg\min_{a^{(L)}_{0:H_L-1}} \left\| \hat{z}^{(L)}_{H_L} - g^{(L)} \right\|_2^2
$$

随后每一层的预测轨迹变成下一层的 **子目标**。下层把 rollout 结果用上层编码器 $E^{(\ell+1)}$「抬升」到上层潜空间，与子目标比对：

$$
a^{(\ell),*}_{0:H_\ell-1} = \arg\min_{a^{(\ell)}_{0:H_\ell-1}} \left[ \left\| \tilde{z}^{(\ell+1)}_{K_{\ell+1}} - \hat{z}^{(\ell+1),*}_{K_{\ell+1}} \right\|_2^2 + \beta \sum_{i=1}^{K_{\ell+1}-1} \left\| \tilde{z}^{(\ell+1)}_i - \hat{z}^{(\ell+1),*}_i \right\|_2^2 \right]
$$

其中 $K_{\ell+1}$ 是要匹配的子目标个数：闭环控制取 $K_{\ell+1} = 1$（只追最近的下一个子目标），开环取 $K_{\ell+1} = H_{\ell+1}$。所有层的动作序列都用梯度下降（AdamW）优化，Level 1 输出原始动作执行。这个过程可以类比为「总部定战略路线 → 战区分解路段 → 士兵走每一步」，每层只在自己看得懂的空间里做力所能及的搜索。

![teaser](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/planning_illustrations/teaser.png)

> 图解：左图是 Visual AntMaze 上三层层级规划的可视化。Level 3（每步预测跨 20 个环境步）在最抽象的空间里朝终点规划，它的第一个预测状态成为 Level 2 的子目标（紫框），Level 2 再把子目标下放给 Level 1（红框），最终由 Level 1 输出原始动作。注意解码出来的图像：高层潜状态保留了蚂蚁的位置和迷宫布局，但腿部姿态细节已经丢失——这正是「选择性抽象」。右图是成功率 vs 每 episode 规划算力（TFLOPs，对数轴）的 Pareto 前沿：三层 H-JEPA（紫）在约 1/10 的算力下达到 70%+ 成功率，而单层 LeWM（灰）即使用 1000 TFLOPs 也只有 30% 左右。

## 高层学到了什么：选择性抽象

方法讲完，第一个要回答的问题是：高层潜空间到底留下了什么、扔掉了什么？

作者在四个环境上训练了最多四层的 H-JEPA：导航类的 FourRoomDistractors（四房间 + 一个随机瞬移的干扰点）和 Visual AntMaze（四足机器人走迷宫），操作类的 Push-T（推 T 形块）和 OGBench Cube（抓放方块）。用冻结表征上训练的 MLP probe 测量各实体的可恢复性（NMSE，越接近 0 表示信息保留越好）。

![depth probes](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/abstraction/depth_probes.png)

> 图解：左两图是四层 H-JEPA 各层的 probe NMSE（三次种子均值 ± SE）。AntMaze 中身体状态（body，蓝线）的 NMSE 从 Level 1 的约 0.06 一路涨到 Level 4 的约 0.81——信息被逐级丢弃；而全局位置（position，红线）始终贴在 0 附近——被完整保留。FourRoom 中干扰点位置（distractor）同样逐级丢失，ego 智能体位置保留。右侧是把各层潜状态解码回像素的效果：Level 3 里蚂蚁的腿姿已经模糊难辨，FourRoom 的蓝色干扰点几乎消失，而红色的 ego 依然清晰。

为什么丢的恰好是这些特征？因为它们在高层的预测视野内 **不可预测**：FourRoom 的干扰点平均 35 步随机瞬移一次，视野一长就必然跨过瞬移点；蚂蚁的关节振荡频率比整体位移快得多（频谱质心差距 8.3 倍），长 horizon 下同样无法预测。JEPA 的联合优化让每层只保留自己能预测的东西，这是一种「用进废退」。

但选择性抽象并非处处发生。作者定义了 **实体频率差（entity frequency gap）**：数据集中最快与最慢状态分量的特征频率（频谱质心）之比，并发现它与抽象强度强相关。

![freq scatter](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/freq_abstraction_scatter.png)

> 图解：横轴是每个数据集的快/慢实体频率差（对数刻度），纵轴是 Level 1 → Level 2 的 probe NMSE 变化量（正值 = Level 2 丢失信息）。频率差大的环境（Humanoid 12.7×、AntMaze 8.3×、FourRoom 3.1×）中，快实体（实心蓝点：身体状态、干扰点）在 Level 2 被显著丢弃，而慢实体（空心灰点：位置）的 NMSE 几乎不变。频率差小于 2× 的环境（Push-T 1.2×、DROID 1.4×、Cube 1.6×，阴影区）则几乎不发生选择性抽象——所有实体都被保留。

附录里作者还给出了一条 AntMaze 单条轨迹的直观证据和六个环境的逐实体频谱 + probe 对照：

![entity traces](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/ant/entity_traces.png)

> 图解：一条代表性 AntMaze 训练 episode（81 个 Level-1 帧）。左图红曲线是归一化全局位置，蓝色簇是 13 维身体状态；位置的逆频谱质心约 35 帧，身体状态约 4 帧（8.3× 差距）。右图是各维度累积绝对变化量，身体状态的累积变化约为位置的十倍——肉眼即可验证「快/慢分离」。

![freq bars](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/freq_to_abstraction_no_robocasa.png)

> 图解：六个环境逐实体的频谱质心（上，对数轴）与 Level-1/Level-2 成对 probe NMSE（下）。左组 FourRoom、AntMaze、Humanoid 频率差 3.1–12.7×，快实体在 Level 2 的解码误差显著恶化而慢实体不变；右组 Push-T、Cube、DROID 频率差仅 1.2–1.6×，各实体在两个层级上可恢复性相当。这组图把「频率分离 → 选择性抽象」的相关性坐实到了每个实体上。

## 规划实验：更少算力，更高成功率

抽象层级学出来了，它能不能真正转化为规划能力？作者做了两组对比：一是固定方法、扫规划算力（改变每层梯度优化的候选轨迹条数）；二是固定算力预算（100 TFLOPs/episode）、扫层级深度，并引入关键基线 **HWM**——一个同样做时间层级分解、但所有层共享同一个潜空间的层级世界模型（实现上即把 H-JEPA 上层编码器换成恒等映射，数据与训练设置完全一致）。

![planning combined](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/planning/depth_planning_combined.png)

> 图解：上排是成功率 vs 规划算力（TFLOPs/episode，对数轴），粗线为 Pareto 前沿。FourRoom、AntMaze、Cube 上，每加一层（最多三层），前沿都向左上方移动——更高成功率、更低算力。Push-T 是例外：两层打平或略超 LeWM，三层即使算力拉满也很差，原因是 Push-T 的 episode 太短，三层模型每个 epoch 只能见到 LeWM 58% 的训练转移（四层仅 14%）。下排是成功率 vs 层级数（灰色虚线 LeWM，红 H-JEPA，蓝 HWM）。两个事实清晰可见：(1) HWM 全面超过 LeWM，说明 **时间分解本身就有价值**；(2) H-JEPA 在 AntMaze 上三层达到 73% 而 HWM 只有 39%，Cube 上四层 55% vs 39%，说明 **独立的抽象潜空间带来了时间分解之外的额外收益**。

分层规划在四个环境上的成功执行示例如下：

![eval tasks](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/planning/eval_tasks.png)

> 图解：每个环境一条成功的 H-JEPA 执行轨迹（FourRoom/AntMaze 用三层模型，Cube/Push-T 用两层），每行是均匀采样的 8 帧观测加最右的目标图像，标签为已耗环境步数。行在首次成功处截止（Cube 例外，展示完整 30 步以包含抓取抬升过程）。

## 机制拆解：抽象目标与时间管理

层级规划同时改变了两件事：搜索在时间维度被分解，候选未来在更抽象的空间里被打分。到底哪个因素贡献了多少？作者设计了一组非常干净的归因实验，这也是笔者认为全文最精彩的部分。

### 只换代价空间，不换动力学

固定用每个模型的 Level-1 世界模型做 **扁平规划**（rollout 动力学、规划器设置全都不变），只改变计算目标代价所用的潜空间：Native L1 用原始 Level-1 距离；Level-$k$ 投影则把预测和目标都经上层编码器 $E^{(2)} \circ \cdots \circ E^{(k)}$ 映射后再测距离（因为 $w_\ell = 1$，编码器逐点，单个 Level-1 潜状态可以直接投影到任意层）。

| 代价空间（AntMaze） | 2 层模型 | 3 层模型 | 4 层模型 |
| --- | --- | --- | --- |
| Native L1 | $23.3 \pm 3.3$ | $16.7 \pm 1.8$ | $10.0 \pm 0.0$ |
| L2 投影 | $\mathbf{31.3 \pm 2.7}$ | $20.7 \pm 4.8$ | $20.7 \pm 1.8$ |
| L3 投影 | — | $\mathbf{22.0 \pm 2.3}$ | $21.3 \pm 4.7$ |
| L4 投影 | — | — | $\mathbf{26.7 \pm 2.9}$ |
| 分层规划（参考） | $39.3 \pm 3.7$ | $73.3 \pm 3.5$ | $63.3 \pm 7.7$ |

> 表解：AntMaze 规划成功率（%，三种子均值 ± SE）。每一列是一个独立训练的 H-JEPA；前四行共享同一个 Level-1 世界模型和规划器，只有代价空间不同。结果：每个多层模型都至少有一个上层代价优于 Native L1，且最佳投影代价（如 4 层模型的 26.7%）超过了单独训练的 LeWM 基线（18.0%）——**不做任何时间分解，仅仅换一个更抽象的空间来度量「离目标多远」，规划就已经变好了**。但完整分层规划（末行）仍显著更高，说明两个机制是互补的。

为什么抽象空间的代价更好用？看代价几何就明白了：

![heatmaps](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/heatmaps/composite_levels.png)

> 图解：三层 H-JEPA 各层潜空间中，迷宫里每个状态到锚点（星标）的潜空间距离热力图（蓝近红远）。迷宫里有用的代价应该与「绕过墙的最短路径长度」相关。Level-1 的距离场在远离锚点处几乎是一片均匀的红色——梯度规划器在远处拿不到任何引导信号；Level 2、3 的代价沿走廊从锚点向外平滑爬升，锚点周围的低代价区域也更宽。长 horizon 预测既促成了抽象，也让高层代价对远处目标更有指导性。

这个代价消融扩展到其他环境后呈现清晰的对照：FourRoom（有选择性抽象）同样受益于投影代价；Cube 和 Push-T（无选择性抽象）则没有明确收益，Cube 上 L3/L4 投影甚至让成功率腰斩。

| 环境 | 深度 | Native L1 | L2 投影 | L3 投影 | L4 投影 | 分层 H-JEPA |
| --- | --- | --- | --- | --- | --- | --- |
| FourRoom（LeWM: 40.7） | 2 层 | $9.3 \pm 0.7$ | $\mathbf{38.7 \pm 6.6}$ | — | — | $81.3 \pm 6.7$ |
| | 3 层 | $43.3 \pm 8.7$ | $65.3 \pm 5.5$ | $\mathbf{71.3 \pm 4.1}$ | — | $96.0 \pm 1.1$ |
| | 4 层 | $68.7 \pm 4.7$ | $\mathbf{72.0 \pm 3.1}$ | $66.0 \pm 3.1$ | $70.7 \pm 2.7$ | $80.0 \pm 4.0$ |
| Cube（LeWM: 32.0） | 2 层 | $24.0 \pm 1.2$ | $13.3 \pm 0.7$ | — | — | $47.3 \pm 2.7$ |
| | 3 层 | $24.7 \pm 1.3$ | $28.0 \pm 4.2$ | $14.0 \pm 3.5$ | — | $60.0 \pm 5.0$ |
| | 4 层 | $26.7 \pm 2.7$ | $25.3 \pm 2.4$ | $12.0 \pm 2.0$ | $12.0 \pm 2.3$ | $55.3 \pm 4.4$ |
| Push-T（LeWM: 40.0） | 2 层 | $36.7 \pm 1.8$ | $38.0 \pm 3.1$ | — | — | $45.3 \pm 1.3$ |
| | 3 层 | $30.7 \pm 1.8$ | $28.0 \pm 4.0$ | $24.7 \pm 4.1$ | — | $17.3 \pm 3.5$ |
| | 4 层 | $1.3 \pm 0.7$ | $1.3 \pm 0.7$ | $0.7 \pm 0.7$ | $1.3 \pm 1.3$ | $0.7 \pm 0.7$ |

> 表解：代价消融在三个环境的扩展（成功率 %）。FourRoom 每个深度都有投影代价超过 Native L1 乃至 LeWM 基线（最深的代价并非总是最好，4 层模型的峰值在 L2 投影）；Cube 和 Push-T 上投影代价基本无益。这组阴性对照说明：抽象代价的收益与选择性抽象现象同进退，但作者也坦承这不足以证明选择性抽象本身就是收益的唯一来源。

### 时间分解：把长任务切成「单调下降」的小段

另一个机制是时间维度。当低层只追踪上层规划的前缀（$K_{\ell+1} < H_{\ell+1}$）时，每个子问题的 rollout 步数和动作搜索长度都大幅缩短，直接省算力。更微妙的是：扁平规划器的全程目标代价沿专家轨迹都可能 **停滞甚至上升**，给规划器一个不一致的进度信号；而子目标追踪代价在每段内几乎单调下降。

![tmono](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/tmono/instance.png)

> 图解：上排是「到目标的代价」随时间的变化（灰：Level-1 空间，蓝：Level-2 空间），注意灰色曲线在专家轨迹中段明显上凸——离终点越近，代价反而越高，这对梯度规划是毒药。下排是 Level-1 rollout 到各子目标的代价（绿），每个子目标段内都干净地单调下降。这是两层 H-JEPA（$s_2 = 3$）在四个环境上的代表性专家轨迹。

定量上，用代价与时间的负 Spearman 相关（越接近 1 越单调）衡量：

| 环境 | Level-1 全程目标代价 | HWM L2 代价 | HWM 子目标 | H-JEPA L2 代价 | H-JEPA 子目标 |
| --- | --- | --- | --- | --- | --- |
| Visual AntMaze | 0.48 | 0.52 | 0.94 | **0.68** | **0.94** |
| Push-T | 0.75 | 0.77 | 0.85 | **0.77** | **0.89** |
| OGBench Cube | 0.65 | 0.74 | 0.79 | **1.00** | **0.99** |
| FourRoom Distractors | 0.56 | 0.62 | 0.92 | **0.85** | **1.00** |

> 表解：子目标追踪的单调性（0.89–1.00）全面碾压全程目标代价（0.48–0.75）。HWM 与 H-JEPA 都展现出这一改善，与「时间分解是 HWM 超越 LeWM 的主要原因」相互印证；而 H-JEPA 的 L2 代价又系统性好于 HWM 的 L2 代价，再次指向表示层级的独立贡献。

## 走向真实机器人：DROID 与逆动力学

到此为止的实验都在固定相机、固定背景的仿真里。一旦走向真实世界，DROID 数据集的每个 episode 场景、光照、物体全变，一个新的失败模式出现了：**慢特征坍塌（slow-feature collapse）**——背景成了信号中最可预测的部分，纯预测目标会驱使编码器记住场景身份、丢掉运动中的机械臂，规划保真度直接归零。

![scene diversity](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/scene_diversity_cube_droid.png)

> 图解：三个 Cube 与三个 DROID clip 的首帧对比。Cube 场景几乎不变，DROID 每个 episode 都是全新厨房/桌面。作者实测：DROID 潜空间方差的 75% 分布在 episode 之间，而 Cube 只有 28%——场景多样性把编码器容量全部吸向了「记住这是哪个场景」。

对策是在每层目标中再加一项 **逆动力学（IDM）损失**：从相邻两个状态表示回归出连接它们的动作。

$$
\mathcal{L}^{(\ell)}_{\mathrm{idm}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{1}{d_a} \left\| I^{(\ell)}\left(z^{(\ell)}_t, z^{(\ell)}_{t+1}\right) - \mathrm{sg}\left[a^{(\ell)}_t\right] \right\|_2^2
$$

其中 $\mathrm{sg}[\cdot]$ 是 stop-gradient；$\ell > 1$ 时回归目标是汇聚后的宏动作，因此梯度只经由两个状态流向编码器。

DROID 没有可用的仿真器，作者用 **Fréchet 保真度** 离线打分：比较规划出的末端执行器三维路径 $p$ 与专家路径 $\tilde{p}$ 的离散 Fréchet 距离，并用「零动作路径」$\mathbf{0}$ 归一化：

$$
\text{fidelity} = \frac{d_F(\mathbf{0}, \tilde{p}) - d_F(p, \tilde{p})}{d_F(\mathbf{0}, \tilde{p})}
$$

0% 等于手臂不动，100% 等于完美复现专家路径，负值比不动还糟。它评分的是整条路径的形状而非终点——这正好能区分「直奔放置点」和「先下探抓取再搬运」这类非贪心行为。作者在 OGBench Cube（二值结果）和 RoboCasa CloseDrawer（连续进度）两个可观测结果的环境上校准了该指标：它是成功的必要条件，且能对任务完成度排序。

结果如下两张图：

![cls ladder](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/cls_ladder.png)

> 图解：DROID（5 fps）上的逐级消融，纵轴 Fréchet 保真度（%）。不加 IDM 的 LeWM 直接慢特征坍塌（保真度为 0）；加 IDM 后 LeWM 跃升到约 34%（+34），成为强基线；HWM（Level 2 为恒等编码器）再 +1；H-JEPA（Level 2 为学习到的编码器）再 +5，达到约 40%。在 DROID 上，表示杠杆和时间杠杆 **同时生效**。

![pareto real](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/compute/pareto_real_cls_droid_sharedy.png)

> 图解：Fréchet 保真度 vs 每 episode 规划算力（TFLOPs，对数轴），粗线为 Pareto 前沿。两层 H-JEPA（红）在所有预算下都高于扁平 LeWM+IDM（灰）和 HWM（橙）；两个冻结编码器的公开世界模型（JEPA-WM、V-JEPA 2-AC）需要高出 **几个数量级** 的算力还只能达到更低的保真度——端到端训练 + 层级抽象的组合优势一目了然。

作者还把规划器「想象」的 rollout 解码出来检查：

![decoded plans](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/decoded_plans.png)

> 图解：全部 16 条评测 clip 的解码规划（按 Fréchet 保真度从 +75% 到 0% 排序）。每条 clip 上排是专家视频，下排是 Level-1 rollout 的解码，蓝圈列是 Level 2 宏规划（每三帧插一帧）；绿框是给规划器的上下文帧，红框是想象的未来。两个标量指标看不到的事实：(1) 想象的轨迹普遍先朝被操作物体 **下探** 而非直奔目标帧——正是评测协议想诱发的非贪心行为，即使在保真度为 0 的 clip 上模型也「知道物体在哪」；(2) Level 2 宏规划是同一条轨迹的粗粒度但连贯的版本，没有退化。

### 附录理论：为什么只有 IDM 能挡住坍塌

这是全文理论浓度最高的一部分，作者用三个层层递进的结论说明「为什么 SIGReg 这类正则挡不住慢特征坍塌，而 IDM 可以」。把观测的未观测 latent state 拆成三部分：**背景** $b$（episode 内恒定：场景、相机、光照）、**前景** $x_t$（动作控制但只是部分可预测：接触、遮挡）、**干扰物** $y_t$（运动但与动作无关且不可预测：其他智能体、阴影）。慢特征坍塌的严格定义是：$z_t$ 只关于背景的 $\sigma$-代数可测，即在每个 episode 内恒定；其「程度」用 episode 内方差占比 $\chi(E)$ 度量，完全坍塌时 $\chi = 0$。

**结论一（可行性命题）：任何只看边际分布的正则都拦不住坍塌。** 对任意满足 $\mathcal{R}(\mu_0) = 0$ 的边际分布泛函，总可以构造编码器 $E = Q \circ U \circ b$（先把背景均匀化，再分位映射到目标分布）配上恒等预测器 $F(z, a) = z$，同时达到 $\mathcal{L}_{\mathrm{pred}} = 0$ 和 $\mathcal{R}(\mathrm{law}(z)) = 0$。方差-协方差正则、白化、熵惩罚、SIGReg 全都只看单嵌入的边际 law，无一幸免。反面推论也很有趣：当所有 episode 场景相同（$\mathrm{law}(b)$ 是点质量）时这个构造失效——这解释了为什么固定相机的仿真环境 **不需要** IDM，而 DROID **必须** 要。

**结论二（可预测性界定理）：坍塌不仅是可行解，还是最优解。** 定义不可约风险 $\mathcal{E}(E)$ 和创新敏感度 $\nu(E)$（知道完整 latent state 和动作后仍然残存的下一嵌入方差），则对任意可测 $E$ 有 $\mathcal{E}(E) \geq \nu(E) \geq 0$，且 $\mathcal{E}(E) = 0$ 当且仅当 $\sigma(z)$ 是「动作闭合」的（$z_{t+1} = \Gamma(z_t, a_t)$ 几乎必然成立）。只读背景的编码器 $\nu = 0$、$\Gamma = \mathrm{id}$，于是 $\mathcal{E} = 0$。实证上，坍塌 run 的预测器确实退化为恒等映射：把真实动作换成其他 episode 的动作后 rollout 几乎不变（动作敏感度 < 0.05）。

**结论三（IDM 地板命题）：IDM 提供的是量级完全不同的壁垒。** 若 $E$ 已慢特征坍塌，则对任意 IDM 头：

$$
\mathcal{L}_{\mathrm{idm}} \geq \frac{1}{d_a} \frac{1}{T-1} \sum_{t=1}^{T-1} \left( \mathrm{tr}\,\mathrm{Var}(a_t) - \mathrm{tr}\,\mathrm{Var}\left( \mathbb{E}[a_t \mid \mathcal{C}] \right) \right)
$$

动作按维度归一化后这个地板就是 $1$——坍塌在 $\gamma = 50$ 的 IDM 权重下要付出 $50$ 的代价。相比之下，SIGReg 对坍塌的「壁垒」是 $\Delta = \lambda (T-1) \kappa$，且只在把时间轴放进样本轴的分组方式下才非零；本文使用的 $(T, N, D)$ 分组（每个 normality 检验看 $N$ 条序列的同一时刻）下 $\Delta = 0$。作者还尝试了不需要动作标签的替代方案——时间排斥项 $\mathcal{L}_{\mathrm{sim}}$（负权重奖励相邻嵌入的差异）：它确实能防止预测器退化为恒等（动作敏感度保持 0.2–0.5），但没有任何权重设置能让规划保真度越过「零动作地板」，而 IDM-on 基线以 30.66% 大幅越过。理论窗口 $\theta_F < |\lambda_{\mathrm{sim}}| < \theta_D$ 在实测的 $\hat{\theta} \approx 0.51$ 下薄到基本不起选择作用。

实验网格与理论完全对得上：

![anticollapse](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/anticollapse_cls.png)

> 图解：以 Level-1 DROID 配置为中心的完整网格（SIGReg × IDM 权重 × 是否 detach 预测目标，16 格）。左图是表征有效秩（满秩 384）：只要关掉 SIGReg，无论 IDM 多大权重，表征都坍塌到秩 1–14，规划保真度跌破「零动作地板」（灰色带，标注真值如 −10、−24）；右图显示 **SIGReg 恢复秩，但只有 IDM 能把秩转化为规划保真度**——开 SIGReg + IDM=100/200 的格子保真度达 33–35%。detach 与否对结果几乎没有影响，因此主实验不用目标 detach。

两个 loss 系数需要联合调：Level 1 的大 IDM 权重配小 SIGReg（或反之）都会掉到地板附近，中间一大片组合都表现良好；Level 2 则迟钝得多，九种组合保真度相近。

![crossval](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/crossval_l1grid_cls.png)

> 图解：DROID 5 fps 下的 Fréchet 保真度热力图（三次训练种子均值，各含三个规划种子）。左：Level-1 单层模型的 SIGReg × IDM 联合网格，粗框是论文采用的系数（在最佳格的一个标准误之内）；右：冻结 Level 1 后的 Level-2 网格，九格几乎一样平——层级越深，对正则系数越不敏感。

![latent variance](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/latent_variance_decomp_no_robocasa.png)

> 图解：各环境潜空间总方差拆成 episode 内（即慢特征程度 $\chi$）与 episode 间两份。DROID 的 episode 间占比约 75%，Cube 仅约 28%；多样场景的环境天然更接近坍塌边缘。代价不是「够不着」而是「容量」：episode 均值在同 episode 的「状态减目标」差里会抵消，但它挤占了留给机械臂的潜空间容量。Level 2 几乎不改变这个占比，所以两层都要加 IDM。

### DROID 工程细节与指标校准

评测协议：固定的 16 条抓取-放置 clip（光照良好、左右相机都能看到被操作物体、物体足够大），每条是以抓取帧为中心的 108 帧（约 7.2 秒）定长窗口，最后一帧为目标；5 fps 下 Level-1 horizon $H_1 = 36$。所有 DROID 模型训练在重编码语料上（74,530 episodes，127.6 GB，约为原始 5.5 TB 的 1/45，解码快 17.9 倍，中位 PSNR 33.75 dB，轨迹逐比特一致），且 16 条评测 clip 从训练集中剔除。一个容易踩的坑：DROID 以 15 Hz 录制但容器标记为 60 fps，若按标记推 stride，采样率会缩水四倍——两个冻结编码器基线实际工作在 1 fps 原生速率下。

指标校准的两个结论也值得记住：在 Cube（二值结果）上，没有一个成功 episode 的保真度低于某地板，但越过地板也区分不了谁成功——Fréchet 差距 **低估** 了真实结果差距；在 RoboCasa CloseDrawer（连续进度）上，固定配置内的秩相关为正且中等，控制路径长度和难度后依然成立——所以这个指标度量的是「沿专家路径走了多远」。

![frechet cube](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/frechet_cube_ab.png)

> 图解：OGBench Cube 上 Fréchet 保真度 vs 抓取成功率。一个冻结 checkpoint，9 组配置 × 3 规划种子 × 50 episodes。(a) 每个点是一种规划器配置，阴影带里扁平和层级规划器离线保真度相同但抓取结果相反；(b) 逐 episode 散点：低于保真度阈值无一成功，但阈值之上多数仍失败——任何路径指标的宿命。

![robocasa frechet](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/robocasa_frechet_vs_graded_nomodes.png)

> 图解：RoboCasa CloseDrawer 上同一离线指标 vs 连续化的关抽屉进度（99 种规划配置 × 32 条 held-out episodes）。横轴 Fréchet 保真度，纵轴分级成功率。结论建立在「固定配置内」的秩相关上（排除了配置间均值偏移的虚高），保真度确实对完成度有排序能力。

## 数据分工：每层吃不同的数据

附录里一个很有实用价值的实验：如果允许每层用不同数据训练（分阶段训练：冻结下层再训上层），会发生什么？作者用 AntMaze 的两种数据——explore（高覆盖、弱时间相关的探索）与 stitch（沿最短路的目标导向片段）——及其 union 做了组合网格。

![dataset grid level1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/ant/dataset_grid/level1.png)
![dataset grid stagewise](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/ant/dataset_grid/stagewise.png)
![dataset grid e2e](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/ant/dataset_grid/e2e_joint.png)

> 图解：三张规划成功率热力图（%，GD 在线规划，三种子均值）。左图是 Level-1 扁平规划：explore/stitch/union 分别为 18/11/19。中图是分阶段训练的两层规划（行 = Level-1 数据，列 = Level-2 数据）：最优组合是 **explore 喂低层 + stitch 喂高层，成功率 67**，远超任何同数据组合；反过来（stitch 低层 + explore 高层）直接崩到 2。右图是端到端联合训练（只有对角线）：stitch 从单层 11 → 21，union 从 19 → 54，但 explore 上两层（11）反而不及单层（18）——纯探索轨迹缺乏高层抽象所依赖的时间相关性，加了抽象层也没用。

结论有三层含义：(1) 加抽象层只在数据有时间结构时才有用；(2) 低层偏爱广覆盖的探索数据，高层偏爱时间连贯的目标导向数据，**搭配** 才是关键；(3) 这为分阶段训练提供了实用论据——冻结低层后可以给高层单独定制数据，而默认的端到端流程做不到这一点。

## 训练与实现细节

单场景环境的共享配置：Level-1 编码器是 ViT-Tiny（[CLS] token 经一层 MLP 投影 + BatchNorm；AntMaze 额外拼接 27 维本体感觉的 128 维嵌入），预测器是 4 层 8 头、宽 192、FF 1024 的因果 Transformer（动作经 adaptive LayerNorm 注入）。上层编码器是隐藏宽 384 的两层 MLP；一层 Transformer 把两个下层动作嵌入聚成 4 维宏动作。训练用 AdamW（weight decay $10^{-3}$，线性 warmup 1% + cosine 衰减，梯度裁剪 1.0，bfloat16），Level-1 峰值学习率 $5 \times 10^{-5}$，上层 $10^{-5}$，SIGReg 用 $M = 1024$ 个随机投影，无 stop-gradient、无 EMA target、无预训练权重，三个种子（42/43/44）。Level-1 单步跨 5 个环境步，上层 stride 均为 2。

各环境差异如下：

| 设置 | FourRoom | AntMaze | Push-T | Cube |
| --- | --- | --- | --- | --- |
| 帧尺寸 / patch | 224 / 14 | 64 / 4 | 224 / 14 | 224 / 14 |
| 动作 / 本体感觉维度 | 2 / — | 8 / 27 | 2 / — | 5 / — |
| 潜空间维度 | 192 | 384 | 192 | 192 |
| 上层预测器层数 / 头数 | 4 / 8 | 5 / 12 | 6 / 16 | 6 / 16 |
| 上层预测器宽度 / FF 宽 | 192 / 1024 | 224 / 1536 | 192 / 2048 | 192 / 2048 |
| SIGReg $\lambda$（LeWM / H-JEPA） | 0.72 / 0.18 | 0.09 / 0.09 | 0.09 / 0.09 | 0.09 / 0.09 |
| H-JEPA 训练 epochs | 7 | 5 | 9 | 8 |

层级越深，单个训练样本需要的最短 clip 越长，而 clip 不能跨 episode，因此每 epoch 可用样本随深度下降——这是理解 Push-T 失败的关键数据：

| | LeWM | 两层 H-JEPA | 三层 H-JEPA | 四层 H-JEPA |
| --- | --- | --- | --- | --- |
| 每 clip 的 Level-1 帧数 | 4 | 7 | 13 | 25 |
| 最短 clip 长度（环境步） | 16 | 31 | 61 | 121 |
| FourRoom clips/epoch | 952,708 | 915,298 (96%) | 840,478 (88%) | 690,838 (73%) |
| AntMaze clips/epoch | 9,575,000 | 9,200,000 (96%) | 8,450,000 (88%) | 6,950,000 (73%) |
| Push-T clips/epoch | 1,981,721 | 1,701,446 (86%) | 1,143,623 (58%) | 285,931 (14%) |
| Cube clips/epoch | 1,783,600 | 1,636,600 (92%) | 1,342,600 (75%) | 754,600 (42%) |

> 表解：括号内为相对同环境 LeWM 的百分比。FourRoom 和 AntMaze 的 episode 长达 400 步，四层也只损失 27%；Push-T 平均 125 步，三层只剩 58%、四层只剩 14%——数据饥饿直接解释了它在高层的崩溃。H-JEPA 各深度间是 epoch 对齐的，所以更深的模型梯度步数反而更少。

训练动态健康与否，看两条曲线：SIGReg 统计量应在早期骤降后稳定在「恰高斯嵌入」取值（约 1）的小倍数内；预测损失通常先升后降（SIGReg 先把初始聚集的嵌入摊开），最终停在非零平台——**预测损失趋向 0 恰恰是坍塌的标志**（常数嵌入平凡可预测）。

![training dynamics](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/training_dynamics/depth4_losses.png)

> 图解：四个环境四层模型的逐层训练曲线（三种子均值 + min/max 带，对数轴）。所有层的预测损失和 SIGReg 都未退化，最深的四层设置下端到端训练依然稳定。两项指标都按层排序：越高层的预测损失和 SIGReg 越大，与上层看到的 clip 更少、目标更粗一致；上层开始下降的时间也晚于 Level 1（AntMaze 上最明显）。总目标全程下降并在末段走平。

![droid dynamics](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/H-JEPA-End-to-End-Learning-of-Hierarchical-World-Models-for-Visual-Planning/figures/droid/droid_training_dynamics.png)

> 图解：DROID 模型 100 epochs 的训练曲线（橙虚线 = 扁平 LeWM+IDM，实线 = 两层 H-JEPA 的两个层级；Level 1：SIGReg 0.08、IDM 100；Level 2：SIGReg 0.16、IDM 50）。没有任何一项退化；H-JEPA 的 Level 1 预测与 IDM 损失比同设置单训的 LeWM+IDM 还低——**联合训练上层不会拖累下层**，反而略有助益。

规划器侧的共享设置：每层用 AdamW 最多 90 步（代价停滞即早停），Level-1 候选从标准高斯初始化；上层没有天然动作边界，候选初始化自最近 4000 个训练宏动作拟合的逐维高斯，每步优化后裁剪到 2–98 分位。闭环时每执行两个 Level-1 动作块（10 环境步）就从头重规划；planning 时每层只编码当前帧与目标帧，预测上下文 $c_\ell = 1$。各环境的 episode 预算与 horizon/候选数配置：

| 设置 | FourRoom | AntMaze | Push-T | Cube |
| --- | --- | --- | --- | --- |
| Episode 预算（环境步） | 210 | 350 | 90 | 30 |
| $\eta$，LeWM | 0.1 | 0.3 | 0.05 | 0.05 |
| $\eta$，H-JEPA（L1，上层） | (0.08, 0.1) | (0.06, 0.1) | (0.1, 0.075) | (0.1, 0.075) |
| $H$，LeWM | 23 | 23 | 15 | 4 |
| $H$，两 / 三 / 四层 | (2,12) / (2,2,6) / (2,2,2,3) | (4,9) / (4,2,5) / (3,2,2,3) | (2,8) / (2,2,4) / (2,2,2,2) | (2,2) / (2,2,1) / (2,2,1,1) |
| $S$，LeWM | 200 | 100 | 800 | 800 |
| $S$，两层 | (100, 800) | (25, 200) | (50, 400) | (400, 100) |
| $S$，三层 | (50, 200, 200) | (100, 100, 400) | (25, 50, 200) | (25, 800, 800) |
| $S$，四层 | (50, 200, 200, 400) | (50, 200, 25, 50) | (200, 800, 800, 800) | (200, 50, 800, 800) |

> 表解：深度对比实验的规划器工作点（$S$ = 候选序列数，$\eta$ = AdamW 学习率，$H$ = 该层预测步数计的 horizon，元组从低层到高层列出）。计算扫描只改候选数，其余设置冻结，最终在独立的 50 任务集上评测。DROID 侧为完全开环：$H = (36, 12)$，$S = (16, 16)$，$\eta = (0.01, 0.3)$，子目标权重 $\beta = 0.5$。

DROID 模型本身的超参数也与仿真不同：Level-1 编码器换 ViT-S/16，预测器为宽 384、深 12 的因果 Transformer（$c_1 = 7$）；Level-2 stride 3，宏动作编码器把三段 7 维原始指令拼成 8 维宏动作；SIGReg 权重 0.08（L1）/ 0.16（L2），IDM 权重 100 / 50，另对 Level-2 宏动作嵌入加 0.005 的 SIGReg；总 batch 256，峰值学习率 $5 \times 10^{-4}$，100 epochs。

最后一个工程细节：SIGReg 的 Epps–Pulley 统计量是整个 batch 的函数而非逐样本求和，直接在各 rank 上各算再平均会随设备数漂移。作者的做法是在经验特征函数（ECF）层面做 all-reduce——每次调用只传两个 $M \times 17$ 的小张量，统计量对分片 **精确不变**：

| 分片方式 | 本文实现 | LeWM 单卡参考实现 |
| --- | --- | --- |
| $1 \times 256$ | 1.372619178543 | 1.372619 |
| $2 \times 128$ | 1.372619178543 | 1.180770 |
| $4 \times 64$ | 1.372619178543 | 1.186233 |

> 表解：同一固定高斯矩阵（$n=256$，$D=32$，$M=64$）按不同方式分片时的 $\mathcal{T}_{\mathrm{EP}}$ 值。本文实现对启动拓扑完全不变（梯度范数也逐位一致），而公式相同但按局部 batch 计算的参考实现漂移约 14%——多卡训练时这个漂移是纯粹的噪声。

## 相关工作定位

在「从像素出发、任务无关、免重建、潜空间层级」四个维度上，H-JEPA 是唯一同时满足四条的方法：

| 方法 | 像素输入 | 任务无关世界模型 | 免重建 | 学习到的潜空间层级 |
| --- | --- | --- | --- | --- |
| 经典层级 MPC | ✗ | ✓ | ✗ | ✗ |
| Director | ✓ | ✗ | ✗ | ✗ |
| Puppeteer | ✓ | ✗ | ✓ | ✗ |
| SPlaTES | ✗ | ✗ | ✓ | ~ |
| Schiewer et al. | ✗ | ✗ | ✗ | ✓ |
| THICK | ✓ | ~ | ✗ | ✗ |
| GCP | ✓ | ✓ | ✗ | ✗ |
| HWM | ✓ | ✓ | ✓ | ✗ |
| **H-JEPA（本文）** | ✓ | ✓ | ✓ | ✓ |

> 表解：与层级控制方法的四轴对比（~ 表示部分满足）。层级视频模型（Clockwork VAE、VTA、STREAMER）重建像素且不用于控制；层级 model-based RL（Director、Puppeteer、THICK）靠奖励驱动且任务特定；最接近的 HWM 在共享潜空间里做时间层级——恰好只差「每层独立表示」这一个轴，也因此成为本文最核心的对照基线。

## 总结与展望

- **方法**：H-JEPA 把多个 action-conditioned JEPA 端到端叠成层级，每层在自己的潜空间、以更粗的 stride 预测更远的未来，无奖励、无重建。
- **表示**：当数据中的因子存在时间频率分离时（频率差 > 约 2×），高层自动丢弃不可预测的快变细节、保留慢变的任务相关状态——选择性抽象是「预测什么就保留什么」的自然产物。
- **规划**：自上而下、子目标接力的规划在 AntMaze 上把成功率从 18% 推到 73% 且算力更省；归因实验把收益拆成两个互补机制——更抽象的 goal-matching 代价 + 时间分解出的短子问题。
- **真实机器人**：场景多变时纯预测目标会发生慢特征坍塌；理论上任何边际正则（含 SIGReg）都拦不住它，IDM 项提供了 $\Omega(1)$ 量级的壁垒，使 H-JEPA 在 DROID 上以更低算力取得更高离线规划保真度（约 40% vs HWM 的 35%）。
- **实用启示**：低层爱探索数据、高层爱连贯数据（explore+stitch 组合达 67%）；episode 太短的环境（Push-T）撑不起深层级。

局限与方向同样清晰：一切增益尚待在真实机器人上闭环验证；扩展到无动作标签的视频需要替代 IDM 的抗坍塌手段（时间排斥项目前还不够）；而上层潜状态与语言对齐、用变长片段替代固定 stride，则可能让「用语言下达目标、让模型自己学会事件切分」成为这条路线下一个真正的台阶。

> 本文参考自 [H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning](http://arxiv.org/abs/2610.06805v1)