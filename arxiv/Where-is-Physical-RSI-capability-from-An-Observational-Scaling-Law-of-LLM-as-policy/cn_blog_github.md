# 物理 RSI 能力从哪来？——LLM-as-policy 的观察性 Scaling Law 解读

大模型直接当机器人策略（LLM-as-policy）已经能边试错边改代码、越改越好，但"为什么会越改越好"一直是个黑盒。这篇来自 Galbot 团队的工作把这件事拆成了三个可测量的问题：物理递归自我改进（Physical RSI）的能力从哪来、它遵循怎样的 Scaling Law、改进到底是过拟合还是泛化。作者提出基准 **PhysRSI-eval**，用一套五轴指标在 18 个双臂操作任务、固定 17 轮交互预算下量化 6 个系统的改进过程。最硬的两个结论：空间推理能力与"开局表现"的秩相关高达 $\rho=0.94$，而感知能力才是"最终得分与失败恢复"的最强相关项（$\rho=0.94$）；消融实验中去掉历史经验后，后期成功率从 41.7% 暴跌到 8.3%，证明 agent 写下的经验真的在起作用。

![PhysRSI-eval 总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/teaserv4.jpg)

> 图解：全文三个问题的总览。(Q1) 左侧讨论能力来源与"物理木桶假说"：空间、感知、编码、Harness 操作四类能力像木桶的板子，分别限制改进的不同侧面；(Q2) 中间是五轴量化与观察性 scaling，$T_{50}$ 表示"半增益轮次"，即最佳平均分首次达到 K1–K17 总增益一半的那一轮；(Q3) 右侧检验改进是过拟合于练习场景，还是能迁移到新布局、新任务。

## 三个根本问题

前沿 LLM agent 已经能通过迭代试错不断提升操作能力：更新技能、生成代码、调用工具。作者把这个过程称为 **物理递归自我改进（Physical Recursive Self-Improvement, Physical RSI）** ——模型权重冻结，所有"学到的东西"必须写进它能读回的工件（技能、工具、策略代码、记忆）里。

这篇文章不想再展示"agent 多强"，而是追问三件更本质的事：

- **Q1：Physical RSI 的能力从哪来？** 感知负责解读结果、空间推理把观测和动作关联起来、编码能力负责实现修订——做好第一次尝试所需的能力，和从失败中恢复所需的能力，是同一种吗？
- **Q2：如何量化 Physical RSI 及其 Scaling Law？** 一个终点分数既说不清增益何时发生，也说不清它花了多少物理交互代价。
- **Q3：改进是过拟合还是泛化？** 分数上涨可能只是适应了练习场景，agent 到底留下了可复用的程序，还是只记住了这个场景？

## PhysRSI-eval：把三个问题变成可测量的基准

### 两阶段、一个接口的受控循环

PhysRSI-eval 的核心设计是：模型权重不动，技能、工具、策略代码和记忆存在一个 **持久化 workspace** 里，每次修订都通过一次"被打分的物理尝试"来检验。

![两阶段工作流](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/flowchat.jpg)

> 图解：上方是两个阶段共享的任务输入与机器人接口（观测、物理动作、记忆与文件、历史）。左侧是 **Evaluate 阶段 ** （首次尝试轮 K0）：一个 episode 内完成"观察—推理—行动"，没有 workspace，历史只存在于当前 episode 内。右侧是 **Optimize 阶段** （自我改进轮 K1–K_H）：每一轮从持久 workspace（技能、工具、策略、记忆）出发执行一次物理尝试，由任务原生评分器打分，随后在机器人动作关闭的情况下修订 workspace 并提交，作为下一轮的起点，直到首次成功或轮次用尽。

具体协议要点：

- **K0（首次尝试）**：没有持久 workspace，每个任务跑 3 次取平均，衡量"裸起点"。
- **K1–K17（自我改进）**：K1 时 workspace 为空，每轮执行一次物理尝试，返回原生分数、成功标志和失败原因；agent 在机器人动作关闭的情况下修订 workspace，提交后进入下一轮。每个任务最多 $H=17$ 轮，首次成功即停止。
- **预算**：每轮 4200 秒推理时间，修订窗口 GPT 系统 1800 秒、Claude 系统 600 秒。物理预算按步数计，每任务 $17 L_t$（$L_t$ 为该任务的单轮步数上限）。

一个细节值得注意：K0 到 K1 之间既换了协议（多了 workspace 和程序执行工具）又可能有真实改进，所以作者只把 K1–K17 当作自我改进段来分析。这个处理很诚实——不把"工具变多了"的功劳算到"自我改进"头上。

### 18 个任务、6 个难度分层

任务套件来自 RoboDojo 双臂操作基准，共 18 个任务，覆盖语言条件摆放、堆叠、容器放置、插入、倾倒、推动和动态抓取，横跨 RoboDojo 官方的五大任务类别（Precision、Long-Horizon、Memory、Open、Generalization）。

![任务难度分层采样](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/task_table.png)

> 图解：18 个任务按 RoboDojo 官方成功率排名分成 6 个难度层，每层抽 3 个任务，从 D1（官方成功率 60–68.7%，最容易）到 D6（成功率 0–0.4%，几乎做不动）。分层在主实验之前就固定好了，不是根据改进结果事后贴的，这保证了难度覆盖的均衡性。

被评的 6 个系统是：GPT-6 Astra、GPT-6.1 Sol、GPT-6 Sol、GPT-5.6 Sol、Claude Opus 5.5、Claude Sonnet 5.5。注意"系统"= 模型 + 运行配置（客户端、推理强度、修订窗口等），所以跨系统比较是描述性的。Gemini 4 在另外 10 个任务上单独报告，不与主榜单混算。

### 五轴指标：一个终点分数远远不够

这是本文方法论上最核心的贡献。单看最终分数，你不知道增益是早是晚、代价是大是小、是靠反复碰运气还是真从失败里学了东西。作者把 Physical RSI 拆成五个轴，记第 $k$ 轮的任务平均分为 $q_k$，$C_k$ 为 K1–K$k$ 内解决的任务比例：

**A1：首次尝试表现（First-Attempt Performance）**。$A_1 = q_0$，即 K0 平均分，含部分进展分。回答"第一次上手做得怎么样"。

**A2：效率（Efficiency）**，衡量增益出现得多早：

$$
A_2 = \frac{\frac{1}{H-1}\sum_{k=1}^{H-1} q_k - q_0}{q_H - q_0}
$$

越早出现持续性增益，值越高。它测的是"增益时机"，不是墙钟速度。

**A3：成功（Success）**。先定义达到峰值增益 90% 的首轮：

$$
\tau_{90} = \min\{k \geq 1 : q_k \geq q_0 + 0.9(q_{\max} - q_0)\}
$$

然后取高分窗口的均值 $A_3 = \frac{1}{H - \tau_{90} + 1}\sum_{k=\tau_{90}}^{H} q_k$。它既不是二值成功率，也不是渐近上限，允许部分得分。

**A4：预算（Budget）**，衡量省下了多少物理交互。对已解决任务 $t$，$W_t$ 是首次成功前累计步数，$Z_t$ 是推理超时轮被计满但未用的步数，$e_t$ 是额外尝试次数，$U_t = (W_t + Z_t)/L_t + e_t$：

$$
A_4 = \frac{1}{|\mathcal{S}|}\sum_{t \in \mathcal{S}} \max(0, 1 - U_t / H)
$$

值越高说明物理预算省得越多。时间、token、金钱都不计入——只算物理交互这个最贵的成本。

**A5：改进（Improvement）**。$A_5 = (C_H - C_1)/(1 - C_1)$，即"K1 失败的任务中，有多少后来在预算内被解决"。回答"早期失败有没有被真正克服"。

博主点评：这五轴设计的聪明之处在于它把"高分""快""省"解耦了——一个系统完全可能分数很高但改进很晚、很费交互（后文的 Opus 5.5 就是这样）。单一排行榜会抹掉这些差异，而五轴画像把它们全部暴露出来。

## Q1：Physical RSI 的能力从哪来？

### 编码能力强 ≠ 物理改进强

先泼一盆冷水：GPT-5.6 Sol 在 LiveBench 的代码生成与补全上超过 GPT-6 Astra，但它在 18 个任务里只解决了 1 个，Astra 解决了 15 个。公开编码榜单的高分，完全不能担保物理自我改进。

作者把 17 个公开 benchmark 分数分成四组能力画像—— **空间推理（Spatial）、Harness 操作、感知（Perception）、编码（Coding）**，然后与五轴做秩相关分析：

- **空间推理** 与 A1（首次尝试）关联最强，$\rho = 0.94$；与 A2/A4 也有 $\rho = 0.83$；
- **感知** 与 A3（最终得分）、A5（失败恢复）关联最强，$\rho = 0.94$；
- **编码** 与每个轴的关联都最弱，$\rho \leq 0.37$。

### 物理木桶假说（Physical Barrel Hypothesis）

这个模式催生了全文的核心概念： **不同的改进侧面可能被不同的能力卡住瓶颈**。开局开得好，主要靠空间推理；从失败中解读信息并恢复，更依赖感知。就像木桶能装多少水由最短的板决定，agent 的某项能力短板可能无法被其他强项补偿。

![能力关联矩阵与木桶假说](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/Barrel_matrixv3.png)

> 图解：左侧矩阵报告 6 个系统在 18 个任务上的探索性秩相关。右侧把每个 RSI 轴画成一个木桶，每块板对应一组能力，板的高度直观展示"哪块板最可能成为瓶颈"的解读。注意这些关联只是候选瓶颈（6 个系统的样本量），作者明确说它们不是因果结论，而是留给后续定向能力干预实验去检验的假说。

博主点评：这个假说解释了一个常见困惑——为什么有的模型 benchmark 均分很高、上了机器人却"学不会"。因为它可能有一块短板（比如感知），恰好卡在"从失败中恢复"这个环节上。

## Q2：如何量化 Physical RSI 及其 Scaling Law？

### 高分、快速、省钱，三者可以互相背离

五轴画像立刻暴露了单榜排名的盲区。Opus 5.5 的 Success 轴比 GPT-6.1 Sol 高（0.744 vs 0.703），但 Efficiency 明显更低（0.609 vs 0.822）；在各自解决的任务上，Opus 省下的预算也更少（0.579 vs 0.808），且在两者共同解决的 10 个任务上差距依然存在。 **更高的终点分数，可能伴随更晚的增益和更大的物理代价。**

| 系统 | 任务数 | A1 首次尝试 | A2 效率 | A3 成功 | A4 预算 | A5 改进 |
|---|---|---|---|---|---|---|
| GPT-6 Astra | 18 | 0.368 | 0.859 | 0.851 | 0.843 | 0.750 |
| GPT-6.1 Sol | 18 | 0.284 | 0.822 | 0.703 | 0.808 | 0.571 |
| GPT-6 Sol | 18 | 0.031 | 0.466 | 0.472 | 0.372 | 0.389 |
| GPT-5.6 Sol | 18 | 0.012 | 0.225 | 0.104 | 0.301 | 0.056 |
| Claude Opus 5.5 | 18 | 0.177 | 0.609 | 0.744 | 0.579 | 0.688 |
| Claude Sonnet 5.5 | 18 | 0.124 | 0.695 | 0.701 | 0.698 | 0.563 |
| Gemini 4* | 10 | 0.050 | 0.783 | 0.635 | 0.624 | 0.600 |

整体来看，6 个系统的任务平均分从 K0 的 0.01–0.37 涨到评测结束的 0.10–0.84，分别解决 15、12、7、1、13、11 个任务。Astra 在五轴上全面领先；两个 Claude 系统"得分高、增益偏晚"；GPT Sol 家族内部跨度极大；GPT-5.6 Sol 的累计成功率连续 16 轮贴近零—— **在这个预算内，堆尝试次数并不能填平所有能力差距**。

![五轴雷达图（叠加）](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/blog/radar_overlay_7models.png)

![五轴雷达图（分面板）](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/blog/radar_panels_7models.png)

> 图解：五轴雷达画像。左图把 7 个系统的画像叠加在一起，右图每个系统一个面板。从顶部顺时针依次为 A1 首次尝试、A4 预算、A3 成功、A2 效率、A5 改进，统一 0–1 刻度。可以直观看到 Astra 的五边形最大最饱满，GPT-5.6 Sol 几乎缩成一个点，而两个 Claude 系统在 A2/A4 两轴上明显内凹——"高分但慢且费"的 trade-off 一目了然。

公开 benchmark 均分能多大程度预示这些差异？四个 GPT 系统的 17 项公开 benchmark 均分跨度只有 54.1–78.1（24 分），而五轴等权物理画像均分（$100\times\sum_{j=1}^{5}A_j/5$）的跨度是 13.9–73.4（59.5 分）。公开榜单把系统"挤"在一起，物理评测把它们拉开。

![公开能力与物理 RSI 的分离](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/blog/figure4_success_benchmark_area.png)

> 图解：左图是 18 个任务上改进过程中的累计成功率曲线，横轴为迭代轮次 K，纵轴为累计成功百分比，阴影区标出 GPT-6.1 Sol 与 GPT-6 Sol 之间的差距；注意 GPT-5.6 Sol 的曲线 16 轮内几乎贴地。右图横轴是 17 项公开 benchmark 均分，纵轴是五轴等权物理均分（百分制）。中间标注的"Rapid contraction region"（快速收缩区）说明：在物理评测下，公开分数相近的系统表现急剧分化——这正是木桶假说的阈值式体现：某项关键能力不达标，其他再强也白搭。

### 跨模型预测：公开能力画像能否预言物理改进？

这是观察性 scaling 的第一个视角。假设不跑完整评测，只看公开能力画像，能不能预言一个没见过的新系统的物理表现？作者问了两个子问题：能改进多少，以及增益何时出现。

- **改进多少**：用四组能力画像的等权平均给系统排序，完整复现了 K17 最佳分数 $\bar q_{17}$ 的实际排名；
- **何时改进 **：定义半增益轮次 $T_{50}$（最佳平均分首次达到 K1–K17 增益一半的那一轮）。对每一对系统，用其余系统的画像训练一个核回归器，预测两者谁先到 $T_{50}$——**所有两两排序全部预测正确**。

![从能力画像预测增益时机](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/blog/half_gain_round_prediction.png)

> 图解：左图是 6 个系统在 Spatial / Harness / Perception / Coding 四组上的归一化能力画像（17 个公开 benchmark 的秩均值）。右图是 18 个任务上的最佳平均分 $\bar q_k$ 随轮次 K 的曲线，圆圈标出各系统的 $T_{50}$。曲线旁的标签是"预测名次 / 实际名次"——$T_{50}$ 的两两排序和 K17 最佳分数的排序都被仅用其余系统拟合的预测器完整复现（1/1、2/2……全部命中）。

### 轮次内预测：早期物理反馈能预言后期增益吗？

第二个视角更贴近实用：交互一旦开始，前几轮的表现能否预测继续投预算还值不值？6 个系统到达高分窗口的时间差异很大（$\tau_{90}$：Astra 在 K9，GPT-6.1 Sol 在 K7，Opus 拖到 K15）。

作者的做法很巧妙：相关模型的改进曲线可能 **形状相似**。先把 Weibull 增长模板 $g_A(t) = 1 - \exp[-(t/\tau_A)^{p_A}]$ 拟合到 Astra 的 K1–K17 均分曲线上并冻结，然后对 GPT-6.1 Sol 做形状迁移预测：

$$
\hat{q}(k) = u + (v - u)\, g_A(\rho(k - 1))
$$

起点 $u$、平台 $v$ 和相对速度 $\rho$ 只用 Sol 的前 5 轮校准，且增益与速度向模板正则化。结果： **只用 K1–K5 的观测，预测 K6–K17 的平均绝对误差仅 0.019**，而"保持最后一轮分数不变"的朴素基线误差是 0.112——差了约 6 倍。

![用 Astra 的改进形状预测 GPT-6.1 Sol](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/blog/astra_shape_transfer.png)

> 图解：左图是 Astra 的均分曲线（蓝点）与拟合的 Weibull 模板（橙线）。右图是 GPT-6.1 Sol：竖虚线左侧的 K1–K5 是可见观测，右侧橙色菱形是模板迁移给出的预测，黑色空心圆是被藏起来的真实后续轮次，阴影是 90% 配对任务 bootstrap 置信带。可以看到预测曲线几乎贴着真实走势走，连平台期高度都压得相当准。

博主点评：这个实验把 Physical RSI 和 LLM 预训练的 learning curve 研究（Kaplan、Chinchilla 一脉）接上了——只不过这里的"预算变量"从训练算力换成了物理交互轮次。当然作者自己也很克制：只有 6 个闭源系统、没有训练算力数据，这些分析本质上是观察性、探索性的。

## Q3：改进是过拟合还是泛化？

解决了"怎么测、怎么预测"之后，最后一个问题是：这些增益到底是什么？agent 学到的是可复用的程序，还是对练习场景的特殊适应？

### 去掉历史经验，改进就垮了

Astra 消融实验覆盖 6 个任务、每个任务 2 条链，K1–K6 每轮换布局、成功后也继续执行。实验逐通道切断"前几轮的信息能传给 agent 什么"：workspace、评分器反馈、历史图像、动作记录。

| 条件 | 保留的信息 | K4–K6 后期成功率 |
|---|---|---|
| Full（完整循环） | 全部四个通道 | 15/36（41.7%） |
| No-img（无历史图像） | 无历史图像 | 12/36（33.3%） |
| No-grader（无评分反馈） | 无评分器反馈 | 5/36（13.9%） |
| WS+FB（仅 workspace + 评分反馈） | 仅这两个通道 | 17/36（47.2%） |
| No-hist（无历史） | 全部切断 | 3/36（8.3%） |

结论分两层：其一， **去掉全部历史后后期成功率从 41.7% 跌到 8.3%**，说明改进确实依赖积累的经验，而不是"多试几次总会撞上"；其二，单通道消融里砍掉评分器反馈伤害最大（13.9%），而只保留 workspace + 评分反馈（47.2%）就已足够——历史图像和动作记录的边际贡献很小。

### 冻结 workspace：不写不改，还能用吗？

把 Astra 在 K6 之后的 workspace 直接冻结，禁止任何修订和结果反馈，拿到 5 个没见过的布局上测试：

| 条件 | 语义 | 成功率 |
|---|---|---|
| E3-full | 完整循环的 workspace，测试时可读历史原始记录 | 23/60（38.3%） |
| E3-notes | 完整循环的 workspace，仅 workspace 文件 | 25/60（41.7%） |
| HO-Fixed | 固定布局练出的 workspace，换新布局 | 27/60（45.0%） |

三个点估计都在 40% 上下，说明 **写进 workspace 的工件在布局变化下依然有用**；而测试时额外暴露原始历史记录没有任何帮助。固定布局练出的 workspace 在自家布局上是 14/16（88%），到未见布局降到 45%——既有场景特化的成分，也确实带走了可迁移的东西。作者很谨慎地指出：分母不同，不能做严格的配对比较，且这只在 Astra 和任务子集上成立。

### 跨任务迁移：经验能帮到别的任务吗？

最有意思的证据来自跨任务迁移：把"除目标任务外所有任务"的 workspace 汇合起来初始化新任务，首次成功明显提前——Sonnet 5.5 的 Cover Blocks 从 K9 提前到 K1，Opus 5.5 的 Organize Table 从 K14 提前到 K7，Deposit Coin 从 K15 提前到 K8。

![跨任务 workspace 迁移](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/manip.jpg)

> 图解：上半部分是各任务原始的自我改进轨迹（首次成功在 K6 或更晚的任务）；下半部分是用其他任务的 workspace 汇合初始化后的轨迹。红叉是失败尝试，深色对勾是首次成功，浅色对勾表示成功后不再重复评测。三条轨迹的首次成功都大幅提前，直观展示了迁移 workspace 带来的"热启动"效果。

这些可复用经验有两种形态： **可执行的例程（工具）** 和 **文本化的操作知识（记忆、技能笔记）**。不同系统的依赖模式截然不同：

- Sonnet 5.5 和 Opus 5.5 越来越多地把控制逻辑编码进自己写的工具里，后期超过 96% 的动作由工具生成；
- GPT-6 Sol 也建了个不小的工具库，但只有 67% 的动作走工具；
- Astra 和 GPT-6.1 Sol 约 99% 的动作仍由模型直接生成，只用极少的工具做验证——它们的改进更多体现在推理时间逐轮下降，以及反复查阅记忆/技能/策略笔记上。

![工具库与跨轮复用](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Where-is-Physical-RSI-capability-from-An-Observational-Scaling-Law-of-LLM-as-policy/assets/blog/tools_reuse.png)

> 图解：(a) 各模型的工具总数，以及感知/几何类工具与动作类工具的占比；(b) 工具谱系在后续轮次被复用的比例（直接使用、修改、派生都算复用）；(c) 复用广度——1 减去"被复用最多的前 10% 谱系所占的后续调用份额"，值越高说明复用分散在更多工具上。Opus 的谱系复用率 0.347，远高于 Sonnet 的 0.167 和 GPT-6 Sol 的 0.086；且 Opus 前 10% 谱系只占后续调用的 48%（Sonnet 76%、GPT-6 Sol 83%）——Opus 积累了更"宽"的可复用工具箱，而另外两家的复用集中在少数工具上。

工具使用还反过来印证了木桶假说：空间类工具占 Sonnet 调用的 14.5%、Opus 的 13.7%，而 Astra 总共只调用了 7 次工具（其中空间类 1 次）。Astra 几乎不需要这些模块，暗示其内在的感知与空间能力更强；Sonnet 和 Opus 反复跨任务复用这些模块，更像是在用工具补偿能力短板。

迁移的具体机制也很值得玩味。在 Organize Table 任务里，Opus 把 Pack Objects Into Box 任务中写的一个参数化移动工具，和 Store Tools In Toolbox 任务留下的一条记忆（"俯仰 30 度的抓取能够到 +y 半区……"）组合起来—— **工具提供操作，记忆提供"什么时候管用"的知识**。语义相似性帮 agent 找到候选经验，失败证据帮它判断该复用、改写还是丢弃。

## 局限性

作者对局限的交代相当坦率：

- 主比较只覆盖 6 个"模型 + 配置"系统，每任务仅一条改进链；K0 的三次重复无法估计完整链间的方差；
- 首次成功即停止，只记录一次成功，"可靠地重复执行"需要单独测试；
- Budget 和 Improvement 两轴基于各系统不同的条件任务集，跨系统不严格可比；
- 能力分析依赖公开 benchmark 分数（评测设置与本文不同），且只有 6 个系统、无独立验证——瓶颈与时机预测仍是待检验的假说；
- 消融和冻结测试只覆盖 Astra 和 6 个任务，跨任务迁移只有 3 个案例。

## 总结

- **Physical RSI 可以被拆开测量**：五轴指标（首次尝试、效率、成功、预算、改进）把"高分、快、省"解耦，Opus 分数高但增益晚、费预算，单一排行榜会掩盖这些差异。
- **能力瓶颈因改进侧面而异（物理木桶假说）**：空间推理最强预测开局（$\rho=0.94$），感知最强预测最终得分与失败恢复（$\rho=0.94$），编码与各轴的相关都最弱（$\rho\leq0.37$）——偏科的模型补不齐短板。
- **公开榜单压缩差异，物理评测放大差异**：GPT 家族公开均分跨度 24 分，物理五轴均分跨度近 60 分；堆尝试次数填不平能力差距。
- **改进是可预测的**：能力画像能完整复现跨系统的改进幅度与时机排序；用 Astra 的 Weibull 改进形状 + 5 轮早期观测预测 GPT-6.1 Sol 后续 12 轮，误差仅 0.019。
- **经验是真实的、且部分可迁移**：去历史后后期成功率 41.7%→8.3%；冻结 workspace 在未见布局上仍有约 42% 成功率；跨任务迁移让三个案例的首次成功提前 7–8 轮。场景特化与可复用经验可以共存。

展望来看，这篇工作立的是"假说"而非"定律"：6 个系统的观察性关联能否升格为因果结论，取决于后续定向能力干预实验；而"哪些被保留的技能与记忆真正支撑了迁移"这个问题，恰好把 Q3 绕回了 Q1——理解物理自我改进，最终还是要回到能力从哪来。

> 本文参考自 [Where is Physical RSI capability from? An Observational Scaling Law of LLM-as-policy](https://arxiv.org/abs/2610.26848)