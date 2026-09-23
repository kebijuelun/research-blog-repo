# RULER：用"实例感知评分细则"做奖励，让 RL 真正教会模型画好 SVG

> 论文：RULER: Instance-aware Rubric Rewards for SVG Generation
> 作者团队：蚂蚁集团、香港科技大学（广州）、牛津大学等
> 项目主页：https://hangyuran.github.io/RULER/

这篇文章主要解决的是一个非常"卡脖子"的问题： **开放式 SVG 生成既没有标准答案，也没有可靠的评分信号** 。没有可靠评分，评估测不准；没有可靠奖励，强化学习（RL）训不好。作者的思路很直接：先把"怎么评"这个问题解决掉，再把验证过的评分机制改造成 RL 的奖励信号，最终用一个 8B 模型在 SVG 生成质量上追平了体量远大于它的 DeepSeek-V3。

## 一、背景：SVG 生成为什么难在"评分"上

SVG（Scalable Vector Graphics，可缩放矢量图形）是一种特殊的"视觉代码"：它是结构化、可执行、可控的文本，渲染出来就是精确的图形。近年来 GPT-5、Gemini 2.5、Kimi K2.5 等前沿模型都把 SVG 生成当作展示视觉代码能力的舞台，围绕它的数据集、Benchmark 和专用训练范式也越来越多。

但从自然语言直接生成 SVG 有一个本质特性： **一条指令可以对应无数个语义上合法的渲染结果** ，不存在绝对的视觉 ground truth。这导致领域被两个问题同时卡住：

- **评估指标不可靠** 。CLIPScore、Aesthetic 分类器、HPS（Human Preference Score）这些指标都是在自然照片上校准的，迁移到风格化矢量内容上表现很差——它们经常给"坏掉的" SVG 打出比忠实还原的 SVG 更高的分数，与人类判断的相关性很弱。
- **训练监督信号不忠实** 。在"指令-SVG"配对数据上做 SFT 本质上只是模仿数据集模板，泛化差。RL 是更自然的替代方案，但 RL 的效果被奖励函数主导——而目前可用的奖励恰恰就是上面那些不可靠的指标，于是策略模型会朝着"最容易刷分的信号"漂移，而不是朝着"画得更好"漂移，这就是经典的 **Reward Hacking** （奖励投机）。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Instance-aware-Rubric-Rewards-for-SVG-Generation/figures/teaser-final.png)

> 图解：左图对比了评估指标的实际表现——标准的标量指标（CLIP 和 Aesthetic）会把一个渲染残缺的 SVG 排在忠实作品之前，或者根本无法惩罚它，而本文的 Rubric 分数与人类偏好一致；右图展示了本文设计的结构化六项评分细则（rubric），横跨语义、视觉、风格三条轴线。这张图直观地说明了"为什么老指标不能用、为什么要用 rubric"。

作者把相关工作里的奖励范式做了一次系统性梳理，用五个理想性质（视觉粒度、语义粒度、无需 GT、多轴、实例感知）来打分：

| 奖励范式 | 代表指标 | 视觉粒度 | 语义粒度 | 无需 GT | 多轴 | 实例感知 |
| --- | --- | --- | --- | --- | --- | --- |
| 基于像素 | SSIM / PSNR | 细粒度 | N/A | ✗ | ✗ | ✗ |
| 基于规则 | 代码长度 | N/A | N/A | ✓ | ✗ | ✗ |
| 基于 Embedding | CLIPScore | 粗粒度 | 粗粒度 | ✓ | ✗ | ✗ |
| 通用 Rubric | Universal Rubric | 细粒度 | 细粒度 | ✗ | ✓ | ✗ |
| **实例感知 Rubric（RULER）** | 本文 | **细粒度** | **细粒度** | **✓** | **✓** | **✓** |

可以看到，RULER 是唯一同时满足全部五条性质的方案，这也是它相比前人的核心创新点定位。

## 二、先解决"评得准"：Rubric 评分与人类判断对齐的实证分析

在构造奖励之前，作者先回答了一个基础问题： **让一个视觉语言模型（VLM）按多轴 rubric 打分，真的比标量指标更接近人类吗？**

他们收集了 900 个渲染好的 SVG 样本（分别由 Claude-Opus-4.6、Qwen3-32B、Qwen3-8B 三个能力不同的模型生成，各 300 个），并请 10 位标注员按 0-100 分给出整体质量评分（考虑指令保真度、视觉质量、构图、渲染质量、风格连贯性），之后还有 3 位校验员复核。然后从两个互补的角度检验自动指标与人类评分的一致性：

- **样本级分数相关性** ：用 Spearman 秩相关系数 $\rho$ 衡量。Rubric 评分达到 $\rho = 0.7929$，明显优于 Aesthetic（$\rho = 0.6051$）和 CLIP（$\rho = 0.5518$）。
- **成对排序一致性** ：用 Goodman-Kruskal Gamma 统计量 $\gamma$ 衡量"两个样本谁更好"的判断与人类是否一致。Rubric 达到 $\gamma = 0.7574$，大幅超过 Aesthetic（$\gamma = 0.5465$）和 CLIP（$\gamma = 0.5295$）。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Instance-aware-Rubric-Rewards-for-SVG-Generation/figures/human-study.png)

> 图解：左图是三种自动指标与人类打分的 Spearman 秩相关（越高越一致），Rubric 以 0.7929 领先；右图是成对排序的 Gamma 一致性，Rubric 同样以 0.7574 大幅领先。这组实验证明：把评估拆解成显式的多个维度，比把质量压缩成一个不透明标量更能跟踪人类感知。

值得一提的是，这个对齐实验用的评分器是 GPT-5-mini 加一份 **通用 rubric** ，与训练时用的奖励模型（Qwen3-VL-8B + 实例感知 rubric）是分开的，避免了"自己证明自己"的循环。

## 三、RULER 方法：把评分细则变成 RL 奖励

### 3.1 任务建模：奖励设计才是核心问题

作者把开放式 SVG 生成形式化为一个 **Token 级马尔可夫决策过程（MDP）** 。给定指令 $\mathcal{Q}$，策略模型 $\pi_\theta$ 自回归地生成 SVG 序列 $\mathcal{Y} = (y_1, \dots, y_T)$：第 $t$ 步的状态是 $s_t = (\mathcal{Q}, y_{<t})$，动作 $a_t = y_t$ 是采样的下一个 token，状态转移是确定性的拼接。序列结束后，渲染引擎 $\mathcal{E}$ 把代码渲染成图像 $\mathcal{I}_{\text{gen}} = \mathcal{E}(\mathcal{Y})$，奖励函数 $\mathcal{R}(\cdot)$ 在其上计算。

这个形式化点出了一个关键洞察：既然没有绝对视觉 ground truth， **奖励函数 $\mathcal{R}(\cdot)$ 的设计——而不是优化算法本身——才是这个 MDP 的核心问题** 。这也解释了为什么全文把大部分精力放在奖励设计上。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Instance-aware-Rubric-Rewards-for-SVG-Generation/figures/framework_final.png)

> 图解：RULER 框架总览。左侧：一个前沿模型（如 Claude-Opus-4.6）从每条文本指令出发，推导出一个实例感知 rubric——横跨语义、视觉、风格三条轴线的六个评分项。右侧：策略模型采样多条 SVG rollout，渲染后由 Judge VLM 逐项对照 rubric 打分，各项满意度按权重加权后形成 GRPO 的奖励信号，驱动策略更新。

### 3.2 实例感知 Rubric 的生成

RULER 用前沿模型 $\mathcal{M}_{\text{rub}}$（默认 Claude-Opus-4.6）为每条指令 $\mathcal{Q}$ 生成一份 **专属的六项评分细则** ，形式上定义为：

$$
C_\mathcal{Q} = \{(c_k, w_k)\}_{k=1}^{6}
$$

其中 $c_k$ 包含第 $k$ 项的实例化标题、描述和连续打分指南，$w_k$ 是重要性权重。六个评分项沿三条互补轴线组织：

- **Semantic Fidelity（语义保真）** ：概念整体可读性；主要部件、关键线索与指令特定关系的可见性。
- **Visual Quality（视觉质量）** ：轮廓与形体精炼度；构图与画布设计。
- **Rendering Style（渲染风格）** ：渲染完成度与执行洁净度；风格连贯性与设计感。

设计上有一个很讲究的取舍： **评分项描述的是"设计意图"层面，而不是像素或路径级别的精确约束** ——不能把 rubric 退化成一张"重建清单"，否则就破坏了一对多的开放式解空间。每一项都针对单一可观测维度、可独立从渲染图判断、惩罚一种独特的失败类型，保证多维度覆盖而不冗余。

**为什么 rubric 可以不依赖 GT？** 生成 rubric 时，模型会先"想象"并写出一份理想 SVG 作为中间质量锚点，再从中提炼出开放式的评分标准。Prompt 中明确禁止要求精确复现几何、位置、颜色、部件数量等实现细节（除非指令本身指定）。关键是：这份理想 SVG **只在构造 rubric 时用，RL 打分时完全不看它** ，因此整个流程不需要任何配对 ground truth。六项的固定权重为 $(5, 5, 5, 4, 5, 5)$，结构对所有指令共享，但标题、描述和打分指南逐条指令单独生成。

### 3.3 奖励计算与 GRPO 优化

**奖励计算。** 每个 RL 步，渲染图 $\mathcal{I}_{\text{gen}}$ 交给冻结的 Judge VLM $\mathcal{M}_{\text{judge}}$（Qwen3-VL-8B）。Judge 不打整体分，而是对照 rubric 逐项独立给出连续满意度 $s_k \in [0,1]$，最终奖励是归一化加权平均：

$$
\mathcal{R}(\mathcal{I}_{\text{gen}}) = \frac{\sum_{k=1}^{6} w_k s_k}{\sum_{k=1}^{6} w_k}
$$

这样就把一个不透明的标量换成了 **稠密的多维信号** 。

**组相对优势。** 这里有个很妙的配合：同一条指令的不同 rollout 往往在六个评分项上"各有各的短板"，产生的奖励天然多样——这正是 GRPO（Group Relative Policy Optimization）最擅长利用的结构。对每条指令采样 $G$ 条 rollout，各自渲染打分后，组内归一化得到优势：

$$
A_i = \frac{\mathcal{R}_i - \mathrm{mean}(\{\mathcal{R}_j\}_{j=1}^{G})}{\mathrm{std}(\{\mathcal{R}_j\}_{j=1}^{G}) + \epsilon}
$$

然后最大化 PPO 风格的裁剪代理目标：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{\mathcal{Q}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \rho_i A_i,\; \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \right) \right]
$$

其中 $\rho_i = \pi_\theta(\mathcal{Y}_i \mid \mathcal{Q}) / \pi_{\theta_{\text{old}}}(\mathcal{Y}_i \mid \mathcal{Q})$ 是序列级重要性比率。

### 3.4 训练与数据细节（来自附录）

- **训练框架** ：VERL + vLLM rollout 引擎，策略 backbone 为 Qwen3-8B；每条 prompt 采样 8 条 rollout，温度 1.0；训练 batch 128，学习率 $1\times10^{-6}$，熵系数 0.001，不使用 KL 正则；单机 8×H800 80GB。
- **训练数据** ：从 MMSVG-Icon 和 MMSVG-Illustration 训练集随机采样 20K + 12K 条 prompt，各生成一份实例感知 rubric。过滤分两步：先丢弃生成失败或格式不合格的 rubric；再用 Qwen3-VL-8B 对随 rubric 一起生成的理想 SVG 做"质检打分"，奖励低于 0.9 的 rubric 被移除。最终得到 **30,737 条** prompt-rubric 对。
- **渲染与评测** ：所有 SVG 用 CairoSVG 2.9.0 以 $512\times512$ 白底渲染；空白渲染用与纯白图的 MSE 阈值 $10^{-4}$ 检测；judge 并发 32、超时 180 秒，训练全程冻结。
- **成本** ：rubric 生成是一次性离线预处理。Claude-Opus-4.6 共处理 60.18M 输入 token、生成 68.49M 输出 token，总成本约 \$2,013， **平均每条约 \$0.0655** ——以 RL 数据构建的标准看相当便宜。

## 四、实验：五个研究问题的答案

实验围绕五个 RQ 展开：RQ1 与基线的整体对比；RQ2 奖励设计的对比；RQ3 rubric 各维度的消融；RQ4 跨基座/跨 rubric 生成器的鲁棒性；RQ5 定性分析。评测在 MMSVG-Illustration（插画）和 MMSVG-Icon（图标）两个 benchmark 上进行，指标包括 Token 数（效率）、CLIP、Aesthetic、HPS，以及作者引入的 **Rubric 分数** （用 GPT-5-mini 作为独立 VLM-Judge 按通用 rubric 打分，作为主要质量指标）。

### RQ1：主结果——8B 模型比肩 DeepSeek-V3

| 方法 | Illustration Rubric↑ | Icon Rubric↑ | Tokens |
| --- | --- | --- | --- |
| VectorFusion（扩散优化） | 0.611 | 0.510 | 31.4k |
| SVGDreamer（扩散优化） | 0.547 | 0.413 | 124.3k |
| Qwen3-8B（backbone） | 0.432 | 0.395 | 0.4k / 0.3k |
| Qwen3-32B | 0.651 | 0.681 | ~0.3-0.5k |
| DeepSeek-V3 | 0.686 | 0.673 | 0.5k / 0.2k |
| IconShop | 0.262 | 0.586 | 2.6k / 1.3k |
| OmniSVG-8B | 0.288 | 0.565 | 6.9k / 5.7k |
| JanusCoder-8B | 0.390 | 0.384 | 1.0k |
| **RULER（8B）** | **0.693** | **0.683** | 2.0k / 0.7k |

几个值得注意的点：

- **全面领先** ：RULER 在两个 benchmark 的主指标 Rubric 上都拿到第一，超过所有专用 SVG 模型（说明"堆 SVG 预训练"不足以补上开放式质量的差距），并 **与体量大得多的 DeepSeek-V3 打平甚至反超** 。
- **辅助指标不输人** ：虽然作者论证了 CLIP/Aesthetic 不足以单独评价 SVG，RULER 在这些传统指标上依然保持竞争力，说明 rubric 收益不是以牺牲其他轴为代价换来的。
- **效率优势** ：相比扩散优化方法动辄 31.4k~124.3k token、单样本平均 72.1 分钟的逐 prompt 迭代优化，RULER 一次自回归前向即可完成生成，且 Rubric 分数更高（Icon 上 0.683 vs 0.510）。
- **人类偏好验证** ：在 150 条 MMSVG-Bench prompt 上的盲测成对比较中，RULER 对全部五个基线的非平局胜率都超过 50%——对 Qwen3-8B 89.7%、Qwen3-32B 66.1%、VectorFusion 53.3%、OmniSVG 66.9%、JanusCoder 高达 96.5%。

### RQ2：奖励设计对比——实例感知是关键杠杆

固定 Qwen3-8B 基座和 GRPO 优化器，只换奖励信号：

| 奖励设计 | Illustration Rubric↑ | Icon Rubric↑ | 备注 |
| --- | --- | --- | --- |
| Zero-Shot（无 RL） | 0.432 | 0.395 | 基线 |
| C+A+H RL（CLIP+Aesthetic+HPS 标量组合） | 0.464 | **0.262** | 典型 reward hacking |
| Universal Rubric RL（通用 rubric） | 0.660 | 0.591 | 稳定提升但粒度粗 |
| **RULER（实例感知 rubric）** | **0.693** | **0.683** | 最强且均衡 |

这段实验信息量很大：

- **标量多指标奖励会塌缩成 reward hacking** ：C+A+H RL 把 Aesthetic 刷到了 6.697（远超所有其他变体），但 CLIP 从 0.244 掉到 0.196，Icon 上 Rubric 甚至从 0.395 崩到 0.262、低于 zero-shot 基线。在 Icon 上，策略为了骗过美学分类器，生成大量重叠笔触和重复装饰路径，平均序列长度从 0.3k 暴涨到 6.3k token，而真实视觉质量反而下降。 **没有维度拆解时，优化器会把概率质量集中到最容易刷的那个信号上** ——这正是 rubric 设计要解决的病理。
- **通用 rubric 有效但不够细** ：Universal Rubric RL 证明了"多轴拆解反馈"本身的价值（无严重 hacking），但同一份通用清单对所有指令一刀切，无法区分"极简图标"和"丰富插画"这类 prompt 特定的正确性标准，最终在 Illustration 和 Icon 上分别落后完整方法 0.033 和 0.092。
- **实例感知补齐了最后一块** ：把每条准则锚定到具体指令上，提供了细粒度、与 prompt 对齐的优化信号。

### RQ3：消融——三个轴缺一不可，且"严格"不等于"好"

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Instance-aware-Rubric-Rewards-for-SVG-Generation/figures/ablation.png)

> 图解：rubric 设计消融在 MMSVG-Illustration 与 MMSVG-Icon 上的 Rubric 分数对比。完整 RULER 最高；分别置零语义、视觉、渲染三类评分项后分数均下降，其中去掉 Visual Quality 跌幅最大；换成更严格的 Rubric-S 反而跌得最多。

- **去掉 Visual Quality** （轮廓/形体/构图两项）跌幅最大：Illustration 上 $0.693 \rightarrow 0.580$，说明这两条是 RL 最能帮到策略的地方。
- **去掉 Rendering** （工艺项）次之（$\rightarrow 0.599$），确认工艺维度提供了其他轴捕捉不到的互补信号。
- **去掉 Semantic Fidelity** 跌幅最小（$\rightarrow 0.623$），可能因为 Qwen3-8B 预训练已经具备较强的文图对齐能力，显式奖励语义的边际收益更小。
- **Rubric-S 的反面教材** ：作者设计了一个更严格的变体（五项结构导向条目 + 一个权重为 $-2$ 的固定"文字捷径"惩罚项），结果 Rubric 反而掉到 0.536，比任何单轴消融都惨——而且诱发了前所未见的 **text-hint hacking** ：策略开始把 prompt 里的关键词作为可读文字直接画进 SVG 里，哪怕有显式惩罚也拦不住。结论很微妙： **互补的轴至少和打分严格性一样重要** ，改动 rubric 规格本身就可能重新打开退化的优化通道。

### RQ4：鲁棒性——不挑基座，也不挑 rubric 生成器

- **跨基座规模** ：把策略换成更小的 Qwen3-4B，RULER-4B 仍把 Rubric 从 $0.372/0.338$（Illustration/Icon）提升到 $0.561/0.559$，逼近 Qwen3-32B 的 $0.585/0.566$；RULER-8B 的 $0.692/0.662$ 则双双反超 32B。五次独立随机种子的推理显示方差很小（如 $0.692 \pm 0.007$），收益稳定可复现。
- **跨 rubric 生成器** ：把生成 rubric 的模型从 Claude-Opus-4.6 换成 GPT-5.5，重新训练同样的 Qwen3-8B 策略，Rubric 仍达 $0.646 \pm 0.007$（vs Claude 版 $0.662$），远超基座的 $0.394$——说明方法不绑定某个特定的 rubric 生成器。

### RQ5：定性对比

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Instance-aware-Rubric-Rewards-for-SVG-Generation/figures/ruler_visulization.png)

> 图解：RULER 与四个基线在代表性图标 prompt 上的定性对比。可以看到 RULER 更好地保留了 prompt 特定的细节——睫毛、蜂窝结构、东北朝向、星形放射、同心圆环等——同时配色更丰富、构图更完整；部分基线则渲染失败或遗漏关键细节。

定性观察与实例感知 rubric 的设计初衷一致：逐条指令定制的评分项会"盯着"语义保真、视觉质量和渲染风格的具体细节打分，策略自然学会保住这些细节。相对自家 backbone Qwen3-8B，提升肉眼可见，示例上的视觉质量与 DeepSeek-V3 相当。此外 RULER 的渲染成功率也很高（Icon 99.3%、Illustration 100%），聚合指标几乎不受失败样本影响。

## 五、总结与局限

这篇文章的贡献可以归纳为三件事：

1. **评估层面** ：系统性揭示了标准标量指标在 SVG 领域偏移下的严重失灵，并实证证明多轴 rubric 评分与人类偏好显著更对齐（$\rho = 0.7929$，$\gamma = 0.7574$）。
2. **训练层面** ：提出 RULER，把每条指令转换成横跨语义、视觉、风格的六项实例感知 rubric，作为 GRPO 的稠密奖励，把模糊的视觉判断变成显式可验证的子目标；全程无需配对 SVG ground truth、无需人类偏好标注，可扩展到任意无标注指令集。
3. **效果层面** ：在 MMSVG-Illustration/Icon 上把 Rubric 分数从 $0.432/0.395$ 提升到 $0.693/0.683$，超过所有专用 SVG 模型、追平 DeepSeek-V3，并在奖励设计对比和消融中确认 **rubric 设计本身是开放式视觉代码 RL 的核心杠杆** 。

作者也坦诚列出了局限：其一，框架依赖两个外部模型（生成 rubric 的前沿模型 + 打分的 Judge VLM），奖励质量会继承它们的偏置与失败模式，对训练分布外的 prompt，judge 可能高估表面线索、低估微妙风格；其二，RL 成本升高——每步都要渲染采样 SVG 并查询 judge，比 CLIP 或启发式标量奖励贵得多，限制了向更大模型和更大规模超参搜索的扩展；其三，"语义-视觉-风格"三轴分解虽然好用，但未必能覆盖所有艺术意图和领域偏好，交互式、人类可引导、领域自适应的 rubric 设计留作未来工作。

> 本文参考自 [RULER: Instance-aware Rubric Rewards for SVG Generation](https://arxiv.org/abs/2609.25270)