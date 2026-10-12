# On the estimation and validity of AI time horizons：统计学重审 METR 时间线

METR 的"50% 时间线"（time horizon）是当下衡量 AI 能力进步最出圈的指标：它表示"AI 有 50% 概率独立完成的任务，人类工程师需要多久做完"。这张图被《纽约时报》类比为 AI 界的摩尔定律，也深度影响着 AI 政策与投资讨论。但问题是：时间线并不是物理测量量，而是一个统计估计量——它估得准吗？它真的度量了我们想度量的东西吗？UC Berkeley 统计系的这篇文章用样条（spline）和项目反应理论（IRT）在 METR 公开的 228 个任务、26 个 AI 数据上重新估计，发现"人类用时"与"AI 难度"之间 **并非对数线性关系**：在 2–30 分钟区间几乎是一条平线。这意味着时间线从 3 分钟涨到 30 分钟，远比从 30 分钟涨到 5 小时容易——尽管两者同样是 10 倍。作者的新估计在 12 项交叉验证评分指标上 **全面优于** METR 基线。

![Summary of contributions](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/headline.png)

> 图解：本文贡献总览，共六个子图。 **(a)** 用共享斜率 logistic 方法复现的 METR 时间线图，横轴是 AI 发布日期，纵轴（对数刻度）是 50% 时间线，呈现指数增长； **(b)** 基线方法的条件成功率图，纵轴是不同人类用时任务上的成功概率（logit 刻度），虚线是 50% 线，时间线可从交点读出； **(c)** GPT-4o 的样本内拟合对比，Model 2（紫）比基线（蓝）更贴近分箱观测频率（黑点）； **(d)** 时间-难度转换图：228 个点各代表一个任务，横轴是人类用时，纵轴是估计的 AI 难度（标准化单位），可见 2–30 分钟区间明显变平，而基线假设为线性（虚线）； **(e)** Model 2 给出的新时间线估计，阴影区（2–30 分钟）内估计变化最大、也恰恰是改进最显著的区域； **(f)** 基于 Model 2 的条件成功率图，评分规则评估表明它比 (b) 更好地概括了数据。

## 一张影响政策与投资的图，到底在估计什么

先把背景说清楚。METR 是伯克利的一家研究机构，被《纽约时报》称为"世界非官方 AI 裁判"。他们的做法分两步：

1. 手工设计 228 个软件工程任务（如"实现这个函数""训练这个分类器"），每个任务标注一个 **human time** ——专业软件工程师成功完成它所需的时间（多次成功尝试取几何平均；约 29% 的任务测不到有效值，用专家估计代替）。
2. 每个主流 AI 发布后，让它自主重复尝试这些任务，统计成功概率。某个 AI 的 **50% 时间线** ，定义为"它能以 50% 概率解决的任务所对应的人类用时"。

例如 GPT-4 的时间线约为 4 分钟：一个让人类花 4 分钟的任务（比如"Google 某个事实"），GPT-4 独立完成的成功率约 50%。时间线越长，能"够得着"的任务越长。

这张图 2025 年 3 月发布后逐点更新，直到 2026 年 5 月的 Claude Mythos，呈现出一条漂亮的指数曲线。它被类比为摩尔定律和 Keeling 曲线（大气 CO₂ 浓度曲线），甚至成为 Yoshua Bengio、Bernie Sanders 等人呼吁放缓 AI 发展的论据之一。但作者点出关键区别： **摩尔定律和 Keeling 曲线的纵轴是物理量，而时间线的纵轴是统计估计量** ——它依赖于模型假设，可以被批评，也可以被改进。

这篇文章回答两个问题：

- METR 图想估计的量到底是什么？能不能估得更好？
- 时间线这个指标本身有多大用（即构念效度，construct validity）？如何诊断？

## 统计框架：时间线是成功曲线的反函数

作者首先把问题形式化。设任务从某个任务分布 $\mathcal{D}$ 中抽取，每个任务 $i$ 有人类用时 $T_i$；固定 $J$ 个 AI，第 $j$ 个 AI 在任务 $i$ 上做 $n_{ij}$ 次独立尝试，结果为 $Y_{ijr} \in \{0,1\}$，单次成功概率为 $p_{ij}$。

定义 AI $j$ 的 **成功曲线** （success curve）为给定人类用时的边际成功概率：

$$
p_{j}(t) = \mathbb{E}_{\texttt{task}_i \sim \mathcal{D}}[Y_{ij1} \mid T_i = t]
$$

假设它关于 $t$ 严格单调递减（人做得越久的任务，AI 成功率越低），则其反函数存在，于是 $q$-时间线定义为：

$$
\mathbf{th}_{q}(\texttt{AI}_j) := p_{j}^{-1}(q)
$$

即"AI $j$ 能以概率 $q$ 解决的任务的人类用时"。取 $q = 0.5$ 就是 METR 的招牌指标。 **整个估计问题因此归结为：估好每个 AI 的成功曲线 $p_j(t)$，再取反函数。** 时间线图无非就是散点 $(D_j, \widehat{\mathbf{th}}_{0.5}(\texttt{AI}_j))$ 按发布日期 $D_j$ 排列。

一个值得注意的细节：发布日期 $D_j$ 在建模中完全没有用到，只用于画图。也就是说，观察到的指数增长完全是数据里长出来的，作者并不质疑这一点——他们质疑的是纵轴每个点的估计质量。

## 四种模型设定：从对数线性到解释性 IRT

作者在同一表格数据（AI 编号、任务编号、run 编号、成败指示、人类用时）上比较了四种成功曲线的估计方法，前两种来自 METR，后两种是本文贡献。

### 基线：共享斜率的 logistic 回归

METR 的原始模型对每个 AI 单独做 logistic 回归：

$$
\log\left(\frac{p_j(t)}{1-p_j(t)}\right) = \alpha_j - \beta_j \log_2(t)
$$

后来的 METR 研究笔记发现，让所有 AI 共享同一个斜率 $\beta$（只保留各自的截距 $\alpha_j$）在交叉验证中表现更好，作者复现并确认：共享斜率版在 12 项指标里赢了原始版 11 项，因此以它作为 **baseline** 。图 1(a) 的时间线就来自这个拟合。

这个设定的核心假设是： **AI 难度是人类用时的对数线性函数** 。人用时每翻一倍，任务难度的 logit 就下降固定量。这正是本文要放松的假设。

### Model 1：共享单调样条

第一个改进很直接：把 $\log_2 t$ 上的线性项换成一个所有 AI 共享的单调函数 $f$：

$$
\log\left(\frac{p_j(t)}{1-p_j(t)}\right) = \alpha_j - \beta\, f(\log_2 t), \qquad \beta > 0
$$

$f$ 取带 4 个内节点的单调自然三次 I-spline（系数非负保证单调），并固定两端取值以消除仿射不可辨识性。METR 自己试过样条但没赢过基线——作者指出原因大概是他们给每个 AI 配了 **单独的** 样条，而共享样条才能借到跨 AI 的信息。

这个共享形状 $f$ 有明确解释：它就是"人类用时 → AI 难度"的转换函数，对所有 AI 通用。

### Model 2：带家族效应与过度离散的解释性 IRT

这是本文的主力模型，一个对全部 $Y_{ijr}$ 的完全生成模型。直觉上，它把每个任务看作自带一个 **潜难度** $\theta_i$，每个 AI 自带一个 **能力** $\alpha_j$，成功率由二者之差决定：

$$
\log\left(\frac{\tilde p_{ij}(\theta_i)}{1 - \tilde p_{ij}(\theta_i)}\right) = \alpha_j - \beta \theta_i
$$

潜难度再被人类用时、任务家族等协变量"解释"：

$$
\theta_i = f(\log_2 T_i) + \xi_{F(i)} + \varepsilon_i
$$

其中 $f$ 仍是共享单调样条；$\xi_F$ 是任务家族随机效应（同一家族的任务难度相关，方差 $\tau(\cdot)^2$ 随家族平均用时变化）；$\varepsilon_i$ 是任务特异残差（方差 $\sigma(\cdot)^2$）。方差函数 $\sigma(\cdot)$、$\tau(\cdot)$ 各取带 2 个内节点的自然三次样条的指数。

此外还有一个 **过度离散** （overdispersion）设计：同一 AI 在同一任务上的多次尝试并非条件独立。作者给出的证据很硬——若按基线的独立假设，约 60% 的 runset 应该是全对或全错，实际却是 **83%**。模型用一个 Beta-Binomial 结构引入相关系数 $\rho$：

$$
p_{ij} \mid \theta_i \sim \operatorname{Beta}\big(\tilde p_{ij}(\theta_i)\,\varphi,\ (1 - \tilde p_{ij}(\theta_i))\,\varphi\big), \qquad \varphi = \tfrac{1-\rho}{\rho}
$$

这样 $\mathrm{Corr}(Y_{ijr}, Y_{ijr'} \mid \theta_i) = \rho$，重复次数多的 AI-任务对会被自动降权，因为它们携带冗余信息。模型用 EM 算法配合 Gauss-Hermite 求积做最大似然拟合。

最终的成功曲线要对随机效应做边际化：

$$
p_j(t) = \mathbb{E}\left[\left(1 + \exp(-(\alpha_j - \beta\theta))\right)^{-1}\right], \qquad \theta \sim \mathcal{N}\!\left(f(\log_2 t),\ \sigma(\log_2 t)^2 + \tau(\log_2 t)^2\right)
$$

数值求积后再反解出时间线。笔者认为这个模型的聪明之处在于： **它没有用 METR 那种启发式的 sqrt-family 重加权去"打补丁"，而是把家族相关性直接写进生成模型** ，相关结构本身成为被估计的对象。

## 怎么评判谁估得好：proper scoring rules 全家桶

比模型不能只看某一条曲线顺眼，作者搭了一套 5 折交叉验证（按 79 个任务家族折分）× 4 种 proper scoring rule × 3 种加权方案的评估体系，共 12 项指标。

四种评分分别是：

- **Marginal log score** ：即二元交叉熵，只通过 $p_j(t)$ 依赖模型；
- **Brier score** ：预测概率与结果的平方误差；
- **Elementary binary score** ：在阈值 $q$ 处只惩罚"时间线判错边"的错误， **只通过时间线 $\mathbf{th}_q$ 依赖模型** ——它直接度量时间线估计本身的质量。由于单点 $q$ 噪声大，作者在 $[q-0.05, q+0.05]$ 窗口内取平滑，报告 $q = 0.5$ 和 $q = 0.8$ 两档；
- **Full log-likelihood** ：仅对生成模型可用，评估包括相关结构在内的整个模型。

一个理论上有趣的事实：所有二元结果的 proper scoring rule 都是 elementary score 的非负混合（特别地，Brier 分数满足 $S_{\mathrm{BR}} = 2\int S_q\, dq$），因此把 $q \mapsto S_q$ 画成曲线——即 **Murphy diagram** ——若一条曲线在每个 $q$ 处都低于另一条，就意味着在 **所有** proper scoring rule 下占优。

三种加权方案是 equal-task（$w_{ij} = 1/n_{ij}$）、equal-family（$1/(n_{F(i)} n_{ij})$）、sqrt-family（$1/(\sqrt{n_{F(i)}}\, n_{ij})$，METR 自己用的折中方案）。它们回答的是"我们希望在哪类预测问题上平均"这个规范性问题。

## 实验结果：12 项指标全面占优，改进集中在中等 AI

![Scoring rule suite](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/score_grid.png)

> 图解：Model 1 / Model 2 相对基线在全部评分指标上的配对差值（每行一种评分：marginal log score、Brier、平滑 elementary score $\tilde S_{0.5}$ 与 $\tilde S_{0.8}$；每列一种加权方案）。点值在 0 左侧表示优于基线，粗/细区间分别为 80% / 95% 置信区间（以任务家族为聚类单元，$n = 79$）。两个新模型的 **点估计在所有 12 项指标上全部占优** ；部分区间跨零，说明单指标证据有限，但方向的一致性本身很有说服力。

![Murphy diagrams](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/murphy_diagrams.png)

> 图解：Murphy diagram（equal-family 加权）。左图横轴是阈值 $q$，纵轴是 elementary score $S_q$，越低越好；右图是各模型与基线的配对差（$\times 10^{-3}$）及 95% 置信带。$q > 0.4$ 后基线几乎被全程压制——这解释了为什么 Brier 和 log score（它们都是 Murphy 曲线的加权积分）也一致偏向新模型。

![Success curves for six AIs](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/success_curves_band_ais.png)

> 图解：六个时间线在 2–15 分钟的 AI 的样本内拟合诊断。黑点是把每个 AI 的 run 按 4 倍人类用时分箱后的观测频率（竖线为 95% Wald 区间），三条线是三种模型拟合的 $p_j(t)$。可以看到基线（log-linear）被迫直线穿过，而 Model 1/2 弯曲着贴合 2–30 分钟的平坦区。对这些 AI，50% 时间线的新估计变化显著，有些甚至落出 METR 原图的 95% 置信区间。

附录还给出逐 AI 的细分证据，说明改进的来源非常集中：

![Per-AI contrasts Model 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/per_ai_contrasts_model1.png)

> 图解：Model 1 相对基线的逐 AI 评分差（26 个 AI 按发布时间排列，点为配对差，负值偏好 Model 1，区间是 95% 聚类稳健区间）。橙色标出的是时间线 2–15 分钟的六个 AI——正是受平坦区影响最大的一批，改进几乎全集中在这里。

![Per-AI contrasts Model 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/per_ai_contrasts_model2.png)

> 图解：Model 2 的同款逐 AI 对比（紫色标出同一批中等能力 AI）。单个 AI 的区间大多宽到无法与零区分，优势要汇总到图 2 的整体比较中才积累起来——这正是要看 pooled 证据而非单点的原因。

![Success curves for all AIs](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/success_curves_all_ais.png)

> 图解：全部 26 个 AI 的成功曲线拟合对照。规律清晰：log-linear 基线的形状失配集中在 GPT-4 时代的中等强度 AI；最弱 AI 的差异主要在尾部；最强 AI 上三种设定几乎重合。平坦区只"咬"中间那段。

## 构念效度：时间线这个数字，什么时候有意义

解决了"怎么估得更准"，下一个问题更根本： **估得再准，时间线这个量本身有用吗？** 作者把这类做法抽象为"能力锚定"（capability anchoring）——用 IRT 学出任务难度与 AI 能力，再把难度回归到一个外部可解释的标注（这里是人类用时）上，最后把 AI 能力"翻译"成标注单位。Lexile 阅读分级框架是同一思路的先例。

他们提出一份五问清单来判断构念效度：任务代表性、难度-能力交互建模是否充分、标注的客观含义、 **可预测性** （predictiveness）、 **可比性** （comparability）。本文聚焦后两问：

- **可预测性** ：如果 AI A 和 B 的时间线分别是 $T_A \ll T_B$，但在整个 $[T_A, T_B]$ 区间里两者的成功曲线都很平、几乎贴在一起，那么"时间线大涨 = 能力大涨"的解读就是误导。
- **可比性** ：时间线从 3→30 分钟和从 30 分钟→5 小时，在图上是一样高的台阶（都是 10 倍），但如果时间与难度的关系到处不一样，这两个台阶的含义就不同。可比性问题此前几乎没人讨论过。

### 诊断工具一：时间-难度转换图

回到图 1(d)。Model 2 拟合后，每个任务的难度取后验均值 $\hat\theta_i = \mathbb{E}[\theta_i \mid \mathbf{Y}, \mathbf{T}]$（综合了所有 AI 的表现和先验），紫色曲线是 $f(x) \pm s(x)$，虚线是 $\text{SNR} = 1$ 参考线。看这张图的方式是：

- 曲线整体与 SNR = 1 线相当或更陡 → 可预测性总体良好（METR 软件任务确实如此）；
- 曲线局部变平 → 该区域可比性出问题。2–30 分钟区间正是如此：这段任务的 AI 难度几乎相同，时间线在其中跳动没有多少能力含义。

### 诊断工具二：条件成功率轨迹图

图 1(b) 与 (f) 的读法是：固定若干人类用时（15 秒、1 分钟、4 分钟、15 分钟、1 小时、4 小时、16 小时），画出各 AI 在该时长任务上的成功概率随发布日期的变化。 **曲线之间的间距反映可预测性，间距的变化反映可比性问题。** 在拟合更好的 (f) 中，4 分钟曲线和 15 分钟曲线几乎黏在一起——沿着 50% 虚线从 4 分钟时间线的 AI 读到 15 分钟时间线的 AI，能力差距小得可怜。

### 一个意外发现：METR 图也许一直在画 Rasch 能力

最有意思的一笔在最后。三个模型都直接估计 AI 能力参数 $\alpha_j$（类比 Rasch 模型中的 ability，作者称之为 Rasch abilities）。把 Model 2 的 $\hat\alpha_j$ 做仿射缩放后叠到 METR 原图上：

![Model 2 over METR plot](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/model2_over_metr_plot.png)

> 图解：两图都以 METR 原始时间线（蓝）为底。 **(a)** Model 2 自己的时间线估计（紫），与 METR 的分歧集中在阴影的 2–30 分钟带内； **(b)** Model 2 的 Rasch 能力 $\hat\alpha_j$ 经最小二乘仿射缩放后（绿，$R^2 = 0.996$）几乎与 METR 原图 **完全重合** ——比 Model 2 自己的时间线还像。

这个巧合的解释是：Model 2 的 50% log 时间线是 $f^{-1}(\alpha_j/\beta)$，而线性模型直接估 $\alpha_j/\beta$。平坦区只占整个任务范围的一小段，而 $\alpha_j$ 的估计由 AI 在 **所有** 任务上的表现决定，所以线性假设"平均而言错得不远"，估出的 $\alpha_j/\beta$ 依然很好。但作者强调：如果 $f$ 真的不是线性的，那这个量就不该叫"时间线"——它度量的是另一个（可能同样有价值的）能力构念。

## 讨论：时间线方法的未来

![Hypothetical future conversion](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/On-the-estimation-and-validity-of-AI-time-horizons---a-statistical-look-at-the-METR-plot/figs/conversion_future.png)

> 图解：一张假想图。一天以内的部分就是图 1(d) 的实测结果；超过一天后，作者画出一种可能的未来形态——时间-难度转换曲线再次出现异常的平坦或陡峭段。线性模型的常规检查 **发现不了** 这种结构，而本文的诊断图可以。

作者的提醒很实际：现有任务最长约 30 小时，一旦基准扩展到更长任务（或迁移到机器人等其他领域），没有理由假设近似线性关系还能保持；未来的 $f$ 完全可能再出几个"平台"。此外，把能力锚定用于公众沟通（比如评估 AI 对整体经济的影响）时，最好的任务标注未必是"人类用时"——选什么标注、如何检验其构念效度，应该成为每个新基准的必修课。

## 总结

- METR 时间线把 AI 能力锚定在"人类用时"上，直观但本质是统计估计量，其假设可以被检验和改进；
- 用共享单调样条（Model 1）和解释性 IRT（Model 2）放松"对数线性"假设后，新估计在 12 项交叉验证 proper scoring rule 指标上全面优于基线；
- 时间-难度转换函数在 2–30 分钟区间明显变平：3→30 分钟的 10 倍跳跃，能力含义远小于 30 分钟→5 小时的 10 倍跳跃；
- 本文提供两张诊断图（转换图 + 条件成功率轨迹图）分别检验可预测性与可比性，建议未来所有时间线类基准配套使用；
- Model 2 的 Rasch 能力经缩放后与 METR 原图几乎重合（$R^2 = 0.996$），暗示原图估计的可能更像"能力"而非"时间"。

展望：这项工作是对 METR 式基准的建设性批评——随着任务套件向更长任务和新领域扩张，"标注是否依然能线性兑换成难度"将成为每个时间线数字背后必须回答的问题。

> 本文参考自 [On the estimation and validity of AI time horizons---a statistical look at the METR plot](http://arxiv.org/abs/2610.12466v1)