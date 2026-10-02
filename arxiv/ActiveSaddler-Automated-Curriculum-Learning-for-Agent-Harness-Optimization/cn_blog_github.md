# ActiveSaddler：当 Agent 的"外挂骨架"也要上课表——用自动课程学习优化 Harness

现有的 Agent Harness 优化方法（自动改 Prompt、工具接口、控制逻辑）大多只回答"怎么改"，却把"用哪些训练任务来驱动修改"当作一开始就定死的事情。这篇文章把被忽略的这一维度形式化为 **自动课程学习（Automated Curriculum Learning）** 问题，提出 ActiveSaddler：把课程决策建模为一个臂集合动态扩张的非平稳 Bandit，让训练课程与 Harness 共同进化。结果相当硬：在 GAIA2 和 Terminal-Bench 2.0 上，相比同一个优化器配固定任务顺序，测试 Pass@1 分别提升 **+4.4 个百分点（59.8%）** 和 **+7.5 个百分点（80.0%）** ，在 TB2 上甚至超过人工精调的 Terminus-KIRA Harness 10.8 个点。

## 一、背景：Harness 优化只解决了"怎么改"，没解决"练什么"

LLM Agent 的可靠性很大程度上不取决于模型本身，而取决于包在模型外面的那层 **Harness** ——它规定了 Agent 怎么被提示、能用什么工具、执行过程如何被监控和纠错。Prompt、工具接口、中间件、运行时控制逻辑，都属于 Harness 的参数。

问题是，手工设计的 Harness 会随着模型升级、任务分布变化而失效，于是出现了 **自动 Harness 优化（AHO）** ：跑训练任务、从执行轨迹里诊断缺陷、给 Harness 打补丁，如此循环。

但博主注意到一个被广泛忽略的点：这些方法在执行训练任务时，任务的选择顺序基本是 **优化开始前就定死的** ——要么全量跑，要么按随机打散或预设难度顺序的 mini-batch 走。这带来两个局限：

- **无法响应"修复进度"的变化** 。随着 Harness 不断被修补，未解决的失败值得继续投入，已修复的失败则没什么剩余价值。固定课程表对此无感：可能过早离开一个没修好的弱点，也可能把宝贵的 rollout 浪费在已经没有学习价值的任务上。论文的实证也印证了这一点：固定顺序优化下，训练中观察到的很多失败到最终 Harness 里依然没解决。
- **无法调节"温故"与"知新"的平衡** 。当已知失败里还有可修复的弱点时，应该继续回头修（exploit）；当已知目标所剩无几时，应该去跑没见过的任务、发现新弱点（explore）。这个相对价值随优化进程不断变化，固定课程表同样无从适应。

由于每次任务执行（rollout）都是真金白银的 LLM 调用，预算有限时，"练什么"和"怎么改"对最终 Harness 质量的影响是同等量级的。这就是这篇文章要补上的维度。

## 二、问题形式化：把课程选择建模为"非平稳 Bandit"

### Harness 优化的标准设定

沿用 AutoSaddler 的离线学习式优化设定：Harness 记为 $H_\theta$，参数 $\theta$ 涵盖 Prompt、工具接口和运行时控制逻辑。对任务输入 $x$，执行产生轨迹与答案 $(\tau, \hat{y}) \sim P_\theta(\cdot \mid x)$，Harness 的总体性能为：

$$
J(\theta) = \mathbb{E}_{(x, y^\ast) \sim \mathcal{T}} \, \mathbb{E}_{(\tau, \hat{y}) \sim P_\theta(\cdot \mid x)} \left[ \mu(\hat{y}, y^\ast) \right]
$$

每轮迭代中，优化器 $\mathcal{O}$ 拿到一批训练任务 $X_t \subseteq D_{\mathrm{train}}$ 的执行记录 $\mathcal{E}_t$，结合开发集 $D_{\mathrm{dev}}$ 做候选选择、回归检查和泛化检查，产出下一轮 Harness：$\theta_{t+1} \sim \mathcal{O}(\theta_t, \mathcal{E}_t, D_{\mathrm{dev}})$。

注意这个分工： **$X_t$ 决定"收集什么证据"，$\mathcal{O}$ 决定"证据怎么变成更新"** 。以往工作全在研究后者，ActiveSaddler 研究前者。

### 课程学习目标

把当前 Harness 视为"正在进化的学习者"，课程策略 $\pi$ 根据历史信息 $\mathcal{I}_{<t}$（当前 Harness + 过往执行与更新记录）选择下一批任务：$X_t \sim \pi(\cdot \mid \mathcal{I}_{<t})$。在共享的固定 rollout 预算下，目标是让最终选出的 Harness 期望性能最大：

$$
\pi^\ast \in \arg\max_{\pi} \, \mathbb{E} \left[ J(\hat{\theta}_\pi) \right]
$$

其中 $\hat{\theta}_\pi$ 是按开发集性能从优化轨迹中选出的最终 Harness，测试集只用于最后报告。

### 为什么是 Bandit，而且是非平稳的？

每一轮要在多个候选优化目标之间分配一次宝贵的优化机会——这是典型的序列资源分配问题，天然对应 **Multi-Armed Bandit** ：每个臂（arm）代表一个优化目标，拉一次臂就把下一轮优化指向它。

但和传统课程学习不同，这里的"臂"有特殊性：

- **臂无法预先定义** 。一个训练任务并不天然标明它对应哪个 Harness 弱点——弱点要执行后诊断才能暴露；同一个任务在 Harness 变化后可能暴露不同的失败；同一个弱点又会横跨多个任务反复出现。
- **臂没有现成的学习信号** 。一个诊断出的弱点没有 loss、TD error 之类的指标告诉你"它还值不值得再修"，只能从当前 Harness、历史修复尝试和执行记录里推断。
- **臂集合是动态扩张的** 。新失败模式随优化不断被发现，臂池从空开始一路长大。

所以 ActiveSaddler 的做法是： **从诊断出的失败里在线构建"失败模式臂"，持续重估每个臂的优化价值，并在"回头修已知弱点"与"探索未见任务"之间自适应分配预算** 。这个设计的聪明之处在于，它不要求底层优化器做任何改动——课程层是一个即插即用的外壳。

## 三、ActiveSaddler 方法详解

![ActiveSaddler 总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/main_figure.png)

> 图解：ActiveSaddler 的单轮迭代流程。每轮开始时，系统持有当前 Harness $H_t$、优化历史 $\mathcal{I}_{<t}$、已实例化的臂集合 $\mathcal{A}_t$ 和未见任务池 $U_t$。① Exploration Controller 先决策：本轮去探索未见任务，还是继续优化某个已知失败模式臂；② 若探索，从 $U_t$ 采样组成批次 $X_t$；③ 否则由 Arm Prioritizer 给所有臂打分并采样一个臂 $a_t$，用它的支持任务集组成 $X_t$；④ 选中的任务在 $H_t$ 下执行，执行记录交给底层 Harness 优化器产出 $H_{t+1}$；⑤ Failure-Pattern Extractor 从失败记录中抽取失败模式，匹配到已有臂或实例化新臂；⑥ 更新后的 Harness、历史、臂集合与未见池共同构成下一轮的课程状态。

三个核心组件，下面逐一拆解。

### Failure-Pattern Extractor：把失败抽象成可复用的"臂"

这是整个框架的地基。它的输入是底层优化器在本轮暴露的失败执行记录 $\mathcal{F}_t$ 及诊断证据 $\delta_t(e)$（在 AutoSaddler 实例化中，既包括打补丁前的失败，也包括打补丁后仍失败或新回归的病例），输出是更新后的臂集合。

抽取分两个阶段，且顺序有讲究：

1. **症状抽取（Symptom Extraction）** ：对每条失败独立抽象出一个候选症状 $c_t(e) = g_{\mathrm{ext}}(e, \delta_t(e))$， **不许看现有臂集合** ——这是为了避免锚定偏差，防止 agent 把新失败硬塞进旧模式里。
2. **模式归一化（Pattern Normalization）** ：所有候选症状抽完后，再与现有臂做 **三路判定** ：

$$
\mathcal{A}_{t+1} = g_{\mathrm{norm}} \left( \mathcal{A}_t, \{ c_t(e) \}_{e \in \mathcal{F}_t} \right)
$$

  - (a) **Same** ：匹配到已有原子模式 → 挂载到该臂；
  - (b) **Composition** ：是多个已有原子模式的组合 → 同时打上多个标签；
  - (c) **New** ：确实是新失败类型 → 注册为新臂。

每个臂维护：症状级模式描述、支持任务集 $S_t(a) \subseteq D_{\mathrm{train}}$，以及观察到该模式时的 Harness、轨迹与诊断证据。抽取原则要求 **原子性** （一个补丁能修一个模式）与 **适度抽象** （比单个 root cause 抽象，但足以区分不同失败类型；root cause 本身保留为证据，不直接当模式）。

工程实现上，所有臂存在一个 **Pattern Registry** 里，agent 通过一个 `pattern` CLI 按需访问（`pattern list / show / score / history` 等读操作，`register / tag / rate / decide` 等写操作）。之所以不直接把注册表塞进 prompt，是因为随着迭代积累，臂的观测历史和证据会膨胀到无法紧凑序列化——CLI 提供了按 session 权限裁剪的分视图访问。这是个务实的工程决策，值得做 Agent 系统的同学参考。

### Arm Prioritizer：给每个臂估"学习进度分"

当决定继续优化已知臂时，该拉哪一个？理想标准是该臂被选中后带来的总体性能期望提升，但这个量不可观测。ActiveSaddler 用 LLM 估计一个学习进度分：

$$
\phi_t(a) = g_{\mathrm{select}}(a, \mathcal{I}_{<t}) \in [0, 1]
$$

LLM 被要求沿四个轴打分（各在 $[0,1]$ 内）：

- **Severity（严重性）** ：该失败模式在当前 Harness 下是否仍然活跃；
- **Fixability（可修复性）** ：是否能通过 Harness 干预解决、在哪里下手；
- **Breadth（广度）** ：修好后能泛化到多广的任务面；
- **Side-effect Risk（副作用风险）** ：补丁引入回归的风险。

最终分数为四轴均值（副作用取反）：$\phi = (\text{severity} + \text{fixability} + \text{breadth} + (1 - \text{side\_effect})) / 4$。

博主认为这里有一个细节设计得很专业：评分 prompt 明确告诉评分 agent，"任务在 mini-batch 上通过 ≠ 模式已解决"——如果之前的补丁修了任务但 **掉了开发集准确率** （被回滚），说明还需要一个保 dev 的更好修复，学习进度仍应为高分；反之，若任务反复通过，可以降低分数但不强制归零，因为通过可能是随机的，采样器的概率下限会定期复查低分臂，捕捉"静默回归"。

分数随后通过 softmax 变成随机选择分布：

$$
q_t(a) = \frac{\exp \left( \phi_t(a) / \tau_{\mathrm{sel}} \right)}{\sum_{a' \in \mathcal{A}_t} \exp \left( \phi_t(a') / \tau_{\mathrm{sel}} \right)}, \qquad a_t \sim \mathrm{Categorical}(q_t)
$$

温度 $\tau_{\mathrm{sel}} = 0.15$。随机采样而非贪心取最高分，是为了让低分臂在相对价值变化时仍有机会被重新选中——后面的 RQ2 分析会证明这一点确实发生了。

### Exploration Controller：什么时候"温故"，什么时候"知新"

每轮的第一个决策是二分类：

$$
d_t = g_{\mathrm{explore}} \left( H_t, \mathcal{A}_t, U_t, \mathcal{I}_{<t} \right) \in \{ \mathtt{unseen}, \mathtt{arm} \}
$$

- **Draw（$d_t = \mathtt{unseen}$）** ：从 $U_t$ 无放回采样一批未见任务执行，可能发现全新失败模式（本轮跳过臂评分）；
- **Pull（$d_t = \mathtt{arm}$）** ：进入上面的臂评分与选择流程。

决策没有固定公式或阈值，prompt 要求 agent 综合权衡两个视角：① 已知臂里是否还有值得修的目标（严重、可修、广的优先 Pull；接近解决、修不动、反复尝试低产的倾向 Draw）；② 失败面的覆盖度如何（已发现臂少、未见池大，说明弱点地图还很不全，倾向 Draw；未见池快耗尽则倾向 Pull）。边界情况是强制的：臂池为空只能 Draw，未见池为空只能 Pull。

此外还有一个 **回归再发现机制** ：当所有训练任务都探索过、且每个臂此后都被访问过之后，那些"曾经通过、且从未实例化过臂"的任务重新进入可探索池——类似传统离线学习里的多 epoch。若此时它们失败了，就按正常流程抽取失败模式，从而 **不必每轮全量复测历史成功任务** 也能捕捉回归。

### 算法骨架

把三块拼起来，单轮迭代的完整流程是：

1. **Exploration Controller** ：处理边界情况与回归再发现资格，做出 Pull/Draw 决策；
2. **（若 Pull）Arm Prioritizer** ：对所有臂打分，softmax 采样目标臂 $a_t$，从其支持任务集构造 $X_t$；（若 Draw）从 $U_t$ 采样构造 $X_t$；
3. **Harness Optimizer** ：执行 $\mathcal{E}_t \leftarrow \mathrm{Execute}(H_t, X_t)$，由固定的优化器 $\mathcal{O}$ 产出 $H_{t+1}$。实例化在 AutoSaddler 上时，这一步是：诊断失败 → 生成结构化补丁得到候选 $H'_t$ → 同批次复跑 → 若有改进再在 $D_{\mathrm{dev}}$ 上评估 → Reflection session 记录 fixed / regressed / still-failing / still-passing 四类病例并更新 EvoDAG → Evolution session 从 EvoDAG 合成 $H_{t+1}$；
4. **Failure-Pattern Extractor** ：处理 $\mathcal{F}_t$（打补丁前的失败 ∪ 补丁后仍失败/新失败的病例），更新臂集合；
5. **课程状态更新** ：记录本轮决策与结果，进入下一轮。

预算耗尽后，按开发集性能选出最终 Harness。所有任务级执行（包括优化器评估候选 Harness 的开销）都计入共享预算 $K$，保证各方法公平对比。

## 四、实验设置

- **Benchmark** ：GAIA2（模拟手机环境中的通用助理能力评测，10 个 Universe/人设环境，使用默认 ReAct agent 作为基础 Harness）和 Terminal-Bench 2.0（TB2，89 个跨系统管理、机器学习、网络安全的真实任务，基础 Harness 为 Terminus 2）。
- **数据划分（OOD 泛化）** ：GAIA2 按 Universe 切分——训练用 Universe 29（75 题），开发用 Universe 30（65 题），测试用三个完全没见过的 Universe 21/22/27（共 300 题）。这比随机切分更严格地考察分布外泛化。TB2 没有天然分组轴，采用均匀随机划分 30/19/40。
- **基线** ：GEPA、Meta-Harness、AutoSaddler（均无显式课程），以及两种基于初始 Harness 五次评估准确率排序的 **固定"由易到难"课程** （类别级 / 任务级顺序）；TB2 上额外参考人工精调的 Terminus-KIRA。
- **预算与超参** ：共享 rollout 预算 $K = 10 \times (|D_{\mathrm{train}}| + |D_{\mathrm{dev}}|)$，即 GAIA2 上 1400 次、TB2 上 490 次，相当于 Meta-Harness 跑 10 轮全量训练集的开销；每轮最多执行 3 个任务（与 AutoSaddler 的 mini-batch 一致）；$\tau_{\mathrm{sel}} = 0.15$。
- **模型** ：任务 agent 与优化器均使用 gpt-5.5，任务 agent 的 reasoning effort 为 medium，优化器 session 为 xhigh。测试时每个 Harness 独立执行 3 次，报告 Pass@1 均值与标准差。

## 五、主实验：一致、稳健、且更省钱

### 主结果

| 方法 | GAIA2 Pass@1（300 题） | TB2 Pass@1（40 题） |
| --- | --- | --- |
| Default Agent / Terminus 2（人工） | 53.6 ± 1.1 | 64.2 ± 2.9 |
| Terminus-KIRA（人工精调） | — | 69.2 ± 3.8 |
| GEPA | 54.2 ± 2.2 | 65.8 ± 5.2 |
| Meta-Harness | 54.2 ± 1.2 | 66.7 ± 5.2 |
| AutoSaddler（固定随机顺序） | 55.4 ± 1.2 | 72.5 ± 0.0 |
| AutoSaddler w/ 类别难度顺序 | 55.9 ± 1.3 | 70.8 ± 1.4 |
| AutoSaddler w/ 任务难度顺序 | 55.7 ± 1.2 | 73.3 ± 1.4 |
| **ActiveSaddler** | **59.8 ± 1.0** | **80.0 ± 2.5** |

几个要点：

- 同一个 AutoSaddler 优化器，只把固定任务顺序换成自适应课程，GAIA2 +4.4 pp、TB2 +7.5 pp。相比精心构造的难度顺序基线也全面领先（GAIA2 +3.9/+4.1 pp，TB2 +9.2/+6.7 pp）。
- TB2 上 80.0% 意味着自动优化出的 Harness 超过人工专家精调的 Terminus-KIRA 10.8 个点。
- 消融行（下文详述）全部明显低于完整方法，三个组件缺一不可。

### 稳健性、可迁移性与替代策略

- **优化随机性稳健** ：GAIA2 上独立跑第二次优化，ActiveSaddler 两次分别取得 59.8% 和 61.6%，两次都在每个 held-out Universe 上超过所有固定顺序基线（超出 3.9–5.7 pp）——提升不是某条幸运优化轨迹的产物。
- **迁移到其他优化器** ：把课程层套在 GEPA 上（GEPA 自身的 prompt 更新与候选选择完全不动），平均 Pass@1 从 54.2% 提升到 57.2%（+3.0 pp），三个 Universe 全面上涨。说明"选哪些任务驱动优化"的收益不绑定特定优化器。
- **对比非 LLM 的课程策略** ：把 Arm Prioritizer 换成 TSCL 风格的 EMA 失败残留追踪（$\eta = 0.1$），得 56.7%——比去掉评分模块（55.3%）好 1.4 pp，说明即使简单的失败持续信号也有用，但仍落后完整方法 3.1 pp；把 Exploration Controller 换成 UCB-AIR 风格的计数规则（$N_t^{\alpha} \geq |\mathcal{A}_t|$ 时探索，$\alpha = 0.6$），仅 55.6%，比固定调度好 0.3 pp，落后完整方法 4.2 pp。两个关键课程决策上，LLM 基于完整优化状态的整体判断都明显优于机械规则。

### 成本效率：多花优化器的钱，省下 rollout 的大头

ActiveSaddler 的课程层确实增加了优化器侧开销（GAIA2 上每个生成的补丁 \$11.44 vs AutoSaddler 的 \$7.87），但单次任务 rollout 的成本各方法基本相当——所以总成本差异主要来自 **rollout 被分配到了哪里** 。把两侧开销合并看端到端"成本-准确率"曲线：

![端到端优化效率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/cumulative_4metrics_run1.png)

> 图解：横轴分别为累计金钱成本、LLM 调用次数、输入/输出 token 数（含优化器侧与任务执行侧），纵轴为迄今最佳开发集准确率；上排 GAIA2，下排 TB2。ActiveSaddler 的曲线在四个口径下都位于基线左上方：GAIA2 上它花 \$298 就达到 58.5% 开发准确率，AutoSaddler 要花 \$1,360 才到同一水平，类别顺序要 \$673，任务顺序花 \$1,698 也只到 56.9%；TB2 上 ActiveSaddler 花 \$128 达到 78.9%，AutoSaddler 和任务顺序分别要 \$220 和 \$363。

一句话： **课程层的额外开销被更聪明的 rollout 分配远远赚回来了** 。

## 六、消融与机制分析：三个设计为什么都对

### RQ1：臂的粒度——为什么是"失败模式"而不是类别或任务？

把失败模式臂换成预定义的类别臂或任务臂（课程流程不变）：GAIA2 上分别掉 3.0 和 3.6 pp，TB2 上掉 10.8 和 6.7 pp。粒度选错，代价很大。

两个案例把机理讲得很清楚：

![案例一：tnxtee](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/activesaddler_rq1_case_study_1.png)

> 图解：GAIA2 任务 `tnxtee` 上，同一失败模式臂让一次失败的修复得以在后续迭代中被"重新瞄准"并迭代细化。该任务要求 agent 在朋友确认出席后立即回复"派对 10 点开始"，agent 的肯定回复漏了规定的结尾和署名格式，由此实例化了一个失败模式臂。第 3 轮的第一次修复加了一个 executor hook，但只在模型生成单行回复时触发——实际生成的是多行回复，hook 没生效。因为弱点仍以失败模式臂的形式显式存在，第 5 轮它被重新选中，第二次修复把触发条件扩展到多行回复，任务通过。对比之下，类别臂把 `tnxtee` 和另外 10 个任务混在宽泛的 `category-time` 臂里：第一批时间类任务通过后，整个臂的严重度从 0.62 暴跌到 0.12，`tnxtee` 从此再没被采到—— **不相干任务的成功掩盖了仍未解决的弱点** 。

![案例二：4bytkx](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/activesaddler_rq1_case_study_2.png)

> 图解：GAIA2 任务 `4bytkx` 上，重访同一失败模式臂纠正了一个"插错位置"的 hook。任务要求取消所有 Social/Personal 事件、只对无歧义日期创建替代事件，agent 却为全部 8 个取消事件都建了替代。第 24 轮的修复把 hook 加在澄清消息之前，但此时非法替代事件早已注册完毕。第 25 轮该臂被重新选中，第二次修复把 hook 前移到日历事件预注册阶段，任务通过。任务臂变体虽然也在第 31 轮修好了它，但修复后来随 Harness 更新丢失；而 75 个任务臂各自为政（失败模式臂峰值只有 35 个），该任务十多轮没被复测，评分器基于过期证据判断"此臂最近已修复"，严重度从 0.80 掉到 0.15、排名从第 6 跌到第 52—— **碎片化 + 陈旧证据让它再也没被回头看过** 。

定量上，用 **失败命中率** （一轮迭代的采样批次中至少含一个"仍未解决"任务的比例）衡量三种臂定义的制导能力：

![失败命中率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/rq1_quantitative_analysis.png)

> 图解：失败模式臂在 GAIA2 上的失败命中率为 52.9%，远高于类别臂的 36.4% 和任务臂的 17.6%；TB2 上为 60.4%，对 30.5% 和 36.6%。失败模式臂是从已观察到的失败中归纳出来的，拉臂即直奔已知弱点；类别臂把异质任务混在一个标签下，任务臂则把预算摊薄到整个训练集上。

附录还列出了优化过程中自动发现的全部失败模式：GAIA2 共 35 个（臂池从 6 个一路长到 35 个），TB2 共 15 个。模式与任务之间是 **多对多** 关系——GAIA2 的 P8（跨应用聚合前源枚举不全）横跨 4 个任务，TB2 的 P13（超大内联 setup 载荷超出 exec 参数上限）也横跨 4 个任务；反过来，`dpbafp`、`make-doom-for-mips` 这样的任务各自暴露了多个不同模式。这正是预定义类别/任务粒度都表达不了的结构。

### RQ2：自适应臂评分——优先级如何随证据演化？

去掉 Arm Prioritizer（所有臂同分）：GAIA2 掉 4.5 pp，TB2 掉 7.5 pp。两个对比鲜明的臂生命周期案例说明了评分机制在做什么：

![P3：已解决的弱点](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/rq2_arm_lifecycle_grid_P3.png)

> 图解：GAIA2 的 P3 臂在第 5 轮首次被拉就产出了被接受的补丁，此后优先级骤降；之后再被复访时所有支持任务全部通过、无需再修，臂稳定在低优先级，预算让位给其他弱点—— **已解决的弱点被正确地降权** 。

![P8：反复出现的弱点](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/rq2_arm_lifecycle_grid_P8.png)

> 图解：P8 的前三次被拉都没产出补丁（相关任务重跑时碰巧通过了），但评分器没有因此降权——执行是随机的，且还没有任何补丁真正针对这个模式。后来同一失败模式在更多任务上出现（图中叉号标记），提供了"弱点仍活跃"的新证据，P8 再次被选中并产出被接受的补丁，把最佳开发准确率从 61.5% 推到 63.1%—— **暂时性成功不被当作已解决的证据** 。

全臂池的宏观统计同样支持这一机制：

![Pull 概率质量演化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/pattern_prob_stack_topk_paper.png)

> 图解：GAIA2 全程 34 次臂选择迭代中，平均 pull 概率最高的 10 个臂单独着色，其余 25 个合并为 Other；上图显示臂池从 6 个增长到 35 个。到第 51 轮，35 个臂中有 24 个（68.6%）的支持任务已全部复测通过，但它们只占 26.2% 的概率质量（均匀分配下应占 68.6%，实际仅为其 0.38 倍）；剩下 11 个仍有未成功任务的臂拿走 73.8%（均匀分配的 2.35 倍）。同时，当同一模式在新任务上再次出现时，P8 的 pull 概率平均上升 4.8 pp，P10 上升 9.4 pp——优先级随证据双向流动。

![全臂 pull 概率热力图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/pattern_prob_heatmap_all_paper.png)

> 图解：每行是一个臂（按实例化时间排序），每列是一次臂选择迭代，颜色深浅为 pull 概率，橙色标记为实际被采中的臂，灰色表示该臂尚未进入池中。34 次拉动覆盖了 35 个臂中的 19 个；被选中臂的优先级中位排名是第 3，62% 的拉动落在前三名，但最远也选到过第 20 名—— **聚焦但不贪心** ，随机选择让低排名臂在相对价值变化时仍有机会。灰色阶梯也直观地表明：所有决策都是在一个不断扩张的臂池上相对做出的。

此外，ActiveSaddler 在"训练中观察到的失败最终被转为成功"的转化率上也是最高，优于 w/o Arm Prioritizer 和 AutoSaddler。

### RQ3：自适应探索——什么时候该去发现新弱点？

把 Exploration Controller 换成"每 5 轮固定探索一次"：GAIA2 掉 4.5 pp，TB2 掉 8.3 pp。探索时机本身就是优化变量。

![自适应探索轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/pattern_discovery_activesaddler.png)

> 图解：GAIA2 优化全程的 Pull/Draw 决策轨迹与新失败模式的发现过程。代表性决策显示：当已知臂里没有值得修的目标时策略选 Draw（发现新模式），当存在可行动的未解决弱点时选 Pull。全局上，Draw 率从前半程的 44%（臂池小、失败面未摸清）降到后半程的 20%（臂池大了，温故价值上升）。

![探索决策与未解决弱点的依赖](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ActiveSaddler-Automated-Curriculum-Learning-for-Agent-Harness-Optimization/Figures/rq3_quantitative_analysis.png)

> 图解：Pull 决策时与 Draw 决策时的平均未解决臂数之比，大于 1 说明策略倾向于在"还有更多没修好的弱点"时选择继续修。ActiveSaddler 在 GAIA2 上该比值为 2.0 倍（1.62 vs 0.81），TB2 上高达 6.9 倍（0.76 vs 0.11）；固定调度的比值接近 1（0.8 和 1.3）——它的 Pull/Draw 决策对当前优化状态基本无感。

## 七、局限与责任部署

作者明确将 ActiveSaddler 定位为研究方法而非生产系统，几点提醒值得引用：

- 实验全部在公开/合成 benchmark 上进行， **benchmark 性能提升不等于优化后的 Harness 在生产环境中安全、合规** ——Harness 更新会改 Prompt、工具接口、权限与运行时控制逻辑，可能改变 agent 访问资源和执行动作的方式；
- 方法本身不包含独立的安全过滤或 reward hacking 检测机制，任务级目标不构成可接受行为的完整规范；
- 生产部署需要额外补齐：对抗性误用测试、最小权限工具访问、沙箱、执行轨迹中的 PII/密钥过滤与留存策略、高风险变更的人工审批与回滚机制。

## 八、总结

- **新问题维度** ：Harness 优化的性能不仅取决于"怎么用反馈更新"，同样取决于"用哪些任务产生反馈"——本文首次将后者形式化为自动课程学习问题。
- **非平稳 Bandit 建模** ：臂不是预设的，而是从诊断出的失败中在线实例化的"失败模式臂"，臂池随优化动态扩张，优先级随证据持续重估。
- **三组件缺一不可** ：失败模式臂（粒度：比类别细、比任务聚）、LLM 臂评分（严重性/可修性/广度/副作用四轴 + softmax 随机选择）、LLM 探索控制（Pull/Draw 无公式权衡），消融各掉 3–11 pp。
- **硬结果** ：GAIA2 +4.4 pp（59.8%）、TB2 +7.5 pp（80.0%，超人工精调 Harness 10.8 pp），两次独立优化运行一致领先，迁移到 GEPA 再涨 3.0 pp。
- **更优成本曲线** ：课程层多花优化器侧开销，但端到端达到同等准确率的钱省 4 倍以上（GAIA2 上 \$298 vs \$1,360）。

展望上，这个"课程即优化维度"的视角大概率不止适用于 Harness 优化——凡是"预算受限、目标随学习进程漂移"的迭代式优化场景（prompt 优化、评测任务选择、RL 环境调度），都值得问一句：我的课程表，是死的还是活的？而把安全约束作为独立信号接入课程决策，则是走向生产前必须跨过的一步。

> 本文参考自 [ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization](https://arxiv.org/abs/2610.00906)