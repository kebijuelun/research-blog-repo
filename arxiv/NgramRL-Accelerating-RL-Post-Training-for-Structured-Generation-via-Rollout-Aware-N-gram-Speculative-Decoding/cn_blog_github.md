# NgramRL：用历史答案加速 RL

RL 后训练最耗时的地方，往往不是更新参数，而是让模型反复生成 rollout；这一阶段可占单步训练时间的 70% 以上。NgramRL 不再额外训练草稿模型，而是把同一 prompt 的历史 rollout 变成会自动更新的 n-gram 草稿库，再由目标模型并行验证。配合异步预取，它还解决了 CPU 查缓存、GPU 等结果的流水线空转问题。在两种模型和三类任务上，稳定阶段每次目标模型前向可提交 3.60–7.77 个 Token；Spider 上相对同步 NgramRL 的实测 rollout 吞吐最高提升 1.38 倍。

## 为什么 RL 后训练需要另一种投机解码？

GRPO（Group Relative Policy Optimization，组相对策略优化）的一次训练大致分三步：针对每个 prompt 采样一组回答，用 Reward Model 打分，再根据组内相对优势更新策略。第一步要逐 Token 自回归地产生大量 rollout，因而成为墙钟时间的主要瓶颈。

投机解码本来很适合缓解这个问题：让一个便宜的 drafter 先猜若干 Token，目标模型一次前向并行检查，猜对就批量接收。但直接搬进 RL 后训练，会撞上三个约束。

| 约束 | RL 场景中的矛盾 | 对 drafter 的要求 |
| --- | --- | --- |
| C1：GPU 显存紧张 | 策略、梯度、优化器状态和 KV Cache 已在争抢显存 | 不应常驻 GPU，也不应新增训练参数 |
| C2：策略持续变化 | 固定草稿模型会逐步偏离当前策略，接受率随训练下降 | 必须跟随策略自动更新 |
| C3：重复跨越训练步出现 | 自由推理文本与当前 prompt 未必重叠，但同一 prompt 的历史回答会重复 | 证据应来自跨 epoch 历史，而非单次生成 |

已有路线各自只解决了部分问题。Medusa、EAGLE 等基于模型的方法需要额外参数或草稿头；PLD（Prompt-Lookup Decoding）只从当前上下文复制，REST 使用固定数据库，Token Recycling 复用单次解码中的候选。它们都不增加模型参数，却没有利用 RL 训练中“同一 prompt 被反复采样”这一信号。

最接近的 SuffixDecoding 会把历史输出放进 CPU 后缀树，但它主要面向固定模型和 Agent 推理流水线。NgramRL 的差异不只是换了缓存结构，而是把缓存放进 RL 训练闭环：策略产生新 rollout，新 rollout 再更新下一轮的 drafter。

## 核心观察：历史 rollout 就是免费 drafter

NgramRL 利用的是一种“以 prompt 为条件的时间局部性”。直觉上，它像让模型在再次作答前翻自己的旧作业：题目相同，模型又只更新了一点，新旧答案自然可能共享大段内容。

这对结构化生成尤其明显。SQL、代码、工具调用会反复出现相同的关键字、字段名、表结构和模板，因此比自由文本更适合复制式起草。论文对相邻 epoch 的测量显示，同一 Spider prompt 的旧 rollout 可覆盖新 rollout 的 48%–96%，重复片段长度达到 7–23 个 Token。

下面三组相隔一个 epoch 的 SQL 结果，展示了重复度随训练上升的趋势。R1 是较早结果，R2 是较晚结果。

| 训练阶段 | 较早与较晚 rollout | 共享比例 |
| --- | --- | --- |
| 早期，step 10 → 115 | R1：`SELECT COUNT(*) FROM Companies WHERE Industry IN ('Banking', 'Conglomerate');`<br>R2：`SELECT COUNT(*) FROM Companies WHERE Industry = 'Banking' OR Industry = 'Conglomerate';` | 50% |
| 中期，step 300 → 407 | 两者都查询 `billing_country` 并计数，都使用相同的 `FROM invoices GROUP BY ... ORDER BY ... DESC LIMIT 5` 骨架，主要差异是计数别名 | 81% |
| 晚期，step 1200 → 1305 | 两者均为 `SELECT b.bname FROM bank b WHERE b.state = 'New York';` | 100% |

> 图解：这组原文示例不是图片，而是三对 SQL 文本。早期只共享查询骨架，中期已共享绝大部分语句，晚期则逐字一致。NgramRL 正是把这种“旧答案覆盖新答案”的冗余转化为可接受的草稿。

有了草稿来源，下一步是回答两个系统问题：缓存怎样跟随策略演化，以及怎样让起草与验证真正缩短解码时间。

## NgramRL：按 prompt 建缓存，让目标模型验收

![NgramRL 跨训练 epoch 的 CPU-GPU 架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/ngramrl-framework.png)

> 图解：同一 prompt 在相邻 epoch 中进入 GPU 上的策略模型。CPU 侧的 per-prompt n-gram cache 提供草稿，验证后接受的 Token 又回流更新缓存；缓存跨 epoch 保留，因此草稿来源始终来自策略自己的近期输出。

整个流程可以压缩为“查旧答案—并行验收—写回新答案”。它不增加模型参数，不需要训练 drafter，也不让草稿状态常驻 GPU。更重要的是，验证仍由当前目标策略完成，缓存只影响提议速度，不决定最终输出分布。

### 缓存怎样建立和更新？

对 prompt $p$，系统单独维护查找表 $\mathcal{C}_p$，而不是把所有 prompt 混进一个全局桶。设第 $t$ 步为该 prompt 采样的 rollout 集合为 $R_p^{(t)}$，缓存更新分两步：

1. **构建与计数**：从 $R_p^{(t)}$ 中提取阶数为 $n_{\min}$ 到 $n_{\max}$ 的 n-gram。对每个 $n$，把前 $n-1$ 个 Token 作为上下文 $ctx$，把下一个 Token $x_i$ 作为续写，并记录出现频次和最近步数。
2. **窗口淘汰**：只保留最近 $W$ 步的 rollout 证据，删除满足 $\mathrm{step}<t-W+1$ 的旧条目，限制内存增长并减少与当前策略的错配。

其核心映射可以写成：

$$
\mathcal{C}_p:(x_{i-n+1},\ldots,x_{i-1})\longmapsto (x_i,\ \mathrm{count},\ \mathrm{step}),\quad n\in[n_{\min},n_{\max}]
$$

起草时，系统为当前上下文寻找**最长匹配**的 n-gram，并取出保存的续写作为草稿；如果匹配失败，该位置退回普通解码。采用最长匹配，是为了用更多上下文减少“相同前缀、不同后续”的歧义。

笔者认为，per-prompt 与滑动窗口是这套设计最关键的两处取舍：前者避免不同题目的模板互相污染，后者承认 RL 的策略不是静态数据库。缓存不是越大越好，而是越贴近当前策略越有价值。

### 为什么加速不会改变训练结果？

这里要区分“草稿分布”和“最终采样分布”。缓存提出的分布记为 $q$，当前目标策略为 $\pi_\theta$。在上下文 $c$ 下，草稿 Token $x$ 的接受概率为：

$$
p_{\mathrm{acc}}(x)=\min\left(1,\frac{\pi_\theta(x\mid c)}{q(x\mid c)}\right)
$$

若拒绝，则从归一化的正残差分布重新采样：

$$
\frac{(\pi_\theta(x\mid c)-q(x\mid c))_+}{\sum_{x'}(\pi_\theta(x'\mid c)-q(x'\mid c))_+}
$$

这就是标准投机采样的修正机制。它保证最终 Token 仍严格满足 $x\sim\pi_\theta(\cdot\mid c)$，与 drafter 是神经网络还是查找表无关。因此，rollout、奖励、优势和梯度在分布上均与非投机训练一致；这里保证的是**分布不变**，不是声称两次随机运行会生成逐 Token 相同的结果。

## 真正的系统难点：CPU 有答案，GPU 却在等待

缓存放在 CPU 上虽然省显存，却引入了主机—设备往返。同步方案中，GPU 必须先完成验证并把结果同步给 CPU；CPU 随后才能查找下一份草稿、处理拒绝、调度下一步并启动 Kernel。论文测得这段空隙让 GPU 每轮约 29% 的时间空闲，而且主要开销不是查表，而是同步和调度。

![同步与异步时间线图例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/timeline_legend.png)

> 图解：该图例对应下面两张 CPU/GPU 时间线，用于区分设备执行、主机工作、同步与调度等环节。阅读重点不是单次查表，而是整条主机依赖链是否暴露在 GPU 关键路径上。

![同步 NgramRL 的解码时间线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/timeline_sync.png)

> 图解：同步基线中，CPU 必须等待本轮 GPU 验证结果，再开始下一步准备，GPU 因而出现空闲区间；图示单轮耗时为 26.1 ms。

![异步 fan-out 预取的解码时间线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/timeline_async.png)

> 图解：NgramRL 在当前前向执行期间预建下一轮草稿表，并重叠调度工作，图示单轮缩短至 19.2 ms。它隐藏的是整条主机链，而不只是 n-gram 查找。

### 验证尚未结束，下一轮上下文怎么提前知道？

第 $t$ 轮验证结果可以写成二元组 $(j,b)$：$j\in\{0,\ldots,k\}$ 是接受的草稿长度，$b$ 是目标模型在第 $j+1$ 个位置额外产生的 bonus token。拒绝时，$b$ 来自残差重采样；全部接受时，$b$ 是目标模型免费产生的下一个 Token。

问题在于，$b$ 直到 GPU 验证结束才确定，而它又是下一轮上下文的最后一个 Token。NgramRL 的答案是：不等结果，提前为最可能的结果分别备好草稿，也就是 fan-out prefetch（扇出预取）。

具体过程如下：

1. 当前 GPU 前向仍在运行时，CPU 为每个获得预算的接受长度分支 $j$，从缓存中选取按频次排序的前 $f_j$ 个 bonus 候选。
2. 对每个 $(j,b)$ 候选各查一次缓存，构造对应的 $k$ Token 草稿，并把结果表在验证返回前传到 GPU。
3. 真实结果 $(j^*,b^*)$ 到达后，GPU 用 $O(1)$ Key Probe 取草稿，不做内容搜索。
4. 命中时直接进入下一步；未命中时仅该请求回退到普通顺序查找，不阻塞整个 batch。

无论命中与否，已接受前缀和 bonus token 都不被预取器改写。覆盖率 $c$ 只决定性能，正确性始终由目标模型验证保证，最坏情况退化为同步基线。

### 有限预算怎样分给不同分支？

总候选预算为 $N=\sum_j f_j$。系统只用过去 $W$ 步的因果统计：接受长度直方图 $\hat P(j)$，以及分支 $j$ 的覆盖曲线 $\hat c_j(f)$——即真实 bonus 落在前 $f$ 个候选中的概率。分配目标是：

$$
\max_{\{f_j\}}\sum_j\hat P(j)\hat c_j(f_j)\quad\text{s.t.}\quad\sum_j f_j=N
$$

求解采用按边际收益分配的贪心策略：哪个分支再增加一个候选最能提升期望覆盖率，就把下一份预算给它。高接受分支会自然得到更多关注；全部分支接受时，bonus 往往就是旧轨迹的下一个 Token，约 90% 的时候排名第一，因此 $f_j=1$ 往往已经足够。

### 覆盖率如何转化为端到端加速？

设每轮 GPU 忙碌时间为 $T_{\mathrm{gpu}}$，需要隐藏的串行主机时间为 $T_{\mathrm{cpu}}$。命中时隐藏主机链，未命中时仍支付这部分时间，因此：

$$
\mathrm{speedup}(c)=\frac{T_{\mathrm{gpu}}+T_{\mathrm{cpu}}}{T_{\mathrm{gpu}}+(1-c)T_{\mathrm{cpu}}}\xrightarrow{c\to1}1+\frac{T_{\mathrm{cpu}}}{T_{\mathrm{gpu}}}
$$

这个上界也解释了收益边界：模型或 batch 越大，$T_{\mathrm{gpu}}$ 相对越长，可隐藏主机时间所占比例越小；CPU 链越重的环境，潜在收益反而越大。预取自身约需每请求 $N$ 次查找，必须塞进 $T_{\mathrm{gpu}}$ 内；当 $N$ 超过约 64，预取链逐渐接近它本想隐藏的主机耗时，因此甜点预算并不大。

工程上，NgramRL 已集成到 verl 的 vLLM rollout worker。缓存挂在 rollout 引擎旁，不要求修改 Reward Model、优化器或策略架构，也可与其他 rollout 效率优化组合；适用前提是训练会重复访问 prompt。

## 实验：历史缓存究竟多能接受草稿？

方法同时包含“接受率”和“系统重叠”两层，实验也分别验证：前者看每次目标前向能提交多少 Token，后者看这些接受收益能否转化为真实吞吐。

### 实验设置与评价口径

| 项目 | 设置 |
| --- | --- |
| RL 训练 | verl + GRPO，2 × NVIDIA A100 80GB，vLLM 提供 rollout |
| 采样 | 每步 64 个 prompt，每个 prompt 采样 5 条回答，共 15 个 epoch |
| 模型 | Qwen3-8B、DeepSeek-LLM-7B |
| 数据集 | GSM8K（自由数学推理）、Spider 1.0（Text-to-SQL）、CodeContests（代码） |
| 基线 | PLD、REST、基于最近 10k 回答全局后缀树的 SuffixDecoding |
| 草稿预算 | 各方法统一 $k=16$ |
| 运行时测量 | Qwen3-8B，bf16，batch 64，单张 RTX PRO 6000 96GB；32 核 Threadripper 9970X 主机 |

主要指标 TPS 在本文中指每次目标模型前向提交的 Token 数，即“1 + 平均接受草稿长度”，并非通常所说的每秒 Token 数。它是硬件无关的无损加速代理指标。结果同时报告全训练均值和最后 30% 步数的均值；后者缓存已充分预热，称为稳定阶段。

运行时实验还用 NVIDIA Nsight Systems 区分设备忙碌时间和暴露的主机时间，避免把接受长度提升直接等同于端到端加速。

### 主结果：稳定阶段每次前向提交 3.60–7.77 Token

下表统一采用“全程均值 / 最后 30% 均值”，加粗为各模型、各任务最优值。

| 模型与方法 | GSM8K | Spider | CodeContests |
| --- | --- | --- | --- |
| **Qwen3-8B：PLD** | 1.31 / 1.31 | 1.10 / 1.11 | 1.28 / 1.30 |
| **Qwen3-8B：REST** | 1.32 / 1.31 | 1.02 / 1.02 | 1.07 / 1.07 |
| **Qwen3-8B：SuffixDecoding** | 2.42 / 2.32 | 4.01 / 4.15 | 2.50 / 2.96 |
| **Qwen3-8B：NgramRL** | **4.00 / 4.36** | **5.94 / 7.77** | **2.70 / 3.60** |
| **DeepSeek-7B：PLD** | 1.47 / 1.49 | 1.24 / 1.24 | 1.19 / 1.15 |
| **DeepSeek-7B：REST** | 1.69 / 1.69 | 1.02 / 1.02 | 1.02 / 1.02 |
| **DeepSeek-7B：SuffixDecoding** | 3.37 / 3.64 | 3.71 / 3.88 | 3.11 / 3.99 |
| **DeepSeek-7B：NgramRL** | **5.83 / 6.81** | **6.37 / 7.71** | **3.92 / 6.19** |

稳定阶段，NgramRL 相对 SuffixDecoding 提升 1.20–1.99 倍，相对 PLD 最高达到 7.0 倍。即便全程均值包含冷启动的第一个 epoch，它在所有模型与任务组合上仍领先。

优势大小与输出规律一致：Spider 的 prompt 特定 SQL 最容易复用；DeepSeek 的 GSM8K 推导也表现出较强重复；CodeContests 的跨 prompt 模板更多，全局缓存因而更有竞争力。

### 训练动态：先冷启动，再随历史积累升温

![Qwen3-8B 三项任务上的 TPS 训练动态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/fig_dynamics_1x3_new.png)

> 图解：横轴是 GRPO 训练步，纵轴是每次目标前向提交的 Token 数，三幅子图依次为 GSM8K、Spider 和 CodeContests。PLD、REST、SuffixDecoding 与 NgramRL 分别用不同线型表示，竖直虚线标出第一个 epoch 结束。NgramRL 起初接近 1，prompt 重复出现后上升；GSM8K 和 CodeContests 持续增强，Spider 则在中期达峰后回落。

Spider 的回落说明“历史”并非永远有效：当策略生成更多样的 SQL，旧 rollout 会逐渐过时。CodeContests 早期由全局缓存领先，随着 per-prompt 历史积累，NgramRL 才完成反超。这也解释了为什么后文要同时消融缓存作用域和历史窗口。

### 接受长度呈重尾分布，$k=16$ 不是随意放大

![不同模型与任务的接受长度分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/fig_accept_dist.png)

> 图解：横轴为单次接受的草稿长度，从 0 到 16；纵轴为对应解码步所占比例，并采用对数刻度。多数步骤集中在零接受，但所有曲线都有长接受尾部，并在完整接受 16 个 Token 处再次抬升。

只看平均接受长度会漏掉这批“整段重放”的机会。Spider 上，14.1% 的步骤至少接受 12 个草稿 Token，完整接受 16 个的比例最高达 10.1%。DeepSeek GSM8K、Qwen GSM8K 和 CodeContests 的长接受步骤占比分别为 13.0%、6.4% 和 3.7%。这些长片段往往来自上一轮解答的逐字复现，若草稿长度过小，就无法把重尾转化为收益。

## 消融实验：缓存该怎样索引、保留和配置？

接受率已经验证，接下来要拆清收益究竟来自 n-gram 结构、per-prompt 索引，还是单纯堆更多历史。

### $n_{\max}=6$ 与 $k=16$ 位于收益拐点

![最大 n-gram 阶数对 TPS 和缓存内存的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/nmax_ablation_qwen3-8b-gsm8k.png)

> 图解：实验使用 Qwen3-8B 与 GSM8K，横轴为最大 n-gram 阶数 $n_{\max}$，两组纵轴分别表示 TPS 和主机缓存内存。提高阶数能增加接受长度，但 6 阶之后收益趋于饱和，内存仍快速增长，因此选择 $n_{\max}=6$。

![投机长度对 TPS 的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/speclen_ablation_qwen3-8b-gsm8k.png)

> 图解：横轴为投机长度 $k$，纵轴为 TPS。延长草稿可提升每次前向提交的 Token 数，但约 16 后边际收益减小；$k$ 不改变缓存本身，因此论文统一选 $k=16$。

作者指出，这两个配置虽在单一模型与数据集上展示，却能较好迁移到其他实验，因而全文共用。

### 关键是按 prompt 索引，而不是无限扩大全局缓存

| 配置 | 冷启动 epoch | 平台期 |
| --- | --- | --- |
| NgramRL，全局桶 | 1.55 | 2.41 |
| NgramRL，per-prompt Key | 1.00 | **3.05** |
| SuffixDecoding | 1.90 | 2.43 |

同样使用 n-gram，换成全局桶后平台期只有 2.41，接近 SuffixDecoding 的 2.43；per-prompt 则达到 3.05。把 SuffixDecoding 的缓存从 10k 扩到 40k 回答，也只提升到 2.65。这说明瓶颈主要是缺少 prompt 条件索引，而不是缓存容量不足。

### 小窗口已保留大部分收益，旧历史还可能帮倒忙

下表每格为“每步接受的草稿 Token 数（缓存 MB）”，$w$ 表示每个 prompt 保留最近多少次出现的历史。

| 模型与任务 | $w=1$ | $w=4$ | $w=16$ | $w=\infty$ |
| --- | --- | --- | --- | --- |
| **Qwen3-8B：GSM8K** | 1.87（841） | 2.81（1878） | 3.38（3181） | 3.34（3634） |
| **Qwen3-8B：Spider** | 3.39（236） | 5.13（255） | 5.82（309） | 5.82（309） |
| **Qwen3-8B：CodeContests** | 1.72（249） | 2.64（591） | 2.89（1742） | 2.91（1744） |
| **DeepSeek-7B：GSM8K** | 3.69（177） | 5.08（293） | 5.74（662） | 5.74（662） |
| **DeepSeek-7B：Spider** | 3.00（230） | 4.63（251） | 5.83（335） | 5.83（335） |
| **DeepSeek-7B：CodeContests** | 3.73（67） | 5.12（89） | 5.78（290） | 5.78（293） |

$w=4$ 已保留完整历史接受收益的 79%–91%，同时显著节省内存。超过 $w=16$ 后基本没有额外帮助，Qwen3-8B 的 GSM8K 甚至从 3.38 降到 3.34，说明旧 rollout 会与当前策略脱节。这与 Spider 训练曲线中期回落相互印证。

## 预取实验：高接受率能否变成真实吞吐？

最后一个问题是，fan-out 是否能以较小预算覆盖真实验证结果，并把主机开销从关键路径上移除。

![预取覆盖率随候选预算变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/fanout_coverage_budget.png)

> 图解：横轴是候选预算 $N$，纵轴是覆盖率 $c$，即真实 $(j^*,b^*)$ 已包含在预取候选中的概率。Spider 在 $N=8$ 时达到 0.95，$N=16$ 时约为 0.97；GSM8K 与 CodeContests 在 $N=32$ 时分别为 0.65 和 0.49，之后只有缓慢的对数级增长。

覆盖率和接受率遵循同一规律：高接受任务更常沿缓存轨迹前进，bonus 也更容易预测。剩余 miss 更多受输出熵限制，而不是历史数据不足；增加 rollout 历史并不会持续推高覆盖率。因此，预算继续越过拐点并不划算，论文实际使用 $N=8$–$32$。

![Spider 上预算为 16 时的模型加速](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/NgramRL-Accelerating-RL-Post-Training-for-Structured-Generation-via-Rollout-Aware-N-gram-Speculative-Decoding/figures/model_speedup_n16.png)

> 图解：纵轴为相对顺序同步基线的 Spider rollout 加速比。$N=16$ 时，Qwen3-8B 的覆盖率为 0.966，实测加速 1.38 倍，接近 1.40 倍重叠上界；DeepSeek-7B 的覆盖率为 0.995，实测 1.21 倍，达到其直接测得的重叠上界。

在熵更高的任务上，预取仍有收益但幅度较小：GSM8K 和 CodeContests 实测为 1.05–1.06 倍，对应重叠上界分别为 1.10 和 1.14 倍。由于 miss 会按请求回退顺序查找，最差表现是同步基线，而不是因错误预取改变输出。

需要注意，这些端到端数字来自论文指定的 8B 运行时测试平台，不应直接外推为任意模型、batch 或 CPU/GPU 配比下的固定加速比；覆盖率、主机链占比和预取成本都会改变结果。

## 总结：把训练中的重复，变成解码中的并行

- NgramRL 用同一 prompt 的近期 rollout 构建 n-gram drafter，无需辅助模型、额外训练或常驻 GPU 的草稿状态。
- Per-prompt 索引负责隔离题目语义，有限历史窗口负责跟踪策略漂移；目标模型验证保证最终采样分布不变。
- Fan-out 预取针对可能的 $(j,b)$ 结果提前备稿，真正隐藏的是同步、查表和调度组成的整条 CPU 链。
- 稳定阶段每次目标前向提交 3.60–7.77 个 Token，最高为 SuffixDecoding 的 1.99 倍、PLD 的 7.0 倍。
- Spider 上实测预取加速最高 1.38 倍，但冷启动、输出熵、prompt 是否复访以及软硬件配比仍决定实际收益。

这项工作更适合 prompt 会重复、输出具有复用结构的 RL 场景；未来值得探索的是如何在任务多样性更高、策略漂移更快时，自适应选择缓存作用域与历史窗口。

> 本文参考自 [NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding](http://arxiv.org/abs/2610.32881v1)