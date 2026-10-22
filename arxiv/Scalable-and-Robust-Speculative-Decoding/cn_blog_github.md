# Sequoia：可扩展、鲁棒的投机解码，让 LLM 推理提速最高 9.5 倍

大模型推理慢的根源是 I/O 瓶颈：每生成一个 token 都要把全部参数从显存里读一遍，GPU 算力大量闲置。投机解码（Speculative Decoding）用小模型"猜"、大模型"验"来破局，但现有方法在猜测预算变大时收益见顶，且对温度等超参数很敏感。这篇 ICML 2024 论文提出的 **Sequoia**，用动态规划搜索最优 token 树结构，配合"无放回采样"的验证算法，在 A100 上对 Llama2-7B 实现最高 **4.04×** 加速，在 L40 的 offloading 场景下对 Llama3-70B-Instruct 实现最高 **9.5×** 加速，单 token 延迟从 5.7 秒压到 **0.60 秒**。

## 背景：投机解码与它的两个"天花板"

### 投机解码在做什么

自回归生成是逐 token 串行的，每步都要完整加载一遍模型参数。在小 batch 或 offloading（模型权重放在内存/硬盘，按需换入 GPU）场景下，生成一个 token 的时间甚至和处理几百上千 token 的 prompt 一样长。

投机解码的思路是：

1. 用一个小的 **draft model**（草稿模型）快速猜出若干个候选 token；
2. 把候选组织成一棵 **token tree**（token 树），每个节点代表一种可能的续写；
3. 用目标大模型**一次前向并行验证**整棵树，接受其中正确的前缀。

关键设计是接受/拒绝规则——它保证了最终输出分布与直接用大模型采样**完全一致**，即"免费加速，不掉精度"。

### 现有树结构：预算越大，浪费越多

以 SpecInfer 为代表的方法，猜测树就是 **k 条独立序列**（从根长出 k 条长度为 L 的链）；Medusa 用满 k 叉树，但中等预算下根本长不深。问题在于：这些结构能接受 token 数的期望值有**理论上界**——

- 单条序列的上界是 $1/(1-P_1)$；
- k 条独立序列只能把上界提高 1（因为它们只增加第一个 token 被接受的机会）；
- 即使是无限深的二叉树，上界也只有 $1/(1-P_2)$。

这里 $P_k$ 表示"一个节点有 k 个孩子时，其中一个被接受"的概率。换句话说，**树再大也没用，期望接受 token 数会饱和**。这在 offloading 这种极端 memory-bound 的场景里尤其致命——那里一次验证的成本几乎与验证 token 数无关，树本应该越大越好，但旧结构接不住这个红利。

### 现有验证算法：低温翻车，高温平庸

另一个观察是验证算法对超参数不鲁棒。SpecInfer 和 SpecTr 都是**有放回**地从 draft model 采样候选：一旦 draft model 对某个错误 token 很自信，它可能被反复采样、反复拒绝，在低温度下表现尤其差。而 naive / top-$k$ 采样则相反——低温不错，高温平庸。没有一种方法能在所有温度下都打。

解决了背景问题，接下来看 Sequoia 如何各个击破。

## Sequoia 方法总览

Sequoia 由三个相互配合的组件构成：

- **可扩展的树构造算法**：把"找最优 token 树"形式化为约束优化问题，用动态规划精确求解；
- **鲁棒的采样与验证算法**：对 draft model 做无放回采样，被拒绝的 token 绝不再提；
- **硬件感知的树优化器**：根据具体硬件的验证耗时曲线，自动选择树的大小和深度。

三者分别回答"树长什么样"、"怎么采样验证"、"在不同硬件上用多大的树"。

## 组件一：用动态规划构造最优 token 树

### 把树构造写成优化问题

Sequoia 首先引入两个关键假设性定义：

- **位置接受假设**（positional acceptance）：一个候选 token 被接受的概率只取决于它是父节点的"第几个孩子"，与具体内容无关；
- **接受率向量** $p = (p_1, p_2, \ldots, p_k, \ldots)$，其中 $p_k$ 是第 k 个孩子位置上的接受概率，可以离线实测。

在这套语言下，给节点 $v$ 定义 score function：

$$
f(v) = \prod_{i \in \texttt{Path}(v)} p_i
$$

其中 $\texttt{Path}(v)$ 是从根到 $v$ 路径上的孩子序号列表（例如 $v$ 是"根的第 2 个孩子的第 3 个孩子"，则路径为 $[2,3]$），根的 score 定义为 1。

论文证明了验证一棵树 $\mathcal{T}$ 期望生成的 token 数有漂亮的闭式表达（命题 1）：

$$
F(\mathcal{T}) = \sum_{v \in \mathcal{T}} f(v)
$$

这个式子可以直观理解为：**每个节点贡献的期望收益，等于"一路接受过来"的概率连乘**。证明用对树规模的归纳完成——删掉一个叶子节点 $v$，期望收益的增量恰好是 $f(v)$。

### 动态规划求解

目标是找一棵大小为 $N$、期望接受 token 数最大的树。朴素枚举不可行（树结构空间指数级），Sequoia 用 DP 填一张二维表：

$$
T(n, b) = \max_{\mathcal{T},\ |\mathcal{T}| = n,\ \texttt{FirstBranch}(\mathcal{T}) = b} F(\mathcal{T})
$$

即"大小为 n、根有 b 个直接孩子的最优树"的期望收益，$B$ 是单个节点孩子数的上界。边界是 $T[1, 0] = 1$（只有根时，每步投机解码保底生成 1 个 token）。

递推的直觉是"砍"掉根的最后一个孩子：把树分成"根 + 前 b-1 个孩子"（tree #1）和"以第 b 个孩子为根的子树"（tree #2）。设 tree #2 有 $m$ 个节点，则：

$$
T[n, b] = \max_{1 \le m \le n-1} \Big( T[n-m, b-1] + p_b \cdot \max_{0 \le j \le B} T[m, j] \Big)
$$

tree #2 的收益要乘上 $p_b$——因为只有第 b 个孩子被接受了，它才有机会被验证。记录每一步取 max 时的 $m$ 和 $b$，就能回溯重建出最优树本身。整个 DP 在**离线**运行，完全不拖慢推理。

![Sequoia 动态规划与树结构示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sequoia_dp_v3.png)

> 图解：左图展示 DP 利用的递归子结构——一棵树可以拆成"前 b-1 个孩子的部分"和"第 b 个孩子为根的子树"两块分别求解。

![Sequoia 树真实例子](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sequoia_tree_flipped.png)

> 图解：右图是一棵真实的大小为 64、最大深度 12 的 Sequoia 树。可以看到它既不像 k 条独立链那样"扁平"，也不像满 k 叉树那样"矮胖"，而是深浅结合、随接受率自适应展开。

这个 DP 还有两个实用扩展：

- **限深**：投机耗时与树深成正比，DP 可以加深度上界 $D$，找限深下的最优树；
- **兼容自投机**：对 Medusa、Eagle、GLIDE 这类复用目标模型表示的 self-speculation 方法，接受率会随树深衰减——DP 可以把一维接受率向量换成二维矩阵（不同深度用不同向量），直接兼容。

### 理论：增长率下界是（准）对数级

这是全文最亮眼的理论结果。先定义：若验证算法"拒绝全部 k 个候选"的概率 $r_k$ 满足幂律上界 $r_k \le 1/k^b$（$b > 0$），则称它有 **b 幂律接受率**。

**定理**：使用满足 b 幂律接受率的验证算法时，大小为 n 的 Sequoia 树期望生成的 token 数满足

$$
G(n) \in \Omega\Big(\frac{b \log n}{\log \log n}\Big)
$$

对比前面列举的旧结构上界（单链 $1/(1-P_1)$、k 链只多 1、无限深二叉树 $1/(1-P_2)$），Sequoia 树的期望收益**无界、随规模对数增长**——预算给得越多，吃得越多。证明思路是构造一族 $k(n)$ 叉树作为下界：取分支数 $k(n) = \lfloor \log(n)^{1/b} \rfloor$，其深度约为 $\log_k(n)$，期望接受长度是几何级数求和 $\frac{1-\alpha_k^{d+1}}{1-\alpha_k}$，再用引理 $\frac{1-(1-x)^m}{x} \ge m/10$（当 $mx \le \log 1.9$）放缩出对数下界；而 Sequoia 树是 DP 找到的最优树，自然不差于这个下界。

## 组件二：无放回采样 + 验证，拒绝"重蹈覆辙"

### 算法改动：两处蓝色小修改

Sequoia 的验证算法以 SpecInfer 为骨架，只改了两处：

1. **无放回采样**：从 draft model 分布 $Q$ 采样候选时，被拒绝的 token 直接从 $Q$ 中置零，绝不重复提出；
2. **兜底均匀分布**：如果 $Q$ 支撑集里的 token 全部被拒绝，就把 draft 分布换成"所有未被拒绝 token 上的均匀分布"，继续采样。

验证的接受判据不变：采样 $s_i \sim D$、$r_i \sim \text{Uniform}(0,1)$，若 $r_i < R[s_i]/D[s_i]$ 则接受 $s_i$；否则把残差分布更新为 $R \leftarrow \text{norm}(\max(R - D, 0))$ 并继续。全部拒绝时从残差 $R$ 采样收尾。

论文在附录中给出了正确性证明（对提议轮数归纳），核心等式是第 n 轮的分解：

$$
t_n(u) = d_n(u) \cdot \min\Big(1, \frac{t_n(u)}{d_n(u)}\Big) + \Big(\sum_x \max(0, t_n(x) - d_n(x))\Big) \cdot t_{n+1}(u)
$$

即"残差分布 = 本轮接受概率 + 全部拒绝概率 × 下一轮残差"，无论 draft 分布如何依赖历史（这正是无放回采样合法的关键），输出分布都与目标模型严格一致。

### 理论：第一个同时满足两条鲁棒性质的算法

论文提出两条理想的鲁棒性质：

- **最优传输性质**（optimal transport）：$k=1$ 时接受率等于 $1 - \|P - Q\|_1 / 2$，这是 $k=1$ 下的理论最优（SpecTr 论文已证）。满足它意味着**高温度**下表现好（此时 $P$、$Q$ 都接近均匀分布）；
- **覆盖性质**（cover）：若 $Q$ 的支撑集大小为 k 且覆盖 $P$ 的支撑集，最多 k 次提议就能把接受率推到 1；k 等于词表大小时接受率恒为 1。满足它意味着**低温度**下表现好。

**定理**：Sequoia 同时满足两条性质；SpecInfer 和 SpecTr 只满足最优传输性质；top-$k$ 采样只满足覆盖性质。

证明很干脆：$k=1$ 时 Sequoia 退化为原始投机解码，天然满足第一条；无放回采样保证 k 次提议恰好穷尽 $Q$ 的支撑集，且残差的支撑只会缩小、被拒绝的 token 会退出残差，因此第 k 次提议必被接受，满足第二条。反例也很好构造：取 $Q = [0.5, 0.5]$、$P = [1.0, 0]$，SpecInfer 有 25% 概率把第二个 token 连采两次、双双被拒——这就是"有放回"的原罪。

![拒绝率对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/figure3.png)

> 图解：横轴是投机的孩子 token 数 k，纵轴是平均拒绝率（1 − 接受率），三个子图分别对应温度 0.2、0.6、1.0（WikiText-103 上测得）。可以看到 Sequoia 的拒绝率在所有温度下都下降最快，且近似满足幂律 $r_k \approx 1/k^b$——这正是树构造理论所依赖的假设。SpecInfer 与 SpecTr 在高温下尚可、低温下明显挣扎，top-$k$ 采样则相反。

**编辑点评**：这个设计的聪明之处在于，"无放回"听起来是显而易见的小改动，但它恰好补上了 SpecInfer 的短板，又通过残差分布的归纳证明保住了"输出分布不变"这条投机解码的生命线。理论和工程价值同时落袋。

## 组件三：硬件感知的树优化器

解决了"树怎么长"之后，下一个问题是"树该多大、多深"。Sequoia 的加速比可以分解为：

$$
\texttt{Speedup}(n, d) = \frac{G(n, d)}{t(n) + d \cdot c}
$$

- $G(n, d)$：大小为 n、深度为 d 的树期望生成的 token 数（DP 直接算）；
- $t(n)$：目标模型验证 n 个 token 的耗时 ÷ 验证 1 个 token 的耗时（硬件相关）；
- $c$：draft model 生成 1 个 token 的耗时 ÷ 目标模型验证 1 个 token 的耗时（硬件相关）。

以往方法可以偷懒令 $t(n) = 1$，因为旧树结构收益早早饱和，最优 n 出现在 $t(n)$ 尚未明显增长的区间。但 Sequoia 的 $G(n, d)$ 是对数级增长的，最优树会大到 $t(n)$ 不能再近似为常数——于是必须**在目标硬件上实测** $t(n)$ 和 $c$，再对 n、d 做小规模搜索取加速比最大者。

![不同硬件上的前向耗时曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/t_n_unnormalized.png)

> 图解：横轴是一次前向处理的 token 数 n，纵轴是前向耗时，不同曲线对应不同模型/硬件组合。可以看到 n 较小时耗时近似常数（memory-bound 区间），超过某个拐点后近似线性增长（compute-bound 区间）；带宽/FLOPS 比值越高的硬件拐点来得越早。正是这条实测曲线决定了最优树的大小。

## 实验：端到端最高 4.04×（A100）/ 9.5×（L40 offloading）

### 实验设置

- **on-device 场景**：draft model 用 JackFram/Llama-68M（JF68M）和 Sheared-Llama-1.3B（SL1.3B），目标模型用 Llama2-7B/13B、Vicuna-33B，跑在 A100-PCIE-80GB 与 L40 上；
- **offloading 场景**：draft model 用 Llama2-7B-chat / Llama3-8B-Instruct，目标模型用 Llama2-70B-chat / Llama3-70B-Instruct，单张 L40（PCIE4），基线是 DeepSpeed-Zero-Inference（5.7 s/token）；
- **数据集**：C4(en)、OpenWebText、CNN DailyMail、MT Bench；prompt 与生成长度均为 128 token（MT Bench 除外）；
- **实现**：基于 HuggingFace Transformers，用 PyTorch CUDA Graphs 消除 kernel launch 开销；无放回采样用 exponential-sort 算法（Gumbel trick）加速；
- **对比对象**：SpecInfer，on-device 用 $5 \times 8$ 树，offloading 用 $16 \times 48$ 树。

### 端到端结果（A100 on-device）

基准的增量解码速度是 24.2 ms/token（HuggingFace），draft model 为 0.5 ms/token。下表中括号内是每步平均生成 token 数，TBT 为 time between tokens：

| 目标模型 | T | 数据集 | 树配置 (size, depth) | Sequoia 加速比 | TBT (ms/token) | SpecInfer 5×8 | 相对 SpecInfer |
|---|---|---|---|---|---|---|---|
| Llama2-7B | 0 | C4 | (128, 10) | **4.04×** (5.08) | 6.0 | 3.45× (3.96) | 1.17× |
| Llama2-7B | 0.6 | C4 | (128, 7) | 3.18× (3.92) | 7.6 | 2.47× (2.97) | 1.29× |
| Llama2-7B | 0 | OpenWebText | (128, 7) | 3.22× (3.86) | 7.5 | 2.79× (3.15) | 1.15× |
| Llama2-7B | 0.6 | OpenWebText | (128, 6) | 2.71× (3.33) | 8.9 | 2.10× (2.54) | 1.29× |
| Llama2-7B | 0 | CNN Daily | (128, 7) | 3.41× (4.05) | 7.1 | 2.95× (3.27) | 1.16× |
| Llama2-7B | 0.6 | CNN Daily | (128, 6) | 2.83× (3.45) | 8.5 | 2.11× (2.58) | 1.34× |
| Llama2-7B | 0 | MT Bench | (128, 10) | 4.03× (4.98) | 6.0 | 3.84× (4.01) | 1.05× |
| Llama2-7B | 0.6 | MT Bench | (128, 7) | 3.18× (3.96) | 7.6 | 2.45× (2.97) | 1.30× |

附录还给出了更大模型与 L40 的完整结果：Llama2-13B 在 A100 上最高 3.73×，Vicuna-33B 最高 2.27×；L40 on-device 上 Llama2-7B 最高 3.95×。目标模型越大，加速比相对越低，因为验证时间随候选数增长得更快，最优树更小。

### 端到端结果（L40 offloading）

offloading 是 Sequoia 的主场——验证一步的成本极高（5.7 s/token），而加大树的边际成本很低，可扩展树结构的红利被充分释放：

| 目标模型 | Draft 模型 | T | 树配置 (size, depth) | Sequoia 加速比 | TBT (s/token) | SpecInfer 16×48 | 相对 SpecInfer |
|---|---|---|---|---|---|---|---|
| Llama2-70B-chat | Llama2-7B-chat | 0 | (768, 18) | 8.6× (10.30) | 0.66 | 5.7× (7.63) | 1.51× |
| Llama2-70B-chat | Llama2-7B-chat | 0.6 | (768, 18) | 8.4× (9.91) | 0.68 | 5.2× (7.03) | 1.62× |
| Llama3-70B-Instruct | Llama3-8B-Instruct | 0 | (768, 18) | **9.5×** (11.68) | 0.60 | 7.0× (9.07) | 1.36× |
| Llama3-70B-Instruct | Llama3-8B-Instruct | 0.6 | (768, 18) | 9.3× (11.37) | 0.61 | 6.1× (8.29) | 1.52× |

70B 模型在单张 L40 上跑到了 0.60 s/token，比 SOTA offloading 系统快 9.5 倍——这是全文最硬的一个数字。

几个值得注意的分析观察：

1. offloading 场景选出的树（768 个节点）远大于 on-device（64–128），印证了"越 memory-bound，树越该大"；
2. 平均生成 token 数与墙钟加速比非常接近，说明硬件感知优化器把账算得很准；
3. 温度、模型对稍有变化，最优树结构可能大不相同；
4. 低温度下选更深的树，因为低温接受率更高，深的链更划算。

## 消融实验：可扩展性与鲁棒性

### 树构造的可扩展性

固定采样验证算法（都用 Sequoia 的），只换树结构，对比"Sequoia 树"与"k 条独立序列"在不同预算下的平均每步生成 token 数（draft=JF68M，target=Llama2-13B，T=0.6，CNN DailyMail）：

![可扩展性对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sec4_scalability.png)

> 图解：横轴是树的 token 预算，纵轴是平均每步生成的 token 数。Sequoia 树的曲线持续增长，而 k 条独立序列早早走平；预算 512 时 Sequoia 平均多生成 **33%** 的 token。

附录中更多模型对（Pythia-410M/2.8B 对 Pythia-12B、Llama2-7B 对 Llama2-70B，WikiText-103）呈现同样的趋势：

![更多模型对的可扩展性](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/scalability_Pythia-410m_Pythia-12b_1.0.png)

> 图解：Pythia-410M → Pythia-12B 组合下，Sequoia 树（蓝线）的生成 token 数随树规模近似对数增长，序列、k 条独立序列、二叉树等基线结构则相继饱和。

![Pythia-2.8B 草稿下的可扩展性](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/scalability_Pythia-2.8b_Pythia-12b_1.0.png)

> 图解：换用更强的 Pythia-2.8B 作 draft，接受率整体提高，所有结构的绝对收益上升，但只有 Sequoia 保持无界增长。

![Llama2 模型对的可扩展性](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/scalability_Llama2-7b_Llama2-70b_1.0.png)

> 图解：Llama2-7B → Llama2-70B 组合同样如此。这解释了为什么 offloading 场景（一次验证 768 个节点也不亏）里 Sequoia 优势最大。

### 采样验证算法的鲁棒性

固定树结构（A100 表中 T=0.6 对应的 Sequoia 树），只换验证算法，扫不同温度（draft=JF68M，target=Llama2-7B，CNN DailyMail）：

![鲁棒性对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sec4_robustness.png)

> 图解：横轴是温度，纵轴是墙钟加速比。Sequoia 在所有温度下都取得最大加速，相对 SpecInfer 最高快 **1.65×**，相对 top-$k$ 采样最高快 **1.27×**；SpecInfer 低温下滑明显，top-$k$ 采样高温下滑明显，与前文理论分析完全吻合。

对 top-$p$ 的鲁棒性（A100，加速比 + 括号内生成 token 数）：

| Top-$p$ | Sequoia (本文) | SpecInfer | top-$k$ 采样 |
|---|---|---|---|
| 0.8 | 2.54× (3.18) | 2.35× (2.93) | 2.43× (2.90) |
| 0.9 | 2.61× (3.27) | 2.42× (3.01) | 2.27× (2.71) |
| 1.0 | 2.69× (3.26) | 2.55× (3.10) | 2.12× (2.44) |

### 硬件感知优化器的效果

![硬件感知优化器](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sec4_hardware_aware.png)

> 图解：橙色曲线是"只最大化生成 token 数、不限深度"的 Sequoia 树在不同大小下的墙钟加速比，绿色横线是硬件感知优化器选出的树的加速比（A100 与 L40 上分别测量）。优化器通过限制树深让投机更快，同时自动找到使总加速比最大的树大小，超过所有不限深的候选。

### 与 SpecInfer 全配置扫描的对比

为排除"SpecInfer 只是树没调好"的质疑，作者在 A100 上扫了 SpecInfer 所有宽度×深度组合（draft=JF68M，target=Llama2-7B，C4）。贪心解码下 SpecInfer 最好的配置也只有 3.70×（8×8），默认 5×8 为 3.45×，而 Sequoia 达 4.04×；随机解码下 SpecInfer 最好 2.42×，Sequoia 达 3.18×。L40 offloading 上 SpecInfer 扫到的最好配置（32×24）为 5.5×，Sequoia 为 8.4×。即 **Sequoia 超过 SpecInfer 所有手工树配置**，赢的是结构搜索本身。

### Sequoia 树长什么样

附录给出了用 Llama3-70B-Instruct（目标）+ Llama3-8B-Instruct（草稿）在 CNN DailyMail 上实测接受率向量后构造的不同规模 Sequoia 树：

![8 与 16 节点的 Sequoia 树](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sequoia_tree_strong_n_8.png)

> 图解：8 节点的小树几乎就是一条带少量分支的链——预算紧张时，把 token 花在"主干的延伸"上最值。

![64 节点的 Sequoia 树](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sequoia_tree_strong_n_64.png)

> 图解：64 节点时树开始呈现"主干深、分支逐渐增多"的形态。

![256 节点的 Sequoia 树](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Scalable-and-Robust-Speculative-Decoding/icml2024/figures/sequoia_tree_strong_n_256.png)

> 图解：256 节点时树的骨架依然是一条很深的主干，但沿途的分支密度明显增加——这正是接受率向量（第 1 个孩子接受率 0.77，其后迅速衰减）在结构上的投影。

## 局限与讨论

- **位置接受假设**：DP 的最优性依赖"接受率只取决于孩子位置"这一简化，忽略了上下文、draft 置信度等因素；
- **幂律假设**：可扩展性定理依赖 $r_k \le 1/k^b$，实验上成立但仍是假设；
- **工程成本**：最优树依赖具体的接受率向量，而后者随模型对、温度、数据域、batch size 变化，每种设置都要先实测再建树，较为繁琐。作者的经验是"为典型场景算一棵树，迁移到其他场景通常也 work"，但更系统的分析留给了未来。

## 总结

- **问题**：现有投机解码的树结构期望收益有上界、验证算法对温度不鲁棒，预算大时浪费严重；
- **树构造**：把建树变成以 $F(\mathcal{T}) = \sum_v f(v)$ 为目标的约束优化，用 DP 精确求解，理论上期望生成 token 数达 $\Omega(b \log n / \log \log n)$，无界增长；
- **采样验证**：无放回采样 + 拒绝兜底均匀分布，是第一个同时满足"最优传输"与"覆盖"两条鲁棒性质的验证算法，且严格保持目标模型输出分布；
- **硬件感知**：用 $\texttt{Speedup}(n,d) = G(n,d) / (t(n) + d \cdot c)$ 在目标硬件上实测选型，避免穷举；
- **结果**：A100 上 Llama2-7B 最高 4.04×，L40 offloading 上 Llama3-70B-Instruct 最高 9.5×（0.60 s/token），比 DeepSpeed-Zero-Inference 快近一个数量级，代码已开源。

一句话展望：Sequoia 把投机解码的"结构上限"往上推了一个量级，而它揭示的接受率幂律与硬件权衡，也为下一代推理系统乃至专用推理芯片的设计提供了理论坐标。

> 本文参考自 [Sequoia: Scalable and Robust Speculative Decoding](https://arxiv.org/abs/2402.12374)