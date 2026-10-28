# Lookahead Decoding：8 卡提速 4 倍

LLM 自回归解码一次只前进一步，既让生成时延随 Token 数线性增长，又因受内存带宽限制而浪费 GPU 的并行算力。对于聊天、搜索和代码生成等长文本场景，这意味着用户等待时间难以随硬件升级自然下降。Lookahead Decoding 的思路是：让模型沿 Jacobi 迭代轨迹并行“预习”多组 n-gram，再在同一步中验证并接收正确片段，全程不需要草稿模型或外部数据库。它在 MT-Bench 上最高提速 1.8 倍，并通过 8 张 GPU 的 Lookahead Parallelism，在代码补全任务上达到约 4 倍提速。

## 自回归解码为什么慢

主流 Decoder-only LLM 的计算可以分成两类：MLP、归一化等逐 Token 模块，以及让 Token 交换信息的 Attention 模块。Attention 的基本形式为：

$$
O = \mathrm{softmax}(QK^T)V
$$

Causal Attention 会给 $QK^T$ 加下三角掩码，使位置 $i$ 只能读取位置 $j \leq i$ 的信息。其他模块又不跨 Token 交换信息，因此第 $i$ 个输出只由前 $i$ 个输入决定。这保证了自回归生成的因果关系，却也形成了严格的串行依赖。

给定 Prompt $\mathbf{x}^0$，贪心解码生成第 $i$ 个 Token 的过程是：

$$
y_i = \arg\max P_M(y_i \mid y_{1:i-1}, \mathbf{x}^0)
$$

每生成一个 Token，都要执行一次完整的前向计算。更关键的是，小批量解码通常不是算力不够，而是反复读取模型权重和 KV Cache 受到内存带宽限制：GPU 有大量空闲计算单元，却无法同时推进后面的位置。

### Speculative Decoding 的启发与包袱

Speculative Decoding 采用“猜测—验证”模式：先用较小的 Draft Model 连续猜出多个 Token，再让原始 LLM 并行验证。验证多个 Token 的成本接近生成一个 Token，因此猜测足够准时就能提速。

问题在于，收益高度依赖 Token 接受率。草稿模型一旦猜错，错误位置及其后的候选都要丢弃；训练高接受率的草稿模型并不简单，而且它还难以跨基础模型和数据集泛化。

围绕“候选从哪里来”，已有工作走出了几条路线：

- SpecInfer 组合多个经过蒸馏、量化或剪枝的草稿模型。
- Medusa、OSD、EAGLE 通过训练获得预测能力。
- REST 从微调数据构建检索库，Prompt Lookup 类方法则直接参考 Prompt。
- Lookahead Decoding 不再引入外部候选来源，而是让目标 LLM 自己并行生成候选，再通过验证保持原输出分布。

理解了“候选必须便宜、验证必须严格”这对矛盾，就可以看清这篇论文真正的切入点：能否把原本闲置的算力变成候选生成能力？

## Jacobi 解码：并行生成，却不会自然提速

答案藏在自回归解码的另一种数学表述里。定义：

$$
f(y_i, y_{1:i-1}, \mathbf{x}^0)
= y_i - \arg\max P_M(y_i \mid y_{1:i-1}, \mathbf{x}^0)
$$

生成 $m$ 个 Token 就等价于求解 $f=0$ 的非线性方程组。Jacobi 解码先随机初始化整段序列 $\mathbf{y}^0$，再用上一轮的所有值并行更新下一轮：

$$
y_i^t = \arg\max P_M\left(y_i \mid y_{1:i-1}^{t-1}, \mathbf{x}^0\right)
$$

可以把它理解为：不再按座位顺序逐个点名，而是让整排人每轮同时修正答案，直到所有位置不再变化。

这种方法有两个重要性质：

1. 每轮第一个尚未确定的 Token 与普通自回归结果一致，所以最多 $m$ 轮就能收敛。
2. 相邻两轮、相邻位置上的 Token 可能恰好组成正确的连续片段，因此一轮有机会确定多个 Token。

但原始 Jacobi 解码很难转化为实际加速。并行生成的 Token 经常落在错误位置；即使某轮出现了正确 Token，后续迭代也可能把它覆盖掉。最终算得更多，墙上时间却没有下降。

Lookahead Decoding 的关键转变，就是不再只盯某一轮的最终序列，而是保存并利用整条迭代轨迹。

## Lookahead Decoding：把迭代轨迹变成候选库

![Lookahead Decoding 的候选生成、验证、收集与窗口更新流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/flow.png)

> 图解：以 $W=5$、$N=3$、$G=2$ 为例，每一步包含四个动作。Lookahead Branch 在 5 个未来位置并行生成 Token；Verification Branch 从 n-gram 池中取出候选并验证；系统把轨迹中新形成的 3-gram 收集进池；最后删除过期 Token、加入新 Token，让窗口大小保持不变。数字表示相对位置，不同颜色表示来自不同迭代步的 Token。

整个算法由两条分支和一个缓存组成：

- Lookahead Branch：沿时间维度和序列维度维护二维窗口，并行生成多组互不连接的 n-gram。
- n-gram Pool：缓存历史轨迹产生的片段，避免正确候选被后续迭代覆盖。
- Verification Branch：从池中挑选可能接在当前答案后的候选，用目标 LLM 严格验证。

笔者认为，这个设计最聪明的地方是把 Jacobi 解码的“缺点”变成了信息来源。位置暂时不正确并不可怕，只要轨迹中的片段未来能被重新检索并通过验证，它就不是无效计算。

### Lookahead Branch：用二维窗口生成 n-gram

二维窗口由两个参数控制：

- $W$：向未来看多少个位置，即每步并行解码的窗口宽度。
- $N$：向过去看多少轮轨迹，并用连续 $N$ 轮的信息组成 n-gram。

原始 Jacobi 解码只使用上一轮历史，相当于只能自然形成 2-gram。Lookahead Branch 使用前 $N-1$ 轮历史，因此可以生成更长的候选。

以 $W=5$、$N=4$ 为例，当前步会回看 3 轮，并在 5 个未来位置同时生成新 Token。沿轨迹依次取 $t-3$ 轮位置 1、$t-2$ 轮位置 2、$t-1$ 轮位置 3 的 Token，再加上当前新生成的 Token，就能组成一个 4-gram。每一步还会移除时间维度和序列维度上最旧的 Token，再追加新结果，保持计算窗口固定。

这样做并非直接宣布这些 n-gram 是答案，而是低成本地积累“可能用得上的连续片段”。是否接收，完全交给验证分支决定。

### Verification Branch：保证结果分布不变

验证时，系统先检查 n-gram 的首 Token，筛选出能接在当前序列最后一个 Token 后面的候选，再把这些候选并行送入目标 LLM。

在贪心解码下，验证逻辑与 Speculative Decoding 类似：逐位置比较候选 Token 与模型在该位置的 argmax 输出。匹配就继续，不匹配就丢弃当前候选；所有候选都失败时，至少接收模型自己算出的一个 Token，保证每一步都会前进。如果 $N-1$ 个候选 Token 全部通过，还可以再接收模型额外算出的一个 Token，因此单步最多前进 $N$ 个 Token。

多个候选还要共享前缀。每接收一个 Token，算法就删除前缀不再匹配的 n-gram，确保下一轮使用相同的条件概率分布。

候选池会随生成过程增大，所以论文用 $G$ 限制每步并行验证的最大候选数。实验上建议让 $G$ 随 $W$ 调整，并直接采用 $G=W$，平衡候选生成与验证成本。

### Sampling 为什么也能保持原分布

普通采样比贪心验证多一个问题：候选被拒绝后，需要更新剩余概率分布。若保存池中每个 n-gram 的完整词表概率，会消耗巨大内存。

论文利用了一个关键事实：验证后的输出分布不依赖候选原本如何采样，候选生成方式只影响接受率。因此 Lookahead Branch 始终用贪心方式生成 n-gram，草稿分布退化为 one-hot 向量，只需记录被选中的 Token。

设目标模型当前分布为 $P$，第 $j$ 个候选 Token 为 $s_j$。算法以 $P(s_j)$ 的概率接受它；若拒绝，就把该 Token 的概率置零并重新归一化：

$$
P_{j+1}(v) =
\begin{cases}
0, & v=s_j \\
\dfrac{P_j(v)}{1-P_j(s_j)}, & v\neq s_j
\end{cases}
$$

所有候选都被拒绝时，从最后归一化的分布中采样。附录用数学归纳法证明：若有 $G$ 个候选，$a_j(v)$ 表示 $v$ 在第 $j$ 个候选处被接受的概率，$r_j$ 表示第 $j$ 个候选被拒绝的概率，那么最终采样到 $v$ 的概率为：

$$
Q_G(v)=a_1+r_1a_2+\cdots+a_G\prod_{k=1}^{G-1}r_k
+a'_{G+1}\prod_{k=1}^{G}r_k
$$

当 $G=1$ 时，无论 $s_1=v$ 还是 $s_1\neq v$，都有 $Q_1(v)=P(v)$。增加一个候选时，新接受项、新回退项与被替换的旧回退项相互抵消，因此 $Q_{G+1}(v)=Q_G(v)=P(v)$。再配合逐轮筛除不匹配前缀的规则，整段采样过程都保持目标 LLM 的原分布。

这也是论文所说的“exact”或“lossless”：不是让候选猜得差不多，而是最终输出仍由目标模型的分布决定。

### 同一步完成生成与验证：定制 Attention Mask

两条分支逻辑上不同，执行上却能合并进同一次前向计算。难点是普通下三角 Causal Mask 无法表达二维轨迹以及候选之间的独立关系，必须设计专用掩码。

![普通 Causal Mask 与 Lookahead Decoding 专用 Attention Mask](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/PadAttention.png)

> 图解：左图是普通下三角掩码，黄色表示允许的 Token 交互，紫色表示屏蔽区域。右图展示 $W=5$、$N=4$、$G=2$ 时的掩码：Lookahead Branch 中每条斜向轨迹只读取自己需要的历史；Verification Branch 独立验证两组候选；两条分支互不可见。上方数字仍表示相对位置，颜色区分不同历史步。

掩码遵循两条原则：每个 Token 只能看见满足因果条件的历史 Token；Lookahead 与 Verification 两条分支不能互相读取。这样既能在一次 LLM 前向中完成全部工作，又不会让候选之间发生不应有的信息泄漏。

标准 FlashAttention 默认使用固定下三角掩码，不能直接套用。作者把可调 $W$、$N$、$G$ 对应的注意力模式写入 FlashAttention 实现，在保留内存高效优势的同时支持该掩码。相比原生 PyTorch 实现，端到端速度进一步提升约 20%。

到这里，单 GPU 上的算法已经完整：候选来自模型自身，验证保证无损，两条分支共享一次前向。下一个问题自然变成：多条互不依赖的分支，能否进一步拆到多张 GPU 上？

## Lookahead Parallelism：按 Token 分支拆到多张 GPU

论文提出的 Lookahead Parallelism（LP）直接把不同计算分支分配给不同设备。

Lookahead Branch 中的多条轨迹互不交互，可以放到不同 GPU；Verification Branch 中各 n-gram 的验证也彼此独立，同样能够拆分。Forward Pass 期间无需通信，只需在计算结束后同步生成的 Token。

![将 Lookahead 与 Verification 分支分配到 4 张 GPU](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/dist.png)

> 图解：图中的 4 张 GPU 分别承接不同长度的 Lookahead 轨迹和不同验证候选。蓝色输入 Token 与部分早期历史会在设备间冗余复制，但每条分支在 Forward Pass 内独立计算，从而避免逐步通信。

LP 与 Tensor Parallelism（TP）、Pipeline Parallelism（PP）有本质区别：TP、PP 切分模型参数或状态，LP 则在每张 GPU 上保留完整模型副本，切分的是 Token 分支。

代价是更多显存；收益是解码关键路径上的通信接近零。对单请求、低时延推理而言，这比增加模型并行通信更符合需求。获得更多 GPU 算力后，还可以同步增大 $W$、$N$、$G$，进一步压缩解码步数。

## Scaling Law：用对数增长的算力换线性减少的步数

增加窗口和候选数并不是无条件提速。论文因此建立了一个简化模型，解释计算量、接受率与步数压缩之间的关系。

### 从 Speculative Decoding 的接受 Token 数开始

设每个候选 Token 的接受率为随机变量 $\beta$，其期望 $E(\beta)=\alpha$，每轮猜测 $\gamma$ 个 Token。单条候选序列的期望接收 Token 数为：

$$
E(\#\mathrm{tokens})=\frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$

附录的推导很直观：只接收 1 个 Token 的概率是 $1-\alpha$，接收 $i\leq\gamma$ 个 Token 的概率是 $(1-\alpha)\alpha^{i-1}$，全部猜中时接收 $\gamma+1$ 个。将这些项加权求和并错位相消，就得到 $1+\alpha+\cdots+\alpha^\gamma$。

如果每轮独立猜测 $b$ 条序列，定义：

$$
p_i=(1-\alpha^i)^b
$$

它表示所有候选最多连续通过 $i$ 个 Token 的概率。利用 $P(\#\mathrm{accepted}=i)=p_i-p_{i-1}$ 求期望，可得：

$$
E(\#\mathrm{tokens})
=(\gamma+1)-\sum_{i=1}^{\gamma}(1-\alpha^i)^b
$$

当 $b=1$ 时，该式退化为单候选公式。这个推导说明，多候选能够突破单条猜测序列的期望上限，但也会增加每步计算。

### Lookahead Decoding 的步数压缩率

论文定义 Step Compression Ratio $S$：生成同样长度的序列，自回归步数与 Lookahead 步数之比。由于自回归一步生成一个 Token，也可写成：

$$
S=\frac{\#\mathrm{generated\ tokens}}{\#\mathrm{Lookahead\ steps}}
$$

把 $G=W=b$、$N-1=\gamma$ 代入多候选公式。实际生成并非每步都有优质候选，因此论文假设平均每 $f$ 步出现一次好候选，其余 $f-1$ 步退化为普通自回归，于是：

$$
S=\frac{f-1+E(\#\mathrm{tokens})}{f}
$$

每步 FLOPs 大致与输入 Token 数 $(W+G)(N-1)$ 成正比。在 $N$ 足够大的理想条件下，解码步数可随每步 $\log(\mathrm{FLOPs})$ 的增加而近似线性下降；反过来说，要让步数持续线性减少，需要指数级增加每步计算量。

![Lookahead Decoding 的步数压缩率与参数规模关系](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/scaling-law.png)

> 图解：左图横轴是实测 $S$，纵轴是 $\log_e(W)$，颜色对应不同 $N$；数据来自 LLaMA-2-Chat-7B 在 MT-Bench 上的实验，且 $G=W$。右图横轴是理论值 $(f-1+E(\#\mathrm{tokens}))/f$，纵轴是 $\log_e(b)$，曲线对应不同 $\gamma$；这里取 $\alpha=0.425$、$f=3.106$。实验散点与理论趋势在较大 $\gamma$ 下较为一致，呈现“对数计算投入换步数压缩”的关系。

这一 Scaling Law 是 LP 能有效利用多 GPU 的理论基础，也同时预示了后文的限制：算力不富裕时，继续扩大窗口很快会遇到收益递减。

## 实验：哪些任务提速最多

### 模型、硬件与数据集

实验覆盖 LLaMA-2 与 CodeLLaMA 的 7B、13B、34B、70B 版本。除特别说明外，均使用 FP16、Batch Size 1，并以 HuggingFace Greedy Search 为主要基线。

两套测试环境如下：

- S1：NVIDIA A100 80GB；7B、13B、34B 使用单卡，70B 使用 2 卡 Pipeline Parallelism。
- S2：带 NVLink 的 DGX，配备 8 张 NVIDIA A100 40GB，用于多 GPU 与 FlashAttention 实验。

| 服务器 | 并行方式 | 模型 | 模型规模 | 数据集 |
|---|---|---|---|---|
| S1 | 无 LP | LLaMA-2-Chat | 7B、13B、70B | MT-Bench |
| S1 | 无 LP | CodeLLaMA | 7B、13B、34B | HumanEval |
| S1 | 无 LP | CodeLLaMA-Inst | 7B、13B、34B | MBPP、GSM8K |
| S2 | 使用 LP | LLaMA-2-Chat | 7B、13B | MT-Bench |
| S2 | 使用 LP | CodeLLaMA | 7B、13B | HumanEval |
| S2 | 使用 LP | CodeLLaMA-Python | 7B、13B | ClassEval |

任务既包含多轮对话，也包含数学、指令代码生成、代码补全与填充：MT-Bench、GSM8K、HumanEval、MBPP、ClassEval。GSM8K 取前 1000 道题；HumanEval 与 ClassEval 的最大生成长度分别设为 512 和 2048。Sampling 质量实验另使用 XSum 与 CNN/Daily Mail。

### 单卡端到端性能：代码任务更受益

![多种模型与数据集上的单卡吞吐对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/all_in_one-flat.png)

> 图解：五组子图分别对应 MT-Bench、GSM8K、MBPP、HumanEval Completion 和 HumanEval Infill；纵轴统一为 Tokens/s，蓝色是不使用 Lookahead 的 HuggingFace 基线，橙色是 Lookahead Decoding，柱顶数字是相对加速比。代码补全的 7B、13B 模型分别达到 2.25 倍和 2.26 倍，是图中最高的两组结果。

论文将整体提升概括为约 1.5～2.3 倍。规律很清楚：代码补全通常优于对话任务，小模型通常优于大模型。

前者是因为代码中存在更多重复片段，历史轨迹生成的 n-gram 更容易再次命中；后者是因为大模型每步本身就需要更多 FLOPs，更快触及 GPU 算力上限，留给额外候选计算的余量更少。

### FlashAttention 与多 GPU：LP 把额外算力变成低时延

![7B 模型在 LP、TP、PP 与 FlashAttention 下的吞吐](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/all-7b.png)

> 图解：横轴为 GPU 数量，纵轴为 Tokens/s，任务依次为 ClassEval、HumanEval、MT-Bench。不同颜色比较单卡、TP、LP、PP，以及是否启用 FlashAttention；柱顶倍数均相对于“单 GPU、无 FlashAttention”的自回归基线。7B 模型在 8 卡 LP 加 FlashAttention 下，三项任务分别达到 3.99、3.87、2.18 倍。

单卡 MT-Bench 上，Lookahead 加 FlashAttention 为 1.90 倍，而自回归加 FlashAttention 为 1.07 倍；两者相比约为 1.8 倍，这就是论文摘要中的 MT-Bench 结果。

在强扩展测试中，GPU 数增加但总工作负载不变。TP 与 PP 多数配置只有约 0.72～0.82 倍，通信开销使小批量单请求反而变慢；LP 则能继续把多卡 FLOPs 用于增加候选和窗口，8 卡 ClassEval 接近 4 倍。

![13B 模型在 LP、TP、PP 与 FlashAttention 下的吞吐](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/all-13b.png)

> 图解：坐标、颜色和基线与上一张图相同。13B 模型在 8 卡 LP 加 FlashAttention 下，ClassEval、HumanEval、MT-Bench 分别达到 3.79、3.42、1.97 倍；TP 与 PP 仍未带来多卡加速。相比 7B，提升略低，符合大模型更快触及算力上限的解释。

### 生成质量：Sampling 下保持 ROUGE，速度仍提升

作者在 LLaMA-2-7B-Chat 上测试摘要生成。Temperature 0.0 表示贪心解码，1.0 表示启用 Sampling；评价指标为 ROUGE-1、ROUGE-2、ROUGE-L。

| 数据集 | Temperature | 方法 | ROUGE-1 | ROUGE-2 | ROUGE-L | 加速比 | $S$ |
|---|---:|---|---:|---:|---:|---:|---:|
| CNN/Daily Mail | 1.0 | 自回归 | 36.55 | 13.20 | 22.68 | 1.00× | 1.00 |
| CNN/Daily Mail | 1.0 | Lookahead | 36.53 | 13.27 | 22.71 | 1.46× | 1.64 |
| CNN/Daily Mail | 0.0 | 自回归 | 37.79 | 14.59 | 23.96 | 1.00× | 1.00 |
| CNN/Daily Mail | 0.0 | Lookahead | 37.79 | 14.59 | 23.96 | 1.57× | 1.72 |
| XSum | 1.0 | 自回归 | 19.15 | 4.53 | 12.84 | 1.00× | 1.00 |
| XSum | 1.0 | Lookahead | 19.20 | 4.53 | 12.87 | 1.50× | 1.67 |
| XSum | 0.0 | 自回归 | 19.38 | 4.78 | 13.05 | 1.00× | 1.00 |
| XSum | 0.0 | Lookahead | 19.39 | 4.79 | 13.06 | 1.60× | 1.77 |

Sampling 下仍获得 1.46～1.60 倍提速，ROUGE 基本一致。Sampling 的加速略低于贪心解码，是因为概率接受机制使候选接受率下降。

附录还检查了数值误差与高级集成的影响：

- FP32 下，Lookahead 在 1、4、8 张 GPU 上的 MT-Bench 输出都与 HuggingFace 基线一致。
- FP16 下，HuggingFace 自身相对 FP32 基线也有 35/160（无 FlashAttention）或 42/160（有 FlashAttention）个回答不完全一致；Lookahead 各设置为 35～44 个，未超出这一数值误差量级。
- 开关 FlashAttention 的平均 $S$ 分别为 3.267 与 3.259，差异小于 0.3%；单 GPU 与 LP 的平均 $S$ 分别为 2.558 与 2.557，差异小于 0.1%。

这些结果支持“算法不改变输出分布”的理论结论，但也提醒我们：工程上的逐 Token 完全一致仍会受到浮点精度影响。

### 消融实验：生成与验证必须平衡

作者在 MT-Bench、LLaMA-2-7B-Chat、A100 和 FlashAttention 下比较不同分支配置，并测试把 Prompt 加入 n-gram 池的效果。

| 编号 | 设置 $(N,W,G)$ | 使用 Prompt 作为参考 | 加速比 | $S$ |
|---:|---|---|---:|---:|
| 1 | 自回归 | 否 | 1.00× | 1.00 |
| 2 | Prompt Lookup | 是 | 1.44× | 1.55 |
| 3 | (10, 1, 3) | 是 | 1.36× | 1.45 |
| 4 | (5, 1, 10) | 是 | 1.36× | 1.51 |
| 5 | (5, 1, 30) | 否 | 1.04× | 1.12 |
| 6 | (5, 1, 30) | 是 | 1.46× | 1.59 |
| 7 | (5, 30, 1) | 否 | 1.61× | 1.79 |
| 8 | (5, 15, 15) | 否 | 1.78× | 1.96 |
| 9 | (5, 15, 15) | 是 | **1.88×** | **2.05** |

只看 Lookahead Branch：$W=1$ 时，即使增大 $N$、$G$，也不如平衡配置；说明仅依赖历史候选或 Prompt，候选覆盖仍有限。

只看 Verification Branch：$W=30$、$G=1$ 能生成很多候选，却每步只验证一个，接受机会不足，速度低于 $W=G=15$。

最后，把 Prompt 作为额外参考可进一步提速到 1.88 倍。两条分支不是越大越好，而是要一起扩展、相互匹配。论文的 Transformers v4.37 Prompt Lookup 基线会匹配多个起始 Token，而当时的 Lookahead 检索只精确匹配一个首 Token，因此独立 Prompt Lookup 会略优于部分弱 Lookahead 配置。

## 局限：这是用空闲算力换时延，不是免费午餐

Lookahead Decoding 的代价是每步执行更多 FLOPs。A100 上，论文给出的近优单 Batch 配置如下，并令 $G=W$：

| 模型 | Window Size $W$ | n-gram Size $N$ |
|---|---:|---:|
| 7B | 15 | 5 |
| 13B | 10 | 5 |
| 34B | 7 | 5 |

按每步输入 Token 数 $(W+G)(N-1)$ 估算，7B、13B、34B 分别需要约 120、80、56 倍的每步额外 FLOPs。数字看起来很大，但解码原本受内存带宽限制，很多计算单元本来就在空转，所以额外 FLOPs 只转化为有限的单步变慢。

一旦环境变为 Compute-bound，例如大 Batch Serving，这些额外计算就会直接挤占有效吞吐，甚至导致减速。GPU 算力较弱时，也不能盲目增大窗口。

![RTX 3090 与 A100 上的压缩率和实际加速比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2402.02057/figures/overhead.png)

> 图解：横轴为 $W=G$ 的取值，均固定 $N=5$；左纵轴是 Step Compression Ratio，右纵轴是实际 Speedup。蓝色与橙色的压缩率曲线几乎重合，说明设备不改变算法接收 Token 的能力；绿色 RTX 3090 加速曲线在 $W=G=30$ 时降至约 1 倍，红色 A100 曲线则在较大窗口下趋于平台。相同的算法压缩率，并不会自动变成相同的墙上时间收益。

因此，这套方法最适合单请求、低时延、算力有富余的场景。论文也给出较温和配置下的参考：MT-Bench 上 RTX 3090 容易获得约 30% 提速，A100 可超过 50%；但根据 Scaling Law，继续追求线性减少步数会遭遇指数计算成本与收益递减。

作者已提供 Python 与 CUDA [实现代码](https://github.com/hao-ai-lab/LookaheadDecoding)，并支持 FlashAttention、Sampling 和多 GPU LP。

## 总结：让目标模型自己成为“草稿生成器”

- 核心问题：自回归解码受串行步数与内存带宽限制，现代 GPU 的并行算力未被充分利用。
- 核心方法：用二维 Jacobi 轨迹生成并缓存 n-gram，再在同一步并行验证，不需要草稿模型、训练或外部数据库。
- 核心性质：贪心与 Sampling 验证都保持目标模型输出分布，FlashAttention 与 LP 不改变算法的步数压缩能力。
- 核心结果：MT-Bench 最高提速 1.8 倍，8 卡代码补全约 4 倍；代码任务、小模型和算力富余环境收益更明显。
- 核心边界：收益来自额外 FLOPs，大 Batch 或 Compute-bound 环境可能不适合，盲目扩大窗口会收益递减。

未来更值得探索的方向，是根据任务重复度、模型规模和实时硬件负载动态调整 $W$、$N$、$G$，让这套“算力换时延”的机制在不同服务环境中自动找到平衡点。

> 本文参考自 [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057)