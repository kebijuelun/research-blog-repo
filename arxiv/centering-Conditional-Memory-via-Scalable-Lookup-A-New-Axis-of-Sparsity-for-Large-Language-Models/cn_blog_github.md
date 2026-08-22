# DeepSeek 新作 Engram：给大模型装上"查表记忆"，开辟 MoE 之外的第二条稀疏轴

> 本文参考自 [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/pdf/2601.07372)

今天聊一篇 DeepSeek-AI 与北京大学联合出品的论文。这篇文章提出了一个很有意思的观点：过去这些年，大模型的"稀疏化"几乎只有一条路——MoE（Mixture-of-Experts，混合专家），也就是 **条件计算** （Conditional Computation）。但语言信号里其实有两大类截然不同的东西：一类需要动态推理，另一类只是静态知识。让 Transformer 用宝贵的计算层去"算"一个本该"查"出来的固定搭配，本质上是一种浪费。

为此，作者提出了 **条件记忆** （Conditional Memory）这条与 MoE 互补的稀疏轴，并给出了具体实现模块 **Engram** ——一个现代化的 $N$-gram 嵌入查找模块，支持 $O(1)$ 常数时间检索。基于它，作者不仅发现了一个 U 形的"稀疏分配" Scaling Law，还把模型做到了 27B 规模，在严格同参数、同 FLOPs 的条件下全面超越 MoE 基线。更反直觉的是：收益最大的居然不是知识类任务，而是推理、代码和数学。

下面我们按"提出问题 -> 分析问题 -> 解决问题 -> 实验验证 -> 结论"的逻辑展开。

## 一、问题：Transformer 没有"知识查找"原语

稀疏性（Sparsity）是智能系统中反复出现的设计原则，从生物神经回路一路延续到现代 LLM。目前这条原则几乎完全由 MoE 承载：每个 token 只激活一小部分专家参数，从而在不增加计算量的前提下把模型容量推到数千亿规模，DeepSeek-V3、Kimi-K2 等前沿模型都是这条路线的受益者。

但作者指出，语言建模其实包含两个性质完全不同的子任务：

- **组合式推理** （Compositional Reasoning）：需要深层、动态的计算；
- **知识检索** （Knowledge Retrieval）：大量文本是局部、静态、高度程式化的——比如命名实体、固定搭配、惯用语。

经典 $N$-gram 模型早就证明，这类局部规律天然适合用廉价的"查表"来刻画。然而标准 Transformer 没有原生的查找原语，只能 **用计算来模拟检索** 。论文引用了 PatchScope 的一个经典案例：要让模型在内部"拼出" `Diana, Princess of Wales` 这个实体，最后一个 token `Wales` 的隐状态需要逐层演化——第 1~2 层只知道是"英国的某个地方"，第 3 层变成"欧洲的国家"，第 4~5 层才浮现出"威尔士王妃"的头衔，直到第 6 层才完整解析出戴安娜本人。

换句话说，模型消耗了 **6 层宝贵的序列深度** ，干了一件本质上等价于"查一张静态表"的事。这些层数本可以被省下来，用于更高层的复杂推理。这就是论文要解决的核心痛点。

## 二、思路：条件记忆——与 MoE 正交的第二根稀疏轴

作者的解法很直接：既然 MoE 是"条件计算"（稀疏激活参数来处理动态逻辑），那就再配一根 **条件记忆** 轴——用稀疏查找操作，为静态知识检索固定的嵌入向量。

具体载体选择了经典的 $N$-gram 嵌入：以局部上下文为 key，通过哈希在一张巨大的嵌入表里做 $O(1)$ 查找。听起来很"复古"，但论文的核心发现是—— **这个静态检索机制只要设计得当，就能成为现代 MoE 架构的理想互补品** 。这里的"设计得当"包括 tokenizer 压缩、多头哈希、上下文感知门控、多分支集成等一系列现代化改造，合起来就是 Engram 模块。

随之而来的一个关键问题是：在总参数预算固定时，MoE 专家和 Engram 记忆之间应该怎么分？作者把它形式化为 **稀疏分配** （Sparsity Allocation）问题，并发现了一个 U 形 Scaling Law（后文详解）。

## 三、Engram 架构：检索 + 融合两步走

Engram 是一个即插即用的条件记忆模块，挂在 Transformer backbone 的特定层上（不是每层都挂），结构上把"静态模式存储"和"动态计算"彻底分开。给定输入序列 $X = (x_1, \dots, x_T)$ 和第 $\ell$ 层的隐状态 $\mathbf{H}^{(\ell)} \in \mathbb{R}^{T \times d}$，每个位置 $t$ 经过两个阶段： **检索** （Retrieval）与 **融合** （Fusion）。

### 3.1 稀疏检索：Tokenizer 压缩 + 多头哈希

**Tokenizer 压缩。** 标准 subword tokenizer 追求无损重建，会给语义等价的串分配完全不同的 ID（比如 `Apple` 和 ` apple`）。Engram 预计算一个满射映射 $\mathcal{P}: V \to V'$，基于 NFKC 归一化、小写化等文本等价规则，把原始 token ID 折叠成"规范 ID"。这一步对 128k 词表实现了约 **23%** 的有效词表压缩（附录里有个很有意思的 case：排名第一的合并项把 `\t`、`\n`、空格、双空格等 163 个 token 全归并成一个规范 ID）。压缩后，位置 $t$ 的后缀 $N$-gram 记为：

$$g_{t,n} = (x'_{t-n+1}, \dots, x'_t), \quad x'_t = \mathcal{P}(x_t)$$

**多头哈希。** 直接参数化所有可能的 $N$-gram 组合空间是不可行的，所以采用哈希技巧。为缓解哈希冲突，每个 $N$-gram 阶数 $n$ 配 $K$ 个独立的哈希头，每个头 $k$ 用一个确定性的轻量乘法-XOR 哈希函数 $\phi_{n,k}$，把压缩后的上下文映射到一张素数大小 $M_{n,k}$ 的嵌入表 $\mathbf{E}_{n,k}$ 中：

$$z_{t,n,k} \triangleq \phi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]$$

最终的记忆向量由所有阶数、所有头的检索结果拼接而成：

$$\mathbf{e}_t \triangleq \mathop{\Vert}_{n=2}^{N} \mathop{\Vert}_{k=1}^{K} \mathbf{e}_{t,n,k}$$

### 3.2 上下文感知门控：给静态记忆装上"开关"

查出来的 $\mathbf{e}_t$ 是上下文无关的静态先验，天然有两个问题：缺乏上下文适应性，且可能被哈希冲突或一词多义污染。为此作者设计了一个受 Attention 启发的门控机制：用当前隐状态 $\mathbf{h}_t$（已经过前面 Attention 层聚合了全局上下文）当动态 Query，检索到的记忆 $\mathbf{e}_t$ 同时作为 Key 和 Value 的来源：

$$\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t$$

为保证梯度稳定，Query 和 Key 先做 RMSNorm，再算标量门 $\alpha_t \in (0,1)$：

$$\alpha_t = \sigma\left( \frac{\mathrm{RMSNorm}(\mathbf{h}_t)^\top \, \mathrm{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}} \right)$$

门控输出为 $\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$。这个设计的语义很明确： **如果查到的记忆和当前上下文矛盾，门就趋向 0，噪声被自动抑制** 。

最后，为了扩大感受野并增强非线性，再接一个短的深度因果卷积（kernel size $w=4$，dilation $\delta$ 取最大 $N$-gram 阶数），配 SiLU 激活和残差：

$$\mathbf{Y} = \mathrm{SiLU}\left( \mathrm{Conv1D}\left( \mathrm{RMSNorm}(\tilde{\mathbf{V}}) \right) \right) + \tilde{\mathbf{V}}$$

Engram 模块通过残差连接注入 backbone：$\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$，之后照常接 Attention 和 MoE。卷积参数零初始化，保证训练起点严格是恒等映射。

### 3.3 与多分支架构（mHC）的集成

本文的默认 backbone 不是标准单流残差，而是更先进的多分支架构（Manifold-Constrained Hyper-Connections，mHC），残差流被扩展成 $M$ 条并行分支（实验中 $M=4$）。为适配它，Engram 采用了一种参数共享策略： **所有分支共享同一张稀疏嵌入表和同一个 Value 投影矩阵 $\mathbf{W}_V$，但每个分支有独立的 Key 投影 $\{\mathbf{W}_K^{(m)}\}_{m=1}^M$** ，从而获得分支特异的门控行为：

$$\alpha_t^{(m)} = \sigma\left( \frac{\mathrm{RMSNorm}(\mathbf{h}_t^{(m)})^\top \, \mathrm{RMSNorm}(\mathbf{W}_K^{(m)} \mathbf{e}_t)}{\sqrt{d}} \right)$$

检索记忆被各分支独立的门调制：$\mathbf{u}_t^{(m)} = \alpha_t^{(m)} \cdot (\mathbf{W}_V \mathbf{e}_t)$。工程上，这 1 个 $\mathbf{W}_V$ 加 $M$ 个 $\mathbf{W}_K^{(m)}$ 可以融合成单次稠密 FP8 矩阵乘，最大化 GPU 计算利用率。

### 3.4 系统效率：存算分离是设计的一等公民

这是 Engram 相对 MoE 的一个结构性优势。MoE 的路由依赖运行时隐状态，而 Engram 的检索 **只由输入 token ID 决定** ——地址在前向开始之前就完全确定。这个确定性带来了两级优化空间：

- **训练时** ：巨大嵌入表按标准模型并行切分到各 GPU，用 All-to-All 通信收集前向需要的行、分发反向梯度，总容量随卡数线性扩展；
- **推理时** ：采用"预取-重叠"（prefetch-and-overlap）策略。既然索引提前已知，系统可以在前面 Transformer block 计算的同时，异步通过 PCIe 从主机内存把嵌入搬到 GPU，通信被计算完全掩盖。

此外，自然语言 $N$-gram 服从 Zipf 分布——极少数模式占绝大多数访问。这天然适合 **多级缓存层次** ：高频嵌入放 GPU HBM 或主机 DRAM，长尾放 NVMe SSD，从而把记忆容量扩到极大的同时几乎不影响有效延迟。也正因如此，Engram 的插层位置需要软硬件协同权衡：插得越深，可用于掩盖延迟的计算窗口越大；但消融实验（见 7.2）表明，建模效果偏好尽早注入。

## 四、Scaling Law：MoE 与 Engram 之间怎么分预算？

### 4.1 有限预算下的 U 形分配律

先定义三个参数量口径：$P_{\mathrm{tot}}$（总可训练参数，不含词表嵌入和 LM head）、$P_{\mathrm{act}}$（每 token 激活参数，决定训练 FLOPs）、以及"免费"的非激活参数 $P_{\mathrm{sparse}} \triangleq P_{\mathrm{tot}} - P_{\mathrm{act}}$。分配比例 $\rho \in [0,1]$ 定义为非激活预算中划给 MoE 专家的份额：

$$P_{\mathrm{MoE}}^{(\mathrm{sparse})} = \rho\, P_{\mathrm{sparse}}, \qquad P_{\mathrm{Engram}} = (1-\rho)\, P_{\mathrm{sparse}}$$

$\rho=1$ 就是纯 MoE；$\rho<1$ 则减少路由专家数，把腾出的参数拨给 Engram 嵌入槽。作者在 $2\times10^{20}$ 和 $6\times10^{20}$ 两个 FLOPs 档位、固定稀疏比 $P_{\mathrm{tot}}/P_{\mathrm{act}} \approx 10$ 下做了系统扫描，结果是教科书级的 **U 形曲线** ：

- 即使把 MoE 的份额砍到 $\rho \approx 40\%$，Engram 混合模型仍能打平纯 MoE 基线；
- 纯 MoE（$\rho=100\%$）被证明是 **次优** 的：把约 20%~25% 的稀疏预算挪给 Engram 效果最好，最优点的位置在两个档位下都稳定在 $\rho \approx 75\%\text{--}80\%$；
- 定量上，10B 档（$C=6\times10^{20}$）的验证损失从 $\rho=100\%$ 时的 1.7248 降到最优点附近的 1.7109（$\Delta = 0.0139$）。

U 形两端各自对应一种失效模式：$\rho \to 100\%$ 时模型没有静态模式的专属记忆，被迫用层深去低效重建；$\rho \to 0\%$ 时模型失去条件计算能力，动态推理任务崩塌。记忆不能替代计算，计算也不该替代记忆——两者是结构性互补的。

### 4.2 无限记忆 regime：对数线性的免费午餐

另一个极端：既然 Engram 每 token 只查常数个槽位、FLOPs 不随表大小增长，那把记忆预算放开猛加会怎样？作者固定一个 3B MoE backbone（$P_{\mathrm{act}}=568$M，训 100B tokens），把 Engram 槽位数从 $2.58\times10^5$ 一路扫到 $1.0\times10^7$（相当于白加约 130 亿参数）。

结果是：验证损失随槽位数呈 **严格的幂律下降** （log 空间线性）。也就是说，Engram 提供了一个可预测的扩展旋钮—— **加记忆就涨点，且不需要额外计算** 。对比同样基于哈希 $N$-gram 嵌入的 OverEncoding（直接与词表嵌入做平均），Engram 在同样的记忆预算下释放出的扩展潜力明显更大。

## 五、大规模预训练：27B 同参同算力全面超车

理论铺垫完，进入真刀真枪的预训练。作者用完全相同的数据课程（262B tokens、相同 token 顺序）训练了四个模型，激活参数严格对齐在 3.8B：

| 模型 | 总参数 | 路由专家 | Engram 参数 | 说明 |
| --- | --- | --- | --- | --- |
| Dense-4B | 4.1B | - | - | 稠密基线 |
| MoE-27B | 26.7B | 72（top-6）+ 2 共享 | - | 稀疏基线 |
| Engram-27B | 26.7B | 55（top-6）+ 2 共享 | 5.7B | 从 MoE 腾出 17 个专家的参数给 Engram（$\rho=74.3\%$） |
| Engram-40B | 39.5B | 55（top-6）+ 2 共享 | 18.5B | 探索记忆扩展性 |

共同配置：30 层、hidden size 2560、MLA 注意力（32 头）、mHC 扩展率 4、Muon 优化器（Engram 嵌入单独用 Adam，学习率 ×5，无 weight decay）。Engram 模块插在第 2 和第 15 层，$N$-gram 取 $\{2,3\}$，8 个哈希头，$d_{\text{mem}}=1280$。

核心结果（262B tokens 训练后）摘录如下：

| Benchmark | Dense-4B | MoE-27B | Engram-27B | Engram-40B |
| --- | --- | --- | --- | --- |
| Pile (loss, ↓) | 2.091 | 1.960 | **1.950** | 1.942 |
| MMLU (5-shot) | 48.6 | 57.4 | **60.4** | 60.6 |
| MMLU-Pro (5-shot) | 21.1 | 28.3 | **30.1** | 31.3 |
| CMMLU (5-shot) | 47.9 | 57.9 | **61.9** | 63.4 |
| C-Eval (5-shot) | 46.9 | 58.0 | **62.7** | 63.3 |
| ARC-Challenge (25-shot) | 59.3 | 70.1 | **73.8** | 76.4 |
| BBH (3-shot) | 42.8 | 50.9 | **55.9** | 57.5 |
| DROP (1-shot, F1) | 41.6 | 55.7 | **59.0** | 60.7 |
| HumanEval (0-shot) | 26.8 | 37.8 | **40.8** | 38.4 |
| GSM8K (8-shot) | 35.5 | 58.4 | **60.6** | 62.6 |
| MATH (4-shot) | 15.2 | 28.3 | **30.7** | 30.6 |

几个值得强调的观察：

- 三种稀疏模型全面碾压同 FLOPs 的 Dense-4B，稀疏路线的扩展优势再次确认；
- **Engram-27B 在严格同参数、同 FLOPs 下一致性地超过 MoE-27B** 。知识类任务的提升在预期之内（MMLU +3.0、CMMLU +4.0、MMLU-Pro +1.8），但真正惊人的是 **通用推理涨得更多** （BBH +5.0、ARC-Challenge +3.7、DROP +3.3），代码数学同样可观（HumanEval +3.0、GSM8K +2.2、MATH +2.4）；
- Engram-40B 进一步降低了预训练 loss，且在训练末期与基线的差距仍在拉大——说明更大的记忆容量在当前 token 预算下还没吃饱。

![Figure: benchmark curves](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/benchmark_curve.png)

> 图解：四个模型在预训练最后 10k step 的各 benchmark 轨迹曲线。横轴为训练步数，纵轴为各任务指标。可以看到 Engram-27B（相对 MoE-27B）的提升不是靠个别噪声点，而是在训练过程中稳定拉开差距；Engram-40B 的曲线在训练末期仍保持上扬态势，印证了"记忆容量未饱和"的判断。

## 六、长上下文：把局部依赖外包后，Attention 解放了

理论上，把局部依赖建模外包给静态查找后，Attention 容量就能腾出来专注全局上下文。作者用 YaRN 做 32k 上下文扩展训练（5000 步、30B 长上下文数据），并在评测设计上做了一个非常严谨的受控实验：

- MoE-27B 用完整 50k step 的预训练 checkpoint；
- Engram-27B 除了 50k，还选了 41k 和 46k 两个中间 checkpoint。其中 **46k 的 Engram 预训练 loss 恰好等于 50k 的 MoE（均为 1.63）** ，构成严格的"同 loss"对照——任何长上下文差异都只能归因于架构本身，而非起点质量。

评测用 LongPPL（长书/论文/代码/长 CoT 四类困惑度）和 RULER（NIAH 大海捞针、变量追踪、词频抽取、QA 等 8 类）：

| 模型 (step, loss) | LongPPL-Book ↓ | LongPPL-L-CoT ↓ | NIAH-MQ ↑ | VT ↑ | FWE ↑ | QA ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| MoE-27B (50k, 1.63) | 4.38 | 14.16 | 84.2 | 77.0 | 73.0 | 34.5 |
| Engram-27B (41k, 1.66) | 4.37 | 14.26 | 89.5 | 83.2 | 99.6 | 44.0 |
| Engram-27B (46k, 1.63) | 4.19 | 13.59 | **97.0** | 87.2 | 98.6 | 37.5 |
| Engram-27B (50k, 1.62) | **4.14** | **13.41** | **97.0** | **89.0** | 99.3 | 40.5 |

结论分三层：

- **同 loss 对照（46k vs 基线）** ：Multi-Query NIAH 从 84.2 飙到 97.0，变量追踪从 77.0 到 87.2——架构优势被干净地剥离出来；
- **同 FLOPs（50k vs 基线）** ：差距进一步拉大，全面最优；
- **只用 82% 预训练算力（41k）的 Engram** ：LongPPL 已经打平满血 MoE，RULER 多项反超。

另外论文还指出一个方法论要点：Engram 41k→50k 的轨迹表明，长上下文能力与基座模型的通用建模能力内在耦合。因此严肃的架构对比必须对齐基座 loss，而不是只对齐训练步数——这一点本身对社区做长上下文研究就很有参考价值。

## 七、机制分析：Engram 到底改变了什么？

### 7.1 有效深度：浅层干深层的活

作者用两个可解释性工具验证了"Engram 等价于加深网络"的假设。

**LogitLens（预测收敛加速）。** 把每层隐状态直接投影到 LM Head，计算中间层输出分布与最终分布的 KL 散度。结果显示 Engram 各层 KL 散度系统性更小，尤其在早期 block 差距最大——曲线下降更陡，说明模型 **提前完成了特征组装** ，更早达到高置信度预测。

**CKA（表征对齐）。** 用 Centered Kernel Alignment 比较两个模型各层表征的相似结构：

$$\mathrm{CKA}(K, L) = \frac{\mathrm{HSIC}(K, L)}{\sqrt{\mathrm{HSIC}(K, K)\,\mathrm{HSIC}(L, L)}}$$

其中 $K = XX^\top$、$L = YY^\top$ 为 Gram 矩阵。进一步定义软对齐指标 $a_j$——Engram 第 $j$ 层对应的"等效 MoE 深度"（取 CKA 最相似的 top-$k$ 层的加权质心，$k=5$）：

$$a_j = \frac{\sum_{i \in \mathcal{I}_j} S_{i,j} \cdot i}{\sum_{i \in \mathcal{I}_j} S_{i,j}}, \quad \mathcal{I}_j = \operatorname{argtopk}_{i}(S_{i,j})$$

结果非常直观： **Engram-27B 第 5 层的表征，最接近 MoE 基线约第 12 层的表征** 。在很宽的层范围内 $a_j > j$，即 Engram 用更浅的层达到了更深的表征——这正是"省下来的层用于推理"的直接证据，也解释了为什么推理类任务涨得比知识类还猛。

### 7.2 结构消融：插在哪里、哪些组件关键？

在 12 层 3B MoE（0.56B 激活，100B tokens）的受控环境里做消融，基线 Val Loss = 1.808，参考配置（1.6B Engram，$\{2,3\}$-gram，插第 2、6 层）= 1.768。

- **插层位置** ：把 1.6B 预算集中成单个模块，从第 1 层扫到第 12 层。第 2 层最优（1.770），越往后越差。这揭示了一个权衡：尽早注入能在 backbone 浪费层深之前接管局部模式重建，但过早的隐状态还没聚合足够上下文、门控精度差。第 2 层恰好是"一轮 Attention 就够门控用、又足够早"的甜点。而把同样预算拆成两个小模块插在第 2、6 层（1.768）比单插第 2 层更好，兼顾早期干预和后期精调；
- **组件重要性** ：回归最大的是三个——多分支特异融合、上下文感知门控、tokenizer 压缩，去掉任何一个都明显掉点；深度卷积影响较小；把容量分给 4-gram 在 1.6B 预算下略亏（摊薄了更高频的 2/3-gram），但不排除更大记忆规模下高阶 $N$-gram 会翻身。

### 7.3 敏感性分析：功能分工的铁证

在推理时完全屏蔽 Engram 的嵌入输出（backbone 不动），观察各任务保留率：

- **事实知识类任务灾难性崩塌** ：只保留原性能的 29%~44%（TriviaQA 仅 29%）——证明 Engram 确实是参数化知识的主要仓库；
- **阅读理解类任务高度稳健** ：保留 81%~93%（C3 达 93%）——这类依赖上下文 grounding 的任务主要靠 backbone 的 Attention。

一个模块管"记住"，一个模块管"看懂"，功能二分非常清晰。

### 7.4 系统实测：100B 参数表卸载到内存，吞吐损失不到 3%

基于 nano-vLLM 实现的推理 harness，在 H800 上测试（512 条序列、长度 100~1024）：把一张 **100B 参数** 的 Engram 表整个放在主机 DRAM，插在第 2 个 block，推理时异步预取、PCIe 传输与第 1 个 block 计算重叠：

| 基座 | 配置 | 吞吐 (tok/s) |
| --- | --- | --- |
| Dense-4B | 基线 | 9,031.62 |
| Dense-4B | + 100B Engram (CPU Offload) | 8,858.28 |
| Dense-8B | 基线 | 6,315.52 |
| Dense-8B | + 100B Engram (CPU Offload) | 6,140.02 |

最大吞吐损失仅 **2.8%** （8B 基座上）。注意这还是一个保守下界：实验强制所有检索都走 PCIe，没有启用基于 Zipf 局部性的缓存。配上多级缓存层次后，开销只会更低。每步的有效通信量只随激活槽位数增长，与表的总大小无关——这意味着 Engram 可以绕开 GPU 显存墙，激进地扩张参数。

### 7.5 案例：门控可视化

可视化 Engram-27B 的门控标量 $\alpha_t$（每个 token 共 8 个门：2 层 × 4 分支），可以看到清晰的选择性：门在 **局部静态模式完成处** 稳定激活——英文里是 `Alexander the Great`、`the Milky Way`、`By the way`、`Princess of Wales` 这类多 token 实体和固定搭配；中文里则精准命中"四大发明""张仲景"等成语和历史实体。由于 Engram 作用于后缀 $N$-gram（$N=3$），token $x_t$ 处的高激活意味着"以它结尾的短语"被识别为静态模式并成功从记忆中检索。定性证据确认：Engram 确实接管了程式化语言依赖，把 backbone 从死记硬背中解放了出来。

## 八、与相关工作的关系

- **$N$-gram 建模与嵌入扩展** ：从 Shannon 的 $N$-gram 到 FastText，再到近年的 N-Grammer、SuperBPE、SCONE、OverEncoding、BLT 等"嵌入扩展"工作，本文与它们的关键区别有二：其一，以往工作多在非严格公平的协议下评估（如 SCONE 有额外训练 FLOPs、OverEncoding 在 MoE backbone 上非等参设置下也没有稳定收益），本文首次在严格 iso-parameter、iso-FLOPs 的稀疏分配框架内验证条件记忆的价值；其二，以往方案把嵌入固定放在第 0 层，串行化了访存与计算，Engram 则把记忆注入更深层实现通信-计算重叠，并利用 Zipf 分布吃满硬件存储层次；
- **高基数类别嵌入** （推荐系统）：共享"超大有偏离散 key 空间"的挑战，但 Engram 的 key 由有序文本 $N$-gram 构造，并通过上下文感知门控注入中间层；
- **MoE 与记忆网络** ：从 Shazeer 的开山之作到 DeepSeekMoE、PKM、PEER、UltraMem，以及 REALM、RETRO 等非参数记忆路线，Engram 定位在二者之间——参数化、可训练，但寻址确定、可存算分离；
- **知识存储机制** ：FFN 作为 Key-Value 记忆、知识神经元、ROME/MEMIT 模型编辑等一系列工作，为"用专用模块承接静态知识"提供了机理层面的支持。

## 九、总结与展望

这篇文章的贡献可以浓缩成四点：

- **提出条件记忆这条新稀疏轴** ：与 MoE 的条件计算互补，用 $O(1)$ 查找承接静态模式，用计算承接动态推理；
- **发现 U 形稀疏分配律** ：固定预算下把约 20%~25% 稀疏参数分给 Engram 最优，且最优点跨规模稳定（$\rho \approx 75\%\text{--}80\%$）；无限记忆 regime 下验证损失随槽位数呈幂律下降；
- **27B 规模实证** ：严格同参同算力下全面超越 MoE，且推理/代码/数学收益大于知识任务；CKA 与 LogitLens 揭示其本质是"等效加深网络"；长上下文上 Multi-Query NIAH 从 84.2 提到 97.0；
- **基础设施感知的设计哲学** ：确定性寻址使存算分离成为可能，100B 参数表放主机内存推理，吞吐损失小于 3%——显存墙被绕开了。

从更大的视角看，Engram 传达的信号是：稀疏化不应该只有"条件计算"一种语言。当记忆能像计算一样被稀疏地、确定性地寻址，模型容量的扩展就不再受限于 GPU 显存，而可以栖身于更便宜、更海量的存储介质上。条件记忆很可能会成为下一代稀疏模型的标配原语。代码已开源：https://github.com/deepseek-ai/Engram

> 本文参考自 [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/pdf/2601.07372)