# YOCO：KV 缓存只存一次的 Decoder-Decoder 架构

长上下文模型最大的部署痛点不是算力，而是显存：一个 65B 模型处理 512K Token 时，仅 KV Cache 就要吃掉约 86GB 显存，一张 H100-80GB 都装不下。微软研究院与清华提出的 **YOCO** （You Only Cache Once）用一种「Decoder-Decoder」架构正面解决这个瓶颈：让模型前半段（self-decoder）负责生产一份全局 KV Cache，后半段（cross-decoder）全部通过 cross-attention 复用这一份缓存，全局 KV 只存一次。结果是：KV Cache 显存最高降低约 80 倍，512K 上下文的 prefill 延迟从 180 秒压缩到 6 秒以内，同时在 1M 长度的大海捞针测试中取得近乎满分的检索准确率，且语言建模性能与同等规模的 Transformer 基本持平。

## 一、提出问题：KV Cache 正在压垮长上下文部署

在聊 YOCO 之前，先回顾一下为什么 Decoder-only Transformer 会成为大模型的标准答案。

回顾语言模型架构的三条路线：Encoder-only（如 BERT）双向编码，不适合自回归生成；Encoder-Decoder（如 T5）可以生成，但输出 Token 无法充分利用 Encoder 参数，多轮对话场景尤其吃亏；Decoder-only（如 GPT）靠 **KV Cache** 缓存历史 Token 的 Key/Value，避免了每生成一个 Token 就重算整段历史，推理速度大幅提升，由此一统江湖。

但天下没有免费的午餐，KV Cache 本身就是一颗定时炸弹：

- **显存爆炸** ：模型每一层都要为每个 Token 存一份 Key 和 Value，显存占用是 $N \times L$ 级别（$N$ 是序列长度，$L$ 是层数）。65B 模型（即使叠加 GQA 和 8-bit 量化）跑 512K Token 就要约 86GB 显存。
- **Prefill 巨慢** ：长序列输入的首 Token 延迟极高。7B 模型用 4 张 H100，prefill 450K Token 要 110 秒，1M 要 380 秒——用户发完一长段材料，要等 6 分钟才能看到第一个字。

这两个瓶颈叠加，让「原生长上下文」的模型几乎无法实际部署。问题的根源在于：Transformer 的每一层注意力都要独立缓存全序列的 KV。那么，能不能只缓存一次？这就是 YOCO 的出发点。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/highlight-cropped.png)

> 图解：论文的 Highlight 图。左上是架构示意：self-decoder 先生成全局 KV Cache，cross-decoder 复用它；下方三组曲线分别展示 YOCO 在训练 Token 数、模型尺寸、上下文长度三个维度上的可扩展性；右侧柱状图是 512K 上下文下的推理成本对比，YOCO 在 KV Cache 显存和 prefill 时间上都比 Transformer 低一到两个数量级。

## 二、核心思路：一半负责记，一半负责读

### 2.1 架构总览

YOCO 的直觉可以打个比方：传统的 Transformer 像是一个图书馆，每个阅览室（每一层）都要自己复印一套全部藏书（KV Cache）；而 YOCO 让模型前半段只设一个「中央档案室」（self-decoder 生成唯一一份全局 KV Cache），后半段所有阅览室都到档案室去查（cross-attention），不用各自复印。

形式上，YOCO 共堆叠 $L$ 个 Block，前 $\frac{L}{2}$ 层是 **Self-Decoder** ，后 $\frac{L}{2}$ 层是 **Cross-Decoder** 。

给定输入序列 $x = x_1 \cdots x_{|x|}$，词向量打包为 $X^0 \in \mathbb{R}^{|x| \times d_{model}}$。数据流分两步：

$$
X^l = \operatorname{Self\text{-}Decoder}(X^{l-1}), \quad l \in [1, \tfrac{L}{2}]
$$

$$
X^l = \operatorname{Cross\text{-}Decoder}(X^{l-1}, \hat{K}, \hat{V}), \quad l \in [\tfrac{L}{2}+1, L]
$$

其中 $X^{L/2}$ 负责产出全局 KV Cache $\hat{K}, \hat{V}$，供所有 cross-decoder 层复用。两部分都沿用 Transformer 的 Block 布局（Attention 与 FFN 交替，叠加 pre-RMSNorm、SwiGLU、GQA 等现代改进），也都使用因果掩码。所以从外部看，整个模型表现得就像一个普通的 Decoder-only Transformer，天然兼容自回归生成——这是它区别于 Encoder-Decoder 架构的关键。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/arch.png)

> 图解：YOCO 架构总览。输入 Token 先经过 $L/2$ 层 self-decoder（使用高效自注意力，如 sliding-window attention 或 gated retention），中间表示 $M = X^{L/2}$ 投影生成全局 $\hat{K}, \hat{V}$（图中高亮的 KV Cache 模块）；随后 $L/2$ 层 cross-decoder 各自用自己的 Query 对这份共享 KV 做 cross-attention。注意全局 KV 只有一份，这是「You Only Cache Once」名字的由来。

### 2.2 Self-Decoder：缓存复杂度为常数

Self-decoder 的每一层计算如下：

$$
\begin{aligned}
Y^l &= \operatorname{ESA}(\operatorname{LN}(X^l)) + X^l \\
X^{l+1} &= \operatorname{SwiGLU}(\operatorname{LN}(Y^l)) + Y^l
\end{aligned}
$$

关键在于 $\operatorname{ESA}(\cdot)$（Efficient Self-Attention，高效自注意力）模块必须满足一个硬约束： **推理显存复杂度为 $O(1)$** ，即缓存大小不随序列长度增长。例如 sliding-window attention 的缓存只取决于窗口大小 $C$，而与输入长度无关。这样 self-decoder 虽然也有缓存，但总量是 $O(CL)$ 级别的常数，长序列下可以忽略。

### 2.3 Cross-Decoder：一份 KV，全体复用

Self-decoder 的输出 $X^{L/2}$ 先投影生成全局 KV Cache：

$$
\hat{K} = \operatorname{LN}(X^{L/2}) W_K, \quad \hat{V} = \operatorname{LN}(X^{L/2}) W_V
$$

之后每个 cross-decoder 层只生成自己的 Query，对共享的 $\hat{K}, \hat{V}$ 做标准 multi-head cross-attention：

$$
\begin{aligned}
\hat{Q}^l &= \operatorname{LN}(X^l) W_Q^l \\
Y^l &= \operatorname{Attention}(\hat{Q}^l, \hat{K}, \hat{V}) + X^l \\
X^{l+1} &= \operatorname{SwiGLU}(\operatorname{LN}(Y^l)) + Y^l
\end{aligned}
$$

Cross-attention 同样使用因果掩码，且兼容 GQA，可以进一步压缩 KV 显存。最后 $X^L$ 接一个 softmax 分类器做 next-token 预测。

笔者认为这个设计最聪明的地方在于： **它没有发明新的全局记忆机制，而是把「全局注意力」与「高效局部编码」这两件已被分别验证的事，用 Encoder-Decoder 的经典结构拼接起来，同时对外保持 Decoder-only 的行为** 。既保留了全局注意力能力，又把 KV Cache 的份数从 $L$ 降到了 1。

### 2.4 推理优势：省显存，还能「提前下班」

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/inference.png)

> 图解：YOCO 的推理流程。Prefill 阶段并行编码输入 Token，只需跑完 self-decoder 即可生成全局 KV Cache，cross-decoder 的部分可以在生成第一个 Token 之前再算——这就是「early exit」；Generation 阶段逐个 Token 解码，cross-decoder 复用已缓存的 $\hat{K}, \hat{V}$。这个计算依赖的解耦不改变最终输出，纯粹是工程收益。

YOCO 的推理优势体现在两个复杂度对比上（$N$ 为序列长度，$L$ 为层数，$D$ 为隐藏维度）：

| | KV Cache 显存 | Prefill 时间（注意力部分） |
|---|---|---|
| Transformer | $O(LND)$ | $O(LN^2D)$ |
| YOCO | $O((N+L)D)$ | $O(LND)$ |

**显存方面** ，Transformer 要存 $N \times L$ 份 KV，而 YOCO 只需存一份全局 KV 加 self-decoder 的常数缓存，约省 $L$ 倍。省出来的显存可以换更大的 batch size，吞吐也随之提升。

**Prefill 方面** 更有意思：因为 cross-decoder 复用 self-decoder 的输出，prefill 阶段可以在进入 cross-decoder 之前「提前退出」。这带来双重加速——首先只需跑一半层数，至少省一半时间；其次 self-decoder 用的高效注意力本身就是线性复杂度。最终 prefill 从二次增长变成线性增长。

## 三、Self-Decoder 的设计选择：Gated Retention

Self-decoder 唯一的要求是 $O(1)$ 推理显存，满足这一点的高效注意力都可以插进来。论文默认使用作者提出的 **Gated Retention** （gRet，又称 gRetNet / RetNet-3），它在 RetNet 的 retention 机制上加入数据相关的门控，同时给出 sliding-window attention 作为备选。

### 3.1 三种等价表示

Gated Retention 的精髓是「一体三面」：同一个数学对象有 parallel、recurrent、chunkwise 三种等价写法，训练用并行形式吃满 GPU，推理用循环形式实现常数显存。

**并行表示** （训练用）。类似 attention，但用数据控制的衰减矩阵 $D$ 替代 softmax：

$$
\begin{aligned}
Q = (X W_Q) \odot \Theta, \quad K &= (X W_K) \odot \overline{\Theta}, \quad V = X W_V, \quad \Theta_n = e^{in\theta} \\
\gamma = \operatorname{sigmoid}(X W_\gamma)^{1/\tau}, \quad D_{nm} &= \textstyle\prod_{i=m+1}^{n} \gamma_i \ (n \ge m),\ \ 0 \ (n < m) \\
\operatorname{gRet}(X) &= (Q K^\intercal \odot D) V
\end{aligned}
$$

其中 $\gamma$ 是由输入决定的逐 head 衰减门（temperature $\tau$ 鼓励 $\gamma$ 趋近 1 以增强记忆），$D_{nm}$ 是从位置 $m$ 到 $n$ 的累积衰减。注意衰减是 head-wise 而非 element-wise，这样能充分利用 NVIDIA Tensor Core。

**循环表示** （推理用）。上式可以等价地写成 RNN 形式，第 $n$ 步只需维护一个固定大小的状态 $S_n$：

$$
S_n = \gamma_n S_{n-1} + K_n^\intercal V_n, \quad \operatorname{gRet}(X_n) = Q_n S_n
$$

推理时 self-decoder 只需保存 $S_n$ 这一个矩阵，与序列长度无关——这就是 $O(1)$ 显存的来源。

**分块循环表示** （训练与 prefill 用）。设 chunk 大小为 $B$，把计算拆成「块内并行 + 块间循环」：

$$
\operatorname{gRet}(X) = \underbrace{(Q_{[i]} K_{[i]}^\intercal \odot D_{[i]}) V_{[i]}}_{\text{块内}} + \underbrace{(Q_{[i]} R_{i-1}) \odot \beta_{[i]}}_{\text{跨块}}
$$

其中 $R_i$ 是第 $i$ 个 chunk 的循环状态，$\beta$ 汇总了 chunk 内的累积衰减。附录中给出了与循环表示的等价性证明：把输出 $O_n$ 按 $n = kB + r$ 拆成「当前 chunk 内的求和」与「历史 chunk 的求和」两部分，前者恰好是块内的 masked 矩阵乘，后者可递归地折叠进状态 $R_{i-1}$，二者相加即得分块公式。这种形式兼有并行的吞吐和循环的低显存，长序列 prefill 时尤其划算。

**多头版本** 。与 multi-head attention 类似，每个 head 独立做 gRet，拼接后经 GroupNorm 归一化，再用 swish 门控输出：

$$
\operatorname{MHGR}(X) = (\operatorname{swish}(X W_G) \odot \operatorname{GroupNorm}(\operatorname{Concat}(\mathrm{head}_1, \cdots, \mathrm{head}_n))) W_O
$$

### 3.2 备选方案：Sliding-Window Attention

Sliding-window attention（SWA）把每个 Query 的注意力范围限制在固定窗口 $C$ 内，通过窗口因果掩码 $B$（窗口内为 0，窗口外为 $-\infty$）实现，其余与标准多头注意力一致。推理时缓存从 $O(N)$ 降为 $O(C)$。论文中窗口取 1024，作为 gRet 的对照组。

## 四、实验验证：性能不掉队，成本数量级下降

解决了「怎么想」，接下来看「做得怎么样」。论文从四个角度验证：训练 Token 可扩展性、模型尺寸可扩展性、1M 长上下文能力、推理效率。

### 4.1 3B 规模语言建模：与 Transformer 平分秋色

作者按 StableLM-3B-4E1T 的训练配方训练 YOCO-3B（hidden size 3072、26 层、GQA、非嵌入参数 2.8B），训练序列长 4096、batch 4M Token，共训练 1.6T Token。LM Eval Harness 零样本结果如下：

| 模型 | ARC-C | ARC-E | BoolQ | Hellaswag | OBQA | PIQA | Winogrande | SciQ | Avg |
|---|---|---|---|---|---|---|---|---|---|
| OpenLLaMA-3B-v2 (1T) | 0.339 | 0.676 | **0.657** | **0.700** | 0.260 | 0.767 | 0.629 | **0.924** | 0.619 |
| StableLM-base-alpha-3B-v2 (1T) | 0.324 | 0.673 | 0.646 | 0.686 | 0.264 | 0.760 | 0.621 | 0.921 | 0.612 |
| **YOCO-3B (1T)** | **0.379** | **0.731** | 0.645 | 0.689 | **0.298** | 0.763 | **0.639** | **0.924** | **0.634** |
| StableLM-3B-4E1T (1.6T) | — | 0.688 | — | — | — | 0.762 | 0.627 | 0.913 | — |
| YOCO-3B (1.6T) | 0.396 | 0.733 | 0.644 | 0.698 | 0.300 | 0.764 | 0.631 | 0.921 | 0.636 |
| YOCO-3B-1M（扩到 1M 后） | 0.413 | 0.747 | 0.638 | 0.705 | 0.300 | 0.773 | 0.651 | 0.932 | 0.645 |

YOCO-3B 在 1T 和 1.6T 两个档位上都优于同数据量的 Transformer 基线，说明性能没有为推理效率付出代价；而且长上下文扩展后（YOCO-3B-1M）平均分反而继续提升到 0.645。

### 4.2 Scaling 曲线：从 160M 到 13B 全程贴身

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/scaling_curve.png)

> 图解：横轴是模型参数量（160M 到 13B，对数刻度），纵轴是验证集 LM loss。三条曲线分别是 Llama 改进版 Transformer、YOCO$_{gRet}$、YOCO$_{SWA}$。三条线全程几乎重合，且都很好地拟合了 scaling law 幂律曲线；YOCO$_{gRet}$ 甚至整体略低于 Transformer（loss 更低更好）。

值得注意的是 YOCO$_{gRet}$ 反而略优于 Transformer 和 YOCO$_{SWA}$。作者解释这来自 attention 与 retention 混合架构的归纳偏置互补——他们还发现按 1:3 交替排布 attention 和 retention 层也有类似增益，与 Jamba 等近期混合架构的发现一致。

### 4.3 1M 长上下文：大海捞针近乎满分

作者在 YOCO-3B 基础上用 64K → 256K → 1M 的渐进式长度调度继续训练（各阶段配合不同的 RoPE $\theta$ 与学习率），得到 YOCO-3B-1M。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/1m_retrieval.png)

> 图解：1M 长度的 Needle-In-A-Haystack 压力测试。横轴是文档深度（needle 插入位置），纵轴是上下文长度（最大 1M Token）。整幅图几乎全绿，表示 YOCO-3B-1M 在任意位置、任意长度下都能近乎完美地检索出埋藏的「针」（一个城市和对应的魔法数字），证明其全局注意力能力没有因只缓存一次而受损。

多针检索（128K 长度下评测）中，YOCO-3B-1M 以一半的参数量与 7B 的 LWM-1M-text 打得有来有回，并明显优于 MiniCPM-128K 和 ChatGLM3-128K：

| 模型 | 规模 | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|---|
| YaRN-Mistral-128K | 7B | 0.02 | 0.12 | 0.08 | 0.20 |
| LWM-1M-text | 7B | 1.00 | 0.90 | 0.76 | 0.62 |
| MiniCPM-128K | 2.4B | 1.00 | 1.00 | 0.54 | 0.56 |
| ChatGLM3-128K | 6B | 0.94 | 0.72 | 0.52 | 0.44 |
| **YOCO-3B-1M** | 3B | 0.98 | 0.98 | 0.84 | 0.56 |

困惑度实验进一步确认模型真的在利用远程信息：

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/book-1m-ppl.png)

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/code-1m-ppl.png)

> 图解：书籍（左）和仓库级代码（右）数据上的累积平均 NLL，横轴为上下文位置（最长 1M Token），纵轴为负对数似然。曲线随长度持续下降，且大致符合幂律——说明 YOCO 越读到后面预测越准，长距离依赖被真实利用了。

### 4.4 推理效率：全面数量级提升

对比对象是用 GQA + Flash-Decoding + kernel fusion 充分优化过的 Transformer，硬件为 H100-80GB，评测长度 32K 到 1M，公平性拉满。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/memory-length.png)

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/memory-arch.png)

> 图解：左图是不同长度下的推理总显存（横轴序列长度，纵轴显存），Transformer 随长度陡峭增长，YOCO 几乎平缓——1M 长度时 YOCO 仅需 12.4GB，Transformer 是其 9.4 倍；32K 时也省约 2 倍。右图是 1M 长度下的显存构成拆解：模型权重是常数，KV Cache 成为 Transformer 的绝对瓶颈，而 YOCO 同时压住了 activation 和 KV Cache 两项。这意味着消费级显卡跑 1M 上下文成为可能。

![Figure 10](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/cachemem-size.png)

> 图解：每个 Token 的 KV Cache 显存随模型尺寸的变化（横轴模型尺寸，纵轴每 Token 缓存字节数）。YOCO 只存一层全局 KV，约省 $L$ 倍，且模型越大（层数越多）省得越多——65B 模型上，1GB 显存 YOCO 能服务 128K Token，而带 GQA 的 Transformer 只能服务 1.6K Token。

![Figure 11](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/prefilling-length.png)

> 图解：Prefill 延迟对比（横轴输入长度，纵轴秒数，对数刻度）。Transformer 的曲线呈二次增长：512K 要 180 秒，1M 要 300 秒；YOCO 是线性增长，512K 不到 6 秒，1M 加速 71.8 倍，即使 32K 短输入也有 2.87 倍加速——这来自「一半层数 + 高效注意力」的双重红利。

![Figure 12](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/throughput-length.png)

> 图解：推理吞吐（横轴上下文长度，纵轴每秒处理 Token 数，含 prefill 与 generation）。512K 时 Transformer 只有 4.5 token/s，YOCO 达 43.1 token/s，提速 9.6 倍。原因有二：prefill 本身快了；显存省下来后可以用更大的 batch size。

## 五、附录精华：训练侧红利与更多架构对比

### 5.1 Chunk Parallelism：分布式长序列训练的通信红利

1M 长度的训练必须把序列切到多张 GPU 上，传统做法中每层 attention 都要 all-gather 一次 KV，通信成为吞吐瓶颈。而 YOCO 的结构恰好解耦了依赖：

- **Self-decoder** ：gRet 只需相邻设备传递循环状态 $S_n$（SWA 只需窗口内的通信），通信量很小；
- **Cross-decoder** ：全局 KV 只需 all-gather **一次** ，而不是每层一次。

![Figure 13](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/chunk_parallelism.png)

> 图解：两台 GPU 上的 chunk parallelism 示意。序列被切成块分到不同设备，self-decoder 各层只在相邻设备间传递状态 $S_n$；self-decoder 输出 $M = X^{L/2}$ 后做一次 all-gather 汇总 KV，此后所有 cross-decoder 层零通信。通信频率的下降直接减少了 1M 训练中的通信开销与显存碎片。

### 5.2 与更多架构的细粒度对比

附录还在 160M 规模上对比了 Mamba、RetNet、H3、gRetNet 等高效架构，并按 Zoology 的方法把困惑度拆成两类： **AR-Hit** （答案可从上下文中召回的二元组，考察联想回忆能力）与 **First-Occur** （无法从上下文召回的常规语言建模）：

| 模型 | 验证集 PPL | AR-Hit | First-Occur |
|---|---|---|---|
| Mamba | 3.645 | 1.555 | 4.126 |
| RetNet | 3.633 | 1.466 | 4.131 |
| Hybrid H3 | 3.591 | 1.251 | 4.130 |
| gRetNet | 3.600 | 1.354 | 4.116 |
| Transformer | 3.564 | 1.219 | 4.104 |
| YOCO$_{SWA}$ | 3.553 | 1.202 | 4.094 |
| **YOCO$_{gRet}$** | **3.530** | **1.199** | **4.067** |

两个 YOCO 变体在全部三项指标上都最好，尤其在考察「从上下文回忆」的 AR-Hit 上领先——说明 cross-decoder 的全局注意力确实保住了 Transformer 最核心的检索能力，而纯线性架构（Mamba、RetNet）在这项上明显吃亏。ZeroSCROLLS 长文本基准上，YOCO 与 Transformer 也一致地优于其他架构：

![Figure 14](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models/figure/scrolls_ppl.png)

> 图解：ZeroSCROLLS 四个长序列任务上，答案困惑度随输入长度（最长 16K）的变化。各架构中 YOCO 与 Transformer 始终处于第一梯队，且随长度增加持续下降；纯线性/稀疏架构则明显落后。

## 六、总结与展望

回顾全文，YOCO 的核心要点可以压缩为五条：

- **架构** ：前半 $L/2$ 层 self-decoder 产出唯一一份全局 KV Cache，后半 $L/2$ 层 cross-decoder 通过 cross-attention 复用，全局 KV 只存一次；
- **模块** ：self-decoder 使用 $O(1)$ 显存的高效注意力，默认 gated retention（parallel/recurrent/chunkwise 三位一体），备选 sliding-window attention；
- **性能** ：3B × 1.6T Token 打平并略超同级 Transformer，160M–13B scaling 曲线贴身，1M 上下文大海捞针近乎满分；
- **效率** ：65B 模型 KV Cache 省约 80 倍，1M prefill 加速 71.8 倍（512K 从 180 秒到 6 秒内），吞吐提升 9.6 倍；
- **系统** ：KV 只需 all-gather 一次，为分布式长序列训练带来通信红利。

YOCO 的局限在于 cross-decoder 的建模能力仍受单份 KV 表达容量的约束（8 针检索已出现下滑），但它的真正价值或许是把「KV Cache」从每层的内部状态提升为架构级的一等公民——显式的高亮缓存模块为缓存压缩、检索索引、预缓存 RAG 等原生记忆机制打开了设计空间，与 BitNet（压权重）这类正交方向组合后，长上下文推理的成本还有数量级的想象空间。

> 本文参考自 [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/pdf/2405.05254)