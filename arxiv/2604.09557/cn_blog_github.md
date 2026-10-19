# SPEED-Bench：投机解码评测新标杆

投机解码（Speculative Decoding, SD）已经是加速大模型推理的主流技术，但它的效果高度依赖输入数据：同样的算法，换个数据集、换个 batch size，加速比可能天差地别——而现有评测恰恰在这两点上都很随意。NVIDIA 这篇被 ICML 2026 接收的工作提出了 SPEED-Bench，一个覆盖 11 个语义类别、支持 1k–32k 输入长度和大并发场景的 SD 统一评测基准，并直接对接 vLLM、TensorRT-LLM、SGLang 等生产级推理引擎。最硬的两个数字：用随机 token 做压测会**平均高估吞吐 23%**；在 $BS=32$ 的真实负载下，EAGLE3 在 Llama 3.3 70B 上能拿到 **1.90×** 的端到端加速，而 N-Gram 方案反而会**拖慢**推理（0.88×）。

## 一、提出问题：投机解码的评测为什么"失真"

先说清楚背景。LLM 的自回归解码在低并发下是典型的 memory-bound 过程：把时间花在把模型参数从 HBM 搬到片上缓存，远大于实际计算。SD 的思路是用一个小 draft model 先猜 $\gamma$ 个 token，再让大模型一次前向并行验证——因为瓶颈在显存搬运而非计算，验证多个 token 的边际成本极低，猜得准就等于一次前向产出多个 token。配合拒绝采样，SD 还是无损加速，输出分布与目标模型严格一致。DeepSeek-R1、Qwen3-Next 等前沿模型甚至把 MTP（Multi-Token Prediction）头原生做进了架构里。

问题在于，SD 的收益是**数据依赖**的，而现有评测方法存在四个系统性缺口：

- **数据域太窄**：acceptance rate 对文本域和熵极度敏感，但新方法常在 MT-Bench 这类小数据上验证——MT-Bench 每类只有 10 条样本，类内多样性极低，跨方法根本没法公平比较。
- **引擎不真实**：很多论文用 HuggingFace 这类高层库测速，完全反映不了 vLLM、TensorRT-LLM、SGLang 里 CUDA Graph、continuous batching 等优化下的真实表现。
- **只测 $BS=1$**：论文爱报单并发加速比，但真实多用户服务追求吞吐，用的是大 batch。batch 一大，系统从 memory-bound 转向 compute-bound，SD 的加速比会缩水甚至反转。
- **ISL 太短**：行业在向长上下文（如编程助手）迁移，而现有基准的输入长度大多不到 100 个 token，长 ISL 场景几乎没人研究。

笔者认为，这四个缺口里最容易被忽视的是第三个。很多团队拿着 $BS=1$ 下 2× 以上的加速比去立项，上线后才发现大 batch 下收益荡然无存——这正是缺乏真实服务场景评测的代价。

还有一个值得一提的背景观察：MoE 模型特别适合 SD。MoE 参数量巨大，但每个 token 只激活一小撮专家，系统很难真正进入 compute-bound；验证 draft token 只需多加载几个专家，想让所有专家全部饱和需要海量的 token。所以即使在大 batch 下，SD 对 MoE 依然有效。

**关键指标**先约定好，后文反复用到：

- **AR（Acceptance Rate）**：条件接受率，即在前缀都被接受的条件下，第 $i$ 个 draft token 被接受的概率。
- **AL（Acceptance Length）**：每次验证步骤期望产出的 token 数（含验证"赠送"的那个 token）。对 draft 长度 $\gamma$：

$$
\text{AL} = 1 + \sum_{i=1}^{\gamma} \prod_{j=1}^{i} \text{AR}_j
$$

- **Throughput（Output TPS）**：所有并发请求每秒产出的总 token 数；**User TPS**：单请求每秒 token 数，是端到端延迟的代理指标。

## 二、SPEED-Bench 总览：一个数据集，两种用途，一套框架

SPEED-Bench 由三部分组成，对应三类评测需求：

![SPEED-Bench 生态总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/main_figure.png)

> 图解：SPEED-Bench 的三驾马车。**左**：Qualitative split 的构建——对 prompt embedding 运行自定义选择算法，最大化类别间的语义多样性。**中**：Throughput split 的构建——数据聚合成固定 ISL 桶（1k–32k），按三种域难度分类，每个桶支持大到 512 的 batch。**右**：统一测量框架，输出标准 SD 指标与加速比。

设计上有一个很务实的取舍：Qualitative Split 追求"小而精"，用选择算法从 18 个公开数据源里挑出语义最多样的 880 条样本，让 AR/AL 测量又快又准；Throughput Split 追求"量足且规整"，用固定 ISL 桶和充足样本量支撑稳定的吞吐-延迟 Pareto 曲线。两者分工明确，互不替代。

## 三、Qualitative Split：用算法"选"出多样性

### 数据构成

作者从 18 个公开数据集聚合出 11 个类别（Coding、Math、Humanities、STEM、Writing、Summarization、Roleplay、RAG、Multilingual、Reasoning、QA），每类选 80 条，共 880 条。选源原则是**语义异质性**——他们发现 SpecBench 里完全基于 MT-Bench 的 Coding 和 Humanities 类别过于简单，不足以代表现代 LLM 任务的复杂度。

元数据也比 SpecBench 丰富得多：每条样本带子类别、多轮对话标记（约 20% 样本为 2–5 轮对话，SpecBench 最多 2 轮），Coding/Math/Humanities/STEM 还带难度标签（约 80% 为难题）。作者还用 GPT-4 验证了样本平均能生成约 650 个 token，保证投机指标有统计意义。

### 选择算法：贪心 + 局部交换

核心想法很直觉：把评测集想象成在语义空间里"撒点"，点越分散，覆盖越全。形式化地说，用 OpenAI 的 text-embedding-3-large 把样本 $t_i$ 映射为归一化向量 $x_i$，余弦相似度即 $x_i^\top x_j$。要从 $N$ 个候选里选 $k$ 个，使子集 $S$ 的总两两相似度最小：

$$
\mathcal{L}(S) = \sum_{i \in S} \sum_{j \in S, j \neq i} x_i^\top x_j
$$

精确求解是 NP-hard，作者采用**贪心选择 + 局部交换精炼（Local Swap Refinement）**：先随机初始化，每轮加入与当前集合总相似度最小的候选；然后反复尝试把集合内的 $i_{out}$ 与集合外的 $i_{in}$ 交换，若交换使 $\mathcal{L}(S)$ 严格下降则执行，直到收敛。交换的增益可以增量计算：

$$
\Delta = \sum_{j \in S \setminus \{i_{o}\}} (x_{i_{i}}^\top x_j - x_{i_{o}}^\top x_j)
$$

作者还对比了二次规划（QP）松弛方案——求解 $w \in [0,1]^N$ 使 $w^\top G w$（$G$ 为 Gram 矩阵）最小——多样性得分相当，但贪心方案更快、更能扩展到大候选池。

效果如何？相比 SpecBench，该算法把平均语义相似度降低了 **40%**（Multilingual 类别高达 83%）。

![语义相似度对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/similarity_spider.png)

> 图解：各类别样本间平均语义相似度的雷达图（**越低越好**）。SPEED-Bench 的选择算法（含/不含局部交换精炼）在所有类别上都优于随机选择和 SpecBench。值得注意的是，在相同数据源上随机选择就已经优于 SpecBench，说明 SPEED-Bench 的数据源本身质量更高；而选择算法又在此基础上进一步压低相似度。

这个"随机选择也优于 SpecBench"的观察很耐人寻味：SpecBench 的问题不仅是没做多样性优化，更是源头数据就同质化——它的 Multilingual 子集全是"Translate German to English:"这一种句式，占了总数据量的约 15%。

附录里的热力图给出了直观证据：

![Multilingual 类别相似度热力图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/similarity_multilingual.png)

> 图解：Translation/Multilingual 类别的两两余弦相似度矩阵，深绿色表示高相似（冗余），浅黄色表示低相似（多样）。左列 SpecBench 呈现大块深绿色——同一数学题的微改版、雷同的翻译句式聚成一团；右列 SPEED-Bench 则是弥散的低相似分布，证实贪心+交换算法有效消除了冗余。

![Math 类别相似度热力图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/similarity_math.png)

> 图解：Math 类别的同款对比，结论一致：SpecBench 有明显的高相似度区块，SPEED-Bench 则均匀分散。

### 选择的稳定性：不只是多样，还要可复现

一个好 benchmark 不仅要多样，还得保证"换一批样本结论不变"。作者各生成 10 个随机采样子集和 10 个算法优化子集，用 Llama 3.3 70B + EAGLE3 测量，结果很有说服力：

| 类别 | 均匀采样 AL 相对波动 | 优化选择 AL 相对波动 |
|---|---|---|
| Roleplay | 22.7% | **0.6%** |
| Multilingual | 14.6% | **5.1%** |
| RAG | 12.1% | **0.9%** |
| Humanities | 9.2% | **3.0%** |
| Writing | 7.2% | **0.9%** |
| STEM | 6.5% | **2.8%** |
| Math | 5.6% | **1.6%** |
| Coding | 4.0% | **0.7%** |

> 表中数值为 $DL=7$ 时不同子集间 AL 的相对极差（最大值减最小值，除以均值），越低越稳定。

类别排序的一致性同样大幅提升：$DL=7$ 时，优化选择的归一化 Kendall-$\tau$ 分歧从 0.056 降到 **0.004**，约 14 倍改善；9 个优化子集里 8 个给出完全相同的类别排序，而 10 个随机子集出现了 8 种不同排序。换句话说，随机采样的 benchmark 连"哪个域对 SD 更容易"这种基本结论都可能是噪声。

## 四、Throughput Split：面向真实服务的吞吐评测

解决了"测得准"之后，下一个问题是"测得真"——系统级加速比必须在真实负载形态下测量。

### 合成数据的两个陷阱

推理压测圈有个通行做法：用随机 token 拼 prompt 模拟负载（vLLM 官方压测脚本就是这么干的，InferenceMax 等公开基准也在用）。对稠密模型的普通解码这没问题，但对 SD 是**根本性错误**的，因为 SD 的表现取决于输入分布的可预测性。随机 token 会触发两种失真模式：

- **Trivial Response（平庸回应）**：模型识别出输入是噪声，回一段"您粘贴的内容似乎不成问题，请问您想让我做什么？"式的模板回答。这种回答极易预测，**虚增 AR**——实测平均 AL 高达 3.44。
- **Topic Latching（话题锁定）**：噪声里偶然出现的关键词被模型当成强信号，幻觉出一篇连贯但文不对题的回答（比如突然输出一份 Unity 2D 平台游戏开发路线图），**压低 AR**——实测平均 AL 只有 1.877。

更隐蔽的是第三个问题：**MoE 专家路由失真**。MoE 的路由器是在自然文本上训练的，面对分布外的随机 token 会"塌缩"到少数几个专家上，破坏推理引擎的负载均衡假设——即使不开 SD，步延迟测量都会失真。

![专家激活分布对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/experts_activation_mid_layer_Random_vs_SPEED.png)

> 图解：GPT-OSS 120B 第 17 层在 prefill 阶段（8k ISL，$BS=32$）top-4 专家的激活频率分布，横轴为专家编号（1–128）。上图使用 SPEED-Bench Throughput Split 数据，激活相对均匀；下图使用随机 token，路由器明显偏向一小撮专家。

![各层激活专家数](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/experts_per_layer_Random_vs_SPEED.png)

> 图解：模型各层被激活的独立专家总数，对比随机 token 与 Throughput Split（8k）。某些层中随机 token 未能激活 **20–30%** 的专家——要知道这是在 $32 \times 8000$ 个 token 的大体量下发生的，说明合成噪声根本无法触发真实语义负载下的路由行为。

### 数据构成

Throughput Split 构造了 5 个固定 ISL 桶（1k、2k、8k、16k、32k，用 `o200k_base` tokenizer 计量），来自 8 个公开数据源；通过截断或追加中性后缀（"please answer now"）对齐长度，保证 prefill 负载确定性的同时不破坏 prompt 语义。样本按域熵分为三档——**低熵**（排序、代码）、**混合熵**（STEM）、**高熵**（创意写作）。每个桶内每类 512 条（每桶共 1536 条），足以画出稳定的吞吐-延迟 Pareto 曲线。作者验证过这些样本平均能生成约 2.4k token（GPT-4，16k ISL）。

### 一个实用彩蛋：跨域加速比估算

这个 split 还有个巧妙用法。注意**每步延迟**主要由系统条件（引擎、batch、ISL、硬件）决定，而 **AL** 主要由数据域决定——两者近似解耦，于是加速比可以解析估算：

$$
S = \frac{AL / t_{sd}}{1 / t_{ar}} = \frac{t_{ar} \cdot AL}{t_{sd}}
$$

实操协议是：在 Throughput Split 对应 ISL 桶上测一次 $t_{ar}$ 和 $t_{sd}$，再用目标域的少量代表性 prompt 测 AL，代入公式即可估算该域的加速比——无需为每个细分域做大规模吞吐测试。作者用 Llama 3.3 70B（$BS=16$，$DL=3$）做了验证：

| Drafter | 目标类别 | ISL | AL | 实测加速比 | 估算加速比 |
|---|---|---|---|---|---|
| EAGLE3 | 高熵 | 1k | 2.12 | 1.63× | 1.61× |
| EAGLE3 | 高熵 | 8k | 1.17 | 0.85× | 0.86× |
| EAGLE3 | 低熵 | 1k | 2.93 | 2.23× | 2.22× |
| EAGLE3 | 低熵 | 8k | 1.19 | 0.87× | 0.88× |
| Vanilla | 高熵 | 1k | 2.47 | 1.37× | 1.38× |
| Vanilla | 低熵 | 2k | 3.16 | 1.67× | 1.67× |
| Vanilla | 低熵 | 8k | 3.21 | 1.36× | 1.35× |

> 步延迟在混合熵类别（STEM/QA）上测量，再与目标域的 AL 组合。估算值与端到端实测值几乎完全吻合，最大误差在 0.01× 量级。

笔者认为这个 proxy 是整篇论文里最"工程友好"的贡献：它把昂贵的全量吞吐测试降维成一次系统测量加一次轻量 AL 测量。当然前提还是那句老话——$t_{ar}$、$t_{sd}$ 必须用真实数据测，随机 token 测出来的步延迟本身就不可信。

## 五、统一测量框架：消除引擎间的"苹果与橘子"

跨引擎评测 SD 有个隐形坑：不同引擎对原始文本的处理不一致——有的自动加 BOS token，有的套不同 chat template——导致 draft 和 target 实际处理的序列都不一样，比较失去意义。

SPEED-Bench 的框架作为一个"薄客户端"，把**外部因素**（tokenization、prompt 模板、特殊 token）全部在框架侧归一化，保证所有后端处理完全相同的 token 序列；而**内部因素**（数值实现、CUDA kernel、调度器、continuous batching）则保留原样——因为这些正是部署环境的本质特征，评测的目的就是测量 SD 在真实部署中的端到端表现。

实现上，框架基于 Python asyncio 异步事件循环并发派发请求，通过分析流式响应 chunk 计算 AR（一个 chunk 含多个新 token 即一次成功投机），并记录 TTFT、步延迟、总延迟，聚合出 User TPS 和 Output TPS。框架原生支持 SGLang、vLLM、TensorRT-LLM，也兼容 SpecBench 的研究模型（附录给出了 Medusa 的接入示例）。作者坦承一个**当前局限**：$BS>256$ 时 Python GIL 会引入客户端开销，正在改进。

## 六、实验与发现：被传统评测掩盖的真相

实验覆盖四类 SD 方法（N-Gram、Vanilla SD 即外部独立草稿模型、EAGLE3、原生 MTP 头）和五个大模型（Llama 3.3 70B、GPT-OSS 120B、Qwen3 235B、Qwen3-Next、DeepSeek R1），只用 draft chain 而非 tree 验证（后者在生产引擎的 $BS>1$ 场景尚非主流）。硬件以单张 B200 为主，DeepSeek/Qwen 推理和 EAGLE3 训练用 8 张。

### 主实验：准确率与加速比

在 Qualitative Split 上以 $BS=32$、$DL=3$ 测量（Temperature=0）：

| 域 | Llama 3.3 70B N-Gram / Vanilla / EAGLE3 | GPT-OSS 120B N-Gram / EAGLE3 | DeepSeek R1 MTP | Qwen3 235B Vanilla / EAGLE3* | Qwen3-Next EAGLE3 / MTP* |
|---|---|---|---|---|---|
| Coding | 1.54 / 2.72 / 3.00 | 1.31 / 2.46 | 2.76 | 2.62 / 2.26 | 3.17 / 3.34 |
| Math | 1.43 / 2.43 / 2.45 | 1.30 / 2.46 | 2.77 | 2.69 / 2.37 | 2.90 / 3.13 |
| Multilingual | 1.91 / 2.59 / 1.71 | 1.58 / 2.32 | 2.68 | 2.87 / 2.33 | 1.90 / 3.19 |
| Roleplay | 1.15 / 2.14 / 2.04 | 1.25 / 1.87 | 2.14 | 1.92 / 1.87 | 1.80 / 2.09 |
| Writing | 1.33 / 2.45 / 2.63 | 1.20 / 1.98 | 2.33 | 2.19 / 2.02 | 2.09 / 2.46 |
| **平均 AL** | 1.41 / **2.44** / **2.44** | 1.31 / **2.25** | 2.55 | **2.43** / 2.22 | 2.36 / **2.81** |
| **平均加速比** | 0.88× / 1.60× / **1.90×** | 0.29× / **1.34×** | 1.45× | 1.17× / **1.33×** | 1.06× / **1.20×** |

> 表中为各类别平均 AL（完整 11 类见原文表 1），* 为 SGLang 上测量。Temperature=1 时所有方法 AL 和加速比均有下降（如 EAGLE3 on Llama 从 1.90× 降至 1.75×）。

三个关键观察：

1. **域熵与 AL 正相关**：低熵的 Coding、Math 显著高于高熵的 Roleplay——可预测性就是 SD 的燃料。
2. **N-Gram 在真实并发下是负优化**：$BS=32$ 时 N-Gram 在两个模型上都造成净减速（0.88×、0.29×），AR 撑不起验证成本。GPT-OSS 上 0.29× 尤其惨烈。
3. **AL 相同不等于加速比相同**：Llama 上 Vanilla 与 EAGLE3 的 AL 同为 2.44，但 EAGLE3 加速比 1.90× 远高于 Vanilla 的 1.60×——外部独立草稿模型的额外开销吃掉了收益。

![AL 随 DL 的变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/AL_BY_DL.png)

> 图解：Qualitative Split 上平均 AL 随 draft 长度（DL）变化的曲线。两个趋势：**原生 MTP 更稳**——Qwen3-Next 预训练的 MTP 头在各 DL 上都优于 EAGLE3 等后训练方案，说明预训练带来的精度红利实打实；**Vanilla SD 更抗长 DL**——尽管草稿开销更大，外部草稿在长投机视野下精度衰减更慢，曲线持续上行。

### 词表剪枝的隐藏代价

EAGLE3 会把草稿词表剪到 32k token 以缓解输出投影层的计算瓶颈。这个优化在标准输入上无伤大雅，但对"长尾"输入是隐性伤害。理论分析显示：剪到 32k 时，整体输出 token 覆盖率还有 94.7%，但 **Multilingual 类别只剩 76.9%**（约 22% 的目标 token 根本不在剪枝词表里，AR 上限被直接砍掉）。

![全词表 vs 剪枝词表](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/EAGLE3-Full-vs-Pruned-Vocab-Comparison.png)

> 图解：GPT-OSS 120B + EAGLE3（$DL=3$）在全词表与剪枝词表下、若干类别的平均 AL 对比。Math 和 Coding 几乎无损，Writing 持平，但 **Multilingual、RAG、Summarization 显著退化**。这类副作用只有在语义覆盖足够广的 benchmark 上才会暴露——如果只在 MT-Bench 上测，你永远不会发现。

### 与 SpecBench 的正面对比

SpecBench 的 Coding、Reasoning 类别各只有 10 条样本，统计噪声大到让轻量的 EAGLE3 看起来和更重的 Vanilla SD 打平。SPEED-Bench 纠正了这个错觉。

![SpecBench vs SPEED-Bench](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/Llama-3.3-70B-TRTLLM-SpecBench-Comparison.png)

> 图解：Llama 3.3 70B（$DL=7$）在 SpecBench 与 SPEED-Bench 上的平均 AL 对比。SpecBench 上 EAGLE3 与 Vanilla SD 差距很小；SPEED-Bench 更大、语义更多样的样本则清晰揭示了外部草稿在长 DL 下的真实优势。Multilingual 类别差异最戏剧性——SpecBench 的多语言子集全是德译英，只显示温和差距；SPEED-Bench 覆盖 23 种语言和多种任务，外部草稿的优势被充分放大。而 Multilingual 和 Coding 恰恰是选择算法降相似度最多的两个类别，印证了"高多样性是暴露方法差异的前提"。

### 延迟与吞吐：真实负载下的系统行为

**随机数据 vs SPEED-Bench。**前文说合成数据失真，这里是实锤：

![随机 token vs SPEED-Bench 吞吐](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/GPT-OSS-120B-TRTLLM-8k-DL_3-speed-vs-random.png)

> 图解：吞吐（Output TPS）随单用户 TPS 变化的 Pareto 曲线，对比随机 token 输入与 Throughput Split（8k）。目标 GPT-OSS 120B + EAGLE3，TensorRT-LLM，$DL=3$，数据点对应 $BS$ 1–128。开 SD 时，合成压测平均**高估吞吐 23%**；有趣的是基线自回归设置也有差距——这正是随机 token 引发 MoE 专家失衡、步延迟失真所致。

**最优 DL 随 batch 变化。**这是全文对部署最有指导意义的一张图：

![不同 DL 的吞吐对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/GPT-OSS-120B-VLLM-2k-DL_1to3.png)

> 图解：吞吐 vs 单用户 TPS，对比 $DL=1$ 与 $DL=3$（Throughput Split 2k 桶，GPT-OSS 120B + EAGLE3，vLLM，$BS$ 2–256）。低 batch 时系统 memory-bound，长草稿占优；batch 增大趋向 compute-bound 后，多验证 token 的成本反超收益，$DL=1$ 反超。两条曲线存在明确的**交叉点**——最优 DL 是并发场景的函数，不存在全局最优。SPEED-Bench 的价值就在于让开发者找到自己的交叉点。

**引擎差异。**TensorRT-LLM 的峰值吞吐更高，秘诀是 one-model 运行时：投机头直接挂在目标模型上，整个"起草-验证"循环能被一张 CUDA Graph 完整捕获；vLLM 采用 two-model 设计，草稿模型作为独立引擎运行，引擎间通信带来额外 host 开销（async/overlap 调度能掩盖一部分），但 piecewise graph 构造对动态草稿策略更灵活。

![引擎对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/GPT-OSS-120B-TRTLLM-2k-DL_2-speed-different-engines.png)

> 图解：吞吐 vs 单用户 TPS，对比不同推理引擎（GPT-OSS 120B + EAGLE3，Throughput Split 2k 桶，$DL=2$，$BS$ 2–256）。TensorRT-LLM 曲线整体位于 vLLM 上方。

### 训练 ISL 与 RoPE：长上下文草稿的"保质期"

作者在 Throughput Split 上发现 HuggingFace 上两个公开的 GPT-OSS 120B EAGLE3 模型在高 ISL 桶上精度大幅退化。为量化原因，他们自训了训练长度分别为 1k、2k、4k 的 EAGLE3 模型：

![训练 ISL 与 YaRN 的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/ISL_vs_Context_Length_noscaling_vs_yarn.png)

> 图解：平均 AL 随推理 ISL 的变化，对比不同训练 ISL（图例 1k/2k/4k 为训练时最大序列长度），虚线为推理时施加 YaRN scaling 的结果。规律很清晰：**推理 ISL 一旦超过训练 ISL，精度快速崩落**；而简单地在推理时加 YaRN scaling，即使训练只用了 2k 序列，也能找回大量草稿精度。

对两个公开 checkpoint 的归因也很有意思：lmsys 版本根本没配 RoPE scaling，推测训练上下文上限 8k，超出即衰；nvidia 版本虽然配了 scaling，但 `original_max_position_embeddings` 设为 8192——疑似训练时沿用了 Llama3 的默认值，没改成真实训练长度，导致 8k ISL 就开始退化。作者的建议：训练 EAGLE3 时把 `max_position_embeddings` 设为真实训练上下文长度，推理配置里施加 RoPE scaling，最好再做一段长上下文微调。

![不同 EAGLE3 模型的 AL 稳定性](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2604.09557/figures/Method_Comparison_Context_Length.png)

> 图解：Throughput Split（1k–32k）上三个 GPT-OSS 120B EAGLE3 草稿的平均 AL。配置得当的 RoPE scaling（自训 4k + YaRN 参考线）能在所有上下文长度上保持稳定，而两个公开模型在高 ISL 明显下滑。

附录还补充了一个细节：GPT-OSS + EAGLE3 组合在 Throughput Split 上出现反常——低熵类别的 AL 随 ISL 增长跌破了混合熵曲线。作者归因于该草稿模型训练语料（UltraChat + Magpie）中代码占比极低（Magpie 整体不到 8%），叠加可能的 RoPE 配置问题。这再次说明：**AL 的异常往往指向草稿模型的训练分布，而不是 SD 方法本身**——没有细粒度分域数据，这种诊断无从谈起。

### 树状验证的兼容性

虽然主实验聚焦 draft chain，框架同样支持 tree 验证。EAGLE3 + Qwen3 235B（SGLang）上的结果显示，增大分支因子（top-$k$）或 DL 都能提升 AL——top-$k$=8、$DL$=7 时平均 AL 达 3.65（top-$k$=2、$DL$=3 时为 2.55），符合"扩大候选验证空间即提升接受"的预期。

## 七、总结

- SPEED-Bench 用"贪心选择 + 局部交换"算法从 18 个数据源精选 880 条高多样性样本，语义相似度比 SpecBench 低 40%，且跨子集的类别排序一致性提升约 14 倍。
- Throughput Split 提供 1k–32k 固定 ISL、三档熵、大 batch 的真实负载，首次支持 SD 的吞吐-延迟 Pareto 评测。
- 随机 token 压测对 SD 根本不可信：虚增吞吐 23%，还会让 MoE 路由塌缩、连基线步延迟都测不准。
- 真实并发下的反直觉结论：N-Gram 是负优化（0.88×/0.29×）；AL 相同不代表加速比相同（草稿开销很关键）；最优 DL 随 batch size 存在交叉点。
- 广覆盖评测暴露了被窄基准掩盖的副作用：EAGLE3 词表剪枝重创多语言（22% token 不可草稿化），公开 EAGLE3 模型普遍存在训练 ISL/RoPE 配置导致的长上下文精度崩塌。

展望：框架目前在 $BS>256$ 时受 Python GIL 限制，作者正在扩展更强的并行能力。更大的意义在于，SPEED-Bench 把 SD 评测从"论文里跑个 MT-Bench"推向"在真实引擎、真实负载、真实数据上说话"——当评测标准对齐了部署现实，社区迭代的 SD 方法才更可能活着走出实验室。

> 本文参考自 [SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](https://arxiv.org/abs/2604.09557)，数据已开源：[HuggingFace: nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)