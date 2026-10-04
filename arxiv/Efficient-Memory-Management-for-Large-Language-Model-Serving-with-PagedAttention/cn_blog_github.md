# PagedAttention：用操作系统的分页思想驯服 LLM 推理的显存怪兽

大模型推理贵在哪里？不是算力不够，而是显存里那块不断膨胀的 KV Cache 把 Batch Size 卡死了——传统系统的 KV Cache 显存实际利用率只有 20.4% - 38.2%，其余全被碎片和预留空间浪费。这篇 SOSP'23 论文把操作系统里经典的虚拟内存与分页（Paging）思想搬进了 Attention 计算：将 KV Cache 切成固定大小的 Block 按需分配、离散存放，并在其上构建了 serving 系统 vLLM。结果是在延迟持平的前提下，吞吐量相对 FasterTransformer 和 Orca 提升 **2-4 倍** ，且序列越长、模型越大、解码算法越复杂，优势越明显。

## 一、问题的提出：LLM Serving 是 Memory-Bound 的

要理解这篇工作，先得搞清楚 LLM 推理的计算模式。

### 自回归生成与 KV Cache

语言建模本质上是对 token 序列 $(x_1, \ldots, x_n)$ 的联合概率建模，通过自回归分解逐个生成：

$$
P(x) = P(x_1) \cdot P(x_2 \mid x_1) \cdots P(x_n \mid x_1, \ldots, x_{n-1})
$$

Transformer 的 Self-Attention 层中，每个位置的 hidden state 先经过线性变换得到 query、key、value：

$$
q_i = W_q x_i, \quad k_i = W_k x_i, \quad v_i = W_v x_i
$$

再计算注意力分数并加权求和得到输出：

$$
a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^{i}\exp(q_i^\top k_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{i} a_{ij} v_j
$$

关键在于：生成第 $i$ 个 token 需要用到前面所有 token 的 key 和 value。为了避免重复计算，这些历史 key/value 向量会被缓存起来，这就是 **KV Cache** 。注意同一个 token 出现在序列不同位置时 KV Cache 是不同的——它依赖于全部前缀。

### Prompt 阶段与生成阶段的两极分化

一次请求的计算被拆成两个阶段：

- **Prompt 阶段（Prefill）** ：整个 prompt 一次性送入，用矩阵乘矩阵并行计算，GPU 利用率很高。
- **自回归生成阶段（Decode）** ：每次迭代只算一个新 token，矩阵乘向量，算力大量闲置，且迭代之间无法并行。

Decode 阶段是典型的 **Memory-Bound** ：瓶颈不在 FLOPS，而在于要不断从显存搬运模型权重和 KV Cache。更雪上加霜的是硬件趋势——从 A100 到 H100，FLOPS 涨了 2 倍多，显存却卡在 80 GB 原地踏步。显存瓶颈只会越来越突出。

### Batch 是解药，但显存卡住了 Batch

既然单次 Decode 喂不饱 GPU，自然的思路是把多个请求攒成一个 Batch，摊薄权重搬运的开销。业界已有的 Iteration-Level Scheduling（如 Orca）解决了请求到达时间不同、长度不一的问题：每轮迭代结束后完成的请求离开、新请求加入，避免了排队和 Padding 浪费。

但 Batch Size 的上限并不由调度决定，而是由 **显存容量** 决定——准确地说，由 KV Cache 占了多少显存决定。以 13B 的 OPT 模型为例，单个 token 的 KV Cache 就要 800 KB（2（key + value）× 5120（hidden size）× 40（层数）× 2 字节（FP16）），一条 2048 token 的请求就是 1.6 GB。一张几十 GB 的 GPU，撑死也就装下几十条请求。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/memory-distribution.png)

> 图解：在 A100（40 GB）上 serving 13B 模型的显存分布。灰色是模型参数（约 65%，静态常驻），红色是 KV Cache（近 30%，随请求动态分配释放），黄色是转瞬即逝的激活值。参数和激活都动不了， **KV Cache 的管理方式直接决定了 Batch Size 的上限** 。

## 二、分析问题：现有系统的显存浪费有多严重？

既然瓶颈在 KV Cache 的显存管理，那现有系统浪费在哪里？论文给出了清晰的解剖。

### 三种浪费：Reserved、内部碎片、外部碎片

现有深度学习框架要求张量连续存储，所以 FasterTransformer、Orca 等系统都为每条请求 **按最大可能长度预分配一整块连续显存** ，不管实际输入输出有多长。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/baseline-memory-management.png)

> 图解：现有系统的 KV Cache 管理。请求 A 按最大长度 2048 预留，请求 B 按 512 预留。浪费分三种： **Reserved** （为未来 token 预留的空位，迟早会用到但长期占着茅坑）、 **内部碎片** （实际长度远小于最大长度，多分配的部分永远用不上）、 **外部碎片** （buddy allocator 等分配器产生的空隙，永远无法使用）。注意同一 token 在不同位置的 KV Cache 不同，所以空位不能随意挪用。

### 量化结果：有效显存利用率最低仅 20.4%

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/memory_breakdown.png)

> 图解：各系统在真实 workload 下的 KV Cache 显存浪费比例。可以看到，现有系统中只有 20.4% - 38.2% 的显存真正存了 token 状态，其余全部被预留和碎片吃掉；而 vLLM 几乎把浪费压到了零。

这个设计的聪明之处不在于发现问题，而在于 **敢于推翻"张量必须连续存储"这条深度学习框架的铁律** ——这正是 PagedAttention 的出发点。

此外还有一个结构性问题：连续存储让 KV Cache **无法共享** 。Parallel Sampling、Beam Search 等解码算法中，多条序列本可以共享大量前缀的 KV Cache，但各存各的连续空间，共享无从谈起。

## 三、解决问题：PagedAttention 与 vLLM

### PagedAttention：让 KV Cache 离散存放的 Attention 算法

核心思路一句话概括： **把操作系统的分页机制搬进 Attention** 。Block 类比 Page，token 类比字节，请求类比进程。

PagedAttention 把每条序列的 KV Cache 切成固定大小的 **KV Block** ，每个 Block 装 $B$ 个 token 的 key/value 向量。记第 $j$ 个 key block 为 $K_j = (k_{(j-1)B+1}, \ldots, k_{jB})$，value block 为 $V_j = (v_{(j-1)B+1}, \ldots, v_{jB})$，Attention 计算改写为分块形式：

$$
A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil}\exp(q_i^\top K_t \mathbf{1} / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top
$$

其中 $A_{ij}$ 是 query 在第 $j$ 个 KV Block 上的注意力分数行向量。Kernel 逐个 Block 取数、算分、加权求和—— **这些 Block 在物理显存里完全可以不连续** 。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/pagedattention.png)

> 图解：PagedAttention 计算示例。Query token 是 "forth"，它的 key/value 历史被分散存放在 3 个物理上不连续的 Block 中。Kernel 依次取每个 Block（比如 Block 0 里是 "Four score and seven" 的 key 向量）与 $q_i$ 计算分数 $A_{ij}$，再与对应 value block $V_j$ 相乘累加，得到最终输出 $o_i$。

### KV Cache Manager：逻辑块与物理块分离

有了能处理离散 Block 的 Attention 算法，上层的内存管理就可以完全照搬 OS 虚拟内存的设计：

- **逻辑 KV Block** ：每条请求视角下自己的 KV Cache，从左到右依次填充，最后一个 Block 的空位留给未来生成的 token。
- **物理 KV Block** ：GPU Worker 上的 Block Engine 把一整块显存切成固定大小的物理块，按需分配。
- **Block Table** ：维护逻辑块到物理块的映射，每项记录物理块号和已填充的位置数。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/system-overview.png)

> 图解：vLLM 系统架构。中央调度器（Scheduler）协调多个分布式 GPU Worker；KV Cache Manager 以分页方式管理显存，通过 Block Table 把指令下发给各 Worker 的 Block Engine。CPU 侧也有一套 Block Allocator，用于 Swapping。

这样，显存按需增长、用完即释放：由于所有 Block 从左到右填满、只有最后一个 Block 可能有空位， **每条请求的浪费被限制在一个 Block 以内** ，内部碎片近乎消除；所有 Block 大小相同，外部碎片直接消失。

### 一次解码的完整过程

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/logical-and-physical-block-table.png)

> 图解：vLLM 的 Block Table 翻译过程。① Prompt 有 7 个 token，系统只分配必需的 2 个逻辑块（0、1），映射到物理块 7 和 1，Prefill 阶段把 KV Cache 填入；② 第一个 Decode step 生成新 token，逻辑块 1 还有空位，直接写入并更新 \#filled 计数；③ 第二个 Decode step 时最后一个逻辑块已满，系统分配新的物理块 3 并登记映射。

全局来看，每轮迭代调度器先选定本批候选序列、为新逻辑块分配物理块，然后把所有输入 token 拼成一条序列送入模型，PagedAttention Kernel 按 Block Table 读写 KV Cache。下图展示了两个请求同时复用显存池的情形：

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/multi-sequence-block-mapping.png)

> 图解：两条序列的 KV Cache 共存于同一物理显存池。两个请求的逻辑块映射到互不连续的物理块，空间被交错利用——这正是"逻辑连续、物理离散"的威力。

## 四、进阶场景：共享 KV Cache 才是大头

解决了单序列的浪费问题后，下一个问题是： **多序列之间能不能共享显存？** 这正是分页设计的杀手锏——Block 粒度的共享 + Copy-on-Write。

### Parallel Sampling：共享 Prompt

代码补全类应用（如 Copilot）常对同一个 prompt 采样多个候选输出。vLLM 让多个输出序列的逻辑块映射到 **同一份物理块** 来共享 prompt 的 KV Cache，并给每个物理块加 **引用计数（Reference Count）** 。

当某个样本要写入一个引用计数大于 1 的共享块时，触发 **Copy-on-Write** ：分配新物理块、拷贝数据、引用计数减一，之后的写入各走各的。这和 OS 里 fork 进程时的写时复制一模一样。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/parallel-decoding.png)

> 图解：Parallel Sampling 示例。样本 A1、A2 的 prompt 部分共享物理块 7 和 1（引用计数均为 2）。A1 要写逻辑块 1 时触发 Copy-on-Write：新分配物理块 3 拷贝原数据，物理块 1 的引用计数降为 1，此后 A2 直接写物理块 1。除最后一个逻辑块外，prompt 的 KV Cache 全部只存一份——prompt 越长省得越多。

### Beam Search：动态演化的共享树

Beam Search 每步保留 top-$k$ 候选，候选之间不仅共享 prompt，还共享大量中间前缀，且共享关系随解码动态变化——就像 OS 里多次 fork 形成的进程树。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/beam-search.png)

> 图解：Beam width $k=4$ 的 Beam Search。虚线之前，4 个候选共享 Block 0（prompt），候选 0-2 共享前 3 个 Block。新一轮迭代后，top-4 候选全部来自原候选 1 和 2，落选候选的逻辑块被释放，引用计数归零的物理块（2、4、5、8）被回收，再分配新物理块 9-12 给新候选。

以往系统处理 Beam Search 需要频繁地大段拷贝 KV Cache（候选 3 要复制候选 2 的大部分历史才能继续生成），vLLM 用物理块共享把拷贝开销降到 **最多一个 Block** ——只有写入共享块时才 Copy-on-Write。

### Shared Prefix：像共享库一样共享系统提示词

很多应用的 prompt 都带着一长串相同的任务描述和示例（System Prompt）。vLLM 可以像 OS 处理共享库一样，预先把这些前缀的 KV Cache 固定在物理块里，用户请求直接把逻辑块映射过去（最后一个块标记 Copy-on-Write），Prefill 只需计算用户自己的输入部分。

![Figure 10](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/share-prompt.png)

> 图解：机器翻译任务中的共享前缀示例。指令和翻译示例构成共享前缀，不同用户的实际输入拼接其后，共享部分的 KV Cache 只算一次、只存一份。

### 混合解码：统一抽象带来的自由

更妙的是，因为所有共享复杂性都被"逻辑块 → 物理块"这一层映射隐藏了，LLM 和 Kernel 看到的只是每序列一串物理块 ID。 **使用不同解码算法的请求可以混在同一个 Batch 里跑** ——这是现有系统做不到的，进一步扩大了 Batch 的空间。

## 五、调度、抢占与分布式

显存总会有耗尽的一刻。流量超过系统容量时，vLLM 需要回答两个经典问题：踢谁？怎么恢复？

### All-or-Nothing 抢占与两种恢复方式

vLLM 采用 FCFS（先来先服务）保证公平、防止饥饿，抢占时最新到达的请求先被踢出。关键洞察是： **一条序列的所有 Block 总是一起被访问** ，所以采用 All-or-Nothing 驱逐——要么整条序列的 Block 全踢，要么全留。同一请求内的多条序列（如 Beam 候选）作为一个 **Sequence Group** 被成组调度。恢复方式有两种：

- **Swapping** ：把被踢 Block 拷到 CPU 内存，恢复时再拷回来。由于被抢占的序列数有限，Swap 空间不会超过 GPU 显存中 KV Cache 的总量。
- **Recomputation** ：直接重算。被抢占序列的已生成 token 可以和原 prompt 拼成新 prompt，一次 Prefill 就把全部 KV Cache 算回来，延迟远低于重新逐 token 生成。

值得一提的是，这两种机制都是 OS 虚拟内存做不到的"魔改"——OS 可没法通过"重算"来恢复被换出的页。笔者认为这正是论文的精髓： **不是照搬 OS，而是用 LLM 的应用语义重新诠释分页思想** 。

### 分布式执行

对于单卡放不下的模型，vLLM 支持 Megatron-LM 式张量并行（SPMD，Attention 按 head 切分）。由于每个模型分片处理的是同一批 token、需要相同位置的 KV Cache，vLLM 只用一个 **集中式 KV Cache Manager** ：每步迭代开始时，调度器把输入 token ID 和各请求的 Block Table 广播给所有 Worker，之后 Worker 按 Block Table 自行读 KV Cache，中间用 All-Reduce 同步，无需调度器介入。显存管理信息随输入一起下发，Worker 之间不需要为内存管理做任何同步。

## 六、实现要点

vLLM 引擎由 8.5K 行 Python（调度器、Block Manager 等控制逻辑）和 2K 行 C++/CUDA（关键 Kernel）构成，前端用 FastAPI 提供兼容 OpenAI API 的接口，支持 GPT、OPT、LLaMA 等模型。三个关键的 Kernel 级优化：

- **Fused Reshape + Block Write** ：把新 KV Cache 的切分、重排、按 Block Table 写入融合成一个 Kernel，减少启动开销。
- **Fused Block Read + Attention** ：改造 FasterTransformer 的 Attention Kernel，按 Block Table 边读边算，每个 Block 分配一个 Warp 保证合并访存。
- **Fused Block Copy** ：Copy-on-Write 会产生大量不连续的小块拷贝，把它们批量化到一次 Kernel Launch 中，避免 `cudaMemcpyAsync` 的频繁调用。

各种解码算法则统一用 `fork`（从已有序列派生新序列）、`append`（追加 token）、`free`（删除序列）三个原语实现——Parallel Sampling 是 fork + append + free，Beam Search 和前缀共享同理，未来的新解码算法也可以靠组合这三个原语支持。

## 七、实验验证

### 实验设置

- **模型与硬件** ：OPT-13B / 66B / 175B 与 LLaMA-13B，Google Cloud A100（13B 单卡、66B 四卡、175B 八卡）。
- **Workload** ：ShareGPT（真实 ChatGPT 对话，输入平均比 Alpaca 长 8.4 倍、输出长 5.8 倍）与 Alpaca 两个数据集，按 Poisson 分布合成请求到达。
- **Baseline** ：FasterTransformer（配动态 batching 调度器）；Orca 三个变体——Oracle（预知输出长度，性能上限）、Pow2（最多超配 2 倍）、Max（按 2048 拉满预留）。
- **指标** ：Normalized Latency（端到端延迟 ÷ 输出长度），高吞吐系统应在高请求速率下保持低 Normalized Latency。

### Basic Sampling：吞吐提升 2-4 倍，对 FasterTransformer 最高 22 倍

![Figure 11](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/n1-sharegpt.png)

> 图解：ShareGPT 数据集上单序列生成的 Normalized Latency 随请求速率变化的曲线。请求速率超过系统容量后延迟爆炸式增长，因此"曲线拐弯点越靠右"代表吞吐越高。vLLM 能承受的请求速率是 Orca (Oracle) 的 **1.7-2.7 倍** 、Orca (Max) 的 **2.7-8 倍** 。

![Figure 12](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/n1-alpaca.png)

> 图解：Alpaca 数据集上的同口径实验，趋势一致。唯一的例外是 OPT-175B + Alpaca：8 卡 640 GB 显存对短序列来说太宽裕，连 Orca 都能装下大 Batch，系统退化为 Compute-Bound，vLLM 的显存优势无从发挥——这反过来印证了"vLLM 的收益来自显存效率"这一论点。

吞吐差距的直接原因是 Batch Size：服务 OPT-13B 时，vLLM 同时处理的请求数是 Orca (Oracle) 的 2.2 倍、Orca (Max) 的 4.3 倍。

![Figure 13](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/batched_requests_sharegpt.png)

> 图解：ShareGPT trace（2 req/s）下各系统的平均并发请求数对比。PagedAttention 省下的显存直接转化成了更大的 Batch。

### Parallel Sampling 与 Beam Search：共享带来额外红利

![Figure 14](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/parallel.png)

> 图解：Alpaca 上 Parallel Sampling 的结果。采样数越多，prompt 共享省下的显存越多，vLLM 相对 Orca 的优势越大。

![Figure 15](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/beam.png)

> 图解：Alpaca 上 Beam Search 的结果。Beam Search 的共享更充分，vLLM 相对 Orca (Oracle) 的优势从 Basic Sampling 的 1.3 倍扩大到 Beam width = 6 时的 **2.3 倍** 。

共享到底省了多少显存？看下面这组数据：

| 场景 | Alpaca 节省比例 | ShareGPT 节省比例 |
| --- | --- | --- |
| Parallel Sampling | 6.1% - 9.8% | 16.2% - 30.5% |
| Beam Search | 37.6% - 55.2% | 44.3% - 66.3% |

> 表解：Block 共享带来的显存节省（节省块数 ÷ 不共享时的总块数）。Beam Search 能省掉一半以上的显存，prompt 更长的 ShareGPT 上收益全面更高。

### Shared Prefix 与 Chatbot

翻译任务（LLaMA-13B + WMT16 英德数据集）中，共享 1 个示例的前缀时 vLLM 吞吐是 Orca (Oracle) 的 **1.67 倍** ，共享 5 个示例（341 token 前缀）时达到 **3.58 倍** ——前缀越长，预缓存的价值越大。

![Figure 16](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/chat-sharegpt.png)

> 图解：Chatbot 场景（ShareGPT 对话历史拼接，prompt 截断到 1024 token）。由于 prompt 普遍很长，三个 Orca 变体都被 buddy allocation 的预留策略拖累、表现趋同，vLLM 则能承受 **2 倍** 的请求速率。

### 消融实验：代价与权衡

天下没有免费的午餐，PagedAttention 的间接寻址是有开销的：

![Figure 17](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Memory-Management-for-Large-Language-Model-Serving-with-PagedAttention/figures/experiments/micro_latency.png)

> 图解：Attention Kernel 的微基准延迟对比。访问 Block Table、额外分支、变长序列处理使 vLLM 的 Attention Kernel 比高度优化的 FasterTransformer 慢 **20% - 26%** 。但开销只影响 Attention 算子，不影响 Linear 等其他算子，端到端依然是数倍的优势——用 20% 的 Kernel 开销换 2-4 倍的吞吐，这笔账非常划算。

**Block Size 的权衡** ：Block 太小，GPU 并行度吃不饱；太大，内部碎片和共享概率恶化。ShareGPT 上 16-128 都不错，Alpaca 上超过 32 就明显变差（序列比 Block 还短），最终默认取 **16** 。

**Swapping vs Recomputation** ：小 Block 时 Swapping 会产生大量细碎传输、跑不满 PCIe 带宽，Recomputation 更优；大 Block 时 Swapping 更优；Block Size 16-64 时两者端到端性能相当，且 Recomputation 的开销从不高于 Swapping 延迟的 20%。

## 八、讨论与结语

论文的讨论章节有一个清醒的提醒：分页思想之所以对 LLM Serving 有效，是因为这个负载 **既需要动态内存分配（输出长度未知），又被显存容量卡住咽喉** 。DNN 训练的张量形状是静态的，非 LLM 的推理通常是 Compute-Bound，硬套 vLLM 的技术反而会引入间接寻址的开销——好思想也要用在对的负载上。

最后回顾一下这篇工作的核心要点：

- **问题** ：LLM Serving 是 Memory-Bound，KV Cache 的显存管理决定 Batch Size 上限，现有系统有效显存利用率仅 20.4% - 38.2%。
- **方法** ：PagedAttention 把 KV Cache 分页化、离散存储，注意力计算改写为分块形式，Kernel 按 Block Table 间接寻址。
- **共享** ：逻辑块/物理块分离 + 引用计数 + Copy-on-Write，让 Parallel Sampling、Beam Search、共享前缀的 KV Cache 以 Block 粒度共享，Beam Search 场景最高省 66.3% 显存。
- **调度** ：All-or-Nothing 抢占 + Swapping/Recomputation 两种恢复方式，是对 OS 虚拟内存的 LLM 语义化改造。
- **结果** ：延迟持平下吞吐提升 2-4 倍（对 FasterTransformer 最高 22 倍），代价仅是 Attention Kernel 20% - 26% 的额外开销。

这项工作最深远的影响或许不在于论文本身，而在于它开启的方向：vLLM 已成为事实上的 LLM 推理引擎标准之一，"把 OS 经典思想引入 AI 系统"也成为了后续大量系统工作的范式。其局限也很明确——收益依赖 Memory-Bound 这一前提，显存充裕或计算主导的负载下优势会收窄。

> 本文参考自 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)