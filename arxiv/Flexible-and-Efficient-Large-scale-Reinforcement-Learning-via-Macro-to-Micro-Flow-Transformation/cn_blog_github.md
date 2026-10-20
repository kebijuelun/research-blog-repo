# RLinf：宏流到微流变换，让大规模 RL 训练又快又灵活

强化学习正在成为大模型继续进化的主引擎，但 RL 训练工作流由推理、生成、训练、仿真器等高度异构的组件拼成，现有框架要么把所有组件挤在同一批 GPU 上排队跑（collocated），要么完全拆开做流水线（disaggregated），两种死板的执行模式都撑不起多样化的 RL 场景。这篇来自清华、无问芯穹等机构的工作提出了 RLinf，核心思路一句话概括：**让用户按“宏观逻辑流”自然地写 RL 流程，由系统自动变换成“微观执行流”去调度硬件**。效果相当硬：在数学推理 RL 上比 veRL、Slime 等 SOTA 系统快 1.07~1.70 倍，在具身智能 RL 上最高提速 2.43 倍，而自动搜索最优执行模式的开销在 1024 卡规模下也不到 6 秒。

## 背景：LLM 时代的 RL 工作流有多复杂

随着预训练 Scaling 的收益放缓，RLHF、GRPO、具身智能、Deep Research 等方向都转向用 RL 继续推高模型智能。OpenAI 等机构甚至预测，RL 消耗的算力很快会超过预训练——这让 RL 训练效率成为最关键的系统问题。

和传统 RL 不同，LLM 时代的 RL 回路里往往挂着好几个大模型。论文梳理了四类代表性工作流：

![多样化 RL 工作流](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/rl_workflows.png)

> 图解：四种典型的 RL 工作流。**GRPO** 最简洁——单个 LLM 对一个 query 生成多条回答（Generation），计算对数概率（Inference），再更新自身（Training）。**RLHF（PPO）** 则要复杂得多：actor 生成回答、reference 模型约束其不要跑偏、reward 模型打分、critic 估计期望回报，四个模型紧密交互。**具身智能**场景中 LLM 与物理仿真器循环交互产生轨迹；**Deep Research** 场景则引入搜索服务器，形成带环的数据流。

这些工作流有两个让人头疼的特征：

- **组件异构性强**：训练需要保存梯度和优化器状态，显存需求远大于推理；生成（Generation）受制于显存带宽，GPU 计算核心经常吃不饱；仿真器可能跑在 CPU 上，或者用 GPU 做 3D 渲染这种非张量计算。并行化方式也千差万别——LLM 训练有数据/张量/流水线并行，仿真器只能靠多开实例。
- **依赖关系复杂**：数据流粒度不一（生成到推理可以是逐条回答，生成到训练至少要一个 micro-batch），有些场景还存在环形数据流；权重更新又充当全局屏障（barrier），把生成和训练同步在一起。

把这么多胃口迥异的组件塞进同一个集群，还要让每块 GPU 都不闲着——这就是 RLinf 要解决的核心矛盾。

## 问题分析：为什么现有执行模式都不行

现有系统的执行模式大致是两个极端，论文用实验数据说明了它们各自的死穴。

### Rollout 的动态性：长尾拖垮一切

![响应时间 CDF](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/rollout_time_cdf.png)

> 图解：一次 7B 数学推理 RL 训练（8 节点 × 8 张 H100）中，生成阶段响应时间的累积分布（CDF）。横轴是响应耗时，纵轴是累积比例，可以看到响应长度差异极大，呈明显的长尾分布。

![未完成响应数随时间变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/rollout_time_fraction.png)

> 图解：同一次实验中，未完成响应占比随时间的衰减曲线。未完成数量很快降到 5% 以下，但就是这极少数长尾响应卡住整个生成阶段，让大量 GPU 空转。GPU 越多，浪费越严重。

Rollout 阶段天然是动态的：同一个 query 的回答长度不一，具身任务的抓取步数不同，Deep Research 的搜索轮数也不同。而 Rollout 是批量执行的，几条“慢性子”响应就会阻塞整个阶段进入推理或训练。collocated 模式（如 veRL）里所有阶段串行共享 GPU，这个问题被无限放大。

流水线模式可以缓解长尾——只给生成阶段分少量 GPU，推理和训练在部分样本就绪后就开工。但它也有自己的坑：推理和训练必须等第一批数据生成完才能启动。**两种模式都不是普适最优，而在一个框架里同时支持两者是个硬骨头。**

### 组件特性多样：混合模式才是刚需

![生成耗时与 batch size 的关系](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/time_vs_batch_size.png)

> 图解：生成阶段的执行时间随 batch size 线性增长，同时 GPU 核心利用率始终高于 70%——生成是“吃算力也吃显存”的组件。

![仿真器耗时与环境数量的关系](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/time_vs_envs.png)

> 图解：仿真器的执行时间随并行环境数量增加得很少，GPU 利用率不足 24%，但显存占用随环境数线性增长——它是典型的“显存大户、算力闲人”。

以具身 RL 为例，三个组件的画像完全不同：

- **仿真器**：加环境几乎不加耗时，但显存线性涨，GPU 利用率低于 24%。它应该尽量多开并行环境，独占 GPU。
- **生成**：耗时和显存都随 batch size 线性涨，核心利用率高。适合和仿真器分开、用流水线衔接。
- **训练**：显存需求最大，但耗时只有生成的三分之一。独占 GPU 纯属浪费，应该在 Rollout 结束后通过换入换出接管 GPU。

结论很清晰：最优编排是**混合模式**——生成和仿真器空间上分离、流水线执行，训练时间上和它们复用同一批 GPU。问题来了：面对特性各异、依赖复杂的任意工作流，靠人肉枚举找到这个最优编排既繁琐又容易漏掉更好的选择，而且没有任何现成指南。

论文由此点题：**灵活性本身就是效率的关键**。系统必须能让开发者保持直觉化的流程编程，同时在执行层自由切换粗粒度（阶段级）和细粒度（batch 级）的调度。

## 核心设计：M2Flow 范式

### 总体架构

![RLinf 系统架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/rlinf_arch.png)

> 图解：RLinf 的架构。用户通过过程式编程接口构建 RL 工作流；每个 RL 组件被封装为 **Worker**，自带通信能力和资源 onload/offload 机制。核心的调度模块基于 **profiling 引导的调度策略** 搜索每个 Worker 的最优执行模式，由 Controller 负责设备分配、连接管理和函数分发。**弹性流水线（elastic pipelining）**和**上下文切换（context switch）**分别支撑空间和时间维度的编排，点对点通信与数据通道（data channel）构成数据面，整个系统基于 Ray 启动和管控 Worker。

M2Flow（macro-to-micro flow transformation）是全文的核心范式：**宏观逻辑流负责表达，微观执行流负责效率**。开发者用命令式的方式、以粗粒度描述组件之间的逻辑通信关系；系统自动把它变换成细粒度的执行流。本质是**把程序语义和物理执行解耦**——代码逻辑不变，调度方式可以在巨大的空间里自由探索。

这个范式通过三个维度的调度落地：

- **空间调度**：决定 Worker 被放到哪些加速器上；
- **时间调度**：决定 Worker 的执行时段，即设备的时间复用；
- **时空联合调度**：决定流水线执行的数据粒度。

### 编程接口：过程式而非图声明式

与 TensorFlow 式的图声明编程不同，RLinf 选择过程式编程范式——牺牲一点点“全局图优化”的机会，换来控制流的灵活性、可调试性和透明度。一个 RLinf 程序由两部分组成：

- **Worker 程序**：定义每个 RL 组件的逻辑。基类 `Worker` 提供 `send`/`recv` 通信原语，子类继承后自动获得跨组件通信能力；同时必须实现 `onload`/`offload` 来申请和释放 GPU 显存等有限设备资源。
- **Workflow Runner**：编排整体流程。以 SPMD 方式在集群上启动各 Worker，同一 Worker 的所有进程由 `WorkerGroup` 统一管理；调用 WorkerGroup 的函数是异步的，返回结果句柄，`wait()` 则提供同步屏障。

笔者认为这里有个聪明的设计：**数据通道（data channel）把控制流和数据流解耦**。比如 GRPO 里 Rollout 可以逐 query 推进，但归一化必须凑齐一个 query 的所有回答——用 `wait` 屏障配合数据通道，就能精确表达“在哪一步停一下”的粒度，同时给系统留出广阔的优化空间。论文给出的一个典型 LLM 推理 RL 工作流 Runner 不到 100 行代码，且在时间/空间两种调度间切换**不需要改任何代码**。

### M2Flow 变换的两大机制

有了逻辑流，接下来要回答“每个 Worker 在哪跑（空间）、何时跑（时间）”。RLinf 用两个机制打通这条路。

**机制一：弹性流水线（Elastic Pipelining）支撑空间调度。** 关键洞察是：RL 训练和 Agent 场景中，大多数 Worker 遵循 SPMD 模式，天然支持不同 batch size 的执行——SGLang、vLLM 可以一次处理一个 prompt 或一串 prompt，推理（prefill-only 计算）也支持单批或多批。因此执行流管理器可以**动态调整数据粒度**：输出攒够配置大小就转发给下游，让下游用小 batch 早开工，或攒大 batch 再开工。训练 Worker 还有 micro-batch（决定前反向单元）和 global-batch（决定何时更新模型）两个旋钮，进一步扩大了调度空间。

**机制二：自动上下文切换（Context Switching）支撑时间调度。** 当多个 Worker 无法同时驻留同一块 GPU（显存不够）时，让它们分时复用。实现上是一把**分布式设备锁**（`device_lock`）：Worker 用资源前先抢锁，锁状态全局一致、只能原子修改，保证独占访问；抢到锁后若资源已被换出则自动 `onload`，干完活释放锁并 `offload` 腾出空间。这把锁比普通锁多两点考量：

- 利用数据通道里的数据依赖关系定义锁优先级——下游 Worker 必须等上游把数据入队并释放锁后才能抢锁，避免锁竞争和死锁；
- 利用 Controller 的设备放置信息，Worker 在不同设备上时跳过不必要的换入换出。

![M2Flow 执行逻辑](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/m2flow.png)

> 图解：M2Flow 的执行逻辑。用户写的是命令式 for 循环（Rollout、Inference、Training 三个 Worker），Execution Flow Manager 可以把输入数据切成更小的块让 Worker 细粒度处理（空间维度的弹性流水线），也可以合并成更少、更大的子任务（时间维度调整）；设备锁则在有数据依赖的 Worker 之间自动管理资源换入换出。

![Worker 的空间与时间调度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/worker_scheduling.png)

> 图解：三种代表性执行模式。**左（纯时间调度）**：每个 Worker 独占全部加速器，依次换入换出——适合大模型训练这种“必须占满所有卡才跑得动”的场景，但受 Rollout 长尾拖累，GPU 空闲明显。**中（纯空间调度）**：Worker 各据一批 GPU，靠流水线掩盖空闲，关键是平衡各方资源让执行时间对齐。**右（混合调度）**：一部分 Worker 分离式流水线执行，阶段完成后换出、由后续 Worker 接管——复杂 RL 工作流的最优解往往在这里。

### 调度策略：Profiling 引导的自动搜索

调度空间这么大，怎么找到最优执行模式？RLinf 拆成两个模块。

**Profiler** 负责测量每个组件在不同数据并行度下的执行时间和显存占用（数据并行度和吞吐通常呈正相关甚至近线性关系）。对非模型组件（如仿真器），数据并行度就是实例数；对模型组件，由用户提供模型并行配置。然后用多项式外推得到更大并行度下的估计，输出每个组件的执行时间估计函数 $E$。与此同时，整个 RL 工作流通过追踪通信原语以 JIT 方式被捕获为一张工作流图。

**Scheduler** 拿到工作流图 $G$、估计函数 $E$ 和 GPU 总数 $N$ 后，用动态规划递归地把图沿 s-t 割（s-t cut）切成两个子图 $G_s$ 和 $G_t$，对每一刀分别评估两种代价：

- **时间调度**：两个子图共享同一批设备，代价是执行时间之和加上换出换入开销；
- **空间调度**：两个子图分占不相交的设备集做流水线，运行时估计为

$$
T_{\text{critical}} + (M/m - 1) \times T_{\text{bottleneck}}
$$

其中 $T_{\text{critical}}$ 是流水线的预热和排空时间，$T_{\text{bottleneck}}$ 是最慢子图的运行时，$M$ 是总 batch 大小，$m$ 是数据处理粒度。

递归直到子图缩为单节点，返回其 profile 出的执行时间。对于具身、Deep Research 这类带环的工作流，图在预处理阶段把环坍缩成单节点，递归到该节点时把计算均匀切分到各 GPU，避免穷举分区的同时保持接近最优。

### 自适应通信：数据面的地基

灵活的编排要求通信层满足两个目标：**灵活**（任意两个 Worker 无论放在哪都能通信）和**自适应**（数据在 CPU/GPU、跨节点之间都能跑满底层链路带宽）。NCCL、Gloo、MPI 这些库都是为“固定进程数的标准通信模式”设计的，既不支持灵活的 rank 伸缩和高效的同设备通信，也处理不了“多个大小不一的 buffer 组成的复杂数据结构”。RLinf 的应对方案分三层：

- **协议层**：透明的连接生命周期管理。Worker 启动时把放置位置、IP、端口注册到全局 Worker 管理器，连接在首次调用通信原语时才惰性建立；Worker 终止时连接管理器通知所有对端拆除连接。
- **原语层**：提供同步/异步 `send`/`recv` 和 `broadcast` 等集合通信，并根据通信双方的位置自动选最优后端——GPU-GPU 走 NCCL、同 GPU 内走零拷贝 cudaIPC、CPU 走 Gloo。payload 支持任意 Python 对象，采用结构感知的序列化：buffer 直接从对象中抽出传输，不经序列化开销，结构信息随通信元数据捎带。
- **数据通道**：FIFO 队列式的生产者-消费者通信设施，由专门的 channel Worker 进程维护队列，任何 Worker 凭句柄访问；支持 GPU 数据卸载到 CPU 以省显存；还带负载均衡能力——入队条目可带权重，消费者可自定义均衡策略，在每次出队时挑选合适的条目。

## 实现：20K 行 Python 撑起通用性

RLinf 用约 20K 行 Python 实现：核心 Worker、Controller、调度器占 5K 行；基于 SGLang/vLLM/HuggingFace 的 Rollout Worker、基于 Megatron/FSDP 的训练 Worker、具身仿真器等开箱即用的通用 Worker 占 2K 行；其余 13K 行是 PPO、GRPO、DAPO、REINFORCE++ 等算法（含异步 off-policy 变体）的丰富支持。

两个值得一提的实现细节：

- **集群管理**：用 Ray 启动和管控远程 Worker 进程，但**不依赖 Ray 分配设备**——Ray 只有 packed/spread 两种刚性策略。RLinf 允许通过全局设备 ID 把任意节点上的任意设备分给任意 Worker 进程，甚至机械臂这类非加速器硬件也被抽象为同样的可调度设备。
- **性能剖析**：为每个远程调用的公开函数提供 Worker 组级计时器，各进程计时值可按 mean/max/min 等方式归约；开发者还能对任意代码段加自定义计时器。这既方便用户定位瓶颈，也是调度策略的数据来源。

## 实验：推理与具身双线验证

实验在 32 节点集群上进行，每节点 8 张 H100-80GB GPU、2 颗 Intel Xeon Platinum 8558 CPU、2TB 内存；节点内 NVLink，节点间 8 张 Mellanox ConnectX-7 400Gbps RDMA 网卡（RoCEv2）。覆盖 Qwen2.5（1.5B/7B/14B/32B）、Qwen3-30B-A3B（MoE）、OpenVLA、OpenVLA-OFT 四类模型，GRPO 和 PPO 两种算法。

### 数学推理 RL：Qwen2.5 + GRPO/PPO

![Qwen2.5 GRPO 吞吐对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/rlinf_vs_verl_all.png)

> 图解：Qwen2.5 1.5B/7B/32B 在 64/128/256 张 GPU 上的 GRPO 训练吞吐（tokens/sec），RLinf（时间模式）全面超过 veRL，加速比 1.10~1.58 倍。

时间模式的收益主要来自两点：更好的显存管理让 Rollout 能分配更大的 KV-cache，以及中间推理阶段上下游同步开销的降低。值得注意的是 veRL 随 GPU 数增加扩展性变差——推理阶段占总耗时的比例从 64 卡的 15.2% 涨到 256 卡的 19.9%，且其 Rollout 引擎峰值显存过高，压缩了 KV-cache，Rollout 时间只能亚线性下降。

![7B 训练延迟分解](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/latency_breakdown_7B.png)

> 图解：Qwen2.5-7B 训练的延迟分解对比。veRL 的瓶颈集中在 Rollout（KV-cache 受限）和更长的推理阶段，这正是 RLinf 拉开差距的地方。

纯空间模式在这个场景反而比 veRL 慢 44.3~68.6%（7B）：Rollout、推理、训练各占一小撮 GPU 都变慢了，加上 28672 的超长序列让训练等首批数据等得更久。**这恰好印证了论文的立场——没有普适最优的模式，灵活切换才是王道。**

![Qwen2.5 PPO 吞吐对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/rlinf_vs_verl_ppo.png)

> 图解：PPO 设置下（actor 和 critic 都用 Qwen2.5，1.5B/7B/14B，16~256 GPU）的吞吐对比。RLinf-Spatial 按 4:1:1:1:1 给 Rollout、actor 推理、actor 训练、critic 推理、critic 训练分卡。

PPO 的结论与 GRPO **正好相反**：空间模式全面胜出。1.5B 模型上 RLinf-Spatial 比 veRL 快 35.0~69.6%，比 RLinf-Temporal 快 18.6~39.8%；7B 上分别快 38.7~60.7% 和 19.0~44.8%；14B 上也有 27.2~56.5% 的优势。

![7B PPO 延迟分解](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/rlinf_vs_verl_breakdown.png)

> 图解：32 GPU 上 Qwen2.5-7B PPO 的延迟分解，条宽代表 GPU 数量。空间模式下 Rollout 虽比时间模式慢 39.3%，但推理和训练在 Rollout 结束后迅速完成，组件间重叠充分，端到端更快；时间模式相对 veRL 的收益则主要来自 “Others” 一项（上下文切换、参数 resharding、优势计算、阶段间同步）的压缩。

同一个系统，GRPO 选时间模式、PPO 选空间模式——这种“对症下药”的能力正是 M2Flow 的价值所在。

### MoE 模型：Qwen3-30B-A3B + GRPO

![Qwen3 MoE 吞吐对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/rlinf_vs_slime_all.png)

> 图解：Qwen3-30B-A3B（MoE）在 32/64/128 GPU 上与 Slime 的对比。Slime 不支持流水线，其空间模式 Rollout 和训练各占一批卡毫无重叠，最慢；RLinf-Spatial（1:1 分卡）在 32 和 64 卡上比 Slime-Colocate 分别快 31.2% 和 7.2%。

![Qwen3 MoE 延迟分解](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/rlinf_vs_slime_breakdown.png)

> 图解：32 GPU 上 Qwen3-30B-A3B GRPO 的延迟分解，条宽与 GPU 数成正比。时间模式下 Rollout 和训练争抢显存，SGLang 的 max_running_requests 被迫从 256 降到 128；空间模式则避免了显存竞争并实现了良好的 Rollout-训练重叠。128 卡时 RLinf-Temporal 反超 3.7%，因为此时空间模式的 Rollout 结束后训练还要再跑 80 秒，重叠不佳。

### 具身智能 RL：ManiSkill 与 LIBERO

![具身 RL 端到端吞吐](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/eval_embody_all.png)

> 图解：(a) ManiSkill 上 OpenVLA 训练（256 并行环境、每轮 80 步）三种模式的吞吐对比，RLinf-Hybrid 比 Temporal 高 52.2~69.1%，比 Spatial 高 60.7~87.2%；(b) LIBERO 上 OpenVLA-OFT 训练（512 并行环境、64 步），RLinf-Temporal 在 8/16/32 GPU 上比 SimpleVLA-RL 分别快 37.8%、42.6%、143.4%。

两个环境的最优模式不同，原因恰好呼应前文的组件画像分析：

- **ManiSkill**（GPU 渲染）：仿真器加环境几乎不加耗时但吃显存，适合独占 GPU 多开环境；Rollout 和训练都要显存，适合时间复用。Hybrid 模式两者兼得，大幅领先。
- **LIBERO**（CPU 密集）：把环境绑到部分 GPU 上反而会限制 CPU 核的利用，Hybrid 的 Rollout 阶段比 SimpleVLA-RL 还慢；Temporal 模式独占全部资源反而最优，加上消除了冗余的环境初始化、用单次前向同时算动作和 log probability 两个优化，拉开巨大差距。

![具身 RL 延迟分解](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/latency_breakdown_libero_maniskill.png)

> 图解：ManiSkill 与 LIBERO 的延迟分解。RLinf-Temporal 的 Rollout 和训练双双快于 SimpleVLA-RL，收益来源是冗余环境初始化的消除和“一次前向算两样”的融合计算。

### 调度策略本身够不够用

![调度策略评估](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Flexible-and-Efficient-Large-scale-Reinforcement-Learning-via-Macro-to-Micro-Flow-Transformation/figures/evaluation/static_scheduling.png)

> 图解：(a) 四种 Qwen2.5-GRPO 配置下预测与真实迭代时间的对比——时间模式估计误差低于 2%，空间流水线模式因响应长度波动导致的流水线不均衡，误差略高但也在 5% 以内，不影响模式排序；(b) 3 节点工作流图在 8~1024 GPU 下的搜索耗时，随 GPU 数指数增长但始终低于 5 秒（论文报告全场景范围约 $7\times10^{-4}$~5.98 秒）。

### 训练出的模型质量如何

系统提速不能以牺牲效果为代价。论文验证了 RLinf 训出的模型性能：

| 模型 | AIME24 | AIME25 | GPQA | 平均 |
| --- | --- | --- | --- | --- |
| **1.5B 组** | | | | |
| Base model | 28.33 | 24.90 | 27.45 | 26.89 |
| DeepScaleR | 40.41 | 30.93 | 27.54 | 32.96 |
| AReaL | 44.42 | 34.27 | 33.81 | 37.50 |
| FastCuRL | 43.65 | 32.49 | 35.00 | 37.05 |
| **RLinf** | **48.44** | **35.63** | **38.46** | **40.84** |
| **7B 组** | | | | |
| Base model | 54.90 | 40.20 | 45.48 | 46.86 |
| AReaL | 61.66 | 49.38 | 46.93 | 52.66 |
| Skywork | 66.87 | 52.49 | 44.43 | 54.60 |
| Polaris | **68.55** | 51.24 | 43.88 | 54.56 |
| AceMath | 67.30 | **55.00** | 45.57 | 55.96 |
| **RLinf** | 68.33 | 52.19 | **48.18** | **56.23** |

RLinf 1.5B 在三个基准上全面领先，比基座提升最多约 20 分；7B 在 GPQA-diamond 上最高，平均分 56.23 也是全组最佳。

| 模型 | Vision | Semantic | Position | 平均 |
| --- | --- | --- | --- | --- |
| RL4VLA | 80.47% | 75.00% | 81.77% | 79.15% |
| **RLinf** | **82.03%** | **78.35%** | **85.42%** | **81.93%** |

> ManiSkill3 上 OpenVLA 经 RL 训练后的成功率，RLinf 全面超过 RL4VLA 基线。

| 模型 | Spatial | Object | Goal | Long | 平均 |
| --- | --- | --- | --- | --- | --- |
| OpenVLA-OFT (one-traj) | 56.45% | 25.60% | 45.59% | 9.68% | 34.33% |
| **RLinf** | **98.99%** | **98.99%** | **98.99%** | **94.35%** | **97.83%** |

> LIBERO 上 OpenVLA-OFT 经 RL 训练后的成功率，平均从 34.33% 跃升至 97.83%，提升超过 63 个百分点——系统效率和训练效果实现了双赢。

## 相关工作

RL 训练框架大体分两派：collocated 派（DeepSpeed Chat、veRL）把所有阶段放在共享 GPU 上以简化编排；分离派（NeMo-Aligner、OpenRLHF）把组件拆到不同设备换取模块化和扩展性，AReal 还在其上引入异步模型更新。RLinf 与它们的区别在于提供了更灵活的“组件-设备”放置机制，允许针对工作负载特性探索和部署更优的执行配置。此外，Megatron-LM 的混合并行、DeepSpeed 的 ZeRO 系列解决的是 LLM 分布式训练的扩展与显存问题，而 RL 工作流因交互式数据生成和异步更新更加动态；Ray 这类 actor 模型框架则提供了 RLinf 赖以构建的去中心化执行底座。

## 总结

- **问题**：RL 工作流组件异构、依赖复杂，collocated 受长尾拖累、disaggregated 负载不均，单一执行模式无法通吃。
- **范式**：M2Flow 把“宏观逻辑流”与“微观执行流”解耦——用户写直觉化的过程式代码，系统自动完成时空调度变换。
- **机制**：Worker 抽象 + 弹性流水线（空间）+ 自动上下文切换（时间）+ profiling 引导的动态规划调度 + 自适应通信，一套组合拳撑起完整调度空间。
- **结果**：推理 RL 提速 1.07~1.70 倍，具身 RL 最高提速 2.43 倍；调度搜索在 1024 卡下不到 6 秒、估计误差小于 5%；训出的模型在 AIME/GPQA/LIBERO 等基准上达到或超过 SOTA。
- **开放**：代码已开源（GitHub: RLinf/RLinf），20K 行 Python，典型工作流不到 100 行且切模式零改动。

一句话展望：论文把 M2Flow 视为未来 AI runtime 的蓝图——训练、推理、仿真、推理链等异构组件在同一执行框架下灵活编排，RLinf 是迈向“AI 工作负载操作系统”的早期一步；其在超大规模异构硬件上的调度精度与在线动态重调度能力，值得继续关注。

> 本文参考自 [Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation](https://arxiv.org/abs/2509.15965)