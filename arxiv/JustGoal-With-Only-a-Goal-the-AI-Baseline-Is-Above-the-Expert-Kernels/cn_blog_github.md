# JustGoal：只给目标，8/10 算子更快

大语言模型训练和推理的效率，往往取决于少数几个 GPU kernel，但高性能实现仍需要专家围绕硬件反复调优，耗时可达数周甚至数月。随着基础模型变强，一个关键问题浮现：专用 Kernel Agent 中的评测设施、领域知识和搜索流程，到底还能带来多少增量价值？JustGoal 给出的测量方法很直接：只告诉通用 Agent 优化目标与约束，不提供任何 kernel 专用工具链，让它自己搭建评测、定位瓶颈并迭代。在 NVIDIA B200 上，它在进入比较的 10 个算子中有 8 个快于人类维护的开源库，领先算子的几何平均加速为 1.14–2.34×；在 9 个存在公开专用 Agent kernel 的算子中，也有 5 个快于其中的最快版本。

## 为什么需要“只给目标”的基线

高性能 kernel 的难点，不是把数学公式翻译成 CUDA，而是同时决定 tiling、存储层级利用、异步流水线和硬件指令。这些选择彼此耦合，最优解还会随 GPU 代际变化。

FlashAttention 的演进就是典型例子：初代通过 tiling 减少 HBM（高带宽存储器）与片上 SRAM 之间的数据搬运；FlashAttention-3 围绕 Hopper 的异步执行能力重排计算和传输；FlashAttention-4 又针对指数运算、shared memory 访问等新瓶颈重做算法与流水线。算子定义没变，硬件一变，实现仍要重新审视。

模型架构也在持续制造新需求。DeepSeek 的 MLA（Multi-head Latent Attention）、DSA（DeepSeek Sparse Attention），以及 GDN（Gated DeltaNet）、Kimi Delta Attention，都带来了不同的状态更新和访存模式。再叠加 prefill/decode、前向/反向、精度、序列长度和 batch size，kernel 空间不断分叉。

现有 LLM Kernel Agent 通常是在通用 Agent 上叠加三层专用机制：

- **执行反馈与搜索**：用编译、运行和 profiling 结果组织迭代，例如进化搜索、树搜索或停滞干预。
- **独立验证**：防止实现钻评测漏洞。KernelBench-Verified 曾报告，一个生成 ReLU kernel 对已知测试 shape 直接返回输入，表面加速达到 374×。
- **领域知识与诊断工具**：提供 CUDA 文档、生产代码经验、NCU 指标解析或 kernel 内事件追踪，帮助模型回答“为什么慢”。

这些机制都针对当时模型的短板而设计，但模型升级后，短板可能已经成为原生能力。若不拿掉专用层做一次对照，就无法知道收益究竟来自新模型，还是来自外挂系统。

| 系统 | 专用评测机制 | 专用知识 | 专用工作流 |
|---|---|---|---|
| KDA | KDA-0.1 使用 FlashInfer-Bench；KDA-0.5 未公开 | KernelWiki 生产 PR 语料；NCU profiling skill；IKET kernel 内事件追踪 skill | humanize2 工作流；research–iterate–tune 三阶段提示；每步由独立 Codex 验收 |
| AVO | 参考实现正确性检查与吞吐量评分 | CUDA/PTX 文档、Blackwell 规格、FA4 源码 | 单谱系进化；只提交不退化的版本；停滞时由 supervisor 干预 |
| AKO | 编译、正确性（含 sanitizer）、计时与反作弊检查 | DSL 专用 skill（含 NCU）；AKO4X 算子陷阱库 | profile–modify–benchmark–commit 循环与停滞规则；AKO4X 使用主/子 Agent 多轮运行 |
| CAKE | Typed IR 预编译验证、代价模型排序、局部诊断、CUPTI 计时 | 将重复失败提炼为验证规则、IR primitive 和优化策略 | generate–filter–evaluate–route；先单 shape 进化，再推广为库 |
| JustGoal | — | — | — |

**笔者认为，JustGoal 最重要的定位不是“又一种 Agent”，而是一把随模型更新的尺子**：专用机制只有在同模型、同任务范围下明显越过它，才证明自己仍有净收益。

## JustGoal 怎么做：把工程闭环交给模型

### 完整输入只有目标与四条约束

Goal-only 可以理解为“只给终点和比赛规则，不发训练手册”。严格来说，它是一种控制原则：保留通用 Claude Code 的工具调用、上下文管理和长时间自主执行能力，保留目标硬件访问权，但移除所有 kernel 专用评测、知识和流程。

论文给出的完整任务输入如下：

> 这是目标算子及其配置。先在 B200 上测量参考实现的 device time；从零开始，绝不查看或照抄参考实现，写出一个 device time 至少快 1.2× 的 kernel。
>
> 1. 使用独立、干净的环境，不使用已有 skills、memory 或其他本机文件。
> 2. 确保正确性。最好先写一个缓慢的 naive FP32 实现，将它与参考实现的误差作为容差下限，再乘以一个系数，用该容差比较候选实现与 naive FP32 实现。
> 3. 不要提问，以 `--dangerously-skip-permissions` 模式运行。
> 4. 使用 B200 集群。

模型随后自己编写编译、正确性检查和计时代码：先测参考 kernel，建立要超越的目标；再写 naive FP32 实现，作为正确性 oracle（判定基准）；最后在迭代中自行决定是否 profiling、换工具链或调整搜索方向。

提示词并非完全没有人类经验：第二条包含了一种常见数值验证方法。但它只规定如何设置误差容差，不告诉模型任何 kernel 优化技巧，因此仍符合作者对 goal-only 的定义。

### 生成与最终验收必须分开

Agent 自己说“正确且更快”并不算实验结果。作者在生成结束后独立复测，只有通过验收的实现才进入表格。

- **开源库基线**：官方 FlashAttention-4、FlashInfer（含 FlashKDA）等人类维护实现；每个 shape 选择通过检查的最快候选。
- **专用 Agent 基线**：MLSys'26 竞赛提交，以及 KDA-0.5、AKO4X、CAKE、AVO 的公开 kernel；同样逐 shape 取最快可用版本。FA4 backward 没有可比的公开专用 Agent kernel。
- **正确性标准**：FA4 使用官方 FlashAttention 标准，其余算子使用 FlashInfer-Bench 标准。
- **计时方法**：统一使用 FlashInfer-Bench；每次测量前清空 L2 cache，即 cold L2；CUPTI（CUDA Profiling Tools Interface）记录入口启动的第一个 kernel 到最后一个 kernel 结束的设备时间，排除 JIT 编译、清缓存和同步开销。

所有主实验使用 Claude Code 2.1.304，模型为 Opus 4.8、Opus 5 和 Fable 5.1，reasoning effort 均为 `high`。硬件环境是 NVIDIA B200，驱动 580.105.08，CUDA 13.0；论文不采用 Agent 自报计时作为最终结果。

## 总览：8/10 超过开源库，5/9 超过公开 Agent kernel

总览实验覆盖 11 个算子、42 个 shape。每个 shape 使用一次独立 session，主要模型是 Fable 5.1，少数回退任务由其他模型完成。论文按算子报告几何平均加速，避免不同延迟量级直接做算术平均：

$$
\bar{s}_{\mathrm{geo}}=\left(\prod_{i=1}^{n}s_i\right)^{1/n}
$$

其中 $s_i$ 是第 $i$ 个 shape 相对基线的加速比，大于 1 表示更快。

![JustGoal 与公开 Kernel Agent 相对开源库的逐算子几何平均加速](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/JustGoal-With-Only-a-Goal-the-AI-Baseline-Is-Above-the-Expert-Kernels/figures/released_kernels.png)

> 图解：横轴是相对开源库的几何平均加速，虚线 $1\times$ 表示与开源库持平；纵轴列出 10 个进入正式比较的算子。蓝条是 JustGoal，橙条是每个 shape 上最快的竞赛或公开 Agent kernel，括号标注来源。JustGoal 在 GDN forward 上达到 2.34×，在 FA4 forward 和 Kimi Delta Attention 上分别为 0.77×、0.99×；FA4 backward 没有可比 Agent kernel。

### 对开源库：并非全胜，但已经越过一条能力线

排除基线协议仍未验证的 DSA top-k indexer 后，10 个算子进入比较。JustGoal 在其中 8 个上更快，加速范围从 FA4 backward 的 1.14× 到 GDN forward 的 2.34×；落后的两个是 FA4 forward 的 0.77× 和 Kimi Delta Attention 的 0.99×。

### 对公开 Agent kernel：比较的是产物，不是系统净贡献

9 个算子存在可比 Agent kernel。JustGoal 在其中 5 个上更快，MoE FP8 基本持平（1.79× 对 1.80×），落后项为：

- FA4 forward：0.77×，AVO 为 1.18×；
- DSA forward：1.66×，公开 Agent 为 1.72×；
- Kimi Delta Attention：0.99×，CAKE 为 1.27×。

这些公开 kernel 各自使用不同模型和优化预算，因此该结果只能说明 JustGoal 的产物与公开产物谁更快，不能据此断定某个 harness 没有价值。系统机制的净收益，必须由后面的同模型、同范围实验回答。

在 KDA-0.5 已发布的 9 个 DSA、GDN、MoE shape 上，公开 kernel 的几何平均加速为 1.29×，单 shape JustGoal 为 1.20×，整体接近；但发布所用模型与投入未知，不能把它当作受控对比。

### 42 个 shape 的延迟与来源

下表保留附录中的逐 shape 数据，单位为 $\mu$s，越低越好。“公开 Agent”与“竞赛 kernel”共同构成上图的 Agent 侧候选，逐 shape 取更快者。

| 算子 | Shape | JustGoal | 开源库 | 竞赛 kernel | 公开 Agent kernel |
|---|---|---:|---:|---:|---:|
| FA4 backward | MHA | 1357.9 | 1481.9（FA4） | — | — |
| FA4 backward | GQA | 1321.2 | 1675.5（FA4） | — | — |
| FA4 backward | 变长 MHA | 1581.9 | 1705.6（FA4） | — | — |
| FA4 forward | MHA (8,4096) 非因果 | 874.0 | 981（FA4） | — | 699（AVO） |
| FA4 forward | MHA (1,32768) 非因果 | 6709.6 | 5930（FA4） | — | 5270（AVO） |
| FA4 forward | MHA (8,4096) 因果 | 544.0 | 460（FA4） | — | 395（AVO） |
| FA4 forward | MHA (1,32768) 因果 | 4188.9 | 3170（FA4） | — | 2690（AVO） |
| FA4 forward | GQA g8 (8,4096) 非因果 | 2277.2 | 1558（FA4） | — | 1336（AVO） |
| FA4 forward | GQA g8 (1,32768) 非因果 | 13205.8 | — | — | — |
| FA4 forward | GQA g8 (8,4096) 因果 | 2501.4 | 960（FA4） | — | 749（AVO） |
| FA4 forward | GQA g8 (1,32768) 因果 | 6667.6 | 5750（FA4） | — | 5410（AVO） |
| FA4 forward | GQA g4 (8,4096) 非因果 | 1767.0 | 1406（FA4） | — | 1347（AVO） |
| FA4 forward | GQA g4 (1,32768) 非因果 | 15886.9 | — | — | — |
| FA4 forward | GQA g4 (8,4096) 因果 | 1003.0 | 929.5（FA4） | — | 751（AVO） |
| FA4 forward | GQA g4 (1,32768) 因果 | 8321.2 | 6030（FA4） | — | 5340（AVO） |
| DSA forward | Heavy workload | 11.42 | 13.71（FlashInfer） | 10.27（MLSys'26） | 10（KDA-0.5） |
| DSA forward | Small request | 2.891 | 7.03（FlashInfer） | 4.037（MLSys'26） | 3（KDA-0.5） |
| DSA forward | Mixed lengths | 7.115 | 11.07（FlashInfer） | 8.427（MLSys'26） | 7（KDA-0.5） |
| DSA top-k indexer | 单请求短序列 | 47.59 | 154.88（DeepGEMM + FlashInfer） | 2.35（MLSys'26） | 1.47–1.55（AKO4X，FP32 fix） |
| DSA top-k indexer | 中 batch | 3531.5 | 315.59（DeepGEMM + FlashInfer） | 17.21（MLSys'26） | 17.39–17.70（AKO4X，FP32 fix） |
| DSA top-k indexer | 大 batch 混合长度 | 13.25 | 313.28（DeepGEMM + FlashInfer） | 21.73（MLSys'26） | 24.24–24.49（AKO4X，FP32 fix） |
| GDN forward | 长序列 | 158.5 | 145.6（FlashInfer） | 209.5（MLSys'26） | 126（KDA-0.5） |
| GDN forward | 单条短序列 | 4.133 | 45.5（FlashInfer） | 5.936（MLSys'26） | 6（KDA-0.5） |
| GDN forward | 少量长序列 | 105.5 | 133（FlashInfer） | 131.6（MLSys'26） | 105（KDA-0.5） |
| GDN decode | 单请求 | 2.347 | 3.39（FlashInfer） | 2.35（MLSys'26） | 2.31（AKO4X） |
| GDN decode | 中 batch | 4.725 | 5.56（FlashInfer） | 5.27（MLSys'26） | 4.75（AKO4X） |
| GDN decode | 大 batch | 11.91 | 13.66（FlashInfer） | 12.51（MLSys'26） | 12.79（AKO4X） |
| Kimi Delta Attention | Packed 混合长度 | 740.0 | 455.6（FlashKDA） | — | 391（CAKE） |
| Kimi Delta Attention | 单条定长长序列 | 454.9 | 658.7（FlashKDA） | — | 470（CAKE） |
| Kimi Delta Attention | Packed 等长 | 506.7 | 545.1（FlashKDA） | — | 435（CAKE） |
| MLA paged decode | 单请求 | 6.320 | 15.98（FlashInfer） | — | 9.38（AKO4X） |
| MLA paged decode | 中 batch 混合长度 | 15.57 | 31.25（FlashInfer） | — | 17.36（AKO4X） |
| MLA paged decode | 大 batch | 57.58 | 77.5（FlashInfer） | — | 69.97（AKO4X） |
| MLA paged prefill | 短 prefill | 4.981 | 22.886（FA4） | — | 10.04（AKO4X） |
| MLA paged prefill | 中 prefill | 23.65 | 34.73（FA4） | — | 22.29（AKO4X） |
| MLA paged prefill | 大 prefill 混合长度 | 430.6 | 553.0（FA4） | — | 2491（AKO4X） |
| GQA paged decode | 单请求 | 4.853 | 7.488（FA4） | — | 7.74（AKO4X） |
| GQA paged decode | 中 batch 短序列 | 10.15 | 16.07（FlashInfer） | — | 11.55（AKO4X） |
| GQA paged decode | 大 batch | 60.22 | 69.61（FA4） | — | 63.47（AKO4X） |
| MoE FP8 | 单 Token | 50.09 | 86.41（FlashInfer） | 57.15（MLSys'26） | 49.05（KDA-0.5） |
| MoE FP8 | 中 batch | 296.9 | 377.41（FlashInfer） | 302.3（MLSys'26） | 278.29（KDA-0.5） |
| MoE FP8 | 大 prefill | 967.1 | 2544（FlashInfer） | 1173.6（MLSys'26） | 1042.65（KDA-0.5） |

阅读这些绝对延迟时要注意三点：

1. 因模型回退，两个 DSA top-k 任务使用 Opus 5.5；两个 MLA paged prefill 与两个 GQA paged decode 任务使用 Opus 5，其余目标模型为 Fable 5.1。
2. 四个 shape 的原始长度分布未能恢复，采用了前一轮的替代分布：GQA paged decode 中/大 batch、Kimi Delta Attention packed 混合长度、MLA paged prefill 大 prefill。
3. 不同测量批次的绝对延迟不可直接横比，每个加速比只使用自己配对批次的基线。例如 FA4 backward 基线在不同批次中，MHA 为 1371–1482 $\mu$s，GQA 为 1456–1676 $\mu$s，变长 MHA 为 1572–1706 $\mu$s。

DSA top-k 的三组候选仅作为审计记录，不进入加速图。三个历史 top-k 基线会先把 score 压成 BF16 再选择，在 18 种输入配置中的 9 种上未通过严格 FP32 membership 检查，而且基线未与候选在同一 GPU 上配对复测。

## 换模型之后，专用 harness 的相对优势会翻转

总览说明了“能做出多快的 kernel”，接下来的受控实验追问“专用 harness 本身还贡献多少”。作者选取 FA4 backward、DSA forward、GDN forward、MoE FP8 四个算子，每个三个 shape，共 12 个 shape；三种模型都使用相同的 24 小时预算。

实验包含四种配置：

- **JustGoal 单 shape**：一次 session 只优化一个 shape；3 模型 × 12 shape，共 36 次。
- **JustGoal 完整定义**：一次 session 覆盖算子 FlashInfer-Bench 定义中的全部 workload，可按 shape 分派或特化；DSA、GDN、MoE、FA4 backward 分别有 23、100、19、3 个 workload，共 12 次。
- **KDA-0.5 复现，完整定义**：根据公开仓库和技术报告重建 humanize2 `flame_chase` 双 chaser 流程、KernelWiki、NCU 报告解析、`iket-profiling` 和独立验收，共 12 次。
- **KDA-0.5 复现，单 shape**：只用 Fable 5.1 跑 12 个 shape，为后续同范围比较补齐一格。

FA4 backward 的参考实现是 FA4 库，其他三个算子的参考实现是 MLSys'26 竞赛 kernel。两次运行违反了约束，论文仍如实列出：Opus 4.8 的 JustGoal DSA heavy load 读取了参考实现；Opus 4.8 的 KDA-0.5 MoE 复现调用了 FlashInfer 库 kernel。

![三种模型下 JustGoal 与 KDA-0.5 复现的逐算子加速](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/JustGoal-With-Only-a-Goal-the-AI-Baseline-Is-Above-the-Expert-Kernels/figures/crossover.png)

> 图解：横轴先按算子分组，再列出 Opus 4.8、Opus 5、Fable 5.1；纵轴是相对参考 kernel 的逐算子几何平均加速，虚线为参考实现的 $1\times$。蓝条是单 shape JustGoal，橙条是完整定义 KDA-0.5 复现。Fable 5.1 上蓝条在四个算子都超过 $1\times$，而 GDN 的橙条仅为 0.58×；但两组任务范围不同，不能把差距直接归因于某项 harness 机制。

跨全部 12 个 shape，单 shape JustGoal 的几何平均加速从 Opus 4.8 的 0.51×、Opus 5 的 0.67×，升至 Fable 5.1 的 1.18×；同模型 KDA-0.5 完整定义复现则从 0.65×、0.87× 升至 0.97×。从 Opus 4.8 到 Fable 5.1，前者约提升 2.3×，后者约提升 1.5×，相对关系由 JustGoal 落后转为反超。

### 三种模型的逐 shape 原始结果

下表中的 JustGoal 都是单 shape，KDA-0.5 都是完整定义复现；数值为相对参考 kernel 的加速比，越高越好。最后一行是“单 shape JustGoal ÷ KDA-0.5 复现”的几何平均比值。

| 算子 | Shape | 参考实现 | Opus 4.8 JustGoal | Opus 4.8 KDA | Opus 5 JustGoal | Opus 5 KDA | Fable 5.1 JustGoal | Fable 5.1 KDA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FA4 backward | MHA | FA4 | 0.45× | 0.97× | 0.96× | 0.87× | 1.09× | 1.09× |
| FA4 backward | GQA | FA4 | 0.42× | 1.17× | 1.12× | 0.92× | 1.27× | 1.26× |
| FA4 backward | 变长 MHA | FA4 | 0.55× | 1.01× | 0.79× | 0.88× | 1.08× | 1.14× |
| DSA forward | Heavy load | 竞赛 kernel | 0.60× | 0.90× | 1.04× | 1.17× | 0.90× | 1.37× |
| DSA forward | Small request | 竞赛 kernel | 0.96× | 0.68× | 1.07× | 0.73× | 1.40× | 0.91× |
| DSA forward | Mixed lengths | 竞赛 kernel | 0.30× | 0.93× | 1.14× | 1.07× | 1.18× | 1.33× |
| GDN forward | 长序列 | 竞赛 kernel | 0.58× | 0.26× | 0.14× | 1.02× | 1.32× | 0.53× |
| GDN forward | 单条短序列 | 竞赛 kernel | 0.38× | 0.30× | 1.02× | 1.05× | 1.44× | 1.09× |
| GDN forward | 少量长序列 | 竞赛 kernel | 0.20× | 0.24× | 0.05× | 0.79× | 1.25× | 0.34× |
| MoE FP8 | 单 Token | 竞赛 kernel | 1.09× | 0.96× | 1.18× | 0.83× | 1.14× | 1.18× |
| MoE FP8 | 中 batch | 竞赛 kernel | 0.57× | 0.80× | 0.86× | 0.67× | 1.02× | 1.01× |
| MoE FP8 | 大 prefill | 竞赛 kernel | 0.65× | 0.65× | 1.12× | 0.60× | 1.21× | 1.12× |
| **单 shape JustGoal / KDA 复现** | **12 shape 几何平均** | — | **0.79×** |  | **0.78×** |  | **1.22×** |  |

再把逐 shape 数据聚合成算子级结果，可以看到模型升级并不是均匀地抬高所有配置：

| 算子 | 模型 | JustGoal 单 shape | JustGoal 完整定义 | KDA-0.5 完整定义复现 |
|---|---|---:|---:|---:|
| FA4 backward | Opus 4.8 | 0.47× | 0.50× | 1.07× |
| FA4 backward | Opus 5 | 0.95× | 0.83× | 0.96× |
| FA4 backward | Fable 5.1 | 1.14× | 0.79× | 1.18× |
| DSA forward | Opus 4.8 | 0.55× | 0.45× | 0.67× |
| DSA forward | Opus 5 | 1.08× | 0.96× | 1.01× |
| DSA forward | Fable 5.1 | 1.14× | 0.74× | 1.20× |
| GDN forward | Opus 4.8 | 0.36× | 0.13× | 0.18× |
| GDN forward | Opus 5 | 0.19× | 0.64× | 0.92× |
| GDN forward | Fable 5.1 | 1.33× | 0.69× | 0.57× |
| MoE FP8 | Opus 4.8 | 0.74× | 0.32× | 0.63× |
| MoE FP8 | Opus 5 | 1.05× | 0.43× | 0.69× |
| MoE FP8 | Fable 5.1 | 1.12× | 0.64× | 1.11× |

这组结果真正支持的是“配置差距随模型改变”，而不是“JustGoal 一定优于 KDA”。要分离模型、harness 和任务范围，还必须把最后一项也对齐。

## 同模型同范围的 2×2：harness 的价值转向任务组织

作者用 Fable 5.1 构造 $\{\text{JustGoal},\text{KDA-0.5}\}\times\{\text{单 shape},\text{完整定义}\}$ 的 2×2 对照。完整定义允许一个实现内部按 shape 分派或特化，但最终必须通过该定义的全部 workload。

| 算子 | 单 shape JustGoal | 单 shape KDA-0.5 | 完整定义 JustGoal | 完整定义 KDA-0.5 |
|---|---:|---:|---:|---:|
| FA4 backward | 1.14× | 1.28× | 0.79× | 1.18× |
| DSA forward | 1.14× | 0.94× | 0.74× | 1.20× |
| GDN forward | 1.33× | 1.13× | 0.69× | 0.57× |
| MoE FP8 | 1.12× | 1.17× | 0.64× | 1.11× |
| **12 shape 几何平均** | **1.18×** | **1.12×** | **0.71×** | **0.97×** |

结论可以拆成两半：

- **单 shape**：JustGoal 为 1.18×，KDA-0.5 为 1.12×。KDA 达到 JustGoal 的约 0.95×，完整专用 harness 没有带来整体增益。
- **完整定义**：JustGoal 降至 0.71×，KDA-0.5 为 0.97×。同范围下 KDA 相对 JustGoal 有 1.37× 优势，说明 harness 仍能帮模型推进更宽的任务。

但完整定义两格都低于各自的单 shape 结果。作者把这一现象称为 **generality tax（通用性税）**：让一个实现覆盖所有 workload，需要牺牲针对单一配置的峰值性能。一个待验证的解释是，模型尚不善于按计算特征聚类 shape，并把大任务拆成可分别优化的子问题；实验本身没有证明这一因果机制。

### 2×2 背后的 12 个 shape

| 算子 | Shape | JustGoal 单 shape | KDA 单 shape | JustGoal 完整定义 | KDA 完整定义 |
|---|---|---:|---:|---:|---:|
| FA4 backward | MHA | 1.09× | 1.35× | 0.71× | 1.14× |
| FA4 backward | GQA | 1.27× | 1.35× | 0.93× | 1.25× |
| FA4 backward | 变长 MHA | 1.08× | 1.16× | 0.75× | 1.14× |
| DSA forward | Heavy load | 0.90× | 1.05× | 0.76× | 1.36× |
| DSA forward | Small request | 1.40× | 1.03× | 0.71× | 0.94× |
| DSA forward | Mixed lengths | 1.18× | 0.78× | 0.74× | 1.35× |
| GDN forward | 长序列 | 1.32× | 0.92× | 0.70× | 0.53× |
| GDN forward | 单条短序列 | 1.44× | 1.41× | 0.90× | 1.03× |
| GDN forward | 少量长序列 | 1.25× | 1.10× | 0.53× | 0.35× |
| MoE FP8 | 单 Token | 1.14× | 1.29× | 0.97× | 1.24× |
| MoE FP8 | 中 batch | 1.02× | 0.90× | 0.64× | 0.98× |
| MoE FP8 | 大 prefill | 1.21× | 1.38× | 0.42× | 1.11× |
| **几何平均** |  | **1.18×** | **1.12×** | **0.71×** | **0.97×** |

**这个设计的聪明之处，是把“专用机制是否有效”改成了一个可证伪的问题**：先固定模型，再固定范围，最后看净差，而不是拿旧模型的收益替新模型背书。

### 三种效应如何计算

附录进一步把每个“模型 × 算子”的三个 shape 几何平均聚合成三个比值：

$$
E_{\mathrm{scope}}=\frac{S_{\mathrm{JustGoal,whole}}}{S_{\mathrm{JustGoal,single}}}
$$

$$
E_{\mathrm{harness}}=\frac{S_{\mathrm{KDA,whole}}}{S_{\mathrm{JustGoal,whole}}}
$$

$$
E_{\mathrm{vs.single}}=\frac{S_{\mathrm{KDA,whole}}}{S_{\mathrm{JustGoal,single}}}
$$

Scope effect 小于 1，表示扩大目标造成损失；harness effect 大于 1，表示完整定义范围内 KDA 更快；vs. single 则比较 KDA 完整定义与 JustGoal 单 shape 的差距。

| 算子 | Opus 4.8 scope | Opus 4.8 harness | Opus 4.8 vs. single | Opus 5 scope | Opus 5 harness | Opus 5 vs. single | Fable 5.1 scope | Fable 5.1 harness | Fable 5.1 vs. single |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FA4 backward | 1.07× | 2.13× | 2.27× | 0.87× | 1.16× | 1.01× | 0.69× | 1.49× | 1.03× |
| DSA forward | 0.81× | 1.50× | 1.21× | 0.89× | 1.05× | 0.93× | 0.65× | 1.63× | 1.05× |
| GDN forward | 0.37× | 1.41× | 0.52× | 3.32× | 1.45× | 4.81× | 0.52× | 0.83× | 0.43× |
| MoE FP8 | 0.44× | 1.97× | 0.86× | 0.41× | 1.59× | 0.66× | 0.57× | 1.73× | 0.99× |
| **几何平均** | **0.61×** | **1.72×** | **1.05×** | **1.02×** | **1.30×** | **1.32×** | **0.60×** | **1.37×** | **0.82×** |

12 个“模型 × 算子”组合中，有 11 个完整定义 JustGoal 落后于同模型 KDA 复现，而且完整定义 JustGoal 没有一个组合跨过参考 kernel。Opus 4.8 的 scope/harness 效应为 0.61×/1.72×，Fable 5.1 为 0.60×/1.37×。这些仍只是完整配置之间的差距，不能隔离某一个机制的贡献。

### 完整定义实验的逐 workload 延迟

下表单位为 $\mu$s，越低越好；每个 JustGoal/KDA 单元格依次是 Opus 4.8 / Opus 5 / Fable 5.1。数据由作者在同一 B200 上用官方 evaluator 至少两轮交替复测，采用 cold L2、CUPTI kernel span 和中位数。

| 算子 | Shape | JustGoal 完整定义 | KDA-0.5 完整定义复现 | 竞赛 kernel | 开源库 |
|---|---|---:|---:|---:|---:|
| FA4 backward | MHA | 2675 / 1702 / 1932 | 1337 / 1416 / 1200 | — | 1371 |
| FA4 backward | GQA | 2701 / 1647 / 1565 | 1238 / 1478 / 1163 | — | 1456 |
| FA4 backward | 变长 MHA | 3426 / 1985 / 2110 | 1557 / 1697 / 1378 | — | 1572 |
| DSA forward | Heavy load | 19.3 / 9.58 / 13.3 | 14.1 / 8.30 / 7.43 | 10.13 | 585 |
| DSA forward | Small request | 10.0 / 4.02 / 4.73 | 5.93 / 4.41 / 3.56 | 3.36 | 661 |
| DSA forward | Mixed lengths | 13.5 / 6.90 / 9.30 | 9.22 / 6.29 / 5.11 | 6.92 | 652 |
| GDN forward | 长序列 | 1126 / 213 / 178 | 710 / 126 / 234 | 124.9 | — |
| GDN forward | 单条短序列 | 13.96 / 5.96 / 5.36 | 15.2 / 4.74 / 4.67 | 4.81 | — |
| GDN forward | 少量长序列 | 1863 / 198 / 203 | 973 / 138 / 312 | 107.7 | — |
| MoE FP8 | 单 Token | 294 / 300 / 53.7 | 135 / 72.9 / 42.1 | 52.0 | 149 |
| MoE FP8 | 中 batch | 551 / 433 / 431 | 336 / 386 / 280 | 275.8 | 435 |
| MoE FP8 | 大 prefill | 2703 / 1390 / 2431 | 1271 / 1582 / 923 | 1028.5 | 2515 |

这里的 DSA“开源库”列是竞赛数据集的 FlashInfer baseline solution，每次调用都会拼接输入，不代表库 kernel 自身速度；MoE 使用 FlashInfer `trtllm_fp8_block_scale_moe`。Opus 5 的 MoE 完整定义方案在单 Token workload 上呈双峰，跨轮为 63–122 $\mu$s，表中取中位数。

作者还记录了 Agent 自测偏乐观的原因：GDN Opus 4.8 把毫秒计时读成微秒；MoE Opus 4.8 的 benchmark 命中了按输入指针缓存的 CUDA Graph；MoE Opus 5 则在自己生成的输入上计时。这再次说明，独立复测不是形式步骤，而是结果可信度的前提。

## 消融与正确性：工具有用，不等于专用流程有用

整体配置比较之后，下一个问题是：KernelWiki、NCU 和双 Agent 工作流，究竟哪一项真正有帮助？作者用 Fable 5.1 在三个 shape 上做单机制消融，每次只向 JustGoal 添加一项：

- **Kernel Wiki**：提示词中加入 KernelWiki 仓库链接；
- **NCU**：要求使用 Nsight Compute 做性能分析；
- **NCU skill**：在 NCU 指令上再增加报告解析 skill；
- **flame chase**：把目标前缀替换为 humanize2 的双 Agent 工作流。

| Shape | JustGoal | Kernel Wiki | NCU | NCU skill | flame chase |
|---|---:|---:|---:|---:|---:|
| FA4 backward 变长 | 1.08× | 0.91× | 1.28× | 1.19× | 0.69× |
| DSA forward heavy | 0.90× | 1.12× | 1.17× | 1.09× | 1.10× |
| GDN forward 长序列 | 1.32× | 1.14× | 1.65× | 1.48× | 1.36× |

NCU 在三个 shape 上都优于默认 JustGoal，但 NCU skill 又都低于直接使用 NCU。Kernel Wiki 和 flame chase 没有一致收益。由此更合理的判断不是“领域机制无效”，而是：**让强模型知道有什么工具可用，可能已经比规定它必须怎样使用工具更重要；每个附加层仍需单独证明价值。**

### 自建正确性检查不能完全替代硬约束

JustGoal 在生成阶段没有外部硬门禁：模型自己写检查，最终版本可能漏过容差错误。KDA 复现则在循环中运行官方 evaluator，失败版本无法提交。

硬约束下，Opus 5 和 Fable 5.1 的 12 次复现全部通过复测；Opus 4.8 的 3 个 DSA kernel 因没有返回 `lse` 而失败，当时的 evaluator 没捕获这一点。软约束下，独立复测通过率如下，分母是复测目录数，分子是发布版本通过的目录数。

| JustGoal 配置 | Opus 4.8 | Opus 5 | Fable 5.1 |
|---|---:|---:|---:|
| 单 shape，4 个主算子 | 11/12 | 9/12 | 12/12 |
| 单 shape，其余 7 个算子 | 1/3 | 4/7 | 22/26 |
| 完整定义 | 5/5 | 4/5 | 5/5 |

失败类型包括数值精度、缺失输出、构建或运行时错误。更强模型的自建检查已接近硬约束通过率，但仍不能取而代之。

完整定义复测还有若干边界案例：Opus 4.8/5 的 DSA 在 small request 上 `lse` 超出容差 1.4–1.7×，不过绝对误差都小于 $10^{-5}$；Opus 4.8 的 MoE 在大 prefill 上有 3 行专家选择整体不匹配，可能与近似并列有关，官方 evaluator 因 90% matched-ratio 规则而通过。公开仓库原样复测还发现，Opus 5 MoE 完整定义方案在 19 个 workload 上均出现 Triton 启动参数数量不匹配，预期 36 个、实际给出 27 个。

## 案例一：GDN 长序列，8.9 小时自主跨越竞赛 kernel

整体数字证明了“能做到”，优化轨迹则展示“怎样做到”。GDN forward 案例使用 20 条序列、共 8192 个 Token，单条长度从 15 到 3371。参考实现是 KDA-0.5 提交的 MLSys'26 竞赛 kernel，Agent 只能把它当黑盒调用；完整轨迹审计确认没有读取源码。

Fable 5.1 无人工干预运行 8.9 小时。参考延迟为 125.1 $\mu$s，整个过程可分成三个阶段。

1. **Triton 探索**：首个正确版本为 302 $\mu$s，仅 0.41×。扫描 tile 和 warp 后，模型发现 recurrence 有固定逐步开销、intra-chunk kernel 有寄存器溢出，估计该路径下限约 146 $\mu$s，于是停止继续微调。
2. **CUDA + `mma.sync`**：先做 B200 微基准，再手写 CUDA C++，0.5 小时达到 194.5 $\mu$s。结合 `cuobjdump`、NCU stall/source counter 和 kernel 内 cycle 计数，1.3 小时达到 165 $\mu$s；随后尝试双流重叠、cluster、TMA multicast 和静态任务分配，3.8 小时仅到 153.8 $\mu$s。
3. **切换到 tcgen05**：4.2 小时，模型判断 shared-memory bandwidth 接近上限，把 recurrence 迁到 B200 第五代 Tensor Core。它查阅 PTX ISA 文档和 CUTLASS 头文件，用 probe kernel 验证 descriptor、swizzle 和 TMEM（Tensor Memory）布局。首版反而为 235 $\mu$s；经过 warp specialization、MMA 融合和剩余 SM 重叠执行，6.2 小时以 122 $\mu$s 首次越过参考，8.5 小时达到 94.6 $\mu$s。

![JustGoal 在 GDN forward 长序列上的 8.9 小时优化轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/JustGoal-With-Only-a-Goal-the-AI-Baseline-Is-Above-the-Expert-Kernels/figures/gdn_trajectory.png)

> 图解：横轴是 session 时间，纵轴是相对参考 kernel 的加速比，虚线为 $1\times$。蓝色阶梯是“截至当前的最佳自测结果”，灰点是日志审查确认未提升最佳值的修改批次，背景色区分初始 kernel、profiling/tuning、tcgen05 重写与重叠三个阶段。图中 6.2 小时首次跨过参考，最终最佳自测为 94.6 $\mu$s。

最终版本在三个独立进程中的延迟为 95.6–97.3 $\mu$s，对应 1.29–1.31×；同 GPU 配对复测为 1.32×。输出与状态误差分别是参考误差的 1.00 和 1.30 倍。注意，轨迹图使用 session 自测协议，与主表独立计时不能直接混用。

这个案例最关键的不是某个 CUDA 技巧，而是工具链切换、硬件微基准、profiling 和搜索阶段都不是提示词规定的。KDA 将这些行为编码成 skills 与三阶段流程，JustGoal 则让模型自己长出了类似流程。

## 案例二：FA4 backward GQA，瓶颈藏在 reduction 缓冲区

第二个案例是 FA4 backward GQA：$B=8$、$N=4096$、$H=16$、$H_{\mathrm{KV}}=2$、$D=128$，causal、BF16。参考实现是 Tri Dao 等人的 FlashAttention-4，Agent 只能通过公开 API 黑盒调用。Fable 5.1 选择 Triton Gluon，4 小时无人工干预完成。

初始 kernel 为 2771 $\mu$s，首次即通过正确性检查，但比参考慢约 80%。模型用 NCU、`%clock` 和消融开关拆分开销，发现 MMA 本身只需约 830 $\mu$s，主要时间花在 softmax 与 dQ 原子累加。复用 TMEM、改用 TMA bulk reduction 后，2.5 小时达到约 1320 $\mu$s。

接下来十余个版本只改善到 1278 $\mu$s。模型又写微基准，测得 TMA reduction 每 SM 仅 25–30 B/clk，释放 staging buffer 约需 2800 cycles；禁用 dQ reduction 后，主 kernel 降至 931 $\mu$s。这些证据共同指向：reduction 复用了 dO 的 staging buffer，阻塞了 dO 加载。

最终修复是为 reduction 单独分配 32 KB staging buffer，并用 `cp.async` 把 dO 加载移到 drain warps，让 TMA 专注 reduction，单步降至 1166 $\mu$s。

![JustGoal 在 FA4 backward GQA 上的 4 小时优化轨迹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/JustGoal-With-Only-a-Goal-the-AI-Baseline-Is-Above-the-Expert-Kernels/figures/fa4_trajectory.png)

> 图解：横轴是 session 时间，纵轴是相对 FA4 的加速比，虚线为 $1\times$。蓝线记录历史最佳，灰点是其他通过正确性检查的版本。约 2.5 小时跨过 FA4，随后在 1278 $\mu$s 附近平台化；定位缓冲区冲突后，最佳自测跃迁至 1166 $\mu$s。

统一 cold-L2 配对复测中，最终 kernel 为 1321 $\mu$s，FA4 为 1676 $\mu$s，加速 1.27×。最终实现对 dQ 使用 FP16 累加、对 dS 做 BF16 rounding，内部数值路径与 FA4 不同，但 dQ、dK、dV 都通过官方梯度容差，误差只占允许预算的 20%–33%。

## 讨论与局限：专用机制的价值正在迁移

这些结果并不意味着专家知识不再重要，而是说明它的最佳落点正在变化：单目标峰值性能越来越多地由模型自主探索获得，专用 harness 的剩余价值更可能体现在宽任务组织、硬正确性约束和失败诊断上。

作者据此提出两条路径：

- **把 workload 划分能力本身作为优化目标**：用单 shape 与完整定义之间的性能差，衡量 Agent 是否学会拆解广义任务。
- **Forge Engineering（锻造工程）**：对高价值、固定配置做深度定制，每个实现只优化目标配置，但必须对该配置下的任意输入保持正确。这样不必支付 generality tax，代价是优化 session 随配置数量增长，并且每个配置都要独立验证。

更务实的研发方式，是先让新模型以 JustGoal 独立探索，再由专家分析轨迹、profiling 证据和失败案例，只针对暴露出的短板增加知识、工具或约束。每加一项，都用同模型、同范围的 JustGoal 重新测量净收益。

论文也明确给出了结论边界：

1. **硬件与任务有限**：只验证 NVIDIA B200 上的选定算子和 shape，不能直接外推到其他硬件与任务。
2. **公开产物不是受控系统对比**：公开 Agent kernel 的模型和预算不同；同模型结论只能来自受控实验。
3. **KDA-0.5 复现并不完整**：公开材料缺少完整提示词、工作流配置和搜索过程，复现可能与原系统不同；其公开 kernel 的模型与投入也未知。
4. **机制尚未完全拆清**：KDA-0.5 同时打包领域先验和范围组织机制，论文没有分别消融这两组因素，关于价值迁移的解释仍是基于数据的判断。
5. **通过测试不等于生产就绪**：有限测试输入上的容差检查，不能保证所有合法输入与执行条件都正确，报告结果只代表通过所采用验证流程后的性能。

## 总结：先测裸模型，再决定专家机制放在哪里

- JustGoal 不是新 Agent 架构，而是只保留目标、约束、通用 Agent 与硬件访问权的 goal-only 实验基线。
- 在 B200 上，它快于 8/10 个开源库算子，并快于 5/9 个具有公开可比实现的最快专用 Agent kernel。
- Fable 5.1 同范围比较中，单 shape 下 KDA-0.5 没有整体增益，完整定义下则有 1.37× 相对优势，说明 harness 价值随任务范围变化。
- NCU 本身在三个消融 shape 上均有帮助，但 NCU skill、KernelWiki 和预定义工作流没有一致增量，机制有效性必须随模型重测。
- 模型自建正确性检查已显著改善，却仍不能替代独立验收；速度结果也不能等同于生产正确性保证。

未来更值得追踪的方向，是让模型学会组织更宽的 workload，并以同模型、同范围的 goal-only 基线持续筛掉已经失效的专用层。

> 本文参考自 [JustGoal: With Only a Goal, the AI Baseline Is Above the Expert Kernels](https://arxiv.org/abs/2610.36434)