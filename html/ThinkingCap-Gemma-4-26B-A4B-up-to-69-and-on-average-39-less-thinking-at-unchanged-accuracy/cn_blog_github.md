# ThinkingCap-Gemma-4-26B-A4B：平均少想 39%，准确率几乎不动

推理模型有个越来越明显的成本问题：为了答一道题，模型往往要先生成上万 token 的思考过程。这些 thinking token 不直接出现在答案里，却实打实地占用延迟和算力。BottleCap AI 的 ThinkingCap 系列就是冲着这个问题来的——把模型的“内心戏”压短，同时不动答案质量。2026 年 10 月 23 日发布的这篇是该系列的第五个模型，对象是 Google 的 gemma-4-26B-A4B-it：平均减少 38.9% 的 thinking token，最高降幅 69.1%，而十二个基准上的平均准确率只掉了 0.15 个百分点。

![ThinkingCap Gemma-4-26B-A4B](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/ThinkingCap-Gemma-4-26B-A4B-up-to-69-and-on-average-39-less-thinking-at-unchanged-accuracy/images/press-thinkingcap-gemma-4-26b.jpg)

## 第五款 ThinkingCap，第一次落在 Google 模型上

在此之前，团队已经发布过四款 token 高效的 ThinkingCap 模型，分别基于 Qwen3.6-27B、Qwen3.8-27B、NVIDIA Nemotron-3.5-Lightning-30B-A3B 和 Qwen3.8-Flash-Next。这次是他们第一次对 Google 的模型动手，也是继 Qwen、NVIDIA 之后覆盖的第三个模型家族。

gemma-4-26B-A4B 是 Google 的稀疏混合专家模型：总参数 26B，每个 token 实际激活约 4B。团队表示在这个模型上格外注意保住准确率——平均 thinking token 降 38.9%，代价只有 0.15pp 的准确率，这个差值比大多数基准自身的运行间波动还小，也是全系列到目前为止最小的精度代价。原文还附了一段对比视频：同一个输入，原始模型和 ThinkingCap 模型在相同硬件、相同推理设置下运行，可以直观看到思考长度的差别。

## 训练目标：只砍思考，别的不动

做法的出发点很克制。以 gemma-4-26B-A4B-it 为起点，目标只有一个：用更少的 token 走到答案，其他一切保持不变——知识、推理能力、答案质量、指令遵循和安全行为都不许动。换句话说，这不是一个“更快但略笨”的折中版模型，而是专门削减冗余思考长度的微调。

## 结果：准确率纹丝不动，思考普遍变短

先看准确率。按任务家族分组对比，两个模型几乎没有差别：

![各任务家族准确率对比：gemma-4-26B-A4B-it 与 ThinkingCap 模型](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/ThinkingCap-Gemma-4-26B-A4B-up-to-69-and-on-average-39-less-thinking-at-unchanged-accuracy/images/accuracy.svg)

> 图解：按任务家族分组的准确率对比，深色为原始模型，浅色为 ThinkingCap 模型。所有家族的差距都在 1 个百分点以内，其中指令遵循与智能体（+0.65pp）、数学（+0.39pp）两个家族不降反升，代码基本持平（0.02pp），知识与推理 −0.51pp，长上下文与多模态 −0.92pp。

再看 thinking token，变化全在这里，而且每个家族都在缩短：

![各任务家族平均 thinking token 对比：gemma-4-26B-A4B-it 与 ThinkingCap 模型](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/ThinkingCap-Gemma-4-26B-A4B-up-to-69-and-on-average-39-less-thinking-at-unchanged-accuracy/images/mean-thinking-tokens.svg)

> 图解：按任务家族分组的平均 thinking token 数对比。长上下文与多模态砍得最多（−54.4%），其次是知识与推理（−44.4%）、数学（−36.1%）、代码（−27.6%），指令遵循与智能体最少（−17.2%）。十二个基准平均 −38.9%。

完整的十二个基准数据如下，每一行都是原始 BF16 模型与 ThinkingCap 模型在同一套评测管线、同一批问题、同一组采样设置下的对比：

| 基准 | 准确率（原始） | 准确率（ThinkingCap） | Δ 准确率 | 平均 thinking token | Δ 平均思考 | 截断率 | 循环率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **数学** | | | | | | | |
| AIME 2026 | 86.20% ±0.86 | 86.72% ±0.98 | +0.52pp | 14766 → 8963 | −39.3% | 5.2 → 0.5% | 5.5 → 0.9% |
| HMMT Nov 2025 | 81.51% ±1.06 | 80.89% ±1.19 | −0.62pp | 16462 → 10542 | −36.0% | 9.0 → 1.2% | 8.6 → 1.2% |
| HMMT Feb 2026 | 72.92% ±1.08 | 74.20% ±1.16 | +1.28pp | 22005 → 14717 | −33.1% | 18.6 → 4.6% | 17.0 → 4.5% |
| **代码** | | | | | | | |
| LiveCodeBench | 76.98% ±0.56 | 76.96% ±0.67 | −0.02pp | 11560 → 8375 | −27.6% | 6.8 → 1.3% | 2.2 → 0.3% |
| **知识与推理** | | | | | | | |
| GPQA-Diamond | 79.92% ±0.57 | 78.96% ±0.67 | −0.96pp | 7699 → 4284 | −44.4% | 0.3 → 0.1% | 0.3 → 0.1% |
| MMLU-Pro | 84.55% ±0.29 | 84.64% ±0.16 | +0.09pp | 3907 → 1989 | −49.1% | 0.0 → 0.0% | 0.1 → 0.0% |
| MMMLU | 84.09% ±0.26 | 83.43% ±0.27 | −0.66pp | 1212 → 731 | −39.7% | 0.0 → 0.0% | 0.0 → 0.0% |
| **长上下文与多模态** | | | | | | | |
| AA-LCR | 64.62% ±2.49 | 60.62% ±2.89 | −4.00pp | 6519 → 2655 | −59.3% | 4.0 → 0.0% | 3.6 → 0.4% |
| LongBench v2 | 59.43% ±1.14 | 60.88% ±1.29 | +1.45pp | 9486 → 2934 | −69.1% | 7.7 → 0.5% | 5.9 → 1.2% |
| RealWorldQA | 70.83% ±0.56 | 70.61% ±0.56 | −0.21pp | 819 → 533 | −34.9% | 0.0 → 0.0% | 0.0 → 0.0% |
| **指令遵循与智能体** | | | | | | | |
| IFBench | 76.83% ±0.80 | 78.67% ±0.73 | +1.83pp | 6470 → 5585 | −13.7% | 0.4 → 0.4% | 0.5 → 0.5% |
| τ²-bench | 64.25% ±0.97 | 63.73% ±1.26 | −0.53pp | 4374 → 3466 | −20.8% | 0.0 → 0.0% | 0.0 → 0.0% |
| **平均（12 个基准）** | 75.18% | 75.03% | −0.15pp | 8773 → 5398 | −38.9% | 4.35 → 0.72% | 3.64 → 0.76% |

表格里几处口径值得说明。准确率是多种子的均值，±号后是种子间波动给出的 95% 置信区间。最后一行的“Δ 平均思考 −38.9%”是十二个降幅的平均，不是旁边两个 token 数相除——后者的比值是 38.5%，团队特意区分了这两种算法。另外 τ²-bench 的 thinking token 按任务统计，即一次多轮 rollout 各回合的总和，而不是按单条回复算。

## 逐行看：砍得最深和最浅的地方

十二个基准无一例外都变短了，但幅度差异不小。砍得最深的是 LongBench v2（−69.1%）、AA-LCR（−59.3%）和 MMLU-Pro（−49.1%）；最浅的是 IFBench（−13.7%）和 τ²-bench（−20.8%）。团队对这两条“浅”行的解释很直白：这两类任务本来就不太依赖长篇思考——一个是照着格式约束清单答题，另一个是按固定的客服政策走正确的工具调用，能省的冗余本来就少。

准确率这边，大多数变化在噪声范围内：十二个里有八个的差距小于运行间波动，五个跑赢了原始模型。真正超出置信区间的只有四处：IFBench（+1.83pp，对自己有利）、GPQA-Diamond（−0.96pp）、MMMLU（−0.66pp）和 AA-LCR（−4.00pp）。团队承认 AA-LCR 是整套测试里唯一实打实的代价。

还有一个容易被忽略的改进：循环（looping）。原始模型有 3.64% 的回复会陷入重复循环，ThinkingCap 模型只有 0.76%。在循环最严重的三个数学基准上——AIME 2026、HMMT Nov 2025、HMMT Feb 2026——循环率分别从 5.5%、8.6%、17.0% 降到 0.9%、1.2%、4.5%。

## 预算视角：同样的 token，换到更多准确率

单点对比之外，团队还给了一条更有部署意义的曲线：给定每题的 thinking token 预算，两个模型各自能交出多少准确率。

![准确率随 thinking token 预算变化的曲线：两个模型对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/ThinkingCap-Gemma-4-26B-A4B-up-to-69-and-on-average-39-less-thinking-at-unchanged-accuracy/images/budget-curve.svg)

> 图解：横轴为每题允许的 thinking token 预算，纵轴为准确率。从左往右读，正是部署时真正关心的问题——预算固定时，ThinkingCap 的曲线在整个中段都压在原始模型上方，即同样的准确率只需更小的预算；当预算放宽到两个模型都不受约束时，两条曲线汇合。

这条曲线把“少想 39%”翻译成了工程语言：如果你的服务按 token 计费或有延迟上限，省下的预算就是直接的吞吐和成本。

## 评测方法：同一条管线，跑两遍

推理模型的评测噪声很大。在官方推荐的 temperature 1.0 下，回复的内容和长度每次都变，单个种子说明不了什么。所以这套评测的每个基准都跑了多个种子，准确率报均值和 95% 置信区间。关键配置如下：

- **管线**：团队内部评测套件，两个模型都跑在 vLLM 上，原始模型由团队实测，而非引用官方公布数字。
- **上下文**：每条回复的生成上限为 65536 token，τ²-bench 按回合计。
- **采样**：思考开启，temperature 1.0、top_p 0.95、top_k 64——ThinkingCap 原样继承 Google 的推荐预设，保证对比里没有任何解码层面的变量。
- **种子数**：AIME 2026 和两项 HMMT 各 64 个，GPQA-Diamond 与 LiveCodeBench 各 32 个，RealWorldQA、IFBench、τ²-bench 各 16 个，AA-LCR 与 LongBench v2 各 8 个，MMLU-Pro 与 MMMLU 各 4 个（题目量大，题目采样不是主要噪声源）。
- **题量**：AIME 2026 共 30 题，HMMT Nov 2025 共 30 题，HMMT Feb 2026 共 33 题，LiveCodeBench 175 题，GPQA-Diamond 198 题，MMLU-Pro 12032 题，MMMLU 9996 题，AA-LCR 100 题，LongBench v2 370 题，RealWorldQA 765 题，IFBench 300 题，τ²-bench 含航空、零售、电信三个领域共 278 个任务。
- **打分**：除三项外均为提取答案后精确匹配——LiveCodeBench 运行生成的代码对测试用例，IFBench 走程序化校验器，τ²-bench 和 AA-LCR 由裁判模型评分。τ²-bench 还会对模拟用户跑完整对话，最终报三个领域的等权平均。

表中“截断率”指思考过程没有正常收尾、被判零分的回复比例；“循环率”由一个基于压缩率的循环检测器统计——如果一段文本的压缩率远超普通散文，就被标记为循环。

## 模型获取与 PRO 版

模型已发布在 Hugging Face：`bottlecapai/ThinkingCap-gemma-4-26B-A4B-it`，同时提供 GGUF 构建。采样设置和服务栈都不需要改，属于直接替换式的升级。团队还提到一个 PRO 版本：thinking token 最多可再压到 −87.6%，平均 −62%，面向对推理成本更敏感的规模化部署，需要邮件联系（enterprise@bottlecapai.com）获取。

## 写在最后

这篇发布回答了一个很实际的问题：推理模型的思考能不能只剪长度、不伤脑子？就这份自测数据看，答案是基本可以——平均省 38.9% 的 thinking token，精度代价 0.15pp 低于多数基准的噪声，还顺手把循环率从 3.64% 压到 0.76%。适合关注的人很直接：正在用 gemma-4-26B-A4B 做推理服务、对延迟或 token 成本敏感的团队，替换成本几乎为零。

限制也要说清楚：AA-LCR 上 −4.00pp 的掉分是真实存在的，长上下文任务用之前值得自己复测；全部评测来自模型作者一方的内部管线，虽然方法描述得相当透明，但终归缺少第三方复现。另外官方只描述了训练目标，没有展开训练方法的细节，想照着复现的读者暂时拿不到配方。

> 本文参考自 [ThinkingCap: Gemma-4-26B-A4B — up to 69% and on average 39% less thinking, at unchanged accuracy](https://bottlecapai.com/post/thinkingcap-gemma-4-26b/)