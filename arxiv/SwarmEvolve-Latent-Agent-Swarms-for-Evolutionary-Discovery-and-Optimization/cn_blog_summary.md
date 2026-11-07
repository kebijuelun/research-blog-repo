# SwarmEvolve：评估吞吐最高13倍

程序搜索的瓶颈常常不是评估候选解，而是 LLM 反复生成和读取长文本推理。SwarmEvolve 让大量小模型 Agent 在连续潜空间中思考并直接交换内部表示，在 21 个数学发现与科学计算问题上，将每小时程序评估量提升至基线的 **5–13 倍**；达到其他框架运行 60 分钟的最佳分数，中位时间仅需 **3–8 分钟**。

## 核心创新：思考、通信与去重都不依赖文本

- **Latent Reasoning**：将最后隐藏状态映射回输入 Embedding，连续执行 40 步推理，全程不解码文本；每步只向 KV Cache 增加一个位置。
- **Latent Communication**：Summarizer 将多个 Agent 的 KV Cache 与自身 Prompt Cache 拼接，直接读取推理轨迹，再解码出完整程序。借助 RoPE 的相对位置性质，Cache 移动位置时只需旋转 Key，Value 不变。
- **Novelty Filter**：对 40 步隐藏状态取均值，以一个接近主流方向的 Agent 为锚点，再用最远点选择保留不同思路。每个 Swarm 从 45 个 Agent 筛至 30 个，几乎不需要额外模型调用。

## 一轮搜索如何组织 450 个 Agent

每轮批量运行 **10 个 Swarm、共 450 个 Agent**：4 个精炼强解，3 个混合前十名程序的思想，2 个探索 Archive 中的多样路径，1 个从初始程序重新出发。Agent 的 Prompt 由搜索关注点与九种思考方式组合，同一 Swarm 共享方向，但读取不同父程序。

筛选后，Summarizer 一次融合 30 条推理轨迹与多个父程序，写出新程序。系统另有参数调节 Agent，每轮最多评估 8 次单参数修改；候选经去重、语法检查和确定性评分后进入 Archive。利用与探索因此固定发生在同一轮，而非停滞后才重启。

## 结果：平均分领先，优势主要来自早期搜索

实验统一采用每题 60 分钟、单 GPU、8 个 CPU 核心和 3 个 Seed。SwarmEvolve 使用 Qwen3 8B、14B、32B-FP8 时，平均分分别为 **0.804、0.837、0.869**，均为同模型下最高。其 8B 成绩已超过三个基线使用 32B-FP8 时的最高分 0.774；运行 15 分钟后，也超过各模型基线的 60 分钟平均分。

冻结的 Qwen3 8B 还在三项构造上超过 AlphaEvolve，其中 26 圆正方形 Packing 的已验证半径和达到 **2.635983**。但它并非逐题必胜：Kissing Number、Eigenvalues 及部分 Packing、PDE 题仍由基线领先。

消融进一步支持“系统效率”而非“潜空间单点推理必然更强”的解释：40 步 Latent Reasoning 每轮约 4 秒，256 Token 文本推理约 32 秒；基线把 57%–87% 的 GPU 时间用于文本推理，SwarmEvolve 的潜空间推理与通信合计约 11%，从而将 66% 的时间用于写程序。更大的 Swarm 主要改善早期探索，后期因同一模型共享知识而趋于饱和。

## 边界

直接拼接 KV Cache 要求 Agent 与 Summarizer 使用同一模型；Linear Attention、State-Space 等架构不能照搬该机制；Cache 显存还会随 Agent 数、推理步数和层数线性增长。跨模型通信、Cache 压缩与非标准 Attention 状态合并，是后续扩展的关键。

> 本文参考自 [SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization](http://arxiv.org/abs/2611.08483v1)