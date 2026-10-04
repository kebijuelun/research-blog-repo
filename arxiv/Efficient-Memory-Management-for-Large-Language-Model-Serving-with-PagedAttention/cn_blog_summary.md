# PagedAttention：用分页思想驯服 LLM 推理显存

LLM 推理的瓶颈不在算力，而在显存：传统系统的 KV Cache 有效利用率仅 20.4% - 38.2%，Batch Size 被碎片和预留空间卡死。这篇 SOSP'23 论文把操作系统的分页思想搬进 Attention，构建了 vLLM——延迟持平下吞吐提升 2-4 倍（最高 22 倍）。

## 问题：显存浪费在哪

- Decode 阶段是 Memory-Bound，只能靠大 Batch 摊薄开销，而 Batch 上限由 KV Cache 显存决定（13B 模型单 token 就占 800 KB）。
- 现有框架要求张量连续存储，按最大长度预分配，产生预留、内部碎片、外部碎片三种浪费。
- 连续存储还让 Parallel Sampling、Beam Search 中可共享的前缀 KV Cache 无法共享。

## 方法：分页 + 共享

- **PagedAttention**：KV Cache 切成固定大小的 Block，Attention 改写为分块形式，Kernel 按 Block Table 间接寻址——Block 物理上可以不连续。
- **逻辑块/物理块分离**：按需分配、用完即释放，每条请求的浪费压到一个 Block 以内，碎片近乎消除。
- **Block 级共享**：引用计数 + Copy-on-Write，支持共享 Prompt、Beam Search 动态前缀、System Prompt 预缓存，Beam Search 最高省 66.3% 显存；不同解码算法可混入同一 Batch。
- **抢占恢复**：All-or-Nothing 驱逐 + Swapping / Recomputation，是用 LLM 语义对 OS 分页的"魔改"。

## 结果：数字说话

- 可承受请求速率是 Orca (Oracle) 的 1.7-2.7 倍、Orca (Max) 的 2.7-8 倍；OPT-13B 并发请求数达 Orca (Max) 的 4.3 倍。
- 共享前缀场景吞吐最高达 Orca 的 3.58 倍。
- 代价仅是 Attention Kernel 20% - 26% 的额外开销。

## 结语

分页之所以对 LLM Serving 有效，是因为它既需要动态内存分配、又被显存容量卡住咽喉——好思想要用在对的负载上。vLLM 已成为事实上的推理引擎标准之一，"把 OS 经典思想引入 AI 系统"也成为后续系统工作的范式。

> 本文参考自 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)