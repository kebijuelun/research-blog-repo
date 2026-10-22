# AKV：让 Agent 直接管理 KV Cache

长轨迹 Agent 每多跑一轮推理和工具调用，KV Cache 就持续膨胀——论文在 BCP 任务上发现 KV 读取占了 API 总成本的 82.4%，缓存管理已成为长程 Agent 落地的核心瓶颈。AKV 的答案是：不改写文本轨迹，直接操作 KV 状态。最终效果是任务平均得分提高 1.7 个百分点、成本降低 27.7%，同时带来 1.18× 输出吞吐和 1.51× P95 请求延迟加速。

## 问题：上下文工程的两难

已有 Context Engineering（摘要、裁剪、延迟压缩）都改写文本轨迹，但修改较早前缀后，保留的后缀 KV 不再与文本对应，必须重新 Prefill——修改越早重算越多，为省重算而少保留又伤准确率。实验印证了这一点：只保留最近 12 轮工具响应的文本裁剪成本是全上下文基线的 1.85×；48K 阈值摘要省了约 7% 成本却损失 5 个百分点准确率。

## 方法：三个操作把缓存与文本解耦

AKV 提出 Cache Engineering，区分"逻辑轨迹"（完整文本历史）与"Active Cache"（Attention 实际可见的 KV），用三个操作控制后者：

- **Append**：常规追加新 Token 缓存；
- **Drop**：让选中 KV 退出后续 Attention，文本与位置编码不变；
- **Repos**：用 RoPE 相对旋转 $K\leftarrow R(q)R(p)^{-1}K$ 重排保留缓存的位置，无需重算 Hidden State。

主力策略是 **Tool-response Rolling Drop**（$K=12$）：保留全部推理 KV 和最近 12 轮工具响应，滚动删除更老的工具响应。依据是 Attention 统计：越旧的内容和工具响应获得的 Attention 越少，而推理状态已吸收过早期观察的信息。

## 系统：正确复用、驱逐与恢复

由于相同文本前缀在不同操作历史下 KV 不同，AKV 在 SGLang 之上构建"Stateless Interface, Stateful Serving"：客户端用扩展 Messages API 声明完整消息与 `drop`/`reposition` 操作历史；服务端通过三项设计保证正确性：

- **操作感知 Radix Tree**：插入 Drop/Repos 节点区分操作历史，先 Exact 匹配再找可经位置变换复用的 Retrigger 前缀；
- **Drop-aware Eviction**：跟踪全部计算与恢复引用，优先回收已 Drop 缓存，同时保证请求隔离；
- **Segment-batch Attention**：在操作触发点切分 Query，单次 Forward 恢复多段历史，FFN 仍统一批处理。

## 实验：更准、更便宜、更快

- **主结果**（MiniMax-M2.7、GPT-OSS-120B、MiroThinker-1.7-Mini × BCP/GAIA-Text/SWE-Verified/xbench-DS）：AKV 在三个模型上平均得分均最高，相对 Standard 提高 1.3–2.1 个百分点；成本相对 Standard 降 27.7%，相对 Text Engineering 降 21.9%。注意并非每项都最优，结论是"平均更优"而非"全面碾压"。
- **稳定性**：摘要方法跨过阈值后准确率骤降，AKV 滚动缩减曲线更平缓（BCP 96K+ 区间 37% vs Standard 24%）。
- **服务性能**（SWE-Verified 最长 80 任务）：峰值并发下请求延迟均值/P95 加速 1.36×/1.51×，输入/输出吞吐 1.53×/1.18×；Full Attention 时间减少 59.8%，Decode Step 总时间减少 16.0%。
- **消融**：只删工具响应已大幅降本，但配合 Repos 后 BCP 通过率从 55% 升至 62%；删推理不如删旧工具响应。
- **策略进化**：Cache-As-Tool 让 Agent 自主决定缓存操作（两模型通过率均最高）；RSI 自动搜索出的策略比人工专家设计再降本 7.1%。

## 结语

AKV 的贡献不止一种删除启发式，而是把 Append/Drop/Repos 变成 Agent 可编程的高层操作，并配套支持多请求共享、恢复的系统机制。代码已开源。

> 本文参考自 [Agent-Managed KV-Cache Reduction](http://arxiv.org/abs/2610.28577v1)