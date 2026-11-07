# SpecCTMC：评估减少4.38倍

Masked Diffusion Language Model 的精确随机采样必须逐事件调用模型，成为长文本生成的延迟瓶颈；SpecCTMC 在不改变目标分布的前提下，将文本任务的串行目标模型评估最多减少 **4.38 倍**，并在 NVIDIA B200 上取得最高 **2.88 倍**实际加速。

## 为什么需要新的并行方式？

- MDLM 可以同时预测多个 masked 位置，却不能随意同时提交：位置间的条件依赖会造成 **factorization error**，即使边缘概率正确，联合分布仍可能错误。
- DirectCTMC 每次只提交一个真实的“位置—token”事件，避免时间离散化和并行近似，却必须串行执行。
- 关键性质是：吸收 mask 过程的总离开速率只取决于剩余 mask 数。每次事件恰好减少一个 mask，因此事件时钟可以提前采样，后续纠正不会破坏时钟。

## SpecCTMC：批量计算，顺序验收

它无需训练，也不需要独立草稿模型，而是复用旧目标预测：

1. 预采样合法事件时钟，只对位置—token 事件投机。
2. 均匀选择 masked 位置，复用缓存的 token 概率，串行构造草稿轨迹。
3. 将草稿更新前的状态合批，计算各状态对应的目标分布。
4. 按原顺序验证；接受则继续，首次拒绝时从完整联合空间的残差分布抽样纠正，并丢弃后续草稿。

草稿事件 $z$ 的接受概率为 $\min\{1,P_i(z)/D_i(z)\}$；拒绝分支补上正残差 $[P_i(z)-D_i(z)]_+$。接受与纠正两部分恰好合成为目标核 $P_i$，而并行只用于计算候选状态，不改变顺序条件依赖。

因此，在共享初始状态、合法时钟、归一化目标核、顺序验证及相同截断规则等条件下，SpecCTMC 与 DirectCTMC 的事件数、时间、位置、token、中间状态和最终输出的**联合分布完全一致**。草稿质量只影响速度，不影响正确性。

## 实验：减少调用，但不保证等比例省时

- LLaDA-8B-Instruct 与 Dream-v0-Instruct-7B 覆盖 GSM8K、MATH、MBPP、HumanEval、IFEval，串行评估减少 **1.73–4.38 倍**。LLaDA 未出现统计上更慢的条件，配对准确率差异的 95% 区间均包含 0；Dream 的 GSM8K 则大多接近持平。
- 多模态 LLaDA-V 在 MathVerse、MathVista 上减少 **5.93–6.69 倍**评估，接受率为 0.698–0.806；因评测器不可用，论文未报告任务分数。
- 在 GSM8K 的同覆盖比较中，$K=8$ 时 SpecCTMC 与 256-step $\tau$-leaping 均取得 71.88% Pass@8，评估中位数分别为 724 和 2,056，前者少用约 **2.84 倍**评估。
- 多次随机采样还能扩大覆盖率：$K=32$ 时 DirectCTMC 与 SpecCTMC 的 Pass@32 均为 85.94%，评估中位数由 8,140 降至 2,901，减少 **2.81 倍**。

## 结论与边界

SpecCTMC 的核心价值，是用精确投机执行降低自然事件 CTMC 的串行深度，而非用质量换速度。但 NFE 不等于延迟：合批仍要计算全部候选，长序列、深批次、硬件和显存都会侵蚀收益。与缓存结合时，它保持的是声明后的 cached target process，并不会自动消除缓存近似。

> 本文参考自 [SpecCTMC: Exact Speculative Simulation of Diffusion Language Model Continuous-Time Markov Chains](https://arxiv.org/abs/2611.08362v1)