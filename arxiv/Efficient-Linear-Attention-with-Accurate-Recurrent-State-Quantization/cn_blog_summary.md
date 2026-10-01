# LeapQuant：给线性注意力的循环状态做 8-bit 量化，精度与 FP32 完全持平

线性注意力用固定大小的循环状态替代了随长度膨胀的 KV Cache，但每生成一个 token 都要把状态完整读写一遍，推理吞吐被 HBM 带宽卡死。LeapQuant 提出免训练、免校准的循环状态量化方案，把状态压到 8-bit：在 Qwen、Kimi、GLM 三个家族的 12 组模型-任务对上，精度与 FP32 完全持平（平均 75.5% vs 75.5%），状态显存流量降低 3.4 倍，kernel 加速 2.05–3.70 倍，端到端推理加速 1.47 倍。

## 为什么难：误差会循环累积

- 线性注意力的状态 $S_t$ 每步做 rank-one 更新（$S_t = \mathrm{Diag}(\alpha_t) S_{t-1} + k_t u_t^\top$），带宽瓶颈在状态读写而非计算。
- naive 的 per-step 量化下，量化误差被递推不断携带、相干叠加；状态中的 outlier 又撑大量化 scale，放大每次的单次误差。
- 代价触目惊心：Qwen3.5-9B 的 AIME 得分在 per-step FP8 下从 87.9% 崩到 14.6%；连 per-step BF16 都掉到 72.1%。

## 三板斧

- **Per-window 量化**：不再每 token 量化，而是"跳"过一个 16-token 窗口——固定量化边界状态，窗口内更新以高精度缓冲、即时组合出输出，只在窗口末尾量化一次。量化频率降为 1/16，64K 上下文下状态误差降低约两个数量级（BF16 降 100 倍、INT8 降 39 倍）。工程上无需物化中间状态，开销仅在 buffer 读写。
- **Compensator Tokens**：量化前用幂迭代拟合 rank-one 项逼近状态的主导 outlier 结构，从状态中减掉、只量化残差。补偿项形式与真实 token 的 rank-one 更新完全一致，直接喂进 decode kernel，零 kernel 改动。实际取 $r = 4$，开销被显存读取完全掩盖（$r \le 8$ 时 kernel 开销不超过 7%）。
- **Residual Smoothing**：按 key 行平均绝对值的平方根对残差做可逆缩放，平滑行间幅值再量化。状态动态范围从中位数的 176 倍逐级压到 47 倍、再到 5.3 倍。

## 结果：精度持平，速度显存双赢

- **精度**：8-bit 下平均 75.5% 与 FP32 持平；6-bit 仍有 72.4%（最强基线仅 29.6–31.9%）；4-bit 保持 60.4%，基线几乎全军覆没（9.7–22.9%）。从 KV Cache 量化改造来的 KVQuant、QuaRot、TurboQuant 普遍水土不服。
- **速度**：单层 kernel 在 B200 / RTX PRO 6000 / RTX 5090 上最高加速 2.68 / 3.95 / 4.25 倍（batch 512）；vLLM 端到端吞吐提升 1.23–1.65 倍。
- **显存**：每状态元素仅 1.19 字节（vs FP32 的 4 字节）；prefix caching 下端到端显存最多省 56%（Kimi-Linear-48B），单卡并发请求数提升 1.4 倍。
- **消融**：INT8 per-step 把 AIME 打到 7.1%，per-window 拉回 82.4%，补偿 token 到 86.6%，smoothing 补齐到 87.9%——每个组件既涨精度又不显著牺牲效率（仍保留 2.52 倍 kernel 加速）。

方法对"对角衰减 + rank-one 更新"这一大类 recurrence 通用，覆盖 Gated DeltaNet、KDA、Mamba2 等主流架构；局限在于更复杂状态转移的适配成本，以及与投机解码、prefix caching 等系统机制的深度协同仍待挖掘。

> 本文参考自 [LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization](http://arxiv.org/abs/2609.38166v1)