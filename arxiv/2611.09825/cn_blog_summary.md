# BitDecoding-FP4：原生 FP4 解码，吞吐达 3 倍

长上下文解码同时受 KV cache 容量与显存带宽限制，BitDecoding-FP4 通过量化算法与 GPU kernel 协同设计，让 GQA Attention 的两次矩阵乘原生运行在 FP4 Tensor Core 上，兼顾精度、速度与并发能力。其 NVFP4 版本在 Qwen3 两个模型上的平均准确率与 BF16 KV 相差不足 0.5 个百分点，B300 上 64K–128K 服务的峰值每 GPU 吞吐达到 BF16 KV 的 **2.4–3.0 倍**。

## 三处误差修正，守住 FP4 精度

- **Dual-Residual Query Quantization**：第一次量化后，再量化剩余残差，两项均保留独立 block scales 并走 Tensor Core。普通 NVFP4 Query 曾贡献 60% 的 logit 误差，该方案将 Query 误差占比降至约 1%。
- **Underflow-Aware Probability Scaling**：把常数 $c=8$ 折进指数项，并对小概率块改用 $\max/2$ 缩放，避免 E4M3 scale 下溢导致整块概率归零；下溢边界较标准规则降低 24 倍，尤其改善长上下文检索。
- **Outlier-Aware Key Quantization**：依次使用静态 Key 平滑、保持内积不变的 Hadamard 旋转，以及两个候选值中的 block scale 搜索，在 clipping 与 rounding 误差之间择优，压制持续出现的异常通道。

## 三项系统配合，避免修正拖慢解码

- **Transposed Softmax**：改沿 Key 维度分配线程，以 butterfly shuffle 并行归约，并融合 $P$ 的 FP4 量化，消除逐行串行扫描瓶颈。
- **Stacked Query MMA**：将残差 Query 放入首个 MMA tile 的零填充区域，每个 Key tile 只发一次 FP4 MMA，再在 Softmax 前合并两组分数。
- **混合精度 vLLM 集成**：FP4 pages 与 BF16 sink、最近 tokens 共用 decode 管线，元数据留在设备端，无需 host synchronization，并兼容 CUDA Graph。完整精度修正的额外 kernel 开销控制在 4% 以内，decode 元数据开销降至 95 μs。

## 实验结论：不仅更快，也能装下更多上下文

- **精度**：Qwen3-4B 与 Qwen3-30B-A3B 的 NVFP4 平均分分别为 67.27、72.74，仅比 BF16 KV 低 0.42、0.41 个百分点；RL rollout 全程也紧贴 BF16 曲线。
- **速度**：128K decode kernel 在 B300 上最高加速 2.99×，在 RTX PRO 5000 上最高达 **4.01×**；256K 下两个模型的 TPOT 分别加速 1.81×、1.53×。
- **容量与泛化**：355B GLM-4.7 的 200K KV cache 从 71.88 GiB 降至 20.27 GiB，实现 **3.5×** 压缩；770B MLA 模型的 1M cache 则压缩 2.6×，三项任务与 BF16 的差距均不超过 0.12 个百分点。

当前方案主要面向 NVIDIA Blackwell；MLA 的 context reduction 仍需反量化，Hy4-preview 的结果也主要证明容量收益而非延迟收益。

> 本文参考自 [BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm–System Co-Design](http://arxiv.org/abs/2611.09825v1)