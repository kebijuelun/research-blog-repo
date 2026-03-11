# FlashAttention-4 深度解读：为 Blackwell 时代重写 Attention 内核

## 一、先说结论：这篇工作为什么重要

这篇论文的核心不是“再做一个更快的 Attention”，而是抓住了一个硬件代际变化：在 NVIDIA Blackwell（B200/GB200）上，Tensor Core 计算速度大幅提升，真正的瓶颈反而变成了 shared memory 带宽和 exponential 单元（MUFU）吞吐。

因此，作者做的不是单点优化，而是 **算法与内核流水线协同设计（co-design）** ：
- 前向中重排 pipeline，尽量让 softmax 与 matmul 重叠；
- 用多项式近似分担 exp 计算压力；
- 反向中使用 TMEM + 2-CTA MMA，降低 shared memory 流量和原子加次数；
- 同时给出 deterministic backward，兼顾可复现训练。

在 B200 上，论文报告 BF16 场景下：相对 cuDNN 9.13 最多约 $1.3\times$，相对 Triton 最多约 $2.7\times$，峰值达到 1613 TFLOPS（约理论峰值的 71%）。

---

## 二、问题背景：为什么 FA3 思路在 Blackwell 不够用了？

Attention 基本计算为：

$$
S = \alpha QK^\top,\quad P = \mathrm{softmax}(S),\quad O = PV,\quad \alpha=\frac{1}{\sqrt d}
$$

反向（单头）为：

$$
dV=P^\top dO,\quad dP=dOV^\top,\quad dS=\mathrm{dsoftmax}(dP),\quad dQ=\alpha dSK,\quad dK=\alpha dS^\top Q
$$

FA3 在 Hopper 上已具备很强性能，但 Blackwell 带来了新条件：
- Tensor Core 吞吐翻倍（FP16/BF16 约从 1 PFLOPS 提升到 2.25 PFLOPS）；
- MUFU 与 SMEM 增长幅度未同步；
- 新增 TMEM（每 SM 256KB）和 fully asynchronous MMA；
- 支持 2-CTA Tensor Core 模式。

这意味着，如果仍按旧 pipeline 组织，MMA 会等待 softmax 与 shared memory，导致算力空转。

---

## 三、方法主线：FA4 到底做了什么

### 3.1 前向：围绕异步 MMA 重排流水线

FA4 在前向采用更激进的 ping-pong 式重叠：让一个 tile 做 MMA 时，另一个 tile 做 softmax；并利用 Blackwell 的 TMEM 存放中间结果，缓解寄存器压力。

![Figure: FA4 Forward Pipeline](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/FA4_FWD_p3.png)

> 图解：这张图展示前向流水并行。横向可理解为时间推进，纵向是不同 warpgroup/阶段。核心思想是把 $QK^\top$、softmax、$PV$ 交错起来，尽量让 Tensor Core 与非 matmul 单元同时忙碌，而不是串行等待。

---

### 3.2 Forward 的两个“非 matmul”关键优化

#### 1）部分软件模拟指数函数（exp emulation）

论文将 $2^x$ 拆分为整数部分与小数部分：

$$
2^x = 2^{\lfloor x \rfloor}\cdot 2^{x-\lfloor x \rfloor}
$$

其中小数部分使用多项式逼近：

$$
2^{x_{\mathrm{frac}}}\approx \sum_{i=0}^{n} p_i x_{\mathrm{frac}}^i,\quad x_{\mathrm{frac}}\in[0,1)
$$

直观上，这是把部分 exp 计算从 MUFU 转移到 FMA/ALU，缓解 MUFU 瓶颈。论文还强调仅对部分元素启用 emulation（约 10%–25%），以避免寄存器压力过大导致 spill。

#### 2）Conditional softmax rescaling（跳过不必要重标定）

标准 online softmax 为：

$$
m_j=\max(m_{j-1},\mathrm{rowmax}(S_j)),\quad
\ell_j=e^{m_{j-1}-m_j}\ell_{j-1}+\mathrm{rowsum}(e^{S_j-m_j})
$$

FA4 采用阈值化处理：仅当 $m_j-m_{j-1}>\tau$ 时才触发 rescale（典型 $\tau=8$）。这减少了 elementwise 操作频率，同时最终归一化仍保证数值正确。

---

### 3.3 反向：2-CTA + TMEM 是性能突破口

反向包含 5 个 MMA，shared memory 流量压力更大。FA4 的核心动作包括：
- 用 TMEM 存放更多中间量，减少 SMEM 往返；
- 采用 2-CTA MMA，让两个 CTA 共同完成更大 tile；
- 在 $dQ$ 路径上重排数据与调度，减少全局 atomic add；
- 给出 deterministic 模式（基于 semaphore 序列化归约）以保证可复现。

从论文数据看，2-CTA 设计不仅降低了 SMEM 压力，还能将部分原子归约次数近似减半，这对长序列场景尤为关键。

---

## 四、Roofline 视角：瓶颈迁移被量化了

### 4.1 前向（示例配置）

| 资源 | $128^3$ cycles | $256\times128^2$ cycles |
|---|---:|---:|
| MMA compute | 1024 | 2048 |
| Shared memory | 768 | 1536 |
| Exponential unit | 1024 | 2048 |

结论：前向中 MMA 与 exp 常并列为主瓶颈，因此“只优化 matmul”并不足够，必须同步优化 softmax/exp 路径。

### 4.2 反向（$M=N=d=128$ vs 2-CTA）

| 资源 | 1-CTA ($M=128$) | 2-CTA ($M=256$) |
|---|---:|---:|
| MMA compute | 2560 | 2560 |
| Total shared memory | 3328 | 2688 |
| Exponential unit | 1024 | 1024 |

结论：反向主要受限于 SMEM，2-CTA 可以直接压低该主瓶颈，因此收益更稳定。

---

## 五、实验结果：速度、稳定性、可用性

![Forward non-causal](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/fa4_fwd_causalFalse_hdim128_updated.png)

> 图解：横轴是 sequence length，纵轴是 TFLOPS。非因果场景下，FA4 在中长序列上整体领先，体现了 pipeline 与非 matmul 优化在高负载下的收益。

![Forward causal](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/fa4_fwd_causalTrue_hdim128_updated.png)

> 图解：同样是 TFLOPS 对比，因果掩码下 FA4 的优势更明显，与论文中的 LPT 调度策略（优先处理长任务）一致。

![Forward (192,128) head](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/fa4_fwd_causalTrue_hdim192128_updated.png)

> 图解：这是更贴近 DeepSeek V3 风格的 $(192,128)$ 头维配置。FA4 在这类“非标准头维”上仍保持较高效率，说明框架泛化性较好。

![Backward non-causal](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/fa4_bwd_causalFalse_hdim128_updated.png)

> 图解：反向非因果场景中，FA4 在长序列下持续领先，符合 2-CTA 降低 SMEM 与 atomic 压力的设计预期。

![Backward causal](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/fa4_bwd_causalTrue_hdim128_updated.png)

> 图解：反向因果场景同样受益，说明调度与流水重排并非仅在单一工作负载下有效。

![Deterministic backward ablation](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/causal_bwd_det_FA4.png)

> 图解：这张消融图对比了不同调度策略下 deterministic backward 的表现。SPT/LPT 等顺序设计明显优于 naive，说明“可复现”不必然等于“极慢”。

论文附录还给出了 non-causal deterministic 的结果：

![Deterministic backward non-causal](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/faf-Algorithm-and-Kernel-Pipelining-Co-Design-for-Asymmetric-Hardware-Scaling/Figures/non-causal_bwd_det_FA4.png)

> 图解：非因果场景下 deterministic 版本仍保留较好的吞吐，说明锁与归约顺序带来的额外开销可被调度策略部分抵消。

---

## 六、工程实现：为什么 CuTe-DSL 是加分项

FA4 全部使用 Python 嵌入式 CuTe-DSL 实现，而非 CUDA C++ 模板。论文给出的单 kernel 编译时间大致为：
- FA3：前向 55s，反向 45s；
- FA4：前向 2.5s，反向 1.4s。

即约 20–30 倍编译加速。对内核开发者而言，这意味着调参、试错和迭代速度显著提升，研发门槛也随之下降。

---

## 七、这篇论文的真正贡献（博主视角总结）

- **贡献 1** ：清晰刻画了 Blackwell 的“非对称硬件扩展”问题：Tensor Core 更快，不等于内核自动更快。
- **贡献 2** ：前向通过 exp emulation + conditional rescale，将 softmax 路径从瓶颈中释放出来。
- **贡献 3** ：反向通过 TMEM 与 2-CTA 协同，直接命中 SMEM/atomic 的根因瓶颈。
- **贡献 4** ：给出 deterministic backward 的高性能实现，兼顾训练可复现性。
- **贡献 5** ：CuTe-DSL 的工程路线证明了“底层可控 + 高迭代效率”可以同时成立。

这篇工作的关键价值不在某一个技巧，而在完整方法论：先用 roofline 定位真实瓶颈，再根据硬件代际特征重写算法与 pipeline。

> 本文参考自 [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)