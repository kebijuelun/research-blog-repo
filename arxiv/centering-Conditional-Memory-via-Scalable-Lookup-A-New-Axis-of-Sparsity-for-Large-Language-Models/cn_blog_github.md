# Conditional Memory via Scalable Lookup：给大模型新增一条“记忆稀疏”轴

## 这篇论文在解决什么问题
当前 MoE 通过 ** 条件计算 ** 扩大模型容量，但 Transformer 缺少“知识查表”的原生能力，导致模型用计算去模拟检索，效率低下。作者提出 ** 条件记忆 ** 作为与 MoE 互补的稀疏轴：让静态知识走检索，动态推理走计算。

## 核心贡献一览
- 提出 Engram：将经典 $N$-gram 嵌入现代化，做到 $O(1)$ 可扩展查表。
- 给出 ** Sparsity Allocation ** （稀疏分配）问题，发现 MoE 与 Engram 的最优配比呈 U 型规律。
- 在 27B 规模上，Engram 在等参数、等 FLOPs 下显著超过 MoE。
- 机制分析显示 Engram 让网络 ** “有效变深” ** ，并释放注意力容量，显著提升长上下文能力。
- 系统层面可把超大表放在 CPU 内存，通过确定性索引 ** 预取 ** ，几乎无吞吐损失。

## 方法总览：Engram 是怎么工作的
![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/engram_arch.png)
> 图解：Engram 在特定层插入，先做 $N$-gram 检索，再用上下文门控融合到隐状态，和主干残差相加。

### 1) 稀疏检索：哈希 $N$-gram
- 通过 tokenizer 压缩把等价 token 合并，提高语义密度。
- 对每个 $N$-gram 进行多头哈希，落到多个 embedding 表。
- 拼接所有检索到的向量得到记忆表示 $\mathbf{e}_t$。

核心公式：
$$
z_{t,n,k} = \phi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]
$$
$$
\mathbf{e}_t = \mathop{\Vert}_{n=2}^{N} \mathop{\Vert}_{k=1}^{K} \mathbf{e}_{t,n,k}
$$

### 2) 上下文门控融合
Engram 不直接把记忆加进去，而是让当前上下文决定是否启用：

$$
\alpha_t = \sigma\left( \frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}} \right)
$$

再经过短卷积提升非线性：

$$
\mathbf{Y} = \text{SiLU}\left( \text{Conv1D}( \text{RMSNorm}(\tilde{\mathbf{V}}) ) \right) + \tilde{\mathbf{V}}
$$

### 3) 与多分支架构融合
在多分支 mHC 中共享 $\mathbf{W}_V$，分支独立 $\mathbf{W}_K^{(m)}$，既保持多样性又可融合成单次矩阵乘。

### 4) 系统效率与可扩展性
![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/engram_system.pdf)
> 图解：训练期在 GPU 间分片并 All-to-All 拉取；推理期用确定性索引提前预取 CPU 内存表项，通信与计算重叠。
> 注意：图片路径 `figs/engram_system.pdf` 可能缺失。

---

## 稀疏分配定律：MoE 与 Engram 怎么配最优
![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/scaling_law.png)
> 图解：左侧是 allocation ratio $\rho$ 与验证损失的 U 型曲线；右侧是 Engram 在“大记忆”下的对数线性 scaling。

定义三个量：
- $P_{\mathrm{tot}}$：总参数
- $P_{\mathrm{act}}$：每 token 激活参数（对应 FLOPs）
- $P_{\mathrm{sparse}} = P_{\mathrm{tot}} - P_{\mathrm{act}}$：稀疏预算

分配公式：
$$
P_{\mathrm{MoE}}^{(\mathrm{sparse})} = \rho P_{\mathrm{sparse}}, \quad
P_{\mathrm{Engram}} = (1-\rho) P_{\mathrm{sparse}}
$$

关键结论：
- 最优 $\rho$ 约在 $75\% \sim 80\%$，即 ** MoE 为主、Engram 为辅 ** 。
- $\rho \to 100\%$ 会缺少知识查表，$\rho \to 0\%$ 会缺少动态推理，两端都差。
- 在“无限记忆”设定下，Engram 规模继续增大仍能稳定下降损失。

---

## 大规模预训练结果（27B / 40B）
作者在 262B tokens 上训练 4 个模型：Dense-4B / MoE-27B / Engram-27B / Engram-40B。

下面是核心任务的代表性对比（Acc 或 EM）：

| 任务 | MoE-27B | Engram-27B | 增益 |
| --- | --- | --- | --- |
| MMLU | 57.4 | 60.4 | +3.0 |
| CMMLU | 57.9 | 61.9 | +4.0 |
| BBH | 50.9 | 55.9 | +5.0 |
| ARC-Challenge | 70.1 | 73.8 | +3.7 |
| HumanEval | 37.8 | 40.8 | +3.0 |
| MATH | 28.3 | 30.7 | +2.4 |

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/benchmark_curve.png)
> 图解：最后 10k 步训练曲线，Engram 在损失和稳定性上持续领先。

结论非常稳定：Engram 在知识类、推理类、代码数学类全线胜出，且 Engram-40B 还有进一步提升空间。

---

## 长上下文能力：释放注意力后优势更明显
长上下文训练在 32k context 上评测 LongPPL 与 RULER。

核心对比（Iso-Loss 设置）：
- Multi-Query NIAH：$97.0$ vs $84.2$
- Variable Tracking：$87.2$ vs $77.0$

这说明 Engram 把局部模式交给查表后，注意力资源更集中到全局依赖上。

---

## 机制分析：为什么 Engram 有效

### 1) 有效深度提升
![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/logitlens_and_cka.png)
> 图解：LogitLens 早期 KL 更低，CKA 显示 Engram 浅层对应 MoE 深层，说明“有效深度”变大。

关键公式（CKA）：
$$
\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K)\text{HSIC}(L, L)}}
$$

### 2) 结构消融与层敏感性
![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/arch_ablation.png)
> 图解：单层插入最优在 Layer 2，多层插入（2 和 6）更稳；去掉多分支融合或门控损失明显。

### 3) 模块敏感性
![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/sensitivity_analysis.png)
> 图解：去掉 Engram 后，事实性任务大幅崩溃，而阅读理解保留大多性能，说明事实知识主要由 Engram 承担。

### 4) 门控可视化
![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/centering-Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/figs/visualization.png)
> 图解：门控对固定短语或命名实体触发明显，例如 “Alexander the Great” 或中文成语，证明查表是有效触发的。

---

## 系统效率：巨表也能跑得快
作者在 nano-vLLM 上测试 100B Engram 表完全 CPU offload：

| 模型 | 基线 tok/s | 加 100B Engram tok/s | 降幅 |
| --- | --- | --- | --- |
| 4B Dense | 9031.62 | 8858.28 | 1.9% |
| 8B Dense | 6315.52 | 6140.02 | 2.8% |

关键结论：确定性索引 + 预取机制让通信被计算隐藏，规模化记忆几乎不影响吞吐。

---

## 相关工作脉络
- $N$-gram 复兴：FastText、SuperBPE、OverEncoding、SCONE 等证明大表有效，但缺少严格等算力对比。
- MoE 系列：Switch、GLaM、DeepSeek-MoE 等把计算稀疏做到极致。
- Memory Network：PKM、PEER、SelfMem 等用 KV 稀疏表扩容。
- 机制研究：FFN 被视为 Key-Value 记忆，知识神经元与模型编辑方法支持这一假设。

Engram 的差异点在于：把 ** 记忆查表 ** 作为系统级一等公民，而不是附加技巧。

---

## 总结与展望
Engram 提供了一个非常清晰的新轴： ** 条件记忆 ** 。它不像 MoE 依赖动态路由，而是用确定性哈希查表处理静态模式，让模型把计算预算留给真正需要推理的部分。实验上，它在多域全面获益，机制上解释清晰，系统上可扩展性强。对于下一代稀疏模型设计，Engram 基本已经给出了一条可落地的路线。

> 本文参考自 [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://www.arxiv.org/abs/2601.07372)