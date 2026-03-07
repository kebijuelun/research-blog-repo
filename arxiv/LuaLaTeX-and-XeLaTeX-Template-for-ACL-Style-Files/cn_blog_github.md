# AdaptEvolve：让进化式 AI Agent“按需调用大模型”的自适应路由方法深度解读

## 背景：这篇工作在解决什么问题？

如果你在做 Evolutionary Agent（比如 OpenEvolve / AlphaEvolve 路线），一个非常现实的问题是：每一代都调用大模型（如 32B）太贵，但只用小模型（如 4B）又不够强。

这篇工作抓住了核心矛盾： **推理能力** 和 **计算成本** 的拉扯。它提出的 AdaptEvolve 本质上是一个“智能分流器”：

- 简单样本尽量交给小模型处理
- 难样本再升级到大模型
- 决策依据不是外部重路由器，而是小模型生成时自带的置信度信号（entropy/logprob 统计）

一句话概括：它不是“固定配比混用模型”，而是“按题目当下难度动态调度模型”。

---

## 方法主线：从固定进化到自适应进化

### 1）进化框架中的路由改造

传统 mutation 是统一模型 $\mathcal{M}$。AdaptEvolve 改成条件路由：

$$
x_i'=
\begin{cases}
\mathcal{M}_S(x_i), & \Phi(C(x_i))=1 \\
\mathcal{M}_L(x_i), & \text{otherwise}
\end{cases}
$$

其中：

- $\mathcal{M}_S$：小模型（4B）
- $\mathcal{M}_L$：大模型（32B）
- $C(x_i)$：由小模型生成过程提取的置信度特征
- $\Phi$：轻量二分类路由器（先静态树，再在线自适应树）

这一步的关键不是“多模型”，而是“多模型调用时机的动态化”。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/LuaLaTeX-and-XeLaTeX-Template-for-ACL-Style-Files/images/mermaid-diagram-2025-12-15-031057.png)

> 图解：这是 AdaptEvolve 的核心流程图。先用小模型生成候选，再计算 LGC/MC/TC/BWC 四类置信度特征，最后由路由器决定是否升级到大模型。横向可理解为一次迭代流程，纵向可理解为从“低成本尝试”到“高能力兜底”的逐级决策链路。

---

### 2）为什么用“内生置信度”做路由？

论文观点非常实用：外部路由器需要额外模型与训练成本，而且在进化搜索里数据分布会漂移（non-stationary），容易失配。  
相反，token 级置信度是生成时顺手可得，额外开销很低。

核心 token 置信度定义为：

$$
c_i=-\frac{1}{k}\sum_{j=1}^{k}\log P_i(j)
$$

在此基础上构造 4 个特征：

- **MC**：全局平均置信度，衡量整体稳定性
- **LGC**：最弱窗口置信度，抓“局部推理崩塌”
- **TC**：末尾窗口置信度，抓收尾阶段是否失稳
- **BWC**：底部 $K\%$ 窗口均值，区分偶发噪声与系统性幻觉

---

### 3）为什么是决策树 + 在线自适应树？

论文给了一个很实在的工程结论：置信度到“是否该升级模型”的关系是非线性的。  
例如，可能出现“全局看起来很自信，但尾部明显不稳”的组合，线性阈值难以处理。

所以做了两层：

- **静态冷启动树**：用 $N=50$ warm-up 样本训练浅层 Decision Tree（Gini, depth=5）
- **在线阶段 HAT**：Hoeffding Adaptive Tree 持续增量学习，遇到 concept drift 自动换枝/重生长

这非常契合进化式 Agent：前期探索和后期精修的数据分布本来就不一样。

---

## 实验设计与评估口径

### 1）任务与模型

- 基座框架：OpenEvolve
- 模型组合：Qwen3-4B（小）+ Qwen3-32B（大）
- 数据集：
  - LiveCodeBench v5（880）
  - MBPP（974）

### 2）成本与效率定义

论文把一次 32B 调用成本定为 $1.0$，4B 调用成本为 $0.125$。  
效率指标定义为：

$$
\text{Eff}=\frac{\text{Accuracy}}{\mathcal{C}_{\text{total}}}
$$

这比只看 Accuracy 更公平，因为它直接惩罚“堆算力”。

---

## 关键结果：不是只省钱，而是“省钱还保精度”

### 1）主结果（跨基准）

| 配置 | LiveCodeBench 成本 | LiveCodeBench Acc | LiveCodeBench Eff | MBPP 成本 | MBPP Acc | MBPP Eff |
|---|---:|---:|---:|---:|---:|---:|
| 4B Only | 0.46 | 62.3 | 135.4 | 0.37 | 80.1 | 216.5 |
| 32B Only | 3.17 | 75.2 | 23.7 | 1.18 | 94.0 | 79.7 |
| 静态树 | 2.02 | 71.2 | 35.2 | 0.46 | 82.2 | 178.7 |
| 在线 HAT（AdaptEvolve） | 2.08 | 73.6 | 35.4 | 0.69 | 91.3 | 132.3 |

可以读出三层信息：

- 在 LiveCodeBench 中，AdaptEvolve 以更低成本接近 32B 精度（73.6 vs 75.2）
- 在 MBPP 中，路由更激进地偏向小模型（85:15），成本更低且精度保持高位
- 平均上，论文报告约 **37.9%** 的推理成本下降，同时保留约 **97.5%** 的上界精度

### 2）对比 Cascading

| 配置 | Ratio (S:L) | Cost | Accuracy | Eff. |
|---|---|---:|---:|---:|
| Cascading Baseline | 36:64 | 2.81 | 73.8 | 26.3 |
| AdaptEvolve（HAT） | 42:58 | 2.08 | 73.6 | 35.4 |

这组结果很有说服力：Accuracy 基本持平，但效率显著提升。说明路由不是“少用大模型就好”，而是“在正确的样本上用大模型”。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/LuaLaTeX-and-XeLaTeX-Template-for-ACL-Style-Files/images/adapt-1.jpeg)

> 图解：横轴是 32B 调用占比，纵轴是最终 Accuracy。红色 AdaptEvolve 点位更靠近“高精度、低大模型占比”的区域，说明它在 Pareto 前沿上优于随机与级联策略。

---

## 附录里最值得看的两件事

### 1）统计可分性：置信度特征确实能分开“解得出/解不出”

论文给出的统计显示，Solved 样本在 LGC/MC/TC/BWC 的均值普遍更高；Unsolved 的偏度更重，说明“高不确定性尾部”更明显。这为树模型学习决策边界提供了可解释基础。

### 2）非线性路由优于线性阈值

Pilot（$N=50$）显示决策树在成本更低时得到更高精度，优于 z-score 阈值切换，印证“置信度签名”不是单阈值可分问题。

---

## 复现视角：这套方法为什么工程上可落地？

从实现角度看，AdaptEvolve 的优点在于：

- 不需要额外大路由模型
- 只依赖生成时可读取的 logprob/entropy
- 可先离线 warm-up，再在线持续自适应
- 与现有进化 Agent 兼容性高（主要改 mutation 路由层）

关键前提也很明确：如果目标闭源 API 不暴露 token 级概率，方法会受限。

---

## 总结

AdaptEvolve 的核心价值，不是再造一个更强 Agent，而是把“算力预算”作为一等公民写进进化过程。  
它证明了：在 Evolutionary Agent 里， **不确定性感知 + 在线路由自适应** 可以在几乎不掉精度的情况下显著降本，并把系统推到更好的质量-成本 Pareto 前沿。

> 本文参考自 [arXiv:2602.11931](https://arxiv.org/abs/2602.11931)