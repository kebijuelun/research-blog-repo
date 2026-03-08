# Learning to Discover at Test Time：让模型在测试时继续训练，去“发现”新 SOTA

## 一句话结论

这篇工作提出了 **TTT-Discover**（Test-Time Training to Discover）：不是把 LLM 冻结后反复采样，而是让模型在同一个测试问题上边搜索边进行 Reinforcement Learning（RL）更新。目标也不是“平均表现更好”，而是“找到一个能打破现有最优解的候选”。  
在数学、GPU Kernel、算法竞赛、单细胞分析四类任务里，它都给出了很强的结果，且主要基于开源模型 `gpt-oss-120b` 完成。

---

## 这篇论文到底在解决什么问题？

传统 test-time scaling（如 Best-of-N、evolutionary search）有一个核心限制：  
- 你可以反复 prompt 一个冻结模型。  
- 但模型本体不会吸收这次任务里的新经验。  

这对“发现型任务”（discovery problem）很致命，因为这类任务通常是 OOD（分布外）问题，靠已有参数很难直接命中突破点。

作者的核心观点是：  
- 对 discovery 来说，不需要学到“普适策略”。  
- 只需要在当前问题上找到一个足够好的“单点突破”解。  

因此，它把目标从“最大化平均 reward”改成“偏向最大值、偏向突破 SOTA”。

---

## 方法总览：TTT-Discover 的最小闭环

算法循环可以概括为四步：

1. 从历史 buffer 里选一个初始状态（候选解）。  
2. 用当前策略采样新动作（代码/推理），得到新候选。  
3. 评估 reward，写回 buffer。  
4. 用这条新轨迹做一次 test-time RL 参数更新。  

这和普通 RL 的最大差异在于：目标函数与 reuse 机制都围绕“找到最好一个解”设计，而不是均值最优。

> 图解：横向是不同任务结果，核心曲线展示了训练 step 从 0 到 49 时 reward 分布如何右移。可以看到 TTT-Discover 随训练推进不断提升上尾表现（最优样本更优），并超过“冻结模型 + 同预算 Best-of-N”的上限。  
![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/vspace--1ex/figures/figure1.png)

---

## 关键设计 1：Entropic Objective（偏向极值，而非均值）

标准策略梯度优化的是期望回报，这对 discovery 不够“激进”。  
作者使用了熵风险（entropic utility）目标：

$$
J_{\beta}(\theta)=\mathbb{E}_{s \sim \text{reuse}(\mathcal{H})}\left[\log \mathbb{E}_{a\sim \pi_{\theta}(\cdot|s)}\left[e^{\beta(s)R(s,a)}\right]\right]
$$

对应梯度权重是指数重加权：

$$
\nabla_{\theta}J_{\beta}(\theta)=
\mathbb{E}_{s,a}\left[w_{\beta(s)}(a)\nabla_{\theta}\log\pi_{\theta}(a|s)\right],\quad
w_{\beta(s)}(a)=
\frac{e^{\beta(s)R(s,a)}}{\mathbb{E}_{a' \sim \pi_{\theta}}[e^{\beta(s)R(s,a')}]}
$$

直觉上：  
- $\beta$ 越大，越偏向高 reward 尾部样本。  
- 当 $\beta \to \infty$ 时，目标更接近“max”。  

但固定 $\beta$ 会不稳定：前期太大容易爆炸，后期太小又学不动。  
因此，论文使用自适应 $\beta(s)$，并通过 KL 预算约束求解：

$$
\mathrm{KL}\big(q_{\beta(s)}(\cdot|s)\,\|\,\pi_{\theta}(\cdot|s)\big)=\gamma,\quad \gamma=\ln 2
$$

这相当于“每个状态单独控制步长”，既能偏向高分样本，又不至于训练崩掉。

---

## 关键设计 2：PUCT Reuse（选起点也要为“爆款解”服务）

仅从空状态开始会导致有效 horizon 太短。  
TTT-Discover 使用 PUCT 风格的状态复用，在 exploitation 和 exploration 之间做平衡：

- 评分核心：$Q(s)+$ 探索项。  
- $Q(s)$ 不看均值，而看“从该状态扩展出的最好子结果”。  
- 高分状态有更高先验概率，但低访问状态也会被探索。  

这点非常关键：  
- evolutionary 方法往往依赖复杂手工启发式。  
- 这里给出了更统一、可解释、跨任务可复用的选择规则。  

---

## 任务设定：统一为“可验证连续 reward 的单问题环境”

论文把每个应用都表示为：  
- state：候选解（代码、步函数、算法等）。  
- action：模型输出（推理 + 代码）。  
- transition：解析或执行 action 得到新 state。  
- reward：可验证连续指标（如 $1/\text{runtime}$、竞赛得分、$1/\text{MSE}$）。  

并且将发现定义为：

$$
R(s) > R(s_{\text{sota}})
$$

即只要超过当前 SOTA 就算 discovery，且超越幅度越大价值越高。

---

## 实验结果：四个方向表现都很强

## 数学问题（Erdős / Autocorrelation / Circle Packing）

- Erdős Minimum Overlap：做到 **0.380876**，优于此前 AI 结果 **0.380924**。  
- 第一自相关不等式 AC1：做到 **1.50287**，优于 1.50314/1.50317 等先前结果。  
- 第二自相关 AC2：未超过最佳 0.9610，但达到 0.9591，接近 SOTA。  
- Circle Packing：在若干设置上追平最好已知结果。  

> 图解：该图对比了 Erdős 问题中的 step function 构造。横轴可理解为定义域位置，纵轴为分段常数高度。TTT-Discover 找到的是更高分辨率、非对称构造（600 段），而先前代表性结果段数更少、结构更对称。  
![Erdos Comparison](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/vspace--1ex/figures/erdos_comparison.png)

> 图解：该图叠加了 AC1 任务中不同方法的 step function 及其自卷积形态。关键观察是 TTT-Discover 在“接近极值约束区域”的形状控制更细，因此把上界继续压低到了更优水平。  
![AC1 Overlay](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/vspace--1ex/figures/ac1_overlay.png)

---

## GPU Kernel Engineering（TriMul / MLA）

TriMul 任务中，TTT-Discover 结果非常亮眼：  
- A100：约 **2198 μs**（人类最佳约 4531 μs）。  
- H100：约 **1161 μs**（人类最佳约 1371 μs）。  
- 其他卡型也有显著优势。  

策略上，其本质是学会了“找瓶颈 + 融合算子 + 把重 matmul 交给 cuBLAS/rocBLAS”。

---

## AtCoder Heuristic Contest（算法工程）

- AHC039：**567,062**，超过人类最佳 **566,997**。  
- AHC058：**848,414,228**，也超过公开人类最佳。  

在同等采样预算下，Best-of-25600 依然明显不够，说明“只采样不学习”的天花板确实存在。

---

## Single-cell Denoising（OpenProblems）

在 PBMC / Tabula 两个数据集上：  
- 综合 score 提升到 **0.71 / 0.73**。  
- 相比 MAGIC 等基线，在 MSE 指标上更好，同时满足 Poisson 约束。  

这说明该框架不只对代码/数学任务有效，对科学计算流程同样可迁移。

---

## 消融：两个组件缺一不可

> 图解：不同配置下 reward 分布随 step 的演化。完整 TTT-Discover 曲线右移最明显，说明其既提升均值也拉高尾部最优。去掉 test-time training 或去掉 reuse，分布提升都会明显停滞。  
![Ablations](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/vspace--1ex/figures/reward_stats_trimul.png)

论文在 TriMul 上的消融结论很清晰：  
- 自适应 entropic objective 比固定 $\beta$ 和期望 reward 更稳、更强。  
- PUCT reuse 比 no reuse、$\epsilon$-greedy 更可靠。  
- naive RL（期望回报 + 无复用）几乎退化到 Best-of-N 水平。  

---

## 我认为最有价值的三点

1. 把 discovery 问题从“平均优化”重写成“极值优化”。这不是小改动，而是目标函数层面的对齐。  
2. 把“搜索启发式”变成更通用的 reuse 机制，减少了大量任务特定手工工程。  
3. 在开源模型 + 中等预算下实现跨域 SOTA，工程可复现性比很多只展示闭源 frontier 模型的工作更强。  

---

## 局限与未来方向

论文也明确了当前边界：  
- 主要适用于连续可验证 reward。  
- 对稀疏/二值奖励或不可验证任务，尚未形成成熟方案。  

下一步方向也很自然：  
- 扩展到 sparse reward discovery。  
- 引入更稳健的奖励估计与长时 credit assignment。  
- 提升跨任务自动化调参与训练稳定性。  

---

> 本文参考自 [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)