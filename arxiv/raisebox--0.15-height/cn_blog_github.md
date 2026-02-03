# Kimi K2.5：面向通用 Agent 的多模态协同与并行编排

## 这篇报告在解决什么问题
大模型正在迈向 **Agentic Intelligence** ，但瓶颈很明确：  
- **多模态冲突** ：视觉与文本联合训练常出现相互干扰。  
- **顺序执行过慢** ：复杂任务需要大量工具调用，单 Agent 串行导致延迟爆炸。  

Kimi K2.5 的核心答案只有两句：  
1. 让文本与视觉从训练到 RL 都 **协同优化** 。  
2. 用 **Agent Swarm** 把复杂任务拆成并行子任务。  

---

## 模型总览：K2.5 的整体架构与表现
K2.5 在 K2 的 MoE 基座上加入原生视觉编码器与多模态训练机制，并在 Agent 框架中引入并行编排。

![Kimi K2.5 Main Results](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/raisebox--0.15-height/figures/k25-main-result.png)  
> 图解：整体性能结果图，展示 K2.5 在多领域任务（语言、视觉、Agent）上的综合优势与 SOTA 位置。

---

## 核心一：文本-视觉联合优化（Joint Optimization）

### 关键设计：早期融合 + 低比例视觉
传统多模态常在后期注入视觉，K2.5 发现 **早期融合 + 较低视觉比例** 更优，原因是早期让模型形成统一的多模态表示，避免后期强行对齐引发的语言退化。

![Vision-Text Joint Training Curves](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/raisebox--0.15-height/figures/vision-joint.png)  
> 图解：不同视觉比例下训练曲线对比。早期低比例融合更稳定，文本能力不出现明显“掉落再恢复”。  
> 注意：图片缺失 `figures/vision-joint.png`。

### MoonViT-3D：图像与视频共享同一编码器
K2.5 使用 **MoonViT-3D** ，核心点是：  
- 图像与视频完全共享参数  
- 4 帧为一组进行时间维压缩  
- 通过 $4\times$ 时间压缩在同样上下文窗口内处理更长视频  

![Multi-Agent RL System](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/raisebox--0.15-height/figures/multi-agent-rl-system.png)  
> 图解：模型整体训练流程和多 Agent RL 架构示意，强调多模态训练与 RL 的协同。

---

## 核心二：Zero-Vision SFT + 联合 RL

### Zero-Vision SFT：只用文本 SFT 也能激活视觉能力
K2.5 的关键发现：  
- 只用文本 SFT，也能触发视觉推理与工具调用  
- 强行加入人工视觉轨迹反而伤泛化  

这意味着：多模态能力可以通过 **文本任务迁移激活** 。

![Vision RL Curves](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/raisebox--0.15-height/figures/k25_visionzerorl_curves.png)  
> 图解：从 zero-vision SFT 起步，随着 RL 训练，视觉任务性能持续上升。

---

### 联合 RL 目标函数（核心公式）
K2.5 的 RL 采用 token 级裁剪策略，核心目标函数如下：

$$
L_{\mathrm{RL}}(\theta) = \mathbb{E}_{x \sim\mathcal{D}}
\left[ \frac{1}{N} \sum_{j=1}^K \sum_{i=1}^{|y_j|}
\mathrm{Clip}
\left(
\frac{\pi_{\theta}(y_j^i | x, y_j^{0:i})}{\pi_{\mathrm{old}}(y_j^i | x, y_j^{0:i}) }, \alpha, \beta
\right)
({r}(x, y_j) - \bar{r}(x)) - \tau \left( \log \frac{\pi_{\theta}(y_j^i | x, y_j^{0:i})}{\pi_{\mathrm{old}}(y_j^i | x, y_j^{0:i}) } \right)^2
\right]
$$

这里的核心思想：  
- 仅保留 log-ratio 在 $[\alpha, \beta]$ 区间的梯度  
- 避免 off-policy 偏移导致的大模型 RL 崩溃  

---

### 视觉 RL 反而提升文本能力
这点非常关键：视觉 RL 不仅没有损害文本，还提升了文本基准分数。  

| Benchmark | Before | After | Improvement |
|---|---|---|---|
| MMLU-Pro | 84.7 | 86.4 | +1.7 |
| GPQA-Diamond | 84.3 | 86.4 | +2.1 |
| LongBench v2 | 56.7 | 58.9 | +2.2 |

---

## 核心三：Agent Swarm 并行编排

### 为什么需要 Agent Swarm
单 Agent 的顺序执行会带来两个问题：  
- 推理深度被推理步数限制  
- 推理延迟随任务复杂度线性增长  

Agent Swarm 通过 **多 Agent 并行 + RL 协调** 打破线性瓶颈。

---

### Agent Swarm 架构
![PARL Orchestration](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/raisebox--0.15-height/figures/multi-agent-rl-system.png)  
> 图解：可训练 Orchestrator 动态创建子 Agent，子 Agent 冻结，仅优化主调度策略。

### PARL 奖励定义

$$
r_{\mathrm{PARL}}(x, y) = \lambda_1 \cdot r_{\text{parallel}}
+ \lambda_2 \cdot r_{\text{finish}}
+ r_{\text{perf}}(x, y)
$$

- $r_{\text{parallel}}$：防止只用单 Agent  
- $r_{\text{finish}}$：避免无意义的“空并行”  
- $r_{\text{perf}}$：最终任务质量  

---

### 关键执行时间指标：Critical Steps

$$
\text{CriticalSteps}
= \sum_{t=1}^{T} \left( S_{\mathrm{main}}^{(t)} + \max_i S_{\mathrm{sub}, i}^{(t)} \right)
$$

直观理解：  
并行任务的时间成本取决于最慢的子任务，而不是所有子任务总和。  

---

### 效率收益
![Agent Swarm Efficiency](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/raisebox--0.15-height/figures/agent_swarm_efficiency.png)  
> 图解：在 WideSearch 中，Agent Swarm 将达到同等性能所需时间缩短 $3\times \sim 4.5\times$。

---

## Token Efficient RL：Toggle 策略
为避免“短链推理过拟合”，K2.5 引入 Toggle：  

$$
\tilde{r}(x, y) =
\begin{cases}
r(x, y) \cdot \mathbb{I}\left\{ \frac{1}{K} \sum_{i=1}^K r(x, y_i) < \lambda\ \mathrm{or}\ |y_i| \leq \mathrm{budget}(x) \right\} & \text{Phase 0} \\
r(x, y) & \text{Phase 1}
\end{cases}
$$

核心直觉：  
- Phase 0 强化预算约束  
- Phase 1 允许长推理  
- 避免模型陷入“只会短推理”  

![Token Efficient RL Radar](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/raisebox--0.15-height/figures/te-k2-thinking-radar.png)  
> 图解：Toggle 在多基准下减少 token 使用量，同时性能几乎不下降。

---

## 训练与基础设施
K2.5 使用 H800 集群训练，并引入 **Decoupled Encoder Process (DEP)** ：  
- 视觉编码器独立前向  
- 主干 transformer 正常训练  
- 视觉后向单独重算  

这解决了多模态训练中 Stage-0 负载不均的问题。

---

## 评测结果概览
K2.5 在多个维度达到 SOTA：  
- **Reasoning** ：AIME 2025 96.1%，HMMT 95.4%  
- **Coding** ：SWE-Bench Verified 76.8%  
- **Agentic** ：BrowseComp 60.6%，Agent Swarm 78.4%  
- **Vision** ：MMMU-Pro 78.5%，OCRBench 92.3%  
- **Video** ：LongVideoBench 79.8%  

---

## 结论与启示
K2.5 的价值不仅在于模型规模，而在于：  
- 证明 **文本与视觉联合优化可相互增强**  
- 证明 **并行 Agent 可以同时提升性能与效率**  
- 提供一个可扩展的 Agentic Intelligence 训练范式  

这意味着未来真正的“通用 Agent”很可能不是一个超大模型，而是 **一个协同系统** 。

---

> 本文参考自 [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)