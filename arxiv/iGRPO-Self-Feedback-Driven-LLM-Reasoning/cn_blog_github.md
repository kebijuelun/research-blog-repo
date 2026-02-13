# iGRPO：用“自反馈”驱动的 LLM 推理强化学习，怎么更强、更稳、更省

## 先说结论：iGRPO 做了什么、为什么有效
iGRPO（Iterative Group Relative Policy Optimization）在 GRPO 的基础上引入 **两阶段自反馈** ：  
第一阶段先采样多个草稿，选出奖励最高的“最佳草稿”；第二阶段把这个草稿拼回原始问题，让模型在“最强一次尝试”的上下文中再生成一组答案，并只对第二阶段的输出做梯度更新。

核心价值在于两点：  
1. **自适应条件化** ：训练中的“条件示例”不是固定的，而是随模型能力一起变强。  
2. **在同等采样预算下更高效** ：把同样数量的采样分成“草稿 + 精炼”，比单轮生成更能逼近正确答案。

---

## 背景：GRPO 在做什么
GRPO 是一种不用价值函数（critic）的 PPO 变体，核心步骤是：

1. 对同一 prompt 采样 $G$ 个回答  
2. 用奖励模型打分  
3. 在组内做标准化，得到优势函数  
4. 用 PPO-style clipping 更新策略  

公式上，GRPO 的奖励归一化形式是：

$$
\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_1,\dots,R_G\})}{\mathrm{std}(\{R_1,\dots,R_G\})}
$$

---

## iGRPO 的核心机制：两阶段自反馈
### 阶段 1：草稿探索
给定 prompt $q$，采样 $N$ 个草稿：  

$$
d_i \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)
$$

用奖励 $R_\phi(\cdot)$ 选出最好的草稿：  

$$
\hat{d} = \arg\max_i R_\phi(d_i)
$$

### 阶段 2：条件精炼
将草稿拼接到原问题：  

$$
q' = \text{Concat}(q, \hat{d})
$$

在 $q'$ 上再采样 $G$ 个完成，做 GRPO 更新。 **只对 Stage 2 进行梯度更新** 。

---

## 为什么这套机制更强：一个直观的理论解释
如果奖励是二值（对 / 不对），那么“草稿阶段选到正确草稿”的概率是：

$$
1 - (1 - V_\theta(q))^N
$$

其中 $V_\theta(q)$ 是单次成功率。  
这意味着： **模型越强，草稿越好；草稿越好，后续精炼越容易再提升。**  
它形成了一个“自举闭环”。

---

## 关键算法框架（简化版）
1. 采样 $N$ 个草稿  
2. 选出最优草稿  
3. 拼接形成新 prompt  
4. 采样 $G$ 个回答  
5. 只对 Stage 2 的回答做 GRPO-style 更新  

---

## 图解：iGRPO 的训练流程
![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/iGRPO-Self-Feedback-Driven-LLM-Reasoning/figures/framework.png)  
> 图解：左侧是 Stage 1 草稿探索，右侧是 Stage 2 条件精炼。最佳草稿被拼回原问题形成 $q'$，使得第二阶段学习在“最好尝试”上进一步修正。

---

## 实验结果：iGRPO 在多个模型和规模上稳定提升

### 1. 标准基准下全面提升（7B / 8B / 14B）
在 Nemotron、DeepSeek-R1 等模型上，iGRPO 的表现稳定超过 GRPO 和多种自反馈/自验证基线。

**示例：OpenMath-Nemotron-14B**  
- AIME25：64.53 → **65.57**  
- AIME24：74.79 → **76.72**  
- Avg：76.73 → **78.00**

---

### 2. 大规模训练强化：OpenMath-Nemotron-14B
![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/iGRPO-Self-Feedback-Driven-LLM-Reasoning/figures/pass_n_aime.png)  
> 图解：AIME24 在 pass@16 后趋于饱和（93.33%），AIME25 则一直提升到 pass@256（96.67%），说明更难的题对多次采样收益更大。

---

### 3. OpenReasoning-Nemotron-7B 的泛化收益
![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/iGRPO-Self-Feedback-Driven-LLM-Reasoning/figures/pass_at_1_bars.png)  
> 图解：iGRPO 不仅提升数学任务（AIME24/25），还迁移到 GPQA、MMLU-Pro 等一般推理任务，说明其“自反馈精炼”并非仅对数学特化。

---

## 训练超参与系统配置（7B 版本）
**训练设置要点**  
- 2 节点 × 8 卡 A100  
- 1 个节点专门用于 vLLM 生成  
- bf16 + FlashAttention-2  
- 学习率 $1\times 10^{-6}$  
- 温度 0.7  
- 奖励函数：accuracy + format（权重 1.0 / 1.0）

**表格：7B 训练超参**
| 参数 | 值 |
| --- | --- |
| bf16 | true |
| attn_implementation | flash_attention_2 |
| use_vllm | true |
| vllm_gpu_memory_utilization | 0.85 |
| gradient_accumulation_steps | 8 |
| gradient_checkpointing | true |
| learning_rate | 1e-06 |
| lr_scheduler_type | cosine_with_min_lr |
| lr_scheduler_kwargs | min_lr_rate: 0.1 |
| warmup_ratio | 0.1 |
| num_train_epochs | 1 |
| per_device_train_batch_size | 16 |
| max_completion_length | 4096 |
| num_generations | 4 |
| temperature | 0.7 |
| reward_funcs | accuracy, format |
| reward_weights | 1.0, 1.0 |

---

## 资源消耗：两阶段带来的代价有多大？
iGRPO 的代价主要来自多了一次生成，但实际消耗非常温和：

- Peak Memory：54.9286 GB → 54.9349 GB（几乎不变）  
- Throughput：0.41 → 0.34 samples/s  
- GPU Hours：83.3 → 94.1（约 **+13%** ）

结论： **精度显著提升，成本只增加约 13%** ，可接受。

---

## 训练过程中的动态差异
### 奖励走势与长度对比
![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/iGRPO-Self-Feedback-Driven-LLM-Reasoning/figures/reward_plot.png)  
> 图解：iGRPO 的平均奖励曲线始终更高，说明训练过程中一直更稳定地提升。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/iGRPO-Self-Feedback-Driven-LLM-Reasoning/figures/length_plot.png)  
> 图解：输出长度几乎一致，说明 iGRPO 的提升不是靠“更长输出”，而是更高质量。

---

## 消融实验：机制本身的价值
### 1. KL 系数影响很小
最佳 $\beta$ 为 0.0001，但所有设置差距不大，实践中 $\beta=0$ 更简单且几乎不掉效果。

### 2. 完成数量的影响
从 4 → 8 增益明显，16 / 32 的收益边际递减。

---

## Appendix 关键推导：iGRPO 的策略梯度
论文附录给出了从 REINFORCE 到 PPO-style clipping 的完整推导。  
核心形式可以概括为：

$$
\nabla_\theta \mathcal{J}_{\mathrm{iGRPO}}(\theta)
=
\mathbb{E}\Biggl[
\frac{1}{G}\sum_{j=1}^{G}\frac{1}{|o_j|}\sum_{t=1}^{|o_j|}
\Bigl[
\mathbb{I}_{j,t}(\theta)\,\hat A_j\,r_{j,t}(\theta)
+
\beta(\rho_{j,t}(\theta)-1)
\Bigr]\;
\nabla_\theta \log \pi_\theta(o_{j,t}\mid q',o_{j,<t})
\Biggr]
$$

其中：  
- $r_{j,t}$ 是旧策略与新策略的 token-level importance ratio  
- $\mathbb{I}_{j,t}$ 对应 PPO clipping 的“有效区间”  
- $\rho_{j,t}$ 是参考策略的 KL 正则项  

重要点： **Stage 1 不参与梯度，仅 Stage 2 参与更新** ，保证稳定性。

---

## 结论：iGRPO 的实际意义
iGRPO 不是一个复杂的新 RL 目标，而是一个 **低侵入、可复用的训练包装层** 。  
它用“最强草稿”作为动态示例，逼迫模型去 **修正自己最好的答案** ，而不是一次次从零开始。

这带来了三个直接收益：  
- 更强的推理性能（尤其是 AIME / MATH / Minerva 这类长链任务）  
- 更稳定的训练过程（延迟 entropy collapse）  
- 代价有限，工程可落地  

---

> 本文参考自 [iGRPO: Self‑Feedback–Driven LLM Reasoning](https://arxiv.org/abs/2602.09000)