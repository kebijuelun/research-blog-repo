# Memex(RL)：用“可索引经验记忆”扩展长程 LLM Agent 的上下文极限

## 这篇论文在解决什么问题？

长程（Long-Horizon）Agent 的核心矛盾很直接：

- 任务步数越来越多，工具调用日志、观察结果、中间推理会不断堆积。
- LLM 的 context window 是固定的，迟早会爆。
- 传统做法要么截断历史，要么做摘要压缩，但这两种方式都容易丢失关键证据。

这篇工作提出的关键思想是： **压缩上下文，但不丢证据** 。  
具体做法是把“工作记忆”和“完整证据”分离：

- 上下文里只保留一个短小的可执行摘要（indexed summary）。
- 完整历史证据放到外部存储（key-value experience store）。
- 需要时通过稳定索引（index）精确取回，而不是做模糊相似度检索。

## 方法总览：Memex 的 Indexed Experience Memory

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/sysrlt-Scaling-Long-Horizon-LLM-Agents-via-Indexed-Experience-Memory/figures/intro.png)

> 图解：这张图展示了 Memex 的主循环。左侧是不断增长的工具轨迹，经过 `CompressExperience` 后，工作上下文被重写成“摘要 + 索引映射”；右侧的外部数据库保存原始高保真内容。后续如果需要某段历史，Agent 通过 `ReadExperience(index)` 精确回填，避免重复调用工具或重新探索环境。

论文把记忆接口定义成两部分：

1. in-context 的 `IndexedSummary = (s, I)`
  - $s$：可执行的当前进度状态（已验证事实、计划）。
  - $I$：索引映射集合，每项是 `(index, description)`。
2. off-context 的外部存储 $\mathcal{D}: index \mapsto content$

核心操作有两个：

- `CompressExperience(summary, blocks)`：把 block 写入 $\mathcal{D}$，然后把工作上下文重写为短摘要。
- `ReadExperience(index)`：按索引取回原始内容并追加回上下文。

这套机制的本质是把“上下文 token”变成“指针系统”：

- 平时维护短状态；
- 证据按需解引用（dereference）；
- 减少无意义的重复工具调用。

## 为什么仅靠 Prompt 规则不够？MemexRL 的必要性

论文强调一个很实用的点：  
“何时压缩、存什么、怎么命名索引、何时读取”是延迟反馈问题。  
一个看似合理的压缩策略，可能在十几步后才暴露灾难性后果（找不到关键 ID、重复查同一资源、上下文超阈值）。

所以作者用 RL 联合学习 task-solving 和 memory-behavior，提出 MemexRL（GRPO 风格）。

### 奖励函数设计

$$
R = R_{\text{task}} - P_{\text{context}} - P_{\text{redundancy}} - P_{\text{format}}
$$

- $P_{\text{context}}$：上下文溢出处罚，鼓励提前压缩。
- $P_{\text{redundancy}}$：重复工具调用惩罚，鼓励“读记忆”而非“再执行”。
- $P_{\text{format}}$：工具调用格式错误惩罚，保证可执行性。

### 组相对优势与 PPO 风格优化

$$
A^{(g)}=\frac{R^{(g)}-\mathrm{mean}(\{R^{(h)}\})}{\mathrm{std}(\{R^{(h)}\})+\epsilon}
$$

$$
\max_{\theta}\frac{1}{G}\sum_{g=1}^{G}\sum_t
\min\!\Big(r_t^{(g)}A^{(g)},\ \mathrm{clip}(r_t^{(g)},1-\eta,1+\eta)A^{(g)}\Big)
-\beta\,\mathrm{KL}(\pi_{\theta}\|\pi_{\text{ref}})
$$

这意味着：模型不仅学会“解题”，还学会“像工程师一样管理工作记忆”。

## 训练细节里最关键的两个设计

### 1) Segmented Trajectory Processing（按压缩边界切段）

一条轨迹中每发生一次压缩，就形成新的 segment。  
因为压缩会改变后续生成的 conditioning prefix，切段训练更符合 autoregressive 模型本质。  
但同一原始轨迹的所有 segment 共享终局奖励 $R$，这样早期压缩决策也能获得延迟信用分配。

### 2) Soft Triggering（软触发压缩）

不是超过阈值就系统强制压缩，而是每步注入 context status，让 Agent 自主决策。  
这把“压缩时机”从硬规则变成可学习技能。

## 理论部分：这套记忆机制为什么可能成立？

论文给了两个命题，核心结论非常清晰。

### 命题 1：有限次解引用下可保持最优决策质量

如果 indexed summary 对决策是 $B$-bounded sufficient（每步最多读 $B$ 个块就够），那么 Memex 策略可达到与 full-history 最优策略同等的期望回报。

直观理解：

- 不需要把完整历史塞进 prompt；
- 只要摘要 + 少量精确证据就能恢复最优决策。

### 命题 2：工作上下文上界可控

若摘要长度 $\le \tau_\sigma$，每步读取块数 $\le B$，单块长度 $\le L$，则：

$$
C_t^{\mathrm{work}} \le \tau_\sigma + BL
$$

这说明即使 full history 持续增长，工作上下文仍可保持常数上界，压缩比会持续增大。  
对长程 Agent 来说，这是“可扩展性”最核心的理论抓手。

## 实验：改造版 ALFWorld 上的效果

作者使用 Qwen3-30B-A3B-Thinking-2507，在更难的 ALFWorld 设定中评估：

- 隐藏 admissible commands；
- 隐藏初始 room 描述；
- `look` 仅允许一次；
- `CompressExperience` 生成的 summary 截断到 300 tokens。

这些改造强迫模型必须依赖“外部可索引记忆”，而不是把 ID 直接堆在上下文里。

![Rollout Success](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/sysrlt-Scaling-Long-Horizon-LLM-Agents-via-Indexed-Experience-Memory/figures/base_success_rate.png)

> 图解：横轴是训练进程，纵轴是 rollout 成功率。成功率大约从 20% 提升到 90%+，说明模型确实学到了稳定可用的记忆读写策略，而不只是提示词技巧。

![Penalty](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/sysrlt-Scaling-Long-Horizon-LLM-Agents-via-Indexed-Experience-Memory/figures/penalty_total_mean.png)

> 图解：总惩罚（越接近 0 越好）从约 -0.4 改善到约 -0.1，表示上下文管理和工具调用行为显著优化。

![Eval](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/sysrlt-Scaling-Long-Horizon-LLM-Agents-via-Indexed-Experience-Memory/figures/bar_success_rate.png)

![Eval Context](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/sysrlt-Scaling-Long-Horizon-LLM-Agents-via-Indexed-Experience-Memory/figures/bar_peak_context.png)

> 图解：最终评测显示，任务成功率从 24.22% 提升到 85.61%；峰值工作上下文从 16934 降到 9634 tokens。也就是约 3.5 倍成功率提升，同时上下文峰值下降约 43%。

![Memory Usage](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/sysrlt-Scaling-Long-Horizon-LLM-Agents-via-Indexed-Experience-Memory/figures/eval_compress_count.png)

![Memory Usage 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/sysrlt-Scaling-Long-Horizon-LLM-Agents-via-Indexed-Experience-Memory/figures/eval_retrieve_count.png)

> 图解：训练后 `CompressExperience` 次数从约 6.5 降到 3，但 `ReadExperience` 从约 1 升到 6~7。说明策略从“频繁重写上下文”转向“建立可复用索引并按需读取”，这正是 Memex 的设计目标。

## 我的解读：这篇工作的真正创新点

1. 把 memory 从“摘要文本”升级为“摘要 + 可解引用索引”。
2. 把 memory 行为纳入 RL action space，而不是当成静态工程规则。
3. 通过分段训练 + 共享终局奖励，处理了长程记忆决策的延迟信用分配。
4. 理论与实验形成闭环：既说明“为什么可能有效”，也展示“确实能够学出来”。

## 结论

Memex(RL) 给长程 Agent 提供了一条很实用的扩展路径：  
不是一味追求更大的 context window，而是让模型学会“把上下文当工作台、把证据放档案库、用索引精确回读”。  
在工具密集、步骤长、证据回访频繁的任务里，这种范式比纯摘要压缩更稳、更可审计，也更接近人类处理复杂任务的方式。

> 本文参考自 [Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory](https://arxiv.org/abs/2603.04257)