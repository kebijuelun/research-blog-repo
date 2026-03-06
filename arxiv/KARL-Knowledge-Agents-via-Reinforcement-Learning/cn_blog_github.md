# KARL：用强化学习训练知识型 Agent，如何在企业检索任务上实现高质量与低成本双赢

## 一、这篇工作到底在解决什么问题？

如果你做过企业级 AI 搜索/问答系统，很快会遇到一个现实问题：  
模型并不是“知道一切”，而是需要在 **外部知识库** 里一边检索一边推理，最后给出可验证答案。

这类任务的难点不在单轮问答，而在 **Grounded Reasoning（基于外部证据的推理）** ：

- 需要多步检索（multi-step retrieval）
- 需要跨文档整合（cross-document synthesis）
- 需要在长上下文里做判断、压缩、再推理
- 常常是“难验证”的开放式答案，而非简单标准答案

这篇论文提出的 KARL（Knowledge Agents via Reinforcement Learning）核心目标是：  
把“会搜索、会整理、会推理、会取舍”的能力，直接通过 RL 训练进 Agent。

---

## 二、核心贡献概览（先看结论版）

论文的四个关键贡献可概括为：

- 构建多能力评测套件 **KARLBench**
- 提出两阶段 **Agentic Synthesis** 数据生成流水线
- 提出迭代式大批量离策略 RL 方法（OAPL）
- 证明多任务 RL + Test-time Compute 可带来更强泛化和更优成本/时延曲线

先看总览图：

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/pareto_only.png)

> 图解：这张图是全文最关键的结论之一，展示了 KARL 在 **成本-质量** 与 **时延-质量** 两个维度的 Pareto 前沿。横轴通常是成本或延迟，纵轴是质量分数。KARL 在更低成本和更低时延下，达到或超过强闭源模型质量，说明它不是单纯“堆算力提分”，而是策略效率本身提升。

---

## 三、KARLBench：为什么这个评测设计很关键？

很多已有 benchmark 只覆盖一种搜索行为，而企业真实任务是多形态的。KARLBench 将任务拆成 6 种能力面向：

1. 约束驱动实体搜索（BrowseComp-Plus）
2. 跨文档报告综合（TREC-Biogen）
3. 长文档表格数值推理（FinanceBench）
4. 穷举式实体检索（QAMPARI）
5. 技术文档流程推理（FreshStack）
6. 企业内部笔记事实聚合（PMBench，私有）

这意味着它不是“在某一个 dataset 刷榜”，而是在评估 Agent 的 **检索策略迁移能力** 。

论文还强调了一个非常工程化的设计点：  
他们刻意只给一个外部工具—— **向量检索** ，避免多工具编排带来的额外变量，从而更干净地评估“检索 + 推理”本体能力。

---

## 四、Agent Harness：为什么只给一个工具反而更强可解释？

论文中的 Agent 交互范式很简单：

- Agent 生成查询
- 调用 Vector Search 获取证据
- 更新轨迹并继续搜索
- 最后产出答案

但它加入了一个非常关键的机制： **上下文压缩（compression）** 。  
当历史轨迹过长时，Agent 会先将历史压缩为摘要，再继续推理。

这不是传统“外置摘要器”，而是把压缩行为也纳入 RL 优化目标。  
换句话说，模型要学会“为了最终答对而压缩”，而不是“为了好看而摘要”。

![Agent Harness](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/aroll_karl_figure.png)

> 图解：该架构图体现了 Dispatcher-Strategy-Environment-Agent 分层。横纵流程分别对应任务分发与单轨迹循环。重点在于生命周期插件（如 compression）可插拔，并且训练与推理阶段共用同一交互接口，降低了 train-serving mismatch。

---

## 五、数据是怎么造出来的？两阶段 Agentic Synthesis

论文的数据构建不是普通 SFT 合成，而是两阶段 Agent 流水线。

### Stage I：问题-答案合成（QA Synthesis）

![Stage1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/stage_1_agentic_synthesis.png)

> 图解：流程从少量种子样例出发，Agent 在语料中主动搜索后生成新问题和参考答案。去重模块会过滤与测试集重复或近重复的样本，避免污染评测。

关键点：

- 生成的问题和答案必须由检索证据支撑
- 去重包含精确去重和语义近重复去重
- 强调多样性和难度，而非简单模板改写

### Stage II：解题轨迹合成（Solution Synthesis）

![Stage2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/stage_2_agentic_synthesis.png)

> 图解：对每个合成题运行多个 solver rollout，根据 pass-rate 筛掉“太简单全对”和“太难全错”的样本，保留学习信号最密集的中间区间；再经质量过滤器剔除歧义题或错误标注题。

这一步本质上是在做 “curriculum via filtering”：  
保留可学但不 trivial 的样本，并显式清洗噪声标签。

---

## 六、方法核心：OAPL 离策略 RL 目标到底在优化什么？

KARL 使用 OAPL（Optimal Advantage-based Policy Optimization with Lagged Inference policy）。

核心目标来自 KL 正则 RL：

$$
\max_{\pi}\ \mathbb{E}_{x,\ y\sim\pi(\cdot|x)}\left[r(x,y)-\beta \, KL\left(\pi(\cdot|x)\|\pi_{ref}(\cdot|x)\right)\right]
$$

进一步可得到一个可回归的优势形式，并在离策略样本上做最小二乘优化：

$$
\min_{\pi}\sum_x\sum_{i=1}^{G}
\left(
\beta_2 \log\frac{\pi(y_i|x)}{\pi_{ref}(y_i|x)}
-
\left(r(x,y_i)-\hat{V}^{\star}(x)\right)
\right)^2
$$

其中价值基线估计为：

$$
\hat{V}^{\star}(x)=
\beta_1\log\left(\frac{1}{G}\sum_{i=1}^{G}\exp\left(\frac{r(x,y_i)}{\beta_1}\right)\right)
$$

直观理解：

- 不是纯在线采样-更新
- 可以大批量离线采样后多轮优化
- 能复用 rollout，训练成本更可控
- 对 trainer 与 serving 引擎差异更稳健

---

## 七、多任务 RL：为什么比“多专家蒸馏”更泛化？

作者对比了两条路线：

- 路线 A：分别训练单任务专家，再蒸馏成一个模型
- 路线 B：直接多任务 RL 联合训练（BrowseComp + TREC）

结果是：两者在 in-domain 表现接近，但多任务 RL 在 OOD 更好，且随着测试时算力提升可继续增长。

![SFT vs RL](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/sft_vs_rl.png)

> 图解：图中横向对比显示，蒸馏模型在已见分布上可提升，但在 OOD 上扩展性有限；RL 模型在 in-domain 与 OOD 都能随着并行采样继续抬升，说明学到的是更通用的搜索策略，而非任务模板。

---

## 八、Test-time Compute：并行思考不是简单投票

论文主要使用两种 TTC：

1. **Parallel Thinking** ：并行生成多个 rollout，再聚合生成最终答案
2. **VGS（Value-Guided Search）** ：训练 value model，在每步分支里选高价值路径

### 8.1 Parallel Thinking

![Parallel Thinking](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/parallel_thinking_karl.png)

> 图解：先并行生成 $N$ 条轨迹，再让同一模型作为 aggregator 读入候选答案并综合。与 Best-of-N 不同，它可以“融合”多个答案信息，而不是只能选一条。

全基准扩展趋势如下：

![Parallel Lines](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/parallel_thinking_lines.png)

> 图解：随着并行数增加，KARL 在各任务持续优于基座模型，且 OOD 提升也明显。曲线后段收益递减，通常来自 pass@k 饱和与聚合上下文变长。

### 8.2 Value-Guided Search（更任务特化）

![VGS Flow](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/bfs_inference_vgs.png)

> 图解：每个步骤先生成多个候选 continuation，用 value model 评估并选最优分支继续。重复多棵搜索树后，再做最终聚合。

![VGS Result](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/vgs.png)

> 图解：在可离散投票任务（如实体答案）里，VGS + Weighted Majority Vote 表现最好，说明 value model 提供了比频次更有效的质量信号。

---

## 九、主结果：KARL 到底赢在哪里？

从论文主表可提炼三条结论：

- **单任务 RL** ：各自任务上能冲高，但跨任务迁移弱
- **多任务 RL** ：不加 TTC 已具强竞争力
- **多任务 RL + 并行思考** ：在总分上达到或超过顶级闭源模型区间

训练后模型还能在更高并行 budget 下继续上涨，这说明 RL 不是只做了“分布锐化（sharpening）”，而是扩展了可解问题覆盖面。

---

## 十、行为分析：RL 后模型到底“学会了什么”？

论文做了大量行为层分析，以下几点尤其有启发。

### 10.1 轨迹更短，但不是盲目早停

![Traj by pass](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/traj_length_flow_by_pass.png)

> 图解：按 Pass@16 分类看，训练后各类样本轨迹整体缩短，尤其已可解样本缩短明显，说明模型减少了无效搜索。

### 10.2 搜索多样性变高

![Search diversity](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/bcp_trec_search_diversity.png)

> 图解：累计独立文档数随步数增长更快，说明 query 合成更丰富，重复检索更少。

### 10.3 搜全证据后的“浪费搜索”变少

![Search efficiency](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/bcp_stacked_searches_clean.png)

> 图解：下部浅色是拿齐关键证据所需搜索，上部深色是拿齐后额外搜索。RL 迭代后，上部显著下降，直接转化为更低成本与更短延迟。

### 10.4 行为模式从“搜不完”转向“探索后提交”

![Category pie](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/category_piecharts.png)

> 图解：KARL 的行为分布更接近强基线模型，`Exhaustive Search, No Convergence` 占比下降，`Explore then Commit` 占比上升。

---

## 十一、压缩能力不是附属功能，而是性能关键项

论文里有一个非常实用的 ablation：去掉 compression 后性能明显下滑。  
这说明长链路 Agent 的关键瓶颈不只是检索质量，还有 **记忆管理策略** 。

![Compression](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/KARL-Knowledge-Agents-via-Reinforcement-Learning/figures/compression.png)

> 图解：每题压缩次数呈右偏长尾分布。大多数样本压缩次数有限，但困难样本需要多次压缩才能维持有效推理链。RL 把“压缩什么、保留什么”学成了任务相关能力。

---

## 十二、这篇论文真正的创新价值在哪里？

KARL 的价值不只是“一个新 SOTA 分数”，更在于给出了一套可落地的方法论：

- 评测上：从单任务刷分转向多能力、异构语料统一评估
- 数据上：让 Agent 在环境中主动发现题目与证据，而非静态拼接
- 训练上：离策略大批量 RL 降低工程复杂度与试验成本
- 推理上：TTC 不只做采样，还做结构化聚合与价值引导
- 系统上：同一 harness 贯穿数据生成、训练、评测、服务，减少分布漂移

如果在做企业知识助手，可优先借鉴以下路径：

1. 先统一 harness 与 reward
2. 再做两阶段数据合成和质量过滤
3. 最后叠加离策略 RL 与并行思考

---

## 十三、局限与后续方向（论文也很诚实）

论文也明确了当前边界：

- 工具空间仍较单一（主要是 vector search）
- 某些算术/表格推理场景仍会早停失败
- 压缩策略还偏 prompt-level，可继续做层级记忆

下一步可自然扩展为：把代码执行、结构化检索、子代理协作并入动作空间，再配合多任务奖励继续拓展能力边界。

---

> 本文参考自 [KARL: Knowledge Agents via Reinforcement Learning](https://arxiv.org/abs/2603.05218)