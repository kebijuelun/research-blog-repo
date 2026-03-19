# Complementary Reinforcement Learning：让 Agent 与经验提炼器共同进化的 RL 新范式

## 一句话结论
这篇工作提出了 **Complementary RL**：不再将经验库视为静态外挂，而是让策略模型（actor）与经验提炼器（extractor）在同一个 RL 闭环中共同训练。核心收益是：经验不会越训越过时，反而会随 actor 能力提升而同步升级，最终在单任务与多任务场景中都带来更高成功率和更好样本效率。

## 1. 这篇论文在解决什么问题？

传统 outcome-based RL（只看任务成败）有两个硬伤：

- 奖励过于稀疏：每条轨迹通常只有最终 $0/1$ 反馈。
- 轨迹信息浪费：中间过程里大量“为什么成功/失败”的行为信号未被充分吸收。

论文将问题形式化为一个 MDP，目标是最大化任务成功率：

$$
\mathcal{J}(\theta)=\mathbb{E}_{\mathcal{E},\,g,\,\tau\sim\pi_\theta}\big[R(\tau)\big]
$$

若引入经验库 $\mathcal{M}$，目标变为：

$$
\mathcal{J}(\theta)=\mathbb{E}_{\mathcal{E},\,g,\,m\sim\mathcal{M},\,\tau\sim\pi_\theta(\cdot|g,m)}\big[R(\tau)\big]
$$

问题在于：现有方法中的经验库常常是 **静态** 的（离线构建，或在线但提炼器不更新），从而与 actor 当前能力分布错配（distributional misalignment）。

## 2. 关键洞察：经验必须“共进化”，而非“只积累”

作者先做了一个关键对比实验：Baseline、Offline Exp.、Static Online Exp.。结论直观明确：

- 离线经验前期有增益，后期会衰减甚至被反超。
- 在线但固定提炼器，提升依然有限。
- 只有 actor 与 extractor 联合演化，才能持续带来正增益。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/method-ablation/Ablation-Compare-training_comparison.png)

> 图解：横轴是训练进程，纵轴是任务表现。Offline/Static 曲线后期趋于停滞甚至回落，说明经验内容未跟上策略分布变化；Co-evolve 曲线持续上行，体现“经验质量—策略能力”的正反馈闭环。

## 3. 方法核心：Complementary RL 的双模型协同优化

### 3.1 Experience Extractor 如何训练？
每条 episode 结束后，extractor 生成经验 $m$，并根据该经验后续是否帮助 actor 成功，赋予二值回报 $r(m)\in\{-1,+1\}$。随后使用 CISPO 目标优化 extractor：

$$
\mathcal{J}_{\text{CISPO}}(\phi)=
\mathbb{E}\!\left[
\frac{\sum_{i=1}^{O}\sum_{t=1}^{|m_i|}
\mathrm{sg}\!\left(\mathrm{clip}(\rho_{i,t})\right)\hat{A}_i\log\pi_\phi(m_{i,t}|g_i,\tau_i,m_{i,<t})}
{\sum_{i=1}^{O}|m_i|}
\right]
$$

其中 $\rho_{i,t}$ 是 token-level importance ratio，$\hat{A}_i=r(m_i)-\bar r$。  
选择 CISPO 而非 REINFORCE 的原因在于：clip 机制能抑制 extractor 更新过猛导致的经验分布突变。

### 3.2 Actor 如何训练？
常规目标是 GRPO：

$$
\mathcal{J}_{\text{GRPO}}(\theta)=
\mathbb{E}\!\left[\frac{1}{K}\sum_{k=1}^K
\min\left(\rho\hat A,\mathrm{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat A\right)\right]
$$

但作者发现：若所有 rollout 都依赖经验，actor 会形成“经验依赖症”，离开经验后能力不足。  
因此将 rollout 分成两组：experience-guided 与 experience-free，并在组内独立归一化 advantage：

$$
\mathcal{J}_{\text{GRPO}}^{\text{split}}(\theta)=
\mathbb{E}\!\left[
\frac{1}{2}\sum_{c\in\{m,\varnothing\}}
\frac{1}{K_c}\sum_{k=1}^{K_c}
\mathcal{L}_{\text{clip}}(\rho_c,\hat A_c)
\right]
$$

该设计避免跨组尺度差异带来的 advantage 偏置与训练塌陷。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/method-ablation/Subgroup.png)

> 图解：横轴为训练步，纵轴为成功率。组内 advantage（Subgroup Adv.）明显比跨组估计更稳定，说明“同条件比较”对 RL 信号质量至关重要。

## 4. 工程实现：异步双轨训练框架是落地关键

算法之外，系统设计同样是贡献点。论文基础设施包括：

- 主训练环：actor 持续 rollout + RL 更新。
- 后台环：extractor 异步处理轨迹、提炼经验、维护经验库。
- 中央组件 `ExperienceManager`：统一执行并发读写协调（读锁检索、写锁更新）。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/pipeline/Pipeline.png)

> 图解：这张图展示了 actor 与 extractor 的异步协作。左侧是并行环境持续产出轨迹，右侧是经验提炼与合并；中间的 Manager 负责检索批处理、队列调度与一致性控制，避免训练线程互相阻塞。

### 4.1 Experience Consolidation（经验整合）
extractor 对每条请求可发起三类操作：

- `Add`：新增经验。
- `Update`：更新已有经验。
- `Return`：无有效经验，不写入。

并周期性执行 `Merge`，去除冗余与冲突条目，提升检索质量。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/ablation/Ablation-wo_Merge-training_comparison.png)

> 图解：去掉 Merge 后曲线显著变差，说明经验库“越攒越脏”会反噬训练；定期合并是经验系统长期可用的必要条件。

### 4.2 Experience Retrieval（经验检索）
检索层采用三项机制提升吞吐：

- Query batching（按 batch 或超时阈值触发）。
- Embedding cache（同任务重复 query 直接命中）。
- Parallel workers（并行语义搜索）。

同时引入 `search_and_ask`，允许 actor 在中间步骤主动二次提问。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/ablation/Ablation-wo_SearchAsk-training_comparison.png)

> 图解：禁用 `search_and_ask` 后性能下降，说明“动态追问”能让经验从“静态前置提示”升级为“过程内决策支持”。

## 5. 实验结果：不仅更高分，也更省动作

### 5.1 单任务（MiniHack / WebShop / ALFWorld / SWE-Bench）
论文报告 Complementary RL 在四个环境中都优于无经验基线，且在探索型任务上的提升更明显。  
例如 MiniHack 与 ALFWorld 约有 $1.3\times$ 优势，SWE-Bench 也有约 $+3\%$ 增益。

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/experiments/single_task/ROOM.png)

> 图解：单任务曲线整体趋势是 Comp. RL 上升更快、上限更高，尤其在需要长期规划与探索的环境中优势更突出。

### 5.2 动作效率
论文还分析了平均动作数：在多个任务中，Comp. RL 用更少步骤达成更高成功率（例如 MiniHack、ALFWorld）。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/experiments/single_task/num_actions/ALF.png)

> 图解：横轴是训练进程，纵轴是平均动作数。成功率提升同时动作数下降，说明经验并非“多想一步”，而是“少走弯路”。

### 5.3 多任务：共进化优势更明显
在多任务对比中，Static Online Exp. 甚至不如 baseline；而 Complementary RL（有/无测试时经验）均领先，说明 actor 参数本身确实学习到了经验，而非仅依赖推理时外挂。

| 方法 | MiniHack | WebShop | ALFWorld | 平均 |
|---|---:|---:|---:|---:|
| Baseline | 0.68 | 0.81 | 0.72 | 0.75 |
| Static Online Exp. (w/ exp) | 0.41 | 0.67 | 0.69 | 0.59 |
| Comp. RL (w/ exp) | 0.78 | 0.87 | 0.82 | 0.82 |
| Comp. RL (w/o exp) | 0.75 | 0.84 | 0.74 | 0.78 |

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/experiments/multi_task/ALL-training_comparison.png)

> 图解：多任务总曲线中，Comp. RL 始终高于 baseline 与静态经验方案，验证了“经验-策略双向更新”在复杂分布下更稳健。

## 6. 高价值附录结论（复现建议重点关注）

### 6.1 两个稳定性技巧
- **Retrieval Diversification**：检索时对高频经验加惩罚，避免反复投喂同一条经验。  
  评分函数：

  $$
  s(m)=s_{\text{rank}}(m)-\lambda\log(1+c(m))-\mathbb{1}[\text{recent}(m)]
  $$

- **Training-count-aware Reweighting**：经验被训练过多次会降权，并设置 cooldown 以避免短期重复优化：

  $$
  w(m)=
  \begin{cases}
  0,& (t-t_{\text{last}})<\delta \\
  (1+c_{\text{train}}(m))^{-\alpha},& \text{otherwise}
  \end{cases}
  $$

### 6.2 一个“有效但变慢”的增强：Actor-Critic
actor 会对每条检索经验先做 `accept/refine/reject` 打分，再反馈给 extractor。该机制可提升早期性能，但会引入显著时延，因此主实验未默认启用。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Complementary-Reinforcement-Learning/figs/appendix/actor_critic/actor-critic-perf.png)

> 图解：性能曲线前期更优，但对应速度图显示检索/决策链路更慢，体现了典型的“效果-吞吐”权衡。

### 6.3 工程教训
- actor 与 extractor 不应共用同一套参数：目标冲突会导致不稳定。
- 给 extractor 直接任务回报通常优于相对回报。
- 加入“熵下降奖励”在直觉上合理，但文中未观察到稳定增益。

## 7. 方法意义与评价

这篇工作的核心贡献不只是“加入经验库”，而是将经验学习从 **检索增强（RAG-like）** 提升为 **可训练的闭环系统**：

- 算法层：缓解了“经验指导”与“无经验泛化”之间的张力（split rollout + subgroup advantage）。
- 系统层：异步双轨 + 中央管理器将方法落地为可扩展工程。
- 实证层：单任务、多任务、动作效率与延迟分析构成了较完整证据链。

对 Agent RL、长期交互任务与工业后训练而言，这篇论文最值得借鉴的观点是：  
经验不是静态资产，而是一个需要被优化、被去重、被调度、被评估的动态模型对象。

## 8. 总结
**Complementary RL** 给出了一个实用方向：让“执行决策的模型”与“提炼经验的模型”在同一训练生态中相互塑造。它既解释了静态经验为何失效，也提供了可复用的算法与系统范式。对于大模型 Agent 的下一阶段训练，这一思路很可能成为标准组件之一。

> 本文参考自 [Complementary Reinforcement Learning](https://arxiv.org/abs/2603.17621)