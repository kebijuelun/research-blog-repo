# AgentConductor：让多智能体拓扑“边执行边进化”的竞赛级代码生成新范式（论文 2602.17100 深度解读）

## 1. 这篇论文到底在解决什么问题？

在竞赛级代码生成（如 APPS、LiveCodeBench、CodeContests）中，单个 LLM 往往会遇到三个瓶颈：

- 难题需要多步拆解、检索、验证、调试，单链路推理不够稳定；
- 固定的多智能体流程（固定角色 + 固定通信图）对简单题又过于“重”；
- 即便代码执行失败，很多方法也只是在同一拓扑上迭代，无法根据错误反馈重构协作结构。

这篇论文的核心主张很直接：  
 **拓扑结构本身应该成为可学习、可反馈、可进化的策略变量** ，而不是预先写死的“工作流模板”。

---

## 2. 核心思路一图看懂：从静态拓扑到动态拓扑

> 图解：这张图对比了传统基线与 AgentConductor 的优化范式。传统方法要么固定拓扑，要么一次性生成后冻结；AgentConductor 在每一轮执行后读取错误反馈，并在下一轮生成更匹配当前状态的 DAG 拓扑，实现“结构级自适应”。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.17100/AgentConductor-case_show_0129.png)

> 图解：左侧展示 YAML 拓扑描述，中间是其映射后的真实图结构，右侧是两轮演化示意。横向可看作不同 step（层），纵向是该层并行 agent；跨层 ref 边代表依赖与信息流。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.17100/only_topo_1.png)

---

## 3. 方法总览：AgentConductor 的三阶段训练与推理

> 图解：框架分三阶段：先做 SFT，学会“规范地写拓扑”；再用 GRPO 做轨迹级 RL，学会“按难度和反馈调拓扑”；最后冻结 orchestrator 做多轮推理。图中同时展示了角色池、执行环境与反馈闭环。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.17100/topoweaver-main.png)

### 3.1 交互范式：Orchestrator 生成 YAML，环境执行并回传反馈

给定题目 $x$，第 $k$ 轮由策略 $\pi_\theta$ 生成拓扑 token 序列：

$$
o_k = (o_{k,1}, \ldots, o_{k,|o_k|})
$$

并解析成分层 DAG：

$$
\mathcal{G}^{(k)} = \mathrm{DecodeTopo}(o_k)
$$

随后环境执行该拓扑并返回反馈 $z_k = (z_k^{\text{roles}}, z_k^{\text{code}})$。  
整个多轮过程可分解为：

$$
p_\theta(o_{1:K}, z_{1:K}\mid x)
= \prod_{k=1}^{K}
\pi_\theta(o_k \mid x, H_k)\cdot
P_{\text{env}}(z_k \mid x,\mathcal{G}^{(k)},H_k)
$$

其中 $H_k$ 是历史拓扑与执行结果。关键点是： **下一轮拓扑显式依赖上一轮执行反馈** 。

### 3.2 拓扑结构设计：分层 DAG + 层内并行 + 跨层连接

论文提出的拓扑并非简单 chain/tree，而是“受约束的分层 DAG”：

- 支持层内并行（提高吞吐）；
- 支持跨层引用（增强信息复用）；
- 避免全连接 mesh（控制成本和复杂度）。

使用 YAML 表示拓扑也很实用：可读、可编辑、可被 LLM 直接生成。

---

## 4. 为什么它能兼顾准确率和成本？关键在“难度感知密度控制”

论文提出了一个拓扑复杂度/稀疏度打分函数 $\mathcal{S}_{\text{complex}}$，基于三个量：

- 节点数 $|V|$（agent 调用规模）；
- 边数 $|E|$（通信负载）；
- 层数 $s$（近似图深度，影响串行延迟）。

先做归一化：

$$
\begin{aligned}
S_{\text{node}} &= \exp\!\left(-\frac{|V|}{N_{\max}(l)}\right),\\
S_{\text{edge}} &= \exp\!\left(-\frac{|E|}{|V|(|V|-0.5)}\right),\\
S_{\text{depth}} &= 1 - \frac{s}{|V|}.
\end{aligned}
$$

再组合成总体指标：

$$
\mathcal{S}_{\text{complex}}
= \exp\!\left(
S_{\text{node}} + 2S_{\text{edge}} + S_{\text{depth}}
\right)
$$

- $\mathcal{S}_{\text{complex}}$ 越大，表示拓扑越稀疏、成本越低；
- 难度分级上限 $N_{\max}(l)$：easy/medium/hard 分别对应 4/7/10（每轮）。

这避免了“所有题一刀切变稀疏”的粗暴策略，转为 **按题目难度动态分配拓扑容量** 。

---

## 5. 强化学习目标与奖励：结构正确 + 代码正确 + 密度约束

即时奖励写为：

$$
r_\phi(\mathcal{G}^{(k)}, z_k^{\text{code}})
= r_e(\mathcal{G}^{(k)}, z_k^{\text{code}}) + r_g(\mathcal{G}^{(k)})
$$

- $r_e$：执行正确性奖励（含 YAML 合法性 + 代码执行错误类型）；
- $r_g$：拓扑密度奖励（是否满足难度上限，以及稀疏程度）。

论文使用 GRPO 做轨迹级优化，并将拓扑 token 作为主要策略优化对象。  
一个很实用的点是：YAML 错误会被明确惩罚（如无 YAML、解析错误、schema 错误、逻辑错误），这显著提高了可执行拓扑比例。

---

## 6. 实验结果：准确率、成本、拓扑稀疏性三赢

### 6.1 主结果（5 个代码数据集）

论文报告 AgentConductor（3B）在五个基准上都达到最优平均表现。最亮眼的是竞赛级任务：

- APPS：58.8（相对次优 +14.6 个百分点）；
- LiveCodeBench：46.3；
- CodeContests：38.8。

基础代码任务也领先：

- HumanEval：97.5；
- MBPP：95.1。

### 6.2 成本与密度（以 APPS 为例）

> 图解：(a) 同时比较了 pass@1、completion tokens、$\mathcal{S}_{\text{complex}}$。AgentConductor 在更高准确率下，用更少 token，并获得更高稀疏度分数。(b) 展示了代表性方法的性能对比，AgentConductor 处于领先位置。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.17100/fig2bc.png)

论文给出的关键结论是：  
在 APPS 上，相比强基线可实现 **更高准确率 + 更低 completion token（最高约降低 68%）+ 更优稀疏拓扑** 。

### 6.3 难度分层分析：密度真的会随难度调节

> 图解：横轴是难度等级，纵轴是平均 $\mathcal{S}_{\text{complex}}$（越高越稀疏）。可以看到 AgentConductor 在不同难度上呈现更细粒度的密度调控，而不是几乎不变的固定风格。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2602.17100/graph_density_3_datasets.png)

---

## 7. 消融实验告诉我们：哪些模块最关键？

- 去掉 SFT：有效拓扑率骤降（几乎无法稳定生成可执行结构）；
- 去掉 RL：准确率与有效率明显下降；
- 去掉 YAML 错误奖励：拓扑合法性大幅退化；
- 去掉代码错误奖励：pass@1 明显下降；
- 去掉密度奖励项：成本控制变差，准确率也受影响。

结论很清晰：  
 **这不是“单靠更强模型”得到的提升，而是奖励设计 + 拓扑表示 + 反馈进化共同作用的结果。**

---

## 8. 附录里值得关注的两个点

### 8.1 理论推导：把 token 成本映射到图结构

论文从每个 agent 的 prompt / 引用 / 输出成本出发，推导出平均成本近似：

$$
\bar{\mathcal{C}} \propto 1 + |V| + 2\frac{|E|}{|V|} + d
$$

再结合分层 DAG 的性质，用 step 数 $s$ 近似深度项，得到可优化的拓扑密度目标。这部分为“为什么要同时约束节点、边、层”提供了数学依据。

### 8.2 跨域泛化

作者还做了 GAIA/HLE/PopQA 等非代码任务实验（通过扩展角色池并替换执行验证器）。结果显示，3B 模型在这些任务上也具备迁移能力，说明这种“拓扑策略学习”并不只适用于代码场景。

---

## 9. 我对这篇工作的评价

这篇论文最有价值的地方，不是再造一个多智能体模板，而是把 **拓扑生成本身纳入 RL 闭环** ：

- 结构可变，而非流程固定；
- 目标多元，而非只看正确率；
- 粒度细化到“按难度设拓扑预算”；
- 通过执行反馈实现结构级在线修正。

如果你在做 Agentic Code Generation，这篇工作给出的启发是：  
与其持续调 prompt 和角色话术，不如把“谁在什么时候与谁沟通”建模成可学习策略，这通常更接近性能与成本的根因。

> 本文参考自 [AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation](https://arxiv.org/abs/2602.17100)