# LoongFlow：用 “Plan‑Execute‑Summarize” 把 LLM 进化搜索从随机突变变成有方向的认知进化

## 一句话先懂
这篇论文关注的是 **“LLM 自我进化”** 这件事：如何让模型不仅能写代码，还能持续改进自己的解法。作者提出 LoongFlow，用 “Plan‑Execute‑Summarize (PES)” 把进化搜索变成有计划、有验证、有反思的闭环，并用混合记忆结构解决多样性崩塌，最终在 AlphaEvolve 与 Kaggle 任务上实现更高质量、更低成本的进化。

## 背景：为什么现有方法不够？
现有 LLM 进化框架（如 OpenEvolve、ShinkaEvolve）通常把 LLM 当成随机突变器，带来三个瓶颈：

-  **探索低效** ：没有计划层，基本靠随机采样，计算与 token 消耗巨大  
-  **多样性崩塌** ：Top‑K 选优会提前收敛，错过 “潜力解”  
-  **缺乏反思记忆** ：不像梯度学习有明确反馈，失败原因无法累计成知识  

LoongFlow 的设计目标就是把进化从 **“随机试错”** 变成 **“可复用的认知进化”**。

## 关键思想：PES + Hybrid Evolutionary Memory
LoongFlow 的核心是两个模块：

1.  **PES 认知进化闭环**  
2.  **混合进化记忆系统** （Multi‑Island + MAP‑Elites + 自适应 Boltzmann 选择）

这两者组合起来，让进化既有方向性，又保持多样性。

---

## 方法细解

### 1. 将进化建模为 MDP
论文把 “代码进化” 形式化为 MDP，状态是代码，动作是代码修改，奖励是评测得分：

$$
s^* = \mathop{\arg\max}_{s \in \mathcal{C}} R(s)
$$

这一步的核心意义在于：把进化看成 **可决策过程**，从而可以引入结构化推理。

### 2. PES：Plan‑Execute‑Summarize 的三阶段推理闭环
#### (1) Plan：制定蓝图
Planner 结合当前代码与历史记忆生成计划：

$$
b \sim \pi_\theta(b \mid s_t, \mathcal{M}_t, \mathcal{I}_{plan})
$$

关键点是 **“基于谱系的上下文检索”**，它不是简单 RAG，而是沿着 parent‑child 关系回溯历史计划与总结。

#### (2) Execute：可执行生成 + 本地验证
Executor 把计划转为代码并进行本地检查，快速剔除无效候选：

$$
s' \sim \pi_\theta(s' \mid b, s_t, \mathcal{I}_{exec})
$$

这样避免无效方案进入昂贵的评测流程。

#### (3) Summarize：生成反思记忆
Summarizer 结合得分与日志，总结出可复用的经验：

$$
z \sim \pi_\theta(z \mid s', r, b, \mathcal{I}_{sum})
$$

并将其写入记忆：

$$
\mathcal{M}_{t+1} \leftarrow \mathcal{M}_t \cup \{z\}
$$

**核心价值**：失败原因被结构化保存，下一代可以直接避坑。

---

### 3. Hybrid Evolutionary Memory：解决多样性崩塌
#### (1) Multi‑Island：并行进化
不同岛屿独立进化，防止单一策略垄断。周期性迁移精英，增强全局多样性。

#### (2) MAP‑Elites：保留行为多样性
把代码映射到特征空间 $\mathcal{F}$，每个格子只保留最优个体：

$$
\mathcal{A}_{archive}(\mathbf{v}) = \{ s \in \mathcal{C} \mid \Phi(s) \in \text{Cell}(\mathbf{v}) \land R(s) = \max_{s' \in \text{Cell}(\mathbf{v})} R(s') \}
$$

避免进化只围绕 “单一最优解” 打转。

#### (3) 自适应 Boltzmann 选择
基于熵自动调节探索‑利用比例：

$$
P(s_i) = \frac{\exp(R(s_i) / \tau)}{\sum_{j=1}^{N} \exp(R(s_j) / \tau)}
$$

当群体趋同（低熵）时升温，逼迫探索；当多样性高时降温，强化利用。

---

## 框架全景图

![Figure 1](media/LoongFlow-overview.jpg)  
> 图解：整体流程由 PES 认知闭环驱动，外层用混合记忆结构维持多样性与长期进化稳定性。  
> 注意：图片文件可能缺失：media/LoongFlow-overview.jpg

![Figure 2](media/LoongFlow-frame.jpg)  
> 图解：Planner 负责基于谱系检索制定计划，Executor 负责实现与本地验证，Summarizer 负责总结反思并写回记忆。  
> 注意：图片文件可能缺失：media/LoongFlow-frame.jpg

---

## 算法视角：LoongFlow 主循环
LoongFlow 的主流程融合了 **自适应选择 + PES 执行 + MAP‑Elites 更新 + 周期迁移**，让进化既具方向性，又具全局多样性。

---

## 实验结果：效果与效率双赢

### 1. AlphaEvolve：算法发现能力
LoongFlow 在多个数学问题上超过 AlphaEvolve 与其他基线：

-  **Autocorrelation II** ：从 0.8962 提升到 0.9027  
-  **Circle Packing** ：小幅持续突破最优  
-  **多项 lower‑is‑better 指标** 也实现更优  

这表明 PES 的全局规划能力能帮助系统走出随机突变的瓶颈。

### 2. MLEBench：ML 工程能力
LoongFlow 在 Kaggle 任务上拿到 **14 个 Gold**，涵盖 CV、NLP、Tabular 等多种任务，说明框架具有跨域可扩展性。

### 3. 效率提升
在 Circle Packing 任务中：

-  LoongFlow 平均 **258 次评测** 达到目标  
-  OpenEvolve 需要 **783 次**  
-  提升幅度 **>60%**

### 4. 复杂任务突破
在 100 次迭代限制下：

-  LoongFlow 三次全部突破 1.0  
-  两个基线完全失败  

说明 PES 不只是 “更快”，而是能探索到基线无法触达的解空间。

---

## 消融实验：PES 的必要性被验证

![Figure 3](media/ablation1.png)  
> 图解：去掉 Planner 后性能明显下降，收敛时间显著变长，说明规划能力是加速进化的关键。  
> 注意：图片文件可能缺失：media/ablation1.png

![Figure 4](media/ablation2.png)  
> 图解：无 Summary 的曲线长期震荡，说明反思记忆是避免循环错误的必要模块。  
> 注意：图片文件可能缺失：media/ablation2.png

-  **无 Planner** ：陷入盲搜，难以突破 0.96  
-  **无 Summary** ：反复犯错，35 小时仍停留在 0.95  
-  **Fuse Executor** ：在稳定性与效率之间取得最好平衡  

---

## 总结：LoongFlow 的真正贡献
LoongFlow 的最大价值不只是 “更高分” 或 “更快收敛”，而是：

-  把 **进化搜索** 从随机变成 **认知驱动**  
-  把 **局部反馈** 变成 **长期记忆**  
-  让 **进化多样性** 不再靠运气，而是结构化维持  

这意味着 LLM 不再只是代码生成器，而是能 **持续进化的智能体**。

---

> 本文参考自 [LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm](https://arxiv.org/abs/2512.24077v1)