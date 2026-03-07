# CodeEvolve 论文深度解读：用 LLM + 进化搜索做算法发现，开源方案如何对标 AlphaEvolve

## 先看结论

这篇论文的核心价值非常直接：作者将“让 LLM 自动写算法、再自动筛选和改进”的流程落成了完整的开源框架 CodeEvolve，并在 AlphaEvolve 使用的同类基准上，取得了 9 个任务中 5 个任务不弱于或优于 AlphaEvolve 的结果。更有意思的是，在不少任务里，开源模型 Qwen3-Coder-30B 在效果接近甚至更好时，成本只有 Gemini 方案的一小部分。

## 1. 论文到底在解决什么问题

传统 Genetic Programming（遗传编程）有两个痛点：

- 代码语义复杂，随机变异很容易产生不可执行代码。
- 搜索空间过大，容易陷入局部最优。

LLM 出现后，代码变异从“字符级拼接”升级为“语义级改写”，但也带来了新问题：

- 单次生成不稳定，仅靠 prompt 工程难以持续提升。
- 闭源系统（如 AlphaEvolve）细节不公开，难以复现，也难做系统性消融。

CodeEvolve 要做的是：把 LLM 作为“语义变异器”，把进化算法作为“全局搜索器”，再结合群岛模型、元提示词、灵感交叉、MAP-Elites 等机制，拼成一个可复现的开源工程体系。

## 2. 方法总览：CodeEvolve 是怎么运作的

> 图解：这是整体框架图。每个 island（岛）都有自己的解和提示词种群，并独立进化；周期性迁移会把优秀个体传播到邻居岛，避免全局早熟收敛。LLM 不是单一调用，而是按探索/开发策略动态调用不同模型。  
![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/codeevolve_diagram_V8.png)

### 2.1 核心对象定义

论文将问题形式化得很清晰：

- solution：一个可执行程序。
- prompt：生成 solution 的提示词。
- 评估函数：$h:\mathcal{S}\mapsto\mathbb{R}^d$，返回多维指标（如目标值、时间、内存）。
- solution fitness：$f_{\mathrm{sol}}(S)$，主优化目标。
- prompt fitness：由它生成过的最好 solution 决定：

$$
f_{\mathrm{prompt}}(P)=\max_{S:P(S)=P}\{f_{\mathrm{sol}}(S)\}
$$

这个定义很关键：它奖励“有潜力产出好解”的 prompt，而不是奖励平均表现。

### 2.2 两类进化算子：开发 + 探索

CodeEvolve 每一步根据探索率 $p_{\text{explr}}$ 选择算子：

- 深度开发（Depth exploitation）：围绕高分父解做局部精修。
- 元提示探索（Meta-prompting exploration）：让辅助 LLM 改写 prompt，再生成新解，主动跳出当前谱系。

开发阶段父解采样使用 rank-based 概率：

$$
\Pr(S)=\frac{\mathrm{rk}(S)^{-1}}{\sum_{S'\in\mathcal{S}_t^i}\mathrm{rk}(S')^{-1}}
$$

排名越高，被继续优化的概率越大。

### 2.3 “灵感式交叉”是一个非常实用的创新点

作者没有采用传统代码拼接（容易语法崩溃），而是给 LLM 多个 inspiration solutions，让模型“借鉴思路、结构、函数模式”进行语义重组。这个设计在消融中被证明非常关键：它单独就有机会超过 AlphaEvolve，而单独 Depth 并不稳定。

### 2.4 调度与种群管理

- 探索率调度：
  - Decay：前期高探索，后期降探索。
  - Plateau：若窗口期无提升就抬高探索率，破局后再降回。
- 执行评估：沙箱运行 + 超时/内存限制；失败解记 0 分并保留日志。
- 迁移策略：精英迁移但抑制循环迁移，且保留每岛最优个体不迁走。
- 质量多样性：并行维护 MAP-Elites/CVT-MAP-Elites 档案，按特征格子存 elite，增强多样性覆盖。

## 3. 实验任务与评测问题

基准覆盖三类问题：

- 几何 packing：CirclePackingSquare/Rect、HexagonPacking。
- 距离比优化：MinimizeMaxMinDist。
- 自相关不等式构造：First/Second AutocorrIneq。

对比对象主要是 AlphaEvolve 与 ThetaEvolve。论文还做了 Qwen3 与 Gemini 配置对比，重点回答三个研究问题：

- 能否推进 SOTA。
- 开源小模型是否可与闭源大模型竞争。
- 各组件到底贡献多大。

## 4. 主结果：9 个任务里 5 个不弱于或超过 AlphaEvolve

### 4.1 代表性结果图

> 图解：这是 CirclePackingSquare($n=26$) 的最优构型对比图，通常展示圆心位置与半径分布。视觉上越“填满边界且冲突少”的排布，对应总半径和越高。  
![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/results/results_circlepackingsquare_qwen.png)

> 图解：这是 MinimizeMaxMinDist 的二维实例对比，目标是降低“最大距离/最小距离”比值。图中点集分布越均匀、极端远近点对越少，指标越优。  
![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/results/results_minmaxmindist_gemini.png)

> 图解：这是三维实例的结果投影/可视化对比。虽然可视化有投影损失，但仍能看出解在空间分散与局部拥挤上的差异。  
![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/results/results_minmaxmindistd3_gemini.png)

### 4.2 关键数字（论文汇总）

- CirclePackingSquare($n=32$)：CodeEvolve（Qwen）达到 2.93954，优于 AlphaEvolve 的 2.93794。
- MinimizeMaxMinDist($n=16,d=2$)：CodeEvolve（Gemini）为 12.88923，略优于 AlphaEvolve 的 12.88927（该任务越小越好）。
- MinimizeMaxMinDist($n=14,d=3$)：CodeEvolve（Gemini）为 4.16579，优于 AlphaEvolve 的 4.16585。
- 自相关任务上 ThetaEvolve 有优势，但覆盖任务较少；CodeEvolve 在更多类型任务上保持了稳定竞争力。

## 5. 成本效率：开源模型配置的性价比非常突出

> 图解：左图是 Qwen3-Coder-30B 在 $n=26$ 上的解历史。横轴通常是 LLM 调用步数，左纵轴是性能变换指标（论文用 $-\log(M-y+\epsilon)$），右纵轴是累计 API 成本。曲线上升代表逼近最优。  
![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/sol_hist_qwen_circlepackingsquare_26.png)

> 图解：右图是 Gemini 配置在同任务下的曲线。它往往以更少调用数达到强解，但单次调用更贵，总成本显著高于 Qwen。  
![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/sol_hist_gemini_circlepackingsquare_26.png)

论文给出的典型结论：

- Qwen 配置约 900 次调用后超过 AlphaEvolve，成本约 \$6。
- Gemini 配置约 400 次调用达到类似里程碑，成本接近 \$35。
- 这说明在此类算法搜索任务中，“编排策略”比“堆最大模型”更决定最终性价比。

## 6. 消融实验：哪些组件是真正的胜负手

### 6.1 全量方法 vs 简化版本

> 图解：这是 $n=26$ 的组件消融曲线。full method 在均值与最优包络上都优于 naive/no evolution，且更快跨过 AlphaEvolve 阈值。  
![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/operator_ablation_circlepackingsquare_26.png)

> 图解：这是 $n=32$ 的同类消融。full method 是唯一稳定超过 AlphaEvolve 的配置，说明组件协同不是“锦上添花”，而是“能否破纪录”的必要条件。  
![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/operator_ablation_circlepackingsquare_32.png)

### 6.2 深度上下文与灵感数量

> 图解：深度 $k$ 消融显示，仅靠祖先链深挖并不能稳定越过 SOTA，说明局部精修容易卡住策略空间。  
![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/depth_ablation_circlepackingsquare_32.png)

> 图解：灵感数 $\iota$ 消融显示，$\iota=2,3$ 时能超过 AlphaEvolve，验证了“语义交叉”比“单线继承”更能产生突破型方案。  
![Figure 10](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/insp_ablation_circlepackingsquare_32.png)

### 6.3 MAP-Elites 与迁移拓扑

> 图解：CVT-MAP-Elites 在均值和样本效率上优于普通网格与 naive 选择，表明“按行为特征保留多样 elite”能有效降低早熟收敛。  
![Figure 11](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/elite_ablation_circlepackingsquare_32.png)

> 图解：Cycle 拓扑优于 Complete 与 Empty。迁移太少则信息不流动，迁移太多会导致同质化，环形拓扑在探索与扩散之间更平衡。  
![Figure 12](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/method/figs/ablations/top_ablation_circlepackingsquare_32.png)

## 7. 附录里的数学任务与公式要点

两类自相关不等式任务本质上是“构造非负阶梯函数”以优化特定比值。

FirstAutocorrIneq（最小化）：

$$
\max_{-1/2\le t\le 1/2}(f*f)(t)\ge C_1\left(\int_{-1/4}^{1/4}f(x)\,dx\right)^2
$$

目标是让对应比值尽可能小。

SecondAutocorrIneq（最大化）：

$$
\|f*f\|_2^2\le C_2\|f*f\|_1\|f*f\|_\infty
$$

目标是构造函数，让相关比值尽可能大（在理论上界约束下给出更优下界构造）。

这类问题对“可执行评估器 + 演化生成”特别友好：评价信号明确，且可自动批量验证。

## 8. 对复现者最有用的工程启发

- 先打磨“执行沙箱与评估函数”，再追求更大模型。
- 探索率要动态调度，固定值在后期很容易浪费预算。
- 灵感式交叉优先级很高，是突破局部最优的重要来源。
- 建议保留 MAP-Elites，尤其在搜索空间多峰、策略多样时。
- 可先用开源模型跑通低成本实验，再按收益决定是否引入昂贵闭源模型。

## 9. 局限与后续方向

作者也坦诚指出了边界：

- Gemini 版本未做同等规模消融，模型无关性仍需更多证据。
- 对比基线多采用文献报告值，未完全重跑。
- 超参数较多（islands、depth、inspirations、topology），迁移到新任务仍需调参。
- 大规模进化依然有推理预算门槛。

总体来看，CodeEvolve 已经把“可复现的 LLM 进化式算法发现”从概念推进到可操作系统，社区可以直接在其框架上迭代新算子和新任务。

> 本文参考自 [CodeEvolve: an open-source evolutionary framework for algorithmic discovery and optimization](https://arxiv.org/abs/2510.14150)