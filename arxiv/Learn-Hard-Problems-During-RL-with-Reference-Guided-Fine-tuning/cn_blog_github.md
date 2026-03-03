# ReGFT：先把“不会做的题”教会模型，再让 RL 发挥威力

## 一句话先看懂
这篇工作要解决的核心问题很直接：在数学推理的 RL（尤其是 RLVR）里，很多难题一开始根本采不到正确轨迹，奖励几乎全是 0，训练就会“卡死”。作者提出了 **Reference-Guided Fine-Tuning（ReGFT）** ：先用“参考解的部分提示”引导模型生成 **自己的** 正确推理轨迹，再做监督微调，最后接 DAPO 做 RL。结果是在 AIME’24、AIME’25、Beyond-AIME 上都表现为收敛更快、上限更高。

## 背景：为什么 RL 会在数学题上失灵？
在 RLVR 里，学习信号来自“是否答对”。如果模型在某类难题上完全不会，就会出现：

- 采样很多条推理，但全错
- verifier 给不出正奖励
- 梯度信息几乎无效
- 算力大量浪费在无信息样本上

作者把这件事称为 **reward sparsity** （奖励稀疏）。这不是优化器的小问题，而是“根本拿不到正样本”的数据分布问题。

## 现有办法的短板：ReFT 为什么不够？
论文先回顾了 ReFT（只用模型自己采样到的正确轨迹做 SFT）：

- 优点：对“本来就偶尔能做对”的题有效，能提升正确轨迹概率
- 缺点：对“完全不会”的题无能为力，因为它依赖模型先采到正确样本

所以，ReFT 更像“强化已有能力”，较难扩展能力边界。

## 核心方法：ReGFT 到底做了什么？

### 方法总览
![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/approach_4.png)

> 图解：上半部分是 ReFT，只训练普通采样中验证正确的轨迹；下半部分是 ReGFT，对“普通采样无正确轨迹”的难题，引入参考解的部分提示进行引导采样，再把这些“模型自己写出来但受提示启发”的正确轨迹加入训练。横向可理解为数据来源变化，纵向可理解为从“可解题”扩展到“原本不可解题”。

ReGFT 的关键不是“把人类参考解喂给模型背下来”，而是：

- 给题目 + 参考解前 80% 作为 Hint
- 让模型从头自己推理（不是补全参考解）
- 仅保留验证正确的轨迹
- 与普通自生成正确轨迹混合做 SFT

这样做的动机是：既借助参考解的结构信息，又保持轨迹处在模型自身推理分布里，减少“学会看、不会用”的分布错配。

### 训练对象如何选？
作者只在“难题子集”上做 ReGFT。定义是：原模型对该题采样 16 次准确率 < 25%。  
这能避免在简单题上过拟合，并把学习预算集中在 reward sparsity 最严重的区域。

### 与 RL 的关系：为什么 ReGFT + DAPO 能叠加？
作者选择的 RL 算法是 DAPO（GRPO 变体，强调 decoupled clipping + dynamic sampling）。论文强调：

- DAPO 解决的是“优化与采样机制”
- ReGFT 解决的是“初始化能力与正样本密度”

两者是正交的。即使 DAPO 已经很强，ReGFT 仍能继续抬高性能。

## 实验设置（复现重点）
- 基座模型：Qwen3-4B-2507-Instruct
- 训练集：OmniMath（4428 题）
- 评测集：AIME 2024 / AIME 2025 / Beyond-AIME
- RL 框架：verl
- 生成长度：16384 tokens
- 采样参数：temperature = 0.7，top-$p$ = 0.9

附录里的 RL 细节包括：

- AdamW，学习率 $1 \times 10^{-6}$
- 前 20 rollout steps 线性 warm-up
- response size = 16 时 prompt batch = 512
- response size = 64 时 prompt batch = 128
- mini-batch = 2048
- Clip-Higher：$\varepsilon_{low}=0.2,\ \varepsilon_{high}=0.28$

## 主结果一：ReGFT 初始化让 RL 更快、更高
![AIME24](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/aime24_acc_comparison_main.png)
![AIME25](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/aime25_acc_comparison_main.png)
![BeyondAIME](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/beyond_aime_acc_comparison_main.png)

> 图解：横轴是 RL 训练步数，纵轴是评测准确率。三条 benchmark 上，ReGFT 初始化曲线整体高于 raw 初始化：前期爬升更快，中后期平台更高，说明它不仅提升收敛速度，也提升最终可达上限。

## 主结果二：ReGFT 明显优于 ReFT 终点性能
![AIME24 ReFT vs ReGFT](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/aime24_acc_comparison_ReFT_ablation.png)
![AIME25 ReFT vs ReGFT](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/aime25_acc_comparison_ReFT_ablation.png)
![BeyondAIME ReFT vs ReGFT](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/beyond_aime_acc_comparison_ReFT_ablation.png)

> 图解：ReFT 在早期也有加速，但收敛后普遍被 ReGFT 超过；在 Beyond-AIME 上甚至可能不如 raw DAPO。这说明“只强化已有成功轨迹”不足以打开难题能力边界。

## 主结果三：直接 SFT 人类参考解不行
![AIME24 SFT ablation](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/aime24_acc_sft_ablation.png)
![AIME25 SFT ablation](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/aime25_acc_sft_ablation.png)
![BeyondAIME SFT ablation](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/beyond_aime_acc_sft_ablation.png)

> 图解：直接拿人类参考解做 SFT，后续 RL 也达不到 ReGFT 水平。核心原因是分布不对齐：模型不一定能内化“非自身生成风格”的推理链。

这基本回答了“为什么 ReGFT 不直接蒸馏参考 CoT”：关键不是正确性本身，而是“可被该模型生成机制吸收”的正确性。

## 主结果四：推理时扩展（pass@k）更稳
![AIME24 pass@k](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/test_time_scaling_aime24.png)
![AIME25 pass@k](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/test_time_scaling_aime25.png)
![BeyondAIME pass@k](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learn-Hard-Problems-During-RL-with-Reference-Guided-Fine-tuning/figures/test_time_scaling_beyond-aime.png)

> 图解：横轴是 $k$（测试时采样预算），纵轴是 pass@k。ReGFT + DAPO 在各个 $k$ 区间都更稳健，优势不会随着 $k$ 增大而消失，说明它提升的是“可发现正确解的覆盖能力”，而不只是 pass@1 的偶然增益。

论文使用的估计式为：

$$
\mathrm{pass@k}=1-\frac{\binom{N-c}{k}}{\binom{N}{k}}
$$

其中 $N=1024$ 为采样总数，$c$ 为其中正确样本数。

## 表格结果（关键数字）
先看 SFT 后检查点（pass@64）：

- Raw：AIME24 59.2，AIME25 46.7，BeyondAIME 30.5
- ReFT：AIME24 62.1，AIME25 46.8，BeyondAIME 31.3
- ReGFT：AIME24 60.0，AIME25 47.8，BeyondAIME 31.3

这个结果有个有趣点：SFT 阶段 ReGFT 并非在所有外部集都压倒性领先，但进入 RL 后优势会放大，说明它的关键价值在于“为 RL 提供更可学习的起点”。

再看采样规模（response size 16 vs 64）：

- 纯 DAPO：16 → 64 有明显提升
- ReFT + DAPO：16 → 64 也提升，但 Beyond-AIME 不占优
- ReGFT + DAPO：64 时三榜单最优（70.0 / 61.6 / 40.3）

结论是：增加采样预算有用，但“只加采样”不够；更强初始化 + 更大探索预算才是组合拳。

## 附录里最有价值的证据
作者比较了两种采样策略（每题各采 64 条）：

- 普通采样可解题比例：68.58%
- reference-guided 采样可解题比例：70.82%
- 其中有 5.85% 题目是“仅 reference-guided 能解”
- 也有 3.61% 题目是“仅普通采样能解”

这说明两种采样具有互补性，因此 ReGFT 在训练时混合两类正确轨迹是合理的。

## Prompt 设计（工程可迁移）
论文给了两个模板思想：

- 普通解题：要求 step-by-step，最终答案放 $\boxed{}$
- 参考引导解题：给 Question + partial reference solution，强调“你必须自己推导，可参考提示思路”

这个提示词设计的重点是“允许借鉴，不允许照抄”，本质在于保持 on-policy 风格。

## 我的解读：这篇论文真正的新意
这篇工作的新意不在于提出更复杂的 RL 算法，而在于提出了一个实用的“前置能力扩展器”：

- 在 RL 前把“零奖励题”先变成“偶尔有奖励题”
- 用参考解做脚手架，但输出必须是模型自生成轨迹
- 让 RL 从“盲训”变成“有信号可学”

可以把它理解为：ReGFT 不是替代 RL，而是给 RL 做“点火系统”。

## 局限与风险
论文也承认了两类限制：

- 模型对复杂人类数学推理的吸收能力有限
- verifier 对开放式证明题有误判，可能把正确解当错（false negative）

所以即使有参考引导，性能距离 100% 仍有明显差距。

## 总结
如果你在做数学推理 RL，且遇到“难题全 0 奖励”导致训练停滞，ReGFT 给出了一条非常落地的路径：

- 先做 reference-guided 的轨迹合成与 SFT
- 再接强 RL（如 DAPO）
- 在相同 RL 预算下获得更快收敛和更高上限
- 在 pass@k 扩展上也更稳

它的核心价值不在“把答案告诉模型”，而在“让模型在自己可生成的分布中学会原本不会的题”。

> 本文参考自 [Learn Hard Problems During RL with Reference Guided Fine-tuning](https://arxiv.org/abs/2603.01223)