# REAP the Experts：为什么 One-shot MoE 压缩中“剪枝”胜过“合并”

这篇论文聚焦一个很现实的问题： **SMoE（Sparsely-activated Mixture-of-Experts）模型很快、很强，但太大** 。研究者想压缩专家（experts），常见做法是 **剪枝（pruning）** 或 **合并（merging）** 。过去合并在 MC 题上效果不错，但在生成任务上到底谁更强？本文用理论 + 大规模实验证明： **在生成类任务上，剪枝明显更稳、更强**，并提出了一个新剪枝准则 REAP。

---

## 1. 论文核心问题：合并到底损失了什么？

SMoE 层输出是专家输出的加权和：

$$
h(x) = \sum_{k=1}^{K} g_k(x)\, f_k(x)
$$

其中 $g_k(x)$ 是 router gate，$f_k(x)$ 是专家函数。

当把两个专家 $(f_i,f_j)$ 压成一个专家时：

- **剪枝**：删除 $f_j$，router 仍然对剩余专家独立控制  
- **合并**：把 $f_i,f_j$ 合成一个 $\tilde f$，并把 gate 绑成 $g_i(x)+g_j(x)$  

论文指出合并会导致一个本质问题： **router 原本是“动态混合”，合并后只能“静态平均”** 。

设混合比例：

$$
r(x) = \frac{g_i(x)}{g_i(x)+g_j(x)}
$$

合并相当于用常数 $\alpha$ 去逼近一个随 $x$ 变化的混合。结果是不可消除的误差：

$$
\mathbb{E}\!\left[(g_i{+}g_j)^2\right] \cdot \mathrm{Var}[r(x)] \cdot \|f_i(x)-f_j(x)\|^2
$$

核心结论：  
只要 router 的混合比例在变（$\mathrm{Var}[r(x)]>0$），且两个专家不完全一样，就一定有不可消除的误差。

---

## 2. 直观图像：合并 = “功能子空间塌缩”

论文在多个模型上做 PCA 可视化，揭示了合并对功能空间的毁伤：

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/Qwen3-30B-A3B-layer_0.png)
> 图解：Qwen3 早层的专家子空间。剪枝后点云分布仍覆盖原来的结构，合并后整体收缩到中心。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/Qwen3-30B-A3B-layer_47.png)
> 图解：Qwen3 晚层的专家子空间，合并的塌缩更严重，分布跨度甚至缩小 100 倍量级。

更进一步，附录里的 ERNIE / Mixtral / Llama 结果也一致：

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/ERNIE-4.5-21B-A3B-PT-layer_1.png)
> 图解：ERNIE 早层，剪枝保留流形几何结构，合并将分布压向中心。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/Llama-4-Scout-17B-16E-Instruct-layer_47.png)
> 图解：Llama-4 晚层，合并后专家几乎全部坍缩成一个小团。

结论非常清晰： **合并会“绑死 router 的控制自由度”，功能空间被硬性压缩** 。

---

## 3. REAP：一个更合理的剪枝准则

传统剪枝只看频率（expert 被选中的次数），但这不代表“贡献大”。

REAP 的核心思路是： **衡量专家的“输出贡献量”** ，即 gate 强度乘上输出幅度。

定义 saliency：

$$
S_j = \frac{1}{|\mathcal{X}_j|} \sum_{x \in \mathcal{X}_j} g_j(x)\, \|f_j(x)\|_2
$$

$\mathcal{X}_j$ 是 expert 被选中的 token 集合。

直观解释：  
一个专家如果 **gate 小** 或 **输出很弱**，即使频率高，也可能对整体贡献很小，应该先剪掉。

---

## 4. 实验设计：从 20B 到 1T 的大规模对比

实验覆盖 6 个模型，从 21B 到 1T：

- ERNIE-4.5-21B  
- Qwen3-30B  
- Mixtral-8x7B  
- GLM-4.5-Air  
- Qwen3-Coder-480B  
- Kimi-K2  

任务覆盖：

- MC 多选题  
- 生成任务：代码、数学推理、创意写作  
- 工具调用 / SWE-Bench / Agentic coding  

压缩比例：25% / 50%

---

## 5. 关键结果：生成任务上剪枝全面胜出

### 5.1 总体表现

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/combined-all-tasks_qwen_and_glm.png)
> 图解：Qwen3 和 GLM 在多任务上的对比。50% 压缩时，REAP 的降幅明显更小，而合并在生成类任务上崩得更厉害。

### 5.2 生成质量差异分析

论文额外分析了合并后生成文本的质量：

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/ngram_diversity.png)
> 图解：N-Gram 多样性。合并模型输出更重复，REAP 更接近原模型。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/cross_perplexity.png)
> 图解：交叉困惑度。合并模型的输出偏离 baseline 更严重。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/jsd_small.png)
> 图解：Logit 分布的 JSD 随输出步数增长，合并模型更快偏离原模型。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/combined_weight_dist.png)
> 图解：专家之间权重差距巨大，表明直接合并在权重空间上很难“对齐”。

---

## 6. 扩展讨论：合并困难的结构性原因

论文进一步指出：

- 合并会形成 **“巨大簇 + 大量单例”**  
- 这会导致 **超大簇内部差异极大**，合并难度指数上升  

![Figure 10](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/restricted_cluster_size_m_smoe.png)
> 图解：限制聚类最大大小时，生成质量急剧下降，说明聚类难以兼顾“压缩率”和“功能一致性”。

---

## 7. Domain-specific 校准很关键

压缩前校准数据很重要，尤其是生成任务：

![Figure 11](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/dataset-ablation-all-models.png)
> 图解：用通用数据（c4）校准时，代码生成质量严重崩溃，很多模型甚至输出不可用。

![Figure 12](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/REAP-the-Experts-Why-Pruning-Prevails-for-One-Shot-MoE-compression/fig/bar-combined-dataset-all-tasks-Qwen3-30B-A3B.png)
> 图解：混合校准数据在 25% 压缩还能撑住，但 50% 压缩时仍远不如领域数据。

---

## 8. 一句话总结

- **合并会导致“功能子空间塌缩”，这是结构性损失**  
- **REAP 用“router gate + 输出幅度”衡量贡献，剪枝更精准**  
- **生成任务上，剪枝明显优于合并，尤其是 50% 压缩**

---

## 9. 你可以如何理解 REAP 的价值

- 对于生成任务，router 的“动态控制能力”就是模型的核心  
- 合并相当于把“动态行为”换成“静态平均”，必然失真  
- REAP 不改动 router 的控制结构，只删掉“贡献最小的专家”  

一句话： **它不是把专家“揉成一坨”，而是把“弱的剔除，强的保留”** 。

---

## 10. 可复现性与资源

论文开源了代码和部分压缩模型：
- GitHub: https://github.com/CerebrasResearch/reap  
- HuggingFace: https://hf.co/cerebras/Qwen3-Coder-REAP-363B-A35B-FP8  

---

## 结论与展望

REAP 证明了： **在 MoE 压缩的生成任务场景中，保留 router 的独立控制是第一原则** 。  
这意味着未来的压缩方向，应该更关注 **router-专家协同机制**，而不是简单的参数融合。

---

本文参考自 REAP the Experts: Why Pruning Prevails for One-Shot MoE compression  
原文链接：https://www.arxiv.org/pdf/2510.13999

如果你想进一步探索，可以考虑：
1. 试着用 REAP 在你自己的 MoE 模型上做 25% / 50% 剪枝对比  
2. 对比 REAP 与 EAN 在不同领域数据校准下的稳定性  
3. 研究是否能把 REAP 和量化、LoRA 等压缩方法组合使用