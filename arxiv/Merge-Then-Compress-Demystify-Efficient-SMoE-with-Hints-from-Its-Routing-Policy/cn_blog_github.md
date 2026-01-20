# Merge, Then Compress：用路由策略解开高效 SMoE 的压缩之谜

## 读完论文先给你结论
这篇文章解决的核心问题是： **SMoE 很强但太占内存且专家冗余严重** 。作者从路由策略里“挖线索”，提出 **先合并再压缩** 的两阶段方案 **MC-SMoE** 。它通过路由激活频率识别“关键专家”，把冗余专家合并进主导专家，再利用合并后权重的低秩特性继续压缩。实测在多任务上能做到 **最高 80% 内存节省、20% FLOPs 降低** ，性能几乎不掉。

---

## 背景：SMoE 为什么既香又难用
SMoE（Sparsely activated Mixture-of-Experts）把 Transformer 中的 FFN 替换成多个专家，每个 token 只激活少量专家，从而在 **计算量几乎不变** 的情况下扩大模型容量。但真实场景里有两个硬伤：

- **内存爆炸** ：每层一堆专家，参数量飙升。
- **专家冗余** ：路由会出现 representation collapse，很多专家长期不工作。

所以目标是： **压缩 SMoE 的参数，同时保住它的能力** 。

---

## 方法总览：MC-SMoE = Merge + Compress
核心思想： **路由策略里藏着“专家重要性”和“专家相似性”** 。先合并冗余专家，再在合并后的权重上做结构压缩。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Merge-Then-Compress-Demystify-Efficient-SMoE-with-Hints-from-Its-Routing-Policy/Figs/teaser.png)
> 图解：整体流程分三步：路由器按 token 选择专家；根据路由统计把专家分组并合并；合并后权重低维化，再做低秩 + 稀疏分解，进一步压缩。

---

## 关键观察 1：路由激活频率揭示专家重要性
不同任务下专家激活频率差异巨大，很多专家几乎不被用到。作者用路由日志统计每个专家被激活的频率，把高频专家当作 **dominant experts** 。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Merge-Then-Compress-Demystify-Efficient-SMoE-with-Hints-from-Its-Routing-Policy/Figs/usage.png)
> 图解：横轴为专家索引，纵轴为 MoE 层。颜色越深说明激活频率越高。可以看到很多专家几乎不激活，说明冗余严重，且不同任务的分布差异很大。

---

## 关键观察 2：路由输出可衡量专家相似性
作者不用参数相似度，而是用 **路由输出 logits 的相似度** 来判断专家是否冗余。公式如下：

$$
\texttt{Sim}(E_i, E_j) = \texttt{cosine}(H_{i,*}, H_{j,*})
$$

其中 $H = W_r X^T$ 是路由器 logits，行向量 $H_{i,*}$ 代表专家 $E_i$ 对输入的响应分布。

这种相似性更贴近“实际被分配到的样本集合”，比直接比权重有效。

---

## M-SMoE：基于路由的专家合并
合并流程分三步：

1. **专家排列对齐** ：不同专家初始化不同，需要先做 permutation alignment，避免错位融合。
2. **确定主导专家** ：按激活频率选出 dominant experts。
3. **分组并加权合并** ：每个非主导专家挂靠到与其最相似的主导专家，再做频率加权平均。

合并公式：

$$
E_{\text{merged}} = \frac{\sum_{i=1}^k \alpha_i E_i}{\sum_{i=1}^k \alpha_i}
$$

这里 $\alpha_i$ 是激活频率，能自然压制长期不工作的专家。

---

## 关键发现：合并后的权重更低秩
合并后权重出现明显低维趋势，用 stable-rank 衡量：

$$
\texttt{stable-rank}(\sigma) = \frac{\sum_i \sigma_i^2}{\max \sigma_i^2}
$$

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Merge-Then-Compress-Demystify-Efficient-SMoE-with-Hints-from-Its-Routing-Policy/Figs/stable-rank.png)
> 图解：图中多数层的 stable-rank 变化为负，说明合并后权重更低秩，适合进一步做低秩分解与稀疏化。

---

## MC-SMoE：合并后继续压缩
作者在合并后做 **低秩 + 稀疏** 分解：

$$
W \approx U V + S
$$

- $U V$ 是低秩部分，$r \ll \min(d_1, d_2)$
- $S$ 是残差稀疏项，按重要性剪掉整列

重要性得分：

$$
\mathcal{I}(s_{i,j}) = |s_{i,j} \cdot \nabla_{s_{i,j}} \mathcal{L}|
$$

这样能让每一层自适应地保留最关键的结构。

---

## 实验效果：更小但几乎不掉分
论文在 8 个 NLP 任务上做了系统对比。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Merge-Then-Compress-Demystify-Efficient-SMoE-with-Hints-from-Its-Routing-Policy/Figs/intro-teaser.png)
> 图解：在 COPA 任务上，MC-SMoE 以接近原模型的准确率换取最高 80% 内存节省，说明合并 + 压缩不会明显伤性能。

更多细节见主表（Switch-Base-32）：

- **M-SMoE** ：可减小 60% 内存，部分任务还略涨分。
- **MC-SMoE** ：可减小 80% 内存 + 20% FLOPs，性能下降 <1%。

---

## 补充实验：方法设计的有效性
### 1. 自适应合并比例优于固定比例
![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Merge-Then-Compress-Demystify-Efficient-SMoE-with-Hints-from-Its-Routing-Policy/Figs/grouping-results/expert-grouping-copa.png)
> 图解：不同任务下专家聚类结构不同，自适应合并可以更好匹配每层冗余程度。

### 2. 路由 logits 相似度最稳
![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Merge-Then-Compress-Demystify-Efficient-SMoE-with-Hints-from-Its-Routing-Policy/Figs/grouping-results/expert-grouping-squad.png)
> 图解：路由 logits 直接反映“专家接收到的 token 分布”，比权重或特征相似更可靠。

---

## 附录关键点：延迟问题与工程化启示
论文指出： **即便专家数量下降，路由器输出维度不变，实际延迟仍然可能偏高** 。如果能进一步剪裁路由器输出通道，延迟还能进一步下降。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Merge-Then-Compress-Demystify-Efficient-SMoE-with-Hints-from-Its-Routing-Policy/Figs/dominant-teaser.png)
> 图解：后层的主导专家更容易压缩，说明越深层越可能冗余，给部署优化留出空间。

---

## 总结与启发
这篇工作给出一个非常实用的 SMoE 精简路径：

- **用路由统计找冗余** ，而不是只靠权重。
- **先合并再压缩** ，避免直接压缩造成性能崩坏。
- **合并后低秩性增强** 是关键突破点。

如果你在做 MoE 相关落地（比如推理部署、移动端推理、边缘设备），MC-SMoE 的思路值得直接借鉴。

---

> 本文参考自 [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/abs/2310.01334)