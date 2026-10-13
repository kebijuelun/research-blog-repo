# Surfing the Activation Wave：用熵和几何选对投机解码层

Speculative Decoding（投机解码）是当下 LLM 推理加速的主流路线，EAGLE3 这类方法需要从目标大模型中挑几层 Hidden States 喂给草稿模型——但 **挑哪几层，目前基本靠“拍脑袋”**：EAGLE2 用最后一层，EAGLE3 用固定的 $[2, N//2, N-3]$ 位置规则。这篇来自 Qualcomm AI Research 和 Stanford 的工作指出，选层本质上是一个组合优化问题，并给出了一套有理论依据的解法：用 **矩阵熵** 衡量层集合的信息量、用 **CKA** 衡量层间的冗余度，提出 ELSI / ELSI-C 两个选层算法。最硬的结果是：在 Qwen3-4B 上，ELSI 选出的 3 层组合比 EAGLE3 默认规则平均提升 **1.6%–2.2%** 的接受长度；更令人意外的是，**只用 ELSI 选出的 1 层，就能打平甚至超过 EAGLE3 的 3 层启发式**。

## 提出问题：三层组合千千万，凭什么总是 $[2, N//2, N-3]$？

先快速回忆一下 EAGLE3 式投机解码的 Conditioning 方式。设目标模型有 $N$ 层，第 $l$ 层输出的 Hidden States 为 $h_l \in \mathbb{R}^{T \times d}$。选定层集合 $\mathcal{L}$ 后，把这些层的 Hidden States 在特征维上拼接：

$$
\bar{h}_{\mathcal{L}} = \operatorname{concat}(\{h_l\}_{l \in \mathcal{L}}) \in \mathbb{R}^{T \times |\mathcal{L}| d}
$$

再经过一个下投影层 $W_{\text{down}} \in \mathbb{R}^{|\mathcal{L}| d \times d}$ 压缩回 $d$ 维，喂给草稿模型：

$$
U_{\mathcal{L}} = \bar{h}_{\mathcal{L}} W_{\text{down}}
$$

可以看到，$\mathcal{L}$ 是整个信息通路的第一道 **瓶颈**：草稿模型能“看到”什么，完全由选了哪几层决定。

问题在于，选层是一个组合问题。一个 36 层的模型只选 3 层，就有 $\binom{36}{3} = 7140$ 种组合，逐一训练草稿模型做消融完全不现实。于是大家退而求其次用位置启发式：EAGLE2 取最后一层 $[N-1]$，EAGLE3 取 $[2, N//2, N-3]$（早、中、晚各采一层）。这些规则在实践中确实管用，但没人说得清 **为什么** 管用、什么时候会失效。

笔者认为，这篇文章的价值恰恰在于把“玄学”变成了“可测量的科学”：先用信息论和几何工具回答“哪些层组合理论上是好的”，再反过来解释现有启发式为什么碰巧有效。

## 分析问题：给隐藏层“画像”的两把尺子

要科学地选层，首先需要两个量化指标：一个衡量 **信息量**（这层值不值得选），一个衡量 **冗余度**（这层和已选的层是不是重复了）。

### 尺子一：矩阵熵——衡量信息量

直觉上，如果一层的 Hidden States 在特征空间里“摊开”得越均匀（有效维度越多），它携带的信息就越多。严格做法是看协方差矩阵的谱分布。对 $X \in \mathbb{R}^{T \times D}$，令 $K_X = XX^\top$，定义归一化谱：

$$
p_i(X) = \frac{\lambda_i(K_X)}{\operatorname{tr}(K_X)} = \frac{\sigma_i(X)^2}{\|X\|_F^2}
$$

其中 $\sigma_i(X)$ 是奇异值。Rényi-$\alpha$ 矩阵熵定义为：

$$
S_\alpha(X) = \frac{1}{1-\alpha} \log \sum_i p_i(X)^\alpha, \quad \alpha > 0, \alpha \neq 1
$$

取 $\alpha \to 1$ 的极限就是 Shannon 矩阵熵：

$$
S_1(X) = -\sum_i p_i(X) \log p_i(X)
$$

对层集合 $\mathcal{L}$，直接对拼接后的表示 $\bar{h}_{\mathcal{L}}$ 计算 $S_1$。这个设计有一个细节值得注意：相比前人基于矩阵的 Joint Entropy 定义，对拼接特征算熵恰好对应 EAGLE3 里 $W_{\text{down}}$ 对拼接特征做 **线性组合** 的实际结构，指标和下游用法是对齐的。

### 尺子二：CKA——衡量几何冗余

CKA（Centered Kernel Alignment，中心核对齐）是衡量两组表示相似度的经典工具。对表示 $X$ 和 $Y$，其 Gram 矩阵为 $K = XX^\top$、$L = YY^\top$，则：

$$
\operatorname{CKA}(X, Y) = \frac{\langle HKH, HLH \rangle_F}{\|HKH\|_F \, \|HLH\|_F}, \quad H = I - \frac{1}{T} \mathbf{1}\mathbf{1}^\top
$$

CKA 越接近 1，两组表示越相似。进一步，令 $M_{ij} = \operatorname{CKA}(h_i, h_j)$ 为逐层两两 CKA 矩阵，定义相邻层之间的 **转变分数（transition score）**：

$$
\delta_\ell = \|M_{\ell+1,:} - M_{\ell,:}\|_2, \quad \ell = 0, \dots, N-2
$$

$\delta_\ell$ 出现尖峰，意味着第 $\ell$ 层到第 $\ell+1$ 层之间表示发生了剧烈变化——也就是一个“表示隔间（compartment）”的边界。这个公式还可以自然推广到拼接层上，后面 ELSI-C 会用到。

### 发现一：激活波——层的信息量沿深度剧烈起伏

作者用 200 条 ShareGPT 样本做校准，统计了两个模型每一层的 $S_1$。

![激活波](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/entropy_filtered_per_prompt_curves_ShareGPT_default_for_intro.png)

> 图解：Qwen3-4B-Instruct 上每条 prompt 的逐层 Shannon 熵曲线，横轴是层索引（0–35），纵轴是谱熵 $S_1$（单位 nats）。每条细线是一条 prompt。可以看到熵沿深度呈现明显的“波浪”形态——作者称之为 **激活波（Activation Wave）**：早期层（0–6）和末尾层（34 附近）熵值冲高，中层（7–28）相对平缓。更关键的是，早期层和末尾层的 **跨 prompt 方差极大**（细线散得很开），这意味着不加处理的话，最优层排序会随 prompt 变化，无法固定使用。

Token 过滤后的逐层平均熵曲线更能看清两个模型的差异：

![逐层熵](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/entropy_per_layer_combined_shannon_token_filter_ff30.png)

> 图解：两个模型逐层 Shannon 熵 $S_1$ 对比，横轴层索引，纵轴熵（nats）。蓝色 Qwen3-4B 呈“高-低-高”形态，在第 28 层附近达到峰值（约 0.47 nats），最后一层骤降到接近 0；红色 Llama-3.1-8B 第 0 层熵极高（约 0.78）后骤降，随后一路爬升，在第 27 层左右达到峰值。两条曲线形状截然不同，说明 **不存在跨模型通用的固定选层规则**。基于 Llama 第一层熵虚高的现象，作者还加了一条工程约束：单层选择除了要高熵，层索引还必须在中点之后。

### 发现二：层拼接的熵增益高度不均匀

单层熵高不代表组合起来也好——两层如果信息重叠，拼接后的熵并不会涨多少。作者把按单层熵排名的 Top-10 层作为“种子层”，分别与所有其他层两两拼接，看联合熵 $S_1(\{H_i, H_j\})$：

![Qwen 层对熵](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/topk_joint_entropy_s1_qwen3-4b_token_filter_ff30.png)

> 图解：Qwen3-4B 的层对联合熵热力图。纵轴是 Top-10 种子层 $i$，横轴是搭配的伙伴层 $j$，颜色越黄联合熵越高、越紫越低。可以明显看出增益不均匀：比如种子层 L23 与中层（6–22）搭配时熵值反而塌陷成深紫色（信息高度重叠），而几乎所有种子层与最后一层 L35 搭配都飙出亮黄色。不同的种子层偏好的伙伴层也不同——这正是“必须联合优化层集合”的直接证据。

![Llama 层对熵](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/topk_joint_entropy_s1_llama-3.1-8b_token_filter_ff30.png)

> 图解：Llama-3.1-8B 的同款热力图，整体趋势与 Qwen 不同：深层区域作为伙伴层普遍增益更高，与其逐层熵“越往后越高”的曲线一致。两张图放在一起看，再次印证了选层规则必须因模型而异。

### 发现三：CKA 揭示深度方向的“表示隔间”

把两两层间 CKA 画成热力图，并叠加用转变分数 $\delta_\ell$ 检测出的隔间边界（黄色虚线）：

![Qwen CKA](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/cka_transition_qwen3-4b_default.png)

> 图解：Qwen3-4B 的两两层间 CKA 热力图，横纵轴均为层索引，颜色越红相似度越高。黄色虚线是检测到的转变边界，大约位于第 1、5、16、33 层附近，把网络切成了若干隔间，中部有两个大隔间。Qwen 的转变边界较多，说明其表示沿深度变化更剧烈、层次分工更细。

![Llama CKA](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/cka_transition_llama-3.1-8b_default.png)

> 图解：Llama-3.1-8B 的同款图。对比非常鲜明：除了第一层和最后两层，整个中间区域几乎是一整块深红色（CKA 接近 1），只在第 1 层和第 30 层附近有边界。也就是说 Llama 的中间层彼此高度可互换。

这个发现顺手解释了一个悬而未决的问题：**为什么 EAGLE3 的 $[2, N//2, N-3]$ 规则碰巧有效？** 因为它选的早、中、晚三层大概率落在不同的 CKA 隔间里，天然比 EAGLE2 只用最后一层冗余更低；但它仍然可能踩在同一隔间内造成信息重叠，所以不是最优。笔者认为这种“用几何结构反向解释既有工程经验”的视角，是全文最有洞察力的部分。

### 插曲：Token 熵过滤——让统计量更稳

回到发现一里的问题：早期层和末尾层的熵随 prompt 剧烈波动，会导致层排序不稳定，而选出的层集合在训练和推理时必须固定。作者的解法很直接：很多 token（比如常见的接续词）预测起来毫无悬念，它们的 Hidden States 对区分各层信息量没有贡献，不如丢掉。对每个 token 位置 $t$，计算预测熵：

$$
\mathcal{H}_t = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)
$$

然后只保留预测熵最高的前 $\rho$ 比例的 token：

$$
\Omega = \operatorname{TopFrac}(\{\mathcal{H}_t\}_{t=1}^T, \rho), \quad \rho \in (0, 1]
$$

实验中 $\rho = 0.3$，即只保留 30%“最难预测”的 token 参与统计。效果：跨 prompt 熵方差在早期层（0–6）降低 **48.4%**、中层（7–28）降低 **45.8%**、晚期层（29–35）降低 **35.6%**，同时激活波的整体形状保持不变。

![Token 过滤](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/20260721_entropy_filtered_per_prompt_curves_token_filter_clean_ShareGPT_ff30.png)

> 图解：Token 熵过滤后（$\rho=0.30$）的逐 prompt 熵曲线，坐标轴与前面的激活波图相同。对比过滤前可以看到：曲线的“浪形”完全保留，但各条 prompt 细线明显收拢——尤其早期层（0–6）的离散度大幅下降，选层统计量因此更稳定。

## 解决问题：ELSI 与 ELSI-C 选层算法

有了“信息量 + 冗余度”两把尺子和稳定的统计量，接下来就是把它们变成可执行的选层算法。整体流水线分三步：

1. **Token 熵过滤**：按上节方法保留高预测熵 token 的 Hidden States $h_l^{(\Omega)}$；
2. **分块平均**：把 token 轴切成不重叠的块（块大小 $B=32$）并做平均，得到每层代表 $Z_l \in \mathbb{R}^{B \times d}$；
3. **选层搜索**：ELSI（纯熵驱动）或 ELSI-C（熵初始化 + CKA 多样性扩展）。

### ELSI：熵驱动的 Beam Search

穷举 $\binom{N}{m}$ 种组合不可行，ELSI（**E**ntropy-based **L**ayer **S**et **I**teration）用 Beam Search 近似求解，以拼接表示的矩阵熵作为扩展准则。记 $\bar{Z}_{\mathcal{L}}$ 为层集合 $\mathcal{L}$ 的特征维拼接，算法流程为：

- **[S1] 种子初始化**：取单层熵 $S_1(Z_l)$ 最高的 $K_{\text{beam}}$ 层构成初始 Beam $\mathcal{B}_1$；
- **[S2] 逐层扩展**：对 Beam 中每个集合 $\mathcal{L}$，尝试加入所有未选层 $l$，保留使 $S_1(\bar{Z}_{\mathcal{L} \cup \{l\}})$ 最大的前 $K_{\text{expand}}$ 个扩展；
- **[S3] 去重**：不同父集合可能扩展出相同的子集合，去重；
- **[S4] Beam 剪枝**：按集合熵保留前 $K_{\text{beam}}$ 个候选进入下一层；
- **[S5] 输出**：到达目标大小 $m$ 后，返回熵最大的集合 $\mathcal{L}^\star = \arg\max_{\mathcal{L} \in \mathcal{B}_m} S_1(\bar{Z}_{\mathcal{L}})$。

Beam 搜索过程的完整可视化如下：

![ELSI Beam](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/entropy_beam_expansion_s1_qwen3-4b_token_filter_ff30_pure_entropy_top_k=10.png)

> 图解：Qwen3-4B 上 ELSI 的 Beam 展开树（$K_{\text{beam}}=10$），每一列是 Beam 的一层深度，方框内是该候选的层集合与熵值 $S_1$，颜色越红熵越高。每一列的第 1 名就是对应 $m$ 的最优解：单层最优是 $\{L28\}$（0.4687），双层最优是 $\{L23, L35\}$（0.6253），三层最优是 $\{L5, L23, L35\}$（0.6272）。注意双层到三层的熵增益已经很小（0.6253 → 0.6272），边际收益递减清晰可见。

### ELSI-C：用 CKA 转变分数做多样性过滤

ELSI 只看熵，有个隐患：高熵扩展仍可能引入与已选层几何上冗余的层。ELSI-C 在 ELSI 框架上加了一道“多样性闸门”：

- **[C1]** 与 ELSI 相同，用单层熵初始化 Beam；
- **[C2] 条件 CKA 矩阵**：对 Beam 中每个集合 $\mathcal{L}$ 和候选层对 $(i, j)$，计算 $M_{\mathcal{L}}(i, j) = \operatorname{CKA}(\bar{Z}_{\mathcal{L} \cup \{i\}}, \bar{Z}_{\mathcal{L} \cup \{j\}})$；
- **[C3] 转变分数筛选**：计算每个候选层的转变分数 $\delta_{\mathcal{L}}(r) = \|M_{\mathcal{L}}(r,:) - M_{\mathcal{L}}(r-1,:)\|_2$，只保留分数最高的 $K_{\text{pool}}$ 个候选层进入扩展池——直觉上，转变分数高的层位于隔间边界附近，带来的新信息最多；
- **[C4]** 去重后仍按熵剪枝；
- **[C5]** 最终同样按熵选出最优集合。

这个设计的聪明之处在于：CKA 只负责 **缩小候选池**（保证多样性），熵仍负责 **最终打分**（保证信息量），两个信号各司其职，避免了多目标加权调参的麻烦。

## 实验验证：选对层，比多选层更重要

### 实验设置

- **目标模型**：Qwen3-4B-Instruct-2507（36 层）与 Llama-3.1-8B-Instruct（32 层）；
- **草稿模型训练**：EAGLE 式草稿模型，用 ShareGPT + UltraChat 生成的蒸馏数据，在 8 张 H100 上训练；
- **评测**：SpecBench 基准，覆盖 Coding、Extraction、Humanities、QA、RAG、Roleplay、STEM、Summarization、Translation、Writing、Math/Reasoning 等类别，报告平均接受 token 数 $\tau$（每次前向传播平均接受的 token 数，同等条件下直接正比于吞吐）；
- **变量消融**：草稿树节点数 30 / 60，采样温度 T=0.0（贪心）/ T=1.0；
- **对照组**：EAGLE2（$[N-1]$）、EAGLE3（$[2, N//2, N-3]$）、低熵负对照（故意选熵最低的层 $\{0, 1, 35\}$）。

### 各算法选出了哪些层

先看选层结果和对应熵值，这是理解下游表现的钥匙：

| 方法 | Qwen3-4B 选层 | 熵 | Llama-3.1-8B 选层 | 熵 |
|---|---|---|---|---|
| **m=1** | | | | |
| EAGLE2 | $\{L35\}$ | 0.044 | $\{L31\}$ | 0.530 |
| ELSI | $\{L28\}$ | 0.469 | $\{L26\}$ | 0.718 |
| ELSI-C | $\{L28\}$ | 0.469 | $\{L26\}$ | 0.718 |
| **m=2** | | | | |
| ELSI | $\{L23, L35\}$ | 0.625 | $\{L22, L31\}$ | 0.824 |
| ELSI-C | $\{L25, L35\}$ | 0.588 | $\{L25, L31\}$ | 0.797 |
| **m=3** | | | | |
| EAGLE3 | $\{L2, L18, L33\}$ | 0.441 | $\{L2, L16, L29\}$ | 0.687 |
| ELSI | $\{L5, L23, L35\}$ | 0.627 | $\{L1, L22, L31\}$ | 0.910 |
| ELSI-C | $\{L6, L28, L35\}$ | 0.593 | $\{L1, L25, L31\}$ | 0.889 |

两个值得盯住的数字：EAGLE2 在 Qwen 上选的最后一层 L35，熵只有 **0.044**，而 ELSI 单层 L28 的熵是 **0.469**——信息量差了 10 倍；同样选 3 层，ELSI 集合的熵（0.627）也明显高于 EAGLE3（0.441）。指标上的优势是否能转化为下游收益？看 SpecBench。

### Qwen3-4B：ELSI 全面领先

下表为 Qwen3-4B 在四组配置下的平均接受 token 数 $\tau$（每块内加粗为最优）：

**Draft 节点 = 30，T = 0.0**

| 方法 | Coding | Extr. | Human. | QA | RAG | Role. | STEM | Summ. | Transl. | Writing | M/R | AVG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EAGLE2 | 4.73 | 4.21 | 3.69 | 3.90 | 3.94 | 3.67 | 3.98 | 3.38 | 3.39 | 3.95 | 4.91 | 4.00 |
| ELSI (m=1) | 4.83 | 4.79 | 3.80 | 3.97 | 4.17 | 3.63 | 4.14 | 3.56 | **3.85** | 4.06 | 5.21 | 4.24 |
| ELSI (m=2) | 4.90 | 4.61 | 3.86 | 4.05 | **4.20** | 3.75 | 4.22 | 3.58 | 3.83 | 4.17 | **5.26** | 4.28 |
| ELSI-C (m=2) | **4.93** | 4.77 | 3.82 | 4.07 | 4.18 | 3.76 | 4.22 | 3.59 | 3.76 | 4.19 | 5.20 | 4.26 |
| Low-Entropy | 4.74 | 4.41 | 3.71 | 3.90 | 4.03 | 3.63 | 4.07 | 3.38 | 3.42 | 4.00 | 4.96 | 4.04 |
| EAGLE3 | 4.88 | 4.75 | 3.79 | 4.02 | 4.13 | 3.73 | 4.14 | 3.52 | 3.60 | 4.07 | 5.18 | 4.20 |
| ELSI (m=3) | 4.90 | **4.80** | **3.87** | **4.08** | 4.20 | **3.86** | **4.30** | **3.61** | 3.80 | **4.22** | 5.25 | **4.29** |
| ELSI-C (m=3) | 4.92 | 4.70 | 3.85 | 4.05 | 4.15 | 3.76 | 4.19 | 3.56 | 3.71 | 4.16 | 5.22 | 4.24 |

**Draft 节点 = 30，T = 1.0**

| 方法 | Coding | Extr. | Human. | QA | RAG | Role. | STEM | Summ. | Transl. | Writing | M/R | AVG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EAGLE2 | 4.51 | 4.41 | 3.37 | 3.59 | 3.76 | 3.44 | 3.91 | 3.26 | 3.35 | 3.52 | 4.85 | 3.86 |
| ELSI (m=1) | 4.55 | **5.03** | 3.48 | 3.60 | 3.98 | 3.31 | 3.95 | 3.41 | **3.81** | 3.63 | 5.06 | 4.06 |
| ELSI (m=2) | **4.68** | 4.88 | **3.51** | 3.72 | **4.02** | 3.44 | 4.04 | 3.45 | 3.79 | 3.75 | 5.13 | 4.11 |
| ELSI-C (m=2) | 4.68 | 4.82 | 3.49 | 3.74 | 3.97 | 3.42 | 3.94 | **3.47** | 3.74 | 3.70 | 5.13 | 4.09 |
| Low-Entropy | 4.50 | 4.56 | 3.39 | 3.66 | 3.86 | 3.31 | 3.87 | 3.26 | 3.37 | 3.62 | 4.84 | 3.89 |
| EAGLE3 | 4.52 | 4.81 | 3.42 | 3.65 | 3.96 | 3.37 | 3.93 | 3.38 | 3.59 | 3.62 | 5.10 | 4.02 |
| ELSI (m=3) | 4.67 | 4.96 | 3.50 | **3.74** | 4.01 | 3.34 | **4.06** | 3.45 | 3.74 | **3.91** | **5.17** | **4.11** |
| ELSI-C (m=3) | 4.48 | 4.85 | 3.50 | 3.74 | 4.00 | **3.50** | 3.89 | 3.44 | 3.71 | 3.70 | 5.11 | 4.08 |

**Draft 节点 = 60，T = 0.0**

| 方法 | Coding | Extr. | Human. | QA | RAG | Role. | STEM | Summ. | Transl. | Writing | M/R | AVG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EAGLE2 | 5.03 | 4.54 | 4.04 | 4.19 | 4.22 | 3.89 | 4.36 | 3.61 | 3.57 | 4.27 | 5.23 | 4.28 |
| ELSI (m=1) | 5.17 | **4.97** | 4.07 | 4.26 | 4.45 | 4.01 | 4.40 | 3.81 | 4.02 | 4.37 | 5.49 | 4.50 |
| ELSI (m=2) | 5.23 | 4.85 | 4.17 | 4.37 | 4.48 | **4.21** | 4.49 | 3.82 | 3.97 | **4.52** | **5.56** | 4.55 |
| ELSI-C (m=2) | **5.23** | 4.94 | **4.18** | 4.37 | 4.43 | 4.04 | 4.55 | 3.83 | 3.96 | 4.48 | 5.54 | 4.54 |
| Low-Entropy | 5.09 | 4.65 | 3.98 | 4.25 | 4.30 | 4.01 | 4.39 | 3.64 | 3.59 | 4.25 | 5.28 | 4.33 |
| EAGLE3 | 5.16 | 4.91 | 4.13 | 4.32 | 4.42 | 4.10 | 4.50 | 3.79 | 3.83 | 4.38 | 5.46 | 4.48 |
| ELSI (m=3) | 5.23 | 4.91 | 4.15 | **4.40** | **4.49** | 4.18 | **4.61** | **3.87** | **4.02** | 4.51 | 5.54 | **4.57** |
| ELSI-C (m=3) | 5.19 | 4.91 | 4.14 | 4.32 | 4.46 | 4.16 | 4.49 | 3.83 | 3.92 | 4.47 | 5.52 | 4.52 |

**Draft 节点 = 60，T = 1.0**

| 方法 | Coding | Extr. | Human. | QA | RAG | Role. | STEM | Summ. | Transl. | Writing | M/R | AVG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EAGLE2 | 4.71 | 4.76 | 3.70 | 3.87 | 4.04 | 3.57 | 4.17 | 3.48 | 3.54 | 3.73 | 5.16 | 4.12 |
| ELSI (m=1) | 4.81 | 5.04 | 3.74 | 3.95 | 4.16 | 3.72 | 4.31 | 3.63 | 3.98 | 3.79 | 5.41 | 4.31 |
| ELSI (m=2) | 4.88 | 5.03 | **3.81** | 3.99 | 4.23 | 3.78 | 4.30 | 3.69 | 3.93 | 3.94 | **5.49** | 4.36 |
| ELSI-C (m=2) | 4.95 | **5.07** | 3.73 | 4.05 | 4.24 | 3.52 | 4.35 | 3.69 | **4.01** | 3.98 | 5.43 | 4.36 |
| Low-Entropy | 4.84 | 4.57 | 3.66 | 3.85 | 4.10 | 3.62 | 4.17 | 3.51 | 3.54 | 3.87 | 5.22 | 4.14 |
| EAGLE3 | **5.03** | 5.01 | 3.71 | 3.99 | 4.25 | 3.63 | 4.23 | 3.58 | 3.83 | 3.88 | 5.38 | 4.30 |
| ELSI (m=3) | 5.00 | 5.04 | 3.77 | **4.06** | **4.28** | 3.64 | **4.37** | **3.73** | 3.93 | **4.03** | 5.43 | **4.37** |
| ELSI-C (m=3) | 4.85 | 5.06 | 3.72 | 4.02 | 4.26 | **3.78** | 4.25 | 3.69 | 3.90 | 3.90 | 5.39 | 4.33 |

从这四张表中可以提炼出四个结论：

- **ELSI (m=3) 是所有配置下平均表现最好的方法**：比 EAGLE3 提升 1.6%–2.2%（平均 $+0.085\,\tau$），比 EAGLE2 提升 6.1%–7.3%；
- **单层就能打三层**：ELSI (m=1) 在四组配置下平均都追平或超过 EAGLE3（$+0.01 \sim +0.04\,\tau$）。一个精心挑选的层，抵得上默认的三层启发式——而且更少的层意味着下投影矩阵 $|\mathcal{L}|d \times d$ 更小，在显存受限设备上还有额外吞吐红利；
- **ELSI 在 EAGLE3 最弱的类别上收益最大**：Translation 提升 2.6%–5.6%、Summarization 提升 2.1%–4.2%，在本身较强的 STEM 上也能再涨 2.4%–3.9%；
- **跨温度、跨树大小收益一致**，不挑部署场景。

低熵负对照组（故意选 $\{0, 1, 35\}$）在所有配置下几乎都是垫底，反过来证明了“高信息量 + 低冗余”确实是接受率提升的因果来源，而不是别的因素。

### Llama-3.1-8B：收益变小，但这本身就是发现

Llama-3.1-8B 上的结果（展示部分类别，AVG 为全类别加权平均）：

| 方法 | Nodes | T | Coding | QA | Role. | Writing | M/R | AVG |
|---|---|---|---|---|---|---|---|---|
| EAGLE2 | 30 | 0.0 | 5.36 | 4.09 | 4.15 | 4.41 | 4.89 | 4.40 |
| ELSI (m=1) | 30 | 0.0 | 5.51 | 4.19 | 4.25 | 4.54 | 5.15 | 4.58 |
| ELSI (m=2) | 30 | 0.0 | 5.51 | 4.22 | 4.26 | 4.57 | 5.14 | 4.58 |
| ELSI-C (m=2) | 30 | 0.0 | 5.49 | 4.19 | 4.13 | 4.54 | 5.08 | 4.53 |
| EAGLE3 | 30 | 0.0 | 5.54 | **4.24** | 4.25 | 4.62 | **5.22** | **4.64** |
| ELSI (m=3) | 30 | 0.0 | **5.55** | 4.22 | **4.30** | **4.62** | 5.14 | 4.59 |
| ELSI-C (m=3) | 30 | 0.0 | 5.47 | 4.15 | 4.24 | 4.51 | 5.10 | 4.52 |
| EAGLE2 | 30 | 1.0 | 4.59 | 3.24 | 3.06 | 3.36 | 4.01 | 3.64 |
| ELSI (m=1) | 30 | 1.0 | 4.53 | 3.27 | **3.47** | 3.76 | 4.34 | 3.82 |
| ELSI (m=2) | 30 | 1.0 | 4.65 | 3.20 | 3.38 | 3.82 | 4.32 | 3.81 |
| ELSI-C (m=2) | 30 | 1.0 | 4.77 | 3.29 | 3.22 | 3.79 | 4.32 | 3.81 |
| EAGLE3 | 30 | 1.0 | 4.67 | 3.32 | 3.21 | **3.89** | **4.44** | **3.88** |
| ELSI (m=3) | 30 | 1.0 | **4.81** | **3.43** | 3.36 | 3.72 | 4.33 | 3.86 |
| ELSI-C (m=3) | 30 | 1.0 | 4.68 | 3.21 | 3.20 | 3.79 | 4.35 | 3.77 |
| EAGLE2 | 60 | 0.0 | 5.58 | 4.36 | 4.48 | 4.61 | 5.24 | 4.66 |
| ELSI (m=1) | 60 | 0.0 | 5.73 | 4.42 | 4.57 | 4.85 | 5.47 | 4.84 |
| ELSI (m=2) | 60 | 0.0 | 5.70 | 4.49 | 4.54 | 4.84 | 5.42 | 4.83 |
| ELSI-C (m=2) | 60 | 0.0 | 5.67 | 4.44 | 4.46 | 4.81 | 5.42 | 4.79 |
| EAGLE3 | 60 | 0.0 | **5.75** | **4.51** | **4.60** | **4.88** | **5.51** | **4.89** |
| ELSI (m=3) | 60 | 0.0 | 5.75 | 4.47 | 4.59 | 4.85 | 5.45 | 4.84 |
| ELSI-C (m=3) | 60 | 0.0 | 5.66 | 4.39 | 4.52 | 4.80 | 5.37 | 4.78 |
| EAGLE2 | 60 | 1.0 | 4.72 | 3.48 | 3.57 | **4.06** | 4.41 | 3.95 |
| ELSI (m=1) | 60 | 1.0 | **5.16** | 3.40 | 3.64 | 3.74 | 4.66 | 4.06 |
| ELSI (m=2) | 60 | 1.0 | 4.55 | 3.64 | 3.69 | 3.86 | 4.60 | 4.06 |
| ELSI-C (m=2) | 60 | 1.0 | 5.05 | 3.43 | 3.69 | 4.02 | 4.62 | 4.06 |
| EAGLE3 | 60 | 1.0 | 4.91 | **3.67** | 3.56 | 3.87 | 4.59 | **4.12** |
| ELSI (m=3) | 60 | 1.0 | 5.07 | 3.65 | **3.77** | 3.72 | 4.54 | 4.06 |
| ELSI-C (m=3) | 60 | 1.0 | 4.94 | 3.60 | 3.48 | 3.93 | **4.71** | 4.05 |

在 Llama 上，格局变了：

- ELSI (m=1) 仍稳定超过 EAGLE2 **2.8%–4.9%**，并且用 **单层就达到 EAGLE3 约 99% 的水平**；
- 但继续加层没有额外收益：$m=1, 2, 3$ 的 AVG 差距只有 $0.017\,\tau$；ELSI (m=3) 平均比 EAGLE3 低约 1.0%；
- 分项来看仍有亮点：T=1.0 时 ELSI (m=3) 在 Roleplay 上比 EAGLE3 高 4.7%–5.9%，在 Coding 上高 3.0%–3.3%。

为什么会出现这种差异？答案就在前面的 CKA 图里：Llama 的中间层 CKA 普遍接近 1，层与层高度可互换，**随便均匀采三层就已经足够多样**，精细选层的空间自然被压缩。笔者认为这个“负结果”恰恰彰显了框架的解释力：ELSI 不只是一个选层工具，还是一个诊断工具——它能告诉你当前模型处于“表示多样化”还是“表示冗余”的 regime，前者值得精挑层集合，后者直接砍层数省钱即可。

## 附录精选：Token 过滤不改变 CKA 隔间结构

有人可能会担心：Token 过滤砍掉了 70% 的 token，会不会把 CKA 隔间结构也搞坏了？附录做了对照实验，在相同的边界检测参数下比较过滤前后的 CKA 转变图：

![CKA 对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Surfing-the-Activation-Wave-Analyzing-the-Information-and-Geometry-of-LLM-Hidden-States-for-Speculative-Decoding/figures-v2/cka_transition_qwen3-4b_token_filter_ff30_msf001.png)

> 图解：Qwen3-4B 在 Token 过滤（$\rho=0.30$）后的 CKA 转变图，黄色虚线为检测边界。与未过滤版本对比，检测参数完全一致，边界位置和隔间结构基本保持一致——差异可归因于过滤本身极小。Llama-3.1-8B 上的对照同样如此。这说明 Token 过滤只是降低了统计噪声，并没有扭曲模型的几何结构，可以放心作为预处理步骤使用。

## 总结与展望

- **问题**：EAGLE 式投机解码的选层规则（$[N-1]$、$[2, N//2, N-3]$）是位置启发式，缺乏理论依据，且组合爆炸使穷举不可行；
- **工具**：矩阵熵衡量层集合信息量，CKA 转变分数揭示深度方向的表示隔间，两者互补；
- **发现**：熵沿深度呈“激活波”形态、层拼接熵增益不均匀、模型间隔间结构差异巨大——选层必须因模型制宜；
- **方法**：ELSI 用熵驱动的 Beam Search 选层，ELSI-C 加 CKA 候选池过滤保多样性，Token 熵过滤将跨 prompt 方差降低 35%–48%；
- **结果**：Qwen3-4B 上 ELSI (m=3) 平均超 EAGLE3 1.6%–2.2%，单层选择即可打平 EAGLE3 三层；低熵负对照垫底验证了因果性。

一句话展望：这套“熵 + 几何”的选层框架本质上是通用的表示选择工具，未来既可扩展到更多目标模型和草稿 Conditioning 方式，也有望迁移到 Embedding 检索、可解释性分析等需要“从模型中挑层”的场景。

> 本文参考自 [Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding](https://arxiv.org/abs/2610.12939)