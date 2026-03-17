# Attention Residuals：把“残差连接”升级成“按深度做注意力”的一次硬核改造

## 一句话先看懂

这篇论文的核心观点很直接：传统 Transformer 的 Residual 是“ **无差别累加** ”，而 AttnRes 把它改成“ **按内容选择性聚合** ”。  
也就是说，后层不再被迫接收一个混在一起的历史状态，而是可以像做 Attention 一样，从前面各层里“挑重点”。

---

## 论文想解决的根问题

在 PreNorm Transformer 里，经典残差是：

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1})
$$

展开后得到：

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)
$$

这会带来 3 个长期问题：

- 所有历史层被等权混合，缺少 selective access（选择性访问）。
- 一旦信息在聚合时被冲淡，后续层无法“点名找回”某一层表示。
- 随深度增长，隐藏态幅度容易不断变大，出现 PreNorm dilution（贡献被稀释）与训练不稳定。

---

## 方法核心：Full AttnRes

作者提出把“固定累加”改成“深度维度 Attention 聚合”：

$$
h_l = \alpha_{0 \to l} \cdot h_1 + \sum_{i=1}^{l-1} \alpha_{i \to l} \cdot v_i
$$

其中 $v_i = f_i(h_i)$，权重来自 softmax：

$$
\alpha_{i \to l} = \frac{\exp\left(q_l^\top \mathrm{RMSNorm}(k_i)\right)}
{\sum_{j=0}^{l-1}\exp\left(q_l^\top \mathrm{RMSNorm}(k_j)\right)}
$$

并且作者使用的是每层一个可学习 pseudo-query 向量 $w_l$（不是 token 相关 query），因此参数开销很轻量。  
直观上，这让每一层都能“回看全历史”，但不是平均看，而是按输入内容动态分配注意力。

![AttnRes 总览](figures/model)

> 图解：这张结构图对比了 Standard Residual、Full AttnRes、Block AttnRes。横向看是信息来源，纵向看是网络深度。AttnRes 的关键变化是把“固定加法路径”替换为“可学习权重的跨层路由”。

---

## 为什么还要 Block AttnRes

Full AttnRes 的问题不在算力，而在大模型工程里的内存与通信。  
当需要保存并跨 stage 传递大量层输出时，代价会随层数增长到 $O(Ld)$。

Block AttnRes 的做法：

- 把 $L$ 层分成 $N$ 个 block。
- block 内先做局部累加形成 block 表示。
- 跨 block 只对 block 级表示做 Attention。

这样把关键开销从 $O(Ld)$ 压到 $O(Nd)$，在工业训练中更可落地。

![Block 大小消融](figures/block)

> 图解：横轴是 block size（或等价的 block 粒度），纵轴是验证集 loss。可以看到从 Full 到适度分块，性能下降很平滑；约 8 个 block 已能恢复 Full AttnRes 的大部分收益。

---

## 系统实现：论文里很实用的两招

### 1) 训练阶段 Cross-stage caching

在 Pipeline Parallel 下，历史 block 表示如果每次全量传输会很浪费。作者做了缓存复用，只传增量 block，显著减少跨 stage 冗余通信。

![Pipeline 缓存通信](figures/pipeline)

> 图解：图中每个 rank 的 cache 会保留已收到的 block 表示。后续 virtual stage 只需补传新增部分，减少重复带宽占用。

### 2) 推理阶段 Two-phase 计算

- Phase 1：同一 block 的多层 query 批量计算 inter-block attention。
- Phase 2：顺序处理 intra-block 依赖，再与 Phase 1 结果做 online softmax merge。

结果是：训练额外开销很小，推理延迟额外开销 < 2%。

---

## 实验结果（重点数字版）

### 1) Scaling Law：全尺度稳定收益

拟合形式：

$$
\mathcal{L} = A \times C^{-\alpha}
$$

- Baseline：$\mathcal{L} = 1.891 \times C^{-0.057}$
- Block AttnRes：$\mathcal{L} = 1.870 \times C^{-0.058}$
- Full AttnRes：$\mathcal{L} = 1.865 \times C^{-0.057}$

在 5.6 PFLOP/s-days 附近，Block AttnRes 对 Baseline 大约等价于 ** 1.25x ** 计算优势。

![Scaling law](figures/scaling)

> 图解：横轴是计算量（PFLOP/s-days），纵轴是验证 loss。三条曲线斜率接近，但 AttnRes 族在全区间都位于 Baseline 下方，说明这不是“小模型特供收益”。

### 2) 大模型主实验（48B 总参数 / 3B 激活）

论文在 Kimi Linear 体系上做 1.4T tokens 预训练。Block AttnRes 在下游几乎全线提升，尤其是多步推理任务：

- GPQA-Diamond：+7.5
- Math（Minerva 风格）：+3.6
- HumanEval：+3.1
- MMLU：+1.1
- TriviaQA：+1.9

![训练动态对比](figures/dynamic)

> 图解：这组图通常包含 loss、输出幅度、梯度幅度三部分。AttnRes 版本在训练全程 loss 更低；同时输出/梯度在深层分布更均匀，缓解了 PreNorm 下“越深越失衡”的现象。

### 3) 消融实验：哪些设计真的有效

- DenseFormer（静态跨层系数）几乎无提升，说明“跨层连接”本身不够，关键在输入相关权重。
- softmax 优于 sigmoid，说明竞争归一化有价值。
- 去掉 RMSNorm 会退化，说明需要抑制不同层幅度差异对注意力打分的偏置。
- input-dependent query 还能更强，但推理 I/O 和参数成本更高，因此默认方案选择了工程上更优的 learned query。

---

## 机制分析：它到底学到了什么

![注意力热力图](figures/attn_res_weights)

> 图解：行是当前层，列是历史 source（Full 为层级 source，Block 为块级 source），颜色是权重大小。可见主对角线仍最亮（局部性保留），但有明显非对角峰值（学到跨层跳连），且 embedding source 在多层持续有权重。

![结构矩阵视角](figures/depth_matrices)

> 图解：作者把残差机制统一成深度 mixing matrix。Residual/Highway/mHC 可以看成不同约束下的线性混合，AttnRes 对应更一般的 softmax 深度注意力。

![架构扫描](figures/arch_sweep)

> 图解：固定算力和参数预算下，AttnRes 的最优点更偏“深而窄”（更低的 $d_{model}/L_b$），说明它能更有效利用额外深度。

---

## 我认为最有价值的结论

- 这篇工作的创新不只是“换个残差公式”，而是把“深度维的信息路由”从固定规则升级为内容驱动机制。
- 论文最强的地方是把 **算法 + 系统** 一起打通：不仅有 loss 改善，还有 pipeline 与 inference 侧的可部署方案。
- Block AttnRes 的意义很现实：在当前硬件约束下拿到接近 Full 的收益，给大规模训练提供了一个可落地的中间解。

---

## 局限与后续方向

- Full AttnRes 仍受限于跨层状态保留与通信开销，尤其在超深模型上。
- 目前 block 粒度是工程折中，未来更高带宽、更大显存可能让更细粒度甚至 Full 成为默认。
- 另一个方向是探索线性复杂度或稀疏化的 depth attention 内核，继续压缩 I/O 成本。

---

> 本文参考自 [Attention Residuals](https://arxiv.org/abs/2603.15031)