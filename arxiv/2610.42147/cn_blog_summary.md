# LSA：10M 上下文单层提速 72.4 倍

当上下文扩展到 1000 万 Token，拖慢稀疏注意力的往往不是注意力计算，而是为每个 Query 扫描全部 Key 的 Indexer，这直接卡住长周期科研与代码 Agent 的响应速度。Latent Sparse Attention（LSA）把 Key 聚类到学习得到的潜空间，让 Query 搜索簇中心而非逐个扫描：在 10M Token 下，D-LSA 使 Qwen-3.8-27B 单层 Prefill、Decode 分别提速 **72.4×、42.5×**。

## 瓶颈不在算注意力，而在找 Key

- 传统稀疏注意力虽只计算约 $k$ 个重要 Key，Indexer 仍扫描全部历史，复杂度没有根本改变。10M 时，Indexer 占 Prefill 时间的 **98%**、Decode 时间的 **89%**。
- Top-$k$ Oracle 实验说明稀疏化可行：128k 上下文、$k=512$ 时，每个 Query 仅关注约 **0.8%** 的 Key，仍保留 Full Attention 的 **98%** 准确率。难点是以低成本找对 Key，而非简单少看 Key。

## 用语义目录替代全量扫描

LSA 像图书馆检索：先按主题建目录，再取回相关条目。Key 经 L2 归一化、4:1 池化和 Ward 聚类，写入包含簇中心与成员位置的 Bucket Cache；Query 选择 Top-$p$ 桶并展开为长程 Key，再补上 4 个 Sink 与最近 512 个 Key 的 Window。索引按 512 个 Token 局部增量构建，无需每步重排历史。

Flat 仍扫描所有桶中心；Tree 用 Beam Search 逐层展开，使索引 Prefill 从 $\mathcal{O}(T^2)$ 降至 $\mathcal{O}(T\log T)$。但 Tree 的串行开销使其单请求 Decode 未必更快，因此主实验 Decode 仍用 Flat。

## 两条训练路线，覆盖新旧模型

- **D-LSA**：冻结底座，只蒸馏 64 维索引投影与温度，用前向 KL 让学生分布逼近原注意力；无需微调全模型，也提供免训练变体。
- **N-LSA**：直接聚类注意力 Key，各 Query Head 分别打分并取最大值；选择停止梯度，仅靠 Next-Token Loss 训练，不增加参数、不需要教师或辅助 Loss。

## 超长上下文提速，同时守住准确率

- **准确率总体贴近基线**：Qwen 在 10M 上，D-LSA Flat 的 RULER 为 **63.9**，高于 Full Attention 的 59.9；PaperTrail 为 **29.1**，也高于基线的 19.8。GLM 的 10M RULER 差距在 4 分以内。
- **收益随长度增长**：GLM 相对原生稀疏注意力的单层 Prefill、Decode 提速 **22.7×、4.6×**。端到端收益小于单层，H200 上两者 10M TTFT 分别改善 **9.21×、2.24×**。
- **原生训练同样可行**：N-LSA 在约 0.5B 模型、128k 训练中取得与 Full Attention 相同的 **1.085** 验证 Loss，整模型达到相同 Loss 快 **1.24×**。

LSA 的边界也很清楚：短上下文固定开销可能抵消收益；Tree 在 GLM 超长区间会损失准确率；完整 KV Cache 并未压缩；N-LSA 的 128k 峰值内存为 **6.8 GiB**，高于 Full Attention 的 5.8 GiB。笔者认为，其真正价值是把语义索引变成可增量维护的推理结构，但多事实检索与跨文档事实链仍是所有方法共同的能力瓶颈。

> 本文参考自 [Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention](http://arxiv.org/abs/2610.42147v1)