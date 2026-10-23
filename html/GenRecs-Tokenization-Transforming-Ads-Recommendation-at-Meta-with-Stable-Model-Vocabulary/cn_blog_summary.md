# GenRecs Tokenization：Meta 用"稳定语义词汇表"重构广告推荐

一双刚上架的新款越野跑鞋，没有历史点击、没有广告互动记录——在传统推荐系统里，它几乎等于陌生人。Meta 公开的 GenRecs Tokenization 要解决的正是这个问题：不再给每个广告、商品、页面分配孤立 ID，而是像语言模型用词表拼词一样，用一组稳定的语义 token 表示每个条目，新跑鞋可直接"继承"模型对上一代跑鞋品类的认知。这套表示与生成式广告模型 GEM 及 Adaptive Ranking Model 搭配后，在 2026 年 2 月的内部测量中为广告主带来 +0.35% 的跨界面转化提升。

## 三阶段流水线：编码、分解、量化

预训练多模态模型先读取每个条目的文本与图片，编码成携带世界知识的 embedding；Tokenizer 再结合用户互动行为学到的表示，经三阶段流水线量化成一小段离散 token：编码进语义空间、分解到若干独立子空间、每个子空间映射到 codebook 中的一个码字。相似条目共享 token，就像相关的词共享词根。Meta 的经验是不存在单一"最优配方"，需按实体和特征的语义组合各阶段，例如许多广告表示采用 prefix n-gram 方案（学习型编码器＋残差分解＋学习型向量码本）。

## Tokenizer 质量可以直接预测排序精度

迭代 tokenizer 通常要等一次完整下游训练才能验证，代价很高。Meta 的结论是：不必等。他们测试八个版本，发现三个内在质量指标与下游 CTR 预测精度强相关：簇纯度 0.84、分布均衡 0.72、码本利用率 0.64，其中分布均衡是最强的单一预测因子。团队因此可围绕 tokenizer 快速迭代——用 diversity loss、dead code reset 等手段优化码本——而不必每次跑完整训练周期。

![三个内在指标与 CTR 精度的相关性](images/GenRecs-Tokenization-image3.png)

> 图解：三个内在指标与 CTR 预测精度均呈正相关。

## 尾部广告的飞轮与一处取舍

对展示量稀少的尾部广告，tokenization 作用最关键：模型借助共享内容结构跨条目泛化、修复冷启动，并形成自增强飞轮——预测变准，尾部广告赢得更多展示和点击，产生更多正向数据，反过来提升整体预测质量。基于共享 LLM 编码器的双塔检索也达到 SOTA，只用经典双塔一小部分数据就追平其性能，说明码本质量提升对下游是普适的。

代价同样明确：完全文字化（verbalized）的 tokenization 信息最丰富，但所需上下文窗口在广告规模下不现实；Meta 把压缩后的文字化 token 与语义 ID 结合，在信息保真与服务延迟之间取一个点。

## 结语

回到开头的问题：当推荐从判别式打分走向生成式推理，数据地基也要从易失效的 ID 换成稳定的语义表示。但要注意，+0.35% 转化提升与 0.64–0.84 的相关系数均来自 Meta 内部数据和自家生产系统，外部难以复现，绝对幅度也需结合其广告大盘基数理解。最值得借鉴的经验是：tokenizer 的内在质量指标可作为下游效果的先行指标，让词表迭代不必绑死在完整训练周期上。

> 本文参考自 [GenRecs Tokenization: Transforming Ads Recommendation at Meta with Stable Model Vocabulary](https://engineering.fb.com/2026/10/19/ml-applications/genrecs-tokenization-transforming-ads-recommendation-at-meta-with-stable-model-vocabulary/)