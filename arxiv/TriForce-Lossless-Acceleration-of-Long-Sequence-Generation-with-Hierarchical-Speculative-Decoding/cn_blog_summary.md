# TriForce：检索稀疏 KV，分层加速长序列生成

长序列生成的核心瓶颈是模型权重与持续膨胀的 KV Cache 反复搬运，TriForce 以两级投机解码兼顾速度与完整上下文，在单张 A100 上最高加速 **2.31×**，双 RTX 4090 的 CPU Offloading 场景中最高加速 **7.78×**，且不改变目标模型的输出分布。

## 核心洞察：完整 KV 必要，但不必每步都读

StreamingLLM、H$_2$O 等方法通过逐出 KV 降低成本，却会永久丢失长程信息；直接使用小草稿模型，又会因缓存近似与目标分布逐渐偏离而降低接受率。TriForce 基于两个观察另辟蹊径：

- **Attention 具有稀疏性**：在 120K 上下文中，除前两层外，几乎所有层仅取 4K token 就能恢复超过 **96%** 的 Attention Score，稀疏 KV 足以生成高质量候选。
- **相邻步骤具有局部性**：一次检索得到的 Top-K 上下文可服务多个生成步骤，缓存构建成本能够被摊销，并可定期或按接受率自适应重建。

## 关键方法：检索起草，分层验证

TriForce 始终保留 Full Cache，把完整 KV 分块，根据当前 Query 与各 Chunk 平均 Key 的相关度，在固定预算内检索 KV。近似只发生在草稿侧，检索失误只会影响接受率和速度；最终仍由完整 KV 的目标模型验证与纠正，因此 **Lossless 指输出分布无损**。

系统用三级推理分别处理两个瓶颈：

1. **小模型 + StreamingLLM Cache**：低成本起草，减少完整模型权重的重复加载。
2. **目标模型 + Retrieval Cache**：利用稀疏全局上下文继续起草与验证，减少完整 KV 加载。
3. **目标模型 + Full Cache**：并行执行最终验证，精确保留原模型输出分布。

在 120K 上下文、4K KV Budget 的对比中，Retrieval 在 Needle Retrieval 上接受率达到 **0.9878**，接近 Top-K 的 0.9989，而 StreamingLLM 与 H$_2$O 仅为 **0.0519** 和 **0.0739**。

## 实验结论：接受率与草稿成本共同决定收益

- A100 上，TriForce 在温度 0 时达到 **2.31×**、平均接受率 **0.9234**；温度升至 1.0 时接受率仍高于 0.9。仅有 Retrieval 或将检索换成 StreamingLLM 时，加速分别只有 1.80× 和 1.90×，说明检索与层级结构缺一不可。
- Offloading 收益更显著：双 RTX 4090 上 Llama2-13B-128K 达 **7.94×**；单张 L40 上，256K/512K 输入分别达 **11.81×/12.10×**。这些结果使用不同硬件与基线，不能直接与 A100 数字混比。
- 4K KV Budget 是接受率与读取成本的较优平衡点；最优两级投机长度为 $\gamma_1=2$、$\gamma_2=6$。输出从 256 延长至 2048 token 时，A100 加速仍稳定在约 **2.3×**。

> 本文参考自 [TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding](https://arxiv.org/abs/2404.11912)