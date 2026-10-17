# Beam：501B 开放 MoE 实战

把前沿编程与 Agent 模型部署到自有硬件，难点是权重、KV cache 和后训练同时落地。Reflection 的首个模型 Beam 采用 Apache 2.0，开放 501B 参数 MoE：每个 token 激活约 23B，支持 1M token 上下文，并提供 Transformers、vLLM、SGLang、TRL 和 Miles 配方。NVFP4 把约 1 TB 的 BF16 权重压到约 280 GB；官方报告的 SWEBench Verified 在 BF16、FP8、NVFP4 下分别为 81.6%、80.2%、80.7%。它比单独开放 checkpoint 更完整，但并不等于低成本。

## 稀疏计算不等于小内存

Beam 的 52 层 decoder 中，后 49 层各有 256 个专家，每个 token 只选 8 个，因此降低的是计算量；全部专家仍须驻留内存或显存，实际速度还受通信、带宽和 serving 实现影响，原文未给出与 23B 稠密模型的延迟对比。

1M 上下文由 43 个 512 token 滑窗层和 9 个全局层共同支撑。按每条序列、BF16、Transformers dynamic cache 计算，KV cache 为 36.08 GiB，而全全局注意力需 208 GiB；这不包括权重、激活和并发请求。Blackwell 可原生运行 NVFP4，H200 等应选 FP8。官方建议按 4 张 B200 规划长上下文与并发，2 张虽能装下权重，却难留运行空间。

## 证据支持什么

三种精度的结果并非单向变化：NVFP4 在 SWE Bench Pro v2-Hard 上为 78.6%，高于 BF16 的 76.5%，却在 Terminal Bench v2.1 上以 78.2% 低于 80.1%。原文未给重复运行方差或跨模型比较，因此既不能说“量化完全无损”，也不能凭单项下降判定低精度不可用。

Hugging Face 团队在 Reflection 平台做了五项 low/high 单次试用，并非正式 benchmark。High 五项都消耗更多 token：长代码上下文任务从 1,613 增至 5,007，并补上 Low 缺失的格式要求；数学任务从 829 增至 2,809，补上了证明。每项只跑一次，且前者还受 prompt 歧义影响，不能据此推算稳定成功率。

后训练同样只支持有限结论。TRL LoRA SFT 后，16 条留出 prompt 中按要求语言推理的数量由 3 条升至 16 条，但不代表推理更正确。Miles 全参数 GRPO 让 AIME 2025 从 46.25% 升至 55.8%，LoRA 版从 42.9% 升至 64.2%，平均输出长度还从 3,188 降至 2,197 token；两者都是特定数学配方结果，参考硬件仍分别达 96 和 32 张 H200。

回到开头，Beam 的价值是把开放权重、服务化与训练配方连成一套可自主定制的工程栈。已有多张 Blackwell 或 Hopper GPU 的团队可做真实业务回放；没有集群，可先用 Inference Providers 或 Endpoints 比较精度、推理档位、token 和延迟。现有证据尚缺独立复测、跨模型基线与 1M 真实负载数据，不宜把官方演示当成普遍结论。

> 本文参考自 [Welcome Beam: a 501B open model you can own, customize, and scale](https://huggingface.co/blog/reflection-beam)