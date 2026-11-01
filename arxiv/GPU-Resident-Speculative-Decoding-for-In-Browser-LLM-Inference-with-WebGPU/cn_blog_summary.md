# WebSpec：GPU 常驻解码最高提速 8.6 倍

浏览器本地运行 LLM 能保护隐私、支持离线，却常被逐 Token 的 CPU 回读和 GPU 调度拖慢；WebSpec 将投机解码的完整决策回路留在 GPU，相对 WebLLM 最高提速 **8.6 倍**，可选 Draft Heads 达 **10.2 倍**，且输出与同一量化运行时的 Greedy Decoding 逐 Token 一致。

## 浏览器为何需要重新设计投机解码

WebLLM 每生成一个 Token，都要经历 Kernel 提交、状态回读和 JavaScript 决策；浏览器安全校验与进程间通信进一步放大等待。一次前向计算包含 250–400 次 Dispatch，Qwen2.5-0.5B 每 Token 的 Host 往返、采样和簿记约占 8 ms。

因此，草稿接受得多并不等于更快，验证成本可能抵消收益：固定 8 节点树在工作站提速约 1.5 倍，在笔记本核显上反而慢 **2.7 倍**。

## 四个部件组成 GPU 常驻闭环

- **确定性小批量 Kernel**：采用 4-bit 权重、8-bit Activation 和整数点积，复用权重与 Activation 加载；固定矩阵乘法及 Attention 的归约顺序，保证批量验证与逐 Token Greedy 计算逐 bit 一致。
- **GPU 常驻循环**：Draft、Verify、Accept、KV Cache Compact 全在 GPU 完成。一次编码 $W$ 步并保持两个窗口在途，CPU 不再逐 Token 询问是否接受。
- **零额外模型草稿**：Context Mixing 混合网页、对话和历史会话中各阶后缀匹配；Session Memory 不进入 KV Cache。可选 Draft Heads 提供更高接受率，但增加 99–151 MB 下载。
- **设备自适应控制**：依据实测验证成本和历史接受率选择树宽，最大化每步期望接受 Token 数与耗时之比；同时在 $W=1,2,4$ 中选择小窗口，避免弱 GPU 上宽树倒亏。

## 结果：小模型和快速 GPU 收益最大

- 相对 WebLLM，笔记本独显、核显和工作站分别提速 **3.5–5.4 倍、4.8–6.2 倍、4.8–8.6 倍**；7B/8B 因更受显存带宽限制，提速回落至 4.8–5.2 倍。
- 工作站 Qwen2.5-0.5B 延迟由 Greedy 的 4.59 ms 降至 GPU Loop 的 2.13 ms，完整方案再到 **1.44 ms**。分解显示，Kernel 是最大收益来源，GPU Loop 与投机继续放大。
- 首 Token 延迟改善 2.4–3.7 倍；GSM8K 平均准确率与 WebLLM 均为 52.3%。主实验及每模型 1,060 条 Prompt 的扩展测试中，投机输出均与本运行时 Greedy 完全一致。

## 结论与边界

- 核心创新不是单个 Drafter，而是 Kernel 确定性、GPU 控制流、上下文复用和在线设备测量的协同。
- 投机宽度是硬件参数，不能跨设备复制；默认 Context/Memory 方案还避免了第二个模型的下载与显存成本。
- 当前无损路径仅覆盖 Greedy Decoding；首次校准成本未计入稳态加速，随机 Sampling、核显页面响应及更多浏览器和移动 GPU 仍待验证。

> 本文参考自 [GPU-Resident Speculative Decoding for In-Browser LLM Inference with WebGPU](http://arxiv.org/abs/2610.39761v1)