# DeepInfer：DeepSeek 开源的生产级推理引擎

做推理框架的不少，但把支撑线上大规模服务的引擎原样开源的并不多。DeepSeek 近日开源的 DeepInfer，按官方说法正是驱动其生产服务的高性能推理框架，本次开放离线推理能力和规模化服务的核心特性——它的价值在于把生产环境中真正起作用的机制摆到了明面上。

目前只支持一个模型：DeepSeek-V4.1-Flash；硬件覆盖 NVIDIA SM100（Blackwell B 系列）和华为昇腾 950，环境要求较新（GCC 13+、Python 3.12，CUDA 13 或 CANN 9.2）。官方也明确列出不包含的部分：KV cache 系统、PD 分离部署、prefill/decode 副本的负载均衡——开源的是强引擎核心，这三块要自己补齐。

## 核心特性：从 kernel 到调度

底层整合了此前多以独立项目发布的 FlashMLA、DeepGEMM、DeepEP 等 kernel；并行上注意力用 DP+CP、MoE 用 EP 配合 EPLB，与 DeepSeek 此前披露的线上架构一脉相承。更值得单独看的机制有：

- **DSpark 投机解码**：按置信度调度验证，默认最大草稿长度 5；
- **Decoder SWA 有界重放**：只重放有界后缀（V4.1 默认保留 140 token），削减 decoder 的 prefill 量；
- **零开销异步调度**：CPU 批次调度与设备执行重叠，默认开启；
- **批次无关推理与确定性采样**：不同批次下结果完全一致，相同输入产生相同输出（仅 NVIDIA）——这对 RL 训练采样和可复现评测尤其关键，也是大规模推理系统里很少被显式承诺的性质；
- **结构化输出**：可将生成约束在正则、JSON schema 或 EBNF 内。

## 性能：两条帕累托曲线

官方基准的测试条件是 prefill 用 4 张 SM100、decode 用 64 张，上下文长度 131,072。

![DeepSeek V4.1 Flash prefill 阶段「非缓存输入吞吐 vs. 延迟」帕累托曲线](images/prefill_pareto.png)

> 图解：prefill 阶段测试条件为 EP4DP4、384+32 个专家，启用 Decoder Bounded Replay。横轴为单步 prefill 延迟（毫秒），纵轴为每 GPU 非缓存输入吞吐（tokens/s），各点标注每个 DP rank 的 batch size。

prefill 侧，batch 从 4,096 提到 16,384，吞吐从约 60,000 涨到约 81,000 tokens/s/GPU，延迟从约 70ms 涨到约 200ms；再加到 65,536，吞吐只爬到约 88,000，延迟却冲过 700ms——4 卡规模下，16K 上下的 batch 是比较均衡的工作点。

![decode 阶段「输出吞吐 vs. 交互性」帕累托曲线](images/decode_pareto.png)

> 图解：decode 阶段测试条件为 EP64DP64、384+64 个专家、nextn=3 固定草稿、逐 token 接受率 75%。横轴为单用户速度，纵轴为每 GPU 输出吞吐，各点标注每个 DP rank 的并发请求数。

decode 侧是吞吐与单用户体验的经典权衡：8 并发时单用户约 325 tokens/s、整卡约 2,600；256 并发时整卡约 26,300 tokens/s、单用户降到约 103。75% 的草稿接受率（等效每步约 3.25 个 token）说明投机解码在这条曲线上起了实质作用。

## 写在最后

上手很简短：Python 3.12 环境导出 `DEEPINFER_ACCELERATOR=nvidia`（昇腾用 `ascend`）后执行 `./build.sh`，跑 `examples/offline_inference.py` 即可离线推理。对想自建 V4.1-Flash 推理的用户，这是官方口径的高性能起点；对做推理系统研究的人，它更像一份生产系统的「解剖样本」——置信度调度、SWA 有界重放、批次无关性这些机制，此前多散见于论文和博客，现在有了可直接阅读的实现。框架以 Apache 2.0 开源，官方说明其最初主要构建在 vLLM 之上，并参考了 SGLang 和 TensorRT-LLM——想评估它的工程底子，这三条技术谱系是很好的参照系。

> 本文参考自 [GitHub - deepseek-ai/DeepInfer: DeepInfer: High-performance, large-scale inference framework for DeepSeek models](https://github.com/deepseek-ai/DeepInfer)