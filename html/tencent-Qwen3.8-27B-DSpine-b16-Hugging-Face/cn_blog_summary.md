# DSpine：并行草稿让 Agent 吞吐最高达 3.84 倍

腾讯发布的 DSpine 将推测解码的草稿生成改为并行，并用五层草稿模型服务 Qwen3.8-27B。团队报告的 WorkBuddy Office 低推理强度 Agent 回放中，DSpine 达到 805.53 tokens/s，是自回归解码 210.03 tokens/s 的 3.84 倍；官方 DFlash2 为 664.77 tokens/s。这个结果面向长输出、多轮工具调用负载，并不代表所有生成任务都能获得同等加速。

## 并行草稿如何保留因果依赖

推测解码由轻量草稿模型提出候选 token，再交给目标模型验证。DSpine 并行生成这些 token，同时让每层把预测出的前驱特征传给相邻位置，使因果信息沿网络深度传播，而非让每个位置完全独立猜测。

当前 checkpoint 的草稿块大小为 16，滑动窗口为 8,192 token，推理采用 DSpark 动态验证并启用 ReplaySSM。需要注意，对照的 DFlash2 草稿块为 8，因此实测比较的是两套完整配置，不能把收益全部归因于相邻因果注入或并行草稿中的某一项。

## 接受长度提升如何转化为吞吐

通用基准在 xhigh 设置下测试贪心采样与温度采样，每次最多生成 4,096 个新 token。DSpine 在十组比较中的接受长度均高于 DFlash2：贪心采样 GSM8K 从 5.69 升至 7.20，而温度采样 MT-Bench 仅从 3.89 升至 4.03。温度采样固定为 `temperature=1.0`、`top_p=0.95`、`top_k=20`。接受长度反映草稿通过验证的程度，并非任务正确率，收益也随任务和采样方式变化。

Agent 测试覆盖 WorkBuddy Bench 的 Code、Web、Office 子集，以及 SWE-bench Pro 和 τ²-bench。WorkBuddy 每个子集回放 20 段 CodeBuddy 沙箱对话，启用前缀缓存，每次最多生成 32,768 个新 token。十组接受长度和端到端吞吐比较均由 DSpine 领先；吞吐按总输出 token 数除以并发回放实际耗时计算，达到自回归解码的 2.71 至 3.84 倍。WorkBuddy Web 的 xhigh 设置取得最高绝对吞吐 840.14 tokens/s，高于 DFlash2 的 734.04 tokens/s。接受长度与吞吐并非固定比例，验证、缓存和运行配置都会影响结果。

## 复现与外推都有边界

上述结果使用 SGLang 0.5.22、BF16、FlashAttention 3 和并发 8，需同时加载目标模型与 DSpine 草稿模型，并配置 ReplaySSM、Mamba 缓存、上下文长度及 Qwen3 解析器，不能只替换 checkpoint 就默认复现。

模型卡未列出 GPU，也未提供任务正确率、显存占用或单请求延迟。因此，对使用 Qwen3.8-27B 和 SGLang 服务长轨迹 Agent 的团队，更稳妥的做法是用自己的采样参数和真实轨迹，同时测量端到端耗时、资源占用与任务结果，再判断 16-token 草稿块及配套配置是否值得采用，而不是直接外推最高 3.84 倍。

> 本文参考自 [tencent/Qwen3.8-27B-DSpine-b16 · Hugging Face](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16)