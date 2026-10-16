# Muse Spark Open：Meta 开源 816B 多模态 MoE，2-bit 量化版跑在单台 Mac 上

816B 参数的模型需要多大的机器？Meta 给出的答案跨度很大：BF16 原版要 4 台 8×H200 节点，而 2-bit 量化版能在一台 512GB 内存的 Mac M3 Ultra 上跑出 18 tokens/s。

10 月 15 日，Meta 发布 Muse Spark Open——Muse 系列的全尺寸开放权重模型。它是混合专家（MoE）架构：总参数约 8160 亿、每个 token 激活约 420 亿，支持最长 100 万 token 上下文，能读文本、图片、音频和视频（输出为文本），定位是智能体编程——读代码库、写代码、调工具，还能一边看截图、听语音一边干活。

## 816B 的体量，42B 的速度

语言模型部分 62 层，每层 256 个专家、每个 token 选 8 个，专家参数约占总量 97%——这个模型的"大"几乎全大在专家上。MoE 的取舍因此很直白：计算量相当于 42B 模型，但 256 个专家都得驻留显存，部署要按 816B 准备内存，BF16 checkpoint 约 1.5 TB。图像和视频共用一套 18 亿参数的感知编码器；音频编码器每 80 毫秒输出一个 token（每秒 12.5 个），可以据此估算音频对上下文的占用。1M 上下文的真实成本主要在 KV cache：填满约需 266 GB，还不含模型本身，GQA 在这里起了关键的压缩作用。

## 推理与微调

一个绕不开的设定：模型强制推理（reasoning），low/medium/high/xhigh 四档只能调低、不能关闭，简单任务想省 token 就设 low。部署路径按硬件分得很清：有 Blackwell 节点用 NVFP4 + SGLang/vLLM；有 512GB 的 Mac 用 llama.cpp 或 oMLX 量化版（模型本体分别约 304 GB 和 344 GB）；都没有就用 Inference Endpoints 一条命令起步。

微调方面，官方在 8 节点 × 8 H100 上用 TRL 跑了 LoRA 示例：只训四个注意力投影，每层 256 个专家全部冻结。500 步约 2 小时，平均 token 准确率从 72% 升到 84%，loss 从 1.26 降到 0.57——配方可以直接套到自己的数据上。

## 能做什么，以及要留意什么

官方做了三个可复现的演示：Mac 上 18 tokens/s 的量化模型配合 Pi 智能体迭代生成 Three.js 3D 资产；只看截图和历史动作（没有 DOM）的纯多模态浏览器智能体；语音描述需求、自动在 Amazon 搜索比价的购物助手。官方团队称，在 computer use 任务上测试时被其表现惊到。

需要留意的是：官方博客没有给出与同类模型的基准对比，"迄今最强的美国开放权重模型"目前只有 Meta 自己的表述支撑；推理无法关闭、KV cache 占 266 GB、BF16 版 1.5 TB 的体量，都是实打实的使用成本。回到开头的问题——816B 要多大机器？答案取决于你接受多少-bit 的量化。

> 本文参考自 [Welcome Muse Spark Open: open-weight frontier multimodal mixture-of-experts model from Meta](https://huggingface.co/blog/meta-muse-spark-open)