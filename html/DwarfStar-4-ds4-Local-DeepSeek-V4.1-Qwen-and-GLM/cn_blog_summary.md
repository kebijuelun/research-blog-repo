# DwarfStar 4：2840 亿参数旗舰模型，塞进 128 GB 单机

DeepSeek V4.1、GLM 5、Qwen3.8 等开源旗舰模型的能力已逼近闭源前沿，但动辄数百 GB 的权重把"数据不出本机"的用户挡在门外。DwarfStar 4（ds4）用一个纯 C 推理引擎加非对称 2-bit 量化，把 2840 亿参数的 DeepSeek V4 Flash 压进 128 GB 单机：在 M5 Max 上，32K 上下文跑出 **34.4 t/s 生成、557 t/s 预填充**，短上下文预填充高达 **790 t/s**——本地旗舰推理正式进入实用区间。

## 矛盾在哪

- DeepSeek V4 Flash 是 284B 参数的 MoE 模型，2-bit 理论下限约 71 GB，再加关键路径精度和 KV Cache，只有 128 GB 级机器才装得下；
- 前提还是量化不能把模型"压傻"——一刀切全员 2-bit 会显著损伤能力。

## 三大核心设计

- **非对称 2-bit 量化，压专家保主干**：MoE 的参数大头在路由专家（routed experts），天然耐量化；Attention、共享专家、Embedding 等所有 token 必经的共享路径保持高精度，再用 imatrix 校准挑选该保的精度。官方说法是 "Compressed, not lobotomized"——压缩，而不是脑叶切除。
- **KV Cache 落盘**：长上下文前缀存入 SSD，通过 prompt hash 匹配恢复，重启不再全量重跑 prefill。对系统提示词和仓库上下文稳定的 Coding Agent，收益极大。
- **一个引擎，三个接口**：CLI、本地 HTTP 服务（同时讲 OpenAI 和 Anthropic 两种 API 方言）、原生 Coding Agent 共享同一份常驻模型状态，单机只需加载一份权重。

## 架构哲学：窄而深

ds4 明确 "Not a generic GGUF runner"，只支持少数端到端验证过的模型家族（DeepSeek V4/V4.1、GLM 5.x、Qwen3.8）和自带 GGUF 布局；纯 C 引擎覆盖 Metal / CUDA / ROCm 三后端，支持张量并行、会话批处理、投机解码与视觉输入。

## 实测数据（DeepSeek V4 Flash Q2）

- **M5 Max 128 GB**：2K 上下文 790/39.4 t/s（预填充/生成），65K 降到 398/27.6——解码吃显存带宽，统一内存占优；
- **DGX Spark 128 GB**：预填充 2K 达 825.8 t/s，65K 仍有 823 t/s，几乎不掉速，适合重 prefill 批处理；
- 读表要点：预填充决定"开工等多久"，生成决定"干活流不流畅"，两列必须分开看。

## 生态与局限

`ds4-server` 兼容 OpenAI/Anthropic API，Codex、Claude Code、OpenCode 改一个 base URL 即可切换到本地模型。代价同样明确：只伺候一小撮定制布局的模型，通用 GGUF 不在射程内。若"为特定旗舰模型做定制引擎"的路线被验证，本地推理或将从"什么都能跑一点"走向"少数模型跑得极好"——ds4 正是这条路上值得观察的样本。

> 本文参考自 [DwarfStar 4 (ds4): Local DeepSeek V4.1, Qwen and GLM](https://dwarfstar.sh/)