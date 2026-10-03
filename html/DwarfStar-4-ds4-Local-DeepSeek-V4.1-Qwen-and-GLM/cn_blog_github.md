# DwarfStar 4（ds4）：把 DeepSeek V4.1、GLM 5、Qwen3.8 塞进本地机器

手头只有一台 128 GB 内存的 Mac 或工作站，却想本地跑 2840 亿参数级别的开源旗舰模型——这件事过去基本不可能，而 **DwarfStar 4（ds4）** 就是冲着这个矛盾来的。随着 DeepSeek V4.1、GLM 5.x、Qwen3.8 等开源权重模型能力逼近闭源前沿，"数据不出本机"的本地推理需求越来越刚性，但动辄数百 GB 的权重把绝大多数个人机器挡在门外。ds4 的思路一句话概括：用一个刻意做"窄"的 C 语言推理引擎，配合 **非对称 2-bit 量化** ——只大幅压缩 MoE 模型中的路由专家（routed experts），保住关键共享路径的精度——让旗舰模型真正落进单机内存。效果有多硬？在 M5 Max 128 GB 上，DeepSeek V4 Flash 的 Q2 版本在 32K 上下文下跑出 **34.4 t/s 生成、557 t/s 预填充** 的速度，2K 短上下文下预填充更是达到 **790 t/s**。

![封面图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/DwarfStar-4-ds4-Local-DeepSeek-V4.1-Qwen-and-GLM/images/og-preview-v2.png)

> 图解：ds4 的项目封面。"Run open weights locally"（在本地运行开源权重）是整个项目的宣言，底部标签点明了支持范围：DeepSeek、GLM 两大模型家族，以及 Metal、CUDA、ROCm 三种后端——分别对应苹果芯片、NVIDIA 显卡和 AMD 显卡。

## 背景：巨型 MoE 与本地推理的矛盾

先看清问题的体量。ds4 的主力支持对象 **DeepSeek V4 Flash** 是一个 2840 亿参数的混合专家模型（Mixture-of-Experts, MoE）。这类模型的常规玩法是远程部署：云端多卡集群对外服务（Serving），用户调 API。

但 ds4 从完全相反的约束出发：**模型必须住在自己的机器上**。这个约束背后是很实际的需求——代码、文档、对话记录不离开本机；不依赖网络和订阅额度；Coding Agent 长时间运行时成本可控。

矛盾显而易见：哪怕按最粗的一刀切估算，284B 参数在 2-bit 下的理论下限也是：

$$
\text{权重大小} \approx 284 \times 10^9 \times 2\ \text{bits} \approx 71\ \text{GB}
$$

这还没算关键路径的更高精度和 KV Cache 的开销。也就是说，即便把量化压到极限，也只有 128 GB 级别的高内存机器才装得下，而且前提是量化本身不能把模型"压傻"。

## 核心思路：塌缩成"矮星"，而不是被截肢

项目名 DwarfStar（矮星）本身就是一个很贴切的比喻，官网用三个阶段讲清了它的哲学：

- **Phase 1 · 巨星（The Giant）**：DeepSeek V4 Flash 这样的巨型 MoE，默认归宿是远程集群。
- **Phase 2 · 塌缩（The Collapse）**：用非对称量化压缩路由专家、保留关键路径，让模型在高内存机器上变得实用——是"压缩"而不是"脑叶切除"。
- **Phase 3 · 矮星（The Dwarf Star）**：塌缩后的模型致密、常驻、完全属于你。本地引擎对外暴露 CLI、HTTP API 和原生 Agent，三者共享同一份模型状态与缓存。

这个叙事的关键在第二阶段：**压缩必须有选择性**。一刀切的全员 2-bit 会显著损伤模型能力，ds4 的选择是区别对待模型内部的组件。下一节就拆解它具体怎么做的。

## 三大核心设计

### 非对称 2-bit 量化：压专家，保主干

MoE 模型的参数大头在 **路由专家（routed experts）** 上——每次前向只激活其中一小部分，单个专家对单个 token 的贡献有限，天然更耐量化。而真正"伤不起"的是所有 token 都要经过的共享路径（如 Attention、共享专家、Embedding 等）。

ds4 的做法是：**路由专家压到 2-bit，关键共享路径保持高精度**，并配合 **imatrix**（重要性矩阵，一种用校准数据统计各权重重要性的量化辅助技术）来挑选该保的精度。官网的说法是 "Compressed, not lobotomized"——这正是其支持的 routed-MoE 构建版本能塞进目标机器的原因。

> 笔者认为这个设计的聪明之处在于：它把"量化预算"花在了刀刃上。MoE 架构本身就暗示了参数重要性的分布极不均匀，非对称量化等于顺着架构的纹理下刀，而不是横着切。

### KV Cache 作为"磁盘公民"

长上下文场景里，KV Cache（注意力机制缓存的历史 Key/Value）的体积和权重一样致命，而且机器一重启就要重新预填充（prefill）——对动辄 64K、100K 上下文的 Agent 工作流来说，这等于每次开机先白等几分钟。

ds4 把 KV Cache 变成 **SSD 上的一等公民**：长前缀可以落盘保存，通过 **prompt hash** 匹配恢复。重启不再意味着全量重跑 prefill。这个设计对 Coding Agent 尤其关键——系统提示词和仓库上下文往往是稳定的长前缀，缓存复用的收益非常大。

### 一个引擎，三个接口

ds4 不做多面手，而是把同一份常驻的模型状态同时喂给三个入口：

- `./ds4`：交互式 CLI，直接对话；
- `./ds4-server`：本地 HTTP 服务，同时讲 **OpenAI 和 Anthropic 两种 API 方言**；
- `./ds4-agent`：面向持久编码会话的原生 Agent。

共享模型状态意味着：CLI 里聊了一半的上下文，Server 和 Agent 不需要再加载第二份权重——单机内存本来就紧张，这个"一份权重、多个门面"的设计是刚需而非锦上添花。

## 架构：刻意做"窄"的技术栈

解决了压缩和缓存之后，剩下的问题是引擎本身怎么组织。ds4 的答案可以用一句话概括：**窄而深，而不是宽而浅**。

它不是又一个通用 GGUF 加载器（官网明确说 "Not a generic GGUF runner"），只支持一小撮经过端到端验证的模型家族和布局：

- **模型层**：DeepSeek V4 / V4.1、GLM 5.x、Qwen3.8 Flash Next，仅限项目自带的 GGUF 布局，统一使用非对称 2-bit + imatrix 方案，覆盖文本和视觉模型；
- **引擎层**：纯 C 编写的 ds4 engine，后端覆盖 Metal / CUDA / ROCm，支持张量并行（Tensor Parallelism）、会话批处理（Session Batching）、投机解码（DSPark + MTP）、视觉输入；
- **缓存层**：KV Cache 在 RAM 与 SSD 之间流动，跨重启存活；
- **接口层**：CLI / OpenAI + Anthropic API / 原生 Coding Agent，记忆能力从个人级延伸到分布式。

支持视觉输入（Vision Input）这一点值得单独提一句：它意味着截图、UI 界面这类视觉上下文也能喂给本地模型，对 Agent 场景是实质性的能力扩展。

## 上手：三步跑起来

部署流程刻意保持了极简，核心就是"拿权重、编后端、开聊"：

**第一步：拉取项目与权重**

```bash
$ git clone https://github.com/antirez/ds4
$ cd ds4 && ./download_model.sh ds4f-q2
```

**第二步：针对你的后端编译**

```bash
$ make
$ make cuda-spark
```

**第三步：启动 CLI 或本地服务**

```bash
$ ./ds4
$ ./ds4-server --ctx 100000
```

注意两点：一是 **必须使用项目提供的 GGUF 权重**，通用 GGUF 文件不是目标（量化布局是定制验证过的）；二是 `--ctx 100000` 这样的参数表明 10 万 token 级上下文是一等支持场景，配合前面说的 KV Cache 落盘，长上下文 Agent 是明确的目标负载。

## 硬件适配：你的机器能不能跑？

ds4 提供了一个"fit check"思路：按平台和内存选路径。基线是 **DeepSeek V4 Flash Q2**；在 128 GB 内存档位，GLM 5.3 Q2 和 Qwen Q4 也能装下，而更大的 V4.1 Q2 则需要从 SSD 流式加载（streaming）。

参考起点是一条命令：

```bash
./download_model.sh ds4f-q2 && make
```

## Benchmark：预填充与生成要分开看

官方给出的参考数据（DeepSeek V4 Flash，Q2 量化）：

| 机器 | 上下文 | 预填充 t/s | 生成 t/s |
| --- | --- | --- | --- |
| M5 Max, 128 GB | 2,048 tok | 790.2 | 39.4 |
| M5 Max, 128 GB | 65,536 tok | 398.5 | 27.6 |
| DGX Spark, 128 GB | 2,048 tok | 825.8 | 18.1 |
| DGX Spark, 128 GB | 65,536 tok | 823.0 | 13.8 |

> 表解：两张卡的性格差异很明显。M5 Max 生成速度占优（39.4 t/s vs 18.1 t/s），因为解码是显存带宽敏感型负载，统一内存的高带宽优势明显；DGX Spark 预填充更强且在长上下文下几乎不掉速（825.8 → 823.0 t/s），适合重 prefill 的批处理场景。对长上下文 Agent 工作负载，读表时务必把两列分开看——预填充决定"开工要等多久"，生成决定"干活流不流畅"。

另一个值得注意的数字是上下文长度对 Apple 平台的影响：从 2K 拉到 65K，预填充从 790 t/s 掉到 398 t/s，生成从 39.4 t/s 掉到 27.6 t/s——KV Cache 增长的代价是实打实的，这也反过来说明 KV Cache 落盘复用设计不是可有可无的点缀。

## 生态：直接对接 Coding Agent

`ds4-server` 同时兼容 OpenAI 和 Anthropic 风格的 API，意味着 **Codex、Claude Code、OpenCode** 这类主流本地 Coding Agent 只需要改一个 base URL，就能从云端模型切换到自己机器上的 ds4。对想"把 Agent 的脑子留在本地"的用户来说，这是整个故事闭环的最后一环：引擎、量化、缓存、接口，最终都是为了让你顺手的工具能无缝接进来。

## 总结

- ds4 是一个 **刻意做窄** 的纯 C 推理引擎，只支持 DeepSeek V4 / V4.1、GLM 5.x、Qwen3.8 等少数经验证的模型家族，覆盖 Metal / CUDA / ROCm 三种后端；
- 核心压缩手段是 **非对称 2-bit 量化 + imatrix**：路由专家压到 2-bit，关键共享路径保精度，让 284B 级 MoE 落进 128 GB 单机；
- **KV Cache 落盘 + prompt hash 恢复** 让重启不再等于全量重跑 prefill，直击长上下文 Agent 的痛点；
- 一个引擎三个接口（CLI / OpenAI + Anthropic API / 原生 Agent），共享同一份模型状态，可直接对接 Codex、Claude Code、OpenCode；
- 实测 M5 Max 128 GB 上 32K 上下文 34.4 t/s 生成、557 t/s 预填充，短上下文预填充达 790 t/s，已进入实用区间。

局限也很明确：它只伺候一小撮特定布局的模型，通用 GGUF 不在射程内——这是"窄而深"策略的一体两面。如果这种"为特定旗舰模型做定制引擎"的路线被验证成功，本地推理或许会从"什么都能跑一点"走向"少数模型跑得极好"，而 ds4 正是这个方向上一个值得持续观察的样本。

> 本文参考自 [DwarfStar 4 (ds4): Local DeepSeek V4.1, Qwen and GLM](https://dwarfstar.sh/)