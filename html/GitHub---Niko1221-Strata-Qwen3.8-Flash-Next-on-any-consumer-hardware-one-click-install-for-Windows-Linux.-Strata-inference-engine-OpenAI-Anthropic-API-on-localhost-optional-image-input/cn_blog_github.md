# Strata：把 125B 参数大模型塞进你的游戏显卡

一台普通的游戏 PC，一张 12 GB 显存的 RTX 5070，能不能跑起一个 1250 亿参数的大模型？Strata 的回答是：不但能跑，还能以 94 tokens/s 的速度生成回答。这个开源项目解决的是本地部署领域最现实的矛盾——聪明的大模型需要服务器级的显存，而大多数人手里只有一张游戏显卡。它的核心思路可以概括为一句话：把模型拆开，常用的"专家"放显卡，其余的放内存和 SSD，再加上投机解码加速。实测结果是：RTX 5070 上短对话生成 94 tokens/s，读取 32K token 的文档达到 2650 tokens/s，全程数据不出本机。

![pagoda-preview.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---Niko1221-Strata-Qwen3.8-Flash-Next-on-any-consumer-hardware-one-click-install-for-Windows-Linux.-Strata-inference-engine-OpenAI-Anthropic-API-on-localhost-optional-image-input/images/pagoda-preview.webp)

> 图解：这是 Strata 在一张 RTX 5070 上、以 IQ3_S 量化和 128K 上下文一次提示生成的"体素风格宝塔花园"场景，完整生成过程仅用 49 秒。这张图是整篇文章最好的广告——一个此前需要服务器集群才能跑的模型，现在在游戏显卡上完成了高质量的复杂创作。

## 先说痛点：125B 模型为什么跑不进消费级显卡

像 Qwen3.8-Flash-Next 这个级别的大模型，按惯例需要在拥有数百 GB 显存的服务器上运行。而消费级显卡的显存天花板只有 12-24 GB，差距是一个数量级。

常规的妥协方案有两条：要么换小模型（能力断崖式下降），要么做极致量化（画质"压成马赛克"，智能明显受损）。Strata 走的是第三条路： **不做减法做调度** 。模型还是 125B 参数，只是不同时刻只有一小部分参数在干活，那就把"在干活的"放在最快的硬件上，其余的存在便宜的大容量介质里。

这个思路为什么可行？后文"工作原理"一节会展开，这里先给出关键事实：Qwen3.8-Flash-Next 是一个 MoE（Mixture of Experts，混合专家）模型，内部有 24,576 个"专家"子网络，而生成每个词只调用其中 10 个。这就是 Strata 一切优化的立足点。

## 跑得到底有多快：两台普通游戏 PC 的实测

空口无凭，作者在两台再普通不过的游戏电脑上做了实测。衡量指标有两个：

- **生成速度** （Writes answers）：短对话中回答出现的速度。60 tokens/s 已经超过人眼阅读速度，一个 token 约等于 ¾ 个英文单词。
- **读取速度** （Reads your prompt）：模型"吃下"你发送内容的速度，测试场景是 32K token 的长文档、代码或聊天记录。

NVIDIA 平台：RTX 5070（12 GB）+ Ryzen 5 7600 + 64 GB RAM：

| 量化规格 | 生成速度 | 读取速度 |
| --- | --- | --- |
| Q2_0 | 94 tokens/s | 2,650 tokens/s |
| IQ2_XS | 79 tokens/s | 2,090 tokens/s |
| IQ3_XXS | 62 tokens/s | 1,750 tokens/s |
| IQ3_S | 53 tokens/s | 1,620 tokens/s |
| Coder | 55 tokens/s | 2,180 tokens/s |

AMD 平台：RX 9070 XT（16 GB）+ Ryzen 9 3900X + 47 GB RAM：

| 量化规格 | 生成速度 | 读取速度 |
| --- | --- | --- |
| Q2_0 | 60 tokens/s | 1,160 tokens/s |
| IQ2_XS | 52 tokens/s | 1,110 tokens/s |
| Coder | 44 tokens/s | 1,420 tokens/s |

几个值得注意的读数：

- 量化越狠（Q2_0 约 2-bit），速度越快；规格越大（IQ3_S），质量越高但速度下降。这是经典的"速度-质量"权衡曲线。
- 读取速度是生成速度的 20-40 倍，这意味着扔给它一整本代码库让它读，一分钟就能读完 3 万 token。
- 显存越大越快：RTX 3090（24 GB）预计可达 100-140 tokens/s。AMD 平台也能跑，只是同等规格下速度约为 NVIDIA 的六成。

笔者认为，这组数字的关键意义不在于"快"，而在于 **跨过了可用性阈值** ——94 tokens/s 的生成速度已经比人读得快，瓶颈从机器转移到了人。本地大模型从"玩具演示"变成了"日常工具"。

## 你需要什么样的硬件

| 项目 | 要求 |
| --- | --- |
| 显卡 | NVIDIA RTX 20/30/40/50 系列，或 AMD RX 7900 XT/XTX、7800 XT/7700 XT、9060 XT、9070/9070 XT、AI PRO R9700、RX 6800/6900 系列； **显存 12 GB 起步** |
| 内存 | 32 GB 起步，64 GB 可以跑全部规格；内存大小决定能装下哪个模型 |
| 硬盘 | 约 80 GB 空闲空间，强烈建议 SSD（首次启动快很多） |
| 系统 | Windows 10/11 或 Linux，配最新显卡驱动 |

安装器会搞定其余一切。两三张显卡还可以并联分担模型（multi-GPU）。

社区还在自己的机器上实验性验证了更多边缘配置：Tesla P40/V100、GTX 10 系、Radeon VII、RX 6700 XT 等老卡，从源码编译的 Intel Arc，甚至不支持 AVX2 的老 CPU（能跑，但很慢）。

## 安装：一句话交给 AI，或者双击一个脚本

安装路径有两条，设计上非常照顾不同习惯的读者。

**路线一：让 AI 编程助手代劳。** 如果你在用 Claude Code、Cursor、Codex、GitHub Copilot 这类工具，直接把这句话贴给它：

```
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

它会自动检测你的显卡、内存和硬盘，选出匹配的模型规格，完成安装、启动，并告诉你如何连接其他应用。AI 工具甚至可以通过 Strata 的 MCP server 来安装、启动和停止它。

**路线二：自己动手。** 下载解压（或 git clone）后，Windows 双击 `START-HERE.bat`，Linux 在目录里运行 `./setup.sh`。NVIDIA 和 AMD 的步骤完全一致，安装器会自动识别显卡并配置对应引擎。过程中只问三个问题：选哪个模型和规格、要多少上下文长度、要不要读图能力。一路回车就是推荐答案。随后自动下载约 70 GB 的模型文件并启动，下载中断后再跑一遍脚本会断点续传。启动完成后浏览器自动打开 `http://127.0.0.1:8080`。

有一个细节必须提醒首次使用者：

> 模型启动时，你的电脑可能卡顿或无响应 1-3 分钟（第一次最久）。Strata 会往内存里加载 35-55 GB 并锁定一部分给显卡，这是正常现象。耐心等待，不要关窗口，窗口里会显示加载进度。

之后每次使用只需再跑一次启动脚本，秒启动、不重复下载。`UPDATE.bat`（或 `./update.sh`）负责只更新不启动。

## 模型怎么选：一张表对照你的内存

安装器会按内存推荐规格，但理解背后的取舍有助于自己决策。同一个模型有多个量化规格： **越小越快，越大越聪明** 。

| 你的内存 | 推荐选择 | 原因 |
| --- | --- | --- |
| 32 GB | Coder | 唯一能塞进 32 GB 的版本，专为代码优化（配 24 GB 显卡时 Q2_0 和 IQ2_XS 也能跑） |
| 48 GB | IQ2_XS（或最快的 Q2_0） | 更大的规格放不下 |
| 64 GB | IQ2_XS（推荐），或 IQ3_XXS / IQ3_S | 全部规格都能装；IQ3_S 质量最高也最慢 |
| 96 GB+ | IQ3_S 或 Unsloth UD-IQ4_XS（约 4-bit） | 可以在开着其他程序的同时跑最大规格 |

几个特殊版本值得单独说明：

- **Coder** ：砍掉一半专家的代码特化版。SWE-bench Verified 得分达到完整模型的 91%（其作者实测），且能塞进 32 GB 内存。代价是代码之外的能力变弱，包括中文和其他 CJK 文本——需要中文能力请选保留全部专家的 Q2_0、IQ2_XS 或 IQ3_S。
- **Swift 1.5** ：一个思考时间大幅缩短的微调版，回答来得更快，质量基本持平。
- **Unsloth UD-IQ4_XS** ：约 4-bit 量化，质量介于 IQ3_S 和 UD-Q4_K_XL 之间，下载量 94 GB。内存不足约 80 GB 时会边回答边从 SSD 读数据，速度变慢（NVMe SSD 有缓解）。
- **Unsloth UD-Q4_K_XL（实验性）** ：最接近完整模型的版本，但大部分权重需要从 SSD 流式读取，64 GB 内存的机器上生成速度只有 7-8.5 tokens/s——属于"能跑但不实用"的探索档位。

这个版本矩阵的设计很务实：它不是让你追求"最还原"的模型，而是帮你在自己的硬件预算内找到"质量够用的最快版本"。

## 用起来是什么体验

![runpagoda.png](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---Niko1221-Strata-Qwen3.8-Flash-Next-on-any-consumer-hardware-one-click-install-for-Windows-Linux.-Strata-inference-engine-OpenAI-Anthropic-API-on-localhost-optional-image-input/images/runpagoda.png)

> 图解：左边是 Strata 应用的 Monitor 面板，实时显示模型状态和 GPU/CPU/RAM 占用；右边是一个编程 agent 正在生成前文视频中的宝塔花园。这张图展示的是 Strata 的典型工作流——本地模型作为后端，编码 agent 作为前端，两者通过标准 API 协作。

日常使用入口有三类：

- **浏览器** ：打开 `http://127.0.0.1:8080`，内置 Chat（聊天）、Monitor（GPU/CPU/RAM 实时监控）和 About（设置与地址信息）三个页面。
- **对接你的应用和编程 agent** ：添加一个"OpenAI 兼容"的 provider，base URL 填 `http://127.0.0.1:8080/v1`，API key 和模型名随便填都能用。用 Anthropic API 的应用（如 Claude Code）指向 `http://127.0.0.1:8080/v1/messages`，设置 `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` 即可；Codex CLI 等使用 OpenAI Responses API 的应用走 `/v1/responses`。
- **手机或局域网内其他电脑** ：以 `--host 0.0.0.0 --api-key <密钥>` 启动即可开放访问，切记一定要设密钥。

几个进阶开关：

- **思考深度** ：off / low / medium / high 四档，off 最快，high 适合难题，可在聊天菜单或应用的 "reasoning effort" 参数中切换。
- **读图** ：安装时对 "Images?" 回答 yes，之后可在聊天里点 Picture 或在应用中附图。注意 AMD 显卡目前只能在 Linux 下通过 CPU 读图，Windows 暂不支持。
- **并发** ：默认一次只处理一个请求，其余排队；设 `"parallel": 2` 可并行回答，但 12 GB 显卡上每个回答会变慢。
- **长提示** ：一段对话的第一条消息会被完整读取，约每分钟 3 万 token；后续消息几秒内即可开始。

## 工作原理：厨房、专家团与"先猜后验"

终于来到最硬核的部分。解决了"能装上、能用上"之后，下一个问题是：它凭什么能跑起来？

作者用了一个很贴切的厨房比喻： **常用的调料放台面，不常用的收进储藏室** 。关键洞察前面已经提过——这个模型是 24,576 个专家的团队，生成每个词只需要其中 10 个。

![how-it-works.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---Niko1221-Strata-Qwen3.8-Flash-Next-on-any-consumer-hardware-one-click-install-for-Windows-Linux.-Strata-inference-engine-OpenAI-Anthropic-API-on-localhost-optional-image-input/images/how-it-works.svg)

> 图解：Strata 的三级存储调度。显卡（GPU）保存最常被调用的几千个"热专家"，内存（RAM）容纳全部 24,576 个专家并由 CPU 并行处理其余部分，SSD 上则存放一张大型查找表用于快速定位。整套系统的本质是：用容量换显存，用命中率换速度。

这个设计的聪明之处在于，它把 MoE 架构的"稀疏激活"特性吃干榨净。稠密模型（dense model）的每个参数每次都要参与计算，拆分到不同介质上会被传输延迟拖死；而 MoE 模型每个词只激活 10/24576 个专家，只要"热专家"缓存命中率够高，绝大部分计算都发生在显卡上，慢速介质的访问被摊薄到可以忽略。

第二项加速技术是 **投机解码** （Speculative Decoding），作者称之为 "Guess, then check"：

![guess-and-check.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---Niko1221-Strata-Qwen3.8-Flash-Next-on-any-consumer-hardware-one-click-install-for-Windows-Linux.-Strata-inference-engine-OpenAI-Anthropic-API-on-localhost-optional-image-input/images/guess-and-check.svg)

> 图解：一个小的辅助模型先快速"猜"出接下来的几个词，大模型一次性并行验证这些猜测。验证通过的部分直接采纳，答案与逐词生成完全一致，但整体速度快了 1.6-1.8 倍。

为什么"先猜后验"能加速？传统自回归生成是一个词一个词串行蹦出来的，每蹦一个词都要把整个模型跑一遍；而"验证一批猜测"可以并行完成，一次前向计算就能确认多个词。猜得准，一次顶好几次；猜错了大不了回退，结果严格等价，质量零损失。

第三项优化针对长文本：提示词被切成最多 8,192 token 的大块批量读入，读取吞吐超过 1,000 tokens/s——这就是前面实测表里的 32K 文档一分钟读完的由来。

三板斧合起来—— **MoE 专家分层调度解决"装得下"，投机解码解决"生成快"，大块批量读取解决"吃得快"** ——构成了 Strata 把服务器级模型塞进游戏 PC 的完整技术图景。

## 踩坑指南：常见问题速查

作者在文档里坦诚地列出了新手最容易撞上的几个问题：

- **首次启动电脑卡死** ：这是模型加载期的正常现象（35-55 GB 内存占用），别关窗口。超过 10 分钟还没动静？重启电脑、关掉其他程序再来，或者换更小的规格。
- **下载或安装中断** ：再跑一遍启动脚本即可断点续传。
- **速度极慢、硬盘灯狂闪，或提示 "the engine stopped unexpectedly"** ：内存不足的典型症状。关掉浏览器等吃内存的程序，或换 Q2_0 / IQ2_XS 这样的小规格。
- **提示 8080 端口被占用** ：Strata 已经在运行了，去找它的窗口。

更多问题见仓库的 docs/TROUBLESHOOTING.md；提交 issue 时记得附上 Strata 目录下的 `strata-<model>.log`。

## 总结

- **核心命题** ：让 125B 参数的 Qwen3.8-Flash-Next 在 12 GB+ 显存的消费级显卡上本地运行，Windows/Linux 一键安装，数据完全不出本机。
- **性能实测** ：RTX 5070 上 Q2_0 规格生成 94 tokens/s、读取 2650 tokens/s，已越过"比人读得快"的可用性阈值。
- **技术三板斧** ：MoE 专家按热度分层调度（GPU/RAM/SSD 三级存储）、投机解码提速 1.6-1.8 倍、8192 token 大块批量读取突破 1000 tokens/s 吞吐。
- **生态友好** ：提供 OpenAI / Anthropic / Responses 三套兼容 API，可被 Claude Code、Codex 等编程 agent 直接接管，甚至支持 AI 助手通过 MCP 自动完成安装。
- **选型务实** ：按内存给出版本矩阵，32 GB 能跑代码特化的 Coder（SWE-bench 91%），96 GB+ 可上接近原版的 4-bit 量化。

展望来看，Strata 代表的方向比这个项目本身更重要：当模型架构（MoE 稀疏激活）与推理引擎（分层调度 + 投机解码）协同设计时，"大模型必须上云"的前提正在瓦解。它目前的局限也很明确——AMD 平台性能仍有差距、超大规格依赖 SSD 流式读取时速度骤降——但随着消费级显存容量和内存带宽的代际提升，本地运行服务器级模型的体验只会越来越好。

> 本文参考自 [GitHub - Niko1221/Strata: Qwen3.8-Flash-Next on any consumer hardware](https://github.com/Niko1221/Strata)