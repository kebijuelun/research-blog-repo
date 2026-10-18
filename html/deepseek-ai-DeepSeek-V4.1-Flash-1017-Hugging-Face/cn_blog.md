# DeepSeek-V4.1-Flash-1017：一次"小更新"，Agent 与编码能力的大步前进

模型版本号里带日期后缀的更新，通常只是修修补补。但 DeepSeek-V4.1-Flash-1017 是个例外：作为 DeepSeek-V4.1-Flash 的一次 minor update，它在 11 项基准上全面上涨，其中 Terminal-Bench 4.0 从 31.2 跳到 42.3，ProgramBench 从 20.3 跳到 29.1——相对提升分别约 36% 和 43%。更值得留意的是官方强调的另一件事：模型在分数变高的同时，缩短了任务执行时间和输出长度，也就是说完成任务更快、更便宜。性能与成本这对通常互相拉扯的指标，这次被同时推进了。

这个仓库本身是一个 763B 参数的图文多模态模型（Image-Text-to-Text），权重以 Safetensors 格式发布，MIT 许可。下面按模型卡片给出的信息，把这次更新值钱的几个点拆开看。

## 基准成绩：涨分集中在 Agent 和编码场景

模型卡片给出的对比表覆盖了 11 项基准，横轴是六个模型：本次发布的 DS-V4.1-Flash-1017、它的前一版 DS-V4.1-Flash、K3、GLM-5.3、GPT-5.6 Sol 和 Opus 5。完整数据如下：

| Benchmark | DS-V4.1-Flash-1017 | DS-V4.1-Flash | K3 | GLM-5.3 | GPT-5.6 Sol | Opus 5 |
| --- | --- | --- | --- | --- | --- | --- |
| HLE w/o tools | 42.6 (44.0†) | 36.8 (39.1†) | 43.5 | 42.0† | 44.5 | 56.3 |
| Terminal-Bench 4.0 | 42.3 | 31.2 | 12.6 | 37.9 | 39.9 | 51.8 |
| DeepSWE v1.1 | 77.0 | 74.2 | 67.5 | 66.9 | 73.0 | 74.0 |
| ProgramBench | 29.1 | 20.3 | 17.5 | 19.0 | 23.0 | 37.0 |
| NL2Repo-Bench | 68.2 | 65.4 | 58.0 | 58.0 | 56.8 | 75.3 |
| Automation-Bench v1.0.6 | 60.5 | 54.8 | 46.7 | 48.8 | 45.8 | 50.3 |
| Agents' Last Exam | 35.4 | 31.8 | 27.6 | 28.5 | 26.7 | 28.6 |
| Apex Bench v1.1 | 72.1 | 64.6 | 47.8 | 56.6 | 54.3 | 65.8 |
| Chartography (w/tools) | 83.0 | 78.9 | 68.1 | — | 79.9 | 84.0 |
| BabyVision (w/tools) | 92.5 | 89.6 | 85.7 | — | 88.9 | 94.1 |
| ZeroBench-main (w/tools) | 51.0 | 49.0 | 41.0 | — | 53.0 | 52.0 |

† 表示 HLE 的纯文本子集。Terminal-Bench 4.0 使用 Claude Code harness，DeepSWE v1.1 使用 Mini-SWE-V2 harness。

把这张表拆开读，能读出三层信息。

第一层是自我对比：1017 版相对前一版 11 项全胜，没有一项倒退。涨幅最大的恰好是 Agent 类任务——Terminal-Bench（终端操作）、ProgramBench、Apex Bench（+7.5）、Automation-Bench（+5.7）。这呼应了官方"reasoning、coding、agentic tasks"的表述，涨分不是均匀撒胡椒面，而是集中在需要多步执行的场景。

第二层是横向位置。在 Agent 类基准上，1017 版已经压过表内除 Opus 5 以外的所有对手：Automation-Bench 60.5 对 Opus 5 的 50.3 甚至反超，Apex Bench 72.1 对 65.8 也是领先。但在知识推理类的 HLE（42.6 对 56.3）和代码库理解类的 NL2Repo-Bench（68.2 对 75.3）上，与 Opus 5 仍有明显差距。终端操作（42.3 对 51.8）和 ProgramBench（29.1 对 37.0）同样没有追平。也就是说，这次更新把它推到了"多数 Agent 基准的第一梯队"，但还没坐到全榜第一的位置。

第三层是视觉能力。末尾三行带工具的视觉基准（Chartography、BabyVision、ZeroBench）表明这不是纯文本模型；不过这几项上它的表现与 GPT-5.6 Sol、Opus 5 互有胜负，领先优势不如编码类项目稳定。GLM-5.3 在这三项上没有数据（表中为 —），比较时要注意覆盖范围不一致。

需要提醒一点：这些数字全部来自官方模型卡片，测试条件（harness、是否带工具）已在表注中说明，但各家对比模型的分数是否都在完全相同的条件下测得，卡片没有逐项交代。引用时建议保留这个不确定性。

## 比涨分更实际的：更快、更便宜

官方在介绍里专门写了一句：新模型"reduces task execution time and output length"，并声称在多个 agent harness 上推进了性能—成本的 Pareto 前沿。

这句话对真实使用者的意义可能超过任何单项分数。Agent 场景的成本大头是长轨迹多轮调用，输出 token 变少、单任务耗时变短，直接折算成 API 账单和等待时间。不过卡片没有给出具体的时延或 token 消耗数字，这条结论目前只能以官方表述为准，实际收益需要在自有工作流里实测。

## 部署前必须知道的两件事

涨分之外，模型卡片里对开发者最有操作价值的信息，集中在 Prompt Encoding 一节——这里有一个容易踩的坑。

**没有 Jinja chat template。** 这次发布不提供 Hugging Face 生态里常见的 Jinja 格式聊天模板，意味着不能指望 transformers 的 `apply_chat_template` 直接处理对话格式。官方给了两条替代路径：

1. `encoding` 文件夹里有一个自包含的 Python 参考实现（`encoding.py`），附带测试用例，覆盖多轮对话、工具调用、thinking 模式、数值推理强度（numeric reasoning effort）、对话中途插入 system 消息、图文交错内容等场景。适合读源码、理解格式。
2. 生产环境建议用官方同步发布的 **deepseek-recipe** ——一组带 Python 绑定的 Rust 库。它能把 Messages、Chat Completions、Responses API 三种请求格式转成内部的 Conversation 格式，编码成 DeepSeek V4 / V4.1 的 prompt 或 token ID，再把模型输出解析回完整或流式响应，覆盖 thinking、工具调用、图像和生成参数。注意它的边界：模型推理、工具执行和 HTTP 传输都留给调用方自己实现，它不是推理框架，只是协议层的编解码工具。

**推理参数与上下文。** 仓库的 `inference` 文件夹提供了权重转换和本地推理的说明，官方推荐的采样参数如下：

| Parameter | Value |
| --- | --- |
| temperature | 1.0 |
| top_p | 1.0 |
| context_window | 1M tokens |
| max_tokens | ≥ 256K |

1M token 的上下文窗口配上不低于 256K 的输出上限，规格上明显面向长轨迹 Agent 和整库代码理解这类任务——这与前面基准表中 NL2Repo-Bench 这类项目的定位一致。temperature 和 top_p 都给到 1.0，说明官方认为模型在不做截断采样时表现最佳，沿用惯性把 temperature 压到 0.2 之类的做法在这里可能反而损失能力。

权重方面，模型规模 763B 参数，仓库内包含 BF16、F32、F8_E4M3、I8 多种张量类型，标签中也标注了 fp8 和 8-bit precision，说明官方直接提供了量化版本。763B 的体量决定了本地部署门槛不低，具体转换步骤以 `inference` 文件夹为准。

## 许可与引用

仓库和模型权重均采用 **MIT License** ，是目前主流开源许可中最宽松的一档，商用没有额外限制。官方给出的引用条目（标题为 "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"，年份 2026）透露了这条产品线的技术主线是 KV Cache 压缩——这也解释了为什么它能把 1M 上下文做成标配，不过本次 1017 更新卡片本身没有展开技术细节，想深挖的读者需要去翻配套的技术报告（卡片中留有入口）。

## 结语

回到开头的问题：一个带日期后缀的 minor update 值得关注吗？从数据看，值得关注的人群很具体——正在用 DS-V4.1-Flash 做 Agent 或编码工作流的开发者，这次升级没有倒退项，且在终端操作、自动化等场景上收益最大，同时官方承诺更低的执行成本；正在选型中文开源模型做 Agent 底座的团队，1017 版在多项 Agent 基准上已进入表内前二，且 MIT 许可没有商用障碍。证据的边界也要说清楚：所有分数来自官方卡片，视觉基准的领先不稳定，HLE、NL2Repo-Bench 等项目与 Opus 5 仍有差距，"更快更便宜"尚无公开数字支撑。想动手的话，第一步不是下载权重，而是先读 `encoding.py` 或接入 deepseek-recipe——没有 Jinja 模板这一点，会直接决定你的接入成本。

## 原文图表补充

![DeepSeek-V4.1](images/logo.svg)

Technical Report 👁️

![figure-2.svg](images/figure-2.svg)

21 models

> 本文参考自 [deepseek-ai/DeepSeek-V4.1-Flash-1017 · Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)