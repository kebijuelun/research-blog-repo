# SWE-chat：斯坦福用 6000 个真实会话揭开 Coding Agent 的真相——只有 44% 的 AI 代码被留下

AI 编程助手到底提效多少、产出的代码有多少真正被采纳？斯坦福 AI Lab 发布的 **SWE-chat** 首次给出了实证答案：这是目前最大规模的"野外" coding agent 交互数据集，包含近 6,000 个会话、63,000+ 条用户 prompt 和 355,000+ 次工具调用。最扎心的两个数字：**仅 44% 的 agent 产出代码最终进入用户 commit**；全托管式 "vibe coding" 每千行引入的安全漏洞约为人写代码的 **9 倍**。

## 数据集怎么来的

- 开发者主动安装开源工具 Entire.io 的 CLI，自动记录完整会话日志，并通过 checkpoint 与 commit 关联实现**行级人/机代码归属**。
- 覆盖 Claude Code、OpenCode、Gemini CLI、Cursor 等五款主流 agent（约 85% 来自 Claude Code），横跨 200+ 个仓库。
- 分析方法双管齐下：LLM-as-a-Judge 做语义标注；更硬核的是**按时间回放每次文件修改**，精确追踪每行 agent 代码的命运（保留、被改写、被删除）。安全性用 Semgrep 对 commit 前后快照差分统计。

## 核心发现

- **真实用途远超写代码**：最常见的意图是"理解已有代码"（19%），1/3 的工具调用是 bash 命令——只考"生成 patch"的 benchmark 严重失真。
- **编码模式极端双峰**：41% 的会话是纯 vibe coding（99% 以上代码由 agent 写），23% 全人写，37% 人机协作；且 vibe coding 占比三个月内从 20% 翻倍到 40% 以上。
- **效率真相**：只有 44.3% 的 agent 代码被留下，主因是人直接删掉。按每百行提交代码算，vibe coding 烧掉约 204K token、成本 \$0.13、耗时 12.6 分钟，**全面劣于协作模式**（68K token、\$0.05、4.8 分钟）——"全自动化才是未来"的叙事被数据打脸。
- **安全真相**：vibe coding 每千行引入 0.76 个漏洞，是人写代码（0.08）的约 9 倍，类型涵盖路径穿越、命令注入、SQL 注入等。
- **自主性跑赢监督**：agent 只有 1.1%–2.6% 的轮次会主动求助，而用户在约 44% 的轮次里要打断或纠偏；P99.9 的单轮时长已超 100 分钟，silent failure 普遍存在。

## 启示与局限

SWE-chat 是持续增长的 living dataset，为三个方向铺路：用真实会话轨迹建更贴近实战的 benchmark、改进 agent 交互设计、训练用户模拟器做离线评测。局限也很明确：数据来自 opt-in 的开源早期用户，成功率可能被高估，行级归属又可能低估 agent 的真实贡献。它最大的价值是一面"持续更新的镜子"——让我们第一次看清 agent 在真实世界里干得怎么样。

> 本文参考自 [SWE-chat: Coding Agent Interactions From Real Users in the Wild](https://arxiv.org/abs/2604.20779)，原始发布见 [Stanford AI Lab on X](https://x.com/StanfordAILab/status/2106096953495036259)