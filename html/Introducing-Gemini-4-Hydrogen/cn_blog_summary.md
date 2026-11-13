# Gemini 4 Hydrogen：四级思考与多模型分工，加速智能体执行

Google DeepMind 在 2026 年 11 月 12 日发布 Gemini 4 Hydrogen。它要解决的不是所有深度推理问题，而是智能体工作流中更常见的执行环节：快速写代码、跑测试、读仓库、调用工具并验证结果。Hydrogen 提供 low、medium、high、xhigh 四级思考模式，默认使用 high，让用户按任务调整质量、延迟与成本。

## 把深度判断与快速执行拆开

架构分析、隐蔽缺陷排查可交给 Carbon、Argon 等前沿模型，明确的实现与验证则交给 Hydrogen。在 Google Antigravity 中，Orchestrator 能在同一会话动态切换模型；Carbon 也可担任协调者，由 Hydrogen 驱动 worker 和 verifier 智能体，在隔离工作区并行构建和测试。

Hydrogen 的整个 100 万 token 输入窗口统一定价：每百万输入 token 1.50 美元、输出 7.50 美元。Google 报告，Orchestrator 在 Terminal-Bench 4.0 上的任务成功率接近 Carbon，同时平均减少 24% 配额消耗、缩短 19% 完成时间。这是官方测试结果，并非 Hydrogen 全面优于 Carbon，实际收益仍取决于任务与分工方式。

## 升级价值在编码，也在完整交付

相较 Gemini 3.8 Flash，Hydrogen 在默认 high 档下完成任务的速度达到 2.5 倍，Terminal-Bench 4.1 得分从 18.2% 升至 57%，官方还称虚构陈述减少超过 5 倍。它强调范围集中的代码修改，并可批量并行读取、搜索仓库文件。合作伙伴的早期测试则观察到 2 至 2.8 倍速度提升、12% 至 23% 的 token 用量下降，但不能直接外推到所有业务。

Hydrogen 原生支持 100 万 token 输入和最高 100 万 token 输出，可处理 PDF、表格、图表、音频与视频。官方案例涵盖共享体素岛屿 Geminisland、含 WebGL2 着色器的电商店面、在 Google Sheets 中播放动态像素场景，以及生成并自主检查 Slides。它们展示了工作流如何串联，却没有统一披露成功率和人工修改量，不能视为一次生成同等成品的保证。

## 更快，也要验证边界

四档思考不必始终开到最高，但官方没有给出各档完整价格与延迟表。安全方面，官方报告 Hydrogen 相较 Gemini 3.8 Flash 在智能体任务中的过度越界最多减少 81%、服务不足最多减少 79%；“最多”只代表特定评测的最大改进，内部诚实性评测也不等于外部审计。

Hydrogen 已逐步开放，可通过 Gemini API、AI Studio、Antigravity、Gemini 应用和 Google Search 等入口使用。回到智能体执行这一起点，它的价值在于组合快速执行、长上下文、多模态输出与分级推理。现有证据主要来自 Google 及合作伙伴，落地前仍应使用自己的仓库和工具链，分别检查成功率、延迟、token 成本与权限边界，再决定四档思考和多模型分工是否值得启用。

> 本文参考自 [Introducing Gemini 4 Hydrogen](https://deepmind.google/blog/introducing-gemini-4-hydrogen/)