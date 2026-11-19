# Gemini 4 Neon：四档思考平衡复杂任务与成本

Google DeepMind 发布的 Gemini 4 Neon，试图解决一个实际问题：高频的编码、资料分析和多步 Agent 任务，如何兼顾完成质量、响应时间与成本。在 Artificial Analysis Intelligence Index 的 xhigh 思考等级下，Neon 得分 61、总体排名第 3；在 Vals Index 排名第 2，并进入性能与成本的帕累托前沿。

Neon 提供 100 万 token 上下文窗口，输入、缓存输入和输出价格分别为每百万 token 3 美元、0.15 美元和 15 美元。四档思考等级让用户按任务难度分配推理深度，默认 high 面向日常生产任务，xhigh 则用于更困难的多步推理。

## 长流程成绩不能脱离执行框架

企业任务往往混合电子表格、PDF、图表、音频和视频。Neon 可在同一上下文窗口处理这些资料；官方引用的评估中，它以 70.9% 位列 Vals Index 第 2，并在 Legal Research Bench 和 Tax Agent Bench 均排名第 1。Agent Arena Index 覆盖 4,200 多个真实会话，Neon 以 1336 分排名全球第 2。

编码方面，Neon 在 Artificial Analysis Coding Agent Index 以 58.6 分排名总体第 1，在 Terminal-Bench 4.1 得到 65.2%。更能说明问题的是 SWE-sweep：同一模型在 mini-SWE harness 中仅得 12.8%，换用 Google Antigravity 后达到 39.0%。因此，比较 Agent 编码能力时，模型、工具与执行框架不能混为一谈。官方展示的 3D 场景、滑雪模拟器等案例体现了复杂工具链中的端到端执行，但仍不能替代真实仓库和维护成本验证。

## 成本可调，安全仍有条件

按 Google 引用的第三方结果，Neon 在 high 档的单任务成本比 Carbon 低 32%，比头部竞争前沿模型低约 50% 至 80%；切换到 xhigh 后，其单任务成本仍比竞争模型各自最高设置低至少 36%。这里比较的是完成一次任务的总成本，而非单纯的 token 单价。

Neon for cyber 覆盖安全代码生成、漏洞发现、风险验证和修复：困难级 AutoBaxBench 得分 76%，WebExploitBench v1.2 得分 51.26%，SEC-bench Pro 达 90.5%，Collinear CWE-bench v2 得 70.7%、比 Carbon 提升 3.4%。部分测试同时报告约一半的竞争方案成本，但该版本仅向 Fairwind Program 中符合条件的可信防御者逐步开放。

安全数字也需保留条件。Google 称 Neon 在 Gray Swan 间接提示注入测试中，开启实时防护并使用最高思考等级时攻击成功率为 0.0%；这一结果不能外推到所有配置。相关数据多来自官方内部评估，生产环境仍需权限边界、工具审计和人工复核。

在 Gemini 4 系列中，Carbon 面向最难、最长的项目，Hydrogen 负责快速日常执行，Neon 则服务复杂但高频的工作。对有多文件编码、跨格式分析或多步 Agent 需求的团队，更稳妥的验证方式是先用 high 建立质量与成本基线，再只对困难任务启用 xhigh，并单独记录执行框架的影响。

> 本文参考自 [Introducing Gemini 4 Neon](https://deepmind.google/blog/introducing-gemini-4-neon/)