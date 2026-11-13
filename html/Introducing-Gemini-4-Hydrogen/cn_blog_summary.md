# Gemini 4 Hydrogen：四档推理加速日常编程与智能体执行

一个定位“快速执行”的模型，在默认 high 档的 Artificial Analysis Intelligence Index 上得到 51 分，略高于擅长长程推理的 Gemini 4 Argon 的 50 分；切到最高 xhigh 档则为 54 分。Google 同时给出百万 token 上下文内的统一价格：每百万输入 token 1.50 美元、输出 token 7.50 美元。它要回答的问题不是旗舰模型能否作出艰难架构判断，而是日常编码、工具调用与智能体工作流能否更快、更可控地推进。

## 把决策与执行分给不同模型

Google 将 Carbon 和 Argon 留给系统设计、隐蔽缺陷排查等深入推理，把 Hydrogen 定位为写代码、跑测试和验证修改的执行引擎。Antigravity 新增的 Orchestrator 会在同一会话中动态切换 Carbon 与 Hydrogen；需要并行时，还可由 Carbon 协调，Hydrogen 驱动执行和验证智能体，在隔离工作区中同时构建、测试。

Google 报告，这套编排 Terminal-Bench 4.0 上的任务成功率接近 Carbon，平均减少 24% 的配额消耗，完成速度提高 19%。节省成本的关键因此不只是压低 token 单价，而是避免让高成本模型承担所有常规步骤。

## 四档推理覆盖完整执行链

Hydrogen 也能独立工作，提供 low、medium、high 和新增的 xhigh 四档思考级别，默认 high。团队可把快速档用于常规修改，把更高档留给困难分析。Google 称，相比 Gemini 3.8 Flash，Hydrogen 在默认 high 档完成任务的速度达到 2.5 倍，Terminal-Bench 4.1 得分从 18.2% 提高到 57%，虚构陈述减少超过 5 倍；这里的 4.1 与上述编排测试的 4.0 是不同版本。

合作伙伴在复杂图、SQL 和数据密集型企业流程的早期测试中，还观察到相比 3.8 Flash 运行速度提高 2 至 2.8 倍、token 用量减少 12% 至 23%。这些结果有明确的比较对象与测试范围，不能直接外推到所有项目。

从迭代式仓库编码、开放式研究到 Web 开发，Hydrogen 强调局部修改、并行文件读取与搜索，以及受阻后的动作调整。它原生支持 100 万 token 输入和最多 100 万 token 输出，可处理 PDF、电子表格、图表、音频和录像；在 Multimodal Knowledge Agent 基准上得到 53.4%，在 Agents’ Last Exam v1.1 上得到 71.6%。

## 更快，也要守住边界

Google 报告，相比 3.8 Flash，Hydrogen 在智能体任务中的过度执行最多减少 81%，服务不足最多减少 79%；工具使用和编码评估中的不对齐动作与走捷径行为减少 12 倍，内部智能体诚实性评估达到 96.7%。面对间接提示注入，它在 Gray Swan IPI 基准上的抵抗力提高超过 2.3 倍。

这些数字多来自 Google 内部评估或指定基准，“最多减少”也只对应特定测试情形。Hydrogen 已逐步开放，可通过 AI Studio、Gemini API、Antigravity、Gemini 应用和 Search 等入口使用，具体权限取决于订阅与企业方案。Google 尚未给出每种真实业务的最佳选档规则；要判断它是否适合自己的日常执行工作，仍需在自有仓库和工具链上测量质量、延迟与 token 成本。

> 本文参考自 [Introducing Gemini 4 Hydrogen](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-hydrogen/)