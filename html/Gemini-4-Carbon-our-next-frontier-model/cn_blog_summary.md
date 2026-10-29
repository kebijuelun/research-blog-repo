# Gemini 4 Carbon：持续数天修复复杂漏洞

在大型代码库里，真正耗时的不是补一行代码，而是反复排查并修复隐蔽漏洞。Google 新发布的 Gemini 4 Carbon 因此聚焦持续 Agent 工作流和网络安全：官方报告 SEC-bench Pro 88.2%、ExploitGym 54.8%、CWE-bench 71.0%，内部团队还观察到它能持续数天处理复杂工程任务。

## 长时程不等于长上下文

长时程能力关注模型能否推进相互依赖的步骤，在失败后修正路线并完成交付；上下文窗口只决定一次能读入多少信息。由于发布材料未给出任务样本量、人工接管频率或成功率，“运行数天”应视为能力案例，而非普遍稳定性承诺。

综合能力也有进步：Artificial Analysis Intelligence Index v5.0 中，Carbon 以 59 分排名第一，高于 Argon 的 50。但 FrontierSWE 2 的 61.2% 仍低于 GPT-6 Astra 的 65.5%，不能据此宣称所有编程任务都领先。

## 从发现漏洞到生成补丁

Carbon 按防御生命周期组织安全能力：发现真实漏洞、验证可利用性、针对根因修复。以下外部安全基准均在**不启用安全护栏**时评估，不代表实际产品开启防护后的表现。

SEC-bench Pro 采用 Pass@1，Carbon 一次尝试发现复杂缺陷的通过率为 88.2%，高于 GPT-6 Astra 的 85.4% 和 Argon 的 83.4%。ExploitGym 考察漏洞利用验证：Carbon 在 10 小时上限下得 54.8%，6 小时下仍有 53.5%；Argon 对应为 42.0% 和 39.5%。延长时间只增加 1.3 个百分点，真实任务的时间成本则未披露。

修复环节的 CWE-bench 中，Carbon 使用 Antigravity 得到 71.0%，Argon 及其他对手为 68.0%。但其他结果使用不同编码工具，比较的是模型与工具组合。三项结果也不保证自动解决安全问题，部署仍需人工审查、回归测试和权限控制。

## 长任务还要少走捷径、少越界

Google 报告，Carbon 在 Terminal-Bench 4.0 上的成功作弊率降至 0.0%，作弊尝试率为 0.4%；内部无依据声明较 Argon 约减少一半，过度执行和应做未做分别下降 26% 和 29%。Gray Swan 间接提示注入测试中，其攻击成功率为 0.0%，Argon 为 0.7%。这些 0% 只代表特定测试集内已评估攻击未成功，不能外推为零风险。

Carbon 初始价格与 Argon 相同：每百万输入 Token 2 美元、输出 Token 10 美元，缓存输入优惠 95%。目前先向部分网络安全防御者开放，公开日期和端到端任务成本尚未给出。

回到复杂漏洞问题，Carbon 已给出从发现、验证到修复的证据链，但数字主要来自 Google。团队仍应在自己的项目中复测补丁回归、人工接管、越界行为和总成本，才能判断“持续数天”能否转化为可靠交付。

> 本文参考自 [Gemini 4 Carbon: our next frontier model](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/)