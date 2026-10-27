# vLLM：约束解码让工具 Schema 准确率升至 100%

Ubicloud 团队用 vLLM 0.28 部署 Kimi K2.6 时，只有 92% 的工具调用符合预期 schema，约每 25 次就有 2 次失败。对需要连续调用工具的 Agent，错误信息会进入后续上下文，重试还会增加成本。团队最终通过约束解码，在 K2VV ToolCall 基准上把 schema 准确率提升到 100%：生成时直接屏蔽非法 Token，而不是输出完整 JSON 后再报错。

## 工具调用为何会失控

模型调用工具前，需要接收工具名、参数类型、必填项和枚举值等 schema。例如 `webfetch` 必须收到字符串 `url`，返回格式只能是 `text`、`markdown` 或 `html`。推理引擎先把这些声明包装成模型专用 Token；模型生成调用后，解析器再剥离标记，转成 Agent 框架可执行的 JSON。任一环节不符合约定，调用都可能失败。

Moonshot AI 发布的 K2VV ToolCall 包含 2000 个请求，分别衡量“触发相似度”和“schema 准确率”：前者比较服务与 Moonshot AI 自家服务的调用时机，后者检查已生成的调用是否结构合法。Ubicloud 报告的两个数字只对应后者。即使达到 100%，也不代表工具选择、参数语义或最终任务全部正确。

自由生成的根因是模型只按概率选择 Token，并不执行类型检查。当前工具叫 `web_search`，它可能输出训练中学到的 `search_web`；schema 要求 `location`，它也可能填写 `city_name`。

## 把事后检查改成生成时拦截

约束解码像实时限制自动填空的选项：推理引擎根据当前语法位置计算合法 Token 集合，掩码排除其余候选，模型再从合法集合中采样。vLLM 借助 XGrammar 描述 Kimi 的固定路径：工具调用开始、工具名、调用索引、参数开始、符合 schema 的 JSON、调用结束。必填字段、类型和枚举值由此成为运行时保证。

![XGrammar 在生成过程中屏蔽非法 Token 的示意图](images/6ace25a7011f2d0bca7a06ea_LLM-inference.png)

> 图解：XGrammar 根据结构化生成的位置计算合法 Token 集合，并掩码排除非法候选。图源自 Dong、Yixin 等人的论文《XGrammar: Flexible and efficient structured generation engine for large language models》（MLSys 2025）。

原文称实际性能开销可以忽略，因为 XGrammar 会预先计算掩码并高效应用到 GPU。但文章没有延迟或吞吐数据，不能据此推断所有部署都毫无成本。

## 有效不等于适合默认开启

Ubicloud 最初在自有分支实现该功能；vLLM 从 v0.25 开始支持后，团队切换到上游实现，用于 Kimi 和 DeepSeek，准确率却回落到 91%。排查发现实现本身没有出错，变化在于未显式开启功能的请求被调整了默认行为。

通用推理引擎需要适配不同模型。强制约束可能让模型偏离正常生成路径，也可能在语法转换或与投机解码组合时引发失败；gpt-oss-20b、Qwen 3.6 和 GLM 5.1 的用户都报告过相关问题。因此，vLLM 在 0.31 中以服务端参数 `--tool-strict-level` 提供按需开启的严格模式。

Ubicloud 则在目标模型上重新运行基准，确认所有已服务版本的 schema 准确率均达到 100% 后才部署生产。这次修复真正值得借鉴的是：分开测试触发决策与结构合规，检查实际版本和开关，并覆盖真实功能组合。回到最初的 92%，约束解码消除的是可避免的格式错误，而非替模型保证每一次工具调用都正确。

> 本文参考自 [Fixing Tool-Call Accuracy in vLLM](https://www.ubicloud.com/blog/fixing-tool-call-accuracy-in-vllm)