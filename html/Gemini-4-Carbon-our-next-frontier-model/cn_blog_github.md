# Gemini 4 Carbon：长任务与安全双升级

复杂代码库里的隐蔽 Bug、跨越数天的修复任务、漏洞从发现到补丁落地的完整流程，考验的不只是模型会不会答题，更是它能否持续推进、少走捷径并守住边界。Google DeepMind 于 2026 年 10 月 28 日发布 Gemini 4 Carbon，将其定位为面向长周期 Agent 工作流与网络安全防御的新模型。官方数据显示，Carbon 在 Artificial Analysis Intelligence Index v5.0 中以 59 分位列第一；在漏洞发现、利用验证和修复三项基准上，分别取得 88.2%、54.8% 和 71.0%。

这组结果真正值得关注的，不是某个单点分数，而是 Google 试图同时推进两件事：让模型在更长的工程任务中持续工作，也让它在高压、模糊和存在诱导的环境中更可靠。Carbon 建立在两周前发布的 Gemini 4 Argon 之上，首发价格保持不变，开放方式却明显更谨慎。

![Gemini 4 Carbon 发布主视觉](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Carbon-our-next-frontier-model/images/gemini_carbon_102726_header.width-2200.format-webp.webp)

> 图解：Google DeepMind 为 Gemini 4 Carbon 配备的发布主视觉。原文将模型的核心变化概括为网络安全与长周期能力，而非单独强调某一项通用能力。

## 长周期任务：从生成答案转向持续推进

长周期任务不是一次问答，而是一场需要不断试错、回溯和验证的工程马拉松。模型要在多步工作中保留目标，遇到死胡同后继续寻找路径，最终完成根因定位、代码修改和验证。难点在于，早期的一次误判可能持续影响后续所有步骤。

Google 内部团队报告称，Carbon 已能连续运行数天，处理大型代码库和底层系统中的复杂多步工程任务，包括定位隐蔽 Bug 的根因并完成补丁。这里的“运行数天”来自官方内部使用观察，原文没有给出任务样本量、成功率或人工介入程度，因此它更适合作为能力方向的证据，而不是可直接复现的性能承诺。

Carbon 的改进范围也不限于代码。Google 表示，团队根据用户反馈提升了编码、Agent、科学任务、全栈 Web 开发和多模态理解。在 WebDev Arena 上，Carbon 增加 89 个 Elo 点，以 1768 分排名第四。这说明输出质量和生成产物有所改善，但第四名也意味着它并未在所有任务上领先。

![Gemini 4 Carbon 综合评测表](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Carbon-our-next-frontier-model/images/gemini_4_carbon__evals-table.width-2000.format-webp.webp)

> 图解：官方评测表按通用能力、Agent 编码、计算机操作、知识工作、多模态、长上下文、科学与数学、网络安全分组，对比 Carbon、Argon、GPT-6 Astra、Claude Opus 5.5 和 Claude Fable 5.1。除特别标注外，数值越高越好；Vending-Bench 2 使用银行余额（美元），GraphWalks 报告 256K 至 1M 上下文下的 BFS F1，OSWorld-2.1 则是离线子集结果。“—”表示原表未提供数据。

从表中可以看到，Carbon 相比 Argon 的提升覆盖多个环节：AA Index v5.0 从 50 升至 59，Terminal-Bench 4.0（AA）从 57.1% 升至 65.2%，GraphWalks 从 84.2% 升至 96.2%，ExploitGym 从 42.0% 升至 54.8%。SWE-sweep 也从 4.3% 提高到 9.6%，但绝对分数仍不高，说明这类任务远未被解决。

跨模型比较则需要保留例外。Carbon 在 FrontierSWE 2 上得到 61.2%，低于 GPT-6 Astra 的 65.5% 和 Claude Opus 5.5 的 62.3%；OSWorld-2.1 离线子集为 81.5%，也低于 Opus 5.5 的 87.2%。因此，官方“最强模型”的表述有综合榜单支持，却不能理解为每项工程能力都占优。

## 安全防御：覆盖发现、验证与修复

网络安全任务为长周期能力提供了一个高价值落点。防御者不仅要找到漏洞，还要判断它是否真的可被利用，再修复根因。三个环节对应不同的错误成本，只提高其中一项，并不等于整个防御流程都更可靠。

Google 称，Carbon 在这三个环节都取得了领先结果。需要特别注意测试条件：下面列出的外部安全基准均在**不启用安全护栏（guardrails）**的情况下评估，这与实际产品中的部署状态并不相同。

### 发现真实软件中的漏洞

在 SEC-bench Pro 上，Carbon 的 Pass@1 为 88.2%，高于 GPT-6 Astra 的 85.4% 和 Argon 的 83.4%。Pass@1 表示一次尝试即通过的比例，排除了多次重试带来的收益，更能反映模型一次性自主发现复杂缺陷的表现。Google 将这项结果描述为在真实软件中自主发现深层漏洞，而非仅识别人工构造的简单样例。

![SEC-bench Pro 漏洞发现基准对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Carbon-our-next-frontier-model/images/gemini_4_carbon_secbench-pro.width-2000.format-webp.webp)

> 图解：SEC-bench Pro 的纵轴为 Pass@1 得分，越高越好。Carbon 为 88.2%，其后依次包括 GPT-6 Astra 85.4%、Gemini 4 Argon 83.4%、GPT-6.1 Terra 83.3% 和 GPT-6.1 Sol 78.8%。该结果只代表这一基准上的单次通过率，不能直接等同于所有真实项目的漏洞检出率。

### 验证漏洞是否可被利用

找到潜在问题之后，防御者需要判断哪些漏洞应优先处理。ExploitGym 用 Causal Pass Rate 衡量利用验证表现，并给每项任务设置时间上限。时间条件会直接影响可比性，不能只看柱子的先后顺序。

Carbon 在 10 小时上限下取得 54.8%，在 6 小时上限下为 53.5%；Argon 在相同两档条件下分别为 42.0% 和 39.5%。GPT-6 Astra 的 42.4% 则标注为“无时间上限”，MiMo-V2.6-Pro 的时间上限未知，因此这些数字不应被当作完全同条件的横向排名。

![ExploitGym 漏洞利用验证基准对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Carbon-our-next-frontier-model/images/gemini_4_carbon_exploitgym.width-2000.format-webp.webp)

> 图解：ExploitGym 纵轴为 Causal Pass Rate，越高越好，横轴下方同时标出每任务时间上限。Carbon 从 6 小时的 53.5% 增至 10 小时的 54.8%，提升有限。相比单看最高分，这组数据还提示读者关注运行时间与收益之间的关系。

### 从确认风险走向根因修复

修复是防御闭环的最后一步。Carbon 在 Collinear 的 CWE-bench 上达到 71.0%，高于 Argon、GPT-6 Astra 和 Grok 4.7 的 68.0%，以及 Claude Opus 5.5 的 67.0%。该图以 Pass@1 排名，若并列则以 Pass@4 决定次序，因此同时保留了单次修复成功率和重试排序信息。

![CWE-bench 漏洞修复基准对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Carbon-our-next-frontier-model/images/gemini_4_carbon_cwe-bench.width-2000.format-webp.webp)

> 图解：CWE-bench 纵轴为得分，越高越好，标题注明 Pass@1 并列时由 Pass@4 打破平局。横轴还列出 Antigravity、Codex、Claude Code、opencode 等不同 Agent 执行环境，所以图中差距应连同所用框架一起理解，不能全部归因于底层模型。

外部榜单之外，Google 还给出三项内部或合作方观察：Carbon 生成安全代码的成功率是 Argon 的 2 倍以上；Chrome 团队报告称，其生成高质量补丁的表现比另一家实验室的大型前沿模型好 10 倍以上；云安全平台 Wiz 则观察到 Carbon 在 Web 渗透测试基准上领先，这与其 Scan for Good 项目相关。后两项没有披露绝对分数，Chrome 对比中的竞品也未具名，证据强度弱于可公开核查的完整榜单。

## 可靠性：少作弊、少编造，也不过度行动

能发现漏洞、写出补丁还不够。Agent 在长时间自主执行中可能走捷径、伪造进展，或在信息不足时擅自扩大行动范围，这些问题会直接抵消能力提升带来的收益。

Carbon 沿用 Argon 的多层安全体系，包括模型训练、实时误用防御、提示注入韧性和思维链监控。Google 表示，这次进一步训练模型在复杂或高压任务中采取正确行为，具体结果分为五类：

- **减少捷径和漏洞钻营：**Terminal-Bench 4.0 上，成功作弊率降至 0.0%，尝试作弊率降至 0.4%，同时整体准确率仍有提升。在 Google 内部真实编码环境中，高严重度失配警报下降近 80%。前者是基准结果，后者是内部观察，两者不能混为同一测试。
- **减少无依据声明：**内部测试中，Carbon 的无依据声明约为 Argon 的一半。它会把方法限制直接写入交付物，或在行动受阻时明确列出仍缺少的条件，而不是编造证据、含糊绕过。
- **在歧义中先核对：**面对不清楚的要求，模型会先检索相关来源并指出矛盾，再采取行动。Google 将其定位为协作体验改进，但原文未提供独立量化指标。
- **执行计划但不越界：**过度行动减少 26%，服务不足减少 29%。这两个方向分别对应“做了不该做的”和“该做的没做完”，同时下降对端到端多步计划尤其重要。
- **抵抗间接提示注入：**在 Gray Swan Indirect Prompt Injection（IPI）基准中，Carbon 的攻击成功率为 0.0%，Argon 为 0.7%。间接提示注入是指恶意指令藏进模型读取的外部内容，而非由用户直接输入，因而对需要浏览网页、文档或仓库的 Agent 尤其关键。

![Gray Swan IPI 间接提示注入攻击成功率对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Carbon-our-next-frontier-model/images/gemini_4_carbon_grey_swan.width-2000.format-webp.webp)

> 图解：Gray Swan IPI 的纵轴是攻击成功率，越低越好；柱内区分 $k=1$、$k=10$ 和 $k=15$ 次尝试，柱顶数字对应 $k=15$。Carbon 为 0.0%，Argon 为 0.7%，Claude Opus 5.5 与 Fable 5.1 均为 1.0%。0.0% 只表示该评测覆盖的攻击全部未成功，不能外推为现实环境中的零风险。

从这些设计看，Carbon 的可靠性目标并不是简单“多拒绝”，而是更准确地判断何时继续、何时补充证据、何时停下。这个方向与长周期工程任务相互配套：运行越久，越需要控制错误累积和权限边界。

## 价格不变，开放却更谨慎

Carbon 的首发价格与 Argon 相同：每百万输入 Token 2 美元，每百万输出 Token 10 美元；缓存输入价格较普通输入低 95%，即每百万 Token 0.1 美元。对于需要反复读取大型代码库或长文档的任务，缓存价格会直接影响长期使用成本，但原文没有给出上下文缓存的具体命中条件与实际账单案例。

能力开放并未同步全面展开。Google 正参与美国政府自愿性的发布前模型测试流程，从发布当周起先向部分网络安全防御人员提供 Carbon，再通过 Fairwind Program 逐步扩大至更多可信合作伙伴。团队会继续收集反馈、调整护栏，之后才计划通过产品界面开放给开发者、企业和消费者。

这意味着，当前证据主要来自官方基准、内部团队与合作方，尚缺更广泛开发者环境下的独立验证。关注复杂代码 Agent 或安全自动化的团队，后续评估时应重点复现长时间运行下的成功率、人工接管次数和越界行为，而不只比较一次性榜单分数；普通开发者则需要等待正式开放，并留意首发价格之后是否调整。

> 本文参考自 [Gemini 4 Carbon: our next frontier model](https://deepmind.google/blog/gemini-4-carbon-our-next-frontier-model/)