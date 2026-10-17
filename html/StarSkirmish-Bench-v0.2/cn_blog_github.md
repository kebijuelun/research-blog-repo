# StarSkirmish Bench v0.2：一小时写 Bot，仍难敌顶尖人类

写出一个能编译的函数，和写出一个能持续作战、还会根据失败经验调整策略的 Bot，是两种不同难度的编程任务。StarSkirmish Bench v0.2 把后一种挑战放进《星际争霸：母巢之战》：每个模型只有 1 小时，从只会采矿的骨架开始，用 C++ 编写 Protoss（神族）Bot，再用固定对手阵容上的实际胜率评分。

它考察的不只是代码生成，而是长时程推理与 agentic coding（代理式编程）：模型要编译、练习、阅读比赛记录，并把反馈带回下一版代码。结果 GPT-6 Astra 以 57 分领先，GPT-6.1 Sol 得到 46 分；但 Astra 面对 Stardust、PurpleWave 和 BananaBrain 三个顶尖人类编写 Bot 时，胜率仍只有 7%。

## 一小时里，模型要完成的不只是写代码

这个 benchmark 的设计像一场带反馈的编程冲刺。模型从只会采矿的 skeleton bot 出发，可以查看三个 demo Bot 的源码，并使用 FAP（战斗模拟器）和 BWEM（地图分析器）。它还能编译 Bot、发起练习赛，阅读每场比赛的摘要、transcript（比赛过程记录）以及 Bot 自己的日志。时间一到，Bot 文件夹中的内容就是最终版本，因此保持代码能够编译本身就是交付要求。

练习对手按强度分层，模型可以自主选择要把时间花在哪一层：

| 练习层级 | 对手 |
| --- | --- |
| D | 3 个 demo Bot |
| C、B | 各有 3 个中等排名的人类编写 Bot |
| A | BananaBrain、Locutus |
| S | Stardust、PurpleWave |

正式比赛是 Protoss 对 Protoss，基于 BWAPI 4.4.0，在 OpenBW 上运行，使用 Heartbreak Ridge、Benzene 和 Destination 三张 ladder 地图。一方建筑全部被摧毁时比赛结束；如果打到 60 分钟（86,400 帧）上限，就按 BASIL ladder 的规则比较 kills + razings（击杀与摧毁建筑）得分，分高者获胜。

这套规则让“学习”有了具体的闭环：写代码只是起点，模型还必须通过练习赛发现问题，并在倒计时结束前把改进保存下来。它也解释了为什么 StarSkirmish 更适合观察长时程 agentic coding，而不是简单衡量一次代码补全的准确率。

## 排行榜看胜率，也看单次成本

v0.2 的 Bench 分数是 5 次运行各自最终 Bot 的平均胜率，范围为 0–100。9 个模型的结果如下。这里的分数不是 Elo，而是固定对手阵容上的平均胜率。

| 排名 | 模型 | Bench 分数（5 次运行平均胜率，%） |
| --- | --- | ---: |
| 1 | GPT-6 Astra | 57 |
| 2 | GPT-6.1 Sol | 46 |
| 3 | Gemini 4 Argon | 39 |
| 4 | Claude Opus 5.5 | 39 |
| 5 | Claude Fable 5.1 | 37 |
| 6 | Claude Sonnet 5.5 | 26 |
| 7 | Muse Wonder Full | 21 |
| 8 | Claude Haiku 5.5 | 21 |
| 9 | GPT-6 Luna | 13 |

GPT-6 Astra 与第二名拉开了明显差距。Gemini 4 Argon、Claude Opus 5.5 和 Claude Fable 5.1 分别为 39、39、37 分，原文将三者视为实质并列，而不是过度解读细小的名次差异。Google 和 Meta 的新旗舰替代了 v0.1 的 Gemini 3.8 Flash 与 Muse Spark 1.3，分别取得第 3 和第 7；不过，v0.2 同时改变了运行工具与推理档位，不能把这个名次变化单独归因于模型换代。

成本图提供了另一种排序方式：横轴是每次 1 小时运行的平均 API 成本，并采用对数坐标。GPT-6.1 Sol 是唯一落在原文所称“最有吸引力象限”的模型，它排名第二，单次成本仅略高于 GPT-6 Astra 的十分之一。这里的性价比只针对本 benchmark 的分数与 API 成本，不能直接推广到所有编程任务；Muse Wonder Full 的价格则按 Muse Wonder 的 API 定价估算。

不过，固定阵容上的高分并不等于已经击败最强的人类编写 Bot，下一组对照给出了更严格的参照。

## 对顶尖人类 Bot 的胜率，最高只有 7%

57 分并不意味着已经击败最强的人类编写 Bot。原文另外统计了 5 个最终 Bot 对阵 Stardust、PurpleWave 和 BananaBrain 的平均胜率，结果如下。注意，这一列只针对这三个顶尖对手，与上表的 13 个固定对手阵容不是同一统计口径。

| 模型 | 对三大顶尖人类 Bot 的平均胜率 |
| --- | ---: |
| GPT-6 Astra | 7% |
| GPT-6.1 Sol | 4% |
| Gemini 4 Argon | 2% |
| Claude Opus 5.5 | 2% |
| Claude Fable 5.1 | 2% |
| Claude Sonnet 5.5 | <1% |
| Muse Wonder Full | <1% |
| Claude Haiku 5.5 | <1% |
| GPT-6 Luna | 0% |

这组对照说明榜单远没有饱和：即便是第一名，面对最强的人类 Bot 也只赢 7% 的比赛。与此同时，7% 也有明确的适用边界。1 小时推理限制是影响因素，而这些比赛只在三张双人地图上进行；原文指出，真实的人类 Bot 锦标赛条件明显更难。因此，不能把单一数字当成模型在所有《星际争霸》环境中的绝对能力。

要理解这些差距从何而来，还需要看模型在这一小时里如何选择练习对象、修改代码和组织战术。

## 行为轨迹揭示了分数之外的差异

StarSkirmish 保存每次运行的 agent trace（代理执行轨迹），因此可以看到模型如何分配这一小时，而不只是看最后的 Bot。原文把每个模型中位运行（median run）期间对 7 个最强阵容 Bot 的练习局数，与 Bench 分数进行对照，并用 9 个模型的线性拟合展示关系。在这些记录中，针对强对手的练习量是最强的分数预测因素。

GPT-6 Astra、GPT-6.1 Sol 和 Gemini 4 Argon 很早就开始挑战这些强 Bot，即使自己的 Bot 当时仍然打不过它们。Muse Wonder Full、Claude Haiku 5.5 和 GPT-6 Luna 则很少或从不这样做。Claude Fable 5.1 是一个值得保留的例外：它几乎只在较弱对手上练习，仍得到 37 分。

这个例外避免了一个过强的结论：练习强对手与高分同时出现，不等于练习强对手单独造成了高分。更稳妥的看法是，StarSkirmish 把“是否主动寻找更有挑战性的反馈”变成了可观察的行为指标，为解释分数提供了线索，但还没有单独证明其中的因果关系。

原文对新旗舰的行为分析，也展示了不同的编程与战术路线。Gemini 4 Argon 偏好 one-shot（一次性写出）Bot，5 次运行的构建次数和代码版本数在所有被追踪的运行中最少。它 41% 的练习局针对高层级对手，仅次于 Astra，但在这些对手身上的 Bot 胜率只有 7%，Astra 为 28%。这里的 7% 针对的是高层级练习对手，和前文针对三大顶尖 Bot 的 7% 不是同一口径。

Gemini 4 Argon 场均完成 2.5 项升级，为全场最高；它第二好的 Bot 在超过 70% 的对局中生产 Reaver。Muse Wonder Full 则几乎只选择最弱的练习对手，只有 1% 的练习达到 B 级或更高。它的典型策略是先扩张，建造 5–9 个 Gateway，等兵力超过已经观察到的敌军再进攻；场均生产 22 个 Zealot，延续 v0.1 的重 Zealot 风格，场均 3 个 Observer 则为所有模型最高，是全场平均的 1.6 倍。两个案例放在一起看，练习对象、代码迭代次数和战术选择可能形成不同组合，不能只用“练得多”解释排名。

同一模型的 5 次运行也并不稳定。原文用点表示每次运行的最终 Bot，用线段标出最低到最高成绩，用横杠表示均值，并给出一个标准差（$\sigma$）的色带；各模型的 $\sigma$ 如下，单位为胜率百分点。

| 模型 | 5 次运行的标准差 $\sigma$（百分点） |
| --- | ---: |
| GPT-6 Astra | 3.4 |
| GPT-6.1 Sol | 6.0 |
| Gemini 4 Argon | 7.0 |
| Claude Opus 5.5 | 3.2 |
| Claude Fable 5.1 | 2.9 |
| Claude Sonnet 5.5 | 8.4 |
| Muse Wonder Full | 3.0 |
| Claude Haiku 5.5 | 6.1 |
| GPT-6 Luna | 5.6 |

Gemini 4 Argon 的波动尤其明显：最好与最差运行相差 18 个百分点，而 GPT-6 Astra 和 Claude Opus 5.5 分别为 9 和 8 个百分点。由此看，单次运行的冠军或失败都不足以代表模型，5 次平均分和运行间差异应该一起阅读。

这些分数和行为差异，还需要放回统一的比赛设置中理解。

## 固定阵容让分数可复测

v0.2 的 tournament setup（锦标赛设置）用固定对手取代了模型之间的相互对战。每个模型的 5 个最终 Bot 都要对阵同一套 13 个对手；每个 Bot 与每个对手在三张地图上各打 6 场，使用新的随机种子。

| 项目 | v0.2 设置 |
| --- | --- |
| 最终 LLM Bot | 45 个（9 个模型 × 每模型 5 个） |
| 固定对手 | 13 个：10 个竞技人类编写 Bot + 3 个 demo Bot |
| 每个最终 Bot 的比赛数 | 234 场 |
| 每个 Bot 与每个对手的比赛数 | 18 场（3 张地图 × 每图 6 场） |
| 全部比赛 | 10,530 场 |
| LLM Bot 之间的对战 | v0.2 不安排 |
| 随机种子 | 使用 fresh seeds（新的随机种子） |

单个 Bot 的分数是它在 234 场锦标赛比赛中的胜率，模型分数则是 5 个 Bot 分数的平均值，0 表示全败，100 表示全胜。由于每个 Bot 都面对相同的固定阵容，比赛不依赖其他 LLM Bot 的参赛情况，所有运行可以并行；新模型也可以直接评分，不必重打其他模型的比赛。

这也是 v0.2 放弃 v0.1 Elo、改用普通胜率的原因。固定阵容让不同模型代际面对相对稳定的门槛，避免 Elo 在某一批参赛模型内部拟合所带来的依赖。不过，固定对手只能减少对手变化造成的干扰，并不能消除地图、运行工具和推理配置变化的影响。

## 第一方 CLI 与最高推理档位，改变了比较条件

v0.1 将所有模型放在同一个中立的第三方 harness（运行脚手架，负责提供工具并驱动模型执行）中，使用 `high` 推理档位；v0.2 改为各实验室的第一方 CLI，并使用各自支持的最高档位。除 Gemini 4 Argon 外，其他模型都运行在 `max`；Argon 使用它支持的最高档位 `high`。两版都给模型 1 小时。

![v0.1 到 v0.2 的 harness 变化示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/StarSkirmish-Bench-v0.2/images/figure-5.svg)

> 图解：原图用彩色连线示意从 v0.1 的统一第三方 harness 到 v0.2 各实验室第一方 CLI 的变化；具体模型与版本如下表，图中不包含分数数据。下表同时保留原图中的实验室图标，便于对应各实验室的 CLI。

| v0.2 CLI | 对应模型 | CLI 版本 |
| --- | --- | --- |
| ![Claude 图标](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/StarSkirmish-Bench-v0.2/images/claude.svg) Claude Code | Claude Opus 5.5、Fable 5.1、Sonnet 5.5、Haiku 5.5 | 2.1.296 |
| ![OpenAI 图标](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/StarSkirmish-Bench-v0.2/images/openai.svg) Codex | GPT-6 Astra、GPT-6.1 Sol、GPT-6 Luna | 0.162.1 |
| ![Gemini 图标](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/StarSkirmish-Bench-v0.2/images/gemini.svg) Antigravity CLI | Gemini 4 Argon | 1.3.5 |
| ![Meta 图标](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/StarSkirmish-Bench-v0.2/images/meta.svg) Muse Code | Muse Wonder Full | 1.4.7 |

因此，v0.2 更接近对“模型 + CLI + 推理配置”这一整套组合的测量，而不是在完全相同的外部条件下只比较模型本身。所有 v0.2 harness 都会在约 250k tokens 附近压缩上下文：对话接近上限时，CLI 先浓缩上下文，再让模型继续。以 OpenAI 的 Codex 为例，GPT 模型使用默认 272k token 窗口，填充到 90% 时触发压缩。

为保证任务边界一致，CLI 的 web search 和 web fetch 均关闭，agent 也没有其他互联网访问。CLI 在计时开始前安装，1 小时从 agent 启动时开始计算；没有 submit 工具，时间到后由 harness 直接收集 Bot。这些细节决定了结果应被理解为受限环境下的端到端表现，而不是模型在无限制开发条件下的能力。

除了运行条件，StarSkirmish 与其他评测的排名关系，也能帮助界定它到底反映了什么能力。

## 它与部分编程榜单并不同步

StarSkirmish 与一些常见综合或编程评测的排名并不完全同步。下表中的 $\rho$ 是 Spearman 秩相关系数，主要反映排名的一致性，不是胜率提升百分比；WeirdML v3 的样本只有 7 个模型，因为 Gemini 4 Argon 和 Muse Wonder Full 在原文写作时尚未被纳入。

| 对照评测 | 与 StarSkirmish 的关系 | 原文给出的条件或现象 |
| --- | --- | --- |
| Artificial Analysis Intelligence Index | $\rho = 0.26$ | GPT-6 Astra 在这 9 个模型中仅并列第 4 |
| Terminal-Bench 4.0 | 已从 v0.1 的紧密跟随转为偏离 | Claude Sonnet 5.5、Muse Wonder Full 在那里接近榜首，在这里却接近榜尾 |
| Vals IOI | $\rho = 0.95$ | 对照任务是用 C++ 解决奥赛编程题 |
| WeirdML v3 | $\rho = 1.00$，$n = 7$ | 独立的非常规机器学习编程任务评测 |

原文提出的一种解读是，被广泛报道的评测可能已经成为模型优化的目标，而较少被针对的长时程、困难编程任务之间仍然保持同步。这只是对相关性的一种解释，不是对模型训练过程的直接证明；尤其 WeirdML 的相关性只有 7 个样本，不能据此宣称普遍规律。

## 结论：适合看持续改代码的能力，不宜当成通用排名

回到开头的问题：模型能否在一小时内写出强大的《星际争霸》Bot？v0.2 给出的答案是，模型之间已经出现清晰的分数与成本差异，但最强的 GPT-6 Astra 对三大顶尖人类 Bot 仍只有 7% 胜率，且同一模型不同运行之间也会波动。

- 关注 agentic coding 的读者，可以重点看模型是否主动挑战强对手、是否把练习反馈转成可编译的改进，而不只是最终分数。
- 想用榜单选择模型的读者，应同时检查 Bench 分数、单次 API 成本、运行方差，以及 CLI 和推理档位；GPT-6.1 Sol 的第二名与较低成本因此值得单独关注。
- 想复现或比较结果的团队，应保留固定对手、地图、5 次运行和 234 场/Bot 等条件，不要把不同 harness 下的数字直接混排。

这项测试的证据范围也很明确：它是 1 小时、无互联网、Protoss 镜像对战、三张地图和固定阵容下的结果，不能替代真实人类 Bot 锦标赛，也不能代表所有软件工程任务。它的价值在于把长时程编程中的练习、反馈和交付过程变成可观察、可复测的指标。

> 本文参考自 [StarSkirmish Bench v0.2](https://starskirmish.com/bench/v0-2/)