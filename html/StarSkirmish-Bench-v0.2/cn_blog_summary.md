# StarSkirmish Bench v0.2：一小时写 Bot，仍难敌顶尖人类

从只会采矿的骨架开始，在一小时内写出能持续作战、还会根据失败调整策略的 C++ Protoss Bot，比生成一个能编译的函数难得多。StarSkirmish Bench v0.2 用《星际争霸：母巢之战》检验这种长时程 agentic coding：模型要编译、练习、读比赛记录并迭代，最后按固定对手阵容的胜率评分。GPT-6 Astra 以 57 分居首，GPT-6.1 Sol 得 46 分；但 Astra 对 Stardust、PurpleWave、BananaBrain 三个顶尖人类编写 Bot 的平均胜率仍只有 7%。

## 一小时的交付闭环

模型可查看 3 个 demo Bot，借助 FAP（战斗模拟器）和 BWEM（地图分析器），编译、练习并阅读摘要、transcript 与日志；时间到，Bot 文件夹即为交付物，编译成功本身就是要求。正式比赛是 Protoss 镜像，基于 BWAPI 4.4.0/OpenBW，在 Heartbreak Ridge、Benzene、Destination 三张地图进行，60 分钟（86,400 帧）后按 BASIL 的 kills + razings 得分决胜。

## 分数之外，还要看成本与波动

Bench 分数是 5 次运行的平均胜率（0–100），不是 Elo。每个最终 Bot 对同一套 13 个对手（10 个竞技人类 Bot、3 个 demo Bot）在三张地图各打 6 场，共 234 场；9 个模型合计 10,530 场。固定阵容让结果可复测，却不消除地图、工具和配置变化的影响。Gemini 4 Argon、Claude Opus 5.5、Claude Fable 5.1 得 39、39、37 分，原文视为实质并列。

成本图用单次 1 小时 API 成本的对数坐标比较，Sol 是唯一落入原文所称“最有吸引力象限”的模型，成本仅略高于 Astra 的十分之一；这只是本 benchmark 的性价比，Muse Wonder Full 的价格还按 Muse Wonder API 定价估算。

## 练习强对手是线索，不是因果证明

原文对 agent trace 的分析显示，针对 7 个最强阵容 Bot 的练习量是最强的分数预测因素。Astra、Sol、Argon 很早就挑战强对手，Muse、Claude Haiku 5.5、GPT-6 Luna 很少或从不这样做；但 Fable 几乎只练弱对手仍得 37 分，不能据此宣称因果。Argon 偏好 one-shot，41% 的练习针对高层级对手，对其胜率仅 7%，Astra 为 28%。它的最好与最差运行相差 18 个百分点，Astra、Opus 分别为 9、8 个，说明均值应与波动一起读。

v0.2 还从 v0.1 的统一第三方 harness、`high` 档位，改为各实验室第一方 CLI 和各自最高档位（Argon 为 `high`，其余为 `max`），并关闭 web search、web fetch 及其他互联网访问。因此它测量的是“模型 + CLI + 推理配置”的端到端组合，而非纯模型能力；与部分编程榜单的排名也并不同步，相关关系不能解释为训练过程证据。

回到开头：一小时已足以产出可作战、可迭代的 Bot，却远不足以击败顶尖人类。这份榜单适合观察练习、反馈和交付，阅读时还应同时看成本、5 次均值与运行方差，不能替代真实人类 Bot 锦标赛或通用编程排名。

> 本文参考自 [StarSkirmish Bench v0.2](https://starskirmish.com/bench/v0-2/)