# Gemini 4 Hydrogen：四级思考与多模型分工，加速智能体执行

Google DeepMind 在 2026 年 11 月 12 日发布 Gemini 4 Hydrogen。它没有把目标设为取代所有深度推理模型，而是补上智能体工作流中更常见的一环：快速写代码、跑测试、读仓库、调用工具并验证结果。为此，Hydrogen 提供 low、medium、high、xhigh 四级思考模式，还能在 Google Antigravity 中与 Gemini 4 Carbon 动态分工。

这套定位直接影响使用成本。Hydrogen 在整个 100 万 token 输入上下文窗口内采用统一价格：每百万输入 token 1.50 美元，每百万输出 token 7.50 美元。Google 报告，Antigravity 新增的 Orchestrator 在 Terminal-Bench 4.0 上取得接近 Carbon 的任务成功率，同时平均减少 24% 配额消耗、缩短 19% 完成时间。不过，这些结果来自官方测试，实际收益仍取决于任务类型和模型分工方式。

## 为什么不让一个模型包办整个任务

真实软件开发并不总需要同等深度的推理。设计系统架构、分析隐蔽缺陷时，团队可能更依赖 Carbon 和 Argon 这类前沿模型的长程推理；进入实现阶段后，工作则变成大量相对明确的执行步骤，包括修改代码、运行测试、检查变更。让一个高成本模型从始至终处理所有步骤，会把算力花在并不需要深度思考的环节上。

Hydrogen 因而有两种用法：作为日常工作的独立模型，或作为多模型工作流中的执行引擎。Google Antigravity 的 Orchestrator 会在同一会话中根据任务动态切换 Carbon、Hydrogen 等模型，而不是要求开发者手工选择每一次调用。对于可并行的工作，Carbon 可以担任协调者，Hydrogen 则驱动 worker 和 verifier 智能体，在隔离工作区中同时构建和测试修改。

这种分工试图把“判断交给谁”和“执行交给谁”拆开。官方给出的 24% 配额节省和 19% 提速，正来自这种按需分配，而不是单纯宣称 Hydrogen 在所有任务上都优于 Carbon；Orchestrator 的结果描述也是“接近 Carbon 的成功率”，并未声称完全相同。

## 从 Gemini 3.8 Flash 迁移，提升体现在哪里

![Gemini 4 Hydrogen 与其他模型的基准测试对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/gemini-4-hydrogen_benchmarks_table.gif)

> 图解：官方汇总图对比 Hydrogen 与同类模型的多项基准成绩。不同测试衡量编码、自动化和推理等不同能力，不能把一个榜单名次理解为所有场景的全面领先。

对于从 Gemini 3.8 Flash 升级的团队，Google 给出了更具体的对照：在默认 high 思考级别下，Hydrogen 完成任务的速度达到其 2.5 倍；Terminal-Bench 4.1 得分从 18.2% 提升到 57%，超过原来的三倍。官方还称，其虚构陈述减少超过 5 倍，并在工具调用失败或搜索没有返回结果时能更顺畅地调整后续行动。

编码能力的改进不只是生成一段答案。Hydrogen 面向问题修复、调试、快速迭代和代码库问答，强调范围集中的修改，尽量避免重写无关文件；在要求发现并修复缺陷、同时不破坏现有代码的 SWE-sweep 上，官方称其超过同层级模型。探索大型仓库时，它还能原生地批量并行发起文件读取和搜索调用，减少逐文件检查的开销。

合作伙伴在复杂图、SQL 和数据密集型企业工作流的早期测试中观察到，相比 Gemini 3.8 Flash，Hydrogen 的运行速度提升为 2 至 2.8 倍，token 用量减少 12% 至 23%。这些数字有明确的比较对象，但仍属于早期合作测试，不能直接外推到每种代码库和业务数据。

开放式智能体任务则覆盖深度研究、数据分析和广泛的代码库导航。Hydrogen 在 arena.ai 的实时基准 Agent Arena 中排名第 5；在 HLE-Diamond 上以 58.8% 领先其所在模型层级，在 Vals Tax Agent Bench 上以 52.3% 位列总榜第 1，在 Vals Legal Research Bench 上以 55.8% 并列第 1，并在 AutomationBench-AA 上领先同层级模型。这里的“第 1”和“领先”分别对应不同榜单及分组，不能混成同一个综合排名。

多模态知识工作是另一条主线。Hydrogen 原生支持 100 万 token 输入，并可输出最多 100 万 token，用于把密集 PDF、电子表格、图表、音频和录制视频整理成带引用的报告或结构化交付物。在衡量复杂视觉文件端到端知识工作的新基准 Multimodal Knowledge Agent（MKA）上，它取得 53.4%；在面向端到端专业工作流的 Agents’ Last Exam v1.1 上，成绩为 71.6%，官方称两项均超过其他模型。

## 从网页到表格，输出不只是代码片段

网页开发是 Hydrogen 希望体现速度优势的场景，包括实时 UI 原型、设计到代码的迭代和交互应用搭建。Google 的展示案例是 Geminisland：用户输入一条提示，就能向共享体素岛屿添加自定义结构，其他玩家随后可用第一人称在地图中走动和探索。

配合 Google Antigravity 新增的 Gemini 前端设计技能，Hydrogen 还一次性生成了一个电商店面，其中包含 WebGL2 地形着色器、固定四阶段滚动叙事、冷热效果分屏滑块、3D 线框规格卡和实时外观配置器。另一个案例把 Google Sheets 变成动画像素画布：模型缩小数千个单元格，并组合 Python、原生 `=LET(...)` 公式与条件格式，在实时电子表格中绘制和播放全动态场景。

这些案例更接近完整交付流程，而非静态页面生成。模型还能从一段 YouTube 主题演讲视频和设计模板出发，在 Google Slides 中生成多页演示文稿，再自主检查并修复文字重叠、清理页面布局。Google Sheets 中的 Hydrogen 则可把财务数据转成会随预测情景动态调整的交互模型，官方称其输出速度比此前 Sheets canvas 中的模型快 30% 以上。

展示案例证明了工作流可以串起来，却没有提供统一的成功率和人工修改量。因此，它们适合用来说明能力边界正在被推向哪里，不宜当作所有提示都能一次得到同等成品的保证。

## 四级思考，让等待和费用随任务变化

![Artificial Analysis Intelligence Index 分数与时间关系图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/gemini-4-hydrogen_aa-index_score-time_graph.gif)

> 图解：该图把智能指数成绩与任务耗时放在一起，重点不是单一最高分，而是不同思考级别下质量与速度如何共同变化。

Hydrogen 的四个思考级别为 low、medium、high 和新增的 xhigh，默认使用 high。用户可以按任务所需推理深度，在质量、延迟和成本之间选择；Google 表示，各档表现会平滑、可预期地递进，且与部分同层级竞争模型相比，Hydrogen 经常能以更少输出 token 更快完成任务。

在 Artificial Analysis Intelligence Index 上，Hydrogen 默认 high 档得分 51，略高于 Argon 的 50；切换到最高 xhigh 档后达到 54。这个对比说明增加推理预算仍能换取分数，但默认档与 Argon 只有 1 分差距，不能仅凭这一点推断所有任务都存在明显能力差距。

![AutomationBench-AA 基准图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/gemini-4-hydrogen_automation-bench_graph.gif)

> 图解：AutomationBench-AA 用于观察模型在自动化任务中的表现，应与具体任务条件和模型分组一起解读。

![AA Index 5.01 基准图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/gemini-4-hydrogen_aa-index_graph.gif)

> 图解：AA Index 图对应 Artificial Analysis 的综合智能评价；官方文字给出的 Hydrogen 成绩为 high 档 51、xhigh 档 54。

![Terminal-Bench 4.0 基准图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/gemini-4-hydrogen_t-bench_graph.gif)

> 图解：Terminal-Bench 4.0 侧重终端环境中的任务执行。它与正文中比较 Gemini 3.8 Flash 时使用的 Terminal-Bench 4.1 不是同一版本，分数不应直接交叉比较。

![HLE-Diamond 基准图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/gemini-4-hydrogen_hle_graph.gif)

> 图解：HLE-Diamond 用于考察深度领域推理与分析，官方报告 Hydrogen 得分为 58.8%，比较范围是其所在模型层级。

对使用者来说，四档设计的价值在于不必始终开启最高推理级别。简单、明确的执行可以采用较低档，复杂分析再提高到 xhigh；但官方材料没有给出各档完整价格与延迟表，具体选择仍需在自己的任务上验证质量和耗时。

## Gemini 应用与搜索中的变化

普通用户接触到的 Hydrogen，重点不在榜单，而在多轮对话中能否持续遵循要求。Gemini 应用中的写作、研究和日常任务会更贴近用户指定的语气、格式和规则，目标是让邮件草稿、学习指南、改写和文档尽量少经过返工；比较产品或综合密集资料时，则会生成清晰的表格和可视化拆解，同时保留关键细节。

执行任务时，Hydrogen 会逐步拆解目标，并结合用户上传内容、Gmail 和 Google Drive 等 Google 应用，以及 Airtable、Dropbox 等 Connected Apps 中的相关上下文。官方称，它会更频繁地调用正确工具和技能，减少完成任务所需的来回沟通。

原型生成也被带入消费端。一个案例把实时全球雷击数据转成交互式 3D 地球和 Web Audio 音乐引擎，让每次闪电对应视觉涟漪与环境音乐；另一个案例用纯数学和 SDF 纹理构建 WebGL2 水与玻璃可视化，模拟实时光线折射、反射、色散和吸收。借助视频组装技能，Hydrogen 还能依照创意简报搜集公共领域影像，把素材剪成一段关于元素“氢”的连续档案风短片并配上旁白。

在搜索中，Hydrogen 面向三类变化：理解包含多个部分、难辨认错字或高度限定条件的自然提问；生成更适应问题、便于扫读的视觉化答案；升级 query fan-out 技术，把查询扩展并连接到更广泛的相关链接。最后一项意在帮助用户发现过去可能没遇到的网站，而不是只缩短一条固定答案。

![公寓维修搜索中 Gemini 3.5 Flash 与 Gemini 4 Hydrogen 的并排对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/Gemini4Hydrogen-01-Studio_Apt-fu.width-2000.format-webp.webp)

> 图解：手机界面对比同一公寓维修查询下两代模型的回答。图中展示的是搜索呈现方式案例，并非独立的准确率实验。

![波士顿社区搜索中 Gemini 3.5 Flash 与 Gemini 4 Hydrogen 的并排对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/Gemini4Hydrogen-03-Boston-full.width-2000.format-webp.webp)

> 图解：波士顿社区查询用于展示 Hydrogen 如何组织更具体、易于浏览的搜索结果。

![运动鞋搜索中 Gemini 3.5 Flash 与 Gemini 4 Hydrogen 的并排对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Gemini-4-Hydrogen/images/Gemini4Hydrogen-02-Sneakers-full.width-2000.format-webp.webp)

> 图解：运动鞋查询案例体现视觉化答案与相关链接的组织方式；实际结果仍会随查询和用户场景变化。

## 更快的智能体，也需要更明确的边界

Google 将 Gemini 4 的安全栈引入 Hydrogen，把训练阶段的保护与产品界面的实时防护结合起来。针对经常调用工具和修改代码的智能体，官方尤其关注两类失败：越过任务或权限边界，以及本应完成任务却没有提供足够服务。

相比 Gemini 3.8 Flash，官方报告 Hydrogen 在智能体任务中的过度越界最多减少 81%，服务不足最多减少 79%；在工具使用和编码评测中，不对齐行动与走捷径行为减少至原来的十二分之一。面对无法完成或受阻的任务，它绕过权限边界的可能性最多降低一半，并在内部智能体诚实性评测中取得 96.7% 的诚实率。

这些数字中的“最多”很重要，它表示特定评测下的最大改进幅度，不代表每次运行都会获得同等收益。内部诚实性评测也不同于覆盖所有现实场景的外部审计，仍需结合模型卡和后续独立测试判断。

推理透明性方面，Google 称在超过 20 万个受审计对话轮次中，Hydrogen 保留了 99.9% 的思维链可读性，且没有出现欺骗性推理编辑。针对间接提示注入，它在 Gray Swan IPI 基准上的抵御能力提升超过 2.3 倍，官方同时报告工具使用、计算机使用和编码智能体的稳健性均有改善。

日常问答的防护则强调减少误拒。更新后的机制试图区分真实生物安全风险与合法医学、科学问题，并将 HealthBench Pro 上的错误拒绝率控制在 0.4% 以下。这一结果针对特定健康评测，不意味着其他敏感领域也具有相同误拒率。

## 已开放使用，但选择仍应基于工作流

Hydrogen 已面向全球逐步开放。开发者可通过 AI Studio 和 Android Studio 中的 Gemini API 构建应用，也可在 AI Studio 试用；付费 Google AI Pro 和 Ultra 订阅者还能在 Google Antigravity 中使用。企业可通过 Gemini for business 的 API 或模型下拉菜单、Google Antigravity，以及符合条件的企业订阅中的 Google Sheets 部署。普通用户可在 Gemini 应用的模型菜单和 Google Search 中使用，Google AI Pro 与 Ultra 订阅者还可率先在 Sheets 中体验。

回到最初的问题，Hydrogen 最值得关注的不是某一个最高分，而是把快速执行、长上下文、多模态输出和分级推理组合成了一套日常智能体方案。独立使用时，它面向希望从 Gemini 3.8 Flash 获得更快编码与工具执行的团队；与 Carbon、Argon 配合时，它负责把深度判断转成并行实现和验证。现有证据主要来自 Google 及合作伙伴测试，落地前应使用自己的仓库、工具链和任务分布，分别检查成功率、延迟、token 成本以及权限边界，再决定四档思考和多模型分工是否值得启用。

> 本文参考自 [Introducing Gemini 4 Hydrogen](https://deepmind.google/blog/introducing-gemini-4-hydrogen/)