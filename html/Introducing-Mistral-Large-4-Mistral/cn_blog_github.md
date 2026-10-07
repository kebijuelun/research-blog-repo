# Mistral Large 4：万亿开源巨兽登场

开源大模型的天花板又被抬高了一格。今天的主角是 Mistral 刚发布公测的 Mistral Large 4（简称 ML4，官方昵称「Le Chonk」）：一个总参数量 1 万亿、激活参数 490 亿的 MoE 原生多模态模型，目标是让开源权重模型在编程、Agent 工作流和企业级垂直场景上正面硬刚闭源旗舰。最硬的数字有两组：在安全漏洞复现与修复测试中拿到 82% 的全场最高分，Cybench 网络安全挑战赛解题率达到 93%。权重将于本月底放出，API 现在就能在 Mistral Studio 上体验。

![hero-ml4-2x.jpg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/hero-ml4-2x.jpg)

> 图解：ML4 官方宣传图「Le Chonk」——Chonk 意指「大块头」，呼应其万亿参数体量。

## 先说结论：它强在哪

ML4 是 Mistral 迄今最大、能力最强的模型：1 万亿总参数、490 亿激活参数，原生支持多模态输入。官方给出的定位非常明确——性能上与全球最强开源模型同台竞技，同时大幅甩开美国和欧洲的其他开源权重模型。

更值得注意的是垂直领域的表现。在网络安全、金融、法律这些企业级关键负载上，Mistral 声称 ML4 在开源模型中做到了 SOTA；而在视觉定位（Visual Grounding）等个别领域，它甚至超过了前沿闭源模型。

![artificial-analysis---deepswe-1.1-201-alt_Z15N6fD.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/artificial-analysis---deepswe-1.1-201-alt_Z15N6fD.webp)

> 图解：第三方机构 Artificial Analysis 的 DeepSWE 基准对比，展示 ML4 在真实软件工程任务上与各家开源/闭源模型的相对位置。

![code-benchmarks---terminal-bench-4-201-v3_TdIn1.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/code-benchmarks---terminal-bench-4-201-v3_TdIn1.webp)

> 图解：Terminal-Bench 4 基准成绩，衡量模型在终端中完成复杂命令行工作流的能力，ML4 处于开源模型第一梯队。

![cybersecurity-benchmarks---aa-cyber-index-alt_Z10qJ7I.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/cybersecurity-benchmarks---aa-cyber-index-alt_Z10qJ7I.webp)

> 图解：Artificial Analysis 网络安全指数总览，ML4 跻身全球前五，在非中国开发的开源权重模型中明显领先。

![artificial-analysis---automationbench-201_1xA4BS.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/artificial-analysis---automationbench-201_1xA4BS.webp)

> 图解：AutomationBench 总览图，覆盖 Gmail、Slack、Salesforce 等真实办公应用中的 657 条业务流程，ML4 得分领先多款开源对手。

![vals.ai---finance-agent-v2-201_13VnVT.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/vals.ai---finance-agent-v2-201_13VnVT.webp)

> 图解：第三方评测机构 vals.ai 的金融 Agent 基准结果，ML4 在金融分析任务上超过了 GPT-6-Astra。

![vals.ai---harvey-s-legal-agent-benchmark-201_1vHt2w.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/vals.ai---harvey-s-legal-agent-benchmark-201_1vHt2w.webp)

> 图解：vals.ai 上 Harvey 法律 Agent 基准的对比结果，ML4 在所有开源模型中排名第一。

![dense200--bbox-201_Z1KiEQH.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/dense200--bbox-201_Z1KiEQH.webp)

> 图解：Dense 200 视觉定位基准成绩，ML4 以 42% 对 41% 微弱优势超过闭源的 GPT-6-Astra。

在权重正式放出之前，Mistral 正与网络安全厂商、经过审核的合作伙伴以及国家机构做真实环境的红队测试，这些合作方将拿到一个降低内容审核、扩展网络能力的同款模型版本。

## 欧洲锻造：为 AI 主权而生

性能之外，ML4 的另一个叙事主轴是「主权」。

ML4 在 Mistral 位于欧洲的自有数据中心从零训练，使用了 3800 张 NVIDIA Grace Blackwell GPU，公测 API 也跑在同一套基础设施上。这意味着从训练到推理的整条链路，都可以不依赖其他数字服务商、在欧洲法律框架下闭环运行。

这件事为什么在网络安全领域尤其重要？原文给出了一个很实际的理由：闭源模型供应商层面的「拒绝回答」可能直接阻断合法的漏洞研究和应急响应——事件处置进行到一半突然失去模型能力，本身就是安全风险。而开源权重 + 自部署意味着组织能力掌握在自己手里。

顺带一个有趣的事实：ML4 的训练数据中相当大比例是多语言语料，覆盖超过 160 种语言，包括欧盟所有官方语言。另外，ML4 使用的训练、定制化和 RL 环境，与 Mistral 通过 Mistral Forge 向企业客户开放的是同一套——也就是说，客户拿到的工具链和 Mistral 内部炼模用的一模一样。

## 能力详解：五大战场逐一拆解

参数和故事讲完，接下来看真本事。Mistral 把 ML4 的能力拆成网络安全、Agent 编程、Agent 工作流、多模态、科学与数学五个方向，我们逐个来看。

### 网络安全：开源模型里的最强矛与盾

ML4 是目前全球最强的网络安全 AI 模型之一。在 Artificial Analysis Cyber Index（一个独立评测，考察模型在真实软件中发现并修复安全缺陷的能力）上，它排名全球前五，并且在非中国开发的开源权重模型中遥遥领先。

两个数字值得记住：

- 在「复现开源软件真实漏洞并打补丁」这项测试中，ML4 得分 **82%** ，为所有参评模型最高；
- 在 Cybench（40 道来自安全竞赛的题目）上解题率 **93%** ，是开源权重模型报告过的最高分之一。

这个 82% 的含金量在于对手的衬托：包括 Claude Opus 5.5 和 GPT-6 Astra 在内的多个顶级闭源模型，在同一测试上得分接近零——不是因为做不到，而是因为安全过滤器拒绝执行这类任务。而防守方的工作往往恰恰需要从「证明漏洞真实存在」开始。

笔者认为这里的逻辑很犀利：攻击者已经在用越狱手段绕过闭源模型的限制，防守方如果还被同样的拒答机制捆住手脚，等于单方面缴械。开源 + 可自部署的强模型，是防守方为数不多能「对等武装」的路径。

内测中，ML4 还展现了超出显式训练范围的能力：恶意软件分析、漏洞优先级排序、检测规则编写等。对需要主权可控、可审计 AI 的安全团队，它支持私有云和本地部署。

![artificial-analysis-cyber-index-v3_2fl1fS.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/artificial-analysis-cyber-index-v3_2fl1fS.webp)

> 图解：Artificial Analysis Cyber Index 详细榜单，ML4 在多样复杂安全挑战的推理效率上对阵全场模型的表现。

![cybersecurity-benchmarks---cybergym-e2e--aa-201_1dv9gg.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/cybersecurity-benchmarks---cybergym-e2e--aa-201_1dv9gg.webp)

> 图解：CyberGym 端到端基准结果，展示 ML4 在完整的「发现—分析—修复」安全任务链条上的表现。

![cybersecurity-benchmarks---cybench-201_z0QUM.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/cybersecurity-benchmarks---cybench-201_z0QUM.webp)

> 图解：Cybench 基准对比，ML4 解决 40 道安全竞赛题中的 93%，处于开源模型顶端。官方演示中还包括一个分布外（OOD）的恶意软件逆向分析任务，模型在没有专门训练过的调查场景下也能完成推理。

### Agent 编程：编码能力全面开花

解决了「安全」这个最硬的场景，下一个问题自然是：日常开发它能打吗？

ML4 在软件工程、代码库理解和复杂终端工作流上交出了一份不错的成绩单：

- DeepSWE v1.1： **61.7%**
- SWE-Atlas-QnA： **59.4%**
- Terminal-Bench 4： **28.3%**

三项合成的 Coding Agent Index 为 **49.8%** ，领先 DeepSeek V4 Pro 0813 和 Qwen3.8 Max。

![code-benchmarks---swe-atlas-qna-v3_VT0VD.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/code-benchmarks---swe-atlas-qna-v3_VT0VD.webp)

> 图解：SWE-Atlas-QnA 基准对比，考察模型对代码仓库的理解与问答能力，ML4 得分 59.4%。

自动评测之外，Mistral 还联合 Surge AI 做了一轮盲测人工评估：专业标注员在隐藏模型身份的情况下，按 1–5 分给代码输出打分。ML4 Preview 在五款模型中排名第二（3.74 分），超过 Kimi K3（3.59）、GLM-5.3（3.60）和 GLM-5.2（3.40），仅次于 Claude Opus 5（4.22）。

![human-evaluation---surge--human-eval-on-code-201_ZCsHgr.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/human-evaluation---surge--human-eval-on-code-201_ZCsHgr.webp)

> 图解：Surge AI 人工盲评结果。横轴为模型，纵轴为 1–5 分的代码质量主观评分；ML4 Preview 位列第二，与第一名 Claude Opus 5 仍有可见差距。

这个排名其实挺诚实——官方没有回避「还打不过 Claude Opus 5」这个事实，反而把完整数据摆出来，可信度加分。

### Agent 工作流：不止写代码，还能干活

模型会写代码只是第一步，真正的商业价值在于跑通完整工作流。ML4 可以驱动通用 Agent 完成信息收集、工具调用，并产出最终交付物。

在 AutomationBench 上——覆盖 Gmail、Google Sheets、Slack、Salesforce 等应用的 657 条真实业务流程——ML4 得分 **59.9%** ，领先 Kimi K3、MiMo-V2.6-Pro 和 DeepSeek V4 Pro。

在更贴近知识工作产出的 AA-Briefcase（评估长周期知识工作，如表格、幻灯片、PDF 的制作）上，ML4 拿到 **1393 Elo** ，同样超过 DeepSeek V4 Pro。

![artificial-analysis---automationbench-201_1DD5qe.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/artificial-analysis---automationbench-201_1DD5qe.webp)

> 图解：AutomationBench 详细成绩。ML4 在 657 条跨应用业务流程上取得 59.9%，体现其在多步工具调用和长链路任务中的稳定性。

### 多模态：从看图到「看清」

ML4 在图像理解上是一次阶梯式跃升，能在复杂文档、图表和自然图像上做推理，面向工程、制造、地球观测等对感知要求高的行业。

真正的亮点是视觉定位（Visual Grounding）与 Agent 能力的结合：检查千兆像素级的卫星图像帮助灾害响应团队争分夺秒，或者放大工程图纸逐个零件核验直到答案精确。官方演示中，ML4 完成了密集自然场景定位、机械图纸零件核验、PDF 证据检索和大范围地理图像中最难目标的搜索。

在 Dense 200 基准上，ML4 以 **42% 对 41%** 超过 GPT-6-Astra——差距不大，但「开源超过闭源旗舰」这件事本身就有标志性意义。

![multimodal-benchmarks---dense200--bbox--alt_Z1AfJj4.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/multimodal-benchmarks---dense200--bbox--alt_Z1AfJj4.webp)

> 图解：Dense 200 视觉定位基准的模型对比，ML4 以 42% 位列第一，险胜 GPT-6-Astra 的 41%。

![multimodal-benchmarks---chartqa-pro-201_ZNRwVf.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/multimodal-benchmarks---chartqa-pro-201_ZNRwVf.webp)

> 图解：ChartQA Pro 图表问答基准，衡量模型对统计图表的理解与推理能力，ML4 处于开源模型前列。

![multimodal-benchmarks---gdp.pdf---aa-201_70MDW.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/multimodal-benchmarks---gdp.pdf---aa-201_70MDW.webp)

> 图解：GDP 文档理解类基准（Artificial Analysis）结果，考察模型从复杂 PDF 中提取并推理信息的能力。

### 科学与数学：研究者的全流程助手

ML4 的科学能力由 AI 驱动方法与 Mistral 研究团队在数学、物理、化学上的专业积累结合而成。

在 SciCode-Verified（考察模型用代码实现物理、数学、材料、生物等领域复杂科学工作流的能力）上，ML4 是开源权重模型中的 SOTA。实际演示中，它能一次性生成完整的 Hartree–Fock 模拟——一个由多步高级计算例程组成的复杂化学任务。

数学方面，官方内部人工评估显示 ML4 比 GLM-5.3 推理更精确、结构更清晰，还能支撑长时间的领域应用数学任务，包括与前沿理论物理相关的工作。

![scicode-verified-pass-1--n_6--alt_Z1DedbC.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/scicode-verified-pass-1--n_6--alt_Z1DedbC.webp)

> 图解：SciCode-Verified 基准成绩（pass@1 与 n 次尝试口径），ML4 在科学计算代码实现任务上领先其他开源权重模型。

![ml4-vs-glm-5.3--E2-80-94-stem-win-rate-breakdown-201_Z24YWt5.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/ml4-vs-glm-5.3--E2-80-94-stem-win-rate-breakdown-201_Z24YWt5.webp)

> 图解：Mistral 内部 STEM（数学与物理）任务人工评估，ML4 对阵 GLM-5.3 的胜率拆解，ML4 在多数子项上占优。

## 知识工作：法律金融双双超闭源

能力之外，落地到办公室场景才是大多数企业的真实需求。ML4 是 Mistral 最强的日常专业任务模型，可以创建、编辑和修复复杂表格与文档。

Mistral 特意引入第三方评测机构 vals.ai 做背书：在代表性法律与金融任务上，ML4 两个领域都超过了 GPT-6-Astra；在 HarveyAI 的法律 Agent 基准上，它超过所有开源模型。

![finch--finworkbench-201_1fKP7q.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/finch--finworkbench-201_1fKP7q.webp)

> 图解：FinWorkBench 基准结果，考察模型在真实财务与会计场景中创建/编辑电子表格的能力。

官方还放了一个很有意思的金融分析演示：让 ML4 和其他顶级开源模型同时解决同一个多步公司金融挑战——检索 EDGAR 及欧洲同类数据库中的公开财报。演示用一张动画语义地图追踪每个模型的解题路径，标出沿途检索到的每份文档；每条轨道的位置反映已收集的证据、计算结果和未解决的问题，观众可以直观对比不同模型「思考路线」的差异。

## 人工评估：与 GLM-5.3 正面对比

除了分项基准，Mistral 还组织了一轮内部人工评估，让编程、CAD（计算机辅助设计）、金融、数学和物理领域的专家标注员对比 ML4 与 GLM-5.3。结果是：ML4 在 CAD 和 STEM 上更受偏爱，金融和编程上与之持平或接近。

![win-rate_Z1VxWvu.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/win-rate_Z1VxWvu.webp)

> 图解：ML4 对阵 GLM-5.3 的分领域胜率。CAD 与 STEM 上 ML4 明显领先，金融与编码两者基本打平。

## 安全性：能打的矛，也要配坚固的盾

一个网络安全能力拉满的模型，自身的安全性自然会被加倍审视。ML4 在这方面的答卷如下：

- **间接提示注入防护** ：相关基准已被 ML4 打满（饱和），在开源模型中处于前沿（对比对象为 GLM-5.2、GLM-5.3、Kimi-K2.6、Kimi-K3、DS-V4-Pro-0813）；
- **Lakera B3 AI Security Benchmark** ：抵御 **93.3%** 的攻击，官方称未见更高的竞品分数；
- **KORA 基准** （衡量负责任的用户交互）：得分 **1.691** （满分 2，评级「Exemplary」），是开源模型中的最高纪录；
- **恶意网络请求拒答** ：尽管网络能力强大，ML4 在 JailbreakBench、StrongREJECT、AgentHarm 的网络类恶意提示上的平均拒答率高于所有开源模型。

![b3-agent-security-benchmark-201_1yc5n3.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/b3-agent-security-benchmark-201_1yc5n3.webp)

> 图解：Lakera B3 AI 安全基准对比，纵轴为攻击抵御率，ML4 的 93.3% 高于参评竞品。

![refusal-of-harmful-cyber-requests-201_ZSiBSy.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/refusal-of-harmful-cyber-requests-201_ZSiBSy.webp)

> 图解：各模型对有害网络请求的拒答率对比。ML4 在三个越狱/有害提示基准上的平均拒答率位居开源模型之首——能力强和守规矩在它身上并不矛盾。

![kora-benchmark-201_Z1ST57T.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/kora-benchmark-201_Z1ST57T.webp)

> 图解：KORA 基准得分对比，ML4 拿到 1.691（满分 2 对应「Exemplary」评级），为开源模型最高。

## 幕后功臣：大规模强化学习

能力跃升从哪来？原文给出的答案是 RL（Reinforcement Learning，强化学习）。

动机很直接：基座模型进步太快，昨天的后训练配方配不上今天的模型——曾经能把模型逼到极限的 ground truth 样本，现在已经不够看了。RL 的优势在于它会跟着模型一起进化：在模型自己的尝试结果上训练，模型变强了就同步提高任务难度和广度。

具体做法上，Mistral 的 RL 库有三个关键设计：

- **可组合的环境接口** ：单次训练就能混合单轮对话、复杂科学解题、安全对齐、事实性、长周期工具调用等多种任务，各环境共享代码沙箱、网页搜索、外部 API 等脚手架资源；
- **可组合的验证器** ：奖励模型、单元测试、LLM 裁判、静态检查按需组合，为每类任务定制奖励信号；
- **异步大规模 rollout** ：自动扩缩容的 actor 集群并行生成数万条轨迹，训练异步进行；管线针对长轨迹优化，支持百万 token 级的 rollout 预算（跨多次上下文压缩），并通过新方法压低 staleness 和 off-policy 漂移，保证长周期 RL 的稳定性。

规模数字：在当前 3000 张 GPU 的集群上，单次训练运行每天产出约 **330 亿 token** ，其中约 **160 亿** 是过滤和掩码后可训练的 completion token。

![training-reward_1n1Dxj.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/training-reward_1n1Dxj.webp)

> 图解：多个代表性训练环境中的 reward 曲线。随着策略学会解决日益复杂的任务，各环境的训练奖励持续上升，说明 RL 运行健康、尚未饱和。

这些提升并非只在训练环境里「自嗨」——它们能迁移到下游评测，而且最终模型的能力由监督微调（SFT）和 RL 两个后训练阶段共同贡献。

![sft-rl-three-tasks-alt_2tyhkR.webp](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/sft-rl-three-tasks-alt_2tyhkR.webp)

> 图解：三个代表性任务上 SFT 与 RL 各自贡献的拆解曲线。可以看到 RL 在 SFT 之上继续推高表现，验证了「两阶段后训练」配方的有效性。

笔者认为这一节是全文技术含量最高的部分：把环境、验证器、rollout 基建都做成可组合的「积木」，本质上是把 RL 从「一任务一工程」的手工作坊模式推向了工业化流水线——这也是它能同时喂饱编程、安全、数学等多条能力线的原因。

## 价格与规格

模型卡片信息一览：

![big-cat-icon.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Introducing-Mistral-Large-4-Mistral/images/big-cat-icon.svg)

| 项目 | 规格 |
| --- | --- |
| 类型 | 开源权重、指令与推理混合的 MoE，多模态输入 |
| 总参数 / 激活参数 | 1 万亿 / 490 亿 |
| 语言 | 原生流畅支持 160+ 种语言 |
| 输入价格 | $1.36 / 百万 token |
| 输出价格 | $4.18 / 百万 token |
| 获取方式 | Mistral Studio 公测 API；权重月底发布 |

## 总结与展望

- ML4 是 Mistral 首个万亿参数模型（激活 490 亿），原生多模态 MoE，本月底开源权重；
- 网络安全是最大亮点：漏洞复现修复 82% 全场最高，Cybench 解题率 93%，同时保持开源模型中最高的恶意请求拒答率；
- Agent 编程合成指数 49.8% 领先 DeepSeek V4 Pro 与 Qwen3.8 Max，人工盲评仅次于 Claude Opus 5；
- 视觉定位（Dense 200）和法律金融第三方评测上超过闭源 GPT-6-Astra，开源身份打出差异化；
- 背后是工业化 RL 流水线：3000 卡日产 330 亿 token，且 RL 运行仍在继续、未见饱和。

展望来看，ML4 只是 Mistral 30 亿欧元 D 轮融资路线图的第一站——权重、架构细节和后训练方法学都将在本月底陆续放出，且它还将作为基座衍生新一代行业专用模型。唯一的悬念是：公测版的表现能否在权重放出后经得起社区的独立复测。

> 本文参考自 [Introducing Mistral Large 4 | Mistral](https://mistral.ai/news/mistral-large-4/)