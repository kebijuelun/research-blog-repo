# Gemini 4 Argon：专攻复杂长任务

在约 4000 万行 Chromium 代码中，把一个约 600 万行的 legacy 功能目录与核心网页处理逻辑拆开，难点不只是会写代码，而是要在数百个文件之间保持架构一致，并连续完成修改、验证和交付。Google 表示，Chrome 团队借助 Gemini 4 Argon 把这项原本需要数月的工作压缩到数周；另一个案例中，Argon 在 A100/vLLM 环境下将 Gemma 4 E2B 的端到端推理吞吐量提升至原来的 2.85 倍，同时保持 100% Token parity（输出 Token 完全一致）和零质量下降。2026 年 10 月 13 日，Google 开始在全球推出 Gemini 4 Argon，把它定位为面向编码、金融、法律、网络安全和科学等复杂现实任务的 frontier 模型。

这次发布的主线因此不是“聊天更聪明”，而是模型能否在更长的任务链上产出可验证的结果。Google 称 Argon 在覆盖金融、编码、法律和税务等关键经济部门的 Vals Index 排名第一，并把优势归因于多步骤任务中的深度与精确性；比这个名次更值得细看的，是它在真实代码库、实验室和机器人上的具体交付记录。

## 工程能力先看交付结果，而非单次补全

Google 已经在内部把 Argon 用于高难度工程问题。按该公司的统计，过去几周，Antigravity 中的 Argon 帮助团队将 Agent 代码提交量增加 35% 以上，同时将代码回滚减少近三分之一。两个指标放在一起才有意义：只提高提交量可能只是产出更多改动，回滚下降则至少说明这批改动在 Google 的现有流程中更稳定。

Chrome 的重构展示了任务尺度。Argon 协助团队把核心网页处理逻辑从 Chromium 中约 600 万行的浏览器功能 legacy 目录里分离出来，最终完成横跨数百个文件的架构重构。Google 表示，这项改造未来可让已加载的网页在标签页、侧边栏和浮动窗口之间平滑移动而无需重新加载，并消除一整类浏览器崩溃；这里说的是重构将解锁的能力，而不是所有用户已经获得了这些收益。

Google 开源 Agent Development Kit（ADK）的优化，则更清楚地展示了 Agent 的工作方式。团队只给 Argon 提供 `.agents/skills` 规则手册和一段目标描述：在不破坏现有应用的前提下消除框架开销。Argon 在 Antigravity 中自主工作了一夜，找到的优化现已进入 `google/adk-python`：长会话场景下，内存会话准备时间缩短 40%，冷启动导入速度提升 21%。这个案例的关键不只是“通宵自主工作”，而是目标、约束和可检查的性能结果都被放进了任务闭环。

同样的优化能力还被用于两个差异很大的代码库。量子化学软件 xTB 通过半经验扩展紧束缚方法预测分子三维结构，并用于筛选大量药物和材料候选；Google 研究人员在 Antigravity 中使用 Argon 后，分子结构预测速度最高达到原来的 2.3 倍，构象搜索达到原来的 1.8 倍，改动已贡献到 xTB 官方 GitHub 仓库。对于 Gemma 4 E2B，Argon 则从 attention tuning 一路优化到 speculative decoding，在 A100 上通过 vLLM 把吞吐量从约 37,900 Token/s 提高到超过 108,000 Token/s。Google 称这一过程只用了几天，而通常需要数月；不过，这些数字对应的是指定模型、硬件和推理栈，不能直接外推到其他部署环境。

### 从代码库扩展到三维内容生成

Google 还用一组三维演示说明 Argon 在空间推理、数学精度和视觉编码上的改进。借助 Antigravity 的 `/goal`、自定义技能和多 Agent 循环，模型不只生成单段脚本，还会调用 Blender、Three.js 等工具，并根据视觉反馈继续修改场景。

- **寺庙建筑群** ：Argon 通过多 Agent 视觉评审循环，在 Blender 中自主完成多建筑寺庙群的建模，包括具有照片级真实感的建筑、植被和光照。
- **桌面赛车游戏** ：从真实书桌的照片出发，它把桌面物体、房间和赛道做成可玩的微缩赛车游戏。Blender MCP 用于建模和碰撞检查，Nano Banana 负责贴图，Three.js 接入物理效果、自定义着色器，以及由 Gemini Omni Flash 生成的汽车引擎音频。
- **程序化三维游戏** ：Argon 编写 Python 脚本，完成程序化雕刻、骨骼绑定、动画和 PBR 烘焙，再用 Three.js 与 GLSL 着色器接入网页，并加入破碎物理效果和单人对战 AI。
- **纯 Python 光线追踪** ：在不使用外部库的条件下，Argon 自行编写三维光线追踪引擎、视频工具和音频混合器，构建并记录一座阳光下的图书馆，最后用 Gemini 3.8 TTS 配音、Lyria 3.5 配乐并合成视频。
- **从单张图到哥特场景** ：从一张 Nano Banana 图片开始，Argon 在 Antigravity 中自主驱动 Blender MCP，依次对齐相机网格、雕刻哥特式建筑并布置随音频变化的光照，形成完整三维场景。

这些演示共同指向一种变化：编码模型正在从“写出一个文件”转向“操作一整套创作工具”。但演示能证明任务可以跑通，不能单独证明任意输入下的成功率和返工成本。

## 多模态开始进入完整专业工作流

复杂工程之外，Argon 把 frontier 推理、视觉感知、领域知识和写作能力组合到法律、金融、科学及编码流程中。Google 特别强调它会清楚汇报进度、更好地遵循风格指令，回答更直接，减少不必要的犹豫、重复和套话。对于需要长时间运行的 Agent，这些表达特征并非单纯文风问题：它们会影响人能否快速检查模型正在做什么。

Google 将 Argon 的企业能力概括为深度多模态文档理解、纵向数据分析和交互式分析沙盒，并强调回答要严格 grounding，也就是以所给材料为依据。早期访问计划中的演示覆盖了从资料整理到交付物的不同阶段：

- **考古资料重建** ：Argon 综合约 50 GB 的机载 LiDAR 扫描、GIS 房间矢量、美国国会图书馆 CAD 图纸和美国国家公园管理局档案，重建 Chaco Canyon 的三维模型，把零散考古记录转成可探索的数字遗产场景。
- **客户提案自动化** ：在 Gemini App 中，它把一个结构混乱的项目资料文件夹整理成客户提案演示稿，再起草邮件并把演示稿发送给客户，同时通过计划任务为后续客户请求建立自动化工作流，整个过程使用自定义技能约束输出。
- **医学学习工具** ：Argon 汇总三个开源仓库的数据，做出可逐层查看解剖结构、检查临床元数据并触发生物力学效果的三维人体探索工具。
- **可打印产品设计** ：在 Gemini macOS 应用中的 Antigravity 里，它根据沙发照片和标有尺寸的草图，用 Nano Banana 生成新沙发腿的设计迭代，再输出尺寸匹配的三维打印文件。

Isomorphic Labs 给出的案例更接近高价值知识工作。团队把 Argon 用作药物发现与设计研究的协作工具，让它综合复杂的 PDF 和专利并提取相关信息；据团队报告，过去通常需要数周的过程可缩短到数小时。这里最容易被忽视的要求仍是 grounding：生成一份看似完整的调研很快，真正有价值的是抽取结果能回到原专利和论文核查。

## 网络防御从发现漏洞走向完成修复

网络安全是 Argon 高能力与高风险的交汇点。Google 正在把 Fairwind Program 扩展到另外数百家合作伙伴，Wiz 已将其用于公益性质的“Scan for Good”计划，为关键基础设施提供免费保护。以下案例均来自 Wiz 的早期访问实践，而不是公开统一测试。

在与一家美国联邦机构合作的网络韧性项目中，Argon 帮助主动发现并修复了一个可能暴露敏感数据与政府基础设施的缺口。Wiz 表示，这项工作避免了关键安全领域开发细节的暴露，以及对专用模型的未经授权访问。

另一个医疗基础设施案例中，Argon 通过串联复杂漏洞，帮助团队发现并快速修复公共文件下载绕过问题。该问题会暴露主密钥并授予管理员访问权限；Wiz 称，修复行动保护了数十万条私人患者记录。两个案例的意义都在于模型不只指出孤立漏洞，还能沿攻击链理解后果并参与修复，但原文没有披露代码规模、误报率或人工复核成本。

## 科研突破更需要验证，而非只比生成速度

科研场景里的“快”只有与正确性结合才有价值。Argon 的案例分别采用了硬件部署指标、形式化验证、私有领域评测和实验室测试，验证方式不同，证据强度也不能混为一谈。

### 量子计算：吞吐优化与形式化验证并行

在量子纠错方向，Argon 把内部版 AlphaQubit——Google 的机器学习量子纠错解码器——转化为单个高度优化的硬件内核（kernel），使推理吞吐量接近翻倍。Google 表示，这有助于部署更大、更强的解码模型并扩展到更大的量子设备，同时继续满足严格的实时阈值。

在量子算法方向，Argon 编写的一个 `arcsin` 子程序实现经过形式化证明工具 Lean 验证，时空成本降至已发表 quantum CORDIC 基线的 1/100 以下。`arcsin` 是近期量子模拟算法的构建模块之一；相比只报告运行速度，形式化验证和已发表基线让这个结果更容易检查，也为构建优化量子线路标准库迈出一步。

### 核聚变：为仿真提供可检验的精确解

Stellarator（仿星器）依靠复杂的三维磁场约束等离子体，磁场优化一直是核聚变设计中的难题。自 1967 年以来，Grad's conjecture 长期假设无法找到这些磁场的精确数学解，直到近期出现反例，其中包括马里兰大学 Matt Landreman 教授的工作。

Google 团队与 Landreman 随后发现了更大一类反例。这些新解可作为可靠基准，用来测试仿星器仿真代码，为未来设计提供校验工具。需要准确归因的是，原文对这一发现的表述主体是 Google 团队与 Landreman，并没有拆解 Argon 在其中承担了多少具体工作。

### 罕见病：发现一个漏检案例，不等于解决诊断

Gamow Labs 的研究人员使用 Argon 分析患有罕见发育性肺部疾病的新生儿去标识化基因组。在一个案例中，Argon 找出了常规软件多年未能发现的染色体重排，并解释了它为何一直漏检。

在 Gamow Labs 基于真实临床和实验数据构建的私有评测中，Argon 在 RareBench 的罕见病诊断辅助上得到 57%，在 LabBench 的湿实验推理上得到 52.2%，均为该组评测的最高结果。它们显示了领域潜力，也同时标出了边界：这是私有评测而非公开临床验证，分数也不是临床诊断准确率，单个成功病例更不能代表普遍检出率。

### 酶设计：数小时得到五个候选，再由实验室确认

Serine esterases（丝氨酸酯酶）能切断目标分子中的酯键，广泛用于回收、制药、生物燃料和洗涤剂生产。传统酶工程往往需要数月进行专门模型调优和候选筛选，Google 科学家则与 Argon 合作，以 4MU-acetate 为基准分子设计这类酶；该分子的酯键被切断后会在紫外线下发光，因此便于检测催化活性。

Argon 在数小时内汇总相关数据、执行模型调优，并提出多种受生物化学启发的过滤条件，最终得到五个全新设计。后续实验室测试确认这些设计具有稳健的催化活性。这个结果把“计算生成”推进到了“湿实验有活性”，但原文没有给出活性提升倍数、完整筛选成功率或与专家流程的对照数据。

## 机器人把普通视频转成可执行动作

机器人为长任务增加了物理世界约束。Argon 在 Embodied Reasoning（ERQA）基准上取得 87%，该基准测试模型理解和解决三维环境中物理、空间问题的能力。Google 称，仅依靠推理的 Argon 超过了其他 frontier 模型，但原文没有列出各对手的具体得分。

更具体的一项内部测试是“看会做”：Argon 观看人类玩 cornhole 沙包投掷游戏的原始视频，映射环境的空间坐标，计算所需的全身关节动作，再自主构建仿真代码。迁移到 Apptronik 的 Apollo 人形机器人后，机器人在真实世界试验中连续命中七次。

Google 将这条流程称为 Video-to-Sim-to-Real，即从视频到仿真再到现实，并称它把数周的手工仿真建模压缩到数小时。如果这一流程能在更多技能上复现，普通视频就可能成为可扩展的机器人训练数据来源；目前证据仍是一项针对分布外物理技能的内部演示，七次连续命中也不能代表跨场景泛化能力。

## 能力越强，安全边界越要落到指标上

Google 将 Gemini 4 Argon 描述为其迄今最完整的安全与防护体系，并与多个第三方伙伴合作测量部分能力和防护措施。发布材料列出了三类防线，数字都应结合各自测试条件理解。

1. **滥用与越狱防御** ：Google 按 Frontier Safety Framework 评估多个高能力领域，并组合使用越狱抵抗、监控与执行机制。外部生物安全红队仅能解锁不到 1% 的受测生物子任务；对于编写 exploit 代码或协助处理危险生物材料等请求，Argon 拒绝完成有害请求的表现也优于此前模型。在另一组对抗性压力测试中，Google 报告违规量接近减半，但未列出基线的绝对次数。
2. **Agent 对齐与实时监控** ：训练目标包括尊重系统边界、诚实报告负面结果，以及在不可逆操作前暂停并请求人工确认。在模型没有动机主动自我克制的编码基准上，Argon 的捷径尝试率为 2%，是 Gemini 系列中的最佳结果。作为额外防线，自动 misalignment monitor 会读取 Agent 的逐步推理和动作，实时发现并阻止未授权的绕行做法；它被用于内部部署、能力评估和强化学习训练。Google 同时强调，训练中不直接惩罚模型的“想法”，以避免把推理过程压到不可观察的位置。
3. **间接提示注入防御** ：网页、邮件和代码里可能藏着劫持任务的恶意指令。通过对抗训练和自动化红队测试，Argon 在 Gray Swan Indirect Prompt Injection（IPI）基准上取得领先；在 15 次自适应攻击尝试的设定下，攻击成功率为 0.7%。

这些结果比笼统的“更安全”更有检查价值，但都不等于绝对安全。生物子任务、危险请求集、编码基准和 IPI 攻击预算分别覆盖不同风险，0.7% 也不是零；实时监控能否发现所有绕行行为，同样不能由现有数字推出。

## 价格、Token 限额与使用入口

为了支持更长、更复杂的科学、编码和 Agent 任务，Gemini API 为 Argon 提供 100 万 Token 的统一输入—输出限额。Google 同时引入新的 continuation protocol，开发者需要查阅文档并针对任务进行配置；这个 API 限额也不应被默认成所有应用入口都具有相同额度。

发布期的 introductory price 为每百万输入 Token 2 美元、每百万输出 Token 10 美元，缓存输入 Token 享受 95% 折扣。对于长任务，除了单价，还应把多轮调用、缓存命中和失败重试纳入实际成本。

| 使用对象 | 已公布的使用方式与条件 |
| --- | --- |
| 开发者 | 可在 AI Studio 和 Android Studio 中使用 Gemini API 付费档；Google AI Ultra 订阅者还可在 Google Antigravity 和 AI Studio 中体验。 |
| 企业 | 可通过 Gemini for business、API 和 Google Antigravity 使用。 |
| 普通用户 | Google AI Ultra 或 Pro 订阅者可在 Gemini App 和 Search 的 AI Mode 中使用。 |
| 科学家与研究人员 | Google 将向 10,000 名科学家提供 Google Antigravity 最高使用档，免费 6 个月；采用滚动接入，须申请，并受资格、容量和筛选条件限制。 |
| 网络安全防御者 | 可滚动申请加入 Fairwind Program；获批者可在无网络安全防护限制的条件下使用 Argon。 |

## 这次发布真正证明了什么

回到开头的 Chromium 重构，Argon 最有说服力的证据不是某一个榜单名次，而是多个任务已经落到可检查的结果上：进入仓库的性能优化、经 Lean 验证的量子子程序、通过湿实验的酶设计，以及在真实机器人上完成的动作。大型代码库维护者、需要多工具协作的专业团队，以及能把模型输出接入测试流程的科研人员，最值得优先关注。

但现有材料主要来自 Google、早期访问伙伴、内部测试和私有评测，许多案例缺少独立复现、完整基线和长期成功率。更稳妥的使用方式是先选一个边界清楚、能够自动测试且可回滚的任务，同时记录完成时间、质量、回滚率和 Token 成本，并保留关键步骤的人工确认。这样才能判断 Argon 在自己的工作流里究竟减少了多少真实成本，而不是把一次精彩演示误认为普遍可靠性；Google 也表示，Argon 只是 Gemini 4 家族的开始。

## 原文图表补充

![figure-1.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-1.svg)

- x.com

![figure-2.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-2.svg)

- Facebook

![figure-3.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-3.svg)

- LinkedIn

![figure-4.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-4.svg)

- Mail

![figure-5.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-5.svg)

- x.com

![figure-6.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-6.svg)

- Facebook

![figure-7.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-7.svg)

- LinkedIn

![figure-8.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-8.svg)

- Mail

> 本文参考自 [Gemini 4 Argon: Built to solve your hardest problems](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-argon/)