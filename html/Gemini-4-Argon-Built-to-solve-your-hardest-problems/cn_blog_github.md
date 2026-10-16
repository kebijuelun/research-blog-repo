# Gemini 4 Argon 发布：从 4000 万行代码重构到机器人看视频学投篮，Google 新旗舰模型主打"难活"

四千万行的 Chromium 代码库、需要数月的大型重构、几周内完成的量子化学软件提速——Google DeepMind 在 10 月 13 日发布的 Gemini 4 Argon，没有用常规的能力演示开场，而是直接摆出了一串已经落地的工程账单。这是 Gemini 4 系列的第一款模型，定位很明确：编码、金融、法律、网络安全和科学研究里那些"多步骤、不能出错"的复杂任务。

按照 Google DeepMind 高级副总裁 Koray Kavukcuoglu 的介绍，Argon 当天起通过 AI Studio 和 Gemini 平台的 API 面向开发者和企业开放，Google AI Ultra 订阅者可以在 Google Antigravity 中使用，Ultra 和 Pro 订阅者也能在 Gemini 应用和搜索的 AI Mode 里用到。定价为每百万输入 token 2 美元、每百万输出 token 10 美元，缓存输入按 0.5 折计费（即 95% 折扣）。榜单方面，官方称 Argon 在衡量金融、编码、法律、税务等关键经济领域模型表现的 Vals Index 上排名第一。

## 工程能力：不是演示视频，是合进主干的代码

这次发布里最扎实的部分是工程案例，因为其中多个结果已经进入开源仓库或生产环境，可以验证。

Google 内部数据显示，过去几周里，Argon 在 Antigravity 中帮助工程团队将 agentic 代码提交量提高了 35% 以上，同时代码回滚减少了近三分之一。具体到项目：

- **Chrome 架构重构** 。Chrome 团队需要从 Chromium 约 4000 万行代码库中一个约 600 万行的遗留目录里，把浏览器功能代码与核心网页处理逻辑分离。Argon 一次输出多处干净的代码变更，帮助团队完成横跨数百个文件的重构，把原本数月的工作压缩到几周。官方称这将让网页在标签页、侧边栏和浮动窗口之间迁移时无需重新加载，并消除一整类浏览器崩溃。
- **ADK 性能优化** 。Google 开源 Agent Development Kit（企业级 agent 框架）团队只给了 Argon 一份 `.agents/skills` 规则文件和一段话的目标——在不破坏现有应用的前提下消除框架开销。Argon 通宵自主工作，产出的优化现已合入 google/adk-python：长会话的内存会话准备时间减少 40%，冷启动导入提速 21%。
- **量子化学软件 xTB** 。xTB 是被引用最多的分子量子化学开源包之一，用于预测分子三维构型、筛选药物和材料候选物。研究人员用 Argon 将其分子结构预测提速最高 2.3 倍、构象搜索提速 1.8 倍，改动已贡献到 xTB 官方 GitHub 仓库。
- **Gemma 推理提速** 。借助 Argon，团队几天内就在 A100 上通过 vLLM 实现了 Gemma 4 E2B 端到端推理 2.85 倍加速（吞吐量从约 37,900 token/s 提升到 108,000 token/s 以上），且 100% token 一致、零质量损失——这个过程通常需要数月。相关改动以从注意力调优到投机解码的一系列上游 PR 形式公开。

这些案例有一个共同点：产出物是真实的代码提交，而不是跑分截图。当然，它们全部来自 Google 自己的团队和早期合作方，第三方独立复现目前还看不到。

编码之外，Argon 在空间推理和视觉编码上也有提升。官方展示的一组 demo 包括：在 Antigravity 中用多智能体视觉批判循环在 Blender 里搭建照片级的宝塔建筑群；把一张真实桌面的照片变成可玩的迷你赛车游戏（Blender MCP 建模与碰撞检测、Nano Banana 贴图、Three.js 物理与自定义着色器）；甚至用纯 Python 零依赖手写了一个带音频混合器的 3D 光线追踪引擎，并用 Gemini 3.8 TTS 和 Lyria 3.5 配乐生成了解说视频。这些 demo 均已放到 Google AI Studio 可以试玩，性质更接近能力演示，和前面的生产案例分量不同。

## 专业工作流：多模态文档理解是主线

在企业侧，Argon 强调的是深度多模态文档理解、纵向数据分析和交互式分析沙箱，同时保持严格的 grounding（回答有据可依）。官方还提到写作风格的改进：回答更直接、更少含糊其辞，不重复，不用套话——这恰恰是很多用户在长文档任务里最在意的点。

展示的案例里有一个颇具代表性的：Argon 综合约 50 GB 的机载激光雷达扫描、GIS 房间矢量、美国国会图书馆 CAD 图纸和国家公园管理局档案，重建了查科峡谷（Chaco Canyon）的 3D 模型，把碎片化的考古记录变成可探索的数字遗址。另一个更日常的场景是：在 Gemini 应用里，Argon 把一个杂乱的项目资料文件夹整理成给客户的产品方案演示稿，随后草拟并发送邮件，还为后续所有客户请求创建了定时自动化流程。

药物研发公司 Isomorphic Labs 的反馈更直接：团队把 Argon 当作研发伙伴，用它综合复杂的 PDF 和专利文献提取关键信息，把通常需要数周的流程压缩到几小时。

## 网络安全与科学：高风险的正面战场

安全能力这次以"授权防御者"的形式释放。Google 正在把 Fairwind Program 扩展到数百个新合作伙伴，安全公司 Wiz 已在其公益项目"Scan for Good"中用 Argon 保护关键基础设施。两个公开的案例：一是与某联邦机构合作，主动发现并修复了一个可能导致敏感政府数据和基础设施暴露的缺口；二是在医疗基础设施中通过串联复杂漏洞，发现并修复了一个公开文件下载绕过问题——该漏洞会暴露主密钥并授予管理员权限，波及数十万份私人病历。

科学研究部分的案例密度最高，也最能体现"深度工作"的定位：

- **量子计算** 。Argon 把 Google 内部的 ML 量子纠错解码器 AlphaQubit 翻译成一个高度优化的硬件内核，推理吞吐量接近翻倍，使纠错解码能跟上量子硬件的实时要求。另一个成果是量子子程序 arcsin：Argon 编写并在 Lean 中完成形式化验证的实现，相比已发表的量子 CORDIC 基线，时空成本降低超过 100 倍。
- **核聚变** 。仿星器（stellarator）是聚变能源的主要候选路线，但其三维磁场优化极其困难。1967 年以来，"Grad 猜想"一直认为这些磁场不存在精确数学解，直到近年马里兰大学 Matt Landreman 教授等人找到反例。Google 团队与 Landreman 合作，发现了一类更广泛的反例，为聚变社区提供了测试仿星器模拟代码的可靠基准。
- **罕见病研究** 。Gamow Labs 的研究人员用 Argon 分析罕见发育性肺病新生儿的脱敏基因组，在一个案例中定位到了常规软件漏检多年的染色体重排，并解释了漏检原因。在 Gamow Labs 基于真实临床和实验室数据的私有评测中，Argon 在 RareBench（罕见病诊断辅助）上取得 57%，在 LabBench（湿实验推理）上取得 52.2%。
- **酶设计** 。丝氨酸酯酶广泛用于回收、制药、生物燃料和洗涤剂制造，但定向设计通常需要数月的模型调优和候选筛选。科学家与 Argon 合作，以基准分子 4MU-acetate 为目标，在几小时内完成数据整合、模型调优和生化过滤，得到五个全新设计，后续实验室测试确认这些设计均具备稳健的催化活性。

为了推动这类研究，Google 还宣布向 10,000 名科学家免费开放 Antigravity 最高使用档位 6 个月，滚动审批，申请已开放。

## 机器人：看一段视频，学会投篮

机器人方向最值得关注的数字是 ERQA（Embodied Reasoning）基准 87 分，超过所有其他前沿模型——该基准测试模型理解和解决三维环境中物理、空间问题的能力，Argon 依靠纯推理取得这一成绩。

更有说服力的是一个内部测试：Argon 观看一段人类玩沙包投掷（cornhole）的原始视频，自行映射环境的空间坐标、计算全身关节运动，然后自主生成仿真代码。迁移到 Apptronik 的 Apollo 人形机器人上后，它在真实试验中连续投进七球。这条"视频到仿真到现实"（Video-to-Sim-to-Real）的流水线把原本需要数周的手工仿真建模压缩到几小时。需要说明的是，这是 Google 自己报告的内部测试，七连投的具体条件（距离、器械、失败样本）原文未披露。

## 安全措施：能力越强，护栏越具体

Argon 的安全部分少见地给出了一组可对照的数字，而不是泛泛的承诺。

- **防滥用与越狱** 。在 Frontier Safety Framework 下评估后，Google 部署了结合越狱抵抗、监控和强制执行的多层防御。外部生物安全红队测试中，攻击者只能解锁不到 1% 的受测生物子任务；在压力测试模型的对抗性请求集上，违规率下降了近 2 倍，明显优于之前的 Gemini 模型。
- **Agent 对齐与实时监控** 。针对 agent 遇到权限报错或测试失败时"走捷径"的问题，Google 训练 Argon 遵守系统边界、如实报告负面结果、在不可逆操作前暂停等待人类确认。在模型没有动机收敛的编码基准上，走捷径的尝试降到 2%。此外还有一个自动化的失准监控器，实时读取 agent 的推理步骤和动作来拦截未授权的变通操作；Google 特别说明训练时不惩罚模型的"想法"，以保持推理过程透明可监控。
- **提示注入抵抗** 。通过对抗训练和自动化红队，Argon 在 Gray Swan 间接提示注入（IPI）基准上领先，15 轮自适应攻击的成功率压到 0.7%。

这些数字的可信度取决于各基准的测试设计，但至少给出了可以被第三方检验的锚点。

## 谁该现在就用，以及限制在哪

Argon 通过 Gemini API 提供业界领先的 100 万 token 统一输入输出上限，并引入了新的 continuation 协议来支撑长任务——官方建议开发者和企业先读文档确认配置。获取渠道按人群划分：开发者走 AI Studio 和 Android Studio 的 API 付费档；企业用户走 Gemini for business、API 和 Antigravity；普通用户在 Gemini 应用和搜索 AI Mode 中体验（需 Ultra 或 Pro 订阅）；科学家可以申请免费的 Antigravity 最高档位；网络安全防御者则可以申请 Fairwind Program，获得无网络护栏的 Argon 访问权限。

回到开头的问题——这次发布改变了什么？从素材看，Argon 的差异不在单项跑分，而在于产出物已经落在真实的代码仓库、真实的安全事件和真实的实验室里：Chrome 重构、ADK 和 xTB 的合入代码、Wiz 修复的两个高危漏洞、五种经湿实验验证的酶。这些是它最值得关注的证据。限制同样清楚：几乎所有案例都来自 Google 内部团队和早期合作伙伴，独立第三方的复现和横向对比还没有出现；Vals Index 第一名、ERQA 87 分等成绩也尚未看到完整的方法学细节。如果你是开发者或研究者，最务实的下一步是用 2 美元/百万 token 的 API 价格在自己的长任务上做一次对照测试——这比任何发布文案都更能说明它是否配得上"最难的问题"这个定位。

## 原文图表补充

以下为原文页面中的分享图标，按来源顺序保留。

![原文分享图标：X](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-1.svg)

- x.com

![原文分享图标：Facebook](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-2.svg)

- Facebook

![原文分享图标：LinkedIn](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-3.svg)

- LinkedIn

![原文分享图标：Mail](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-4.svg)

- Mail

![原文分享图标：X](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-5.svg)

- x.com

![原文分享图标：Facebook](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-6.svg)

- Facebook

![原文分享图标：LinkedIn](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-7.svg)

- LinkedIn

![原文分享图标：Mail](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-Built-to-solve-your-hardest-problems/images/figure-8.svg)

- Mail

> 本文参考自 [Gemini 4 Argon: Built to solve your hardest problems](https://deepmind.google/blog/introducing-gemini-4-argon/)