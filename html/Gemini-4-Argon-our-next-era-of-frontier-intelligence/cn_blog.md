# Gemini 4 Argon 发布：Google 的新一代前沿模型，把"深度思考"拉满到 100 万 Token

2026 年 9 月 30 日，Google 正式官宣新一代前沿模型 **Gemini 4 Argon**。这篇文章主要解决的问题是：如何让大模型在真实世界的软件工程、金融、法律、网络安全等 **长程复杂工作流** 中持续保持深度推理能力，而不是聊几轮就"断片"。它的核心思路简单粗暴——把输出 Token 上限从 64K 一口气拉到 **行业领先的 100 万**，让模型单次推理轨迹就能生成数十万 Token 的思考。最硬的成绩单：真实软件工程基准 DeepSWE v1.1 达到 **77.9%** 的新 SOTA，安全漏洞修复基准 CWE-bench v1 以 **68%** 并列第一。

![Gemini 4 Argon 官方宣传图](images/g4_30-09-26_key-art_blog.width-200.format-webp.webp)

> 图解：Gemini 4 Argon 官方发布主视觉。这一代模型的定位不是"更快的聊天助手"，而是面向专业复杂任务的"工作伙伴"。

## 不只是跑分：Argon 已经在 Google 内部"上岗"了

很多模型发布时只有 Benchmark 数字，Argon 不一样——它已经在 Google 内部真实干活了，数千名 Google 员工在日常使用。这里有几个非常能说明问题的案例：

- **量子算法优化**：Argon 帮助量子计算研究员优化关键子程序的时空资源（qubits × gates），其中一个例子在 **几分钟内就把已发表基线改进了 40%**。
- **内存效率优化**：一组 Argon Agent 分析全集群的 profiling 遥测数据，自主识别并应用内存优化，落地后释放超过 **300 TiB** 内存，预计总节省 500 TiB 到 1 PiB。
- **大规模代码迁移**：Argon Agent 正在把 Google 的 C/C++ 代码库迁移到 Rust，从 re2、libgav1 这样几万行的核心库，一路扩展到 **80 万+ 行的 Fuchsia Zircon 内核**。

最值得展开讲的是 libgav1 的案例。libgav1 是 Google 开源的视频解码器，Argon Agent 接手一个已有的 Rust 移植版本，通过多轮 profile 引导的实验、研究编译器输出，把 **3.2 万行手写 SIMD 代码** 替换成安全 Rust，让编译器自动向量化。

最终结果：一个内存安全的视频解码器，跑得比 Rust 移植版 **快 2.7 倍**，视频输出完全一致，性能逼近优化过的 C++ 原版。

这个设计的聪明之处在于：它没有让模型"硬写" SIMD，而是让模型理解编译器的行为模式，用编译器友好的高级代码换取自动向量化——既保住了 Rust 的内存安全，又追回了性能。当然，鉴于这些系统的关键性，这类大规模重写都要经过严格的自动化审计、仿真测试和人工 review 才会上线。

## 核心武器：100 万 Token 输出上限

聊完实战案例，我们来看看支撑这一切的关键改动。

Argon 把输出 Token 上限从上一代的 64K 直接拉到 **100 万**，这是目前的行业最高水平。为什么这个数字重要？因为长程任务的本质瓶颈往往不是上下文"读"得不够多，而是"想"得不够久——当模型有余量在单条轨迹中生成数十万 Token 的深度思考时，它就能一口气把难题想透，而不是被迫在中间截断、丢失推理状态。

![Gemini 4 Argon 能力基准表](images/gemini-4-argon_table_blog.gif)

> 图解：Gemini 4 Argon 在各基准上的能力总览。可以看到它在编码、推理、多模态等维度上全面压过前代，1M 输出 Token 是这一代最醒目的规格升级。

定价方面也已经公布：输入每百万 Token **2 美元**，输出每百万 Token **10 美元**，缓存输入 Token 享受输入价 **95% 的折扣**。对长程 Agent 场景来说，缓存折扣是个实打实的成本杀器。

## 编码与企业知识工作：全线领先

有了超长思考预算，Argon 在编码和跨领域企业工作流上的表现如何？我们逐个基准来看。

在衡量真实长程软件工程任务的 **DeepSWE v1.1** 上，Argon 以 **77.9%** 刷新 SOTA。

![DeepSWE 评测对比](images/gemini_4_cyber_evals_deepswe.gif)

> 图解：DeepSWE v1.1 评测结果，横轴为各模型，纵轴为解决率。这个基准测的是真实世界的长程软件工程任务（不是刷 LeetCode），Argon 的 77.9% 意味着它已经能在多步骤、跨文件的真实工程中独立交付。

编码之外，Argon 还是 **Vals Index** 的头名。这个指数按各行业对美国 GDP 的贡献加权，衡量模型在金融、编码、法律、税务工作中的经济影响力——换句话说，它比纯学术基准更接近"这模型能帮企业赚/省多少钱"。

![Vals Index 对比](images/gemini_4_cyber_evals_vals_index.gif)

> 图解：Vals Index 综合排名，覆盖金融、编码、法律、税务四大领域并按 GDP 权重合成。Argon 总分领先，说明它的能力分布不偏科。

细分领域的成绩单同样漂亮：

- **Vals Finance Agent v2**（多步骤金融研究）领先：

![Vals Finance 基准](images/gemini_4_cyber_evals_vals_finance.gif)

> 图解：Vals Finance Agent v2 评测，衡量模型完成多步骤金融研究任务的能力，例如连环的财报分析与数据核查。

- **Harvey Legal Agent Benchmark**（法律研究与文书起草）领先：

![Harvey 法律基准](images/gemini_4_cyber_evals_harveys.gif)

> 图解：Harvey 法律 Agent 基准结果，覆盖法律检索与文书起草。法律场景对长文档理解和严谨推理要求极高，恰好对上 1M 输出的强项。

- **AutomationBench**（Zapier 出品的端到端业务流程执行基准）：Argon 以 **51.3%** 排名第一。

![AutomationBench 基准](images/gemini_4_cyber_evals_automationbench.gif)

> 图解：AutomationBench 衡量模型跨核心业务职能的端到端执行能力——不只是给建议，而是真正把流程跑完。51.3% 看着不高，但已是全场第一，说明这类"真干活"的基准对全行业都还很硬。

值得一提的是多模态：当知识工作需要视觉理解时，Argon 同样能打——专业图表分析、长视频细节识别、基于一连串文档采取行动都不在话下。在长视频理解基准 **LVBench** 上，它以 **91.7%** 拿下 SOTA。

## 网络安全防御：本次发布的重头戏

如果说前面的能力是"全面升级"，那网络安全就是 Argon 的战略主攻方向。Google 专门训练了它的防御能力：Argon 可以 **自主发现、验证并修补** 关键软件漏洞。并且对于可信防御者和 Google 内部团队，Argon 会 **不带网络护栏** 发布，让防御方用上完整的前沿能力——这是个相当大胆的决策，后面安全章节我们再回头看它是怎么做风控的。

实战案例已经有一个很有说服力的：云安全公司 Wiz 通过其 "Scan for Good" 计划（免费保护关键公共基础设施）使用 Argon，模型在一家全球医院都在用的医疗软件中 **挖出了一个暴露敏感个人信息的严重漏洞**——而这个漏洞此前其他前沿模型都没发现。

基准方面，在衡量漏洞修复能力的 **CWE-bench v1** 上，Argon 以 **68%** 并列第一，延续了 3.8 Flash Cyber 在 CWE-bench v0 上的前沿表现。

![CWE-bench 评测](images/gemini_4_cyber_evals_cwe_bench.width-1200.format-webp.webp)

> 图解：CWE-bench v1 评测结果，纵轴为漏洞修复得分。68% 的成绩意味着在常见漏洞类型的自动修复任务上，Argon 已站上第一梯队。

漏洞发现能力相比 3.8 Flash Cyber 的跃升也很明显：

- 在 Google 内部综合漏洞基准上，Argon 在横跨 **20 种编程语言** 的复杂代码库中挖出了大量暴露面。
- 在 Wiz 的内部黑盒渗透测试基准（不看源码、直接分析线上 Web 系统）上，Argon 在攻击面发现、漏洞识别、产出 PoC 验证证据三个环节全面超过 3.8 Flash Cyber。

![安全漏洞发现对比](images/gemini_4_cyber_evals_security_vu.width-1200.format-webp.webp)

> 图解：Argon 与 3.8 Flash Cyber 在漏洞发现各环节的能力对比。黑盒渗透是最接近真实攻击者视角的场景，Argon 在这里的提升意味着防御方终于能在"攻击者之前"用上更强的自动化工具。

## 安全护栏：能力越大，闸门越多

能力拉到这个级别，Google 在发布节奏上明显谨慎：Argon 目前正通过 **Fairwind Program** 向可信网络防御者小范围开放，同时参与美国政府的自愿预发布模型评估流程，之后才逐步推向开发者、企业和消费者。上线前，Google 在四个方向加固前沿护栏：

- **防滥用**：针对网络和 CBRN（化学、生物、放射性、核）攻击风险，模型按 Frontier Safety Framework 拒绝有害请求，同时保留合法的军民两用科研空间。值得一提的是，Google 在加强 **监测模型内部激活** 来识别滥用的技术，且这些护栏经过内外部红队的手工+自动化混合攻击测试。
- **防 Prompt Injection**：间接提示注入（用恶意指令或上下文劫持模型行为）是 Agent 时代最现实的威胁。通过自动化红队和对抗训练，Argon 在 Gray Swan 的间接提示注入（IPI）基准上鲁棒性领先。

![Gray Swan IPI 评测](images/gemini_4_cyber_evals_gray_swan_i.width-1200.format-webp.webp)

> 图解：Gray Swan 间接 Prompt Injection 基准结果，越高代表模型越难被恶意上下文劫持。对要在企业里跑长流程的 Agent 来说，这项指标几乎和能力本身一样重要。

- **对齐监控**：部署 misalignment 缓解措施，监控 Argon 的 Chain-of-Thought 和行为，越界时中止执行。这套系统同样用于监控训练过程并向专门的事件响应团队报警。有个细节很有意思：Google 特意 **不把监控发现回灌进训练**，以免把模型的推理"塑造"成会规避监控的样子——并公开呼吁行业在这个能力跃升期保留推理透明度。
- **加固系统环境**：按照其 Agent 控制路线图，在高风险训练或评估开始前，对沙箱环境进行隔离和密封，并承诺与合作伙伴分享 Agent 安全最佳实践。

## 总结与展望

最后回顾一下这篇文章的要点：

- Gemini 4 Argon 是 Google 新一代前沿模型，主打 **长程复杂工作流** 上的持续深度推理。
- 输出 Token 上限从 64K 暴涨至 **100 万**，是这一代最核心的规格升级。
- 实战成绩：DeepSWE v1.1 **77.9%** SOTA、CWE-bench v1 **68%** 并列第一、AutomationBench **51.3%** 第一、LVBench **91.7%** SOTA。
- 已在 Google 内部创造真实价值：量子算法优化提速 40%、释放 300+ TiB 内存、80 万行内核级 C/C++ 到 Rust 迁移。
- 网络安全防御是主攻方向：不带护栏版本面向可信防御者开放，已挖出全球性医疗软件严重漏洞。
- 定价：输入 $2 / 输出 $10 每百万 Token，缓存输入 95% 折扣；发布走分阶段路线，先 Fairwind 可信测试者，再推向开发者、企业（付费 API 客户和 Google AI Ultra 订阅者优先）与消费者。

展望来看，Argon 最大的启示在于：前沿模型的竞争焦点正从"单次回答质量"转向"长时自主工作能力"，而 1M 输出 Token 加无护栏防御版本这两个动作，分别给 Agent 能力和攻防不对称格局开了新口子——接下来值得观察的是，Google 这套"分阶段发布 + 推理透明监控"的安全范式，能否成为行业级 frontier 模型发布的标准模板。

> 本文参考自 [Gemini 4 Argon: our next era of frontier intelligence](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/)