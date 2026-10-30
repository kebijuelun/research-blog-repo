# Le Chonk：编程 Token 省 47%

编程 Agent 要真正进入日常开发，光会写代码还不够：它既要完成仓库级任务，又不能靠无限增加输出 Token 来换成功率；对安全团队来说，模型能否私有化部署、能否配合防御工作，也同样关键。Mistral 现在开放了其 1 万亿参数、原生多模态模型 Mistral Large 4（Le Chonk）的权重，并把编程、网络防御和推理效率作为这次更新的重点。经过公共预览后的额外三周强化学习，它在 Artificial Analysis Coding Agent Index 上得到 48%，与 GLM-5.3 基本持平，同时平均每个任务少用 47% 的 Token。

这组数字比一句“旗舰模型开放权重”更值得关注：Le Chonk 试图同时改善能力、运行成本和组织对模型的控制权。不过，它并不是每项评测都领先，1 万亿参数也意味着部署门槛不能忽略。下面按原文给出的测试条件和应用案例，看看这次更新到底改变了什么。

## 三周强化学习后，开放的是改进版权重

Mistral Large 4 本月早些时候以公共预览形式发布。Mistral 表示，团队随后根据开发者、早期客户和合作伙伴的反馈继续扩大强化学习训练，主要提升编程和防御性网络安全能力，并改善推理效率；这次开放下载的正是改进后的模型，而不是简单把预览版权重原样放出。

模型本身采用 MoE（混合专家）架构，把指令执行、推理和 Agent 能力放在同一个模型中，支持多模态输入，并原生覆盖 160 多种语言。Mistral 还强调，它从零开始在欧洲基础设施上训练，从训练到发布具有清晰的来源链路。完整技术报告尚未发布，因此原文没有进一步给出专家数量、激活参数、上下文长度或训练数据构成，暂时不能只凭“1 万亿参数”推断实际部署成本。

## 编程能力：先看交付质量，再看 Token 消耗

仓库级编程的难点不是生成一段看起来合理的代码，而是修改能否通过整个项目的约束。Cognition 的私有基准 FrontierCode 2.0 正是从端到端代码质量出发，并考察仓库维护者是否愿意合并 Agent 的改动。Mistral Large 4 在这项测试中得到 19.7%，高于图中 DeepSeek V4.1 Flash 的 19.3%、GLM-5.3 的 17.8% 和 Kimi K3 的 17.3%。

![FrontierCode 2.0 编程基准得分对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-4.png)

> 图解：纵轴为 FrontierCode 2.0 得分。Mistral Large 4 以 19.7% 位列图中第一，但领先 DeepSeek V4.1 Flash 只有 0.4 个百分点；同时，19.7% 也说明这类端到端仓库任务远未被解决。该基准是私有的，外部读者无法仅凭这张柱状图检查题目和评分明细。

只看成功率还不够，Agent 为完成任务要写多少 Token，会直接影响延迟和推理费用。在 Artificial Analysis Coding Agent Index 上，Mistral Large 4 得分 48%，Mistral 将其描述为与领先开放权重模型 GLM-5.3 基本持平；区别在于，前者平均每个任务少用 47% 的 Token。这里的结论不是“分数大幅领先”，而是用更少的输出达到相近表现。

![Coding Agent Index 得分与每任务 Token 消耗对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-5.png)

> 图解：横轴是每个任务平均消耗的 Token，单位为百万；纵轴是 Coding Agent Index 得分，因此越靠左上方，代表得分与 Token 效率的组合越好。图中同时给出 Mistral Large 4 的 High 和 Medium 推理档位，说明效率并非只有一个固定工作点。

终端工作流的变化更直观。Terminal-Bench 4.0 上，Mistral Large 4 从公共预览版的 28% 提升到 41%，增加 13 个百分点，同时输出 Token 减少 30%；图中 GLM-5.3 为 40%。也就是说，这次强化学习带来的不是单纯的分数上涨，而是能力与输出量同时向更有利的方向变化。

![Terminal-Bench 4.0 得分对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-6.png)

> 图解：橙色柱是 Mistral 自报的公共预览版结果，白色柱为本次模型及其他对比模型。Mistral Large 4 的 41% 比预览版高 13 个百分点；原文另给出输出 Token 减少 30%，图中没有展示 Token 的绝对数量。

为了让开发者主动选择性能与成本的平衡，Le Chonk 提供三种推理模式。这里的“推理档位”可以理解为给模型分配不同的思考预算：简单任务不必一律使用最高档，复杂任务则可以保留更充分的推理过程。

| 模式 | 建议用途 |
| --- | --- |
| Low | 简单任务；取代原来的 None 推理模式 |
| Medium | 日常任务 |
| High（默认） | 需要高精度和充分推理的最困难任务 |

除了自动基准，Mistral 还引用 Surge Code 的盲测成对比较作为人类偏好的证据。评测者更偏好 Mistral Large 4 而非 GLM-5.3、Kimi K3 的回答，两种对比中 Mistral 的偏好率均为 64%；面对 DeepSeek V4.1 Flash 时为 56%。面对 Claude Opus 5.5，Mistral 的偏好率只有 22%，所以在该次评测覆盖的模型里，它排在 Claude Opus 5.5 之后，而不是全面领先。

![Surge Code 盲测中 Mistral Large 4 相对各模型的回答偏好率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-7.png)

> 图解：每行左侧数值是评测者偏好 Mistral Large 4 的提示词占比，50% 表示双方打平。它对 GLM-5.3、Kimi K3 和 DeepSeek V4.1 Flash 分别为 64%、64% 和 56%，对 Claude Opus 5.5 则为 22%。原文未在正文中给出样本量，因此不宜把偏好率外推到所有编程场景。

## 网络防御：既要修得了漏洞，也要挡得住滥用

网络安全任务把两种看似相反的要求放在了一起：模型需要理解漏洞、复现问题并修改代码，又不能把同样的能力无条件用于真实攻击。Artificial Analysis Cyber Index 测量的是一条完整防御链路——在代码库中发现漏洞，复现并验证它们，再完成不破坏现有功能的补丁。Mistral Large 4 得分为 51%，Mistral 同时表示，它会执行这些正当防御任务，而不会不必要地拒绝。

![Artificial Analysis Cyber Index 网络安全能力对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-8.png)

> 图解：Mistral Large 4 得分 51%，高于图中的 GLM-5.3 Flash（50%）、Kimi K3（41%）和 GLM-5.3（36%），但低于 MiMo-V2.6-Pro（56%）与 DeepSeek-V4.1-Flash 1017（57%）。因此，这张图支持“具备较强防御能力”，并不支持“所有网络安全指标第一”。

安全护栏的结果来自另一组测试。Mistral 报告，模型在 JailbreakBench、StrongREJECT 和 AgentHarm 的网络安全相关提示上，对恶意请求的平均拒绝率达到 96.6%，高于公共预览版，也是其测得的开放模型最高值。需要区分的是，Cyber Index 衡量防御任务能力，拒绝率衡量对恶意请求的抑制，两者不能合并成一个笼统的“安全分数”。

![恶意网络安全请求平均拒绝率对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-9.png)

> 图解：该图汇总 JailbreakBench、StrongREJECT 和 AgentHarm 中网络安全相关提示的平均拒绝率，柱上数字经过取整。Mistral Large 4 在正文中报告的精确值为 96.6%，预览版约为 95%；纵轴从 40% 起，而不是从 0 起，读图时不应按柱高直接比较比例。

合作伙伴的反馈补充了基准之外的应用视角，但仍应视为各方自报结果。Booz Allen 表示，Le Chonk 在其内部源代码安全分析基准上，漏洞检测召回率可与其他前沿网络安全模型相当，并提到成本曲线和主权部署的吸引力；Elastic 已用真实网络安全工作负载评估模型，ENISA 则在测试漏洞识别、代码审计和防御日志分析。这些说法说明测试正在进入具体业务流程，但原文没有给出统一的样本、召回率或成本数字，不能当成同一条件下的横向实验。

Mistral 对现实风险也保留了一段重要说明：团队监测线上流量时，虽然看到模型拒绝针对真实目标的黑盒攻击，但仍认为它可能提升真实攻击的执行能力，因此正在调整护栏，并计划继续更新。面向高级渗透测试、漏洞评估和红队工作的专用变体，只向经过严格审查的企业设计合作伙伴开放预览，还配有合同约束，测试必须限于自有或明确授权的系统及受控环境。

对企业而言，另一条路径是用 Mistral Forge 加入自有安全数据、策略、检测逻辑和威胁环境。对于边界清晰的任务，Mistral 表示还可以把模型进一步优化成更小的专用变体，以提高准确率；这与“所有安全任务都直接运行一个万亿参数通用模型”是不同的部署思路。

## 企业工作：表格、网页浏览与 Agent 防护

编程和安全之外，这轮强化学习也延伸到常见知识工作。财务与会计场景中，Le Chonk 重点改善电子表格的创建和编辑，在 Finch 上得到 73.3%，在开放模型中仅次于 Kimi K3。图中其公共预览版为 67.4%，新版本提高 5.9 个百分点。

![Finch 财务工作基准得分对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-27.png)

> 图解：Finch 关注财务实际工作流，尤其是电子表格处理。Mistral Large 4 从预览版的 67.4% 提升到 73.3%，高于图中的 GLM-5.3（65.1%）和 DeepSeek-V4-Pro-0813（67.4%），但低于 Kimi K3（77.3%）。

网页浏览能力对应 BrowseComp。Le Chonk 达到 84.5%，高于 GLM-5.3 的 82.5% 和 DeepSeek V4.1 Flash 1017 的 83.4%；与早期监督微调（SFT）检查点的 75% 相比，强化学习阶段带来 9.5 个百分点的提升。图中 Kimi K3 仍更高，为 91.2，所以这项结果同样不是全面领先。

![BrowseComp 网页浏览能力对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-28.png)

> 图解：橙色柱为早期 Mistral Large SFT 检查点，白色柱为强化学习后的 Mistral Large 4 及其他模型。84.5 相比 75 的提升展示了后续训练带来的变化，而模型间比较仍限于 BrowseComp 这一基准及其测试设置。

当 Agent 面向内部员工或外部用户运行时，系统提示词泄露、恶意代码注入等攻击会直接进入应用链路。B3 Agent Security Benchmark 上，Le Chonk 的攻击抵御率从预览版的 93.3% 提升到 97.1%，增加 3.8 个百分点。Mistral 据此认为，自行托管模型的组织无需额外调优也能获得安全的 Agent 基础；更稳妥的理解是，模型内置防护已有改善，但单项基准不能替代应用层权限控制与持续测试。

![B3 Agent Security Benchmark 攻击抵御率对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Mistral-Large-4-Le-Chonk-is-now-open-weight-Mistral/images/figure-29.png)

> 图解：柱上数字经过取整，Mistral Large 4 的精确结果为 97.1%，预览版为 93.3%。该指标描述模型抵抗基准攻击的比例，不等于生产系统中所有 Agent 攻击的总体拦截率。

## 三个案例，展示 Agent 如何形成执行闭环

原文给出的演示比单点问答更接近完整任务：模型不只生成答案，还要获取上下文、执行、检查并迭代。

- **漏洞分诊**：Agent 读取新收到的漏洞报告，结合源代码和产品上下文调查，在隔离沙箱中复现缺陷，再编写并迭代补丁，直到用于验证的漏洞利用不再生效。
- **Agent 式 Web 开发**：在无人协助的情况下，Le Chonk 构建 Orbital Trace，用真实跟踪数据展示约 19,000 个绕地卫星和碎片。它写完应用后会在浏览器中实际使用，查找自身缺陷和性能问题，再根据独立审查者的反馈修复，直到需求全部通过。
- **量子化学求解器**：模型从零编写计算分子中电子分布的软件，并用标准化学工具校验结果；随后再做一个 3D 页面，让用户选择苯或水等分子，拖动滑块拉开分子，观察化学键断裂时电子如何重新分布。

这三个案例分别强调安全修复、软件交付和科学计算，但共同点是“执行—验证—修正”的闭环。原文没有提供这些演示的成功率、耗时或人工筛选过程，因此它们适合说明能力形态，不适合当作稳定性数据。

## 开放权重的价值，在于把控制权交给使用方

Mistral 所说的“主权 AI”，核心不是 API 从哪里调用，而是组织能否检查、定制并在自己的基础设施上运行模型，同时不依赖模型厂商持续提供 API。对金融、政府和关键基础设施团队，这种控制还涉及数据驻留、模型行为可见性，以及 Agent 能访问什么数据和权限。

Le Chonk 的支持者也给出了不同场景：Belfius 看重微调、私有部署以及与银行数字渠道的集成；ESA Φ-lab 希望围绕图像和技术文档，适配科学家的地球观测工作；ABN AMRO 关注金融行业中的透明生态与网络防御。它们说明的是开放权重带来的选择空间，而不是已经验证的普遍业务收益。

主权部署也不等于所有组织都必须自己维护集群。对偏好托管服务的用户，Mistral Large 4 可通过 Microsoft Foundry 使用，Azure 开发者还有私有部署选项。Mistral Studio、Forge 和 Applied AI 团队则分别覆盖应用构建与治理、专有数据定制以及联合交付。这里的取舍很实际：自托管提供更强的直接控制，托管和私有部署则可能降低运维负担，具体边界取决于所选服务。

## 如何试用，以及价格和许可证的限制

如果想快速验证自己的任务，可以通过 Mistral Studio、Mistral API、OpenRouter 或 Vibe Code 试用；需要完整控制权时，可以从 Hugging Face 下载权重，用 vLLM 或 SGLang 部署。NVIDIA 也提供托管端点和适用于 vLLM 的优化 NIM 容器，开发者还可使用 NeMo AutoModel 做定制、TensorRT LLM 做高吞吐服务。完整技术报告仍待发布。

API 定价按每百万 Token 计算，缓存输入与普通输入分开计费：

| 项目 | 每百万 Token 价格（美元） |
| --- | ---: |
| 输入 | 0.70 |
| 缓存输入 | 0.07 |
| 输出 | 2.10 |

模型使用 Mistral Ecosystem Open-Weight 许可证，即修改版 MIT 许可证。用户可以免费使用、修改、微调和商业化，但有一个关键例外：经营 Model-as-a-Service（模型即服务）的企业，在将该模型用于商业服务前，必须从 Mistral 取得单独的商业许可证。因此，“开放权重”不等于所有商业形态都无附加条件，上线前应核对许可证文本。

## 谁值得现在测试 Le Chonk

- 对编程团队，最有价值的信号是 Coding Agent Index 得分相近时少用 47% Token，以及 Terminal-Bench 提升 13 个百分点、输出减少 30%；应直接用自有仓库验证成功率、延迟和总费用。
- 对安全团队，51% 的 Cyber Index 与 96.6% 的恶意请求平均拒绝率需要分开看，前者是防御能力，后者是滥用抑制，任何一项都不能替代授权边界和应用层防护。
- 对重视数据驻留与定制的企业，开放权重、自托管、Forge 定制和私有托管提供了多种控制层级，但 Model-as-a-Service 的商业许可限制需要提前确认。
- 对准备自行部署的开发者，当前最大的未知仍是完整技术报告、硬件需求和可复现的独立评测；1 万亿参数模型的实际资源成本，不能从 API 单价推导出来。

Le Chonk 最有力的证据在于编程效率和从预览版到开放权重版的多项提升，而不是每项榜单都第一。如果开发者能在自己的编程、防御或企业工作流中复现这些收益，开放权重才会从发布口号变成真正的部署优势。

> 本文参考自 [Mistral Large 4 (Le Chonk) is now open weight | Mistral](https://mistral.ai/news/mistral-large-4-oss/)