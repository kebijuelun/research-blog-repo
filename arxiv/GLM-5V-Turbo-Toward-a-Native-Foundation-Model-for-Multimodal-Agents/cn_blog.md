# GLM-5V-Turbo 解读：面向多模态 Agent 的原生基础模型

## 一句话概括

GLM-5V-Turbo 这篇技术报告的核心，不是单纯再做一个“看图更准”的 VLM，而是把 **视觉感知、语言推理、工具调用、代码生成、GUI 操作、长程任务执行** 放到同一个 Agentic 系统中重新设计。

它想解决的问题可以概括为：

> 未来的智能体不只是读文本、写代码、调用函数，而是要能看懂网页、文档、截图、图表、视频和 GUI 环境，并在这些真实数字化环境中完成连续任务。

因此，GLM-5V-Turbo 的定位是一个面向 **multimodal agents** 的原生基础模型：视觉不是外挂接口，而是推理、规划和执行的一部分。

---

## 背景：为什么多模态 Agent 不能只靠“语言模型 + 截图输入”？

过去很多 Agent 系统，本质上是以文本为中心的：

1. 用户给出任务。
2. 模型生成计划。
3. 调用搜索、浏览器、代码执行等工具。
4. 根据返回文本继续推理。

这个范式在纯文本场景里有效，比如代码生成、资料整理、问答检索。但一旦进入真实数字化环境，就会遇到明显瓶颈：

- 网页内容往往是图文混排，关键信息可能藏在表格、图表、按钮、弹窗和布局结构里。
- GUI 操作依赖空间位置和视觉识别，不只是 DOM 文本。
- PDF、PPT、论文、报告中的知识经常以图片、公式、版式或截图形式存在。
- 网站复刻、UI-to-code、PRD-to-app 等任务需要模型理解视觉设计和交互流程。
- 长程 Agent 执行中，模型必须持续记住之前看到过的视觉状态。

所以，论文强调一个观点： **Agentic capability 的边界不再由语言推理单独决定，而是由多模态感知、工具链、执行框架和验证机制共同决定。**

---

## 整体贡献：GLM-5V-Turbo 做了什么？

从论文内容看，GLM-5V-Turbo 的改进主要分为五条线：

1. **模型设计** ：提出面向细粒度多模态理解的视觉编码器 CogViT。
2. **训练机制** ：提出 Multimodal Multi-Token Prediction，兼顾多模态建模和工程效率。
3. **大规模训练** ：在预训练、SFT 和 RL 阶段深度融合视觉、语言、代码、GUI、工具使用等任务。
4. **Agent 工具链** ：扩展多模态搜索、图像处理、网页阅读、内容生成、Deep Research 等工具。
5. **生态集成** ：适配 Claude Code、AutoClaw、OpenClaw 等外部 Agent 框架，并提供官方 skills。

论文给出的代表性结果包括：

| 能力方向 | Benchmark | GLM-5V-Turbo 结果 |
|---|---:|---:|
| 多模态编码 | Design2Code | 94.8 |
| 多模态搜索 | ImageMining | 30.7 |
| 多模态搜索 | BrowseComp-VL | 51.9 |
| 多模态搜索 | MMSearch | 72.9 |
| 视觉问答 | SimpleVQA | 78.2 |
| GUI Agent | AndroidWorld | 75.7 |
| GUI Agent | OSWorld | 62.3 |
| Claw Agent | PinchBench | 87.0 / 80.7 |
| Claw Agent | ClawEval | 57.7 / 75.0 |
| Claw Agent | ZClawBench | 57.6 |
| Text-only Coding | CC-Backend | 22.8 |
| Text-only Coding | CC-Frontend | 68.4 |
| Text-only Coding | CC-RepoExploration | 72.2 |

这里比较值得注意的一点是：GLM-5V-Turbo 在增强视觉能力的同时，并没有明显牺牲 text-only coding 能力。这对 Agent 来说非常关键，因为真实任务往往同时包含文本推理、代码修改、网页操作和视觉判断。

---

## CogViT：面向 Agent 的视觉编码器

GLM-5V-Turbo 的视觉侧核心是 **CogViT** 。论文对它的定位很明确：不是只做粗粒度图像分类，而是要服务于 Agent 场景里的细粒度理解。

![CogViT 性能对比](figures/cogvit.png)

> 图解：这张图展示 CogViT 在通用视觉理解、细粒度识别、几何感知和空间理解等任务上的表现。它的意义在于说明 GLM-5V-Turbo 并不是把现成视觉编码器简单接到语言模型上，而是专门强化了 Agent 需要的视觉能力，比如 UI 结构、局部细节、空间关系和复杂图文内容。

### 第一阶段：蒸馏式 Masked Image Modeling

CogViT 的第一阶段是视觉表征学习，采用 distillation-based masked image modeling。

训练时会随机 mask 图像区域，然后让 student ViT 在教师模型的特征空间里重建被遮挡区域。这里使用了两个 teacher：

- **SigLIP2** ：提供更偏语义的表征。
- **DINOv3** ：提供更偏纹理和局部结构的表征。

训练设置包括：

- mask ratio 为 $35\%$。
- 输入分辨率为 $224 \times 224$。
- 数据混合为 80% 高质量自然图像、10% instruction-following 数据、10% 科学图像。
- 使用 Muon optimizer 和 cosine decay schedule。
- 引入 QK-Norm 来稳定 attention 计算。

QK-Norm 的作用可以简单理解为：在计算 attention logits 之前先规范化 query 和 key，降低大规模训练中 logit 爆炸的风险。

标准 attention 可以写成：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

QK-Norm 的思想是在 $QK^T$ 之前对 $Q$ 和 $K$ 做归一化，从而让 attention 分布更稳定。

### 第二阶段：图文对比学习

第二阶段从纯视觉表征转向 image-text alignment，也就是让图像和文本进入共享语义空间。

这一阶段有三个重要升级：

1. 使用 **NaFlex** 处理可变尺寸图像，保留原始宽高比，而不是强行 resize 到固定大小。
2. 使用 SigLIP loss，并把 global batch size 扩展到 64K。
3. 使用 80 亿中英双语图文语料，增强跨语言多模态理解。

这对中文用户特别重要，因为很多多模态模型在英文图文数据上表现不错，但遇到中文网页、中文文档、中文图表时能力会明显下降。GLM-5V-Turbo 显然把中英双语多模态能力作为基础能力来训练。

---

## MMTP：多模态 Multi-Token Prediction 怎么做？

论文提出的第二个关键设计是 **Multimodal Multi-Token Prediction** ，简称 MMTP。

传统 Multi-Token Prediction 的目标是：模型不仅预测下一个 token，还预测未来多个 token，从而提升训练效率和推理友好性。

但扩展到多模态后，会遇到一个工程问题：

> 图像 token 应该如何传给 MTP head？

论文比较了三种方案：

| 方案 | 做法 | 问题或优势 |
|---|---|---|
| Option 1 | 直接把视觉 embedding 传给 MTP head | 通信复杂，跨 pipeline stage 传输成本高 |
| Option 2 | 在 MTP head 中 mask 掉所有视觉 token | 退化成 text-only MTP，损失多模态位置信息 |
| Option 3 | 保留视觉 token 的位置，但用共享的 `<|image|>` token 替代视觉 embedding | 训练更稳定，工程实现更简单 |

GLM-5V-Turbo 最终选择了第三种方案。

![MMTP 设计示意](figures/model-mtp-v3.png)

> 图解：这张图展示了多模态 MTP 的结构设计。关键点在于，MTP head 并不直接接收复杂的视觉 embedding，而是使用共享的 `<|image|>` token 表示图像位置。左下角的 loss 曲线说明，相比直接传视觉 embedding，这种设计训练 loss 更低、收敛更稳定。

这个设计很有工程味道。直觉上，直接传视觉 embedding 看起来信息最完整，但实际大模型训练要考虑 pipeline parallelism、sequence parallelism、context parallelism 等分布式训练机制。如果视觉 embedding 在不同 stage 之间频繁传输，会显著增加通信复杂度。

使用 `<|image|>` token 的好处是：

- 保留图像 token 在序列中的位置信息。
- 避免跨 stage 传输大块视觉 embedding。
- 让 MTP head 输入分布更接近文本 token。
- 降低工程实现复杂度。
- 在小规模 ablation 中表现出更低 loss。

这个选择体现了论文的一个重要思路： **多模态模型设计不能只看理论表达能力，还必须考虑大规模训练和推理基础设施是否承受得住。**

---

## 训练：从感知、推理到 Agent 能力的联合优化

GLM-5V-Turbo 的训练不是单点强化某个 benchmark，而是覆盖多类任务的广域训练。

预训练阶段的数据包括：

- plain text
- interleaved image-text
- OCR
- coding
- GUI
- video
- multimodal tool-use
- spatial perception
- grounding
- academic problem-solving

其中，论文特别强调了 **multimodal coding data** 。原因很直接：UI-to-code、SVG coding、frontend recreation 这类任务，会强迫模型理解布局、结构、相对位置和局部细节，而不是只做粗粒度图像描述。

### Joint RL：超过 30 类任务的强化学习

GLM-5V-Turbo 在后训练阶段进行了覆盖 30 多个任务类别的 joint RL。论文列出了几个增益：

| 能力类型 | 任务 | 相比 SFT 的提升 |
|---|---|---:|
| 感知 | RefCOCO-avg | +4.8% |
| 感知 | PointBench | +3.2% |
| 视频理解 | MVBench | +5.6% |
| 3D grounding | SUNRGBD | +7.7% |
| OCR | OCRBench | +4.2% |
| 图表理解 | CharXiv | +7.7% |
| STEM 推理 | MMMU / MathVista / LogicVista 等 | +1.8% |
| GUI Agent | OSWorld | +4.9% |
| Coding Agent | CC-Backend | +0.2% |
| 工具使用 | MMSearch | +3.5% |

这里有一个很重要的观察：相比 SFT，RL 在多任务场景中似乎更不容易出现强烈的跨域干扰。也就是说，多类任务可以一起提升，而不是一个任务涨、另一个任务掉。

论文给出的解释偏经验性：多任务 RL 让模型接触到更丰富的策略分布，从而更容易学习可迁移的 thinking patterns。比如，视觉 grounding 中学到的定位和验证习惯，可能迁移到 GUI 操作；UI-to-code 中学到的布局理解，也可能帮助多轮网页开发。

但论文也提醒：如果某类能力在 RL 中完全没有覆盖，后训练后仍可能下降。这说明 RL 的任务覆盖范围会影响模型最终的 generalization boundary。

---

## 大规模多模态 RL 基础设施：难点不只是模型

论文用了相当多篇幅讨论多模态 RL 的训练系统，这部分很容易被忽略，但对 Agent 模型很关键。

多模态 RL 比传统 RLHF 更难，主要因为：

- prompt 和 response 长度差异很大。
- 有 single-step 任务，也有 multi-step 任务。
- 每类任务可能需要不同 verifier。
- 图像和视频会带来巨大的 token 和显存压力。
- rollout 中会出现长尾请求，拖慢整体 pipeline。

GLM-5V-Turbo 从四个方面重构训练系统。

### 统一任务和奖励抽象

论文提出统一的 VLM RL Gym，用一个环境接口处理 single-step 和 multi-step 任务。

同时，reward system 独立管理多个 verifier：

- rule-based verifier 本地同步执行。
- model-based judge 通过 API 异步调用。
- 多个 verifier 的结果通过可配置策略聚合成 reward。

这种设计的好处是，训练主链路不用和具体评测逻辑绑死，便于混合任务训练。

### 全流程异步与阶段重叠

传统 rollout 往往需要等一个 batch 全部完成后再算 reward。GLM-5V-Turbo 的做法是：每个 inference request 完成后立刻触发 callback 和 reward 计算。

这样可以减少 long-tail request 带来的空转。

此外，它还把 rollout inference、reward evaluation、batch construction、weight transfer 解耦，让 CPU-GPU 传输、reference model forward 和训练步骤尽量重叠。

### 多模态显存管理

文本训练里的 activation recomputation 不能直接解决多图、多视频输入带来的显存问题。因此论文对 ViT 和 projector 设计了单独的显存策略，包括 targeted recomputation 和 CPU offloading。

这可以避免 activation memory 随图像数量线性爆炸。

### 视觉输入的负载均衡

长视频输入会产生长度差异很大的视觉 token 序列。如果 naive 地在 forward 中再做切分，每个 rank 可能需要先持有完整 patch tensor，显存和通信都浪费。

论文的方案是把 CP 和 TP partitioning 提前到 data-loading 阶段，并根据 downsample group 对齐边界。随后通过 asynchronous all-to-all 把每个 rank 真正需要的部分分发过去。

这类系统优化说明，GLM-5V-Turbo 的“多模态原生”不只是模型结构上的原生，也包括训练系统对视觉 token 的原生支持。

---

## 多模态工具链：从“会看”到“会做”

论文中非常关键的一节是 **Multimodal Toolchain Expansion** 。GLM-5V-Turbo 扩展了多模态工具链，让模型能形成更完整的 perception-planning-execution loop。

工具大致分为三大类：

| 场景 | 工具集合 | 代表工具 |
|---|---|---|
| General | Recognition Tools | `zai_recognize_plant`、`zai_recognize_location`、`zai_recognize_person` |
| General | Multimodal Search | `zai_search_web_text`、`zai_search_web_by_image`、`zai_search_similar_images`、`zai_search_scholar` |
| General | Browser Tools | `zai_load_image_from_url`、`zai_read_webpage` |
| General | Image Processing | `zai_crop_image`、`zai_draw_image_bounding_boxes`、`zai_draw_video_objects_tracking` |
| Creation | Web Creation | `submit_plan`、`apply_edits`、`zai_generate_web_html` |
| Creation | Slide Creation | `zai_generate_slide_html`、`zai_generate_outline_ppt` |
| Deep Research | Multimodal DR Tools | `zai_dr_python`、`zai_dr_open_url_mm`、`zai_dr_search`、`zai_dr_images_lens` |

这套工具链的意义在于，模型不再只是“看一张图然后回答”，而是可以：

1. 观察截图或网页。
2. 判断需要搜索、裁剪、放大还是标注。
3. 调用工具获取更多证据。
4. 更新任务状态。
5. 继续下一轮操作。
6. 最终生成网页、报告、PPT 或代码结果。

比如网站复刻任务中，模型可以先通过 GUI agent 探索页面结构，截图并理解布局，再收集素材，最后用 UI-to-code 能力生成 HTML。

---

## 与 Claude Code 和 AutoClaw 集成：模型成为系统级协作者

GLM-5V-Turbo 还强调了与外部 Agent 框架的集成，尤其是 Claude Code 和 AutoClaw。

可以这样理解三者分工：

| 组件 | 角色 |
|---|---|
| GLM-5V-Turbo | 视觉语言控制器和高层认知核心 |
| Claude Code | 处理本地代码、终端、文件系统和工程任务 |
| AutoClaw | 提供浏览器和 GUI 自动化执行能力 |

这种组合让 GLM-5V-Turbo 从“被动生成代码”变成“主动参与系统执行”的 Agent 核心。

它可以看屏幕、理解当前状态、规划下一步，再把具体执行交给框架。论文认为，这标志着模型角色发生变化：它不再只是 text-based assistant，而是 grounded in real-world environments 的 multimodal actor。

---

## ImageMining：用图像进行深度搜索的新 Benchmark

论文提出了一个新的 benchmark： **ImageMining** 。

它的核心理念是：

> think with image, deep search with image

也就是说，模型不能只把图像当成一次性输入，而要围绕图像进行多步搜索、裁剪、放大、交叉验证和推理。

ImageMining 包含 217 个人工整理的测试样例，覆盖 7 个领域：

- Social
- Entertainment
- Products
- Places
- Rich Text
- Nature
- Science

以及 5 类推理任务：

| 推理类别 | 含义 |
|---|---|
| Universal Recognition | 细粒度识别动植物、物体、艺术品等 |
| Spatio-Temporal Reasoning | 基于视觉线索推断地点或时间 |
| Event Reasoning | 理解新闻事件、产品发布等 |
| Text-based Reasoning | 解析图中的论文、报告、富文本信息 |
| Visual Search | 通过图像交叉检索具体作品、地点或对象 |

![ImageMining 案例](figures/visual_search/image_mining_case.png)

> 图解：这张图展示的是 ImageMining 风格的复杂视觉搜索任务。模型需要从图像中的对象出发，识别其分布地点，再关联到小说、作者、电影改编和豆瓣页面海报信息。这里的难点不是单次视觉问答，而是图像线索、网页搜索、实体消歧和多跳推理的组合。

论文特别提到一个数据构造约束： **Visual Jump** ，即中间推理跳转必须包含视觉转移。这是为了防止模型只依赖文本捷径或参数知识，而是真正通过图像进行探索。

此外，ImageMining 还构造了 OCR Search 数据，覆盖 charts、maps、posters 等场景，要求模型先做 entity isolation 和 localized cropping，再发起搜索链。

![图像定位与酒店搜索案例](figures/visual_search/image_hotel_search_summary.png)

> 图解：这张图展示了通过输入图像定位地点，并进一步搜索指定日期酒店价格的任务。模型不仅要识别图像中的地点，还要结合用户给出的入住时间，进行真实世界信息检索、价格排序和体验建议整理。这类任务非常接近现实 Agent 使用场景。

---

## Multimodal Deep Research：从检索到图文混排内容生成

GLM-5V-Turbo 的另一个重要能力是多模态 Deep Research。它不只是搜索文字资料，而是能够读取网页、图表、截图、论文 PDF、结构化文档，并生成图文混排的报告、博客或 PPT。

![多模态 Deep Research 与论文博客生成](figures/mm-deep-research.jpg)

> 图解：左侧案例展示了 GLM-5V-Turbo 通过网页搜索收集视觉材料，并组织成图文交错的研究报告。重点在于，模型不是把图片当装饰，而是把视觉证据放到论证链条中，与文字解释共同构成结论。

![文档阅读与技术博客生成](figures/doc-read.png)

> 图解：右侧案例展示了从学术论文中自动裁剪视觉元素，并插入到技术博客中的过程。这说明 GLM-5V-Turbo 可以把论文中的图表、结构和文字结论重新组织为更适合阅读的内容形态。

论文把下游内容生成分成三类：

1. **Interleaved Reports** ：图文交错报告，适合对比分析、文献综述。
2. **Deep Research to PPT** ：把收集材料组织成结构化 slide deck。
3. **Document-Style Write-ups** ：生成博客式解读或结构化笔记。

这背后的关键不是“会写长文”，而是保留 **textual conclusions** 和 **visual evidence** 的对应关系。很多研究型任务中，真正有价值的信息往往藏在图、表、版式和截图里，纯文本 pipeline 会直接丢掉这些证据。

---

## 官方 Skills：把模型能力产品化

论文列出了 GLM-5V-Turbo 支持的一组官方 skills，分为 Native、External Tool 和 Specialized 三类。

| Skill | Type | URL |
|---|---|---|
| `PDF-to-Web` | Native | https://clawhub.ai/zai-org/glmv-pdf-to-web |
| `PDF-to-PPT` | Native | https://clawhub.ai/zai-org/glmv-pdf-to-ppt |
| `Web Replication` | Native | https://clawhub.ai/zai-org/glmv-web-replication |
| `PRD-to-App` | Native | https://clawhub.ai/zai-org/glmv-prd-to-app |
| `Stock Analyst` | Native | https://clawhub.ai/zai-org/glmv-stock-analyst |
| `Image Captioning` | External Tool | https://clawhub.ai/JaredforReal/glmv-caption |
| `Visual Grounding` | External Tool | https://clawhub.ai/jaredforreal/glmv-grounding |
| `Doc-based Writing` | External Tool | https://clawhub.ai/jaredforreal/glmv-doc-based-writing |
| `Resume Screening` | External Tool | https://clawhub.ai/JaredforReal/glmv-resume-screen |
| `Prompt Generation` | External Tool | https://clawhub.ai/JaredforReal/glmv-prompt-gen |
| `General OCR` | Specialized | https://clawhub.ai/JaredforReal/glmocr |
| `Table Recognition` | Specialized | https://clawhub.ai/JaredforReal/glmocr-table |
| `Handwriting Recognition` | Specialized | https://clawhub.ai/JaredforReal/glmocr-handwriting |
| `Formula Recognition` | Specialized | https://clawhub.ai/JaredforReal/glmocr-formula |
| `Image Generation` | Specialized | https://clawhub.ai/JaredforReal/glm-image-gen |

这些 skills 的意义在于降低模型进入实际 Agent 系统的成本。比如：

- `PDF-to-Web` 可以把 PDF 转成网页。
- `PDF-to-PPT` 可以把论文或报告转成演示文稿。
- `Web Replication` 可以做网页复刻。
- `PRD-to-App` 可以从产品需求文档生成应用。
- OCR 相关 skills 则适合文档解析、表格识别、公式识别等垂直任务。

---

## Demo 解读：GLM-5V-Turbo 的典型应用场景

### 股票分析

![股票分析案例](figures/stock_analysis.png)

> 图解：该案例展示 GLM-5V-Turbo 结合 OpenClaw 和 `glmv-stock-analyst` skill 进行股票分析。模型会从多个来源收集信息，并组织技术分析、基本面分析、分析师观点和行动建议。这类任务体现了多源信息检索、结构化总结和报告生成能力。

### GUI 探索与网页复刻

![GUI 探索与网页复刻](figures/mmgui_explore.png)

> 图解：图中展示了模型通过 GUI 探索目标网站，收集必要素材，并复刻网页的过程。这里的关键能力包括截图理解、页面导航、元素定位、素材提取和 HTML 生成。相比单张截图还原网页，这种 workflow 更接近真实前端复刻任务。

### PRD 到网站

![PRD 到网站生成](figures/mmprd2code.png)

> 图解：该案例展示根据 PRD 文档生成网站的过程。模型不仅需要理解需求文本，还要结合项目目录内容，规划页面结构和交互，并在工作目录中实现网站。这类任务把产品理解、工程实现和多模态 Agent 执行连接到一起。

### 多模态 UI-to-Code

![电商网站生成案例](figures/mm_ui2code_web.jpg)

> 图解：这张图展示 GLM-5V-Turbo 根据高层设计需求生成完整电商网站。页面包含欢迎页、购物界面、品牌故事页、暗色模式设计、checkout 流程和按钮交互。它体现的不只是视觉还原，还有前端信息架构和交互设计能力。

![移动端 UI 复刻案例](figures/mm_ui2code_mobile.jpg)

> 图解：该案例中，模型根据移动端心情追踪 App 的参考图生成可执行网页代码，并扩展出风格一致的后续页面和交互。这说明模型能从单个视觉样例推断产品设计语言，而不是机械复制一屏 UI。

![Agentic UI-to-Code](figures/mm_agentic_ui2code.png)

> 图解：这张图强调 agentic UI recreation。模型需要根据截图还原网页，并自动检索截图中出现的图像素材。任务流程包含视觉理解、素材检索和代码生成，是多模态 Agent 能力的典型组合。

### 论文网站与 PPT 生成

![论文网站生成](figures/mmwebgen.jpg)

> 图解：该案例展示模型根据论文生成介绍网站。输出不是简单摘要，而是把论文动机、方法、系统设计和实验结果组织成网页信息结构，并穿插图片说明。

![论文 PPT 生成](figures/mmpptgen.jpg)

> 图解：该案例展示从论文自动生成 PowerPoint。模型需要判断每页 slide 应该承载什么内容，如何分配图文比例，以及如何把方法、架构和结果转换为演示友好的结构。

### 图像素材收集与图文报告

![图像素材收集](figures/image_collection.jpg)

> 图解：这张图展示了为 Apple Wearables 专题报告收集图像素材的过程。重点在于模型不仅找图，还要关注来源权威性、图片质量和图文组织方式。

### 文档写作

![北京旅行指南](figures/travel_guide.jpg)

> 图解：左图是 103 页中文北京旅行指南。模型需要读取长文档，理解其中的景点、路线和文化信息，再面向外国游客筛选“必去景点”。

![旅行解说词生成](figures/commentary.png)

> 图解：右图是基于旅行指南生成的景点解说内容。这里体现的是 document-grounded writing：输出内容必须受源文档约束，而不是自由编造。

### OCR 与文档解析

![多语言 OCR 原图](figures/multilingual.png)

> 图解：原图包含多语言文字，适合测试模型对不同文字系统的识别能力。多语言 OCR 对真实文档处理很重要，因为网页截图、海报和报告中经常混合多种语言。

![多语言 OCR 结果](figures/multilingual_ocr.png)

> 图解：识别结果不仅输出文字，还标注语言类型。这类能力可以作为后续翻译、检索和结构化解析的基础。

![物理教材页面](figures/physics.png)

> 图解：这是一页物理教材，包含正文、公式、图表和表格。文档解析难点在于不同元素的版式关系，而不是单纯 OCR 文本。

![物理教材转写结果](figures/physics_ocr.png)

> 图解：转写结果以 Markdown 形式保留文本、表格和图示结构。这说明模型具备将复杂页面转换为结构化内容的能力。

### Grounding 与视觉定位

![篮球视频目标追踪 1](figures/cases_grounding/mot-basket-1.jpg)

> 图解：该图是视频目标追踪任务中的一帧。模型需要识别每秒画面中所有打篮球的人，并用结构化 JSON 输出对应对象。

![篮球视频目标追踪 2](figures/cases_grounding/mot-basket-2.jpg)

> 图解：连续帧追踪的关键不是单帧检测，而是跨时间保持对象身份一致。这对视频 Agent 和监控分析类任务很重要。

![篮球视频目标追踪 3](figures/cases_grounding/mot-basket-3.jpg)

> 图解：这类场景考察模型在运动、遮挡、姿态变化下的视觉 tracking 能力。

![视频目标追踪案例 1](figures/cases_grounding/mot-thief-1.jpg)

> 图解：该案例要求根据“person committing crime”这样的语义描述定位视频中的对象。模型需要把语言描述和视觉对象对齐。

![视频目标追踪案例 2](figures/cases_grounding/mot-thief-4.jpg)

> 图解：随着视频时间变化，模型需要持续追踪同一目标，并输出 bounding box 和全局一致标签。

![视频目标追踪案例 3](figures/cases_grounding/mot-thief-6.jpg)

> 图解：这种能力对 GUI 操作之外的真实视频理解场景同样重要，尤其是多步观察和事件分析。

![人物识别与 GPU 电路板识别](figures/cases_grounding/recognition-1927-cn.jpg)

> 图解：左图展示人物识别和框选能力。模型需要识别图中所有人物并给出名字，这要求视觉识别、检索和 grounding 结合。

![GPU 电路板组件识别](figures/cases_grounding/recognition-b200.jpg)

> 图解：右图展示 GPU 电路板组件识别和参数对比。模型需要搜索图像、框选组件、识别名称，并与 H100 做参数比较，属于视觉搜索和专业知识结合任务。

![学生手写答案定位](figures/cases_grounding/stem-ground-handwrite.jpg)

> 图解：该图展示教育场景中的 handwriting grounding。模型需要找到每个填空处学生手写答案的位置，这要求细粒度文字区域定位。

![写作错误定位](figures/cases_grounding/stem-correct-writing.png)

> 图解：该图展示写作错误检测和定位。模型不仅要识别错误词，还要在图像中定位对应区域，适合教育批改场景。

![家具 3D Grounding](figures/cases_grounding/3d-furnitures.png)

> 图解：这张图展示 3D grounding。模型输出的不只是 2D bounding box，而是包含中心点、尺寸和旋转角的 3D bounding box，更适合机器人、空间理解和室内场景建模。

![盆栽 3D Grounding](figures/cases_grounding/3d-zhipu.jpg)

> 图解：该案例要求定位第一个盆栽的 3D bounding box。这里考察的是模型从单张图或视觉场景中推断空间位置和物体尺度的能力。

### 空间推理

![手指计数与标注](figures/fingers_marks.png)

> 图解：该任务要求数出图中的手指数量，并用 `[[x,y]]` 格式标注位置。它考察的是细粒度空间定位和计数能力，这类基础感知能力会直接影响更复杂的 GUI 操作和视觉推理。

---

## Benchmark 结果：多模态能力与 Coding 能力并进

论文把评测分为四类：

1. **Multimodal Coding**
  - Design2Code
  - Flame-VLM-Code
  - Vision2Web

2. **Multimodal Tool-use**
  - ImageMining
  - BrowseComp-VL
  - MMSearch
  - MMSearch-Plus
  - SimpleVQA
  - Facts
  - V*

3. **GUI Agent**
  - OSWorld
  - AndroidWorld
  - WebVoyager

4. **Text-only Coding and Claw**
  - CC-Bench-V2
  - PinchBench
  - ClawEval
  - ZClawBench

![多模态 Coding、Tool-use 与 GUI Agent 评测](figures/glm-5v-turbo-benchmark1.png)

> 图解：这张图汇总了 GLM-5V-Turbo 在多模态编码、工具使用和 GUI Agent benchmark 上的表现。可以看到，模型在 UI-to-code、视觉网站开发、多模态搜索和 GUI 操作等任务上都有较强结果，说明视觉理解可以迁移到具体执行任务中。

![Text Coding 与 Claw Agent 评测](figures/glm-5v-turbo-benchmark2.png)

> 图解：这张图展示 GLM-5V-Turbo 在 text-only coding 和 Claw agent benchmark 上的表现。重点是模型在强化视觉能力后，仍然保持了较强代码能力，这对真实 Agent 系统非常重要，因为任务通常需要视觉判断和代码执行协同完成。

论文中的一个关键判断是：

> GLM-5V-Turbo 的多模态能力不是孤立 benchmark 上的提升，而是能够迁移到框架化的端到端 Agent 执行中。

这也是为什么它特别强调 Claw、Claude Code、AutoClaw 这类系统环境。

---

## 三个设计经验：开发多模态 Agent 的方法论

论文在 Design Lenses 部分总结了三个经验，这部分非常值得单独看。

### Lens 1：感知仍然是高层多模态能力的基础

现在很多研究更关注 planning、reasoning、reflection，但论文认为：多模态任务里，很多看似高层的错误，其实起点是模型没看清。

比如：

- 误读按钮文字。
- 混淆界面元素。
- 看错图表趋势。
- 忽略表格中的小字。
- 错判布局和空间关系。

这些感知错误会一路传播到规划和执行阶段。

论文提出一个有意思的观察： **multimodal coding 和 grounding 是很好的感知训练代理任务。**

原因是 UI-to-code、SVG coding、frontend recreation 强迫模型学习：

- 布局结构
- 相对位置
- 局部细节
- 视觉层级
- 元素关系

此外，论文还提到在 GUI-agent instruction tuning 中加入 critic data，让模型批判自己的感知错误，比如误读界面细节、误认目标元素、做出错误下一步动作。这可以减少视觉 hallucination。

### Lens 2：Agent 能力适合通过层级优化构建

长程 Agent 任务很难直接端到端优化，因为：

- 环境搭建成本高。
- 高质量轨迹稀缺。
- 验证困难。
- 解法路径不唯一。
- 结果强依赖环境状态。

因此，论文主张采用 **hierarchical optimization** ，不要只压在高层长程任务上。

以 GUI Agent 为例，可以构造多层任务：

| 层级 | 任务 |
|---|---|
| 低层 | 元素感知 |
| 中低层 | GUI grounding |
| 中层 | 单步动作预测 |
| 高层 | 轨迹级动作预测 |

这样做有两个好处：

1. 低层任务更容易构造、标注和验证。
2. 当底层能力还不稳时，直接训练长程任务收益有限，还可能导致训练不稳定。

这个思路对复现很有启发：如果想训练一个网页 Agent，不一定一上来就做完整浏览任务，可以先训练截图元素识别、按钮定位、单步点击预测，再逐步进入多步执行。

### Lens 3：端到端任务的关键是明确规格、可靠验证和可控评测

论文认为，长程 Agent 任务最难的不是“更长”，而是“可评测”。

很多真实任务存在：

- 目标描述不完整。
- 执行边界模糊。
- 中间路径多样。
- 最终结果难以稳定比较。
- 失败归因困难。

因此，一个好的端到端 Agent benchmark 必须同时设计：

1. **清晰任务规格** ：不仅是 prompt，还可以包括 PRD、mockup、参考网页、素材资源。
2. **可靠 outcome verification** ：最终结果要能被稳定判断。
3. **受控 evaluation procedure** ：评测流程要尽量可复现。

论文提到 Vision2Web 就是这个思路的实例。它不是简单让模型“做一个网站”，而是用 PRD、mockups、reference pages 和 assets 共同定义任务，再通过 workflow-based verification 评估执行过程。

这对未来 Agent benchmark 非常关键：如果评测本身不稳定，RL 和系统优化就很难获得可靠信号。

---

## Remaining Challenges：多模态 Agent 还没解决什么？

论文最后指出了三个核心挑战。

### 挑战一：如何让更好的 Agentic Strategies 自发涌现？

当前 Agent 训练仍然依赖人工构造或强过滤的 cold-start trajectories。这样做能初始化模型，但也会限制策略空间。

模型可能越来越擅长执行熟悉路径，却很难发现真正更优的新策略。

论文观察到，提高 cold-start 阶段的轨迹多样性，可以帮助 RL 发现附近更好的策略变体。但这还不够。更根本的问题是：如何让模型自己发现新的推理和行动策略。

更进一步，未来模型还需要发现更复杂的组织形式，例如：

- sub-agent decomposition
- multi-agent collaboration
- hierarchical decision structures

这意味着 Agent 能力的发展可能不只是单模型能力提升，还涉及组织结构和任务分解策略的涌现。

### 挑战二：多模态长程上下文管理仍是瓶颈

图像和视频比文本更消耗 context budget。很多系统在上下文变长后会丢弃早期视觉观察，但这些视觉信息可能后面还会用到。

文本场景中，可以对历史对话做 summary 或 compaction。但多模态场景更难，因为要保留的不只是语义，还有：

- 布局细节
- 空间关系
- 图像局部信息
- 视频中的时间变化
- 曾经出现但暂时不显眼的视觉线索

论文指出，当前大多数 memory mechanism 仍然是 text-centric，更擅长压缩“说过什么”，不擅长压缩“看到了什么”。

因此，长期多模态 Agent 需要更 multimodal-native 的记忆机制，而不是简单把文本记忆方案套过来。

### 挑战三：模型和 Harness 共同决定能力边界

Agent 系统中，模型能力边界不再由模型单独决定，而是由模型和 harness 一起塑造。

这里的 harness 可以理解为模型外部的执行系统，包括：

- task decomposition
- tool-use policy
- memory mechanism
- verifier
- workflow controller
- browser automation
- code execution environment

同一个模型，在不同 harness 下可能表现差异很大。反过来，看起来像模型能力不足的问题，也可能只是 harness 设计不好。

更复杂的是，harness 的价值还依赖模型所处能力阶段。某种分解策略在弱模型上没用，在强模型上可能变得关键。

所以论文认为，Agentic model development 不能再被理解为单纯的模型提升，而是模型和 harness 的共同优化。

---

## 我的理解：这篇论文真正想表达什么？

如果只看 benchmark，GLM-5V-Turbo 是一个强多模态模型。但从技术报告的叙事看，它真正想强调的是 **范式变化** 。

过去的 VLM 更像是：

> 语言模型 + 图片输入接口

而 GLM-5V-Turbo 想走向：

> 以多模态感知为基础的 Agentic foundation model

这两者区别很大。

第一种模型主要回答“这张图里有什么”。第二种模型要回答“我看到这个环境后，下一步应该做什么，并如何验证我做对了”。

因此，论文的很多设计都围绕这个目标展开：

- CogViT 强化细粒度视觉和空间理解。
- MMTP 在多模态训练中平衡效果和工程效率。
- Joint RL 覆盖感知、推理、工具、GUI 和代码任务。
- Toolchain 支持搜索、裁剪、标注、网页阅读和内容生成。
- Claude Code / AutoClaw 集成让模型进入真实执行环境。
- ImageMining 评估“围绕图像进行深度搜索”的能力。
- Skills 把能力封装成可复用工作流。

换句话说，GLM-5V-Turbo 的技术路线不是把视觉能力当作加分项，而是把视觉作为 Agent 工作流的基础输入层。

---

## 适合复现和借鉴的技术点

如果我们想从这篇论文中提炼对工程实践最有用的部分，我认为有下面几点。

### 1. 多模态 Agent 要优先补感知短板

很多 Agent 失败不是不会规划，而是第一步看错了。尤其在 GUI、网页、表格、文档任务中，视觉细节错误会直接导致后续工具调用错误。

因此，训练或评测时应该单独拆出：

- element detection
- grounding
- OCR
- layout understanding
- chart understanding
- spatial reasoning

这些低层任务不是“边角料”，而是上层 Agent 能力的地基。

### 2. UI-to-code 是很强的多模态训练任务

UI-to-code 同时要求模型理解视觉布局、组件语义、样式结构和代码生成。它天然连接视觉和工程执行，比普通 image caption 更接近 Agent 应用。

论文也观察到，frontend 或 SVG coding 这类任务能帮助模型学习结构化视觉感知。

### 3. 工具链设计会显著改变模型能力边界

一个模型是否“会做任务”，很大程度取决于它有哪些工具，以及工具是否容易被模型规划和调用。

比如 ImageMining 中，如果没有 crop、search by image、similar image search、OCR search 等工具，模型很难完成深度视觉搜索。

### 4. Benchmark 必须重视验证机制

对于端到端 Agent，任务是否真实不够，关键是结果是否能稳定验证。否则训练信号和评测结果都会不可靠。

Vision2Web 这种 workflow-based verification 思路，比单纯让模型自由生成网页更适合用于优化。

### 5. 多模态记忆是下一阶段难点

长程 Agent 中，如何压缩和恢复视觉上下文会变得越来越重要。未来可能需要：

- 可检索视觉记忆
- 局部区域级记忆
- 视频状态摘要
- layout-aware memory
- multimodal episodic memory

这部分目前还没有成熟范式。

---

## 总结

GLM-5V-Turbo 这篇报告的主线非常清晰：未来 Agent 需要原生多模态能力，而不是把视觉当作语言模型的附属输入。

它从模型、训练、RL 系统、工具链、外部框架、benchmark 和 skills 生态多个层面推进这一目标。核心结论可以压缩成三句话：

1. **感知是多模态 Agent 的基础能力** ，很多高层失败源于低层看错。
2. **Agent 能力需要层级优化和工具链协同** ，不能只靠端到端长程任务硬训。
3. **模型和 harness 共同决定系统能力边界** ，未来 Agent 研发会越来越像模型、工具、执行框架和验证系统的联合工程。

> 本文参考自 [GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents](https://arxiv.org/abs/2604.26752)