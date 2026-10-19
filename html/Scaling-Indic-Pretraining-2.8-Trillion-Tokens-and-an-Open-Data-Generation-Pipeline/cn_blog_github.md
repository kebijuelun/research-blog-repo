# Nemotron-Indic：2.8 万亿 Token 数据管线

印度语言模型要稳定地解释数学、处理代码和回答专业问题，一个直接瓶颈是训练文本不足。AI4Bharat 负责人 Mitesh Khapra 将这个问题概括为：“我们没有足够的印度语言数据。”NVIDIA 与 IIT Madras 的 AI4Bharat 因此宣布将推出 Nemotron-pretraining-Indic-v1：约 2.8 万亿 Token，覆盖 22 种印度语言，并同步开放用于构建它的 GPU 加速数据生成管线。

这套方案并非只做批量翻译，而是把文档级翻译、质量评分、定向修复和文字转换串成可追溯的流程。在与该语料相关的 Nemotron 3.5 Super VL Indic 继续预训练（CPT）对照中，5 项印度语言知识与推理基准提高了 4.98～9.82 个百分点。不过，这一阶段同时使用合成和天然印度语言文本，不能把全部提升单独归因于合成语料。

## 2.8 万亿 Token 里到底有什么

这 22 种语言是印度宪法列出的语言。以它们为第一语言的印度人占 97%，按 2011 年人口普查计算接近 12 亿人。AI4Bharat 此前已发布覆盖这 22 种语言的开放预训练语料 Sangraha。这次合作进一步扩大了可公开使用的合成印度语言训练数据，并把数学、代码、问答等经过筛选的英文内容带入这些语言。

语料的“原生文字”与“罗马化”是两个不同维度。原生文字（native script）指使用语言自身的文字系统，文本仍由英文翻译而来；罗马化（romanized）指用拉丁字母书写印度语言，并非把文本翻译成英语。22 种语言都有原生文字数据，其中 12 种另有罗马化数据，合计形成 34 种语言—文字配置。

![Nemotron-pretraining-Indic 语料规模与语言配置](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Scaling-Indic-Pretraining-2.8-Trillion-Tokens-and-an-Open-Data-Generation-Pipeline/images/VmVFMKjOsoNkEKfQ7cRnh.png)

> 图解：按 Gemma-4 tokenizer 统计，语料约为 2.83T 个目标语言 Token。其中，12 种双文字语言贡献 2,273.8B 个原生文字 Token，另外 10 种语言贡献 327.2B 个原生文字 Token，12 种语言的罗马化子集贡献 224.6B 个 Token；原生文字合计 2,601.0B。图中还列出 2.44B 条文档记录，但记录包括同一来源的翻译和文字变体，不等于独立英文源文档的数量。

两组语言配置如下：

- **原生文字 + 罗马化（12 种）**：孟加拉语（Bengali）、古吉拉特语（Gujarati）、印地语（Hindi）、卡纳达语（Kannada）、马拉雅拉姆语（Malayalam）、马拉地语（Marathi）、尼泊尔语（Nepali）、奥里亚语（Odia）、旁遮普语（Punjabi）、泰米尔语（Tamil）、泰卢固语（Telugu）和乌尔都语（Urdu）。
- **仅原生文字（10 种）**：阿萨姆语（Assamese）、博多语（Bodo）、多格拉语（Dogri）、克什米尔语（Kashmiri）、孔卡尼语（Konkani）、迈蒂利语（Maithili）、曼尼普尔语（Manipuri）、梵语（Sanskrit）、桑塔利语（Santali）和信德语（Sindhi）。

罗马化数据并不是为每个原生文字样本都生成一份，而是从这 12 种语言已验收的原生文字译文中选取子集转换而来。这一区别会影响训练配比：不同文字配置提供了不同的书写形式暴露，却可能承载相同的底层内容。Token 数还取决于所用 tokenizer；这里统计的是保留下来的语料库存，包括修复后被接受的文本，不是某一次训练实际消耗的 Token 预算。

## 关键不只是翻译，而是可追溯的生成流程

语料的源文档来自 Nemotron-CC-v2 的 15 个子集，覆盖网页文本、问答、数学、代码和其他专业内容。团队使用 NVIDIA 的开源合成数据框架 NeMo Data Designer 执行翻译、质量检查和定向修复，并全程保留源文档标识。每条输出因而可以追溯到原始文档，便于审计、复现和后续构建。

![十二种语言的印度语言合成数据生成流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Scaling-Indic-Pretraining-2.8-Trillion-Tokens-and-an-Open-Data-Generation-Pipeline/images/PCtm5uAGrdrp8kgHXb6y3.png)

> 图解：十二种语言的流程由 NeMo Data Designer 和 Gemma-4-31B-it 实现。英文文档先经过文档翻译，再接受 0～100 分的质量检查；低于 75 分的候选文本进入定向重写，修复后再次检查，仍不达标则丢弃。通过检查的原生文字文本保留源 ID、语言、文字、正文和质量元数据，其中一部分再经过罗马化及同样的检查、重写、复查流程，进入罗马化语料。人工抽样检查和故意注入错误的测试用于改进提示词与质量检查。

### 十二种语言：先修翻译，再转写

这 12 种语言采用完整文档翻译，让模型利用段落、表格和公式之间的上下文。图中流程还显示，输入选取少于 4,096 个 Gemma-4 Token 的文档，不拆分文档，翻译时保留源元数据、事实和格式。Gemma-4-31B-it 是在完整文档上比较翻译质量与技术内容保留情况后选出的模型；提示词会区分需要翻译的正文与需要保留的代码、公式、URL 和格式。

质量环节可以概括为“翻译—找错—按错修复—再验收”，而不是简单地一次性过滤：

1. **文档翻译**：生成原生文字候选文本，同时保留事实和格式。
2. **质量检查**：同一个 Gemma-4-31B-it 将译文与英文源文档对照，参考多维质量指标（Multidimensional Quality Metrics，MQM）设计评分标准，从含义、流畅度、术语、风格和格式等方面打分并列出错误。MQM 是一套按错误类型和严重程度分类的开放翻译质量框架。达到 75 分的译文直接保留，低于 75 分的进入下一步。
3. **定向重写**：模型收到源文、译文和错误反馈，对低分候选文本进行修复，重点修正含义和语言问题。
4. **最终检查**：修复文本再次评分，达到 75 分才保留，否则丢弃。这里的 75 分是评分标准中的分数，不是“75% 准确率”。

这一修复环节在原生文字和罗马化数据中共挽回 438.4B 个 Token，且已经计入前文的语料总量。团队还结合人工抽样评审和故意引入错误的测试，检查遗漏、残留英语等问题，并据此改进提示词和检查机制。

罗马化被拆成独立步骤：先从已通过检查的原生文字译文中选取子集，再转换成拉丁字母，并对照原生文字输入检查结果。团队的初步人工比较显示，相比直接从英文生成罗马化文本，这种两步法产生的幻觉性增译和残留英语更少。不过，这一结论来自初步比较，并不表示所有语言和领域都必然如此。

### 另外十种语言：针对长文和文字系统改流程

其余 10 种语言没有简单复用同一条链路，因为比较中暴露出长文档质量和文字稳定性问题。团队改用 Bodhan AI 的 Indic Translate 做文档级翻译。Bodhan AI 是 IIT Madras 孵化的非营利教育 AI 中心，Indic Translate 支持英文与 22 种印度语言之间的翻译。随后仍由 Gemma-4-31B-it 对照英文源文评分，保留至少 75 分的结果，但不进行基于反馈的重写，最终只生成原生文字数据。

其中 8 种语言保留整篇文档翻译。桑塔利语（Santali）的长文档采用相互重叠的窗口翻译后再重组；曼尼普尔语（Manipuri）则先译为孟加拉文字，再转换为其原生文字 Meitei Mayek。这些分支说明，所谓“统一管线”实际上仍要针对语言、文档长度和文字系统调整工程路径。

## 继续预训练带来了多少提升

数据生成流程解决了供给和质量控制问题，下一步是验证加入训练后模型是否真正受益。Nemotron 3.5 Super VL 技术报告比较了 Indic CPT 前后的 checkpoint。该阶段配置的训练预算为 100B 个 Token，其中 10%（10B）分配给合成与天然印度语言文本，合成部分来自该语料的十二语言子集；其余 90% 保持此前训练混合数据的相对比例。

![Nemotron 3.5 Super VL 在 Indic CPT 前后的基准对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Scaling-Indic-Pretraining-2.8-Trillion-Tokens-and-an-Open-Data-Generation-Pipeline/images/ZPR2A2wDoNW6OVb4-GX63.png)

> 图解：灰色为 Indic CPT 前的蒸馏 checkpoint，绿色为 CPT 后的 checkpoint。5 项指标均上升，提升幅度从 4.98 到 9.82 个百分点（pp）。这些基准覆盖的语言数并不相同；Indic GSM8K 的 Odia 因评测超出 serving 上下文长度限制而未纳入。

| 基准（语言覆盖） | CPT 前 | CPT 后 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| MILU（10 种） | 72.48 | 77.90 | +5.42 |
| Indic MMLU-Pro（12 种） | 51.83 | 56.81 | +4.98 |
| Indic ARC-Challenge（12 种） | 84.17 | 92.83 | +8.66 |
| Indic ARC-Easy（12 种） | 90.73 | 96.77 | +6.04 |
| Indic GSM8K（strict，11 种） | 72.67 | 82.49 | +9.82 |

这组对照显示，加入合成与天然印度语言文本的 CPT 阶段后，知识和推理基准均有提升。但它支持的是整个训练阶段的效果，而不是合成语料单独的因果贡献；比较对象也是训练阶段 checkpoint，不是两个最终发布版本。对于其他模型和语言，怎样分配天然数据、合成数据与不同文字配置，仍需要单独研究。

配套 Indic 报告进一步考察了源数据混合、合成数据池规模和文字覆盖，但相关受控研究使用的是十二语言组的首轮通过数据。把修复挽回的数据和另外 10 种语言也纳入完整语料评估，仍是下一步工作。

## 发布模型对比：任务涨分，指令遵循略降

CPT 对照只覆盖一个训练阶段。在完整发布模型上，Nemotron 3.5 Super VL 还与上一代 Nemotron 3 Super 做了比较。这些结果包含 Indic CPT 和后训练在内的完整训练流程，因此更不能直接当作语料的消融实验。

![Nemotron Super 系列在印度语言任务和语言质量上的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Scaling-Indic-Pretraining-2.8-Trillion-Tokens-and-an-Open-Data-Generation-Pipeline/images/YQO_-yZWrhhQgueXuxzuM.png)

> 图解：左图展示 6 项任务基准，绿色柱代表的 Nemotron 3.5 Super 在其中 5 项上高于 Nemotron 3 Super，IFEval 1.5 下降 2.04 个百分点。除 WMT24++ 使用 chrF 点外，标签中的变化为百分点。右图是初步 IndiVibe 条件偏好分数，不是直接的头对头胜率；有用性、流畅度、语言与文字忠实度、恰当详略程度四个维度均上升。

| 任务基准 | Nemotron 3.5 Super VL 相对 Nemotron 3 Super 的变化 |
| --- | ---: |
| MILU 1.5 | +7.60 个百分点 |
| GPQA | +11.83 个百分点 |
| MATH500 | +4.58 个百分点 |
| MBPP | +3.28 个百分点 |
| WMT24++ | +13.69 chrF 点 |
| IFEval 1.5 | −2.04 个百分点 |

IndiVibe 是 Sarvam AI 面向印度语言助手式提示词的开放基准。图中四个语言质量维度的变化分别为：有用性 +23.7、流畅度 +25.6、语言与文字忠实度 +22.3、恰当详略程度 +21.0 个百分点。任务基准回答“模型会不会做”，IndiVibe 则补充考察“模型能否用合适的印度语言表达”。IFEval 1.5 的小幅下降也提醒读者，发布模型的能力提升并非每个维度都同步增加。

## 从加载语料到改造生成配方

### 先流式检查，再决定训练配比

原文建议先按目标语言和文字选择配置，检查样本，并在训练前阅读数据集卡片和许可证。配置名由语言、文字和地区标签组成，例如 `hi-Deva-IN` 表示使用天城文的印地语，`hi-Latn-IN` 表示罗马化印地语。下面的示例从数据集卡片取得仓库 ID，发现可用 split，并以流式方式读取一条样本：

```python
from datasets import get_dataset_split_names, load_dataset

dataset_id = input("Repository ID from the dataset card: ").strip()
config = "hi-Deva-IN"

splits = get_dataset_split_names(dataset_id, config_name=config)
print("Available splits:", splits)

data = load_dataset(
    dataset_id, name=config, split=splits[0], streaming=True
)
print(next(iter(data)))
```

Hugging Face Datasets 的 streaming 模式允许先检查记录，而不必下载完整语料。原文发布时，语料仍处于即将开放的状态；数据可用后，实验中应固定 dataset revision，以便复现。

第一轮训练研究至少要处理好三件事：

- 在固定 Token 预算下比较不同混合配比，分别记录天然／合成数据、原生文字／罗马化数据的比例。
- 同时跟踪各语言结果、英文能力保持情况，以及语言和文字质量，而不只看一个汇总平均分。
- 划分留出集（held-out set）时考虑同一源文档跨语言、跨文字派生的记录，避免相关样本同时进入训练与评估而造成泄漏。

若要构建更完整的训练方案，原文还建议将该语料与 Nemotron 的 Chat & Instruction 或 Post-Training Blends 等后训练数据集合搭配。面向印度场景的合成数据，则可以用 Nemotron-Personas-India 作为 NeMo Data Designer 的种子；这些英文和印地语合成 persona 基于印度真实世界分布构建。

### 用自己的源文档改造生成配方

开源的 Nemotron Bharat 翻译插件为 NeMo Data Designer 提供翻译与评估提示词、可选的反馈式重写、文档窗口化和文字重映射支持。使用者可以通过配置参考选择模型与语言、按领域调整提示词，并设置重写参数；配套开发者说明则解释具体流程和工程取舍。

更稳妥的落地顺序是先用一小批自有源文档试跑，检查翻译质量以及代码、数学公式和格式的保留情况，再请母语评审帮助调整提示词和检查规则，确认稳定后扩大规模。还要把“生成候选”和“语料验收”分开：插件返回候选文本与分数，最终哪些记录进入训练集，应由自己的过滤策略决定。

## 这套方案的价值与边界

翻译能扩大英文技术资料的可及性，却不能替代原本就用印度语言写成的文本，也不能保证覆盖地方知识与文化语境。因此，这套语料更适合作为天然印度语言文本的补充，而不是取而代之；实际训练仍需结合来源合适的天然语料，并针对目标用途评估混合结果。

质量控制也有明确边界。评分主要依靠自动化，人工抽样评审在不同语言上的覆盖并不一致；十二语言流程中，翻译模型同时担任评审，可能漏掉自身倾向产生的错误。即使文本通过 75 分门槛，也仍可能含有错误，十语言流程尤其需要更广泛的母语者验证。此外，源 ID 检查只能去除处理过程中产生的重复，并不等于对相似内容做语义去重。

回到开头“印度语言数据不够”的问题，这次发布至少提供了四层可复用的价值：

- **数据供给**：约 2.8 万亿 Token，覆盖 22 种语言和 34 种语言—文字配置。
- **生成方法**：翻译、评分、定向修复和罗马化都有明确流程，且保留源文档追踪信息。
- **训练证据**：Indic CPT 后 5 项基准全部上升，但证据支持的是合成与天然数据的阶段级混合配方。
- **使用路径**：既可以直接流式检查语料，也可以用 Nemotron Bharat 插件和自有文档重建生成流程。

需要构建印度语言模型的团队值得关注这套资源，需要扩充领域语料的团队则更应关注开放的生成配方。可执行的起点是选定少量目标语言和文字配置，先检查样本与许可证，再在固定 Token 预算下比较混合配比；它能缓解技术文本的规模问题，但地方知识、文化覆盖和更充分的母语者验证仍未被这项发布自动解决。AI4Bharat 在 NVIDIA 支持下建设的 Indic-LLM-Eval-Suite，也正是为持续评估任务能力与语言质量而设。

> 本文参考自 [Scaling Indic Pretraining: 2.8 Trillion Tokens and an Open Data Generation Pipeline](https://huggingface.co/blog/nvidia/scaling-indic-pretraining)