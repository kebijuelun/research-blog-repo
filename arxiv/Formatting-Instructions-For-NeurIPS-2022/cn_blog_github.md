# 一句话让大模型"开窍"：解读《Large Language Models are Zero-Shot Reasoners》

这篇 NeurIPS 2022 的论文解决了一个看似简单却影响深远的问题：不依赖任何人工精心构造的示例（few-shot exemplars），能否激发出大语言模型（LLM）的多步推理能力？作者给出的答案简单到令人惊讶——只需要在每个问题后面加一句 **"Let's think step by step"** （让我们一步一步思考）。就这一句话，让 text-davinci-002 在 MultiArith 算术推理基准上的准确率从 17.7% 暴涨到 78.7%，在 GSM8K 上从 10.4% 提升到 40.7%。这个发现直接重塑了此后整个 Prompt Engineering 领域的研究范式。

## 背景：从"举例子"到"说句话"

要理解这篇论文的贡献，先得搞清楚它所处的技术坐标系。

2020 年 GPT-3 横空出世之后，"预训练 + Prompt"逐渐取代"预训练 + 微调"成为使用 LLM 的主流范式。但很快大家发现一个问题：LLM 在那些需要直觉反应的单步任务（所谓 **System-1** 任务，比如翻译、摘要）上表现很好，可一旦遇到需要慢思考、多步推理的 **System-2** 任务（比如数学应用题、逻辑推理），即使模型规模堆到 100B 以上参数，性能曲线依然是平的——模型变大并没有变聪明。

2022 年初，Google 的 Wei 等人提出了 **Chain of Thought** （CoT，思维链）Prompting：在 few-shot 示例中不只给"问题 + 答案"，而是给"问题 + 一步一步的推理过程 + 答案"。模型会模仿示例，先输出推理路径再给答案，效果立竿见影。例如配合 540B 参数的 PaLM，GSM8K 的准确率从 17.9% 跳到 58.1%。

但 Few-shot-CoT 有两个不太体面的软肋：

- **每个任务都得手工写推理示例**。这不是普通的"问题-答案"对，而是要人工编写"问题-推理过程-答案"三元组，费时费力，面对实际场景中无穷无尽的新任务时根本来不及。
- **示例会注入人类偏见**。你写的推理方式是"你认为"模型应该怎么想，这未必是模型自己最擅长的思考方式，可能反而限制了模型的内在能力。

![四种 prompting 方式的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Formatting-Instructions-For-NeurIPS-2022/conceptual_differences.png)

> 图解：这张图对比了四种 prompting 范式。(a) 标准 Few-shot：给几个"问题-答案"示例；(b) Few-shot-CoT：给几个带逐步推理过程的示例（注意每个任务都要单独准备一套）；(c) 标准 Zero-shot：什么都不给，直接问；(d) 本文的 Zero-shot-CoT：不给任何示例，只在问题后加上固定的 "Let's think step by step"。蓝色文字是模型生成的推理链。关键区别在于：Few-shot-CoT 的示例是 **per-task** （每任务定制）的，而 Zero-shot-CoT 的这句咒语是 **task-agnostic** （跨任务通用）的——算术、符号、常识、逻辑推理全都用同一句。

于是一个自然的问题浮出水面：CoT 的成功，到底是因为"模型从示例中学会了推理"，还是因为"模型本来就会推理，只是需要被点一下"？如果是后者，那辛辛苦苦写示例可能完全是多余的。这正是本文要验证的假设。

## 核心方法：Zero-shot-CoT 与两阶段 Prompting

### 一句话的核心思想

方法本身简单到一句话能说完：在问题后面加一句触发语（trigger sentence），比如 "Let's think step by step"，引导模型先输出推理过程，再给答案。

形式化地说，给定问题 $\mathbf{x}$，先用模板把它包装成 prompt $\mathbf{x}^{\prime}$：

$$
\mathbf{x}^{\prime} = \text{``Q: [X]. A: [T]''}
$$

其中 $[\text{X}]$ 是问题占位符，$[\text{T}]$ 是触发语。比如用 "Let's think step by step" 时，prompt 就是 "Q: [问题]. A: Let's think step by step."

**这个设计的聪明之处在于**：它没有描述任务本身（"这是一道数学题"），而是描述了完成任务所需要的 **认知过程** （"一步步想"）。任务千差万别，但"分步推理"这个能力是所有 System-2 任务共享的底层能力——这正是它能一句咒语通吃所有推理任务的根本原因。

### 为什么需要两阶段？

概念虽简单，实现上却要用 prompting 两次，这是很多转述文章会忽略的工程细节。

![Zero-shot-CoT 完整流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Formatting-Instructions-For-NeurIPS-2022/fig_overview_2.png)

> 图解：Zero-shot-CoT 的完整流水线。第一阶段（左）用"推理提取" prompt（Q: [X]. A: Let's think step by step.）让模型生成完整的推理路径 $\mathbf{z}$；第二阶段（右）把 "原始 prompt + 推理路径 + 答案触发语" 拼接起来再次喂给模型，提取出格式规范的最终答案。

**第一阶段：推理提取（Reasoning Extraction）**。把包装好的 $\mathbf{x}^{\prime}$ 喂给模型，用贪心解码（greedy decoding）生成推理文本 $\mathbf{z}$。

**第二阶段：答案提取（Answer Extraction）**。把三样东西拼起来——$\mathbf{x}^{\prime}$、刚生成的 $\mathbf{z}$、以及一个答案触发语 $[\text{A}]$——再喂给模型一次。答案触发语随答案格式而变：

- 数学题（数值答案）："Therefore, the answer (arabic numerals) is"
- 选择题："Therefore, among A through E, the answer is"
- 是非题："Therefore, the answer (Yes or No) is"

为什么 Few-shot-CoT 不需要这一步？因为它的示例本身就教模型在推理结尾输出 "The answer is X" 的格式，答案的位置是确定的。而 Zero-shot-CoT 没有示例约束，模型推理完后不保证乖乖报答案，所以需要第二次 prompt 来"收口"。作者把这个第二阶段称为 **self-augmented prompt** ——因为 prompt 里混入了模型自己刚生成的文本。

**博主点评**：用"少一次人工工程，多一次模型调用"来交换任务无关性，这笔买卖非常划算。人工写示例的成本是随任务数线性增长的，而多一次推理调用的成本是常数级的。

最后是 **Answer Cleansing** （答案清洗）：从输出文本中抽取第一个符合格式的内容作为最终预测。比如数值任务提取第一个数字（"probably 375 and 376" 取 375），选择题提取第一个大写字母。这个细节虽然琐碎，却是保证评测可复现的关键——Zero-shot 设定下贪心解码是确定性的，整个实验完全可复现。

## 实验设置：17 个模型 × 12 个数据集的饱和式验证

论文的实验覆盖面相当扎实：

- **任务**：4 大类共 12 个数据集。
  - **算术推理** （6 个）：SingleEq、AddSub、MultiArith、GSM8K、AQUA-RAT、SVAMP。前两个较简单（单步可解），后四个需要多步推理。
  - **常识推理** （2 个）：CommonsenseQA、StrategyQA。
  - **符号推理** （2 个）：Last Letter Concatenation（拼接每个单词的最后一个字母）、Coin Flip（追踪硬币翻转多次后的朝向），这两个是作者自己构造的。
  - **其他逻辑推理** （2 个）：BIG-bench 中的 Date Understanding 和 Tracking Shuffled Objects。
- **模型**：共 17 个，参数从 0.3B 到 540B，包括 Instruct GPT-3（text-ada/babbage/curie/davinci-001、davinci-002）、原版 GPT-3、PaLM（8B/62B/540B），以及用于规模研究的 GPT-2、GPT-Neo、GPT-J、T0、OPT。
- **解码**：除 PaLM 外用贪心解码（temperature = 0），max_tokens = 128；PaLM 用 TopK=1，max_tokens = 256。

## 实验结果：一句咒语的威力

### 主结果：Zero-shot-CoT vs 标准 Zero-shot

| 任务类型 | 数据集 | Zero-shot | Zero-shot-CoT |
| --- | --- | --- | --- |
| 算术 | SingleEq | 78.7 | 78.7 |
| 算术 | AddSub | 77.0 | 74.7 |
| 算术 | **MultiArith** | 17.7 | **78.7** |
| 算术 | **GSM8K** | 10.4 | **40.7** |
| 算术 | AQUA-RAT | 22.4 | **33.5** |
| 算术 | SVAMP | 58.8 | **62.1** |
| 常识 | CommonsenseQA | 72.6 | 64.0 |
| 常识 | StrategyQA | 54.3 | 52.3 |
| 逻辑 | Date Understanding | 33.6 | **61.8** |
| 逻辑 | Shuffled Objects | 29.7 | **52.9** |
| 符号 | Last Letter | 0.2 | **57.6** |
| 符号 | Coin Flip | 53.8 | **91.4** |

（表中为各方法两组答案提取 prompt 中较好的一组数值，模型为 text-davinci-002）

几个值得品读的观察：

- **提升集中在多步推理任务**。MultiArith（17.7% → 78.7%）和 GSM8K（10.4% → 40.7%）这种需要多步计算的任务提升是数量级的；而 SingleEq、AddSub 这种单步可解的简单题基本持平——这符合预期，因为本来就不需要"一步步想"。
- **符号推理效果惊人**。Last Letter 从几乎全错（0.2%，模型连输出格式都搞不对）涨到 57.6%；Coin Flip 从接近随机猜测（53.8%）涨到 91.4%。
- **常识推理没有提升甚至略降**。这是论文坦诚指出的局限。但作者深挖样本后发现一个有趣现象：即使最终答案错了，生成的思维链本身往往 **逻辑上是通的** ，只是得出了反常识的结论；模型还经常拒绝单一选项，给出"A、B、C、D 或 E 都可能"这种模棱两可的输出。这说明 Zero-shot-CoT 确实激发了更好的推理，只是常识类任务的评测指标没有直接反映出来。

### 和其他基线掰手腕

| 方法 | MultiArith | GSM8K |
| --- | --- | --- |
| Zero-Shot | 17.7 | 10.4 |
| Few-Shot（2 示例） | 33.7 | 15.6 |
| Few-Shot（8 示例） | 33.8 | 15.6 |
| **Zero-Shot-CoT（本文）** | **78.7** | **40.7** |
| Few-Shot-CoT（2 示例） | 84.8 | 41.3 |
| Few-Shot-CoT（8 示例） | 93.0 | 48.7 |
| **Zero-Plus-Few-Shot-CoT（8 示例）** | **92.8** | **51.5** |
| 微调 GPT-3 175B | - | 33 |
| 微调 GPT-3 175B + verifier | - | 55 |
| PaLM 540B：Zero-Shot | 25.5 | 12.5 |
| **PaLM 540B：Zero-Shot-CoT** | **66.1** | **43.0** |
| **PaLM 540B：Zero-Shot-CoT + Self-Consistency** | **89.0** | **70.1** |
| PaLM 540B：Few-Shot-CoT | - | 56.9 |
| PaLM 540B：Few-Shot-CoT + Self-Consistency | - | 74.4 |

这张表里有几个"杀手级"数字：

- Zero-shot-CoT **大幅超过给了 8 个示例的标准 Few-shot** （78.7 vs 33.8）。也就是说，示例本身没有触发 CoT 的话，给再多也没用——鸿沟不在示例数量，而在是否激发多步推理。
- 在 GSM8K 上，一个不调参、不给示例的 prompt 方法（40.7）竟然 **打赢了微调的 GPT-3 175B** （33）。
- 作者还顺手试了一个组合技 **Zero-Plus-Few-Shot-CoT**：在 Few-shot-CoT 每个示例的答案开头也插入 "Let's think step by step"，GSM8K 达到 51.5%，反超原版 Few-shot-CoT 的 48.7%——说明这句咒语和示例是正交的增益。
- 在 PaLM 540B 上结论完全复现（MultiArith 25.5 → 66.1，GSM8K 12.5 → 43.0），证明这不是 OpenAI 模型的专属现象。再叠加 **Self-Consistency** （多次随机采样推理路径后多数投票），GSM8K 冲到 70.1%，已经非常接近 Few-shot-CoT + Self-Consistency 的 74.4%。

### 模型规模：推理能力是一种"涌现"

![模型规模研究](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Formatting-Instructions-For-NeurIPS-2022/fig_model_scale_instructgpt.png)

> 图解：横轴是模型规模（Instruct GPT-3 从 text-ada-001 的 S 到 text-davinci-002 的 XL），纵轴是 MultiArith 准确率。可以看到：不用 CoT 时（Zero-shot、Few-shot 曲线），性能随规模增长极其缓慢，曲线几乎是平的；而一旦启用 CoT（Zero-shot-CoT、Few-shot-CoT 曲线），性能随模型变大陡峭上升。CoT 在小模型上几乎无效甚至有害，规模到了 100B 级别才"开窍"。

具体数字很有说服力：MultiArith 上，原版 GPT-3 从 0.3B 到 175B，Zero-shot-CoT 从 1.7% 涨到 19.0%；Instruct GPT-3 的 text-davinci-002 达到 78.7%。而 GPT-2（1.5B）、GPT-Neo（2.7B）、GPT-J（6B）、T0（11B）、OPT（13B）这些中小模型上，Zero-shot-CoT 和 Zero-shot 基本没差别。

PaLM 上的 GSM8K 数据同样如此：8B / 62B / 540B 三个规模下，Zero-shot 是 2.1 / 7.0 / 12.5，Zero-shot-CoT 是 2.4 / 10.5 / 43.0——小模型上几乎零增益，540B 上才爆发。

附录里不同规模模型生成文本的对比非常生动：text-davinci-002 能一步步算出 28，而 text-ada-001 只会鹦鹉学舌地复述题目，GPT-2 则陷入 "Todd's brother, Todd, eats 32 cupcakes" 的无限复读。这也与 Wei 等人 Few-shot-CoT 的发现一致： **CoT 是一种随规模涌现（emergent）的能力**。

### 触发语敏感度：话术的讲究

作者系统测试了 16 种触发语模板，分三类，结果（MultiArith，text-davinci-002）：

| 类别 | 模板 | 准确率 |
| --- | --- | --- |
| instructive | **Let's think step by step.** | **78.7** |
| instructive | First, | 77.3 |
| instructive | Let's think about this logically. | 74.5 |
| instructive | Let's solve this problem by splitting it into steps. | 72.2 |
| instructive | Let's be realistic and think step by step. | 70.8 |
| instructive | Let's think like a detective step by step. | 70.3 |
| instructive | Let's think | 57.5 |
| misleading | Don't think. Just feel. | 18.8 |
| misleading | Let's think step by step but reach an incorrect answer. | 18.7 |
| misleading | By using the fact that the earth is round, | 9.3 |
| irrelevant | By the way, I found a good restaurant nearby. | 17.5 |
| irrelevant | Abrakadabra! | 15.5 |
| （对照） | 标准 Zero-shot（无触发语） | 17.7 |

规律很清晰： **只要话术在鼓励分步推理，性能就远超基线** ；误导性或无关的句子则完全无效（和 17.7% 的基线持平甚至更低）。"Let's think step by step" 是最强的一句，但并非玄学——同族的话术全都有效，说明起作用的是"鼓励推理"这个语义，而不是某个魔法字符串。

### Few-shot-CoT 的阿喀琉斯之踵：示例错配

解决了"Zero-shot 行不行"之后，下一个问题是：Few-shot-CoT 精心准备的示例到底有多稳健？作者做了一个很损的实验：拿 CommonsenseQA（常识选择题）的 CoT 示例去解算术题。

| 任务 | Zero-shot | Few-shot-CoT（错配示例） | **Zero-shot-CoT** | Few-shot-CoT（匹配示例） |
| --- | --- | --- | --- | --- |
| AQUA-RAT（同为选择题格式） | 22.4 | 31.9 | **33.5** | 39.0 |
| MultiArith（答案格式不同） | 17.7 | 27.0 | **78.7** | 88.2 |

结果触目惊心：示例和任务错配时，Few-shot-CoT 性能崩塌。尤其是答案格式都不同时（选择题示例 → 数值题），27.0% 的成绩被 Zero-shot-CoT 的 78.7% 吊打。而只要答案格式一致（都是选择题），即使领域完全不同，也还能保留不少增益——这印证了 Min 等人的发现： **模型从 few-shot 示例中学到的主要是"输出格式"，而不是任务本身**。

**博主点评**：这个实验其实是对 Few-shot-CoT 的一记釜底抽薪——它说明 Few-shot-CoT 的高分很大程度建立在"示例与任务严丝合缝"的前提上，而这个前提在真实应用场景中恰恰是最难保证的。Zero-shot-CoT 不需要任何示例，天然免疫这个问题。

### 错误分析：两种方法错得不一样

作者在 MultiArith 上对各 50 个正确/错误样本做了人工归因，发现两种 CoT 的"失败人格"截然不同：

| 错误类型 | Zero-shot-CoT | Few-shot-CoT |
| --- | --- | --- |
| 常识性错误 | 10.0% | **23.8%** |
| 计算错误 | (8.0%) | (**26.2%**) |
| 多此一举（推理对后又画蛇添足改错） | (**10.0%**) | (2.4%) |
| 其他（根本没开始推理等） | **20.0%** | 2.4% |

- **Zero-shot-CoT 的典型错误** 是"自由过了火"：要么已经算对又继续输出多余步骤把答案改错，要么干脆不推理、把题目换个说法复述一遍，偶尔还会撞上 max_tokens 长度上限被截断。
- **Few-shot-CoT 的典型错误** 是"被示例带偏"：计算错误比例是 Zero-shot-CoT 的三倍多（26.2% vs 8.0%），尤其在遇到 $(3+2) \times 4$ 这种三元运算时容易翻车——因为示例里没有这种形态，模型硬套示例的格式反而算错；常识性错误也明显更多。

这个对比反过来佐证了开头的动机：人工示例确实会把人类的思维定式（和盲区）注入模型，而 Zero-shot-CoT 让模型"用自己的方式想"，犯的错误反而更"自然"。

## 讨论：从"窄 prompt"到"宽 prompt"

把本文放进 prompt 研究的谱系里看，定位会更清晰。此前的工作大致分三档：微调（需要训练数据，输出 CoT）、Few-shot prompting（需要 per-task 示例）、Zero-shot prompting（需要 per-task 模板）。Zero-shot-CoT 是唯一一个 **单一模板跨任务通用** 且输出思维链的方法——Liu 等人发现的 "Let's solve this problem by splitting it into steps" 与之类似，但当时只被当作个别案例展示，没有做过定量系统评测。

作者借用了 Chollet 的智能层级理论来升华这个发现：以往每个任务定制 prompt 的做法，激发的是 LLM 的 **narrow generalization** （狭窄泛化、任务特定技能）；而一句跨任务通用的 "Let's think step by step"，激发的是 **broad generalization** （宽泛泛化、通用认知能力）——即"逻辑推理"这个 System-2 能力本身。这意味着 LLM 内部可能还藏着大量未被挖掘的 **高层、多任务的 zero-shot 能力** ，等待被简单的 prompt 唤醒。社区一直假设 LLM 是"优秀的 few-shot 学习者"，这篇论文则证明：在准备微调数据集或手工示例之前，模型自带的 zero-shot 能力被严重低估了。

当然也要指出局限：本文结论建立在 InstructGPT、GPT-3、PaLM 等模型之上，这些模型的训练数据细节并不完全公开；不过多个厂商、多个架构模型上结论一致复现，说明模型不太可能只是在"背题"，而是真的具备任务无关的多步推理能力。

## 总结

- **一句话的核心贡献**：不加任何示例，只在问题后追加 "Let's think step by step"，就能让 LLM 以 zero-shot 方式输出思维链并完成多步推理。
- **实现上是两阶段 prompting**：先提取推理路径，再用自增强 prompt 提取规范格式的答案。
- **效果惊人**：MultiArith 17.7% → 78.7%，GSM8K 10.4% → 40.7%，零示例超过 8-shot 标准 prompting，甚至在 GSM8K 上超过微调 GPT-3。
- **关键前提**：CoT 是规模涌现的能力，只有百亿参数以上的大模型才能被这句话点醒。
- **深层启示**：人工示例会注入偏见、且对任务错配极其脆弱；LLM 内部藏着被低估的通用 zero-shot 推理能力，值得在堆数据微调之前先充分挖掘。

这篇论文最大的价值或许不在于方法本身（简单到不像一篇论文），而在于它扭转了整个领域的视角：从"如何让模型学会某个任务"转向"模型本来就会什么、如何用一句话把它唤出来"。今天无处不在的 "think step by step" 类 prompt 以及后续的自动 prompt 优化研究，都可以追溯到这篇文章。

> 本文参考自 [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)