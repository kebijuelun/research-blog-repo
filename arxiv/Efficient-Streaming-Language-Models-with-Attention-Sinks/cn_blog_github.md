# StreamingLLM：保留注意力汇点，让流式对话稳定运行

StreamingLLM 要解决的是：让只有有限训练窗口的 LLM，在多轮对话等持续输入场景中长时间运行，同时避免 KV Cache 无限增长和模型性能崩溃。它的核心发现是，模型会把大量“无处安放”的注意力分配给开头的几个 token；只要保留这些 **Attention Sinks（注意力汇点）**，再配合最近的滑动窗口，无需微调就能稳定生成。实验中，Llama-2、MPT、Falcon 和 Pythia 等多个模型可稳定处理长达 **400 万 token** 的文本；与质量可接受的滑动窗口重计算方案相比，单 token 解码最高提速 **22.2 倍**。需要注意，它延长的是模型可持续运行的时间，而不是模型能够记忆的上下文长度。

## 为什么“一直聊下去”比想象中困难

自回归 Transformer 在解码时，会缓存历史 token 的 Key 和 Value，也就是 KV Cache，避免每生成一个 token 都重算整段历史。但输入不断增长时，两个矛盾会同时出现：

- **效率问题**：缓存所有历史 KV 会持续占用显存，注意力计算和解码延迟也会随序列变长而增加。
- **长度泛化问题**：Llama-2 等模型在固定长度窗口内预训练，输入一旦明显超过该窗口，性能就可能下降。

一个直观方案是 **Window Attention（窗口注意力）**：只保留最近固定数量的 token。缓存填满后，它的显存占用和每步计算量基本固定，看起来很适合流式应用。然而，一旦最早 token 的 KV 被移出窗口，模型质量就会突然崩溃。

另一种方案是 **Sliding Window with Re-computation（滑动窗口重计算）**：每生成一个 token，都基于最近一段文本重新构建 KV。这种方式质量较好，但窗口内部仍需进行二次复杂度的注意力计算，速度难以满足持续部署需求。

![Dense Attention、Window Attention、滑动窗口重计算与 StreamingLLM 的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/scheme.png)

> 图解：设模型预训练长度为 $L$，当前要预测远超出该范围的第 $T$ 个 token。Dense Attention 的缓存持续增长，时间复杂度为 $O(T^2)$；Window Attention 只留最近 $L$ 个 token，却会在开头 token 被淘汰后失稳；重计算方案的整体复杂度为 $O(TL^2)$；StreamingLLM 则同时保留开头少量 token 和最近 token，以固定缓存换取稳定的流式生成。图中困惑度来自 Llama-2-13B 在 PG19 首本约 65K token 书籍上的测试。

这也解释了 StreamingLLM 与几类长文本技术的边界：

- **Length Extrapolation（长度外推）** 关注训练短、测试长，RoPE 和 ALiBi 都属于相关位置编码方案，但它们并不自动保证无限长输入下仍稳定。
- **Context Window Extension（上下文窗口扩展）** 通过位置插值、微调或系统优化扩大一次前向计算可见的范围，但窗口依然有限。
- **长上下文利用** 研究模型能否真正找到并使用窗口内的信息；窗口更大，不代表信息利用就更好。

StreamingLLM 属于第一类，并可与后两类方法组合。它不重训模型、不扩大实际记忆窗口，而是让已有模型在“开头锚点 + 近期上下文”中持续工作。

## 窗口注意力为何崩溃：开头 token 是注意力汇点

要修复窗口注意力，先要定位性能骤降发生在何时。论文在 20K token 文本上比较多种方案，结果非常有规律性：Dense Attention 在超过预训练窗口后失效；Window Attention 在超过缓存长度、开始移除初始 token 后失效；StreamingLLM 的困惑度则接近重计算基线。

![20K token 文本上的语言模型困惑度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/nll_curve.png)

> 图解：横轴方向表示输入文本不断变长，纵轴比较语言建模困惑度，越低越好。不同模型都呈现相同趋势：性能崩溃并非简单地由“文本很长”触发，而是与初始 token 被移出缓存紧密相关；保留这些 token 后，曲线重新稳定。

### Softmax 需要一个“剩余注意力出口”

论文进一步可视化 Llama-2-7B 的全部层和注意力头。前两层主要表现为局部注意力，而最近 token 更重要；从更深层开始，大量层和头都会把注意力集中到序列开头，即使开头内容与当前预测没有明显语义关系。

![Llama-2-7B 各层、各注意力头的平均注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/attention_weights.png)

> 图解：结果在 256 个长度均为 16 的句子上取平均。前两层的热图更偏向邻近 token；其余大多数层和头中，序列最左侧的初始 token 明显更亮，说明它们获得了不成比例的注意力权重。

这种现象被称为 **Attention Sink**。可以从 Softmax 的归一化约束建立直觉：注意力权重总和必须为 1，即使当前 Query 不需要从历史 token 获取多少信息，也不能把所有权重都置零，总得把剩余权重“倒”到某个位置。

$$
\operatorname{SoftMax}(x)_i = \frac{e^{x_i}}{e^{x_1}+\sum_{j=2}^{N}e^{x_j}},\qquad x_1 \gg x_j,\quad j\in\{2,\ldots,N\}
$$

如果第一个位置承担了大量权重，删除它的 KV 就会显著改变 Softmax 分母，进而让整份注意力分布偏离模型训练时熟悉的状态。这个设计的聪明之处在于：它没有把初始 token 错当成“必须长期保留的语义记忆”，而是把它们视为维持计算稳定的结构锚点。

为什么恰好是开头？在自回归注意力中，初始 token 几乎能被所有后续 token 看见，而较后的 token 只能被更有限的后续位置看见。因此，初始位置最容易在训练中成为统一的注意力出口。

### 是语义重要，还是位置重要？

论文用换词实验排除了“开头语义不可替代”的解释：把最初四个 token 全部换成换行符 `\n`，再保留最近 1020 个 token，困惑度仍能恢复到接近原始初始 token 的水平。

下表的缓存配置 $x+y$ 表示保留 $x$ 个初始 token 与 $y$ 个最近 token；PPL 为困惑度，越低越好，评测使用 PG19 测试集首本 65K token 的书。

| Llama-2-13B 缓存配置 | PPL（↓） |
|---|---:|
| 0+1024，纯窗口注意力 | 5158.07 |
| 4+1020，保留原始开头 | 5.40 |
| 4 个 `\n`+1020 | 5.60 |

可见，真正关键的是 **起始位置及其 KV 所承担的注意力汇点功能**，而不是最初几个词的具体含义。

那么只留一个初始 token 够吗？消融实验表明，通常不够。已有模型往往把汇点功能分散到多个起始位置，四个初始 token 基本足够，再增加到八个收益很小。

| 模型 | 0+窗口 | 1+窗口 | 2+窗口 | 4+窗口 | 8+窗口 |
|---|---:|---:|---:|---:|---:|
| Falcon-7B | 17.90 | 12.12 | 12.12 | 12.12 | 12.12 |
| MPT-7B | 460.29 | 14.99 | 15.00 | 14.99 | 14.98 |
| Pythia-12B | 21.62 | 11.95 | 12.09 | 12.09 | 12.02 |
| Llama-2-7B | 3359.95 | 11.88 | 10.51 | 9.59 | 9.54 |

其中前三行总缓存为 2048，Llama-2-7B 总缓存为 4096；困惑度在拼接后的 PG19 测试集 400K token 上计算。论文因此把 **4 个初始 token** 作为默认设置，而不是武断地认为“第一个 token 就是唯一汇点”。

## StreamingLLM：固定大小的滚动 KV Cache

理解崩溃原因后，方法本身非常简洁：在普通滑动窗口之外，永久保留少量初始 token 的 KV。

缓存由两部分组成：

1. **Attention Sinks**：默认四个初始 token，用于稳定 Softmax 注意力分布。
2. **Rolling KV Cache**：持续滚动更新，保留最近 token，为语言建模提供真实的局部上下文。

![StreamingLLM 的滚动 KV Cache](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/rolling_kv.png)

> 图解：缓存左侧是固定不动的初始 token，右侧是随生成不断滑动的近期 token。新 token 到来后，只淘汰滚动区中最旧的 KV，不动注意力汇点，因此缓存总量始终受限。

这套机制可以直接接入采用 RoPE 或 ALiBi 等相对位置编码的现有自回归模型，不需要微调。

### 位置必须按“缓存内坐标”计算

StreamingLLM 有一个容易忽略但至关重要的细节：**位置编码不能把缓存中不连续的原始文本编号直接交给模型，而要重新按缓存内顺序编号。**

例如，缓存里的原始 token 编号为：

$$
[0,1,2,3,6,7,8]
$$

接下来要解码原文第 9 个 token。StreamingLLM 分配给当前计算的位置是：

$$
[0,1,2,3,4,5,6,7]
$$

而不是 $[0,1,2,3,6,7,8,9]$。这样，模型看到的仍是一段位置连续、相对距离符合预训练经验的序列。

对 **RoPE**，实现上要先缓存施加旋转变换之前的 Key，再在每轮解码时按缓存内位置重新做位置变换。对 **ALiBi**，则给当前连续缓存施加线性距离偏置，不能使用原始编号造成的“跳跃”偏置。

这也说明 StreamingLLM 并非简单地“少删几行缓存”；若位置处理不一致，保留汇点也无法完整复现论文中的稳定效果。

## 从头预训练一个专用 Sink Token

现有模型需要多个初始 token，很可能是因为预训练样本没有统一、固定的起始符号。以 Llama-2 为例，`<s>` 在文本切块之前添加，切块后第零个位置仍可能是任意 token，模型只能把汇点功能分摊到若干起始位置。

论文因此提出：在每个训练样本开头加入一个可学习的占位 token，让它成为专门的 Sink Token。另一种思路是 SoftMax-off-by-One：

$$
\operatorname{SoftMax}_1(x)_i = \frac{e^{x_i}}{1+\sum_{j=1}^{N}e^{x_j}}
$$

分母额外加 1，使所有上下文 token 的注意力不必再严格合计为 1。它等价于在注意力计算前添加一个 Key、Value 都为零的 token，论文称之为 **Zero Sink**。

作者使用相同的 Pythia-160M 训练配置，在去重后的 Pile 数据上训练 143,000 步，仅将训练 batch size 调为 256。单 Sink Token 的加入没有破坏收敛，也没有损害七项零样本 NLP 基准表现。

![有无 Sink Token 的预训练损失曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/learnable_sink_curve.png)

> 图解：两条训练损失曲线的收敛趋势接近，说明增加专用汇点没有引入明显的训练困难。

下表为零样本准确率，单位为 %。ARC-c、ARC-e 分别表示 ARC Challenge 和 ARC Easy；HS、LBD、OBQA、WG 分别表示 HellaSwag、LAMBADA、OpenBookQA 和 Winogrande。

| 训练方式 | ARC-c | ARC-e | HS | LBD | OBQA | PIQA | WG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 18.6 | 45.2 | 29.4 | 39.6 | 16.0 | 62.2 | 50.1 |
| +1 Sink Token | **19.6** | **45.6** | **29.8** | **39.9** | **16.6** | **62.6** | **50.8** |

更关键的是流式表现。下表在 PG19 测试集第一个样本上报告困惑度：Vanilla 仍需要多个初始 token；Zero Sink 虽有缓解，却仍依赖其他初始位置；Learnable Sink 只需保留一个专用汇点即可稳定。

| 预训练方式 | 0+1024 | 1+1023 | 2+1022 | 4+1020 |
|---|---:|---:|---:|---:|
| Vanilla | 27.87 | 18.49 | 18.05 | 18.05 |
| Zero Sink | 29214 | 19.90 | 18.27 | 18.01 |
| Learnable Sink | 1235 | **18.01** | 18.01 | 18.02 |

注意，$0+1024$ 对 Learnable Sink 仍失败，是因为此时连专用 Sink Token 本身也没有被保留；$1+1023$ 才是“一个汇点 + 最近 token”的目标配置。

![有无 Sink Token 预训练模型的注意力对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/attention_vis_pretrain.png)

> 图解：左右分别是未加入和加入 Sink Token 的模型，注意力在 256 个长度为 16 的句子上取平均，层与头一一对应。没有专用汇点时，浅层偏局部、深层转向多个初始 token；加入后，各层和头更集中地关注专用 Sink Token，对其他初始 token 的依赖随之减弱。

## 实验：从 400 万 token 到真实流式问答

实验覆盖 Llama-2、MPT、Pythia 和 Falcon 四个模型家族，同时涵盖 RoPE 与 ALiBi 两类位置编码。除专门消融外，StreamingLLM 默认保留四个初始 token。

### 多模型、多规模的超长文本建模

作者拼接 PG19 测试集中的 100 本长书进行语言建模。Llama-2 的缓存设为 2048，Falcon、Pythia 和 MPT 设为 1024，均为相应预训练窗口的一半。

![StreamingLLM 在 400 万 token 文本上的困惑度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/long_lm.png)

> 图解：横轴持续推进到超过 400 万 token，纵轴为困惑度。Llama-2-7B/13B/70B、Falcon-7B/40B、Pythia-2.8B/6.9B/12B 和 MPT-7B/30B 整体都保持稳定；局部波动主要来自不同书籍之间的切换，而不是随流式长度增长持续恶化。

这组结果证明的是“可以长期稳定地基于缓存生成”，并不表示模型能回忆 400 万 token 前的具体信息。

### 更大的缓存不总是更低困惑度

在固定四个汇点后，作者继续改变近期缓存大小。结果没有呈现“窗口越大越好”的单调规律，说明模型未必能充分利用所有可见上下文。

| 模型 | 较小缓存 | 次小缓存 | 次大缓存 | 最大缓存 |
|---|---:|---:|---:|---:|
| Falcon-7B | 4+252：13.61 | 4+508：12.84 | 4+1020：**12.34** | 4+2044：12.84 |
| MPT-7B | 4+252：**14.12** | 4+508：14.25 | 4+1020：14.33 | 4+2044：14.99 |
| Pythia-12B | 4+252：13.17 | 4+508：12.52 | 4+1020：**12.08** | 4+2044：12.09 |
| Llama-2-7B | 4+508：9.73 | 4+1020：9.32 | 4+2044：**9.08** | 4+4092：9.59 |

数值仍为拼接 PG19 测试集 400K token 上的困惑度。这里暴露的是模型上下文利用能力的上限，而非 StreamingLLM 缓存机制本身失效。

### 连续问答：保留汇点可恢复指令模型准确率

作者将 ARC Easy 和 ARC Challenge 的问答对连续拼接，交给 Llama-2-Chat 逐题作答，用 Exact Match 评价。Dense Attention 因 OOM（显存溢出）无法完成；Window Attention 虽能运行，却在初始 KV 淘汰后近乎随机输出；StreamingLLM 与逐样本 One-shot 基线相当，部分结果还略高。

| 模型与方法 | ARC-E | ARC-C |
|---|---:|---:|
| Llama-2-7B-Chat，One-shot | 71.25 | 53.16 |
| Llama-2-7B-Chat，StreamingLLM | 71.34 | 55.03 |
| Llama-2-13B-Chat，One-shot | 78.16 | 63.31 |
| Llama-2-13B-Chat，StreamingLLM | 80.89 | 65.61 |
| Llama-2-70B-Chat，One-shot | 91.29 | 78.50 |
| Llama-2-70B-Chat，StreamingLLM | 91.37 | 80.20 |
| 三种规模的 Dense Attention | OOM | OOM |
| 7B / 13B / 70B，Window Attention | 3.58 / 0.25 / 0.12 | 1.39 / 0.34 / 0.32 |

表中数值为准确率（%），Window Attention 与 StreamingLLM 的缓存大小均为 1024。为保持每种模型与基线的对应关系，Window 一行依次列出 7B、13B、70B 的结果。

### StreamEval：问题只询问近期信息

为模拟更真实的持续问答，论文构建 StreamEval。它不同于只在长文末尾提一次问题的 LongEval，而是每输入 10 行新信息就提问一次，答案固定位于当前问题之前 20 行。

![StreamEval 的第一个样本](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/streameval_example.png)

> 图解：信息流不断追加，查询穿插在流中，而不是等到全文结束后才出现。该设置考察模型能否持续运行，并检索距离当前较近的答案，符合日常对话主要依赖近期上下文的特点。

![StreamEval 上不同注意力方案的准确率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/kv_bench.png)

> 图解：结果为 100 个样本的平均准确率。Dense Attention 在预训练长度附近失败，Window Attention 在缓存开始淘汰初始 token 后失败；StreamingLLM 在输入接近 120K token 时仍保持合理准确率。LongChat-7B-v1.5-32K 和 Llama-2-7B-32K-Instruct 的结果还表明，上下文扩展可与 StreamingLLM 互补：前者扩大可用的近期窗口，后者控制持续运行的成本。

### 效率：最高 22.2 倍单 token 解码加速

延迟与显存测试在单张 NVIDIA A6000 上进行，使用 Hugging Face Transformers 实现，模型为 Llama-2-7B 和 Llama-2-13B。对比对象是质量可接受的滑动窗口重计算方案。

![StreamingLLM 与重计算基线的延迟和显存对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/efficiency.png)

> 图解：横轴是缓存大小，也就是注意力窗口大小，图中分别比较单 token 解码延迟和显存占用。随着窗口增大，重计算基线的延迟呈二次增长，StreamingLLM 的解码开销则近似线性变化，最高取得 **22.2 倍**每 token 加速；两者显存占用相近。这个倍数是特定测试设置下的峰值，不应理解为所有部署都固定提速 22.2 倍。

## 能力边界：可持续生成不等于长期记忆

StreamingLLM 最容易被误读的地方，是把“处理无限输入流”理解成“拥有无限上下文”。实际上，模型只能利用当前缓存里的近期信息和汇点，被滚出窗口的历史内容无法直接检索。

### 答案距离超过缓存，准确率就会归零

作者在 Llama-2-7B-32K-Instruct 上改变 StreamEval 中问题与答案的距离。每行含 23 个 token，因此 token 距离为行距的 23 倍；每个配置在 100 个样本上取平均，每个样本含 100 次查询。

| 行距 | Token 距离 | 4+2044 | 4+4092 | 4+8188 | 4+16380 |
|---|---:|---:|---:|---:|---:|
| 20 | 460 | 85.80 | 84.60 | 81.15 | 77.65 |
| 40 | 920 | 80.35 | 83.80 | 81.25 | 77.50 |
| 60 | 1380 | 79.15 | 82.80 | 81.50 | 78.50 |
| 80 | 1840 | 75.30 | 77.15 | 76.40 | 73.80 |
| 100 | 2300 | 0.00 | 61.60 | 50.10 | 40.50 |
| 150 | 3450 | 0.00 | 68.20 | 58.30 | 38.45 |
| 200 | 4600 | 0.00 | 0.00 | 62.75 | 46.90 |
| 400 | 9200 | 0.00 | 0.00 | 0.00 | 45.70 |
| 600 | 13800 | 0.00 | 0.00 | 0.00 | 28.50 |
| 800 | 18400 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | 23000 | 0.00 | 0.00 | 0.00 | 0.00 |

当答案距离超过对应缓存容量时，准确率最终降为零。即便答案仍在缓存内，更大窗口也不总带来更高准确率，再次反映了长上下文利用问题。

### LongBench：丢掉关键提示，汇点也救不回来

LongBench 测试使用最大上下文为 4K 的 Llama-2-7B-Chat，涵盖单文档问答、多文档问答和摘要。默认截断基线保留开头 1750 个与结尾 1750 个 token。StreamingLLM 若采用 4+3496，只保留四个开头 token，会丢失大量初始提示，表现因此下降；改为 1750+1750 后，整体恢复到接近截断基线。

| 方法 | NarrativeQA | Qasper | HotpotQA | 2WikiMQA | GovReport | MultiNews |
|---|---:|---:|---:|---:|---:|---:|
| Truncation 1750+1750 | 18.7 | 19.2 | 25.4 | 32.8 | 27.3 | 25.8 |
| StreamingLLM 4+3496 | 11.6 | 16.9 | 21.6 | 28.2 | 23.9 | 25.5 |
| StreamingLLM 1750+1750 | 18.2 | 19.7 | 24.9 | 32.0 | 26.3 | 25.9 |

因此，StreamingLLM 适合日常多轮对话、短文档问答等主要依赖短期记忆的场景；长文档问答、长文摘要等需要完整历史依赖的任务，仍需上下文扩展、检索或其他记忆机制配合。

## 附录证据：注意力汇点并非单一模型的偶然现象

主文结论之外，附录从更长序列、更大模型和其他 Transformer 架构验证了 Attention Sink 的普遍性。

### 长序列中的定量统计

![Llama-2-7B 第 4096 个 token 对首个 token 的注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/Llama-2-7B-first-token-attn.png)

> 图解：横轴为模型层，纵轴为 Softmax 后分配给第一个 token 的注意力分数。结果在 256 个长度均为 4096 的序列上取平均，误差棒表示同层不同注意力头之间的标准差。除最底部两层外，许多层中首个 token 获得的注意力超过总量的一半，为“删除初始 KV 会大幅改变 Softmax 分母”提供了定量证据。

### 长度 128 的 Llama-2-7B 注意力热图

![Layer 0 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-7b-128-attn/0.png)
![Layer 6 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-7b-128-attn/6.png)
![Layer 12 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-7b-128-attn/12.png)
![Layer 18 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-7b-128-attn/18.png)
![Layer 24 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-7b-128-attn/24.png)
![Layer 30 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-7b-128-attn/30.png)

> 图解：六幅热图依次展示 Llama-2-7B 的第 0、6、12、18、24、30 层第 0 个头，在 256 个长度为 128 的句子上取平均。短序列中的规律在长一些的输入上仍成立：浅层更偏局部注意力，多数后续层则持续关注初始 token，且这种集中不依赖它们与当前位置的距离。

### Llama-2-70B 的跨层、跨头结果

![Layer 0 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/0_0.png)
![Layer 0 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/0_1.png)
![Layer 8 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/8_0.png)
![Layer 8 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/8_1.png)
![Layer 16 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/16_0.png)
![Layer 16 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/16_1.png)
![Layer 24 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/24_0.png)
![Layer 24 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/24_1.png)
![Layer 32 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/32_0.png)
![Layer 32 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/32_1.png)
![Layer 40 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/40_0.png)
![Layer 40 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/40_1.png)
![Layer 48 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/48_0.png)
![Layer 48 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/48_1.png)
![Layer 56 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/56_0.png)
![Layer 56 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/56_1.png)
![Layer 64 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/64_0.png)
![Layer 64 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/64_1.png)
![Layer 72 Head 0](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/72_0.png)
![Layer 72 Head 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/llama-2-70b-avg-attn/72_1.png)

> 图解：这些热图覆盖 Llama-2-70B 从第 0 层到第 72 层的多个层，并在每层展示第 0、1 两个头；统计同样来自 256 个长度为 16 的句子。除浅层的局部模式外，大多数层和头都给初始 token 很高注意力，说明该现象并不限于 7B 规模。

### BERT 中也存在类似汇点

![BERT 第 0 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/0.png)
![BERT 第 1 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/1.png)
![BERT 第 2 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/2.png)
![BERT 第 3 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/3.png)
![BERT 第 4 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/4.png)
![BERT 第 5 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/5.png)
![BERT 第 6 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/6.png)
![BERT 第 7 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/7.png)
![BERT 第 8 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/8.png)
![BERT 第 9 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/9.png)
![BERT 第 10 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/10.png)
![BERT 第 11 层注意力](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/bert-attn/11.png)

> 图解：图中依次展示 BERT-base-uncased 的 12 层注意力，输入句子为“StreamingLLM can work on infinite-length texts without compromising efficiency and performance.”。多数层把异常高的注意力分配给始终存在的 `[SEP]`，说明注意力汇点也出现在 Encoder Transformer 中。它与 ViT 中承担全局信息收纳作用的 registers 有相似性，但位置和功能并不完全相同。

### 预训练时增加第二个 Sink Token 没有额外收益

论文还比较了零个、一个和两个可学习 Sink Token。三者的预训练损失曲线接近，但第二个汇点没有带来稳定的基准提升。

![0、1、2 个 Sink Token 的预训练损失](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Efficient-Streaming-Language-Models-with-Attention-Sinks/figures/two_learnable_sink_curve.png)

> 图解：三条损失曲线整体趋势相似，说明增加汇点数量没有显著改变优化过程；是否值得增加，还要看下游和流式表现。

下表为零样本准确率（%）：

| 训练方式 | ARC-c | ARC-e | HS | LBD | OBQA | PIQA | WG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 18.6 | 45.2 | 29.4 | 39.6 | 16.0 | 62.2 | 50.1 |
| +1 Sink Token | **19.6** | **45.6** | **29.8** | **39.9** | **16.6** | 62.6 | **50.8** |
| +2 Sink Tokens | 18.7 | 45.6 | 29.6 | 37.5 | 15.8 | **64.3** | 50.4 |

流式困惑度同样没有显示更多汇点的优势。两个 Sink Token 的模型甚至需要同时保留二者才能稳定，只保留其中一个时困惑度为 25.73。

| 预训练方式 | 0+1024 | 1+1023 | 2+1022 | 4+1020 |
|---|---:|---:|---:|---:|
| Vanilla | 27.87 | 18.49 | 18.05 | 18.05 |
| +1 Sink Token | 1235 | **18.01** | 18.01 | 18.02 |
| +2 Sink Tokens | 1262 | 25.73 | 18.05 | 18.05 |

这组结果支持“一个专用汇点已经足够”，也提示不能直接把 ViT 中“多个 registers 有益”的经验照搬到自回归语言模型。

## 总结：保留的是稳定锚点，不是全部历史

- 窗口注意力的崩溃点与初始 token 被淘汰高度一致，根因是大量注意力集中在这些位置，删除其 KV 会破坏 Softmax 分布。
- StreamingLLM 用“四个初始 token + 最近滚动窗口”构成固定大小缓存，并按缓存内连续位置计算 RoPE 或 ALiBi，无需微调已有模型。
- 多个模型家族和规模在最长 400 万 token 的流式语言建模中保持稳定；相对滑动窗口重计算，单 token 解码最高加速 22.2 倍。
- 预训练时加入一个可学习 Sink Token，可让流式部署只保留一个专用汇点，且不损害所测基准上的常规模型能力。
- StreamingLLM 不扩展真实上下文或长期记忆；答案超出缓存后仍无法检索，长文档任务需要与上下文扩展、检索等方法结合。

未来值得推进的方向，是让模型更充分利用缓存内的上下文，并把专用汇点、长期记忆与高效检索组合成真正可持续运行的 LLM 系统。

> 本文参考自 [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)