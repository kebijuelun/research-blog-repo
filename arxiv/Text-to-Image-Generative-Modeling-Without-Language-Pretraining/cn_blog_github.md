# 不要语言预训练，文生图反而更强？斯坦福 Halcyon 的"从零开始"实验

现代文生图（T2I）系统几乎无一例外地依赖互联网规模的**语言预训练**——要么挂一个预训练好的大语言模型做文本编码器，要么在多模态训练里混入海量纯文本。这篇来自 Stanford 和 Radical Numerics 的工作问了一个釜底抽薪的问题：**语言预训练对 T2I 生成真的必要吗？**他们提出 Halcyon，一族完全从零训练的早期融合（early-fusion）多模态自回归模型，只用 1 亿对图文数据（GPIC）预训练。结果相当硬：Halcyon-0.52B 以**18.2 倍**更少的总预训练算力击败 LlamaGen-XL/8（gFD 48.0 vs. 63.3），Halcyon-1.6B 以**275 倍**更少的算力击败 JiT-T2I-XXL/8++（gFD 35.3 vs. 41.0）。控制实验进一步表明：纯文本预训练对 T2I 生成**没有帮助**。

![总预训练算力与生成质量对比及 Halcyon 样例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/fig1_release_v2.png)

> 图解：左图横轴是构建整套 T2I 系统所需的**总预训练 FLOPs**（含语言模型、图像 tokenizer、图像生成器三部分，对数刻度），纵轴是 GPIC 上的生成质量 gFD（越低越好）。每颗星代表一个系统，旁边的环形图拆解了语言与视觉算力的占比。可以看到 LlamaGen、JiT-T2I 等系统的算力绝大部分（90% 以上）烧在语言预训练上，而高亮标注的 Halcyon 完全从零训练，以低得多的总算力取得更低的 gFD。右图是 Halcyon-4B 生成的 256×256 样例。

## 一、背景：T2I 对语言预训练的"隐形依赖"

T2I 生成的任务是把自然语言描述映射为图像。当前主流系统对语言预训练的依赖有两种形态：

- **晚期融合（late fusion）**：主流范式。先用纯文本预训练一个大的语言模型（如 T5-XXL、Qwen3-1.7B），冻结后用它产出文本嵌入，再通过 cross-attention 去条件化一个独立的图像生成器。Imagen、PixArt-α、LlamaGen、SANA、PixelGen 都属此类。
- **早期融合（early fusion）**：去掉独立的文本编码器，用一个统一的多模态模型同时处理文本和图像 token，但仍会在预训练中混入大量纯文本数据，或直接从一个预训练 LLM 初始化（Chameleon、Emu3、Transfusion、Tuna-2 等）。

问题在于：**语言预训练吃掉了绝大部分算力**——在代表性的 T2I 系统中，它占总预训练 FLOPs 的 80% 以上。而它到底给 T2I 生成带来了多少收益，此前从未被系统量化过；更糟的是，论文里报成本时常常不把语言预训练算进去，严重低估了构建一套 T2I 系统的真实代价。

一个有意思的对照是：不同系统花在语言和视觉上的 token 预算差异巨大。带语言预训练的系统通常消耗 $10^{12}\sim10^{14}$ 的纯文本 token，常常超过视觉预算本身；而 Halcyon 的语言信号只来自与图像配对的 caption token，比这些语言预算低两个数量级以上。

其实早期的 DALL-E、CogView、GLIDE 就是直接从图文对里学文本条件的，没有用语言预训练。但自从 Imagen 报告了"放大预训练文本编码器有好处"之后（Parti 的结论其实相反：收益很边缘），整个领域就默认了大语言模型是必需品。这篇论文就是要把这个"默认"放到实验台上检验。

## 二、Halcyon：完全从零训练的多模态自回归模型

Halcyon 的核心思路一句话概括：**把文本和图像都变成 token，塞进同一个词表，用一个 decoder-only Transformer 做 next-token prediction，所有组件——文本 tokenizer、图像 tokenizer、生成模型——全部从零训练。**架构上遵循 Chameleon、Emu3 的早期融合路线，backbone 采用 Qwen3 风格的 decoder-only 结构。

### 2.1 两个从零训练的 tokenizer

**图像 tokenizer。**采用 Cosmos Tokenizer 的编码器-解码器设计，配合有限标量量化（FSQ, Finite Scalar Quantization）。空间下采样 8 倍，码本大小 64,000（FSQ levels 为 $[8,8,8,5,5,5]$），decoder 宽度从 Cosmos 的 128 加宽到 192。每张 $256\times256$ 图像被表示为 $32\times32$ 网格按光栅序展开的 $N=1024$ 个图像 token $\mathbf{x}=(x_1,\ldots,x_N)$。训练分两个阶段：先 1 个 epoch 用重建损失 + LPIPS 感知损失，再 0.5 个 epoch 加 PatchGAN 对抗损失。

**文本 tokenizer。**在 GPIC caption（21.3 GB）加 OpenWebText（11.9 GB）上从零训练 byte-level BPE，词表仅 16,384。注意 OpenWebText 只用于训练 BPE 分词器本身，不用于训练多模态生成模型。

**联合词表。**文本 token、图像 token 和特殊 token 组成大小为 80,640 的单一词表（16,384 文本 + 64,000 图像 + 256 特殊），模型共享 embedding 表和输出头。用 `[SOV]`/`[EOV]` 标记图像的起止，`[EOS]` 标记序列结束，`[PAD]` 做填充。

### 2.2 序列格式与训练目标

预训练使用四种序列格式：

- **T（纯文本）**：$w_1\cdots w_M$ + `[EOS]`，packed 排列；
- **I（纯图像）**：`[SOV]` + $x_1\cdots x_N$ + `[EOV]` + `[EOS]`；
- **I2T（图生文）**：先生成图像 token，再以图像为条件生成 caption token；
- **T2I（文生图）**：caption 在前作为前缀，`[SOV]` 之后以文本为条件生成图像 token。

含图像的序列 pad 到固定长度 1280，padding 位置不计入损失。这里有一个精巧的观察：**I2T 序列与 T2I 序列做 caption dropout 后得到的"无条件图像"前缀完全相同**（都是 `[SOV]` + 图像 token + `[EOV]`），这个等价性在后面的 I2T 实验里会被巧妙利用。

训练目标是加权的 next-token 交叉熵：

$$
\mathcal{L} = \frac{\mathcal{L}_{\mathrm{text}} + \lambda_{\mathrm{img}}\,\mathcal{L}_{\mathrm{img}}}{M}
$$

其中

$$
\mathcal{L}_{\mathrm{text}} = -\sum_{i\in\mathcal{T}}\log p_{\theta}(s_i\mid s_{<i}), \quad
\mathcal{L}_{\mathrm{img}} = -\sum_{i\in\mathcal{I}}\log p_{\theta}(s_i\mid s_{<i})
$$

$\mathcal{T}$ 和 $\mathcal{I}$ 分别是非 padding 的文本与图像 token 集合，$M=|\mathcal{T}|+|\mathcal{I}|$。设 $\lambda_{\mathrm{img}}=0.5$（与 Emu3 一致），防止数量上占优的图像 token 主导训练损失。

推理时使用 classifier-free guidance（CFG）：训练时对 $p_{\mathrm{drop}}=0.1$ 比例的 T2I 样本丢掉 caption 以训练无条件分支，采样时组合两路 logits：

$$
\boldsymbol{\ell}_{\mathrm{cfg}} = \boldsymbol{\ell}_{\mathrm{u}} + \gamma\,(\boldsymbol{\ell}_{\mathrm{c}} - \boldsymbol{\ell}_{\mathrm{u}})
$$

其中 $\gamma$ 是引导强度，CFG 只施加在图像 token 的 logits 上。

### 2.3 数据组成与调度：五种受控配置

为了量化"语言预训练到底有没有用"，作者在**固定预训练预算**下设计了五种数据组成/调度方案（这是全文实验设计的关键骨架）：

- **Baseline T2I**：只用 T2I 样本训练；
- **Staged: Language → T2I**：先用 $\alpha$ 比例的训练做纯文本预训练（文本 token 量超过该模型规模的 Chinchilla 最优预算），再做 T2I——模拟 Tuna-2 这类"语言优先"的日程；
- **Staged: I2T → T2I**：先用 $\beta$ 比例做 I2T，再做 T2I；
- **Mixing: T2I + I2T**：全程交织 T2I 与 I2T，I2T 占 $\eta$；
- **Mixing: T2I（无 caption dropout）+ I2T**：设 $\eta = p_{\mathrm{drop}}$ 并关掉 caption dropout——由于 I2T 与"丢 caption 的 T2I"共享同一图像前缀，这个配置在不改变条件/无条件图像 token 数量的前提下，白捡了 I2T 监督。

这个设计是本文方法论上最值得称道的地方：所有比较都在**同一总算力预算**下进行，排除"钞能力"干扰，纯粹比较数据配比本身的效果。

## 三、实验设置

**数据与训练协议。**所有模型在 GPIC（约 1 亿图文对）上以 $256\times256$ 分辨率预训练一个 epoch，global batch size 1024，统一使用 warmup-stable-decay（WSD）学习率调度，AdamW 优化器（$\beta_1,\beta_2=0.9,0.95$），bf16 精度。语言预训练数据用 ClimbMix。

**模型规模。**Halcyon 训练了五个尺寸：0.1B / 0.25B / 0.52B / 1.6B / 4B，序列长度统一 1280。

**基线选择。**选基线并不容易——现有系统在训练目标、语言条件方式、图像表征上都不同，而本研究需要公开代码、能做受控干预的系统。最终选了两个代表性晚期融合家族：

- **LlamaGen**（自回归）：FLAN-T5-XL 文本编码器 + 离散图像 tokenizer + GPT-XL 图像生成器。三个变体：默认版（256 图像 token）；换用本文 GPIC tokenizer 的 LlamaGen-XL/8（1024 token，生成器 0.90B）；再把 FLAN-T5-XL（1.22B）换成 FLAN-T5-Base（0.11B）的版本。
- **JiT-T2I**（扩散 / flow matching）：预训练 Qwen3-1.7B + 像素空间生成器（无图像 tokenizer）。三个变体：默认 XXL/16（$16\times16$ patch，256 token）；加 LPIPS 感知损失的 XXL/16++；再把 patch 缩到 $8\times8$ 的 XXL/8++（1024 token）。

这两个家族覆盖了自回归 vs. 扩散、encoder-decoder vs. decoder-only 语言模型、256 vs. 1024 图像序列长度三个维度，对照相当全面。

**评估协议。**三个 benchmark：GPIC Test-50K（主指标，报告 gFD、Precision、Recall、Density、Coverage、MMD）；GenEval（组合式生成，使用 PixelGen 的去模板化 prompt）；DPG-Bench（复杂 prompt 遵循）。每个模型都在 GPIC Validation-50K 上扫 CFG 强度，取 gFD 最低者用于全部评测。另外对 Halcyon 报告四种设置下的 token 级 NLL，以及用 Qwen3-VL-Reranker-8B 打分的图像描述（captioning）对齐度。

## 四、主结果：不靠语言预训练，反而更强

先看两个"前菜"实验，它们本身就很能说明问题。

**图像表征和感知监督的影响。**把 LlamaGen-XL 的原版 tokenizer 换成 GPIC 训练的 tokenizer（图像序列从 256 变 1024），gFD 从 147.0 降到 63.3——**换 tokenizer 比换语言模型重要得多**。JiT-T2I 这边，加 LPIPS 感知损失把 gFD 从 67.9 降到 48.9，再把 patch 从 $16\times16$ 缩到 $8\times8$ 进一步降到 41.0。

**语言编码器规模的影响。**把 LlamaGen 的 FLAN-T5-XL（1.22B）换成 FLAN-T5-Base（0.11B），语言模型训练算力减少 13.5 倍（$9.06\times10^{21}$ vs. $6.69\times10^{20}$ FLOPs），gFD 却只是从 63.3 轻微变差到 66.7。这是第一个直接证据：**语言预训练算力增加一个数量级，只换来 T2I 质量的边际改善**。

然后是主结果：

| 模型 | 语言模型 | 视觉参数 | gFD ↓ | Prec. ↑ | Rec. ↑ | Dens. → 1 | Cov. ↑ | MMD ↓ |
|---|---|---|---|---|---|---|---|---|
| LlamaGen-XL | 1.22B | 0.78B | 147.01 | 0.914 | 0.715 | 0.925 | 0.822 | 0.174 |
| LlamaGen-XL/8 | 1.22B | 0.90B | 63.29 | 0.948 | 0.827 | 1.058 | 0.925 | 0.041 |
| LlamaGen-XL-T5-Base/8 | 0.11B | 0.90B | 66.69 | 0.941 | 0.822 | 1.016 | 0.924 | 0.045 |
| JiT-T2I-XXL/16 | 1.70B | 1.10B | 67.92 | 0.934 | 0.822 | 0.979 | 0.923 | 0.038 |
| JiT-T2I-XXL/16++ | 1.70B | 1.10B | 48.86 | 0.947 | 0.861 | 1.022 | 0.939 | 0.022 |
| JiT-T2I-XXL/8++ | 1.70B | 1.10B | 40.97 | 0.949 | 0.880 | 1.049 | 0.952 | 0.016 |
| **Halcyon-0.10B** | — | 0.10B | 144.16 | 0.924 | 0.636 | 1.029 | 0.853 | 0.131 |
| **Halcyon-0.25B** | — | 0.25B | 72.55 | 0.946 | 0.796 | 1.050 | 0.922 | 0.050 |
| **Halcyon-0.52B** | — | 0.52B | **48.02** | 0.953 | 0.858 | 1.064 | 0.938 | 0.027 |
| **Halcyon-1.60B** | — | 1.60B | **35.28** | 0.955 | 0.891 | 1.071 | 0.955 | 0.019 |
| **Halcyon-4.00B** | — | 4.00B | **28.43** | 0.951 | 0.905 | 1.034 | 0.960 | 0.014 |

几个关键读数：

- Halcyon-0.52B（单一模型、从零训练）gFD 48.0，**超过**用 FLAN-T5-XL 的 LlamaGen-XL/8（63.3），逼近用 Qwen3-1.7B 的 JiT-T2I-XXL/8++（41.0）；
- Halcyon-1.6B 降到 35.3，**同时超越两个晚期融合系统**；4B 进一步降到 28.4；
- 组合生成能力上，Halcyon-0.52B 在 GenEval 上已超过 JiT-T2I-XXL/8++（44.9 vs. 43.2），DPG-Bench 打平（76.84 vs. 76.85）；1.6B 则两项全超（GenEval 53.5 vs. 43.2，DPG-Bench 77.9 vs. 76.9）。

真正"杀人诛心"的是算力账。把构建每个 T2I 系统的**全部**预训练 FLOPs（语言模型 + 图像 tokenizer + 图像生成器）摆在一起：

| 方法 | 总 FLOPs | 语言占比 | gFD ↓ |
|---|---|---|---|
| LlamaGen-XL/8 | $9.68\times10^{21}$ | 93.6% | 63.29 |
| JiT-T2I-XXL/8++ | $3.68\times10^{23}$ | 99.8% | 40.97 |
| **Halcyon-0.52B** | $5.33\times10^{20}$ | 15.0% | 48.02 |
| **Halcyon-1.6B** | $1.34\times10^{21}$ | 17.9% | 35.28 |
| **Halcyon-4B** | $3.08\times10^{21}$ | 19.0% | 28.43 |

也就是说，Halcyon-1.6B 用 JiT-T2I-XXL/8++ **1/275** 的总预训练算力取得了更好的生成质量。注意 Halcyon 的"语言 FLOPs"只是联合训练中文本 token 那部分，不存在外部语言预训练。

![各模型在相同 prompt 下的生成样例对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/collage_4types_v4.png)

> 图解：从左到右依次是 JiT-T2I-XXL/8++、LlamaGen-XL/8，以及 Halcyon 0.1B 到 4B 在相同 prompt（GPIC Test-50K）下的生成结果。基线模型分别标注了语言模型和图像生成器的参数规模。可以直观看到 Halcyon 随规模增大，图像质量与 prompt 遵循度（服饰、物体属性、场景布局等细节）持续提升。

## 五、控制实验一：语言预训练到底有用吗？

主结果说明"不用语言预训练也能赢"，但还没回答"加了会不会更好"。作者用两组受控干预来回答。

**干预 1：在固定预算内划一部分给纯文本预训练。**对 Halcyon-0.52B，把 $\alpha=0.1, 0.2, 0.5$ 的训练先分给纯文本（量都超过 Chinchilla 参考预算），再做 T2I：

| 配置 | $\alpha$ | gFD ↓ |
|---|---|---|
| Baseline（纯 T2I） | — | **48.02** |
| Staged: Language → T2I | 0.10 | 51.56 |
| Staged: Language → T2I | 0.20 | 52.76 |
| Staged: Language → T2I | 0.50 | 65.13 |
| Staged: Language → T2I（补满一个 T2I epoch，$1.5\times$ 算力） | 0.33 | 49.22 |

语言预训练占比越高，gFD 越差——因为挤占了图文对训练。有人会抬杠：这是预算被分走的缘故。于是最后一行做了对照：语言预训练之后**仍然补满整个 T2I epoch**（总预算 1.5 倍），gFD 49.2，依然不如 baseline 的 48.0。也就是说，即使不算"挤占预算"的账，先做语言预训练也带不来收益。

**干预 2：从 36T token 预训练好的 LLM 继续训练。**用 Qwen3-0.6B-Base（36T 纯文本预训练）初始化，扩词表加图像 token，在 GPIC 上继续训练（Qwen3-0.6B-CPT），与完全随机初始化的同架构模型对比：

| 模型 | 语言预训练 token | gFD ↓ |
|---|---|---|
| Qwen3-0.6B（随机初始化） | 0 | **46.61** |
| Qwen3-0.6B-CPT（预训练初始化） | 36T | 47.81 |
| Qwen3-0.6B-CPT（更低峰值学习率 $10^{-4}$） | 36T | 63.62 |

36T token 的语言预训练不仅没帮忙，反而略微拖了后腿；为了防止"太大学习率破坏了预训练权重"而降低峰值学习率，结果更差。训练曲线还显示，随机初始化的模型在整个预训练过程中图像 NLL 持续更低，而语言预训练初始化的只是文本 NLL 更低——**强大的纯文本建模能力并不能转化为图像生成的有用初始化**。

![随机初始化与语言预训练初始化的 NLL 对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/qwen3_init_nll.png)

> 图解：左图为以文本为条件的图像 token 验证 NLL（$\mathcal{I}\mid\mathcal{T}$），右图为无条件文本 token NLL（$\mathcal{T}\mid\varnothing$），对比随机初始化与语言预训练初始化的 Qwen3-0.6B。随机初始化的图像 NLL 全程更低，语言预训练初始化的文本 NLL 更低——两者各赢各的赛道，但 T2I 要的是前者。

为什么会这样？作者的解释很直白：纯文本预训练优化的是 $p(\mathcal{T}\mid\varnothing)$，而 T2I 需要的是 $p(\mathcal{I}\mid\mathcal{T})$，而且图像 token 还来自一个独立训练的 tokenizer——两套表征之间没有必然的可迁移性。FLAN-T5-XL 的 caption NLL 确实比 FLAN-T5-Base 低（2.76 vs. 3.24），算力多 13.5 倍，但 gFD 只从 66.7 改善到 63.3，与 Parti 当年的观察一致。

## 六、控制实验二：I2T 监督的妙用

解决了"语言预训练没用"之后，下一个问题自然浮现：既然预算省下来了，能不能顺便让模型学会别的——比如图像描述（image captioning）？

**先泼一盆冷水：把 I2T 当独立任务加进来，对 T2I 没帮助。**在固定预算下，无论 staged（先 I2T 后 T2I）还是 mixing（交织），I2T 占 10% 时 gFD 与 baseline 基本持平（48.4 / 48.2 vs. 48.0）；占 50% 时条件图像 token 减半，两者都变差（staged 54.2，mixing 76.0），尽管 mixing 的图像描述 NLL 大幅降低（1.009 vs. staged 的 20.437）。

**但把 caption dropout 换成 I2T，就是白捡的能力。**关键技巧：不再随机丢 caption，而是把原本会被丢 caption 的那 10% T2I 样本直接换成 I2T 样本（$p_{\mathrm{drop}}=0$，$\eta=0.1$）。由于两类序列共享 `[SOV]` + 图像 + `[EOV]` 前缀，**条件和无条件图像 token 的数量完全不变**，CFG 照常工作，同时还多了 I2T 监督。这个设计的聪明之处在于零成本——预训练预算、图像 token 统计量一概不动。

结果：在所有模型规模上，条件文本 NLL 下降超过一个数量级（0.52B 处 18.414 → 1.102），而 gFD、GenEval、DPG-Bench 与 baseline 基本持平；caption 对齐分数随规模稳步上升（0.1B 的 0.090 → 4B 的 0.559，GPIC 真实 caption 的对齐分数为 0.891）。

| 规模 | 配置 | $\mathcal{T}\mid\mathcal{I}$ NLL ↓ | gFD ↓ | GenEval ↑ | DPG-Bench ↑ | Caption 对齐 ↑ |
|---|---|---|---|---|---|---|
| 0.10B | baseline | 19.066 | 144.16 | 27.31 | 66.36 | — |
| 0.10B | + I2T | **1.410** | 144.14 | 27.66 | 68.89 | 0.090 |
| 0.52B | baseline | 18.414 | 48.02 | 44.91 | 76.84 | — |
| 0.52B | + I2T | **1.102** | 46.48 | 44.60 | 76.14 | 0.315 |
| 1.6B | baseline | 20.426 | 35.28 | 53.48 | 77.90 | — |
| 1.6B | + I2T | **0.977** | 35.91 | 53.51 | 78.21 | 0.467 |
| 4B | baseline | 16.738 | 28.43 | 56.39 | 77.90 | — |
| 4B | + I2T | **0.890** | 29.30 | 55.33 | 78.61 | 0.559 |

![Halcyon-4B 的图像描述示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/caption-with-gt-updated.png)

> 图解：Halcyon-4B 的图像描述示例。每张图上方是 GPIC 真实 caption，下方是模型生成的 caption。模型仅用 10% 的预训练预算（替换 caption dropout 而来）就学会了看图说话，生成描述在语义上与真实 caption 相当接近。

## 七、Scaling 行为、CFG 与 caption 长度

**Scaling。**从 0.1B 到 4B，图像与文本的验证 NLL 都随模型规模持续下降，且大模型在整个训练过程中始终保持更低的 NLL——两个模态的建模能力同步改善。

![Halcyon 各规模的验证 NLL scaling 曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/scaling_v2.png)

> 图解：Halcyon 在 GPIC 预训练过程中，图像 token（左）与文本 token（右）的验证 NLL 曲线。横轴为训练进程，纵轴为 NLL。从 0.1B 到 4B，两条曲线都随规模一致地下移，说明统一架构下容量增加对两种模态都有益。

**CFG 与模型规模的关系。**最优 CFG 强度随规模增大而降低：从 0.1B 的 7.5 降到 4B 的 4.5，对应的 gFD 从 142.7 改善到 28.4（验证集数字）。这与直觉一致——**更强的模型需要更少的外部引导**，不过 CFG 在所有规模上仍然有益。

![Halcyon 各规模的 CFG 强度扫描](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/cfg_sweep_v2.png)

> 图解：Halcyon 各规模在 GPIC Validation-50K 上的 CFG 强度扫描。横轴为引导强度 $\gamma$，纵轴为 gFD。星号标出每个模型的最优点：随着模型变大，最优 $\gamma$ 左移（7.5 → 4.5），gFD 曲线整体下移。基线模型的扫描同理进行。

**caption 长度的影响。**用 GPIC Test-1M 的短、中、长 caption 分别评测：

| 模型 | Test-50K | Short | Medium | Long |
|---|---|---|---|---|
| 0.1B | 144.16 | 149.03 | 139.27 | 147.85 |
| 0.25B | 72.55 | 77.18 | 70.14 | 72.53 |
| 0.52B | 48.02 | 51.99 | 46.12 | 46.83 |
| 1.6B | 35.28 | 38.59 | 34.21 | 34.33 |
| 4B | 28.43 | 31.48 | 27.91 | 27.77 |

所有规模下，中长 caption 的 gFD 都优于短 caption——**更详细的文本描述确实提升生成质量**；但长 caption 相比中 caption 没有稳定优势。这从侧面印证了模型真的在学文本条件，而不是把 caption 当噪声。

## 八、附录细节（复现向）

详细模式下，把附录里对复现最关键的信息也整理出来。

**Halcyon 模型配置。**五个尺寸均为 Qwen3 风格 decoder-only，tied embeddings，head dim 128：

| 配置 | 0.1B | 0.25B | 0.52B | 1.6B | 4B |
|---|---|---|---|---|---|
| 层数 | 14 | 21 | 28 | 28 | 36 |
| 隐藏维 | 512 | 768 | 1024 | 2048 | 2560 |
| FFN 维 | 1536 | 2304 | 3072 | 6144 | 9728 |
| Q / KV 头 | 8/4 | 12/6 | 16/8 | 16/8 | 32/8 |
| 总参数 | 96.4M | 247.7M | 523.0M | 1574.5M | 3840M |
| 峰值学习率 | 1e-3 | 7.5e-4 | 7.5e-4 | 7.5e-4 | 7.5e-4 |
| 最优 CFG | 7.5 | 6.0 | 6.0 | 5.0 | 4.5 |

统一超参：97,632 步（GPIC 一个 epoch），global batch 1024，1000 步 warmup，学习率先恒定到 49,595 步再 cooldown 到峰值的 1/10，weight decay 0.1，梯度裁剪 1.0，$\lambda_{\mathrm{img}}=0.5$，$p_{\mathrm{drop}}=0.1$。

**Tokenizer 细节。**图像 tokenizer 为 Cosmos 编码器-解码器 + FSQ，136M 参数，8 倍下采样，码本 64,000，在 GPIC 训练图像上训 1.5 个 epoch（242,172 步，batch 640，学习率 1e-4）。文本 BPE 词表 16,384，GPIC caption 平均 44.5 token（p50 36，p90 64，p99 182）。

**重建与压缩质量。**从零训练的图像 tokenizer 全面超过 Cosmos-DI-8x8：rFD 18.86 vs. 37.93，PSNR 25.61 vs. 25.38，SSIM 0.792 vs. 0.776，LPIPS 0.062 vs. 0.153（Real-vs-Real 参考 rFD 为 7.59）；与所有离散 tokenizer 一样，小字号文字和精细字形的重建仍是短板。文本 tokenizer 用词表只有 Qwen3 的 1/9.3，压缩率却相当（4.774 vs. 4.763 bytes/token），优于 FLAN-T5（4.152）。

**种子方差（评估结论可靠性的底气）。**训练种子：Halcyon-0.10B 用三个种子重训，gFD 的 std 为 5.54，其余样本质量指标稳定——这给出了"小模型之间 gFD 差多少才算真差异"的参考尺。采样种子：固定 checkpoint 换三个采样种子，各规模 gFD 的 std 最大仅 0.14，评估噪声可以忽略。

**推理成本。**在单台 8×H100 节点上用 vLLM 生成 GPIC Test-50K 全部 5 万张图（含模型加载与 CUDA graph 捕获，不含约 198 秒的图像解码）：0.1B 约 395 秒（126.5 张/秒），0.52B 约 663 秒，4B 约 936 秒（53.4 张/秒）。自回归 + vLLM 的吞吐相当可观。

**采样设置。**所有模型每张 caption 生成一张 $256\times256$ 图。Halcyon 在 caption 前缀 + `[SOV]` 之后自回归采样 1024 个图像 token，temperature 1.0，关闭 top-k/top-p，固定种子；CFG 只作用于图像 token logits，无条件分支用 `[SOV]` 前缀（无 caption）。基线用官方采样代码与设置，同样在验证集上选最优 CFG。

![tag 式 caption 的生成样例对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/collage_tags_v4.png)

> 图解：GPIC 四种 caption 类型中"tag"（标签式）prompt 的生成对比，从左到右为 JiT-T2I-XXL/8++、LlamaGen-XL/8 与 Halcyon 0.1B–4B。

![short caption 的生成样例对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/collage_short_v4.png)

> 图解："short"（短句）caption 的生成对比。随 Halcyon 规模增大，细节还原与 prompt 遵循明显改善。

![medium caption 的生成样例对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/collage_medium_v4.png)

> 图解："medium"（中等长度）caption 的生成对比。

![long caption 的生成样例对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Text-to-Image-Generative-Modeling-Without-Language-Pretraining/figures/collage_long_v4.png)

> 图解："long"（长描述）caption 的生成对比。大模型能捕捉服饰、物体属性与场景布局等更细粒度的要求；不过在 256×256 分辨率下，渲染清晰可读的文字对所有模型都仍是难题。

## 九、总结与展望

- **核心结论**：完全从零训练、不使用任何互联网规模语言预训练的 Halcyon，在 GPIC 上以远更低的总算力超越依赖预训练 LLM 的强晚期融合系统（0.52B 胜 LlamaGen-XL/8，1.6B 胜 JiT-T2I-XXL/8++）。
- **算力账**：语言预训练占主流 T2I 系统 80% 以上的预训练 FLOPs，却对生成质量贡献甚微；把这部分算力记账后，Halcyon-1.6B 的性价比是 JiT-T2I-XXL/8++ 的 275 倍。
- **控制实验**：固定预算下，纯文本预训练（无论是先验阶段还是从 36T token 的 Qwen3-0.6B 继续训练）都不改善 T2I 生成——$p(\mathcal{T}\mid\varnothing)$ 的强大不保证对 $p(\mathcal{I}\mid\mathcal{T})$ 有用。
- **免费午餐**：用 I2T 序列替换 caption dropout 样本，不改变任何图像 token 预算，就白得了图像描述能力，且 T2I 质量不降。
- **实践配方**：从零训练图像 tokenizer（FSQ + 感知/对抗损失）+ 小词表 BPE + 早期融合自回归 + GPIC 规模图文对，就是一套完整可复现的低成本 T2I 方案。

局限与方向：实验集中在 256×256 的 GPIC 单一数据集，受控干预主要用 0.5B–0.6B 规模、且聚焦自回归模型；更高分辨率、更大规模、Transfusion 式混合架构，以及"文生视频能否同样摆脱语言预训练"（Wan、CogVideoX、Mochi 都重度依赖大语言编码器），都是这个结论自然延伸出的开放问题。更广义地说，这项工作提示我们：在多模态生成里，"先预训练一个单模态底座再迁移"的惯性思维，值得用同样的控制实验重新审视。

> 本文参考自 [Text-to-Image Generative Modeling Without Language Pretraining](http://arxiv.org/abs/2610.26863v1)