# EAGLE-3：训练时测试提速 6.5 倍

大语言模型逐 Token 生成带来的高延迟，在需要长链推理的模型上尤其突出，但直接增加草稿模型的训练数据，却并不能让 EAGLE 持续提速。EAGLE-3 抓住了这个反常现象的根因：不再强迫草稿模型回归目标模型的特征，而是在训练时模拟多步生成，直接学习预测 Token。它进一步融合目标模型低、中、高层的特征，让草稿模型既能利用更丰富的信息，又能适应自己的预测误差。最终，EAGLE-3 最高实现 **6.5 倍加速**，相对 EAGLE-2 的延迟加速约提升 **1.4 倍**；在 SGLang、batch size 为 64 时，吞吐量仍提高 **38%**。

## 为什么草稿模型“吃不动”更多数据

### 自回归解码的成本，正在被推理模型放大

自回归生成像一场必须按顺序完成的接力：生成下一个 Token 前，必须等前一个 Token 确定。每向前走一步，都要读取大模型的全部参数，因此参数规模越大，解码的显存访问和计算成本越高。

量化、蒸馏等方法可以压缩单次计算，却往往需要在模型质量与速度之间权衡。Speculative Sampling（投机采样）提供了另一条路径：先用便宜的草稿模型连续猜几个 Token，再让目标模型并行验收，从而在不改变目标模型输出分布的前提下减少串行等待。

EAGLE 则让小模型“站在大模型肩膀上”猜测：它复用目标模型 LM Head 前的顶层特征，在特征空间做自回归预测，再借用目标模型的 LM Head 得到草稿 Token。相比完全独立的小模型，这种做法显著提高了接受率。

问题在于，当作者沿用大模型领域“扩大训练数据提升能力”的思路训练 EAGLE 时，加速比很快趋于饱和。要理解这个瓶颈，需要拆开 EAGLE 的训练目标。

### 特征回归既提供帮助，也成为约束

EAGLE 同时优化两类 Loss：

- 特征预测 Loss $l_{\mathrm{fea}}$：要求草稿模型的输出逼近目标模型顶层特征；
- Token 预测 Loss $l_{\mathrm{token}}$：要求经过 LM Head 后的分布预测正确 Token。

$l_{\mathrm{fea}}$ 的好处是让只训练一步的模型，也能在多步预测时以“模拟目标特征”为中间桥梁。但预测 Token 才是最终目的，额外拟合特征相当于要求模型同时完成一项并不必要的回归任务，限制了表达能力。

直接删掉 $l_{\mathrm{fea}}$ 也不是完整答案。第一步预测通常会变好，可草稿模型随后要把自己输出的自由向量 $\hat a_{t+1}$ 当作输入；训练时它却只见过目标模型的真实特征，于是第二步输入偏离训练分布，接受率明显下降。

![EAGLE、移除特征约束与 EAGLE-3 的训练测试差异](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/nofe.png)

> 图解：上部的 EAGLE 用 $l_{\mathrm{fea}}$ 约束每一步特征预测；中部直接移除该约束后，第一步尚可，第二步因输入分布变化而失准；底部的 EAGLE-3 把第一步输出 $\hat a_{t+1}$ 回传给第二步训练，让训练和真实多步测试走同一条路径。$f$ 表示目标模型特征，$a$ 表示不受特征回归约束的草稿模型输出，$t$ 表示 Token，带帽符号表示模型预测。

数据扩展实验把这个矛盾展示得更清楚。下面的横轴均为相对 ShareGPT 的训练数据规模，$0\text{-}\alpha$ 是第一个草稿 Token 的接受率，$1\text{-}\alpha$ 是输入已含一个模型自预测值时的接受率。

![首个草稿 Token 接受率随数据规模变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/alpha0.png)

> 图解：仅移除特征预测的黄线让 $0\text{-}\alpha$ 随数据增加持续上升，并从 2 倍数据规模起高于 EAGLE；EAGLE-3 的蓝线在小数据时起点较低，但从 4 倍数据规模起超过 EAGLE，8 倍时约为 0.80，说明完整方法可以持续从数据中获益。

![第二个草稿 Token 接受率随数据规模变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/alpha1.png)

> 图解：黄线虽改善了首个 Token，却让 $1\text{-}\alpha$ 降到约 0.20—0.32，暴露了多步输入失配；EAGLE-3 的 $1\text{-}\alpha$ 从约 0.70 增至约 0.79，始终高于 EAGLE。两张图合起来说明：真正有效的不是简单删 Loss，而是删除约束后再训练模型适应自身输出。

因此，EAGLE-3 的核心问题被重新定义为：**如何让草稿模型摆脱特征模仿，同时在多步生成中保持稳定？** 在给出答案前，先补齐它与 Speculative Sampling、EAGLE-2 共用的基础框架。

## 从草稿到验证：无损加速如何成立

### Speculative Sampling：小模型猜，大模型验收

设 $t_i$ 为第 $i$ 个 Token，$T_{a:b}$ 表示从 $t_a$ 到 $t_b$ 的序列，已有前缀为 $T_{1:j}$。标准流程分为两步：

1. **Drafting**：草稿模型自回归生成 $k$ 个 Token，形成 $\hat T_{j+1:j+k}$，并记录每个草稿 Token 的概率 $\hat p$。
2. **Verification**：目标模型一次并行计算整段草稿的概率 $p$，再从左到右依次验收。

对第 $i$ 个草稿 Token $\hat t_{j+i}$，接受概率为：

$$
\min\left(1,\frac{p_{j+i}(\hat t_{j+i})}{\hat p_{j+i}(\hat t_{j+i})}\right)
$$

如果拒绝，就从归一化的差分分布 $\operatorname{norm}(\max(0,p_{j+i}-\hat p_{j+i}))$ 中采样替代 Token，并丢弃其后的草稿。这个接受—修正规则保证最终分布与目标模型普通自回归解码一致，而不是“看起来差不多”的有损近似。

### EAGLE 与 EAGLE-2：从特征自回归到动态草稿树

EAGLE 在特征层做自回归，并把上一时间步实际采样出的 Token Embedding 一并输入草稿模型，以消除 Token 采样结果的随机性。它还会在同一位置生成多个候选，形成树状草稿，再通过 Tree Attention 并行验证。

静态树无法适应上下文难度：简单片段上分支太少，困难片段上又可能浪费候选。EAGLE-2 因而用草稿模型的置信度近似接受率，动态扩展并剪枝草稿树。EAGLE-3 保留这一机制，改造重点放在草稿模型本身。

## EAGLE-3 的两项关键改造

### 融合多层特征：不只盯着“下一 Token”

EAGLE 和 Medusa 通常只使用 LM Head 前的顶层特征。对于满秩 LM Head，这些特征与下一 Token 的 logits 一一对应，信息天然偏向“下一个 Token 是什么”；但草稿模型还要预测更后面的 Token，只看顶层并不充分。

EAGLE-3 在目标模型 forward 时，同时记录低层、中层和高层特征 $l_i,m_i,h_i$。三者各为 $k$ 维，拼接后经过全连接层压回 $k$ 维：

$$
g_i=\operatorname{FC}\left([l_i;m_i;h_i]\right),\qquad [l_i;m_i;h_i]\in\mathbb{R}^{3k},\quad g_i\in\mathbb{R}^{k}
$$

其中 $k$ 是目标模型的 hidden size。$g_i$ 同时保留不同层级的语义信息，而移除特征回归约束后，草稿模型也不必再把输出伪装成某一层的目标特征。

笔者认为，这里最值得注意的不是“多用几层特征”本身，而是两项改造的依赖关系：**正因为取消了特征对齐，输入才有自由从单一顶层扩展到多层融合。**

### 推理流水线：真实特征起步，自预测向量续跑

以“How can”为前缀，目标模型先在 prefill 或上一轮 verification 中生成“I”，同时留下“How”“can”等位置的 $l,m,h$，融合得到 $g$。草稿阶段如下：

1. **Step 1**：复用 $g_{\mathrm{how}}$ 和 $g_{\mathrm{can}}$，并加入已采样 Token“I”的 Embedding $e_{\mathrm I}$。拼接后的输入经 FC 降维，再进入一个 Transformer Decoder Layer，输出不受特征回归约束的 $a$；$a_{\mathrm I}$ 经目标模型 LM Head 采样出“do”。
2. **Step 2**：前缀变为“How can I”，但“I”尚未经过目标模型验证，无法取得 $g_{\mathrm I}$。于是用上一轮的 $a_{\mathrm I}$ 替代它，并拼接新采样 Token“do”的 Embedding，继续预测“it”。
3. **Step 3 及以后**：同理，用 $a_{\mathrm{do}}$ 替代不可获得的 $g_{\mathrm{do}}$，加入 $e_{\mathrm{it}}$ 后继续向后生成。

![EAGLE-3 推理流水线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/method.png)

> 图解：左侧目标模型的参数被冻结，雪花标记表示不参与草稿训练；低、中、高特征 $l,m,h$ 经拼接和 FC 得到 $g$。右侧展示三步草稿：已有上下文使用目标模型的 $g$，新生成位置改用草稿模型自身的 $a$，每个位置还拼入下一实际 Token 的 Embedding $e$，最后共享目标模型的 LM Head 产生候选。

这个设计也明确了训练要求：草稿模型必须同时见过“目标特征 $g$”和“自己的输出 $a$”，否则训练与推理仍会在第二步以后分叉。下一项改造正是为此而来。

## Training-Time Test：把多步测试搬进训练

Training-Time Test 可以直译为“训练时测试”。它不是在训练集之外额外做一次评测，而是在训练过程中实际生成 $a$，再把 $a$ 回传给草稿模型，构造后续步骤并继续计算 Token 预测 Loss。这样，模型在训练时就经历了推理中的输入分布。

草稿模型主体只有一层 Transformer Decoder。除 Self-Attention 外，其他模块无需特殊处理；真正需要修改的是 Attention Mask，因为多个草稿分支的上下文不是简单的一条直线。

原始训练序列“How can I”使用标准下三角 Mask。三个位置分别预测“are”“we”“do”后，第二轮得到三个分支：

- “are”只能看“How”和自身；
- “we”只能看“How can”和自身；
- “do”只能看“How can I”和自身。

第三轮再沿各分支生成“you”“help”“it”时，每个 Query 只能关注原始前缀、自己的父节点和自身，不能关注旁支。这样既模拟多步生成，也保留不同位置并行训练的效率。

![Training-Time Test 的因果注意力 Mask](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/train.png)

> 图解：灰色是原始训练 Token，蓝色和黄色分别是第一、第二轮预测。矩阵行表示 Query、列表示 Key，红勾表示允许关注。左上是普通下三角 Mask；右上和下方逐步变为分支感知的 Mask，防止不同草稿路径互相看见。除访问原始训练数据的区域外，需要计算的位置主要位于对应对角线上，可用向量点积代替完整矩阵乘法，减少无效计算。

HASS 也在训练中模拟多步草稿，但两者目标不同。HASS 仍保留 $l_{\mathrm{fea}}$，通过减少不准确特征的误差累积来改善 EAGLE；EAGLE-3 则彻底取消特征拟合，让输出不再追随目标顶层特征，并由此获得多层输入自由度和数据扩展能力。换句话说，前者是在修正特征预测，后者是在取消这项任务。

## 数据越多，加速是否还能继续增长

方法改造完成后，作者在 MT-bench 上以 LLaMA-Instruct 3.1 8B 为目标模型，把训练数据从 ShareGPT 的 1 倍扩展到 2、4、8 倍。结果不再只是首个 Token 接受率提高，而是端到端加速和平均接受长度同步增长。

![加速比随草稿训练数据规模变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/scaale.png)

> 图解：横轴是相对 ShareGPT 的数据规模，纵轴是相对普通自回归解码的 Speedup。EAGLE-2 从约 3.16 倍增至约 3.30 倍后基本饱和；EAGLE-3 则从约 3.70 倍持续升至约 4.40 倍，没有出现同样的平台期。

![平均接受长度随草稿训练数据规模变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/scaletau.png)

> 图解：纵轴是平均接受长度 $\tau$。EAGLE-2 从约 4.05 缓慢增至约 4.15；EAGLE-3 从约 5.2 增至约 6.1。两条曲线共同支持论文提出的“推理加速 Scaling Law”：草稿模型扩大训练数据后，最终解码效率仍能提高。

接下来要验证的是，这种优势能否跨任务、跨目标模型成立，以及它能否走出单请求测试，在真实推理框架的大 batch 场景中保留收益。

## 实验设置：五类任务与三项指标

### 模型、任务与训练配置

实验覆盖 Chat 和 Reasoning 两类目标模型：Vicuna 13B、LLaMA-Instruct 3.1 8B、LLaMA-Instruct 3.3 70B，以及 DeepSeek-R1-Distill-LLaMA 8B。所有任务共用同一套权重，不针对任务单独微调。

| 任务 | 数据集 |
| --- | --- |
| 多轮对话 | MT-bench |
| 代码生成 | HumanEval |
| 数学推理 | GSM8K |
| 指令遵循 | Alpaca |
| 文本摘要 | CNN/Daily Mail |

草稿模型使用 AdamW，$(\beta_1,\beta_2)=(0.9,0.95)$，学习率为 $5\times10^{-5}$，梯度裁剪为 0.5。训练数据包括约 68K 条 ShareGPT 和 464K 条 UltraChat-200K；回答由目标模型现场生成，而不是直接采用固定数据集中的答案。训练 DeepSeek-R1-Distill-LLaMA 8B 的草稿模型时，还加入 OpenThoughts-114k-math。

### 指标为什么只看速度，不看生成质量

EAGLE-3 不修改目标模型权重，并使用严格的 Speculative Sampling 接受条件，因此理论输出分布不变，无需重新评估回答质量。论文采用三项指标：

- **Speedup Ratio**：相对普通自回归解码的实际加速比，基线为 1.00 倍；
- **平均接受长度 $\tau$**：每轮 drafting—verification 平均生成的 Token 数；
- **$n\text{-}\alpha$ 接受率**：此前草稿均被接受时，输入含 $n$ 个自预测值后的下一 Token 接受率，测试时使用链式草稿。

这里的 $n$ 很关键：$n$ 越大，草稿模型越依赖自己的历史输出，越能检验多步误差是否累积。

## 主实验：不同模型和任务上全面提速

下图在 Temperature=0 时比较代表性方法。Chat 模型使用 MT-bench，Reasoning 模型使用 GSM8K；标准 Speculative Sampling 在 Vicuna-13B 上以 Vicuna-68M 为草稿模型。

![四类目标模型上的代表方法加速比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/r1.png)

> 图解：横轴列出四种目标模型，纵轴为 Speedup，普通自回归 Vanilla 均为 1.0 倍。EAGLE-3 在 Vicuna 13B、LLaMA-Instruct 3.1 8B、LLaMA-Instruct 3.3 70B 和 DeepSeek R1 LLaMA 8B 上分别约为 5.6、4.4、4.1、5.0 倍，均高于同组其他方法。

完整结果如下。每格写作“Speedup / $\tau$”，保留五类任务及均值，便于同时观察实际速度和草稿接受情况。

**Temperature=0：**

| 目标模型 | 方法 | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vicuna 13B | SpS | 1.93× / 2.27 | 2.23× / 2.57 | 1.77× / 2.01 | 1.76× / 2.03 | 1.93× / 2.33 | 1.92× / 2.24 |
| Vicuna 13B | PLD | 1.58× / 1.63 | 1.85× / 1.93 | 1.68× / 1.73 | 1.16× / 1.19 | 2.42× / 2.50 | 1.74× / 1.80 |
| Vicuna 13B | Medusa | 2.07× / 2.59 | 2.50× / 2.78 | 2.23× / 2.64 | 2.08× / 2.45 | 1.71× / 2.09 | 2.12× / 2.51 |
| Vicuna 13B | Lookahead | 1.65× / 1.69 | 1.71× / 1.75 | 1.81× / 1.90 | 1.46× / 1.51 | 1.46× / 1.50 | 1.62× / 1.67 |
| Vicuna 13B | Hydra | 2.88× / 3.65 | 3.28× / 3.87 | 2.93× / 3.66 | 2.86× / 3.53 | 2.05× / 2.81 | 2.80× / 3.50 |
| Vicuna 13B | EAGLE | 3.07× / 3.98 | 3.58× / 4.39 | 3.08× / 3.97 | 3.03× / 3.95 | 2.49× / 3.52 | 3.05× / 3.96 |
| Vicuna 13B | EAGLE-2 | 4.26× / 4.83 | 4.96× / 5.41 | 4.22× / 4.79 | 4.25× / 4.89 | 3.40× / 4.21 | 4.22× / 4.83 |
| Vicuna 13B | **EAGLE-3** | **5.58× / 6.65** | **6.47× / 7.54** | **5.32× / 6.29** | **5.16× / 6.17** | **5.01× / 6.47** | **5.51× / 6.62** |
| LLaMA 3.1 8B | EAGLE-2 | 3.16× / 4.05 | 3.66× / 4.71 | 3.39× / 4.24 | 3.28× / 4.12 | 2.65× / 3.45 | 3.23× / 4.11 |
| LLaMA 3.1 8B | **EAGLE-3** | **4.40× / 6.13** | **4.85× / 6.74** | **4.48× / 6.23** | **4.82× / 6.70** | **3.65× / 5.34** | **4.44× / 6.23** |
| LLaMA 3.3 70B | EAGLE-2 | 2.83× / 3.67 | 3.12× / 4.09 | 2.83× / 3.69 | 3.03× / 3.92 | 2.44× / 3.55 | 2.85× / 3.78 |
| LLaMA 3.3 70B | **EAGLE-3** | **4.11× / 5.63** | **4.79× / 6.52** | **4.34× / 6.15** | **4.30× / 6.09** | **3.27× / 5.02** | **4.12× / 5.88** |
| DeepSeek-R1-Distill-LLaMA 8B | EAGLE-2 | 2.92× / 3.80 | 3.42× / 4.29 | 3.40× / 4.40 | 3.01× / 3.80 | 3.53× / 3.33 | 3.26× / 3.92 |
| DeepSeek-R1-Distill-LLaMA 8B | **EAGLE-3** | **4.05× / 5.58** | **4.59× / 6.38** | **5.01× / 6.93** | **3.65× / 5.37** | **3.52× / 4.92** | **4.16× / 5.84** |

**Temperature=1：**

| 目标模型 | 方法 | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vicuna 13B | SpS | 1.62× / 1.84 | 1.72× / 1.97 | 1.46× / 1.73 | 1.52× / 1.78 | 1.66× / 1.89 | 1.60× / 1.84 |
| Vicuna 13B | EAGLE | 2.32× / 3.20 | 2.65× / 3.63 | 2.57× / 3.60 | 2.45× / 3.57 | 2.23× / 3.26 | 2.44× / 3.45 |
| Vicuna 13B | EAGLE-2 | 3.80× / 4.40 | 4.22× / 4.89 | 3.77× / 4.41 | 3.78× / 4.37 | 3.25× / 3.97 | 3.76× / 4.41 |
| Vicuna 13B | **EAGLE-3** | **4.57× / 5.42** | **5.15× / 6.22** | **4.71× / 5.58** | **4.49× / 5.39** | **4.33× / 5.72** | **4.65× / 5.67** |
| LLaMA 3.1 8B | EAGLE-2 | 2.44× / 3.16 | 3.39× / 4.39 | 2.86× / 3.74 | 2.83× / 3.65 | 2.44× / 3.14 | 2.80× / 3.62 |
| LLaMA 3.1 8B | **EAGLE-3** | **3.07× / 4.24** | **4.13× / 5.82** | **3.32× / 4.59** | **3.90× / 5.56** | **2.99× / 4.39** | **3.45× / 4.92** |
| LLaMA 3.3 70B | EAGLE-2 | 2.73× / 3.51 | 2.89× / 3.81 | 2.52× / 3.36 | 2.77× / 3.73 | 2.32× / 3.27 | 2.65× / 3.54 |
| LLaMA 3.3 70B | **EAGLE-3** | **3.96× / 5.45** | **4.36× / 6.16** | **4.17× / 5.95** | **4.14× / 5.87** | **3.11× / 4.88** | **3.95× / 5.66** |
| DeepSeek-R1-Distill-LLaMA 8B | EAGLE-2 | 2.69× / 3.41 | 3.01× / 3.82 | 3.16× / 4.05 | 2.64× / 3.29 | 2.35× / 3.13 | 2.77× / 3.54 |
| DeepSeek-R1-Distill-LLaMA 8B | **EAGLE-3** | **3.20× / 4.49** | **3.77× / 5.28** | **4.38× / 6.10** | **3.16× / 4.30** | **3.08× / 4.27** | **3.52× / 4.89** |

Temperature=1 的表格没有列入 Medusa 等放宽非贪心接受条件的方法，因为它们不再严格保证无损加速，不适合与 EAGLE-3 直接比较。

整体看，EAGLE-3 相对普通解码约加速 3.0—6.5 倍，相对 EAGLE-2 提升约 20%—40%。HumanEval 上表现尤其突出，因为代码包含较多固定模板，草稿更容易命中；DeepSeek-R1-Distill-LLaMA 8B 则在 GSM8K 上最快，作者认为这可能与额外使用数学数据训练有关。

## 多步接受率：优势不是靠单步猜准

单步接受率无法解释长草稿的稳定性，因此作者在 MT-bench、LLaMA-Instruct 3.1 8B 上继续测试 $n\text{-}\alpha$。

![EAGLE 与 EAGLE-3 的多步接受率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/figs-logo.png-EAGLE-3-Scaling-up-Inference-Acceleration-of-Large-Language-Models-via-Training-Time-Test/figs/alpha.png)

> 图解：横轴从 $0\text{-}\alpha$ 到 $7\text{-}\alpha$，表示输入中自预测值逐步增多；纵轴为接受率。EAGLE 从约 0.71 降到 0.51—0.53 附近，而 EAGLE-3 始终约为 0.78—0.81。后者没有随草稿变长出现明显衰减，直接验证了 Training-Time Test 对多步一致性的作用。

平均接受长度和端到端 Speedup 都已提高，下一步还需拆分收益来源：究竟来自删除特征约束，还是来自多层特征融合？

## 消融实验：两项改造都不可缺少

作者以 LLaMA-Instruct 3.1 8B 为目标模型，从 EAGLE-2 出发依次加入两项改动。

| 方法 | MT-bench Speedup | MT-bench $\tau$ | GSM8K Speedup | GSM8K $\tau$ |
| --- | --- | --- | --- | --- |
| EAGLE-2 | 3.16× | 4.05 | 3.39× | 4.24 |
| + 移除特征预测约束 | 3.82× | 5.37 | 3.77× | 5.22 |
| + 多层融合特征（EAGLE-3） | **4.40×** | **6.13** | **4.48×** | **6.23** |

第一步已经带来明显提升，说明特征回归确实是瓶颈；加入多层融合后又进一步上升，说明更丰富的输入信息并非附带装饰。两项改动分别解决“输出受约束”和“输入信息不足”。

## 生产框架：大 batch 下还能否提高吞吐量

单请求加速并不自动等于服务吞吐提升。小 batch 解码常受显存带宽限制，还有空闲算力可用于验证草稿；batch 增大后，冗余算力减少，验证成本可能抵消收益。生产框架本身的高度优化也会压缩投机采样的空间。

### SGLang：batch size 64 仍有 1.38 倍吞吐

SGLang 团队在单张 H100、LLaMA-Instruct 3.1 8B、MT-bench 上测试 SGLang v0.4.4。该实验不用树结构，链长设为 3；不开启 Speculative Sampling 的 SGLang 为 1.00 倍基线。

| Batch size | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 56 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EAGLE | 1.40× | 1.38× | 1.23× | 1.02× | 0.93× | 0.94× | 0.88× | 0.99× | 0.99× |
| EAGLE-3 | **1.81×** | **1.82×** | **1.62×** | **1.48×** | **1.39×** | **1.32×** | **1.38×** | **1.34×** | **1.38×** |

EAGLE 从 batch size 24 起已降低吞吐，EAGLE-3 到 64 仍有 38% 增益。这说明更高的接受率不仅减少延迟，也降低了大 batch 下无效验证的代价。

同一团队还测试了 batch size 1 的单请求吞吐：

| 方法 | 吞吐量（batch size=1） |
| --- | --- |
| SGLang，无 Speculative Sampling，1×H100 | 158.34 tokens/s |
| SGLang + EAGLE-2，1×H100 | 244.10 tokens/s |
| SGLang + EAGLE-3，1×H100 | **373.25 tokens/s** |

### vLLM：收益区间向更大 batch 延伸

作者还在 vLLM 上测试 LLaMA-Instruct 3.1 8B、MT-bench。实验不用树结构，最大链长为 2，无 Speculative Sampling 的 vLLM 为 1.00 倍基线。

| Batch size | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 56 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EAGLE | 1.30× | 1.25× | 1.21× | 1.10× | 1.03× | 0.93× | 0.82× | 0.71× |
| EAGLE-3 | **1.75×** | **1.68×** | **1.58×** | **1.49×** | **1.42×** | **1.36×** | **1.21×** | **1.01×** |

EAGLE-3 在 batch size 56 时刚好持平基线，而 EAGLE 已降至 0.71 倍。需要指出的是，原论文正文将硬件写作 RTX 3090，表注则写作 A100，两处存在不一致；这里保留这一事实，不据此推断具体硬件。

## 与相关方法的关系

大模型推理加速大体有三条路线：

- **量化与蒸馏**：压缩模型或计算精度，通常要重新权衡质量与速度；
- **复用目标模型信息**：EAGLE、Medusa、Hydra 复用特征，GLIDE、CAPE 复用 KV Cache；
- **复用部分计算或参数**：Draft & Verify、Layer Skipping、Early Exit 等方法减少草稿成本。

EAGLE-3 属于第二类，并继续使用目标模型验证来保持无损。它与 HASS 都关注训练—推理不一致，但只有 EAGLE-3 同时放弃特征回归、引入多层特征，并把数据扩展转化为持续加速。EAGLE 的特征自回归还曾启发 DeepSeek-V3 预训练中的 Multi-Token Prediction，而后者又反过来启发了 EAGLE-3 的结构设计。

## 复现时需要注意的实现细节

普通自回归基线使用 Hugging Face Transformers 的 PyTorch 后端，并预分配 KV Cache；标准 Speculative Sampling 使用该库的 assisted generation。PLD、Lookahead、Medusa 和 Hydra 采用官方默认设置与公开权重，EAGLE 的 Vicuna、LLaMA2-Chat 草稿模型也用官方权重，LLaMA3-Instruct 则与其他方法一致使用 ShareGPT 训练。

EAGLE-2 对 7B/8B、13B、70B 目标模型分别设置 60、50、48 个草稿 Token，树深均为 6，扩展阶段选择 10 个节点。EAGLE-3 因接受率更高，在保持节点数不变的情况下把树深增加到 8，让更深的草稿路径有机会被利用。

## 总结：从“模仿特征”转向“学会起草”

- EAGLE 的瓶颈不只是模型容量，而是特征回归 Loss 限制了草稿模型的表达能力，使增加数据的收益快速饱和。
- EAGLE-3 用 Training-Time Test 在训练中回传自身输出，直接预测 Token，消除了多步草稿的输入分布失配。
- 低、中、高层特征融合补足顶层特征偏向下一 Token 的信息局限，与取消特征约束形成相互依赖的两项设计。
- 五类任务、四种目标模型上，EAGLE-3 最高加速 6.5 倍，并在多步接受率、消融实验和大 batch 吞吐测试中得到一致验证。
- 更重要的是，它展示了一条推理加速的 Scaling 路径：增加草稿训练数据，仍能持续提高接受长度与实际 Speedup。

目前实验尚未覆盖 405B 和 671B 模型，作者也预计更大数据规模还能继续提速；这一趋势能否在更大模型、更多框架和更广任务上保持，是下一步值得验证的方向。

> 本文参考自 [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)