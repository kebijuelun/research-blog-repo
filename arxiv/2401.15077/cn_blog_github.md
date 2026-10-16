# EAGLE：特征外推让 LLM 推理提速 3 倍

大语言模型（LLM）逐 token 生成的自回归解码方式，让推理又慢又贵——每次前向传播只能产出一个 token。投机采样（Speculative Sampling）用"小模型起草 + 大模型并行验证"的思路破局，但草稿质量一直是瓶颈。EAGLE 这篇工作重新审视了投机采样，提出一个反直觉的关键洞察： **与其让草稿模型直接预测 token，不如让它在"特征层"（LM Head 之前的倒数第二层隐藏状态）做自回归外推，再把超前一个时间步的 token 序列作为输入，消除采样带来的不确定性** 。效果非常硬：在 LLaMA2-Chat 70B 上实现 **2.7x-3.5x** 的延迟加速、吞吐量翻倍，且理论上保证输出分布与原模型完全一致——也就是说，加速是无损的。

## 提出问题：自回归解码为什么慢，投机采样卡在哪

LLM 生成文本是一个 token 一个 token 串行吐出来的，每次前向传播都要把全部权重读一遍，却只能换来一个 token，GPU 算力大量闲置。这是典型的 memory-bound（访存受限）场景。

投机采样把生成拆成两步：

1. **起草（Drafting）** ：用一个便宜的小模型快速猜出 $\gamma$ 个候选 token；
2. **验证（Verification）** ：目标大模型一次前向传播并行验证这批候选，接受的留下，拒绝的丢弃重采样。

妙处在于：验证阶段的接受/拒绝规则（接受概率为 $\min(1, p(\hat{t})/\hat{p}(\hat{t}))$，拒绝后从修正分布 $\text{norm}(\max(0, p - \hat{p}))$ 重采样）在数学上保证了输出分布与目标 LLM 完全一致。加速是"免费"的，不改结果。

那问题出在哪？ **草稿模型不好找、也不好训** 。想用 LLaMA2-7B 给 70B 当草稿还行，但谁来给 7B 当草稿？TinyLLaMA 的指令模板与 LLaMA2-Chat 不一致，用不了；专门训一个小模型，动辄要几千 B 的 token（TinyLLaMA 用了 3000B），成本太高。另一条路是压低草稿开销：Lookahead 用 n-gram + Jacobi 迭代，Medusa 用一组 MLP 基于大模型的特征直接预测后续 token，但它们牺牲了草稿精度——Medusa 的接受率约 0.6，Lookahead 更低，加速效果被草稿质量卡住了天花板。

笔者认为，这篇文章最漂亮的地方就在于：它没有继续在"怎么压开销"上卷，而是回头问了一个更本质的问题—— **草稿模型到底该预测什么？**

## 分析问题：两个关键观察

### 观察一：特征级自回归比 token 级更简单

这里的"特征"指目标 LLM 倒数第二层的隐藏状态，即 LM Head 之前的高维连续向量。

直觉上可以这样理解：token 是离散的自然语言符号，同一句话可以有千变万化的表达；而特征序列是模型内部"消化后"的语义表示，规律性更强。用一个比喻：让草稿模型直接猜"下一个词"就像猜一个人下一句话说什么，而猜"下一个特征"更像猜他下一步的"思路走向"——后者的空间更连续、更平滑，自然更好学。

实验也证实了这一点：在 Vicuna 7B 上，基于特征做自回归的草稿模型加速比达到 1.9x，而基于 token 的只有 1.5x（见下方图 3）。

### 观察二：采样带来的不确定性，限制了特征预测的上限

但特征级自回归有一个隐藏坑。文本生成时，目标 LLM 先输出一个分布，再 **采样** 出具体 token——这一步有随机性。而特征是高维连续向量，没法像 token 那样"采样"，只能回归。

这就产生了歧义：如下图所示，在特征 $f_I$（对应 token "I"）之后，可能采样出 "am"（概率 0.6）也可能采样出 "always"（概率 0.4）。这两条分支通向完全不同的后续特征 $f_{am}$ 和 $f_{always}$。 **仅凭 $f_I$，草稿模型根本不知道该往哪条路走。**

![特征序列中的不确定性](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/fig2.png)

> 图解：中间的 $f_I$ 是 token "I" 对应的特征。$p_I$ 显示下一步采样有 0.6 概率得到 "am"、0.4 概率得到 "always"，两个分支分别产生不同的特征 $f_{am}$ 和 $f_{always}$，进而导出不同的后续分布（"am" 后大概率接 "ready"，"always" 后大概率接 "begin"）。红色虚线的 "sampling" 箭头表明：分叉发生在采样这一步，特征本身无法预知结果。

Medusa 也踩了同样的坑：它用 $f_I$ 直接预测后续 token，训练目标在 $p_{am}$ 和 $p_{always}$ 之间摇摆，精度自然上不去。

EAGLE 的解法简洁得令人拍案： **把超前一个时间步的 token 序列也喂给草稿模型** 。采样结果已经确定性地写在 token 里了——预测 $f_{always}$ 时，输入是 $(f_I, t_{always})$；预测 $f_{am}$ 时，输入是 $(f_I, t_{am})$。歧义被彻底消除，而代价几乎为零。

两个观察叠加的效果，用一张图说话：

![token、feature 与 feature&shifted-token 三种草稿模型的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/ob12.png)

> 图解：横轴为训练 Epoch，左图纵轴为加速比（Speedup），右图纵轴为草稿准确率（Acc），测试在 Vicuna 7B + MT-bench、temperature=0 下进行。三条曲线中，纯 token（橙色）最差（加速约 1.5x）；纯 feature（绿色）明显更好（约 1.9x）；feature & shifted-token（蓝色，即 EAGLE）一骑绝尘，收敛后加速约 2.8x、准确率约 0.8——远超 Medusa 的 0.6 左右。注意蓝线相比绿线的提升纯粹来自"token 前移一步"这一个改动，模型复杂度没有任何增加。

分析到这里，问题就清晰了：草稿要在特征层做，且必须把采样结果（即已生成的 token）作为输入告诉草稿模型。下一节看 EAGLE 具体怎么落地。

## 解决问题：EAGLE 的方法设计

### 起草阶段：特征 + 前移 token 的自回归外推

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）的草稿模型只包含三个模块：

- **Embedding 层** ：直接复用目标 LLM 的参数，冻结不训练；
- **LM Head** ：同样复用目标 LLM 的参数，冻结；
- **Autoregression Head（自回归头）** ：唯一需要训练的部分，仅由一个 FC 层 + 一个 Transformer Decoder 层组成。

前向过程是：输入特征序列 $F_{1:i}$（形状 (bs, seq\_len, hidden\_dim)）与前移一步的 token 序列 $T_{2:i+1}$；token 经 Embedding 后与特征拼接成 2 倍维度的融合序列；FC 层把维度压回 hidden\_dim；Decoder 层预测下一个特征 $\hat{f}_{i+1}$；再经冻结的 LM Head 得到分布并采样出下一个 token；最后把新特征和新 token 拼回输入，继续自回归。

![EAGLE 的整体流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/fig1_space.png)

> 图解：上半部分是计算流程，下半部分是对应的生成结果。左侧是目标 LLM：处理 "How can" 后产出特征 $f_{how}$、$f_{can}$，并采样出 token "I"（蓝色模块带雪花图标，表示复用目标 LLM 参数、不训练）。右侧是草稿模型：绿色块是 token embedding，橙色块是特征，红框是草稿模型的预测。注意每个位置的输入都是"上一步特征 + 当前已确定的 token embedding"的组合——这正是 shifted-token 设计。草稿模型只用 3 次前向传播（Forward 1/2/3），配合树注意力（Tree Attention）就展开出一棵 10 个 token 的草稿树（"I can make a..." 与 "I can help you..." 等多个分支）。

与各家方法对比，差异一目了然：投机采样和 Lookahead 是"用 token 预测 token"，Medusa 是"用特征独立预测若干 token"，而 EAGLE 是"用特征 + 前移 token 预测特征"，三者中唯一把采样随机性纳入输入的方案。

### 训练目标：回归 + 分类双损失

预测下一个特征本质上是回归任务，用 Smooth L1 损失：

$$
L_{reg} = \text{Smooth L1}(f_{i+1},\ \text{Draft\_Model}(T_{2:i+1}, F_{1:i}))
$$

但特征只是中间目标，最终要的还是能生成正确 token 的分布，所以再加一个分类损失直接优化最终目标：

$$
p_{i+2} = \text{Softmax}(\text{LM\_Head}(f_{i+2})),\quad \hat{p}_{i+2} = \text{Softmax}(\text{LM\_Head}(\hat{f}_{i+2}))
$$

$$
L_{cls} = \text{Cross\_Entropy}(p_{i+2},\ \hat{p}_{i+2})
$$

总损失为 $L = L_{reg} + w_{cls} L_{cls}$。由于分类损失在数值上比回归损失大约一个数量级，作者取 $w_{cls} = 0.1$ 做平衡。

还有两个实用的训练技巧：

- **固定数据集即可** ：理想情况下草稿模型应该用目标 LLM 自己生成的文本来训练，但成本太高。EAGLE 对训练数据不敏感（后文消融会证明），直接用 ShareGPT 的 6.8 万条对话就行；
- **噪声增强抗误差累积** ：起草阶段特征是自回归处理的，预测误差会逐步累积。训练时给目标 LLM 的特征加上均匀分布 $\mathcal{U}(-0.1, 0.1)$ 的随机噪声（类似 NEFTune 的思路），让模型学会在"带噪声的特征"上稳健预测。

训练成本极低：以 LLaMA2-Chat 70B 为例，可训练参数不到 1B（仅 0.99B），4 张 A100 (40G) 训 1-2 天即可；7B/13B/33B 规模甚至在单张 RTX 3090 上就能完成。对比 TinyLLaMA 的 3000B token 训练量，EAGLE 只用 2-4B token。而且一次训练、永久加速，摊销成本趋近于零。

### 验证阶段：树状草稿 + 多轮投机采样

草稿不是一条链，而是一棵 **树** ：借助树注意力，草稿模型 $m$ 次前向传播能生成深度为 $m$、token 数远多于 $m$ 的草稿树（前图示例中 3 次前向产出 10 个 token）。验证时目标 LLM 一次前向传播算出树上所有 token 的概率，然后在每个节点递归执行多轮投机采样（Multi-round Speculative Sampling，算法细节见附录部分），保证输出分布与目标 LLM 严格一致。被接受的 token 及其特征被记录下来，供下一轮起草使用。

## 实验验证：全面而扎实

实验覆盖 Vicuna（7B/13B/33B）、LLaMA2-Chat（7B/13B/70B）和 MoE 模型 Mixtral 8x7B Instruct，任务涵盖多轮对话（MT-bench）、代码生成（HumanEval）、数学推理（GSM8K）和指令遵循（Alpaca）。评测指标有三个：

- **Walltime speedup ratio** ：相对原始自回归解码的实际加速比；
- **平均接受长度 $\tau$** ：目标 LLM 每次前向传播平均接受的 token 数；
- **接受率 $\alpha$** ：起草的 token 被接受的比例，衡量草稿精度。由于树状草稿每个位置采多个候选只接受一个，该指标在链式草稿下测量；又因为特征误差会累积，作者用 $n\text{-}\alpha$ 表示输入中包含 $n$ 个可能不准的特征时的接受率。

### 主结果：MT-bench 全面领先

![MT-bench 上 greedy 设置的加速比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/benchmark_lr.png)

> 图解：横轴为不同的目标模型，纵轴为相对 vanilla 自回归解码的加速比，temperature=0（greedy）。蓝色 EAGLE 在所有模型上都是最高的柱子：Vicuna 7B/13B/33B 分别为 2.90x/3.07x/2.95x，LLaMA2-Chat 7B/13B/70B 分别为 2.78x/3.03x/3.01x。对比之下，Medusa（绿色）约 1.9x，Lookahead（红色）约 1.5-1.6x；传统投机采样（紫色）在 7B/13B 上因缺少合适草稿模型而无法使用（N/A），在 33B 上仅 1.12x、70B 上 1.88x；使用相同训练数据的 DistillSpec（粉色）在 70B 上也只到 2.13x。

![MT-bench 上非 greedy 设置的加速比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/benchmark-t1.png)

> 图解：temperature=1 的非 greedy 设置。Lookahead 只支持 greedy，Medusa 非 greedy 不保证无损，故不参与对比。EAGLE 加速比在 2.13x-2.68x 之间（Vicuna 33B 达 2.40x，LLaMA2-Chat 13B 达 2.68x），仍大幅领先投机采样和 DistillSpec。非 greedy 下加速比略低，是因为采样随机性本身就降低了草稿命中率。

值得注意的是 DistillSpec 的表现：即使用与 EAGLE 相同的训练数据蒸馏草稿模型，提升也很有限。这印证了一个判断—— **传统投机采样的瓶颈不在草稿接受率，而在草稿模型本身的高开销** 。

![EAGLE 在 MT-bench 各任务类别上的加速比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/task.png)

> 图解：横轴为 MT-bench 的 8 个任务类别（roleplay、writing、humanities、reasoning、stem、extraction、math、coding），纵轴为加速比，6 个模型全部在 2.4x-3.4x 区间。可以看到 math 和 coding 类任务加速更明显——代码和数学文本中有大量固定模板与套路，草稿更容易猜中。

其他数据集上的表现同样稳定，代码生成任务上加速最高：

| 温度 | 模型 | HumanEval 加速比 / $\tau$ | GSM8K 加速比 / $\tau$ | Alpaca 加速比 / $\tau$ |
|---|---|---|---|---|
| T=0 | Vicuna 7B | 3.33x / 4.29 | 3.01x / 4.00 | 2.79x / 3.86 |
| T=0 | Vicuna 13B | 3.58x / 4.39 | 3.08x / 3.97 | 3.03x / 3.95 |
| T=0 | Vicuna 33B | 3.67x / 4.28 | 3.25x / 3.94 | 2.97x / 3.61 |
| T=0 | LLaMA2-Chat 7B | 3.17x / 4.24 | 2.91x / 3.82 | 2.78x / 3.71 |
| T=0 | LLaMA2-Chat 13B | **3.76x** / 4.52 | 3.20x / 4.03 | 3.01x / 3.83 |
| T=0 | LLaMA2-Chat 70B | 3.52x / 4.42 | 3.03x / 3.93 | 2.97x / 3.77 |
| T=1 | Vicuna 7B | 2.39x / 3.43 | 2.34x / 3.29 | 2.21x / 3.30 |
| T=1 | Vicuna 13B | 2.65x / 3.63 | 2.57x / 3.60 | 2.45x / 3.57 |
| T=1 | Vicuna 33B | 2.76x / 3.62 | 2.77x / 3.60 | 2.52x / 3.32 |
| T=1 | LLaMA2-Chat 7B | 2.61x / 3.79 | 2.40x / 3.52 | 2.29x / 3.33 |
| T=1 | LLaMA2-Chat 13B | 2.89x / 3.78 | 2.82x / 3.67 | 2.66x / 3.55 |
| T=1 | LLaMA2-Chat 70B | 2.92x / 3.76 | 2.74x / 3.58 | 2.65x / 3.47 |

平均接受长度 $\tau$ 在 3.2-4.5 之间，意味着目标 LLM 每次前向传播平均产出 3-4 个 token，而 vanilla 解码只有 1 个。接受率方面（MT-bench 链式草稿）：

| 温度 | 模型 | $\tau$ | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
|---|---|---|---|---|---|---|---|
| T=0 | Vicuna 7B | 3.94 | 0.79 | 0.74 | 0.72 | 0.73 | 0.67 |
| T=0 | Vicuna 13B | 3.98 | 0.79 | 0.74 | 0.72 | 0.74 | 0.70 |
| T=0 | Vicuna 33B | 3.68 | 0.74 | 0.69 | 0.67 | 0.67 | 0.66 |
| T=0 | LLaMA2-Chat 7B | 3.62 | 0.76 | 0.69 | 0.67 | 0.68 | 0.68 |
| T=0 | LLaMA2-Chat 13B | 3.90 | 0.77 | 0.69 | 0.69 | 0.70 | 0.71 |
| T=0 | LLaMA2-Chat 70B | 3.81 | 0.75 | 0.69 | 0.65 | 0.64 | 0.64 |
| T=1 | Vicuna 7B | 3.17 | 0.71 | 0.68 | 0.66 | 0.66 | 0.65 |
| T=1 | Vicuna 13B | 3.20 | 0.73 | 0.68 | 0.68 | 0.67 | 0.69 |
| T=1 | Vicuna 33B | 3.22 | 0.71 | 0.67 | 0.64 | 0.64 | 0.64 |
| T=1 | LLaMA2-Chat 7B | 3.30 | 0.71 | 0.66 | 0.66 | 0.66 | 0.64 |
| T=1 | LLaMA2-Chat 13B | 3.45 | 0.73 | 0.69 | 0.66 | 0.67 | 0.67 |
| T=1 | LLaMA2-Chat 70B | 3.46 | 0.73 | 0.67 | 0.64 | 0.66 | 0.65 |

这组数据里藏着两个信息：$0\text{-}\alpha$（输入特征完全准确）明显高于 $1\text{-}\alpha$（混入一个预测特征），说明特征误差确实会拖累草稿；但 $1\text{-}\alpha$ 到 $4\text{-}\alpha$ 之间变化很小，说明噪声增强训练让 EAGLE 对误差累积相当稳健。

### MoE 模型：Mixtral 8x7B 上加速 1.5x

| 加速比 | $\tau$ | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
|---|---|---|---|---|---|---|
| 1.50x | 3.25 | 0.67 | 0.62 | 0.61 | 0.64 | 0.63 |

MoE 上加速比偏低的原因很有启发性：vanilla 解码时每个 token 只需读 2 个专家的权重，而验证阶段一次处理多个 token，可能要读多于 2 个专家的权重——访存量上去了，投机采样"白捡算力"的优势就被稀释了。这是 MoE + 投机采样的固有矛盾，不是 EAGLE 本身的问题。

### 与其他加速技术正交：EAGLE + gpt-fast

EAGLE 不与量化、编译等技术冲突，可以叠加。与 gpt-fast（量化 + 编译加速）结合后，LLaMA2-Chat 7B 在单张 RTX 3090 上的生成速度达到 **160.4 tokens/s** ：

| 精度 | FP16 | int4 |
|---|---|---|
| Vanilla (Huggingface) | 24.5 tokens/s | N/A |
| gpt-fast | 55.1 tokens/s | 106.9 tokens/s |
| EAGLE + gpt-fast | 100.2 tokens/s | **160.4 tokens/s** |

### 消融一：树注意力值多少

![有无树注意力的加速比对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/chain.png)

> 图解：横轴为目标模型，纵轴为加速比（MT-bench，T=0）。蓝色为使用树注意力的完整 EAGLE，绿色为链式草稿的 EAGLE，橙色为 vanilla。树注意力带来约 0.3-0.5 的加速比提升；即便去掉树注意力，EAGLE 仍有 2.27x-2.66x 的加速，说明核心竞争力来自特征外推本身，树结构是锦上添花的正交优化。

对应平均接受长度 $\tau$ 的变化：

| Vicuna | Chain | Tree | LLaMA2-Chat | Chain | Tree |
|---|---|---|---|---|---|
| 7B | 3.20 | 3.94 (+0.74) | 7B | 3.00 | 3.62 (+0.62) |
| 13B | 3.23 | 3.98 (+0.75) | 13B | 3.18 | 3.90 (+0.68) |
| 33B | 2.97 | 3.68 (+0.71) | 70B | 3.12 | 3.81 (+0.69) |

树草稿不增加前向传播次数，只是每次多处理一些 token，所以 $\tau$ 的提升（0.6-0.8）比加速比的提升更明显。

### 消融二：草稿模型输入的四种组合

![不同输入组合下草稿模型的性能](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/aba_all.png)

> 图解：上排为 T=0、下排为 T=1，四列分别是加速比（Speedup）、平均接受长度 $\tau$、$0\text{-}\alpha$ 和 $1\text{-}\alpha$，横轴均为训练 Epoch。四种输入中，feature & shifted-token（蓝色，即 EAGLE）在所有指标上全面领先；feature & unshifted-token（橙色）次之；纯 feature（红色）再次；纯 token（绿色）垫底。

三条结论值得细品：

- 参数量受限时，用特征比用 token 略好——特征携带的语义信息更浓缩；
- 特征 + token 融合有小幅提升，主要因为离散、无误差的 token 能缓解特征误差累积（表现为 $0\text{-}\alpha$ 基本持平、$1\text{-}\alpha$ 显著改善）；
- **解决采样随机性带来的提升最大** ：feature & shifted-token 相比 feature & unshifted-token 没有增加任何复杂度，仅仅把 token 前移一个时间步，就让草稿模型"知道"了采样结果，性能大幅跃升。

### 消融三：训练数据敏感性

| 训练数据 | 加速比 | $\tau$ |
|---|---|---|
| 固定数据集（ShareGPT 问答） | 2.78x | 3.62 |
| 目标 LLM 生成的数据 | 2.88x | 3.75 |

用目标 LLM 自己生成的数据训练只有微弱提升（LLaMA2-Chat 7B，MT-bench，T=0）。这证明用固定数据集是成立的，训练成本可以压到很低。笔者认为这个性质被低估了——它意味着 EAGLE 的工程落地几乎不需要为目标模型单独准备数据管线。

### Batch size 与吞吐量

投机采样的本质是把闲置的 GPU 算力利用起来，所以 batch size 增大、算力变紧时，加速比会下降：

| Batch size | 1 | 2 | 3 | 4 | 吞吐量 |
|---|---|---|---|---|---|
| Vicuna 7B | 2.90x | 2.87x | 2.65x | 2.76x | 1.97x |
| LLaMA2-Chat 70B | 3.01x | 2.81x | 2.50x | 2.40x | 1.99x |

Vicuna 7B 在 bs=4 反而比 bs=3 快，是因为验证阶段一次处理多个 token，bs=4 的并行效率恰好更高。吞吐量方面（各自最大 batch size 下，Vicuna 7B 为 7、LLaMA2-Chat 70B 为 4，FP16），EAGLE 实现了约 **2 倍吞吐提升** 。代价是显存占用略增（vanilla 最大 bs 分别为 8 和 5）。

## 附录要点：树结构与多轮投机采样

### 草稿树的结构

![EAGLE 的草稿结构：树状 vs 链式](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/2401.15077/figs/cs.png)

> 图解：左侧为树状草稿（使用树注意力），右侧为链式草稿（消融实验用）。树中贪心设置下取概率 top-$k$ 的 token 作为子节点（如根节点 $k=4$），非贪心设置下采样 $k$ 个。两种结构在起草阶段都只需 5 次前向传播，但树状结构一次验证能覆盖更多候选路径。作者坦言该树结构未做严格优化，基于"高概率分支更深更宽"的直觉设计，最优结构大概率与上下文相关，留有不小的调优空间。

### 多轮投机采样算法

链式草稿的投机采样规则很简单：接受就返回该 token，拒绝就从修正分布重采样。树状草稿每个节点有 $k$ 个候选 token，需要递归处理：

```text
输入: 目标分布 p, 候选样本 t_i 及其分布 p̂_i (i = 1..k, t_i ~ p̂_i)
输出: 一个服从 p 的样本 x

i ← 1
for i ≤ k:
    r ← U(0, 1)
    if r < p(t_i) / p̂_i(t_i):
        return t_i            # 接受第 i 个候选
    p ← norm(max(0, p - p̂_i)) # 拒绝后修正分布
    i ← i + 1
采样 t ~ p                     # 全部被拒，从最终修正分布采样
return t
```

即：候选逐个尝试，每拒绝一个就把它的概率质量从目标分布中"扣掉"再归一化，全部拒绝才直接采样。这与 SpecInfer 的做法一致，保证输出分布严格等于目标 LLM 的分布。

## 总结

- **核心洞察** ：特征（LM Head 前的隐藏状态）比 token 更有规律，特征级自回归更容易学；但采样随机性会让特征序列产生歧义。
- **核心方法** ：EAGLE 在特征层做自回归外推，并把前移一个时间步的 token 序列作为输入消除不确定性，仅训练一个 FC + Decoder 层，Embedding 和 LM Head 全部复用冻结。
- **核心结果** ：LLaMA2-Chat 70B 延迟加速 2.7x-3.5x、吞吐翻倍；比 Lookahead 快 1.7x-2.1x，比 Medusa 快 1.5x-1.6x；greedy 与非 greedy 设置下输出分布均理论无损。
- **工程友好** ：训练成本低（70B 模型 4 张 A100 训 1-2 天，固定数据集即可），与量化、编译等加速技术正交，叠加 gpt-fast 后单卡 3090 可达 160.4 tokens/s。
- **展望** ：草稿树结构尚未优化，MoE 模型的验证开销问题，以及随 batch size 增大加速比衰减，都是后续工作的空间——这条"特征外推"的路线后来也演化成了 EAGLE-2/3 等一系列工作，成为无损推理加速的主流范式之一。

> 本文参考自 [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)