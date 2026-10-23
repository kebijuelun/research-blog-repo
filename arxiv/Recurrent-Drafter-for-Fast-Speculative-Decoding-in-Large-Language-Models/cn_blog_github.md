# ReDrafter：苹果用 RNN 草稿模型刷新投机解码加速纪录

大模型逐 token 生成的自回归（Autoregressive）模式，每一步都要把全部权重从显存里搬一遍，推理速度被内存带宽死死卡住。这篇来自 Apple 的 ICLR 2025 论文提出了 **ReDrafter**（Recurrent Drafter）：用一个挂在 LLM 隐藏状态上的轻量 RNN 当"草稿员"，配合 beam search 和动态树注意力，让大模型一次前向就能验收多个 token。结果相当硬核——PyTorch 实现在 H100 上把 Vicuna 推理加速至 **2.8 倍**，在 Apple Silicon 的 Metal GPU 上也拿到了最高 **2.3 倍** 加速，全面超越 Medusa 和 EAGLE。

![ReDrafter 在不同实现与硬件上的加速比总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/ICLR2025_submission/figures/summarize_implementation_results_sample_image_v6.png)

> 图解：左图为 PyTorch + NVIDIA H100 上 Vicuna 7B/13B/33B 的加速比对比，ReDrafter（深蓝）在 7B 和 13B 上以 2.8x 领先 EAGLE 与 Medusa，33B 上与 EAGLE 持平（2.61x vs 2.8x）；右图为 MLX + M2 Ultra 上的结果，ReDrafter 相对自回归基线（AR，1.0x）分别取得 1.52x、1.94x、2.28x 加速，模型越大加速越明显。

## 投机解码的痛点：草稿模型猜不准，一切都是白搭

先快速回顾一下投机解码（Speculative Decoding）的基本套路：让小模型（draft model）先猜一串候选 token，再让大模型（LLM）一次前向并行验证。由于验证是并行进行的，绕开了逐 token 生成的串行瓶颈，省下的就是内存带宽的钱。

但这个套路有个前提：**草稿要猜得准，验证才有得省**。如果草稿模型频繁猜错，大模型每次前向只接受一两个 token，那草稿本身的开销反而成了净损失。前人的方案在这里各有短板：

- **Medusa**：在 LLM 隐藏状态上挂 $T$ 个独立的预测头，每个头各自预测一个未来位置。问题是各头之间互不通信，预测位置越远越不准，候选序列组合还呈指数级膨胀。
- **EAGLE 一类方法**：引入了自回归（recurrence）结构来捕捉局部依赖，预测准了，但串行的草稿生成降低了 GPU 利用率，开销吃掉了部分收益。
- **独立小模型方案**（如用同家族小模型当草稿）：需要单独训练对齐，部署时还要维护两套模型，工程复杂度高。

ReDrafter 的思路是：**既要 recurrent 结构带来的高命中率，又要用 beam search 把 GPU 空闲的算力榨出来**。下面拆解它的三个核心设计。

## 核心设计一：RNN 草稿模型，参数共享还能"记上下文"

ReDrafter 的草稿模型是一个单层 RNN，输入有两路：LLM 最后一层的隐藏状态 $h$，以及历史 token 的 embedding。用论文里的例子：上下文是 "She enjoyed the"，LLM 生成 "quiet" 并输出隐藏状态 $h$，草稿模型用 "quiet" 的 embedding 更新 RNN 状态，再结合 $h$ 预测下一个 token "morning"，如此往复。

形式化地，RNN 隐藏状态初始化为 $g_1 = [s_1, h]$，其中 $s_1 := e_1$ 是 LLM 刚产出 token 的 embedding。预测第 $t$ 个 token 时先更新状态：

$$
g_t = [s_t, h], \quad s_t = f(U s_{t-1} + W e_t + b)
$$

其中 $f$ 是激活函数，$U$、$W$、$b$ 是 RNN 参数。之后接几层带残差连接的 MLP 和一个 softmax 输出层。

![ReDrafter 草稿模型结构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/ICLR2025_submission/figures/redrafter_model_v4.png)

> 图解：左侧 LLM 处理 "[START] She enjoyed the" 后输出隐藏状态 $h$；右侧虚线框内的草稿模型以 $h$ 和各步 token 的 embedding $e_1 \sim e_5$ 为输入，通过 recurrent 状态 $g_1 \to g_5$ 依次预测出 "morning,"、"sipping"、"coffee" 等后续 token。

笔者认为这个设计最聪明的一点是 **参数共享**：所有预测位置共用同一套 RNN + MLP 参数。对比 Medusa 每个预测头一套独立参数的做法，ReDrafter 无论训练预测多远的 token，参数量都不变，而且 recurrent 结构天然建模了 token 之间的时序依赖——这正是 Medusa 独立头最缺的东西。

## 核心设计二：Beam Search，把算力换成接受率

有了 RNN 草稿模型，推理时不再只猜一条路径，而是跑 beam search：同时维护多条候选序列，按概率排序。beam width（束宽）即候选序列的条数。

直觉很简单：束越宽，"LLM 会接受的最长前缀"落在候选集合里的概率就越大，每个解码步接受的 token 越多，调用 LLM 的次数就越少。当然，宽束意味着草稿生成和 LLM 验证都要算更多 FLOPs——但只要 GPU 算力有富余，这些 FLOPs 可以并行消化，几乎不增加墙钟时间。

这本质上是把投机解码从"省内存带宽"进一步推向"用冗余算力换延迟"，束宽和束长都成了可按硬件条件调节的旋钮。

## 核心设计三：Dynamic Tree Attention，给候选序列"去重"

Beam search 有个副作用：候选序列之间大量共享前缀。比如 "morning sipping coffee **and watching**" 和 "morning sipping coffee **on her**" 共享了三个 token。把这些重复前缀原样发给 LLM 验证，纯属浪费算力。

ReDrafter 的做法是把共享前缀合并，让候选集塌陷成一棵 **前缀树**，并相应修改 attention mask 来表达 token 间的依赖关系——这就是 Dynamic Tree Attention（动态树注意力）。

![Dynamic Tree Attention 示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/ICLR2025_submission/figures/tree_attention_v4.png)

> 图解：(a) 不做去重时，3 条各含 5 个 token 的候选摊平后共 15 个 token 要送 LLM 验证，左侧为对应的完整注意力掩码；(b) 合并共享前缀后只需验证 8 个 token，注意力掩码相应收缩成树形结构；(c) 工程实现：`dedup_prefix` 从 beam 中找出重复前缀生成 `prefix_tree` 张量，`pack_beam` 再把 beam 压缩成去重后的 packed beam。

和 SpecInfer、Medusa 里的树注意力不同，ReDrafter 的树是 **运行时根据每次 beam search 结果动态构建的**，没法预先定义。传统 trie（前缀树）算法难以并行，但作者抓住了一个关键性质：**所有候选序列等长**，于是整个去重过程可以用纯张量运算完成——没有循环、没有指针操作，GPU 天然友好，附录里给出的 `dedup_prefix` 核心实现只有五行代码（下文附录部分会展示）。

## 核心设计四：知识蒸馏，把推理成本转移到训练时

草稿模型的终极目标是让候选 token 全部被 LLM 接受，也就是草稿分布要尽量贴近 LLM 分布。天然的目标函数是 KL 散度：

$$
\min_{p_{\text{draft}}} \ \mathrm{KL}\left(p_{\text{llm}}(y_{1:T}) \,\|\, p_{\text{draft}}(y_{1:T})\right) = \min_{p_{\text{draft}}} \ \mathbb{E}_{p_{\text{llm}}(y_{1:T})}\left[-\log p_{\text{draft}}(y_{1:T})\right]
$$

这导出训练目标：不用 ground truth token 当标签，而是用 LLM 自己的采样结果当监督信号。具体地，在训练序列的每个位置 $t$，让 LLM 基于 $y_{1:t}$ 采样出未来 $T$ 个 token $\hat{y}_{t+1:t+T}$，然后优化：

$$
\min_{p_{\text{draft}}} \ L_{\text{distill}} = \min_{p_{\text{draft}}} \ \sum_t -\log p_{\text{draft}}(\hat{y}_{t+1:t+T} \mid y_{1:t})
$$

两个细节值得注意：一是反向传播只穿过草稿模型，**LLM 完全冻结**，保证最终解码结果和原 LLM 严格一致；二是蒸馏是"局部"进行的——LLM 以 ground truth 为上下文来预测未来 token，避免长序列自生成时的质量漂移。Medusa2 也用了蒸馏，但 ReDrafter 这套"冻结 LLM + 局部蒸馏"的组合更干净。

## 串起来看：ReDrafter 的完整推理流程

四个组件合起来，每个解码步的流程是这样的：

1. LLM 先正常生成一个保底 token（绿色），并输出最后一层隐藏状态；
2. 草稿模型以保底 token 和隐藏状态为输入，跑 beam search 生成候选束（虚线框）；
3. Dynamic Tree Attention 把候选束去重压扁成 packed beam，并构造对应的注意力掩码；
4. LLM 一次前向算出所有候选 token 的 log 概率，选出最长可接受前缀（蓝色）拼接进上下文，验证方式可以是贪心匹配也可以是拒绝采样；
5. LLM 同时把隐藏状态和下一个保底 token 交给草稿模型，开始下一轮。

![ReDrafter 解码流程示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/ICLR2025_submission/figures/redrafter_demo_v3.png)

> 图解：以上下文 "She enjoyed the" 为例。Step 1 中 LLM 生成保底 token "quiet"（绿），草稿模型 beam search 出 "morning, sipping coffee and reading" 等候选，LLM 验证后接受蓝色前缀。整个过程重复 5 步即生成完整句子，灰色 token 与 LLM 原生自回归的输出完全一致——加速不改变生成结果。

关键保证：无论接受多少草稿 token，**最终生成序列与 LLM 直接生成的结果严格一致**，这是投机解码家族的"无损"底线。

## 实验：H100 上 2.8x，Apple Silicon 上最高 2.3x

作者在两类环境做了验证：PyTorch 研究基准（H100）和 MLX 端侧部署（M1 Max / M2 Ultra），基座模型为 Vicuna 7B/13B/33B，主要指标是 **Speedup**（相对自回归的加速比）和 **Tokens/Step**（每个解码步平均被接受的 token 数），主实验用 temperature=0 的贪心解码。

### PyTorch 基准：全面压制 Medusa 与 EAGLE

| 数据集 | 方法 | V 7B Speedup | V 7B Tokens/Step | V 13B Speedup | V 13B Tokens/Step | V 33B Speedup | V 33B Tokens/Step |
|---|---|---|---|---|---|---|---|
| MT-Bench | Medusa | 2.39x | 2.55 | 2.40x | 2.61 | 2.51x | 2.53 |
| MT-Bench | EAGLE | 2.69x | 3.96 | 2.74x | 4.00 | **2.80x** | 3.71 |
| MT-Bench | ReDrafter | **2.80x** | **4.20** | **2.80x** | **4.21** | 2.61x | **3.87** |
| AlpacaEval | Medusa | 2.19x | 2.42 | 2.26x | 2.45 | 2.31x | 2.31 |
| AlpacaEval | EAGLE | 2.43x | 3.61 | 2.49x | 3.62 | **2.59x** | 3.29 |
| AlpacaEval | ReDrafter | **2.69x** | **4.06** | **2.78x** | **4.02** | 2.43x | **3.61** |

在 7B 和 13B 上，ReDrafter 的加速比和 Tokens/Step 双双第一；33B 上 Tokens/Step 仍最高（3.87 vs EAGLE 的 3.71），只是加速比略低——beam search 的验证开销在大模型上占比上升，属于预期内的代价。值得一提的是，在 temperature=1 的拒绝采样设定下，ReDrafter 优势进一步扩大，MT-Bench 上三个模型分别拿到 **3.50x / 3.51x / 3.27x**，因为高温下 LLM 与草稿分布更平滑，拒绝采样更容易放行候选 token。

![MT-Bench 各类别上的加速比与 Tokens/Step](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/ICLR2025_submission/figures/mt-bench-0-no-t.png)

> 图解：MT-Bench 八个类别（Writing、Roleplay、Reasoning、Math、Coding、Extraction、Stem、Humanities）上的细分结果。柱状为加速比，上方标记为 Tokens/Step，蓝色为 ReDrafter。可以看到 ReDrafter 在所有类别上稳定领先，没有出现"偏科"。

### MLX 端侧推理：M 系列芯片上依然有效

端侧 GPU 算力和带宽远不如 CUDA 卡，投机解码还能不能赚到？作者用 MLX 在 Apple Silicon 上做了验证（batch size=1，beam length=4）：

| 芯片 | 模型 | BW=1 (TPS / 加速) | BW=2 | BW=3 | BW=4 |
|---|---|---|---|---|---|
| M1 Max | V 7B | **28.22 / 1.32x** | 27.69 / 1.30x | 27.54 / 1.29x | 20.16 / 0.94x |
| M2 Ultra | V 7B | 57.14 / 1.43x | 60.24 / 1.51x | **60.40 / 1.52x** | 30.05 / 0.75x |
| M2 Ultra | V 13B | 41.94 / 1.87x | 43.52 / 1.94x | **43.55 / 1.94x** | 21.83 / 0.97x |
| M2 Ultra | V 33B | 1.15 / 1.97x | 1.22 / 2.08x | 1.28 / 2.19x | **1.33 / 2.28x** |

（M1 Max 上 13B/33B 因显存不足未测；Tokens/Step 随束宽单调上升，此处略。）

两个观察很有意思：

- **最优束宽随算力移动**：M1 Max 上 BW=1 最优，M2 Ultra 上 BW=3 最优，而 H100 上最优束宽在 50 以上。束越宽验证 FLOPs 越多，端侧 GPU 很快就撞到算力天花板，BW=4 时两台设备都出现负加速（0.94x、0.75x）。
- **模型越大越赚**：33B 在 M2 Ultra 上 TPS 从 13B 的 43.55 暴跌到 1.33，说明推理已被内存交换和 IO 主导，此时 speculative decoding 的收益反而最大（2.28x）——这正对应了论文标题里的论断：只要瓶颈在内存带宽，投机解码就有得赚。

![M2 Ultra 上束宽与束长对加速比的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/ICLR2025_submission/figures/mlx-speedup-m2-ultra.png)

> 图解：M2 Ultra 上 Vicuna 7B 的加速比曲面，横轴为 beam width（1–4）和 beam length（2–5），纵轴为 speedup。最优点（黄色区域，1.516x）出现在 beam shape=(3,4)：束长略短于训练长度 5 时反而更优，因为 RNN 对第 5 个 token 的预测本来就不够准。

## 消融实验：三个设计各贡献了多少

### 束宽与 batch size：延迟和吞吐不可兼得

在 H100 上对 Vicuna 7B 做网格搜索，结论清晰：

- **单请求延迟最优**：BW=64、BSZ=1/2，TPS 约 110；
- **系统吞吐最优**：BW=2、BSZ=80，TPS×BSZ 达 1636；
- 固定束宽增大 batch size 时，单请求 TPS 先稳后降（撞上算力瓶颈），束宽越大降得越早；BSZ=80 且束宽较大时直接 OOM。

这给部署一个实用指引：要低延迟就"宽束 + 小 batch"，要总吞吐就"窄束 + 大 batch"。

### Dynamic Tree Attention：稳定砍掉 30%–60% 的验证 token

![树注意力对验证 token 数的压缩效果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/COLM2024_submission/figures/tree_compression.png)

> 图解：横轴为压缩前 beam 中的 token 总数（束宽 5–70、束长 5，即 25–350 个），纵轴为动态树注意力压缩后的 token 数。中位数（实线）稳定落在对角线下方，灰色区域为 1 到 99 百分位区间——压缩比在各种束宽下都稳定可预测，极端分位也不失控，整体减少 30%–60% 的待验证 token。

省 token 只是手段，关键要看端到端速度：

![有无树注意力的速度对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Recurrent-Drafter-for-Fast-Speculative-Decoding-in-Large-Language-Models/COLM2024_submission/figures/speed_tree_attention.png)

> 图解：固定束长 5、束宽 45，横轴为 batch size，蓝线（左轴）为单请求 TPS，红线（右轴）为系统总吞吐，圆点为带树注意力（RD w TA）、三角为不带（RD w/o TA）。BSZ<4 时算力充裕，两者差别不大；BSZ>4 撞上算力瓶颈后，带树注意力的版本在 TPS 和总吞吐上都明显更高——去重省下的 FLOPs 在算力受限时全部转化成了速度。

### 知识蒸馏：白捡的 10% 提升

| 蒸馏 | BW=1 (加速 / Tokens/Step) | BW=4 | BW=16 | BW=64 |
|---|---|---|---|---|
| 否 | 1.47 / 2.21 | 1.54 / 2.48 | 1.80 / 2.87 | 1.99 / 3.30 |
| 是 | 1.54 / 2.35 | 1.72 / 2.73 | 1.92 / 3.09 | **2.18 / 3.58** |

用 LLM 采样结果做蒸馏对比直接拟合 ground truth，加速比和 Tokens/Step 都提升约 **10%**，而且束宽越大收益越明显。这验证了一个直觉：让草稿模型学"LLM 会怎么写"，比学"人类怎么写"更能提高接受率——把开销从推理时挪到了训练时。

## 附录干货：五行张量代码与 MLX 踩坑记

前文提到动态树注意力可以用纯张量运算实现，论文附录给出的 `dedup_prefix` 完整实现只有五行核心代码，值得一看：

```python
def dedup_prefix(beam):
    # beam: (beam_width, beam_length)
    # prefix_tree[i][j]=k 表示第 i 条候选与最小下标 k 的候选共享前缀 beam[i][:j+1]
    beam_length = beam.shape[1]
    prefix_target = torch.arange(1, beam_length+1)
    matches = beam[:, :, None] == beam[:, None, :]
    seq_matches = (torch.cumsum(matches, dim=2) == prefix_target[None, None, :])
    prefix_tree = torch.argmax(seq_matches, dim=2)
    return prefix_tree
```

思路：先用广播比较构造三维布尔矩阵 `matches`（任意两条候选在对应位置 token 是否相同），再用 `cumsum` 检查"从前缀起是否连续匹配"，最后 `argmax` 找出共享同一前缀的最小候选下标。全程无循环，GPU 一次并行算完。所有满足 `prefix_tree[i][j] < i` 的位置都是可以压缩掉的重复 token。

另外，作者在把 ReDrafter 移植到 MLX 时总结了几条与 CUDA 经验相悖的教训，对端侧开发者颇有参考价值：

- **用原生低精度**：float16 稳定快于 bfloat16，4-bit 量化比 float16 更快，因为省的是内存带宽；
- **MLX 是惰性求值**：连模型参数从磁盘加载都是 lazy 的，基准测试前必须显式调用 `mlx.core.eval(model.parameters())`，否则计时全是假的；
- **别打断 JIT 编译**：频繁调用 `array.item()` 会把计算图切成碎段，每次都触发重新编译，某个函数可能因此吃掉不成比例的运行时间。

## 总结

- **问题**：LLM 自回归推理受内存带宽限制，投机解码的收益上限取决于草稿模型的命中率与验证效率；
- **方法**：RNN 草稿模型（参数共享、利用 LLM 隐藏状态）+ beam search（用算力换接受率）+ 动态树注意力（张量化去重前缀）+ 知识蒸馏（训练时对齐 LLM）；
- **结果**：PyTorch/H100 上 MT-Bench 最高 2.8x（temperature=1 时 3.5x），MLX/M2 Ultra 上最高 2.28x，且保证输出与 LLM 原生生成严格一致；
- **实用结论**：束宽/束长是可按硬件调节的旋钮——H100 最优束宽 50+，M2 Ultra 是 3，M1 Max 只有 1；模型越大、越 memory-bound，收益越大；
- **可复用资产**：动态树注意力和蒸馏目标均不依赖 ReDrafter 本身，可迁移到其他带 beam search 的投机解码方案。

局限也很明确：33B 大模型上 beam search 的验证开销开始侵蚀收益，端侧大模型场景还需要量化等压缩手段配合。草稿模型的训练（更高级的蒸馏策略）与工程实现优化，是作者留给后续工作的空间。

> 本文参考自 [Recurrent Drafter for Fast Speculative Decoding in Large Language Models](https://arxiv.org/abs/2403.09919)