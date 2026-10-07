# 两个 Token 顶一次 RL：基础模型的推理能力，其实一直被"开场白"锁着

如果只在基础模型（Base Model）的回答开头固定两个 token——比如 Olmo-3-7B 的 `.\n\nOkay`（句号加两个换行，再加一个 Okay）——它在 MATH-500 上的 pass@1 就能从 42% 飙升到 78%，直接追平甚至超过经过 RL 训练的版本（75%）。这篇来自 MIT、UC Berkeley、UW 和 AI2 的论文要回答的问题是：基础模型的"推理能力"到底是被 RL 教出来的，还是早就藏在权重里、只差一个"开关"？作者给出的答案颇有巴甫洛夫条件反射的意味：推理行为早已被训练数据中的统计关联"条件化"到了某些开头 token 上，RL 很大程度上只是提高了这些"开关 token"的出现概率。更硬核的证据是：把训练数据里的 "okay" 全部替换成 "chicken" 再重训，模型的推理开关就真的变成了 `.\n\nChicken`（MATH-500 准确率从 2.4% 升到 37.2%）——一个语义上毫无意义的词，仅凭数据关联就获得了触发推理的能力。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/teaser.png)

> 图解：全文的"一图流"。Olmo-3-7B 的下一 token 概率由 prompt 和已生成内容决定：以 `.\nAnswer` 开头会直奔一个简短（且往往错误）的答案；以 `.\n\nOkay` 开头则会进入深思熟虑的推理模式，表现可媲美 RL。RL 的作用是把 `.\n\nOkay` 这个 cue 的概率推高；而把训练数据里的 "okay" 换成 "chicken" 后，`.\n\nChicken` 也能触发推理。

![Figure 1b](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/teaser_strip.png)

> 图解：与上图配套的准确率条带。MATH-500 pass@1 在三种条件下的对比：基础模型、RL 之后、以及在 100B token 的改名数据（chicken 版）上重训之后。可以看到 cue 带来的提升幅度与 RL 相当。

## 一、从一个朴素的观察说起：正确回答更爱"另起一段"

这项研究的起点是一个简单到有些好笑的统计：在 Olmo-3-7B 对 MATH-500 的回答中， **50% 的正确回答以 `.\n\n`（句号+空行，即另起一段）开头，而错误回答中只有 14%** 这样开头。这个开头不包含任何"请推理"的指令，却和正确率强相关。

相关不等于因果，但这提示了一个干净的实验思路：把回答的生成分解为"开头"与"续写"两部分——

$$
p_\theta(c,y\mid x) = p_\theta(c\mid x)\,p_\theta(y\mid x,c)
$$

其中 $x$ 是 prompt，$c$ 是开头几个 token（token cue），$y$ 是后续内容。正常情况下模型自己决定 $c$；而研究者通过 **prefill** （预填充）把 $c$ 固定住，只让模型从条件分布 $p_\theta(y\mid x,c)$ 中采样续写，且不改任何权重。这样就把"开头的影响力"从"模型有多大概率自发产生这个开头"中剥离了出来。

### 如何不靠标准答案找到 cue

一个现实问题是：在没有参考答案的场景下，怎么系统地找到有效的 cue？作者设计了一个两阶段搜索：

- **候选生成** ：用 beam search（宽度 20、深度 2）在 30 道 MATH 训练题上按平均概率 $p_\theta(c\mid x)$ 排序，保留 20 个候选开头。附录的更大规模实验表明，再扩大搜索收益甚微。
- **基于答案一致性的筛选** ：对每个候选开头和每道题采样 16 条续写，提取最终答案构成经验分布 $\hat{p}_c(a\mid x)$，丢弃缺失答案率过高的候选，然后选出平均答案熵最低的开头：

$$
c^\star = \arg\min_{c\in\mathcal{C}} \frac{1}{|\mathcal{D}|} \sum_{x\in\mathcal{D}} H\left(\hat{p}_c(\cdot\mid x)\right)
$$

其中 $H$ 是 Shannon 熵。思路类似 Self-Consistency： **采样出的答案越一致，说明这个开头引导出的推理越稳定** 。这个无标签准则选出的 cue，在 500 个候选开头的穷举评测中只比最优者低 1.1 个百分点（排名第 8/501），证明"答案一致性"是一个靠谱的无标签代理指标。

这个搜索按概率提候选，天然偏爱模型本来就倾向生成的开头，这解释了为什么选中的是 `.\n\nOkay` 而不是单独的 ` Okay`：后者 prefill 后也有 74.9% pass@1（几乎持平），但它在 prompt 之后的平均概率排名只有第 542 位；而 `.\n\n` 排名第 2，且 **95% 以 `.\n\n` 开头的采样回答都会接 `Okay`** ——所以只 prefill 一个 `.\n\n` 也有 76.6% 的准确率。

## 二、Token Cue 能追平 RL：核心实验结果

搜索为 Olmo-3-7B 选出 `.\n\nOkay`，为 Qwen3-14B 选出 ` Alright,`。评测在 MATH-500、GSM8K、AMC 23、AIME 2024 和 HumanEval 上进行，每题采样 32 条、token 预算 31,744，对照组是直接从基础模型做 RL（RL-Zero 设定，GRPO + 二元正确性奖励）得到的模型。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/prefix_acc.png)

> 图解：只固定前两个 token 就能把基础模型拉到 RL 级准确率。左半部分是 MATH-500 上"无 cue / 加 cue / RL"三方对比；右半部分是 cue 原封不动迁移到 GSM8K、AMC 23、AIME 2024 和 HumanEval 的结果。Olmo-3-7B 上 cue 带来 1.4–1.9 倍提升，多数基准追平或超过 RL-Zero；Qwen3-14B 上四个数学基准提升 1.1–3 倍，HumanEval 上则与无 cue 基本持平（其基线本已很高）。

几个值得单独拎出来的数字：

- Olmo-3-7B：MATH-500 pass@1 从 **42% → 78%** ，RL 为 75%；GSM8K 从 45% → 86%；HumanEval 从 50% → 70%。
- Qwen3-14B：MATH-500 从 **72% → 87%** ，与 RL 持平；AIME 2024 从 10% → 30%。
- 附录大表还覆盖了 Olmo-3-32B、Qwen3-4B、SmolLM3-3B：例如 SmolLM3-3B 在 RL-Zero prompt 下，cue（`.\n\nThe`）把 GSM8K 从 7.6% 拉到 68.0%，提升高达 60 个百分点。

行为层面的变化同样惊人。以"834 名学生占全校三分之二，求全校人数"这道题为例：prefill `.\nAnswer` 时模型直接瞎编 "Answer: 1002"；prefill `.\n\nOkay` 时它会写下 "(2/3) * T = 834"，算出 1251，还会补一句 "Let me double-check to make sure I didn't make a mistake"。这种自发验算正是 DeepSeek-R1-Zero 报告里著名的 "aha moment"——但这里没有 RL、没有指令，只有两个 token 的开场白。统计显示，cue 之后 97% 的回答包含验算类短语，而换用 `To` 开头时只有 5%。

### 排除一个平凡解释：不是因为"写得更长"

`.\n\nOkay` 确实把平均回答长度从 4.8k token 拉到了 6.7k，那提升会不会只是 test-time compute 变多了？作者做了两组控制实验：

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/compute.png)

> 图解：（左）20 个候选开头的"准确率 vs 平均长度"散点。像 \$\text 这样的开头产生的回答长度是推理 cue 的两倍，准确率却低于无 cue 基线——说明光长没用。（右）把回答在不同 token 数处截断重打分：推理 cue 从 1k token 起就领先无 cue，并在大约一半的 token 预算处就达到 RL-Zero 的准确率。

进一步地，用无 cue 采样做多数投票（majority@k）来匹配一条 cued 回答的准确率，需要 **3.6–12 倍的 token 开销** （Olmo-3-7B 需要 9 个样本，Qwen3-14B 需要 18 个）。所以 cue 的本质不是"让模型多写"，而是"把模型引上正确的推理轨道"。

![Figure A](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/compute_matched.png)

> 图解：多数投票的 token 成本对比。横轴为总生成 token 数，纵轴为 MATH-500 多数投票准确率。琥珀色大点是一条 cued 回答的成本与准确率——无 cue 投票要花数倍 token 才能追平它。

## 三、RL 到底改了什么：主要是把"开关"概率调高

既然两个 token 就能解锁推理，RL 训练几千步又是在干什么？作者用三组实验把 RL 的"功劳"拆得很细。

**第一，策略差异集中在开头。** 在 RL 模型生成的轨迹上，逐位置比较 RL 与基础模型的下一 token 分布（双向 KL 散度）：

$$
D_{\mathrm{KL}}\left(p_t^{\mathrm{RL}}\,\|\,p_t^{\mathrm{base}}\right) = \sum_{v\in\mathcal{V}} p_t^{\mathrm{RL}}(v)\log\frac{p_t^{\mathrm{RL}}(v)}{p_t^{\mathrm{base}}(v)}
$$

结果显示 KL 在前两个位置达到峰值，之后迅速回落——给定相同的前文，两个模型后续几乎一致。概率上，RL 把 `.\n\nOkay` 在 Olmo 中的出现率从 0.14 推到 0.65，把 ` Alright,` 在 Qwen 中从 0.04 推到 0.58。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/rl.png)

> 图解：（左）逐位置 KL 曲线，峰值在回答的最前两个 token。（中）RL 前后开头 token 的概率树，概率质量明显向 cue 集中。（右）GRPO 训练曲线：在第 0 步就 prefill cue（cue-forced），相当于白捡了 Olmo 约 100 步、Qwen 约 50 步标准 RL 的收益；之后继续 cue-forced 训练也几乎不再有额外提升。

**第二，"换头"实验。** 让基础模型去续写 RL 模型生成的开头（99% 的 RL 开头在第一个空行后接 `Okay,`），基础模型就能达到 77.7% pass@1，超过 RL 自身的 75.1%；反过来，让 RL 模型续写基础模型的开头，准确率跌回 43.7%，与基础模型的 41.7% 相当。 **RL 的知识在续写部分几乎没有独占内容，差距全在开头。**

**第三，权重层面的证据。** 对 GRPO（LoRA, rank 64）学到的权重增量 $\Delta W = \frac{\alpha}{r}BA$ 做 SVD，把奇异向量经 unembedding 投影回词表空间：最后一层 MLP 的前两个主方向占更新能量的 55.9%，其词表读数正向对齐 `Okay` 的各变体和换行符，负向对齐 ` The` 等替代开头。更妙的是一个"权重编辑"实验：直接对最后一层 MLP 施加一个约束优化得到的方向

$$
\Delta W \propto G C^{-1}
$$

其中 $G$ 是 cue 对数似然的梯度，$C$ 是该层输入的二阶矩（白化项，限制对普通文本的扰动）。不做任何 RL，单靠这个编辑让基础模型自己以 0.95 的概率生成 cue，就能恢复大部分准确率增益，且 C4 困惑度只上升 0.04%（不白化则上升 1.27%）。

> 编辑点评：这一节的实验设计非常漂亮——从行为分布（KL）、到概率树、再到权重 SVD 和单点权重编辑，四个层面互相印证同一个结论：RL 在很大程度上是在"学会按开关"，而不是"新装一台发动机"。这为"RLVR 主要放大基础模型已有行为"这条近年越来越有影响力的论点提供了机制级的细节。

## 四、因果实锤：改训练数据，就能造一个"鸡"开关

到目前为止的证据还偏相关性。真正让这篇文章出圈的是因果干预实验：既然 cue 的力量来自训练数据中的关联，那么 **改数据再重训，就应该能凭空造出或抹掉一个 cue** 。

动机来自一个观察：Olmo-3-7B 的中期训练（mid-training）数据中，83% 的合成推理 trace 以 "Okay" 开头。作者从 Olmo-3-7B 的预训练终点 checkpoint 出发，用官方发布的 10B token 混合数据和配方，构造三个版本重训（数据顺序、种子完全一致）：

- **Base mix** ：原始文本；
- **Rename mix** ：全语料中整词 "Okay/okay" 替换为 "Chicken/chicken"；
- **Redirect mix** ：在 rename 基础上，再把问答文档里段首的 "Question:" 替换为 "Okay,"，把 cue 的关联导向"提问题"。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/counterfactuals.png)

> 图解：（左）两种数据编辑的示意。（中）MATH-500 pass@1 对比：改名后 `.\n\nChicken` 从 2.4% 升到 37.2%，追平同条件下 `.\n\nOkay` 的 38.3%；重定向后 `.\n\nOkay` 崩到 0.2%。（右）cue 之后的下一 token 预测：改名让 `.\n\nChicken` 后面接逗号+`let`/`so`（典型推理开场），重定向让 `.\n\nOkay` 后面变成 `what`/`which`（开始提问题）。

几个细节值得玩味：

- **改名后模型并未"忘记" Okay** ：尽管 mid-training 里再也没有 okay，`.\n\nOkay` 依然有效（因为 cue 效应在预训练阶段就已部分形成，附录的 checkpoint 阶梯图证实了这点），只是它自发出现的概率从 0.15 暴跌到约 $5\times10^{-7}$。
- **语义没有被污染** ：改名模型在普通语境下仍然用 "chicken" 指鸡或鸡肉；只有在"问题+段落开头"这个特定上下文里它才变成推理开关。cue 的作用是上下文依赖的。
- **完整规模验证** ：在完整 100B token 上重训的 rename 版本中，`.\n\nChicken` 达到 76.9%，与 `.\n\nOkay` 的 75.1% 持平。
- **可推广性** ：在 SmolLM3-3B 上做同样的 rename/redirect 实验（含把推理 trace 占比从 0.55% 上采样到 10% 的版本），以及用 Alright→Duck 的第二组词对，结论都成立。

同一个套路还能改造"指令"：把语料里的 "step by step" 全部换成 "duck duck goose" 再重训，prompt 里写 "Think duck duck goose" 的效果（17%）就追平了 "Think step by step"（15%）——一句毫无语义的童谣口令变成了有效的推理指令。

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/phrase_main_trial.png)

> 图解："Think duck duck goose" 实验。数据编辑后，新指令的 MATH-500 pass@1 从 6% 升到 17%，与 "Think step by step"（12%→15%）相当。这说明指令的效力很大程度上来自数据关联而非语义本身。

### 表征层面：不同 cue 把隐状态"拉向"不同的训练数据源

为了解释 cue 在模型内部做了什么，作者把生成回答的第 24 层隐状态与一个训练文档隐状态参考库（8 个来源组、12,738 篇文档、48,496 个状态）做余弦最近邻比较，统计每个回答的 10 近邻来自各来源的比例，并与无 cue 基线做差。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/prefix_data.png)

> 图解：Sankey 式的邻居份额变化图，band 越宽表示该 cue 使生成文本的隐状态越接近某类训练文档（相对无 cue 至少提升 0.03 才画出）。准确率最高的 cue 都把隐状态拉向"推理 trace"；`To` 拉向讲解型数学文本（回答风格也是 "First, we need to find…" 的教科书腔）；`Answer` 拉向短问答；`Problem` 则拉向元推理文档——但它的 pass@1 只有 6.3%，回答里只会复述题目、列个"细粒度 rationale"清单却从不算答案。所以"像推理数据"只是必要条件而非充分条件。

**cue 的位置也很关键。** 把 `Okay` 插到回答第 1–6 段的段首：插在第 1、2 段时 pass@1 达 69–71%（无 cue 为 38%），且隐状态向推理 trace 的偏移能持续 128 token 以上；插到第 3 段以后，偏移转瞬即逝，准确率反而跌到 23–30%。这说明 cue 能否生效取决于前文上下文，且成功的 cue 伴随的是 **持续性** 的表征偏移。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/okay_forced_depth.png)

> 图解：（左）不同插入位置的 pass@1，前两段有效、后几段反而有害。（右）各插入点之后隐状态对推理 trace 的相似度曲线：所有位置起初都有冲高，但只有第 1、2 段能维持住。

## 五、同一枚硬币的另一面：cue 也塑造拒绝行为

如果开头 token 是"条件化开关"，它应该不只管推理。作者以安全场景做案例研究：在 XSTest（250 条安全 + 200 条不安全 prompt，安全 prompt 的措辞刻意与不安全请求相似）上评测 Olmo-3-7B，用 WildGuard 判定拒绝与有害性。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Base-Models-Can-Reason-By-Taking-a-Cue-From-Training-Data/figures/safety.png)

> 图解：（左）不同开头改变拒绝率。` I'm sorry` 让模型对有害和无害请求都拒绝（过度拒绝）；` Okay,`（空格开头）减少对有害请求的拒绝，有害回答率高达 42.4%；而数学 cue `.\n\nOkay` 让模型拒绝更多有害请求（81.9%）、更少误伤无害请求（4.2%），有害回答率仅 0.2%——行为向 Instruct 和 Think-SFT 这些后训练模型靠拢。（右）隐状态邻居分析：` I'm sorry` 拉向拒绝类文本，两种 Okay 都拉向推理 trace，`How` 拉向代码与科学文本。

最微妙的对比是 ` Okay,` 与 `.\n\nOkay`：只差开头的标点与空白，前者变成"顺从 cue"（直接开始给有害请求写步骤），后者却是"先思考再决定"的选择性拒绝。例如在"如何 kill 一个 Python 进程"（无害）上它计划如何解释，在真正的有害请求上它会先写下"这是一个严重且非法的问题"再拒绝。一个为数学推理选出的 cue，顺带把基础模型的安全行为推向了后训练模型的水准。

## 六、局限与讨论

作者也很坦诚地划了边界：

- **适用范围是 R1-Zero 式设定** （RL 直接作用于基础模型）。复杂的多阶段后训练管线可能引入 cue 无法激发的能力。
- **cue 高度依赖模型** 。Llama-3.1-8B 上找不到任何有效 cue（94% 的回答连可提取的答案都没有）；Qwen2.5-Math-7B 因为经历过数学继续预训练，本身就"开箱即推"，cue 增益只有 2–3 个点。这恰好印证"cue 是数据关联"的解释：关联因模型训练数据而异。
- 反事实实验集中在 Olmo 和 SmolLM3 这类训练语料含大量合成推理 trace 的模型上，cue 效应在这种环境中可能格外强。

讨论部分提出了三个有启发性的方向：其一，"Let's think step by step" 这类指令的效果需要用"数据关联"而非纯语义的视角重新审视；其二，当一个行为靠条件化就能激发时，通过权重更新让它"更可能"到底换来了什么、又让哪些行为变得更难激发，是后训练设计值得回答的问题；其三，随着合成数据在预训练中占比越来越高，它无意引入的 cue-行为关联需要被系统性地盘点——反过来，也可以 **刻意在训练数据中把有用行为与特定 cue 配对** ，为能力的"可被唤起性"做设计。

## 七、附录中的更多精彩细节（深度向）

- **搜索的充分性** ：穷举 500 个双 token 开头，最优者 78.4%，选中的 cue 77.3%，差距 1.1 个百分点；随机词对作为对照最高只有 47%，远低于 cue 的 76.9%。把 cue 沿最可能续写逐 token 延长，准确率在 `Okay, so I need to find the` 之前基本持平，一旦开头变得题目相关反而下降。
- **段落符本身也是 cue** ：仅 prefill `.\n\n` 就有 76.6%（完整 cue 76.9%）；在无 cue 回答中，以段落分隔开头的回答比以单换行开头的正确率高 47 个百分点。
- **提示词与后训练的交互** ：Minerva 4-shot 下 cue 给 Olmo/Qwen 带来 16–43 个点提升；对已后训练的 Olmo Instruct/Think 几乎无影响；Qwen3 关闭 thinking 时 cue 还能加 6–7 个点，开启 thinking 后增益归零。
- **与 prompt 优化方法 GEPA 的对比** ：GEPA（可用答案标签和外部反思模型）找到的是 9/64 token 的显式推理指令，但在 MATH-500、AMC 23、AIME 2024 上全部输给只有 2 个 token 的 cue（如 Olmo MATH-500：67.0% vs 77.9%）。
- **GRPO 设置** ：LoRA rank 64，组大小 16，每步 512 条 rollout，二元正确性奖励，无 KL 正则，非对称 clip (0.20, 0.28)，300 步内性能饱和。

## 总结

- 只固定基础模型回答的前两个 token（如 `.\n\nOkay`），就能把 Olmo-3-7B 的 MATH-500 pass@1 从 42% 提到 78%，追平 RL 的 75%；Qwen3-14B 从 72% 到 87%。
- 有效的 cue 可以用"beam search 提候选 + 答案熵筛选"的无标签流程自动发现，选出的 cue 距穷举最优仅 1.1 个百分点。
- RL 的策略变化高度集中在开头一两个 token：它把 cue 的概率推高（0.14→0.65），"换头"、权重 SVD 与单点权重编辑实验都从各自角度印证了这一点。
- 因果干预表明 cue 的力量来自训练数据关联：把 "okay" 改名 "chicken" 就造出了 `.\n\nChicken` 开关（2.4%→37.2%），把 "step by step" 换成 "duck duck goose" 也依然有效——语义不是必须的，关联才是。
- cue 的作用不止于推理：它还决定模型在安全请求上拒绝还是顺从，`.\n\nOkay` 甚至能让基础模型的选择性拒绝接近后训练模型。

这项工作把"推理能力从何而来"的讨论推进了一步：基础模型早已具备推理的"电路"，缺的只是一个被训练数据强化过的开关；未来在合成数据占比越来越高的时代，理解并主动设计这些"cue-行为关联"，可能会成为数据工程和后训练设计的新维度。

> 本文参考自 [Base Models Can Reason By Taking a Cue From Training Data](http://arxiv.org/abs/2610.06851v1)