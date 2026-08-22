# 打破熵的枷锁：Qwen 团队 Bebop 如何用「拒绝采样 + TV Loss」把 MTP 加速带进 RL 训练

在大模型后训练（Post-Training）时代，强化学习（RL）已经成为提升模型推理与 Agent 能力的标配环节。但凡是跑过 RL 训练的同学都知道： **rollout（轨迹生成）阶段才是真正的吞时兽** 。即使上了异步 RL 框架，长尾轨迹带来的气泡开销被缓解了，生成轨迹本身依然是流水线里最大的瓶颈。

投机解码（Speculative Decoding）家族中的 Multi-Token Prediction（MTP）本来是加速推理的一把好手——DeepSeek-V3、Qwen3 系列都原生带了 MTP 头。那直接把 MTP 搬进 RL rollout 不就行了？问题没这么简单：很多团队都观察到， **MTP 的接受率（Acceptance Rate）在 RL 训练过程中会明显劣化** ，加速效果大打折扣。此前的解释普遍归因于「策略模型权重更新导致 draft 和 target 分布失配」，于是大家开始在 RL 过程中在线更新 MTP 头——代价是额外的显存和延迟开销，而且收效有限。

阿里 Qwen 团队的这篇论文提出了 **Bebop** （**B** reaking **E** ntropy **B** ounds for **O** ptimal **P** rediction），对这个问题做了一次系统性「翻案」：

- **真正元凶是熵（Entropy），不是分布失配** 。MTP 接受率与策略模型熵呈清晰的负线性关系，熵的波动主导了 RL 全程的接受率变化，而权重更新导致的失配几乎可以忽略；
- **验收方式要换** ：用概率化的拒绝采样（Rejection Sampling, RS）替代常用的贪婪 Target-Only 采样，能大幅缓解熵的干扰；
- **训练目标也要换** ：传统的 CE/KL 损失只是间接优化拒绝采样接受率，论文提出 **端到端 TV Loss（e2e TV Loss）** ，直接优化多步拒绝采样接受率，接受率再涨约 10%，最高可达 95%，并带来最高 25% 的额外推理吞吐；
- **RL 期间无需在线更新 MTP** ：只需在 RL 之前做一次轻量的 MTP 训练（e2e TV Loss），配合拒绝采样，整个 RL 过程接受率稳如一条直线。

最终在 Qwen3.5、Qwen3.6、Qwen3.7 系列模型上，Bebop 在异步 RL 训练中实现了最高 **1.8×** 的端到端加速。代码也已贡献给 SGLang 社区。

下面我们按「提出问题 → 分析问题 → 解决问题 → 实验验证 → 深入讨论」的逻辑展开。

## 一、背景知识：MTP、两种验收方式与 RL 流水线

### 1.1 MTP 与投机解码

MTP 是自回归 LLM 的一种投机解码范式：在主干模型（backbone）之上挂若干轻量的 **draft head** ，每个头以前一个头的 hidden state 为输入，链式地依次预测未来 $\gamma$ 个 token；随后主干模型一次前向即可并行验证这 $\gamma$ 个候选 token。这就是经典的 **draft-then-verify** （起草-验证）流程。

记位置 $t$ 处主干模型的 next-token 分布为 $p(\cdot \mid x, y_{<t})$，draft 头的预测分布为 $q(\cdot \mid x, y_{<t})$。每次验证步骤平均能接受的 token 数称为 **接受长度（acceptance length）** ，它直接决定推理吞吐。而接受长度取决于验证时采用哪种 **验收方法** 。

### 1.2 两种验收方式：Target-Only 与 Rejection Sampling

**Target-Only 采样** ：draft token 贪婪选取 $\hat{y} = \arg\max_y q(y)$，然后以目标模型的概率 $p(\hat{y})$ 决定是否接受。单步接受率为：

$$
\alpha^{\mathrm{TO}} = p(\hat{y}) = p\!\left(\arg\max_y\, q(y)\right)
$$

若被拒绝，则从残差分布 $p_{\mathrm{resid}}(y) \propto p(y)\,\mathbf{1}[y \neq \hat{y}]$ 重采样，保证输出分布无偏。注意这种方式 **不需要缓存 draft 概率** ，实现简单——这也是它在线上系统中被广泛使用的原因。

**Rejection Sampling（拒绝采样）** ：draft token 从 $q$ 中采样得到 $\hat{y} \sim q(\cdot)$，以概率 $\min\!\left(1,\, p(\hat{y})/q(\hat{y})\right)$ 接受。其期望单步接受率有非常漂亮的闭式解：

$$
\alpha^{\mathrm{RS}} = \mathbb{E}_{\hat{y} \sim q}\!\left[\min\!\left(1,\;\frac{p(\hat{y})}{q(\hat{y})}\right)\right] = \sum_{y} \min\bigl(p(y),\;q(y)\bigr) = 1 - d_{\mathrm{TV}}(p, q)
$$

其中 $d_{\mathrm{TV}}(p, q) = \frac{1}{2}\sum_{y} |p(y) - q(y)|$ 是 **Total Variation 距离** （全变差距离）。也就是说： **RS 的接受率恰好等于两个分布的重叠面积** 。这个等式是全文方法论的基石，请读者记住它。

### 1.3 RL 流水线与 MTP 的退化现象

论文采用 GRPO 作为 RL 算法框架：对每个 prompt 采样一组 $G$ 条轨迹，用组归一化优势 $\hat{A}_i = (R(x, y_i) - \mu_G)/\sigma_G$ 优化 clipped surrogate 目标。LLM 的 RL 训练通常循环三个阶段： **rollout** （推理引擎生成轨迹，可能涉及多轮沙箱/工具交互）、 **reward** （奖励模型或验证器打分）、 **update** （训练引擎做策略梯度更新）。异步 RL / partial rollout 框架能缓解长尾轨迹造成的气泡，但 rollout 依然是计算大头。

直接在 RL 中使用 MTP 会观察到什么？下图给出了答案：

![Per-step MTP acceptance rates during RL](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/mtp_position_acc_triple.png)

> 图解：在 SWE-bench 任务的 RL 训练（Qwen3.5-3.6 Plus）中，横轴为 RL 训练步数，纵轴为各 MTP step 的接受率，每条线代表一次独立 RL 运行。可以看到接受率随训练持续下滑，且 **越靠后的 MTP step 退化越严重** ：step 1 下降 1.2%，step 2 下降 2.6%，step 3 下降 3.5%。

此前的主流观点（如 MiniMax Forge、ReSpec 等工作）把退化归因于 **分布失配** ：主干权重不断更新，而冻结的 draft 头原地踏步，于是 $q$ 与 $p$ 渐行渐远。但论文指出这个解释 **并不完整** ——目标模型的熵 $\mathcal{H}(p)$ 在 RL 中的漂移是另一个更根本的驱动因素。这两个因素还会通过多步验收结构 **乘法累积** ：单步接受率 $\alpha_i$ 下降一点点，$\gamma$ 步 MTP 的期望接受长度 $\mathbb{E}[L] = \sum_{j=1}^{\gamma} \prod_{i=1}^{j} \alpha_i$ 就会被成倍放大。

## 二、核心发现：熵才是接受率的「隐形天花板」

这一节是全文理论部分的核心。固定生成位置 $t$，定义目标熵：

$$
\mathcal{H}(p) = -\sum_{v \in \mathcal{V}} p(v) \log p(v)
$$

熵低意味着分布尖锐（模型很自信），熵高意味着分布摊平（模型很犹豫）。论文分别推导了两种验收方式下，熵如何约束接受率。

### 2.1 Target-Only 采样：接受率被 $\max_y p(y)$ 死死压住

**命题 1（Target-Only 的熵依赖接受率）** ：对于训练良好的 draft 模型（能正确找出 target 的 top-1 token），有 $\alpha^{\mathrm{TO}} = \max_y p(y)$，它是 $\mathcal{H}(p)$ 的单调递减函数，下界为 $\exp(-\mathcal{H}(p))$，经验上可近似为线性：

$$
\alpha^{\mathrm{TO}} \approx a^{\mathrm{TO}} - b^{\mathrm{TO}} \cdot \mathcal{H}(p)
$$

证明思路很直观：由 Jensen 不等式可得 $\log(\max_y p(y)) \geq -\mathcal{H}(p)$，即 $\max_y p(y) \geq \exp(-\mathcal{H}(p))$；再对递减函数 $f(\mathcal{H})$ 在参考熵 $\bar{\mathcal{H}}$ 处做一阶泰勒展开即得线性形式。直觉上： **熵越高，分布越平，top-1 token 的概率质量越小，接受率的天花板就越低** 。若 draft 不完美（top-1 排序出错），只会把斜率压得更陡，线性关系依然保持。

### 2.2 Rejection Sampling：接受率等于分布重叠面积

切到 RS 后，利用恒等式 $|a-b| = a+b-2\min(a,b)$ 可以把 TV 距离展开：

$$
d_{\mathrm{TV}}(p, q) = 1 - \sum_{v} \min\bigl(p(v), q(v)\bigr) \quad\Longrightarrow\quad \alpha^{\mathrm{RS}} = 1 - d_{\mathrm{TV}}(p, q)
$$

表面上看，接受率不再直接被熵约束，而是取决于 $p$ 与 $q$ 的重叠程度——这正是 RS 相比 Target-Only 的本质优势：Target-Only 的上限 $\max_y p(y)$ 随熵升高直接下降，而 RS 看的是 **整个分布的重叠** ，对熵的敏感度天然更低。

### 2.3 但 CE/KL 训练的 draft 依然怕熵

故事到这里还没完。实验发现换成 RS 后，接受率与熵仍然负相关。为什么？答案藏在 **训练目标** 里。

**命题 2（CE/KL 训练下 RS 的熵依赖）** ：CE/KL 的梯度为 $q_j - p_j$，它对每个 token 施加与概率绝对差成正比的优化压力，在容量有限的 draft 模型上会产生 **近似均匀的逐 token 失配** $|\eta_v| \lesssim \sigma$。而高熵分布的有效支撑集大小约为 $|\mathcal{S}_{\mathrm{eff}}| \approx \exp(\mathcal{H}(p))$，这些均匀的小误差会在支撑集上 **逐个累加** ：

$$
d_{\mathrm{TV}} \approx \frac{\sigma}{2}\exp(\mathcal{H}(p)) \quad\Longrightarrow\quad \alpha^{\mathrm{RS}} \approx 1 - \frac{\sigma}{2}\exp(\mathcal{H}(p)) \approx a^{\mathrm{RS}} - b^{\mathrm{RS}} \cdot \mathcal{H}(p)
$$

在 RL 的熵工作区间内对指数做线性化，就得到与 Target-Only 类似的负线性关系（斜率甚至略陡）。一句话总结： **CE/KL 把优化资源均匀撒向整个词表，熵一高，需要照顾的 token 变多，均匀误差就累积成大的 TV 距离** 。

![Entropy vs Accept Length](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/entropy_vs_accept_length.png)

> 图解：横轴为策略模型的平均熵，纵轴为 MTP 接受长度，每个点来自 Qwen3.5/3.6/3.7 不同任务、不同 RL 训练步的实测均值。传统训练方式下两者呈明显的负线性关系；而经过 e2e TV Loss 训练并配合拒绝采样后，曲线几乎被「拉平」——接受率对熵的依赖被大幅消除。

![Draft/Target Distribution](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/draft_target_comparison.png)

> 图解：CE/KL 训练的 draft 分布与 TV 训练的 draft 分布同目标策略分布的对比。TV 训练得到的 draft 分布与 policy 分布的重叠面积显著更大，这正是更高接受率和加速比的来源。

## 三、Bebop 的核心武器：端到端 TV Loss

既然 RS 的接受率等于 $1 - d_{\mathrm{TV}}$，而 CE/KL 只是在间接优化它，那最直接的想法就是： **把 TV 距离本身当作训练目标** 。

### 3.1 为什么 CE/KL 是次优的

由 Pinsker 不等式：

$$
d_{\mathrm{TV}}(p, q) \leq \sqrt{\frac{1}{2} D_{\mathrm{KL}}(p \| q)}
$$

KL 只是 TV 的一个上界。最小化 KL 并不高效地最小化 TV——KL 会把有限的模型容量浪费在与接受决策无关的长尾 token 上。论文强调：问题不在于这个界松不松，而在于 **KL 梯度的容量分配方式** 从根本上就是错的（详见第二节的均匀失配分析）。

### 3.2 TV Loss 及其梯度

直接最小化 TV 距离（$p$ 视为常量，梯度只流过 $q$）：

$$
\mathcal{L}_{\mathrm{TV}} = d_{\mathrm{TV}}(p, q) = 1 - \sum_{v \in \mathcal{V}} \min\bigl(p(v), q(v)\bigr)
$$

设 draft 头输出 logits $z \in \mathbb{R}^{|\mathcal{V}|}$，$q_j = \mathrm{softmax}(z)_j$，则梯度为：

$$
\frac{\partial \mathcal{L}_{\mathrm{TV}}}{\partial z_j} = -q_j \Bigl[ \mathbf{1}[q_j \leq p_j] - S \Bigr], \quad \text{其中} \quad S = \sum_{v} \mathbf{1}[q_v \leq p_v] \cdot q_v
$$

这个梯度有三个非常优雅的性质：

- **有界性** ：$\left|\partial \mathcal{L}_{\mathrm{TV}} / \partial z_j\right| \leq 1$ 恒成立（因为 $q_j \in [0,1]$ 且指示函数与 $S$ 都在 $[0,1]$ 内），训练天然稳定；相比之下 KL 梯度 $q_j - p_j$ 在分布差异大时幅值不可控；
- **选择性** ：对 $q_j \leq p_j$ 的 token（RS 下会被接受的 token）梯度推高 logit；对 $q_j > p_j$ 的 token（会被拒绝的）梯度压低 logit；对 $q_j \approx 0$ 的无关 token，梯度自动约为 0——不在词表长尾上浪费一分算力；
- **与 RS 决策边界对齐** ：指示函数 $\mathbf{1}[q_j \leq p_j]$ 恰好就是拒绝采样的接受判据，梯度信号直接作用在「接受/拒绝」的分界线上。

三种目标的梯度结构对比（论文 Table 1）：

| 性质 | Forward KL (CE) | Reverse KL | TV Loss |
| --- | --- | --- | --- |
| 梯度 | $q_j - p_j$ | $q_j[\log(q_j/p_j) - C]$ | $-q_j[\mathbf{1}[q_j \leq p_j] - C]$ |
| 是否正比于 $q_j$ | 否 | 是 | 是 |
| 长尾抑制 | 无 | 有 | 有 |

注意 Reverse KL 虽然也满足 $q_j$ 正比性和长尾抑制，但它有 **zero-forcing** 特性（允许 draft 丢掉 $p$ 的某些 mode）和 **不对称惩罚** （把 $q$ 压到全局 $q \leq p$），两者都会减少重叠面积 $\sum_v \min(p, q)$，实验中接受率提升微乎其微。综合适用性排序为： **TV Loss > Reverse KL > Forward KL (CE)** 。

### 3.3 端到端多步 TV Loss

$\gamma$ 步 MTP 的期望接受长度为：

$$
\mathbb{E}[L] = \sum_{j=1}^{\gamma} \prod_{i=1}^{j} \alpha_i = \alpha_1 + \alpha_1 \alpha_2 + \alpha_1 \alpha_2 \alpha_3 + \cdots
$$

如果直接优化各步 TV 距离的 **平均值** $\frac{1}{\gamma}\sum_i d_{\mathrm{TV}}(p_i, q_i)$，就完全忽略了多步验收的 **乘法结构** ——前面步骤的接受率出现在更多乘积项里，理应权重更高。于是论文提出 **e2e TV Loss** ：

$$
\mathcal{L}_{\mathrm{e2e}} = 1 - \frac{1}{\gamma} \sum_{j=1}^{\gamma} \prod_{i=1}^{j} \bigl(1 - d_{\mathrm{TV}}(p_i, q_i)\bigr)
$$

这个损失直接优化归一化的期望接受长度，且可以视为一种 **动态的逐步加权机制** ：每一步的有效权重取决于当前 draft 质量，训练会自动把重心移到当前最拖后腿的 step 上。这与此前工作使用的固定位置权重（如 Medusa/EAGLE-3 的 head 权重、DFlash 的指数衰减权重等）形成鲜明对比。

### 3.4 为什么 TV Loss 能「解耦」熵

这是全文理论上最漂亮的一环。关键在于两种训练目标产生 **不同的失配结构** ：

- **CE/KL → 均匀失配** ：$|q^*(v) - p(v)| \lesssim \sigma$，TV 距离随有效支撑集大小 $\exp(\mathcal{H}(p))$ 增长，接受率被熵绑架；
- **TV → 概率比例失配** ：由于 TV 梯度正比于 $q_j$，每个 token 分到的优化资源正比于其概率，最终失配为 $|q^*(v) - p(v)| \lesssim \delta \cdot p(v)$。

**命题 3（TV 训练下熵依赖消失）** ：在概率比例失配下：

$$
d_{\mathrm{TV}}(p, q^*_{\mathrm{TV}}) \leq \frac{\delta}{2} \sum_v p(v) = \frac{\delta}{2}
$$

由于 $\sum_v p(v) = 1$，这个上界 **与熵完全无关** ，于是 $\alpha^{\mathrm{RS}}_{\mathrm{TV}} \geq 1 - \delta/2$。直观地说：熵高时分布摊开，每个 token 分到的优化资源变少，但它在 TV 距离里的权重也同比变少—— **两个效应正好抵消** 。当然，draft 头容量有限，实践中 $\delta$ 仍有微弱的熵依赖，但实验显示熵-接受率斜率被压低了 **95% 以上** 。

## 四、RL 适配策略：一次预训练，全程免更新

理论立住了，接下来是工程上最关心的问题： **RL 过程中到底要不要在线更新 MTP 头？**

### 4.1 分解分析：熵 vs 失配，各占多少锅

利用已建立的线性熵-接受率关系，可以把 RL 第 $t$ 步的接受率变化分解为：

$$
\Delta\alpha_t = \underbrace{b \cdot (\mathcal{H}_t - \mathcal{H}_0)}_{\text{熵驱动项}} + \underbrace{\Delta\alpha_t - b \cdot (\mathcal{H}_t - \mathcal{H}_0)}_{\text{失配残差项}}
$$

其中斜率 $b$ 从每次实验的早期阶段估计，$\mathcal{H}_0$ 为初始熵。第一项刻画「假设 draft-target 关系不变，仅由熵漂移引起的接受率变化」，残差项则捕捉主干权重更新带来的真实失配。

![Decomposition of acceptance changes](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/decomposition_delta.png)

> 图解：RL 训练中接受长度变化的分解。灰色为总变化 $\Delta\alpha$，橙色为熵驱动分量，绿色为 draft-target 失配分量。三个面板分别对应三种配置： **Target-Only** 下熵增与失配都在拖累接受率； **RS + CE Loss** 下退化几乎全部由熵驱动（失配分量贴着零轴），说明 RL 权重更新并不会显著破坏分布重叠； **RS + TV Loss** 下所有分量都接近零，TV 训练的 draft 对熵漂移和权重更新双重免疫。

这个分解直接挑战了「失配中心论」： **接受率波动的主角是熵，失配只是无关紧要的配角** 。

### 4.2 Pre-RL 适配足矣

由分解分析可得关键的实用结论：既然 RS 下 RL 权重更新引起的失配可忽略， **RL 期间更新 MTP 头就是不必要的** 。只需在 SFT 阶段（RL 开始之前）用 e2e TV Loss 做一次性的 MTP 训练，draft 模型就能在整个 RL 过程中保持高接受率。这省掉了 RL 中维护 MTP 优化器状态的显存开销和梯度更新计算。

更有意思的是「反向伤害」现象：如果在 RL 中继续用 CE Loss 更新一个 TV 训练好的 MTP，接受率反而会 **退化** 回 RS w/ CE 的基线水平——因为 CE 会把 draft 分布重新「抹平」，侵蚀 TV 训练的成果。

### 4.3 交叉训练：万不得已时的最佳实践

对于确实需要 MTP 与主干联合训练的场景（例如 Target-Only 采样下失配不可忽略时），论文发现 **分离学习率 + 分离梯度范数归一化** 的联合训练是最佳折中。由于 MTP loss 的梯度只流经 draft 头，主干梯度不受影响，MTP 训练不会干扰 RL 对主干的优化。

## 五、实验验证

实验分三组：(1) SFT 阶段不同多步 MTP 损失对接受率的影响；(2) e2e TV Loss + RS 在 RL 中的接受率、加速比与稳定性；(3) RL 期间更新 MTP 参数的收益。

**实验设置** ：主实验在 Qwen3.5-35A3B 上用混合 RFT 数据训练，恒定学习率 $3.5 \times 10^{-5}$、3% warmup、1 个 epoch、Megatron 框架、global batch 256、序列长度 256K；多步训练时对 5 个 MTP step 做前反向传播并冻结主干；评估统一用 $\gamma = 3$（目标模型每次验证 4 个 token）。吞吐用 SGLang 的 MTP + RS 实现测量。RL 实验基于 veRL 搭建的异步框架，SGLang 作为 rollout 引擎。

### 5.1 SFT 阶段：e2e TV Loss 全面胜出

五种损失（CE / KL / Reverse KL / TV / e2e TV）在 Qwen3.5-35A3B、$\gamma=3$ 下的拒绝采样接受率（%）对比（论文 Table 2，$\Delta$ 为相对 CE 基线的提升）：

| MTP Loss | Math | Code | SWE | Agent | MTBench (OOD) |
| --- | --- | --- | --- | --- | --- |
| CE loss（基线） | 75.0 | 71.3 | 75.1 | 90.3 | 65.3 |
| KL loss | +0.0 | +0.0 | +0.2 | +0.2 | +0.0 |
| Reverse KL loss | +1.3 | +1.0 | −0.2 | +1.0 | +0.5 |
| TV loss | +2.4 | +2.5 | +3.3 | +5.2 | +1.4 |
| **e2e TV loss（本文）** | **+3.0** | **+3.3** | **+8.0** | **+6.7** | **+2.3** |

几个值得注意的观察：

- KL 与 CE 几乎无差别（二者梯度本就相同）；Reverse KL 提升有限，印证了附录中的理论分析；
- e2e TV Loss 在同分布任务上提升 3–8%，OOD 的 MT-Bench 上也有 2.3%；
- **Agent 任务上接受率从 90.3% 推到 97.0%** ——这个水平对 RL rollout 和 Agent 推理效率都是质变；
- e2e TV 的优势在靠后的 MTP step 更明显：step 3 领先 CE 约 5%，step 2 领先 2.5–5%；
- 泛化性强：完全不用 agent 数据训练的模型，在 agent 任务上仍有约 70% 接受率。

![CE vs TV accept rate (reasoning/conversation)](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/ce_vs_tv_accept_rate.png)

> 图解：SFT 训练过程中 CE（实线）与 TV（虚线）在推理与会话类任务（Math、Code、MT-Bench）上各 MTP step 的接受长度曲线。CE 的 step 1 接受率在训练中会出现持续下滑（优化精力被摊薄到整个词表），而 TV 保持稳定甚至缓升。

![CE vs TV accept length (agentic tasks)](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/swe_vending_accept_length.png)

> 图解：在 agentic 与混合任务（Hybrid、Agent、Long-Horizon、SWE-Bench）上的同类对比。TV Loss 的优势在 agentic 任务上被进一步放大，提升最高可达 8%。

而作为对照，在 **Target-Only 采样** 下，各训练目标的接受率几乎完全一致（差异 < 0.3%）：

![Target-only accept length](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/to_ce_tv_accept_length.png)

> 图解：Target-Only 采样下 CE 与 TV 训练的接受长度几乎重合。这符合理论预期：$\alpha^{\mathrm{TO}} = p(\arg\max_y q(y))$ 在 top-1 排序正确时退化为 $\max_y p(y)$，只取决于目标分布本身，与 draft 分布形状无关。TV Loss 的用武之地恰恰是 RS 所依赖的完整分布重叠。

此外，接受率还随模型规模提升：Qwen3.7-Max 在 Agent 任务上接受率达 94.6%、Qwen3.7-Plus 达 98.6%（e2e TV 训练），说明 $\gamma=3$ 下 draft 已几乎收敛到主干；模型变小则接受率不同程度下滑。接受率提升与吞吐提升近似线性相关（跨 8 个模型、3 类任务，相关系数 $r = 0.81$），e2e-TV 训练的 Qwen3.7 Plus 在所有数据集上吞吐都超过 CE 训练的 Qwen3.6 Plus。

### 5.2 RL 阶段：接受率全程稳定，延迟最高降 1.8×

RL 实验选了两类代表性负载：

- **Reasoning RL** ：数学推理、代码推理、指令遵循等长 CoT 任务，最大生成长度 64K；评测基准 HMMT25、AIME25、LiveCodeBench；
- **SWE RL** ：多轮代码编辑任务（思考 → 工具调用 → 工具执行循环），最大生成长度 128K、最多 200 轮；评测基准 SWE-Verified。

![Accept length during Reasoning RL](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_hybrid_accept_len.png)

![Accept length during SWE RL](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_swe_accept_len.png)

![Accept length during SWE RL on Qwen3.7-Max](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_swe_max_accept_len.png)

> 图解：三张子图分别是 Reasoning RL（Qwen3.6-Plus）、SWE RL（Qwen3.6-Plus）和 SWE RL（Qwen3.7-Max）中接受长度随训练步的变化。横轴为 RL step，纵轴为接受长度。 **RS w/ TV（Bebop）全程维持最高且平稳的接受长度** ；Target-Only 持续下滑；RS w/ CE 居中但仍受熵影响。SWE 负载的熵在训练中略微上升，是对训练目标鲁棒性更直接的考验：RS w/ TV 稳如直线，TO 持续退化。更大规模的 Qwen3.7-Max 上，RS w/ TV 的「熵不变性」趋势更加明显。

对应的延迟收益：

![Reasoning RL latency](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_latency.png)

![SWE RL latency](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_swe_latency.png)

![Agent RL latency](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_agent_latency.png)

> 图解：Qwen3.6-35A3B 与 Qwen3.6-Plus 上 RL 训练的单步延迟对比（无 MTP / Target-Only / RS w/ TV 三种配置）。MTP + 拒绝采样将每步 RL 训练延迟降低 **1.5–1.8×** ；在 Agentic RL 中 rollout 阶段加速最高达 **2.4×** 。在大规模训练（数十万 GPU 小时量级）下，这是非常可观的墙钟时间节省。

熵-接受率关系的直接验证：

![Entropy vs accept length (Reasoning RL)](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/entropy_vs_accept_len_hybrid.png)

![Entropy vs accept length (SWE RL)](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/entropy_vs_accept_len_swe.png)

![Entropy vs accept length (SWE RL, Qwen3.7-Max)](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/entropy_vs_accept_len_swe_max.png)

> 图解：三种 RL 负载下，横轴为熵、纵轴为接受长度，每个点为一个训练步，直线为线性拟合。TO 与 RS w/ CE 呈强负相关（斜率约 **−1.68** ），而 RS w/ TV 几乎水平（斜率约 **−0.06** ）——斜率压缩超过 95%，同时截距整体上移。这从实验上确认了 TV 训练既提升了分布对齐度，又把接受率从熵的枷锁中解放了出来。

### 5.3 RL 期间更新 MTP 权重：得不偿失

论文对比了三种在线更新配置（RS w/ TV + TV loss、RS w/ TV + CE loss、TO + CE loss）：

![Accept length with MTP updates](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_hybrid_mtp_train_accept_len.png)

> 图解：RL 中更新与不更新 MTP 权重的接受长度对比。随着训练推进， **在线更新的接受率会向对应的不更新基线收敛** ：RS w/ TV 起点更高，但用 CE loss 在线更新会让它逐渐退化到 RS w/ CE 的水平；已经训好的 MTP 继续更新也无显著收益；Target-Only 下用 CE 更新甚至因失配而变差。

![Accept rate delta vs throughput](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/accept_delta_vs_throughput.png)

> 图解：横轴为拒绝采样带来的接受率增量（RS − No-RS），纵轴为吞吐加速比（RS / No-RS），覆盖 8 个模型、3 类任务，相关系数 $r = 0.81$。接受率收益几乎线性地转化为吞吐收益，说明「提升接受率」这条技术路线的投入产出非常直接。

## 六、Discussion：机制层面的深入剖析

### 6.1 TV Loss 让 draft 分布更「尖锐」

TV 训练产生的 draft 分布熵更接近目标熵（略高一点），即分布更尖锐、更贴合目标的峰值预测；CE/KL 则倾向产生更平滑、把概率质量摊向整个词表的分布——这对追求重叠面积最大化的 RS 是次优的。这种锐化来自 TV 梯度的选择性：它聚焦在决策边界附近（$q_j \approx p_j$）的 token，忽略无关长尾。

![Entropy gap vs KL](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/entropy_gap_vs_kl.png)

![Entropy gap vs RS accept rate](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/entropy_gap_vs_rs_accept.png)

![KL vs RS accept rate](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/kl_vs_rs_accept.png)

> 图解：三幅散点图横跨多个模型与任务。(a) draft-target 熵差 $\Delta H$ 与 $D_{\mathrm{KL}}(q\|p)$ 的关系：MTP 头训得好的模型熵差更小，但 KL 距离反而更大；(b) 熵差与 RS 接受率负相关（$r = -0.54$）；(c) KL 距离与 RS 接受率几乎无关（$r = 0.13$）。结论很反直觉但很重要： **预测 RS 接受率的有效指标是熵差，而不是 KL** ——这也再次说明以 KL 为训练目标是南辕北辙。

### 6.2 不同 Loss 塑造不同的分布模式

![MTP metrics during RL](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rl_mtp_metrics.png)

> 图解：RL 中用不同 loss 更新 MTP 时各项指标的演化。TV loss 使 draft 熵更贴近目标（但 KL 距离更大），且因分布更尖锐，$\alpha_{p>q}$ 更低、$\alpha_{q>p}$ 更高；中途切换 loss（如 RS w/ TV + CE），各项指标会逐渐滑向新 loss 的特征模式（draft 熵逐渐升高）。这从机制上解释了 5.3 节中「CE 在线更新侵蚀 TV 成果」的现象。

### 6.3 策略更新下的鲁棒性：离散 vs 连续

两种验收方式面对 RL 权重更新的「体质」不同：

- **Target-Only 对排名翻转脆弱** ：其接受判据是 **离散** 的——token 要么被接受要么被拒绝。RL 一步梯度让 top-1 易主（如 $p(v_1): 0.31 \to 0.29$ 而 $p(v_2): 0.29 \to 0.31$），仍钟情旧 top-1 的 draft 就会经历从「接受」到「拒绝」的跳变；
- **Rejection Sampling 平滑退化** ：$\alpha^{\mathrm{RS}} = \sum_v \min(p(v), q(v))$ 是两个分布的 **连续** 函数，同样的排名交换对 TV 重叠的影响微乎其微；
- **高熵放大脆弱性差** ：熵高时多个 token 概率接近，排名翻转更频繁，Target-Only 的离散失效被进一步放大。

有趣的是，实验观察到 TO 与 RS（CE 训练下）的熵-接受率斜率相近（$b^{\mathrm{TO}} \approx b^{\mathrm{RS}}$）——TO 的离散脆弱性与 CE 训练下 RS 的 TV 累积效应在量上恰好「打了个平手」，只是成因不同。

### 6.4 温度的影响

采样温度 $\tau$ 直接推高目标熵（$\mathcal{H}(p_\tau) = \mathcal{H}(\mathrm{softmax}(z/\tau))$ 随 $\tau$ 单调增），结合线性关系可推知： **温度越高，MTP 接受率越低** 。

![Accept length vs temperature](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/accept_length_vs_temp.png)

![Accept rate vs output length](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/accept_rate_vs_position.png)

> 图解：(a) 横轴为采样温度，纵轴为平均接受长度。RS 在各温度下保持相对稳定，而 Target-Only 在高温区急剧劣化——这对常用高温鼓励探索的 RL 训练尤有实际意义，该分析为「探索的温度成本」提供了定量的吞吐核算框架。(b) 横轴为生成位置（输出长度），纵轴为接受率（8 个模型平均）：RS 在所有生成位置上都稳定优于 TO。

### 6.5 RS 决策边界：什么时候该开拒绝采样

RS 优于 Target-Only 的充要条件可以推导出非常简洁的形式：

$$
1 - d_{\mathrm{TV}}(p, q) > p(\hat{y}) \quad\Longleftrightarrow\quad d_{\mathrm{TV}}(p, q) < 1 - p(\hat{y})
$$

即： **只要 draft-target 的 TV 距离小于目标分布在 draft top-1 token 之外的概率质量，就该用 RS** 。

![RS decision boundary](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/rs_decision_boundary.png)

> 图解：8 个原生 MTP 模型 × 3 类任务共 24 个组合在决策边界图中的位置，23/24 都稳稳落在「RS 更优」区域。结论：对于原生 MTP 模型，几乎所有实际部署场景都应该开启拒绝采样。

### 6.6 生成长度与位置效应

接受率随生成位置系统性变化：靠近 prompt 的早期位置熵低（续写更可预测），接受率高；随着生成长度增加（尤其长 CoT 推理），熵升高、接受率可能下降。这提示了 **自适应 MTP 策略** 的优化空间——根据局部熵估计动态调整 draft 长度 $\gamma$。

### 6.7 Agentic RL 与气泡问题

![Accept length during Agent RL](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/agent_accept_len_csv.png)

> 图解：Agent RL（如 SWE-bench 多轮工具交互）中的接受长度。均值稳定在约 3.7，但 min–max 区间呈现随生成推进而加剧的周期性波动——多轮交互与长生成带来了多变的熵剖面。

MTP 在 agentic 场景收益特别大，原因有二：(1) 长生成中包含大量结构化输出（模板代码、工具调用格式、重复模式），高度可预测，这些区段接受率极高；(2) 多轮交互与长尾生成会缩减有效 batch size，推理引擎远离计算饱和区，MTP 的延迟收益被放大。实验中 agentic 负载从 TV Loss 获得的接受率提升也最大（5%）。

### 6.8 Top-K 截断近似的不稳定性

全词表 TV Loss 在大词表上峰值显存很高。论文的解法是 **fused backward kernel** （见附录精选）；也试过 top-$K$ 截断近似来进一步省显存，但即便 $K = 20{,}000$ 也会拖慢收敛并损失性能，更小的 $K$ 则出现明显的 loss 尖峰。

![MTP loss under top-K truncation](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Breaking-Entropy-Bounds-Accelerating-RL-Training-via-MTP-with-Rejection-Sampling/figs/mtp_lm_loss_topk.png)

> 图解：不同 top-$K$ 截断下的 MTP loss 曲线。$K$ 越小 loss 尖峰越明显、训练越不稳定；$K = 20{,}000$ 的收敛仍慢于全词表 TV loss。最终论文采用 fused 全词表实现而非截断近似。

## 七、附录精选：关键推导与工程实现

考虑到「详细」档位的定位，这里把附录中最有价值的推导和系统实现梳理出来，方便复现。

### 7.1 TV Loss 梯度的完整推导

由 $\min(p_v, q_v)$ 对 $q_v$ 的次梯度 $\mathbf{1}[q_v \leq p_v]$，结合 softmax 雅可比 $\partial q_v / \partial z_j = q_v(\delta_{vj} - q_j)$：

$$
\frac{\partial}{\partial z_j} \sum_v \min(p_v, q_v) = \sum_v \mathbf{1}[q_v \leq p_v] \cdot q_v(\delta_{vj} - q_j) = q_j\bigl[\mathbf{1}[q_j \leq p_j] - S\bigr]
$$

从而 $\partial \mathcal{L}_{\mathrm{TV}} / \partial z_j = -q_j[\mathbf{1}[q_j \leq p_j] - S]$，且 $|\cdot| \leq q_j \leq 1$。作为对比，forward KL 的梯度 $q_j - p_j$ 有三个短板：对每个 $q_j \neq p_j$ 的 token（含概率可忽略者）都施加非零力；不区分 RS 下会被接受/拒绝的 token；当 draft 过度自信（$q_j \gg p_j$）时梯度幅值很大。

### 7.2 Reverse KL 为什么仍然不行

Reverse KL 的梯度可推得 $\partial D_{\mathrm{KL}}(q\|p) / \partial z_j = q_j[\log(q_j/p_j) - D_{\mathrm{KL}}(q\|p)]$，结构上同样正比于 $q_j$、同样抑制长尾。但它有三个致命伤：

- **Zero-forcing** ：$q(v) \to 0$ 时 $q\log(q/p) \to 0$，不惩罚丢 mode 的行为，直接放弃这些 token 上的重叠 $\min(p(v), q(v))$；
- **不对称惩罚** ：高估（$q_j > p_j$）的梯度远强于低估，把 draft 推向全局 $q \leq p$，单 token 接受概率虽为 1，但这些 token 被采到的概率变低，总重叠反而次优；
- **间接目标** ：$\log(q_j/p_j)$ 是软的非线性信号，不如 TV 的指示函数那样直接对齐 RS 决策边界。

### 7.3 熵-接受率关系的完整图景

附录把第二节的分析统一在一个框架下：CE/KL 的均匀失配 $|q-p| \lesssim \sigma$ 在有效支撑集上累积出 $d_{\mathrm{TV}} \approx \frac{\sigma}{2}\exp(\mathcal{H}(p))$；而 TV 训练通过自纠错机制（定义 $r_j = q_j/p_j$，$r_j < 1$ 时梯度推升 logit、$r_j > 1$ 时压低，驱动 $r_j \to 1$）在有效支撑集上形成有界的 log-ratio 误差 $|\log(q/p)| \leq \epsilon$，进而得到概率比例失配 $|q - p| \lesssim (e^{\epsilon}-1)\, p$，最终把 TV 距离压到与熵无关的 $\delta/2$。作者也坦诚：Adam 的二阶矩归一化会部分削弱原始的 $q_j$ 正比性，所以比例失配应视为建模近似而非无条件定理。

### 7.4 Fused TV Loss Kernel

为了在大词表上算全量 TV Loss 而不爆显存，论文实现了融合 kernel：前向在单次 kernel launch 中完成 softmax 归一化与重叠/$S$ 累积（按 `BLOCK_V` 分块遍历词表，不物化完整 softmax 输出）；反向复用缓存的 $(m, D, S)$ 计算 $\nabla_z \ell = q \cdot (S - 1 + \mathbf{1}[q > p]) \cdot g_{\text{out}}$。张量并行下，全局 max $m$ 与 exp-sum $D$ 通过 `all_reduce` 跨 TP rank 聚合，局部 overlap 与 $S$ 计算后再做归约。

### 7.5 拒绝采样的推理引擎实现

RS 落地需要同时改 draft 与 verify 两个阶段（采样而非 argmax、缓存 draft 概率、验证时算 $\min(1, p/q)$），论文给出两套开源实现：

- **SGLang（Multinomial 版）** ：draft 阶段对 logits 做温度缩放后多项式采样并缓存完整 $q$ 向量；验证阶段用融合 Triton kernel 做序贯接受（$u_i \cdot q_i(\hat{y}_i) < p_i(\hat{y}_i)$ 则接受，首个拒绝处停止），再从残差分布 $p_{\mathrm{resid}}(v) \propto \max(0, p_j(v) - q_j(v))$ 用两遍 CDF 反演重采样。主要显存开销是每请求 $O(\gamma \times |\mathcal{V}|)$ 的 draft 概率缓存；
- **vLLM（Gumbel-Max 版）** ：draft 采样用 Gumbel-Max 技巧（$v^* = \arg\max_v [\log q(v)/\tau + G_v]$），验证拆成两个 kernel——序贯接受 kernel 记录首个拒绝位置，残差 logits kernel 在 logit 空间算 $\log\max(0, p_j - q_j)$，最后同样用 Gumbel-Max 重采样，避免了显式 CDF 反演。

## 八、相关工作与总结

**与现有工作的关系** ：投机解码方向的 draft 架构百花齐放（独立小模型、early-exit、Medusa/EAGLE 辅助头、MTP 头、扩散模型等），Bebop 专注于共享主干 hidden state 的原生 MTP 头在 RL 动态下的行为。RL 系统侧的异步框架主要隐藏长尾气泡，不改变轨迹生成是瓶颈的事实。此前也有工作（如 LK Losses）尝试面向接受率的目标，但针对的是固定 target 的推理时投机解码；近期 OPD 工作用 reverse KL 优化 student，目标也与直接最大化 RS 接受率相去甚远。 **本文是第一个将 TV 距离直接作为 MTP 头训练目标、并第一个分析其在 RL 训练中行为的工作** 。MTP 与 RL 目标、调度器完全正交，可以即插即用。

**三个核心结论** ：

1. Target-Only 与 Rejection Sampling 下的 MTP 接受率都被目标模型熵 **线性约束** ，熵漂移（而非分布失配）是 RL 中接受率波动的主因；
2. e2e TV Loss 直接优化多步 RS 接受率，带来约 10% 的接受率提升、最高 95% 的接受率、最高 25% 的额外推理吞吐，并把熵-接受率斜率压低 95% 以上；
3. 轻量的 pre-RL 适配（e2e TV Loss 训练 + RS 推理）即可让接受率贯穿 RL 全程保持稳定， **无需昂贵的在线 MTP 更新** ，最终在异步 RL 流水线中实现最高 1.8× 端到端加速。

**局限性** （作者自述，也值得关注）：熵-接受率分析依赖「均匀失配 vs 概率比例失配」的建模假设，这些假设由梯度结构启发而来，尚未被严格证明；TV 训练保证的熵不变性是 **分布条件性** 的——只在 SFT 数据覆盖的熵范围内成立。若 RL 探索把策略熵推出该范围，draft 头会遇到 OOD 的目标分布，失配比 $\delta$ 不再有界，熵-接受率依赖会回到 CE/KL 的水平，此时建议在 RL 中用 TV Loss 做 MTP 协同训练以扩展覆盖范围。

对从业者而言，这篇论文的实操建议可以浓缩成一句话： **给你的 MTP 换上 TV Loss，打开拒绝采样，然后放心地让它陪你跑完整个 RL——熵再也奈何不了你。**

> 本文参考自 [Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling](https://arxiv.org/pdf/2606.12370)