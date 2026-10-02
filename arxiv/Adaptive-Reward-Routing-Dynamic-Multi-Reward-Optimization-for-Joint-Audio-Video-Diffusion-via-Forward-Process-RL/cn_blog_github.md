# Adaptive Reward Routing：让多奖励 RL 的“更新位置”和“奖励配比”都跟着模型一起进化

音视频联合生成模型（如 LTX-2）的后训练需要同时优化画质、音质、文本对齐和音画同步等多个目标，多奖励强化学习是目前的主流做法。但这条路线有两个被忽视的问题： **奖励驱动的梯度应该打在模型的哪些位置** ，以及 **多个互相冲突的奖励该如何动态权衡** ，这两者都会随训练进程不断变化。这篇文章提出 Adaptive Reward Routing，用跨模态注意力响应作为“影响力探针”动态定位更新位置（token 级 + 层级），同时在保留用户偏好先验的前提下用梯度冲突信号对奖励权重做残差修正。在 JavisBench 上，该方法在 LTX-2 和 LTX-2.3 两个骨干网络的 10 项指标中各自拿下 9 项最优，例如在 LTX-2.3 上 DeSync（越低越好）从 OmniNFT* 的 0.335 降到 **0.302** ，视频质量 VQ 从 3.537 提升到 **3.599** 。

## 一、问题：多奖励 RL 的两个“静态”陷阱

### 1.1 背景：音视频联合生成的多目标困境

近年来，以 LTX-2 为代表的联合音视频扩散模型已经能从一段文本同时生成画面和声音。但“好”的音视频内容需要同时满足四个维度的要求：

- 视频本身的视觉质量
- 音频本身的听觉质量
- 音画与文本的语义对齐
- 音频与画面的时间同步（如嘴型对齐）

单一监督目标很难同时刻画这些要求，因此奖励引导的扩散 RL（包括 GRPO 系方法和 DiffusionNFT）成为自然的后训练范式：把每个要求写成一个奖励信号，让模型在多奖励下优化。

问题在于，现有的多奖励 RL 方法在两个关键维度上都是 **静态** 的，而模型本身是 **动态** 演化的。

### 1.2 陷阱一：更新位置是死的（Where to Optimize）

联合音视频模型通常由音频、视频两条分支组成，二者通过双向 cross-attention 耦合。奖励路由器首先要决定每个奖励该由哪条分支负责，而分支级的更新还需要进一步定位到具体的 token 和跨模态交互层。

此前为 DiffusionNFT 设计的 OmniNFT 虽然意识到了这一点，但它的层路由规则是基于 **基座模型** 一次性算好、之后固定不变的。作者对基座模型和 OmniNFT 训练后的 checkpoint 分别做了探查（见下图 a、b），发现跨模态的功能分布和梯度路径在微调过程中发生了明显漂移——静态路由会随训练推进而逐渐“过时”。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Adaptive-Reward-Routing-Dynamic-Multi-Reward-Optimization-for-Joint-Audio-Video-Diffusion-via-Forward-Process-RL/images/teaser.png)

> 图解：联合音视频后训练为什么需要自适应的路由与奖励协调。 **(a)** 前向 KV 消融：按层段屏蔽跨模态 K/V 输出，观察 DeSync 指标的退化程度（退化越大说明该层段对同步越重要），基座与训练后模型的曲线明显分叉，说明“同步关键层”在训练中发生了迁移。 **(b)** 反向梯度分析：统计各层 Q（分支内）与 K/V（跨模态）梯度的 $\ell_2$ 范数，基座与训练后模型的峰值位置错位，说明跨模态梯度路径也在演化。 **(c)** 奖励冲突矩阵：不同奖励在同一批样本上的归一化偏好经常符号相反，固定的分支分配无法处理。 **(d)** 纯梯度几何权重的漂移：完全由梯度几何决定的权重轨迹中，AV-DeSync 的权重几乎衰减到零——纯几何加权会把“弱但关键”的目标直接丢弃。

### 1.3 陷阱二：奖励配比是死的（How to Balance）

多个奖励经常在同一个样本上给出相反的评价，而且合适的配比会随优化进程变化。现有方案各有短板：

- **GDPO** ：对每个奖励独立归一化，但用 **固定权重** 聚合，冲突完全得不到适配；
- **MARBLE** ：根据梯度几何动态调权，但算出来的系数反映的是“梯度兼容性”，而不是与用户偏好对齐的“重要性”。上图 (d) 显示，几何-only 的权重会让 AV-DeSync 这类关键目标趋于消失。

一句话总结：有效的后训练需要 **冲突感知** 的动态适配，同时以用户定义的优先级为锚点。这正是 Adaptive Reward Routing 要解决的问题。

## 二、预备知识：LTX-2 与 DiffusionNFT

在介绍方法之前，先把两个技术底座讲清楚。

### 2.1 双流 Flow Matching 架构

LTX-2 采用音频、视频两条独立的流，共享时间步。每个模态的隐变量遵循标准线性插值 $x_t^m=(1-t)x_0^m+t x_1^m$，其中 $x_1^m\sim\mathcal N(0,I)$，模型联合预测两个速度场。两条流通过双向 cross-attention 交换信息：

$$
o_{a\rightarrow v}^{l,t}
=\operatorname{Attn}\!\left(Q_v(h_v^{l,t}),K_a(h_a^{l,t}),V_a(h_a^{l,t})\right),\quad
o_{v\rightarrow a}^{l,t}
=\operatorname{Attn}\!\left(Q_a(h_a^{l,t}),K_v(h_v^{l,t}),V_v(h_v^{l,t})\right)
$$

即 A2V（音频到视频）方向里，视频 token 做 Query、音频 token 提供 K/V；V2A 方向反之。门控后的输出分别加回视频流和音频流。这个双向耦合结构既是“互相条件化”的来源，也让“奖励该往哪里打梯度”变得不再显然。

### 2.2 DiffusionNFT：前向过程 RL

DiffusionNFT 不优化反向采样链，而是直接在 **前向过程** 上构造一对隐式的正、负策略：

$$
v_\theta^+=(1-\beta)v^{\mathrm{updated}}+\beta v_\theta,
\qquad
v_\theta^-=(1+\beta)v^{\mathrm{updated}}-\beta v_\theta
$$

对每个 prompt，旧策略生成一组 $N$ 个样本，把奖励转成组内相对优势：

$$
A^{(n)}=\frac{R^{(n)}-\mu_R}{\sigma_R+\varepsilon},
\qquad
r^{(n)}=\frac{1}{2}+\frac{1}{2}\operatorname{clip}\!\left(\frac{A^{(n)}}{A_{\max}},-1,1\right)
$$

其中 $\mu_R$、$\sigma_R$ 在 rollout 组内计算。$r^{(n)}>1/2$ 表示该样本优于同组平均，偏向正策略；反之偏向负策略。最终目标为：

$$
\mathcal L_{\mathrm{NFT}}
=\mathbb E_{n,t}\!\left[
r^{(n)}\|v_\theta^+(x_t^{(n)},c,t)-u^{(n)}\|_2^2
+(1-r^{(n)})\|v_\theta^-(x_t^{(n)},c,t)-u^{(n)}\|_2^2
\right]
$$

这个优势估计不需要学习 value function，只表达“样本在同组中处于平均线之上还是之下”。

### 2.3 多奖励聚合的现状

记奖励集合为 $\mathcal K=\mathcal K_v\cup\mathcal K_a\cup\mathcal K_c$（视频、音频、跨模态）。每个奖励独立做组内归一化得到 $A_k^{(n)}$ 后：

- **GDPO** ：固定权重线性组合 $A_{\mathrm{GDPO}}^{(n)}=\sum_k\omega_k^{\mathrm{prior}}A_k^{(n)}$——保留了显式偏好，但无法适配冲突；
- **MARBLE** ：在单纯形上选权重使加权归一化梯度和的范数最小——适配了局部梯度几何，但不编码偏好和模态责任。

本文的思路是 **不做二选一** ：偏好先验打底，冲突信号做残差修正。

## 三、方法：Adaptive Reward Routing

Adaptive Reward Routing 在四个层面上适配奖励驱动的优化：奖励重加权 → 模态分支分配 → token 级信用分配 → 层级跨模态梯度路由。整体流水线如下：

$$
\{A_k\}
\xrightarrow{\text{reward reweighting}}
\{\omega_{m,k}A_k\}
\xrightarrow{\text{branch routing}}
(A_v,A_a)
\xrightarrow{\text{token routing}}
\mathcal{L}
\xrightarrow{\text{layer routing}}
\nabla_\theta\mathcal{L}
$$

直觉上：奖励重加权决定每个目标贡献多强，分支路由把目标指派给负责的模态，token 路由选择模态损失应强调哪些位置，层路由控制梯度如何跨越模态边界流动。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Adaptive-Reward-Routing-Dynamic-Multi-Reward-Optimization-for-Joint-Audio-Video-Diffusion-via-Forward-Process-RL/images/framework.png)

> 图解：Adaptive Reward Routing 总览。左侧多奖励各自归一化后进入“偏好保持的模态感知重加权”，得到分支级优势；中间“跨模态影响力引导路由”利用双向 cross-attention 的 pre-gate 响应生成 token 权重 $\lambda_{m,i}$ 和层级软化系数 $\alpha_m^l$；右侧完成路由后的 loss 反传，梯度按影响力在跨模态路径上有选择地保留或切断。

### 3.1 跨模态影响力引导路由：定位更新位置

最直接的影响力度量是“禁用 A2V 或 V2A 后比较速度预测的变化”，但训练中反复做这种干预需要大量额外前向计算。作者的替代方案非常经济： **直接用前向过程本来就产生的量** ——对应跨模态注意力路径的 pre-gate 响应范数。对目标 token $i$：

$$
d_{v,i}^{l,t}=\|o_{a\rightarrow v,i}^{l,t}\|_2,
\qquad
d_{a,i}^{l,t}=\|o_{v\rightarrow a,i}^{l,t}\|_2
$$

这些方向性响应在一个中后段去噪时间窗内收集，并在进入策略优化前 detach。后文第五节会验证这个代理与直接干预的高度一致性。

**Token 级路由。** 对每个目标 token，在选定时间步和跨模态块上平均响应，经 99 分位截断的 min-max 归一化（$\operatorname{Norm}_{99}$）后变成正的 loss 权重：

$$
\lambda_{m,i}=1+(\lambda_{\max}-1)\operatorname{Norm}_{99}\!\left(
\frac{1}{|\mathcal B||\mathcal T|}\sum_{l\in\mathcal B}\sum_{t\in\mathcal T}d_{m,i}^{l,t}
\right)
$$

注意一个细节：音频响应是全局归一化的，而视频响应在 **每一帧内部** 归一化，避免帧间幅值差异主导权重分布。

**层级路由。** 换个聚合方向——对每一层，把同样的响应在 token 和时间步上平均，得到层分数 $\widetilde\delta_m^l$（块间 min-max 归一化），再转成“软 detach 系数”：

$$
\alpha_m^l=(1-\widetilde\delta_m^l)^{1/\tau}
$$

对源端 K/V 张量 $X\in\{K,V\}$，路由后的表示为：

$$
\widetilde X_{\bar m\rightarrow m}^{l,t}
=\alpha_m^l\operatorname{sg}(X_{\bar m}^{l,t})+(1-\alpha_m^l)X_{\bar m}^{l,t}
$$

其中 $\operatorname{sg}$ 是 stop-gradient。这个操作的巧妙之处在于： **前向值完全不变** （两项相加恒等于 $X$），但反向梯度被缩放为 $1-\alpha_m^l$。影响力强的层保留更多梯度，耦合弱的层逐渐被 detach；A2V 与 V2A 两个方向独立路由。

> 博主点评：这个“前向不动、只调梯度”的设计很聪明——它不改变模型当前的推理行为，因此不会在训练中引入额外的分布漂移，只重塑梯度的流动拓扑。相当于在不改水路的情况下调整各支流的闸门开度。

### 3.2 偏好保持的模态感知重加权：协调奖励冲突

**动机。** 预定义奖励权重表达“用户想要什么”，但不能响应奖励冲突；梯度几何系数能响应冲突，却可能因为某个目标早期梯度噪声大、方向不合群就把它压制掉。作者把两者结合而非二选一。

**实现。** 每个奖励只通过它所监督的分支来探查梯度：视频/音频奖励走各自分支，跨模态奖励走两条分支。MARBLE 在每个分支内产生冲突感知系数 $\gamma_{m,k}$。探查期间 **关闭 token 路由** ，避免当前 token 权重污染几何测量。warm-up 之后，平滑后的系数作为对先验的残差修正：

$$
\omega_{m,k}=
\begin{cases}
\omega_{m,k}^{\mathrm{prior}}, & e<e_{\mathrm{warm}},\\
(1-\kappa)\omega_{m,k}^{\mathrm{prior}}+\kappa C_m\bar\gamma_{m,k}, & e\ge e_{\mathrm{warm}}.
\end{cases}
$$

其中 $C_m$ 把单纯形系数重新缩放，保持分支 $m$ 内的总先验权重不变；$\bar\gamma_m\leftarrow\rho\bar\gamma_m+(1-\rho)\gamma_m^*$ 对连续估计做指数平滑。先验因此提供了一个 **非零下限** ，残差项负责适配当前的冲突格局——用户偏好不会被覆盖，弱但关键的目标也不会被强势奖励吞掉。

### 3.3 完整训练目标

自适应奖励权重先为每个模态分支生成独立的优势：

$$
A_m^{(n)}
=
\sum_{k\in\mathcal K_m\cup\mathcal K_c}
\omega_{m,k}A_k^{(n)},
\qquad m\in\{v,a\}
$$

跨模态奖励同时计入两个分支。每个分支优势再映射为最优概率 $r_m^{(n)}=r(A_m^{(n)})$。样本 $n$ 的 token $i$ 上的负感知损失为：

$$
\ell_{m,i}^{(n)}
=
r_m^{(n)}
\frac{\|v_{\theta,m,i}^{+}-u_{m,i}\|_2^2}{w_m^{+,(n)}+\varepsilon}
+
\bigl(1-r_m^{(n)}\bigr)
\frac{\|v_{\theta,m,i}^{-}-u_{m,i}\|_2^2}{w_m^{-,(n)}+\varepsilon}
$$

其中 $w_m^{\pm,(n)}$ 是对应策略的 detach 平均绝对残差（对模态 $m$ 的全部 token 和特征维度取平均）。token 路由权重构成模态损失：

$$
\mathcal L_m^{\mathrm{policy}}
=
\mathbb E_n\!\left[
\frac{
\sum_{i\in\mathcal I_m}
\lambda_{m,i}^{(n)}\ell_{m,i}^{(n)}
}{
\sum_{i\in\mathcal I_m}\lambda_{m,i}^{(n)}
}
\right]
$$

最后合并两个分支并加 KL 正则（向固定参考策略靠拢）：

$$
\mathcal L(\theta)
=
\sum_{m\in\mathcal M}\mathcal L_m^{\mathrm{policy}}
+
\lambda_{\mathrm{KL}}
\sum_{m\in\mathcal M}\mathcal L_{\mathrm{KL},m}(\theta)
$$

## 四、实验

### 4.1 实验设置

- **骨干网络** ：LTX-2（19B）与 LTX-2.3（22B），均为双流 + 双向 cross-attention 架构；
- **训练数据** ：19,487 条源自 VGGSound 语料的音视频 prompt，每条含模态级描述和联合 prompt；
- **奖励模型** （5 个）：视频质量用 VideoAlign 和 HPSv3；音频质量用 AudioBox Aesthetics；文本-音频对齐用 CLAP；音画同步用 DeSync（训练时转为越高越好的 AV-DeSync，评估时报原始越低越好的 DeSync）；
- **对比基线** ：Base Model（无后训练）、GDPO（固定配比）、MARBLE（全局冲突感知调权）、OmniNFT（静态多模态路由，含作者发布的 checkpoint OmniNFT*）；
- **评测基准** ：JavisBench 全量 10,140 条 prompt，生成统一归一化为 4 秒、24 FPS、16 kHz 音频，报告 AV-Quality（VQ、AQ）、Text-Consistency（TV-IB、TA-IB、CLIP、CLAP）、AV-Consistency（AV-IB、AVHScore）、AV-Synchrony（JavisScore、DeSync）四组共 10 项指标。

### 4.2 主结果

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Adaptive-Reward-Routing-Dynamic-Multi-Reward-Optimization-for-Joint-Audio-Video-Diffusion-via-Forward-Process-RL/images/video.png)

> 图解：定性对比。每个 prompt 展示 LTX-2、OmniNFT、本文方法生成的 5 帧时间有序画面，覆盖中文新闻播报、风格化英语演讲、双人对话和狗叫场景。LTX-2 在人物和动物样例上有明显的主体/外观漂移，OmniNFT 提升了 prompt 忠实度但仍有时间不一致，本文方法的身份与场景结构最稳定。

定量结果如下（3 个随机种子的均值，加粗为最优，下划线为次优）：

**LTX-2 骨干：**

| 方法 | VQ↑ | AQ↑ | TV-IB↑ | TA-IB↑ | CLIP↑ | CLAP↑ | AV-IB↑ | AVHScore↑ | JavisScore↑ | DeSync↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Base Model | 1.883 | 5.201 | 0.265 | 0.143 | 0.312 | 0.358 | 0.180 | 0.177 | 0.153 | 0.604 |
| + GDPO | 2.722 | 5.450 | 0.261 | 0.138 | 0.312 | 0.347 | 0.174 | 0.175 | 0.155 | 0.671 |
| + MARBLE | 2.384 | 5.100 | 0.265 | 0.138 | 0.311 | 0.365 | 0.182 | 0.182 | 0.158 | 0.618 |
| + OmniNFT | 3.136 | 5.614 | 0.265 | 0.145 | 0.312 | 0.416 | 0.222 | 0.219 | 0.195 | 0.390 |
| + OmniNFT* | 3.278 | 5.609 | 0.254 | 0.165 | **0.314** | 0.422 | 0.220 | 0.219 | 0.193 | 0.378 |
| + Ours | **3.336** | **5.868** | **0.268** | **0.167** | 0.314 | **0.425** | **0.235** | **0.234** | **0.206** | **0.341** |

**LTX-2.3 骨干：**

| 方法 | VQ↑ | AQ↑ | TV-IB↑ | TA-IB↑ | CLIP↑ | CLAP↑ | AV-IB↑ | AVHScore↑ | JavisScore↑ | DeSync↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Base Model | 2.032 | 5.218 | 0.271 | 0.151 | 0.308 | 0.387 | 0.205 | 0.202 | 0.175 | 0.504 |
| + GDPO | 2.929 | 5.251 | 0.272 | 0.144 | 0.309 | 0.376 | 0.218 | 0.199 | 0.178 | 0.560 |
| + MARBLE | 2.582 | 5.176 | 0.271 | 0.147 | 0.309 | 0.394 | 0.217 | 0.207 | 0.182 | 0.496 |
| + OmniNFT | 3.489 | 5.693 | 0.271 | 0.163 | **0.316** | 0.449 | 0.250 | 0.238 | 0.224 | 0.369 |
| + OmniNFT* | 3.537 | 5.702 | 0.260 | 0.174 | 0.311 | 0.456 | 0.252 | 0.249 | 0.221 | 0.335 |
| + Ours | **3.599** | **5.979** | **0.274** | **0.177** | 0.314 | **0.460** | **0.267** | **0.266** | **0.236** | **0.302** |

几个值得注意的观察：

- **GDPO 暴露了固定配比的失衡** ：它提升了视觉质量，却反而恶化了多项音频与同步指标（LTX-2 上 DeSync 从 0.604 涨到 0.671）；
- **MARBLE 全局缓解冲突、OmniNFT 引入模态感知优化** ，但两者都只适配了“协调”或“路由”中的一个维度；
- 本文方法在两个骨干上各拿下 10 项指标中的 **9 项最优** ，且趋势一致——说明增益来自协同的多奖励优化，而非牺牲某一目标换取另一目标。

### 4.3 训练动态

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Adaptive-Reward-Routing-Dynamic-Multi-Reward-Optimization-for-Joint-Audio-Video-Diffusion-via-Forward-Process-RL/images/reward.png)

> 图解： **(a)** 五个单项奖励（AudioBox、AV-DeSync、CLAP、HPSv3、VideoAlign）及其均值随训练的轨迹。本文方法的平均奖励最高，且五个分项的轨迹都比较健康；基线方法在音频、视频、同步目标之间的进展不均衡。 **(b)** 组件消融：无论从 GDPO 还是 MARBLE 出发，逐步加入各组件都带来渐进增益，完整方法整体最强。

### 4.4 消融实验

在 LTX-2 上做的消融分为两条相互独立的链路（灰色行为权重链，从 MARBLE 出发且不继承路由组件）：

| 链路 | 配置 | VQ↑ | AQ↑ | CLAP↑ | AV-IB↑ | JavisScore↑ | DeSync↓ |
|---|---|---|---|---|---|---|---|
| Routing | + Token Weighting | 3.008 | 5.663 | 0.388 | 0.210 | 0.179 | 0.482 |
| Routing | + Layer Scale | 3.192 | 5.784 | 0.409 | 0.221 | 0.189 | 0.366 |
| Routing | + Token + Layer | 3.315 | 5.839 | 0.411 | 0.226 | 0.190 | 0.343 |
| Weighting | + Branch-Aware | 2.612 | 5.290 | 0.384 | 0.198 | 0.177 | 0.492 |
| Weighting | + Residual | 2.891 | 5.526 | 0.393 | 0.199 | 0.180 | 0.455 |
| Weighting | + Warm-Up | 2.999 | 5.791 | 0.404 | 0.209 | 0.183 | 0.369 |
| **Routing + Weighting (Ours)** | — | **3.336** | **5.868** | **0.425** | **0.235** | **0.206** | **0.341** |

三个结论：

1. **路由组件互补** ：token 加权通过强调跨模态响应强的 token 改善局部信用分配，层缩放通过保留关键交互层的梯度进一步改善一致性与同步，两者叠加是路由-only 的最强配置；
2. **权重组件各尽其职** ：branch-aware 把奖励交互指派到负责的模态分支；residual mixing 保留偏好先验而不是用梯度系数取而代之；warm-up 延迟动态调权直到梯度关系估计可靠，稳定整个适配过程；
3. **两条轴互补** ：中间配置不一定让每个指标单调变好（各组件偏好的目标不同），但完整模型在质量、语义一致性、同步三方面取得最强的整体平衡，说明“自适应定位”与“偏好保持的协调”解决的是两种不同但互补的失效模式。

## 五、机制验证：跨模态影响力代理靠谱吗？

方法是建立在“pre-gate cross-attention 响应范数 ≈ 真实影响力”这个假设上的，作者用直接干预实验做了三重验证。具体做法：在 4 个训练 checkpoint 上分别禁用 48 个 A2V/V2A 块中的每一个（固定 prompt、噪声隐变量和时间步），最终预测变化越大说明该路径越有影响力。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Adaptive-Reward-Routing-Dynamic-Multi-Reward-Optimization-for-Joint-Audio-Video-Diffusion-via-Forward-Process-RL/images/proxy.png)

> 图解： **(a)** 层级保真度：每个点是一层，x 轴为该层的平均 pre-gate 响应范数，y 轴为禁用该层后最终速度的相对变化（两轴均按 48 层 min-max 归一化），点贴近对角线说明代理与直接干预一致。 **(b)** 动态追踪：y 轴为代理与当前模型干预结果的相关系数，实时重算的代理始终准确，而初始化时冻结的代理随训练迅速失准。 **(c)** token 级验证：分别屏蔽代理得分 top 10%、随机 10%、bottom 10% 的目标 token 所接收的跨模态信息，屏蔽 top 组造成的预测变化最大。

三个数字最有说服力：

- **层级排序一致** ：代理恢复的层排序与干预排序的 Spearman 相关系数达 **0.98** （A2V）和 **0.97** （V2A）；
- **必须动态重算** ：随训练实时重算的代理全程保持在 **0.96** 以上，而初始化时冻结的代理掉到 **0.56 / 0.38** ——影响力层级确实在漂移，固定路由图会过时；
- **高分 token 确实更重要** ：屏蔽 top 10% 高分 token 造成的预测变化是随机 10% 组的 **1.62 倍** （A2V）和 **1.74 倍** （V2A）。

> 博主点评：这组验证是全文最扎实的部分之一。很多“用某信号做代理”的工作止步于直觉合理，本文用等规模消融直接证明了代理的因果有效性，还顺带量化了“静态路由过时”这一核心动机，形成了闭环。

## 六、总结与展望

这篇文章把联合音视频生成的多奖励后训练重新表述为一个 **动态信用分配** 问题，核心要点：

- **问题重述** ：多奖励 RL 的有效性取决于两个随训练变化的量——更新该打在哪里、奖励该如何协调，静态方案（OmniNFT 的固定路由、GDPO 的固定权重、MARBLE 的纯几何调权）各有硬伤；
- **路由组件** ：用双向 cross-attention 的 pre-gate 响应作为零额外成本的跨模态影响力代理，token 级做损失加权、层级做“前向不变、梯度缩放”的软 detach；
- **权重组件** ：偏好先验打底提供非零下限，分支内 MARBLE 冲突系数在 warm-up 后做残差修正，既适配冲突又不让强势奖励吞掉弱目标；
- **实验结论** ：两个骨干、10 项指标中各 9 项最优；消融确认 token/层路由、分支感知、残差修正、warm-up 各自有效且互补；
- **机制验证** ：代理与直接干预的层排序相关达 0.97–0.98，动态重算全程保持 0.96 以上，冻结代理跌至 0.38–0.56，高分 token 被屏蔽后影响是随机组的 1.6–1.7 倍。

局限与方向：目前还缺少一个统一刻画“质量 + 一致性 + 同步”的综合音视频奖励模型；方法依赖 DiffusionNFT 的前向过程形式，但路由思想可推广到其他扩散目标；架构上要求模态 token 可识别、方向性交互响应可分离，对模态表示不可分的模型（作者已在单流 JavisDiT++ 上初步验证奖励协调部分）仍是开放问题。更宏观地看，这项工作指向一个方向：多模态学习系统中的反馈不应按固定配方路由，而应随模型能力的成长持续重组。

> 本文参考自 [Adaptive Reward Routing: Dynamic Multi-Reward Optimization for Joint Audio-Video Diffusion via Forward-Process RL](https://arxiv.org/abs/2609.37200)