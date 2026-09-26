# Rolling-WAM：让机器人"边想边做"的滚动式世界动作模型

World Action Models（WAM，世界动作模型）让机器人在生成动作的同时"想象"未来的画面，决策更有远见；但每次重新规划都要把整段未来视频和动作从纯噪声完整去噪一遍，动辄近 1 秒的延迟让闭环控制跟不上环境变化。USC、Brown、复旦与 Toyota Research Institute 联合提出的 **Rolling-WAM** 换了个思路：不再每步从头算，而是维护一个滑动窗口，近处的动作块彻底去噪立即执行，远处的块保持半成品随窗口滚动逐步细化。效果上，它在 LIBERO 上取得 98.1% 成功率、RoboTwin 2.0 上以 93.3% 刷新 SOTA，同时把单次重规划延迟从 Joint-WAM 的 978 ms 压到 215 ms，实现 **4.5 倍** 稳态加速，且比纯动作 VLA 模型 $\pi_{0.5}$（296 ms）还要快。

## 问题：WAM 的"想象力"太贵了

传统的视觉-语言-动作（VLA）模型只看当前观测出动作，属于"看一步走一步"。WAM 则更进一步：在生成动作序列 $\mathbf{a}_{t:t+H-1}$ 的同时，联合预测未来视频 $\mathbf{v}_{t+1:t+H}$，形式化为学习联合分布 $p_\theta(\mathbf{a}_{t:t+H-1}, \mathbf{v}_{t+1:t+H} \mid c_t)$，其中条件 $c_t = (o_t, s_t, \ell)$ 包含当前图像、机器人状态和语言指令。视觉预测为动作提供了"场景将如何演化"的先验，行为更连贯、更有远见。

问题出在部署端。真实环境是动态的，机器人必须频繁地基于最新观测重新规划（replan），纠正执行误差。而标准的联合去噪采样器在每个重规划周期都要把 **整段** 预测时域从纯高斯噪声去噪到干净数据，扩散模型的多步迭代让单次推理非常昂贵。

这就产生了一个明显的浪费：receding-horizon 控制实际只执行每个周期的前一小段动作，但模型却对整个未来付出了同等算力——那些"还没执行就被丢弃"的远期预测，白算了。有的工作（如 Fast-WAM）干脆在部署时砍掉视频生成来提速，代价是失去了视觉想象这块锚点。

笔者认为，这个矛盾的根源在于 **去噪调度与执行截止时间不匹配**：近期的动作马上要用，理应优先算干净；远期的计划只是"草稿"，完全可以后补。人类规划正是如此——下一秒怎么走很具体，三秒后怎么走边做边想。Rolling-WAM 就是把这个直觉形式化了。

## 方法：把去噪过程"摊"到多个控制周期

Rolling-WAM 的核心是一张 **噪声水平错开的滑动窗口**。借鉴 Rolling Diffusion 在序列生成中的思想，窗口内的视频-动作块按时间先后持有递增的噪声水平：最近的块最干净，最远的块最接近纯噪声。

![Rolling-WAM 整体示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/World-Action-Models-with-Rolling-Imagination/figures/rollingwam-teaser.png)

> 图解：左侧为机器人执行画面；中间上方显示 $t+1$ 时刻的预测窗口，近处的 $t+1$ 帧清晰、远处 $t+3$ 帧仍是噪声，下方曲线为对应的动作轨迹；执行完第一个动作块后，机器人获得新观测 $o_{t+1}$，窗口整体前移，旧的干净块被移除、尾部追加一个纯噪声新块（下方一行）。右侧评估面板直观给出三个核心结论：真机成功率 85.0% 领先，仿真成功率 93.3% 领先，单次重规划延迟 215 ms 仅为 Joint-WAM 的约 1/4.5。

### 滚动去噪：噪声调度的数学设计

把预测窗口划分为 $W$ 个时间对齐的块 $\mathbf{X}_{1:W}$，每块 $\mathbf{X}_j = (\mathbf{v}^{(j)}, \mathbf{a}^{(j)})$ 包含一段视频和 $K$ 个动作，总预测时域 $H = WK$。用 $\tau \in [0,1]$ 参数化一个重规划周期内的去噪相位（从 1 降到 0），$\sigma(\cdot)$ 为单调递增的基础噪声调度，$\sigma(1)=1$ 对应纯高斯噪声。

**滚动模式（Rolling mode）** 下，第 $j$ 个块的噪声水平为：

$$
\sigma_j^{\mathrm{rolling}}(\tau) = \sigma\!\left(\frac{j-1+\tau}{W}\right), \qquad j = 1, \ldots, W
$$

这个设计有一个非常精巧的对齐性质：

$$
\sigma_{j+1}^{\mathrm{rolling}}(0) = \sigma_j^{\mathrm{rolling}}(1), \qquad j = 1, \ldots, W-1
$$

意思是：一个周期结束时，第 $j+1$ 块的噪声水平恰好等于第 $j$ 块在本周期开始时的噪声水平。窗口前移后，每个留存的块 **天然就位** 于它新位置应有的起始噪声，无需任何重采样，只需在尾部补一个 $\sigma = 1$ 的纯噪声块，就恢复了下一周期的初始构型。笔者认为这是全文最漂亮的一笔——窗口滚动与噪声调度被设计成了无缝咬合的齿轮。

**初始化模式（Initialization mode）** 处理 episode 开头没有任何历史预测的情况：

$$
\sigma_j^{\mathrm{init}}(\tau) = \sigma\!\left(\min\left\{1,\; \tau + \frac{j-1}{W}\right\}\right)
$$

所有块从纯噪声出发，近处的块先去噪，远处的块暂时"冻"在噪声里；结束时构型与滚动模式一致，平滑衔接。

### 采样与执行：每周期只需 $N/W$ 步

设每个块完整去噪需要 $N$ 步（取 $W$ 的倍数）。初始化阶段用步长 $\Delta\tau = -1/N$，共 $N$ 步；进入稳态后，每个重规划周期只需 **$N/W$ 步**、步长 $\Delta\tau = -W/N$，就能把第一个块去噪干净。

每步用 Euler 更新联合刷新整个窗口：

$$
\widetilde{\mathbf{X}}_j \leftarrow \widetilde{\mathbf{X}}_j + \left[\boldsymbol{\sigma}_j(\tau+\Delta\tau) - \boldsymbol{\sigma}_j(\tau)\right] f_{\theta,j}\!\left(\widetilde{\mathbf{X}}_{1:W}, \boldsymbol{\sigma}; c_t\right)
$$

第一个块干净后，机器人执行其 $K$ 个动作，窗口前移、追加新噪声块，并以最新相机观测和状态进入下一周期。注意：每个块在被执行前仍然走满了全部 $N$ 步去噪—— **总计算量没有省，只是被分摊到了它"排队"的多个周期里**。这也解释了为什么更大的窗口能降低稳态延迟。默认配置 $N=10$、$W=5$、$K=16$，即每次重规划只做 2 步去噪，预测时域却长达 80 个动作。

### 架构与训练：MoT 双专家 + 匹配噪声剖面

![Rolling-WAM 框架](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/World-Action-Models-with-Rolling-Imagination/figures/rollingwam-framework.png)

> 图解：(a) 联合视频-动作模型：当前观测经 VAE 编码后与带噪未来视频 latent 进入 Video DiT，带噪动作块进入 Action DiT，两者通过 masked joint attention 耦合，文本指令和机器人状态经 cross-attention 注入。(b) 滚动推理：联合去噪 → 执行 $A_1$ → 窗口前移并追加纯噪声块 $V_{W+1}, A_{W+1}$，颜色越深噪声越大。(c) 注意力掩码：动作块可以看见整个视频窗口（粉紫区域），但动作-动作注意力仅限同一块内部（橙色对角块），视频 token 不回头看动作。

架构上采用 Mixture-of-Transformers（MoT）：视频专家是预训练的 Wan2.2-TI2V-5B 视频 DiT，动作专家是一个约 1B 参数、30 层的轻量 Transformer，权重由视频专家插值初始化。视频与动作 token 按所在块的噪声水平做调制，使不同"完成度"的预测能被联合处理。注意力掩码的设计意图很明确： **让每个动作块都能读到整段共享的、不断演化的视觉未来，但动作信息只允许通过视觉通道跨块流动**。

训练采用 flow matching。对每个示范窗口采样 $\tau \sim \mathcal{U}(0,1)$，以概率 $\beta$（实际取 0.8）选滚动模式、$1-\beta$ 选初始化模式，构造带噪样本：

$$
\widetilde{\mathbf{X}}_j = (1-\boldsymbol{\sigma}_j)\mathbf{X}_j + \boldsymbol{\sigma}_j\boldsymbol{\epsilon}_j, \qquad \boldsymbol{\epsilon}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

网络预测目标速度 $\boldsymbol{\epsilon}_j - \mathbf{X}_j$，联合目标为：

$$
\mathcal{L} = \mathbb{E}\!\left[\frac{1}{W}\sum_{j=1}^{W} b_j\, w(\boldsymbol{\sigma}_j)\left(\lambda_v \ell_j^v + \lambda_a \ell_j^a\right)\right]
$$

其中 $\ell_j^v, \ell_j^a$ 分别为视频与动作的逐块 L2 损失，$w$ 为噪声加权，$\lambda_v = \lambda_a = 1$，掩码 $b_j$ 排除初始化阶段仍处于纯噪声的块。关键点是 **训练时的噪声剖面与推理时完全一致** ——模型学到的正是"看着半成品未来做决策"的能力。

## 实验：性能不降，延迟砍到 1/4.5

实验回答一个核心问题：滚动去噪能否在加速的同时保住操作性能？评测覆盖 LIBERO（4 个套件）、RoboTwin 2.0（50 个双臂任务，Clean / Randomized 两种设置）和 Unitree G1 真机的三个任务，并做了受控延迟测量与消融。

### 仿真基准：RoboTwin 上反超所有基线

LIBERO 上的对比如下（P.T. 表示是否经过具身预训练；Rolling-WAM 没有使用任何额外具身预训练）：

| Method | P.T. | Spatial | Object | Goal | Long | Average |
| --- | :-: | ---: | ---: | ---: | ---: | ---: |
| $\pi_0$ | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| $\pi_{0.5}$ | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| Motus | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | **98.5** |
| Fast-WAM | ✗ | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Joint-WAM | ✗ | 99.6 | 99.4 | 98.2 | 96.8 | **98.5** |
| **Rolling-WAM** | ✗ | 98.2 | 98.0 | 98.2 | 97.8 | 98.1 |

Rolling-WAM 平均 98.1%，与并列第一的 LingBot-VA / Joint-WAM（98.5%）仅差 0.4 个百分点，且四个套件成绩均衡（97.8%–98.2%）。值得注意，排在前面的 LingBot-VA 吃了大规模具身预训练的红利，Rolling-WAM 是"裸训"出战。

RoboTwin 2.0 的结果更有说服力：

| Method | P.T. | Clean | Randomized | Average |
| --- | :-: | ---: | ---: | ---: |
| $\pi_0$ | ✓ | 65.9 | 58.4 | 62.2 |
| $\pi_{0.5}$ | ✓ | 82.7 | 76.8 | 79.8 |
| Motus | ✓ | 88.7 | 87.0 | 87.8 |
| LingBot-VA | ✓ | 92.9 | 91.5 | 92.2 |
| Fast-WAM | ✗ | 91.9 | 91.8 | 91.8 |
| Joint-WAM | ✗ | 90.8 | 90.3 | 90.6 |
| **Rolling-WAM** | ✗ | **93.5** | **93.0** | **93.3** |

在 50 个双臂任务、含场景随机化的设置下，Rolling-WAM 以 93.3% 的平均成功率 **同时拿下两种设置的第一**，反超 LingBot-VA 1.1 个点、Fast-WAM 1.5 个点、Joint-WAM 2.7 个点。省算力还能涨点，说明滚动机制带来的更长远视觉上下文本身就是正则化式的收益。

![想象与真实 rollout 对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/World-Action-Models-with-Rolling-Imagination/figures/robotwin-imagined-observed.png)

> 图解：RoboTwin 任务 Put Object Cabinet 中，上行为 Rolling-WAM 在执行过程中"想象"的未来视频，下行为真实观测，各列是连续时刻。可以看到想象中的机械臂运动与任务推进和真实观测高度一致——滚动窗口中留存并持续细化的视觉预测没有漂移。

### 推理效率：215 ms，比纯动作 VLA 还快

延迟在单张 A100、RoboTwin 2.0 设置（$384 \times 320$ 分辨率）下测量，含视觉编码与去噪，不含初始化和预热。

![重规划延迟对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/World-Action-Models-with-Rolling-Imagination/figures/robotwin_replan_latency.png)

> 图解：柱状图为各方法单次稳态重规划的平均延迟（ms，越低越好）。Joint-WAM 每次全量去噪需 978 ms，Fast-WAM 去掉视频生成后为 548 ms，纯动作 VLA 模型 $\pi_{0.5}$ 与 GR00T N1.7 分别为 296 ms 和 285 ms，而 Rolling-WAM 只需 215 ms——相对 Joint-WAM 加速约 4.5 倍、相对 Fast-WAM 约 2.5 倍，甚至比不生成视频的 VLA 基线还快，且它维护的是 80 步的预测窗口（基线只有 16 步）。这些数字未使用 torch.compile、TensorRT 或自定义 CUDA kernel。

窗口大小的影响也值得一看：

![不同窗口大小下的延迟](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/World-Action-Models-with-Rolling-Imagination/figures/replan_annotated_latency_trend.png)

> 图解：横轴为窗口大小 $W$，纵轴为重规划延迟（ms）。Rolling-WAM（蓝线）随 $W$ 增大近乎线性下降——去噪步数 $N/W$ 越来越少——$W=5$ 时约 322 ms（该扫描中用 $N=15$、每周期 3 步，对应 4.75 倍加速），$W=6$–8 进入平台期；而 Joint-WAM（灰）与 Fast-WAM（黄）的延迟几乎不随 $W$ 变化，始终在 800–1600 ms 区间。

### 真机验证：G1 人形机器人上的三个任务

真机实验在 Unitree G1 上进行，三个任务为 Doll Placement（把玩偶狗放进盒子）、Plate Stacking（叠三个盘子）、Bead Pouring（把珠子从瓶中倒入玻璃杯）。策略输入 $320 \times 224$ 第一视角图像和 43 维状态，输出 78 维动作（64 维 SONIC 运动 latent + 双手各 7 维指令），每任务 50 条示范、10 Hz 执行，每任务测 20 次。

![G1 真机任务](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/World-Action-Models-with-Rolling-Imagination/figures/g1-real-world.png)

> 图解：自上而下为 Doll Placement、Plate Stacking、Bead Pouring 三个任务的代表性执行序列，每行从左到右展示任务推进过程，小图为第一视角观测。

| Method | Doll Placement | Plate Stacking | Bead Pouring | Avg. (%) |
| --- | ---: | ---: | ---: | ---: |
| $\pi_{0.5}$ | 55.0 | 70.0 | 60.0 | 61.7 |
| GR00T N1.7 | 75.0 | 65.0 | 60.0 | 66.7 |
| Fast-WAM | 85.0 | 80.0 | 60.0 | 75.0 |
| Joint-WAM | 70.0 | 100.0 | 65.0 | 78.3 |
| **Rolling-WAM** | **85.0** | **100.0** | **70.0** | **85.0** |

Rolling-WAM 以 85.0% 的平均成功率领跑，三个任务上均不弱于任何基线。更有意思的是定性观察：Rolling-WAM 在动作块边界处的执行更连贯，而 Joint-WAM 因重规划停顿明显，有时甚至打断任务节奏——低延迟在物理世界的好处，不只是跑分上的数字。

### 消融：窗口不是越大越好

![窗口大小与成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/World-Action-Models-with-Rolling-Imagination/figures/window_size_ablation.png)

> 图解：横轴为窗口大小 $W$（预测时域 $H=16W$），纵轴为 6 个代表性 RoboTwin Clean 任务的平均成功率。成功率从 $W=1$ 的 63.0% 一路上升到 $W=5$ 的峰值 78.2%，随后回落，$W=8$ 时降至 69.5%。原因可能是：过远的视觉预测受当前观测约束太弱，对即将执行的动作帮助有限，反而引入噪声。这支撑了 $W=5$ 的默认选择。

噪声调度与注意力的消融（6 个 RoboTwin 任务 + LIBERO）：

| Variant | Selected RoboTwin | LIBERO |
| --- | ---: | ---: |
| **Full Rolling (w/o A2A)** | 78.2 | **98.1** |
| w/ A2A | 76.3 | 97.9 |
| Constant ($p=0.2$) | 74.7 | 97.9 |
| Constant ($p=0.5$) | 73.0 | 97.1 |
| Random ($p=0.2$) | 75.3 | 97.0 |
| Random ($p=0.5$) | **78.5** | 97.3 |

两个结论：其一，训练时混入 Constant 或 Random 噪声剖面都无法在两个基准上同时改进，与推理严格对齐的完整 rolling 调度是最稳妥的选择；其二，放开跨块的动作-动作注意力（A2A）反而双降——信息通过共享视觉窗口流动已经足够，直接的动作捷径没有收益。

## 总结

- **核心思想**：把联合视频-动作去噪分摊到多个重规划周期，滑动窗口内近处块去净即执行、远处块半成品随窗口滚动续算；
- **调度设计**：滚动噪声调度满足 $\sigma_{j+1}(0) = \sigma_j(1)$ 的无缝对齐，窗口前移后留存块天然就位；
- **性能**：LIBERO 98.1%（距榜首 0.4 点）、RoboTwin 2.0 以 93.3% 刷新 SOTA、G1 真机平均成功率 85.0% 领先全部基线；
- **效率**：稳态重规划 215 ms，对 Joint-WAM 加速约 4.5 倍，比纯动作 VLA 还快，同时保留完整的未来视频想象；
- **局限与方向**：长窗口下留存的预测可能滞后于快速变化的场景；自适应窗口管理与异步执行是值得探索的方向。

Rolling-WAM 给出的启示是普适的：生成式策略的算力分配应当与执行截止时间对齐——这与一致性蒸馏、特征缓存等优化正交，叠加使用还有进一步压缩延迟的空间。

> 本文参考自 [Rolling-WAM: World Action Models with Rolling Imagination](http://arxiv.org/abs/2609.30247v1)