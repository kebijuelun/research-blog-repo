# Road MOPD：跨教师守护化解蒸馏冲突

多教师在线策略蒸馏（MOPD）想把数学、代码、医学等多个领域专家教师的能力"合并"进同一个学生模型，但主流的"领域路由"做法只管每个样本听哪个老师的话，却不管不同老师给出的更新方向是否在共享参数上打架。这篇文章发现：当教师风格差异大时（如医学搭配数学/代码），联合训练会产生持续的梯度冲突，导致弱势领域能力不进反退。为此作者提出 **Road MOPD**（Cross-Teacher Guarding for MOPD），让"没被路由到的教师"作为跨领域评委，在同一条学生轨迹上从 token 加权和梯度投影两个层面进行"守护"。在高冲突场景下，医学领域准确率比 Vanilla MOPD 提升 **5.6–8.8 个百分点**，并在低冲突和三领域设置下也取得最佳综合表现。

## 一、问题：领域路由只管"听谁的"，不管"打不打架"

先快速回顾两个概念。

**OPD（On-Policy Distillation，在线策略蒸馏）**：学生自己生成回答，教师在这些学生轨迹上提供逐 token 的密集监督。相比离线蒸馏，它的训练分布与学生当前的生成分布一致，信用分配更细。

**MOPD（Multi-teacher OPD）**：有 $K$ 个领域专家教师时，最自然的做法是 **领域路由** ——数学样本只问数学教师，医学样本只问医学教师，然后所有领域的梯度共同更新同一个学生。

问题在于：路由只区分了监督的 **来源**，更新却作用在 **共享参数** 上。单个领域里有效的监督信号，联合优化时可能互相干扰。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/intro_gradient_conflict_and_transfer.png)

> 图解：(a) 训练过程中产生梯度冲突的网络层比例。Math×Code 组合冲突层很少，而 Medical 与 Math/Code 搭配时冲突层比例很高，且随训练推进进一步上升。(b–d) Vanilla MOPD 相对单领域 OPD 的逐 benchmark 分数变化：Math×Code 基本能保住甚至超过单领域水平，而两个含 Medical 的组合都一致地拉低了医学表现。这说明"数学+代码能联合获益"的经验并不能推广到所有领域组合。

笔者认为，这组图的价值在于把"联合训练有时失效"从模糊的经验变成了可测量的现象：**更新兼容性** 才是 MOPD 的关键变量，而单纯靠路由分配监督显然协调不了它。

于是本文的核心问题浮出水面：**没被路由到的教师，能否帮忙预判并缓解更新冲突，同时不破坏路由教师的本领域监督？**

## 二、方法总览：让"旁观者"教师当评委

作者的回答是把非路由教师（auxiliary teacher）引入为 **跨领域评委**：它们在同一条学生生成的轨迹上打分，其反馈在两个层面发挥作用：

- **Token 级**：跨教师一致性引导的信用分配——两个教师都支持的修正方向就加强，意见相左的 token 就削弱；
- **梯度级**：预防性梯度投影——用同轨迹上的辅助梯度做约束，削掉目标领域更新中与之冲突的分量，并用范数保持机制防止"削过头"。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/mopd_method_overview_revised.png)

> 图解：左图是 Vanilla MOPD——各领域路由监督直接对共享参数产生可能互相冲突的更新。右图是 Road MOPD——同轨迹上的辅助教师反馈用于 (A) 基于一致性的 token 级信用分配，和 (B) 带目标梯度范数保持约束的预防性梯度投影。

### 2.1 形式化：OPD 与 MOPD

设学生策略为 $\pi_\theta$，教师为 $\pi_T$，prompt 分布为 $\mathcal{D}$。OPD 最小化序列级 reverse-KL 目标：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta;\pi_T,\mathcal{D})
= \mathbb{E}_{x\sim\mathcal{D}}
  D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_T(\cdot\mid x)\right)
$$

引入冻结参考模型 $\pi_{\mathrm{ref}}$（即学生初始化的副本），可以把 OPD 等价写成"相对奖励最大化 + KL 惩罚"的形式（附录给出推导，见第五节）：

$$
\mathcal{J}_{\lambda}(\theta)
=\mathbb{E}_\theta\!\left[
  \lambda\log\frac{\pi_T(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
  -D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)\right]
$$

其中 $\lambda=1$ 就是标准 OPD；$\lambda>1$ 放大教师的相对奖励（即 ExOPD）。这个 $\lambda$ 是本文第一个切入点：**固定 $\lambda$ 对所有 token 一视同仁，但一条轨迹里既有可迁移的通用推理模式，也有领域专属的修正，理应区别对待。**

MOPD 则是把 $K$ 个领域的 OPD 目标等权平均：

$$
\mathcal{L}_{\mathrm{MOPD}}(\theta)
=\frac{1}{K}\sum_{d=1}^{K}\mathcal{L}_{\mathrm{OPD}}(\theta;\pi_{T_d},\mathcal{D}_d)
$$

以下记路由教师为 $T_{\mathrm{r}}=T_d$，辅助教师为 $T_{\mathrm{a}}$，两者与参考模型在 **同一批学生轨迹** 上按 teacher forcing 打分（无需额外 rollout），对应 log 概率分别记为 $\ell_t^{\mathrm{r}}$、$\ell_t^{\mathrm{a}}$、$\ell_t^{\mathrm{ref}}$。

### 2.2 Token 级：跨教师一致性信用分配

怎么量化两个教师"意见是否一致"？作者把每个教师的 log 概率先减去共享参考模型的值（得到相对证据），过 $\tanh$ 压缩到 $[-1,1]$，再相乘：

$$
s_t
=\tanh\!\left(\frac{\ell_t^{\mathrm{r}}-\ell_t^{\mathrm{ref}}}{\tau}\right)
\times
\tanh\!\left(\frac{\ell_t^{\mathrm{a}}-\ell_t^{\mathrm{ref}}}{\tau}\right),
\qquad s_t\in[-1,1]
$$

直觉很直白：$s_t>0$ 表示两个教师相对参考模型倾向 **同一方向** 的修正；$s_t<0$ 表示两人意见相反，越接近 $-1$ 分歧越大；任一教师与参考模型差不多时，$s_t$ 接近 0（没有额外证据）。

然后用 $s_t$ 把全局固定的奖励尺度换成逐 token 系数：

$$
\lambda_t
=\mathrm{clip}\!\left(1+\rho_{\mathrm{g}}s_t,\ \lambda_{\min},\ \lambda_{\max}\right)
$$

- $s_t>0 \Rightarrow \lambda_t>1$：两个教师都支持，放大路由教师的相对奖励；
- $s_t<0 \Rightarrow \lambda_t<1$：两人打架，衰减该 token 的监督；
- $s_t=0 \Rightarrow \lambda_t=1$：退化为标准 OPD。

实现上，调制后的信号 $\lambda_t(\ell_t^{\mathrm{r}}-\ell_t^{\mathrm{ref}})-k_{1,t}(\pi_\theta\|\pi_{\mathrm{ref}})$ 经 clip 和 stop-gradient 后作为 PPO 风格 surrogate 的 advantage（$k_1$ 是采样 token 的 log-ratio 估计子）。有多个辅助教师时，先把各辅助教师与路由教师的 token 一致性取平均再做同样调制。

这里有一个容易误读的细节，作者特别强调：**辅助教师不是来教目标领域知识的"第二导师"**，它的相对偏好只被当作"该教师所代表能力"的兼容性信号。这个定位决定了它不是多装了一个损失，而是给路由监督装了一个"刹车/油门"。

### 2.3 梯度级：带范数保持的预防性投影

解决了 token 加权之后，下一个问题是：token 级一致了，参数更新就兼容吗？不一定——token 信号要经过共享参数聚合，所以还需要梯度层面的保险。

先从一致性调制后的损失算出目标领域梯度 $g_d=\nabla_\theta\mathcal{L}_d^{\mathrm{agr}}$。辅助梯度则把路由教师换成辅助教师、关掉一致性调制，在 **同一批 domain-$d$ 轨迹** 上算普通 OPD 梯度 $g_d'$。注意 $g_d'$ 反映的是辅助教师对"当前行为"的偏好，而不是它在自己领域数据上的训练梯度——这正是"预防性"与 PCGrad"事后修补"的本质区别。

对每个可训练参数张量（记切片为 $g$ 和 $g'$），若 $\langle g,g'\rangle<0$ 则存在局部冲突。候选修正梯度被限制在 $g$ 与其完全投影之间的线段上：

$$
g_{\mathrm{full}}
=g-\frac{\langle g,g'\rangle}{\|g'\|_2^2}g',
\qquad
\mathcal{S}(g,g')=\mathrm{conv}\{g,g_{\mathrm{full}}\}
$$

完全投影能彻底消除冲突，但当两个梯度近乎反向时会把 $g$ 几乎抹掉——本领域的学习信号也没了。所以作者在 **范数保持下限** $m\in(0,1]$ 下求残余冲突最小的解：

$$
\widetilde g
=\underset{v\in\mathcal{S}(g,g')}{\arg\min}\;-\langle v,g'\rangle
\quad\text{s.t.}\quad
\|v\|_2^2\geq m^2\|g\|_2^2
$$

解是一个"部分投影"：

$$
\widetilde g
=g-\alpha^\star\frac{\langle g,g'\rangle}{\|g'\|_2^2}g',
\qquad \alpha^\star\in[0,1]
$$

完全投影满足范数下限就全投（$\alpha^\star=1$）；否则在下限允许范围内尽量消冲突，此时 $\|\widetilde g\|_2=m\|g\|_2$；无冲突则不动（$\alpha^\star=0$）。闭式解与推导见第五节的附录部分。多辅助教师时，对每个张量依次选余弦最负的 peer 做投影，并相对 **原始** 目标梯度共享同一个范数下限。

这个设计的聪明之处在于：投影强度不是固定的，而是由"消冲突"和"保信号"的权衡自适应决定——$m$ 相当于给本领域梯度模长上了一道保险丝。

### 2.4 整体目标与训练流程

最终的蒸馏目标只包含目标领域损失（辅助损失仅用于梯度修正，不进目标），各领域先做序列内 token 平均、再跨响应和领域平均。更新时只合并投影后的目标梯度：

$$
g_{\mathrm{update}}=\frac{1}{K}\sum_{d=1}^{K}\widetilde g_d
$$

再经全局梯度范数裁剪（阈值 1.0）后做一步 AdamW。完整的一步优化流程为：

1. Rollout：用旧策略 $\pi_{\mathrm{old}}$ 采样，每个 prompt 生成 1 条响应；
2. 打分：$K$ 个教师与冻结参考模型在同批轨迹上打 log 概率分；
3. 对每个领域 $d$：计算一致性 $\bar s_{d,t}$ → 调制系数 $\lambda_{d,t}$ → 调制 advantage → 目标梯度 $g_d$；
4. 对每个辅助教师 $j\neq d$：算普通 OPD 辅助梯度 $g'_{d\to j}$；
5. 张量级投影得到 $\widetilde g_d$，跨领域平均、全局裁剪、AdamW 更新。

实现细节值得一提：梯度快照用 BF16 存储，内积与平方范数用 FP64 计算并跨数据并行组归约，保证同一张量的分片共用同一余弦值与投影系数；数值稳定常数为 $10^{-12}$。

## 三、实验设置

**模型配置**：学生为 Qwen3-4B 和 Qwen3-1.7B（关闭 thinking 模式，全参数训练）；数学/代码教师分别为 Qwen3-4B-Non-Thinking-RL-Math-Step500 / -Code-Step300；医学教师为 II-Medical-8B（基于 Qwen3-8B 的 RL checkpoint）。所有教师与参考模型全程冻结，仅用于打分；各模型词表兼容，保证同一 token 序列可比较。

**训练数据**：每个配对设置每领域 10,000 条 prompt，共 20,000 条，随机打乱。

| 领域 | 来源 | 采样条数 |
|---|---|---|
| Math | DeepMath（难度 ≥ 6，候选池 57K） | 10,000 |
| Code | Eurus-RL-Code | 10,000 |
| Medical | MedMCQA 训练集 | 4,000 |
| Medical | MedReason | 3,000 |
| Medical | Medical-R1-Distill-Data | 2,000 |
| Medical | m23k | 1,000 |

**评测**：3 个领域共 11 个 benchmark——Code（HumanEval+、MBPP+、LiveCodeBench v6）报 Avg@4；Math（AIME 2024/2025、HMMT 2025 Feb./Nov.）报 Avg@32；Medical（MedQA、MedXpertQA、MedMCQA、SuperGPQA-Med）报 Avg@1。

**基线**：Vanilla MOPD（纯领域路由）、ExOPD（固定外推系数 1.25）、CaMOPD（能力恢复/领域保持分流）、Open-MOPD（token 份额均衡 + 动态领域权重 + reward refresh），分析部分还对比了 PCGrad。

**关键超参**：$\tau=1$，$\rho_{\mathrm{g}}=0.5$，$[\lambda_{\min},\lambda_{\max}]=[0.5,1.5]$，范数保持 $m=0.6$；AdamW，学习率 $10^{-5}$ 恒定；每 rollout batch 1,008 条 prompt 对应一次优化更新，配对设置 58 个 batch、三领域 116 个；PPO clip 0.2，advantage 裁剪到 $[-10,10]$（完整方法与所有消融一致）；rollout 温度/top-$p$ 均为 1.0，最大响应长度 16,384 token。硬件为单节点 8 张 NVIDIA B200。

## 四、实验结果

### 4.1 主结果：高冲突配对全面领先

**Qwen3-4B，Math × Medical**（百分制，加粗为最优，下划线为次优）：

| 方法 | Math 域均 | Med 域均 | 两域平均 |
|---|---|---|---|
| Student | 15.37 | 41.01 | 28.19 |
| Single OPD | 44.27 | 47.90 | 46.09 |
| Vanilla MOPD | 45.97 | 39.45 | 42.71 |
| ExOPD | <u>47.19</u> | 43.87 | 45.53 |
| Open-MOPD | 45.34 | 44.52 | 44.93 |
| CaMOPD | 46.64 | <u>47.75</u> | <u>47.20</u> |
| **Ours** | **47.60** | **48.23** | **47.92** |

**Qwen3-4B，Code × Medical**：

| 方法 | Code 域均 | Med 域均 | 两域平均 |
|---|---|---|---|
| Student | 52.97 | 41.01 | 46.99 |
| Single OPD | 60.91 | 47.90 | 54.41 |
| Vanilla MOPD | 60.53 | 40.75 | 50.64 |
| ExOPD | <u>61.65</u> | 44.93 | 53.29 |
| Open-MOPD | 60.83 | <u>47.34</u> | <u>54.09</u> |
| CaMOPD | 60.64 | 47.28 | 53.96 |
| **Ours** | **63.02** | **47.75** | **55.38** |

两个设置下 Vanilla MOPD 的医学均值（39.45 / 40.75）甚至 **低于未蒸馏学生的 41.01** ——联合训练帮了倒忙。Road MOPD 把医学拉回到 48.23 / 47.75，恢复到 Single OPD 参考线（47.90）附近甚至之上，同时 Math/Code 也拿到多领域方法中的最高分：医学的"补课"不是靠牺牲另一门课换来的。

**Qwen3-1.7B 学生**（域均值摘要）：

| 方法 | Math×Med: Math | Math×Med: Med | 平均 | Code×Med: Code | Code×Med: Med | 平均 |
|---|---|---|---|---|---|---|
| Single OPD | 27.60 | 35.67 | 31.64 | 51.68 | 35.67 | 43.67 |
| Vanilla MOPD | <u>28.36</u> | 31.18 | 29.77 | 51.07 | 30.93 | 41.00 |
| ExOPD | **28.44** | <u>33.01</u> | <u>30.72</u> | **54.16** | 29.77 | <u>41.97</u> |
| CaMOPD | 26.69 | 31.03 | 28.86 | 49.98 | <u>31.34</u> | 40.66 |
| **Ours** | 27.21 | **36.89** | **32.05** | <u>53.22</u> | **36.51** | **44.86** |

小学生上所有多领域基线的医学均值都跌破 Single OPD 的 35.67，而 Road MOPD 反超至 36.89 / 36.51，对最强基线的医学领先扩大到 3.88 和 5.17 分。虽然 Math/Code 单项略逊 ExOPD（−1.23 / −0.94），但等权平均后分别领先 ExOPD **1.33 和 2.89 分** ——学生越小，跨领域权衡上的收益越大。

### 4.2 低冲突与三领域：不只治"打架"，还能促"合作"

**Math × Code（低冲突，Qwen3-4B 域均值）**：

| 方法 | Math | Code | All |
|---|---|---|---|
| Single OPD | 44.27 | 60.91 | 52.59 |
| Vanilla MOPD | 46.12 | 61.24 | 53.68 |
| ExOPD | **48.28** | <u>61.82</u> | <u>55.05</u> |
| Open-MOPD | 44.32 | 59.82 | 52.07 |
| CaMOPD | 46.30 | 60.37 | 53.34 |
| **Ours** | <u>47.73</u> | **63.07** | **55.40** |

**Math × Code × Medical（三领域，Qwen3-4B 域均值）**：

| 方法 | Math | Code | Medical | All |
|---|---|---|---|---|
| Student | 15.37 | 52.97 | 41.01 | 36.45 |
| Single OPD | 44.27 | 60.91 | 47.90 | 51.03 |
| Vanilla MOPD | 46.98 | 61.26 | 41.03 | 49.76 |
| ExOPD | <u>47.09</u> | <u>61.52</u> | 45.62 | 51.41 |
| Open-MOPD | 44.06 | 60.56 | 38.83 | 47.82 |
| CaMOPD | 45.96 | 60.97 | <u>48.20</u> | <u>51.71</u> |
| **Ours** | **47.29** | **62.94** | **48.45** | **52.89** |

低冲突场景下，Road MOPD 的 Math 虽比 ExOPD 低 0.55 分，但 Code 高出 1.25 分，总分 55.40 居首。三领域场景是更全面的检验（多个辅助教师、聚合一致性、序列化投影），Road MOPD 在 **三个领域全部领先**，总分 52.89 超过 CaMOPD 1.18 分。值得注意的是 Vanilla MOPD 三域平均（49.76）再次低于 Single OPD（51.03），进一步印证冲突问题不是个别现象。

### 4.3 机制验证：分歧确实预示冲突

一个自然的质疑是：token 级一致性 $s_t$ 真的能反映更新冲突吗？作者在固定 checkpoint 上做了探针实验：按批次的分歧度 $S_-=\mathrm{mean}_t[\max(-s_t,0)]$ 把样本分成五档，统计每档中两个教师原始全参梯度余弦为负的批次比例。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/output/pdf/figure3_v2_agreement_responses.png)

> 图解：左图横轴为按分歧度从低到高的五档分组，纵轴为出现梯度冲突的批次比例——Math（蓝）从 24% 升至 40%，Medical（橙）从 46% 升至 76%，分歧越大冲突越频繁，为一致性引导的信用分配提供了实证依据。右图是 Math（上）和 Medical（下）的响应片段可视化：蓝色 token 被衰减（$\lambda_t<1$），橙色 token 被放大（$\lambda_t>1$）。可以看到两位教师对 *Alternatively*（Math）和 *The*（Medical）这类 token 修正方向相反时被降权，而对 *perhaps* / *Maybe* 的一致修正被增强。

### 4.4 与 PCGrad 对比：预防优于事后修补

PCGrad 是对各自领域数据上独立算出的梯度做事后投影；Road MOPD 则是用辅助教师在 **同一条轨迹** 上的梯度做预防性约束。作者在相同训练设置下对比（双方都保留一致性信用分配与范数保持机制）。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/0913_cross_pcgrad_cosine_accuracy_compact.png)

> 图解：(a) 训练中的平均逐层余弦：同轨迹"目标-辅助"梯度对在大部分训练阶段余弦为负，而各自独立形成的领域梯度对大多非负——两种判据捕捉的是不同的局部兼容性，同轨迹约束能发现独立梯度对齐时仍存在的冲突。(b) 领域准确率：在 Math×Medical 和 Code×Medical 上，Road MOPD 四个领域均值全面超过 PCGrad，提升 0.18–2.12 个百分点（如 Code×Med 中 Code 63.02 vs 60.90）。

### 4.5 消融：每个组件都在干活

| Math × Med. | Math | Med. | Avg. |
|---|---|---|---|
| w/o Credit Assign. | 45.78 | 47.74 | 46.76 |
| w/o Grad. Protect. | <u>47.19</u> | 48.11 | <u>47.65</u> |
| w/o Adaptive Ret. | 45.78 | **48.40** | 47.09 |
| **Road MOPD** | **47.60** | <u>48.23</u> | **47.92** |

| Code × Med. | Code | Med. | Avg. |
|---|---|---|---|
| w/o Credit Assign. | 59.54 | <u>47.46</u> | 53.50 |
| w/o Grad. Protect. | <u>63.01</u> | 46.87 | <u>54.94</u> |
| w/o Adaptive Ret. | 62.85 | 47.03 | <u>54.94</u> |
| **Road MOPD** | **63.02** | **47.75** | **55.38** |

去掉信用分配，Math 和 Code 分别掉 1.82 和 3.48 分；去掉梯度保护，Code×Med 的医学掉 0.88 分；把自适应范数保持换成完全投影，两域平均分别掉 0.83 和 0.44 分。三个组件缺一则总分受损，完整方法在两个设置下都取得最高等权平均。

## 五、附录深挖：推导与训练诊断

以下几个附录要点值得向读者展开。

### 5.1 从序列级 reverse-KL 到逐 token 监督

记学生在前缀 $h_t=(x,y_{<t})$ 上的分布为 $p_t$，教师为 $q_t$。由自回归分解和 KL 链式法则：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
=\mathbb{E}_\theta\!\left[\sum_{t=1}^{T}\log\frac{p_t(y_t)}{q_t(y_t)}\right]
=\mathbb{E}_\theta\!\left[\sum_{t=1}^{T}D_{\mathrm{KL}}(p_t\,\|\,q_t)\right]
$$

即序列级 reverse-KL 恰好等于 **学生访问过的前缀上** 逐 token KL 之和——这正是 OPD"在学生轨迹上打分"的理论出处。引入冻结参考模型加减其 log 概率后，可等价写成相对奖励目标 $\mathcal{J}_1(\theta)$，且 $\arg\min_\theta\mathcal{L}_{\mathrm{OPD}}=\arg\max_\theta\mathcal{J}_1$：奖励最大化目标就是 reverse-KL 损失的负数。对相对奖励项单独乘 $\lambda$ 即得到正文的目标；$\lambda=1$ 时参考项相消，回到 OPD。

实现上的 advantage（$\mathrm{sg}$ 为 stop-gradient，整个重算 advantage 在调制后裁剪到 $[-10,10]$ 并 detach）：

$$
A^{\mathrm{agr}}_{d,i,t}
=\mathrm{sg}\!\left[\mathrm{clip}\!\left(
\lambda_{d,i,t}\bigl(\ell^d_{i,t}-\ell^{\mathrm{ref}}_{i,t}\bigr)
-\bigl(\ell^\theta_{i,t}-\ell^{\mathrm{ref}}_{i,t}\bigr),-10,10\right)\right]
$$

注意 $\lambda_{d,i,t}$ 只缩放教师相对参考的奖励，**而不是乘在整个 OPD 损失上**；$\lambda=1$ 时退化为普通 OPD 信号。策略损失用 PPO 风格 clipped surrogate（ratio clip 0.2），prompt 与 padding 位置不参与损失。辅助监督复用同一批存储轨迹，advantage 换成 $\mathrm{sg}[\mathrm{clip}(\ell^j_{i,t}-\ell^\theta_{i,t},-10,10)]$，梯度只用于投影、永不并入更新。

### 5.2 范数约束投影的闭式解

令 $c=\langle g,g'\rangle/(\|g\|_2\|g'\|_2)$。候选解写成 $v(\alpha)=g-\alpha\frac{\langle g,g'\rangle}{\|g'\|_2^2}g'$，目标 $-(1-\alpha)\langle g,g'\rangle$ 随 $\alpha$ 严格下降，所以取可行域内最大系数。模长关系为：

$$
\frac{\|v(\alpha)\|_2^2}{\|g\|_2^2}=1-(2\alpha-\alpha^2)c^2
$$

由此得闭式解：

$$
\alpha^\star=
\begin{cases}
0, & c\geq 0,\\[4pt]
1, & c<0\ \text{且}\ \sqrt{1-c^2}\geq m,\\[4pt]
1-\sqrt{1-\dfrac{1-m^2}{c^2}}, & c<0\ \text{且}\ \sqrt{1-c^2}<m.
\end{cases}
$$

残余冲突为 $\langle\widetilde g,g'\rangle=(1-\alpha^\star)\langle g,g'\rangle$，当部分投影时仍为负——作者也坦诚说明：这些是关于投影后张量的局部结论，不保证优化器走完一步后辅助损失不降。多教师时相对原始目标梯度共享范数下限 $\|g_d^{(q)}\|_2\geq m\|g_d^{(0)}\|_2$；若每次投影独立用 $m$ 只能得到乘积式下界，并非同一约束。

### 5.3 训练动态与范数保持诊断

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/appendix_training_dynamics.png)

> 图解：三个配对运行（Math×Medical、Code×Medical、Math×Code）各 58 次更新的完整训练轨迹，无平滑。上行为记录的 task-0 surrogate 损失（对称 log 刻度，$\pm0.01$ 内线性）——它可为负，只是单域损失贡献而非完整目标，训练早期快速下降；中行为各领域平均响应长度；下行为达到 16,384 token 上限的响应比例。生成长度与顶格比例在损失下降后仍持续波动。

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/appendix_gradient_retention.png)

> 图解：范数保持机制何时真正起作用。蓝线为自适应投影的实际模长比，橙色虚线为同一批梯度上"如果完全投影"的解析估计，点线为下限 $m=0.6$。在 Math×Medical 的 Math 分支，完全投影估计在 58 次更新中有 40 次跌破 0.6、最低仅 0.253，而自适应投影始终守在 0.600；同运行的 Medical 分支从未触限，两线重合。这说明约束是"按需介入"：完全投影不伤信号时不做任何改变。

### 5.4 更多 token 级可视化

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/appendix_token_contexts.png)

> 图解：正文右栏两条轨迹扩展为 256 token 上下文，含原始 prompt，方框标出正文的 52 token 窗口。蓝/橙分别表示衰减与放大，色标固定为 $\lambda_t\in[0.5,1.5]$，token 下标从零开始、半开区间。这些示例选自每个 prompt 的 20 条采样响应，用于展示答案附近同时存在衰减与放大，不代表整体系数分布。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/appendix_token_samples_math.png)

> 图解：AIME2024_0 的三条额外 Math 响应摘录，分别取固定 192 token 窗口 $[0,192)$、$[928,1120)$、$[1856,2048)$（即开头/中间/结尾）。选择过程不看系数与答案正确性，色标与正文一致；圆点表示纯空白 token。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Beyond-Domain-Routed-Supervision-Cross-Teacher-Guarding-for-Multi-Teacher-On-Policy-Distillation/fig/appendix_token_samples_medical.png)

> 图解：MedQA_test_0 的三条额外 Medical 响应摘录，窗口规则与色标同上。注意这些是模型生成的推理片段，并非经过核实的答案解析。

### 5.5 单领域训练的跨领域对照

附录还给出单领域 OPD checkpoint 在全部 11 个 benchmark 上的交叉评测（Qwen3-4B，域均值，百分制）：

| Benchmark 域 | Teacher | Student | Math-only | Medical-only | Code-only |
|---|---|---|---|---|---|
| Code Avg. | 64.20 | 52.97 | 57.70 | <u>58.73</u> | **60.91** |
| Math Avg. | 46.07 | 15.37 | **44.27** | <u>43.10</u> | 31.46 |
| Medical Avg. | 58.05 | 41.01 | <u>45.16</u> | **47.90** | 43.09 |
| 11-Task Avg. | 55.37 | 34.94 | <u>48.26</u> | **49.11** | 43.72 |

两个有意思的细节：Medical-only 模型在 Math 上拿到 43.10，仅比 Math-only 低 1.17 分，医学训练对数学迁移不错；而 Code-only 模型的数学均值只有 31.46，代码向数学的迁移明显弱。这种不对称性也从侧面解释了为什么不同领域组合的冲突程度差异很大。

### 5.6 局限

- **计算开销**：辅助教师打分与梯度处理带来额外训练成本（复用学生轨迹省掉了额外 rollout，但打分和多次反向仍在）；更高效打分与梯度修正留作未来工作。
- **领域规模**：联合训练最多验证到三个领域，更大领域集合与教师池上的增益与稳定性有待检验。

## 六、总结

- **问题定位**：领域路由只分配监督来源，各领域更新在共享参数上可能持续冲突——医学配对中冲突层比例高且随训练上升，Vanilla MOPD 医学分数甚至低于未蒸馏学生。
- **方法**：Road MOPD 让非路由教师在同一条学生轨迹上当"跨领域评委"，token 级用一致性 $s_t$ 调制奖励系数 $\lambda_t\in[0.5,1.5]$，梯度级用同轨迹辅助梯度做带范数保持下限 $m=0.6$ 的预防性部分投影。
- **高冲突场景**：医学均值较 Vanilla MOPD 提升 5.6–8.8 分，4B 学生恢复到 Single OPD 水平（48.23 / 47.75 vs 47.90），1.7B 学生反超 Single OPD。
- **泛化性**：低冲突 Math×Code 总分 55.40 居首；三领域训练三个领域全部领先，总分 52.89 超 CaMOPD 1.18 分；机制探针显示教师分歧度与梯度冲突率正相关，且预防性投影全面优于 PCGrad 的事后修补。
- **消融**：信用分配、梯度保护、自适应范数保持三者缺一，等权平均均下降。

展望：目前的"每位辅助教师都给每条轨迹打分"在教师数量增多时成本线性上升，如何做高效的教师选择与兼容性估计，是这类跨教师监督方法走向更大领域池的关键下一步。

> 本文参考自 [Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2610.13521)