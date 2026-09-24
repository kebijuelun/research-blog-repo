# 过去塑造未来：自回归视频生成中的「记忆」机制全景综述

视频生成模型这两年进化得非常快，短视频的清晰度、运动流畅度已经相当惊艳。但只要把生成时长拉长到分钟级、或者让用户在生成的世界里持续交互，问题就暴露了：角色长着长着换了一张脸、镜头绕一圈回来后场景完全变样、打开的门过一会儿自己「关上了」。这些都不是画质问题，而是 **记忆问题** —— 模型忘记了几十秒前自己亲手生成的世界。

这篇来自 HKUST、CityUHK、CMU、NVIDIA 等机构近三十位研究者的综述《The Past Frames the Future: Memory for Autoregressive Video Generation》，正是围绕这个核心痛点展开的。它第一次把「记忆」作为自回归（Autoregressive, AR）视频生成的核心设计问题进行系统化梳理，沿着 **形态（Forms）、功能（Functions）、操作（Operations）、学习（Learning）、评估（Evaluation）** 五个维度搭起了一套统一框架。这篇文章就带大家把这篇综述的精华过一遍。

![全景图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure2_v1.png)

> 图解：自回归视频生成记忆机制的全景。上半部分的树状图把代表性方法沿两个维度摆放：记忆的「载体形态」（视觉、隐式、显式、参数化）和「保持目标」（身份、空间、动态、语义、因果）；下半部分则勾勒出记忆的操作生命周期（写入、读取、更新、管理、整合）与训练范式。可以看出整个领域已经从「单纯加长上下文」演化成一个有完整体系的设计空间。

## 背景：为什么 AR 视频生成绕不开记忆？

### 统一的形式化：外层 rollout + 内层生成

先把问题数学化。一段 $T$ 帧的原始视频 $\mathbf{x}_{1:T}$ 可以被抽象为 $N$ 个 **自回归视觉单元** 的序列 $\mathbf{y}_{1:N}$ —— 这里的「单元」很灵活，可以是离散 token 块、连续 latent 帧、短 clip 或者多帧 chunk。AR 生成就是把这个序列的条件分布分解成一步步的因果预测：

$$
p_\theta(\mathbf{y}_{1:N}\mid \mathbf{c}) = \prod_{n=1}^{N} p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n}, \mathbf{c})
$$

其中 $\mathbf{c}$ 是外部条件（文本、参考图、相机轨迹、控制指令等）。

这里有一个特别值得强调的观点： **综述把「自回归」定义为外层在视觉单元上的因果 rollout，而不是某一种具体架构** 。内层每个单元怎么生成，有两条主流路线：

- **离散 token 自回归** （VideoGPT、VideoPoet、Emu3 等）：用 VQ-VAE 把画面量化成离散 token，内层再做逐 token 的 next-token 预测：

$$
p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n}, \mathbf{c}) = \prod_{\ell=1}^{L} p_\theta(\mathbf{z}^{(n)}_{\ell} \mid \mathbf{z}^{(n)}_{<\ell}, \mathbf{y}_{<n}, \mathbf{c})
$$

- **连续帧/块级生成** （MCVD、各类 AR diffusion/flow 模型）：内层是一次以历史为条件的去噪或流轨迹：

$$
\frac{d\mathbf{y}_n^{(\tau)}}{d\tau} = f_\theta(\mathbf{y}_n^{(\tau)}, \tau, \mathbf{y}_{<n}, \mathbf{c})
$$

注意 $\tau$ 是扩散/流的内部时间，和外层 AR 步数 $n$ 是两回事。两条路线内层合成方式完全不同，但面临同一个长程难题： **随着 rollout 推进，直接可用的历史必然受限** 。

![背景框架](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure_background.png)

> 图解：左侧对比了离散 token 自回归与连续帧/块生成两种范式 —— 内层生成过程不同，但外层的因果 rollout 完全一致；右侧展示了「记忆条件生成」的核心思想：在有限局部上下文之外，用一个跨步骤持久维护的记忆状态来保存久远历史，与当前上下文、外部条件一起喂给逐步生成器，从而逼近「看到全部历史」的理想生成。

### 有界历史访问：理想与现实的裂缝

理论上条件历史 $\mathbf{y}_{<n}$ 会随 rollout 无限增长，但 self-attention 的平方复杂度、KV cache 的显存占用、延迟约束都决定了：实际系统只能在一个 **有界局部上下文** 上工作：

$$
\mathbf{C}_n = \mathbf{y}_{\max(1,n-W):n-1}
$$

窗口 $W$ 之外的历史 —— 早期出现的实体、场景布局、运动状态、视觉锚点 —— 对生成器直接不可见。在开放式长视频生成中，这会导致信息丢失叠加误差累积；在动作条件/交互式生成（如 Genie 这类世界模型）中问题更严重：

$$
p_\theta(\mathbf{y}_{1:N} \mid \mathbf{c}, \mathbf{a}_{1:N}) = \prod_{n=1}^{N} p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n}, \mathbf{a}_{\leq n}, \mathbf{c})
$$

综述还做了一个很细的区分： **改变观测的控制** （如移动相机，世界本身没变）和 **改变状态的干预** （如推开门、挪动物体）。前者主要考验空间保持，后者则要求「干预的后果」在证据离开窗口后依然持续约束生成。

关于「长上下文」和「记忆」的边界，综述给出的判据是 **操作性的** 而非看窗口长度：每一步都直接喂给生成器的近期 token/帧/KV 状态是 **活跃上下文** ；而当历史信息被持久保存、并通过选择、压缩、索引、检索、整合、修订等操作显式管理时，它才成为 **记忆** 。

### 记忆条件生成与记忆生命周期

于是全历史条件分布被推广为记忆条件形式：

$$
p_\theta(\mathbf{y}_{1:N} \mid \mathbf{c}) = \prod_{n=1}^{N} p_\theta(\mathbf{y}_n \mid \mathbf{C}_n, \mathbf{M}_n, \mathbf{c})
$$

其中 $\mathbf{M}_n$ 是生成 $\mathbf{y}_n$ 之前可用的持久记忆状态，最简单的情况是对被截断前缀的聚合 $\mathbf{M}_n = \mathcal{A}(\mathbf{y}_{<n-W}, \mathbf{c})$。目标自然是逼近全历史条件： $p_\theta(\mathbf{y}_n \mid \mathbf{y}_{<n}, \mathbf{c}) \approx p_\theta(\mathbf{y}_n \mid \mathbf{C}_n, \mathbf{M}_n, \mathbf{c})$。

每个 AR 步骤中，记忆走完一条完整生命周期 —— 这套流程是全文的总骨架：

$$
\mathbf{q}_n = Q_\theta(\mathbf{C}_n, \mathbf{c}), \qquad \mathbf{r}_n = \mathcal{R}_\beta(\mathbf{q}_n, \mathbf{M}_n), \qquad \mathbf{h}_n = \mathcal{I}_\eta(\mathbf{C}_n, \mathbf{r}_n, \mathbf{c})
$$

生成前先由当前上下文形成查询 $\mathbf{q}_n$，从记忆中读出相关内容 $\mathbf{r}_n$，与局部上下文整合成 $\mathbf{h}_n$ 再生成；生成之后提取写入候选 $\mathbf{w}_n = \mathcal{W}_\alpha(\mathbf{y}_n, \mathbf{C}_n, \mathbf{c})$，更新持久状态 $\widetilde{\mathbf{M}}_{n+1} = \mathcal{U}_\phi(\mathbf{M}_n, \mathbf{w}_n)$，最后由管理算子在预算 $B$ 下做保留/压缩/修订：

$$
\mathbf{M}_{n+1} = \mathcal{G}_\gamma(\widetilde{\mathbf{M}}_{n+1}; B)
$$

### 无记忆 rollout 的六大失效模式

为什么必须这么做？因为没有持久记忆的 rollout（memoryless rollout）会系统性地出现六类症状，它们构成了后文「功能」章节的靶子：

1. **实体遗忘** ：角色或物体离开视野后再出现时被省略、复制或换了属性，人脸身份漂移是典型例子。
2. **外观漂移** ：纹理、颜色、光照、风格逐渐偏移，小误差被反复反馈放大。
3. **空间不一致** ：镜头回到旧场景时布局、几何、物体摆放对不上。
4. **动态退化** ：运动冻结、相位错乱、违反物理接触约束。
5. **语义漂移** ：忘记已发生的事件、角色关系、未完成的目标，叙事失去推进。
6. **因果/状态不一致** ：开过的门自己关上、挪走的物体回到原位 —— 干预被当成了瞬时控制而非持久状态更新。

一句话总结：长程 AR 生成的核心挑战不是「时长」本身，而是 **有界历史访问下的时间持久性** 。

## 形态（Forms）：什么在承载记忆？

第一个问题：历史信息到底存在什么计算对象里？综述按「记忆载体」分出四大类：

- **视觉记忆（Visual Memory）** ：直接保存与观测对齐的视觉证据（RGB 帧、clip 或 VAE latent）。
- **隐式状态记忆（Implicit State Memory）** ：把历史保留在模型原生的神经状态里（KV cache、循环隐状态等），语义是学习出来的而非人为规定的。
- **显式状态记忆（Explicit State Memory）** ：用有明确语义的结构化变量表示历史（实体、位姿、占据场、事件状态等）。
- **自适应参数化记忆（Adaptive Parametric Memory）** ：把序列/实例特定的信息编码进可更新的参数（TTT 快权重、临时 LoRA 等）。

前三类是非参数化的运行时状态，第四类把持久性放进了参数。实际系统完全可以混搭使用。

![记忆载体形态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure_forms.png)

> 图解：四类记忆载体的示意与代表性实现。历史上下文可以持久化为视觉记忆（帧/latent 证据）、隐式状态记忆（KV cache、SSM/循环状态、编码历史）、显式状态记忆（点云、体素、实体记录）或参数化记忆（快权重、LoRA），并且可以组合起来共同条件化因果主干网络。

### 视觉记忆：像素空间与 VAE 空间

视觉记忆的形式化很直白： $\mathbf{M}_n^{\mathrm{vis}} = \{E_{\mathrm{vis}}(\mathbf{y}_i) \mid i \in \mathcal{I}_n^{\mathrm{vis}}\}$，关键在于每个记忆单元都能追溯到具体的历史观测。

**像素空间** 里又分帧级和 clip 级：帧级记忆有三条策略谱系 —— 近期帧保留（MCVD，其实更像活跃上下文）、稀疏锚点帧（ConsistI2V 这类保留长程参考）、检索式（Context-as-Memory 按需激活非局部帧）。clip 级记忆额外保留了短程运动和时间顺序证据（如 LGC-VD、LCT），代价是更高的存储和条件化开销，而且过时的运动先验可能误导当前生成。

**VAE 空间** 记忆存的是可解码回观测的 latent： $\mathbf{m}_i^{\mathrm{vae},0} = E_{\mathrm{VAE}}(\mathbf{x}_{s_i:u_i})$，扩散模型还可以保留在某个中间噪声水平 $\tau$ 上： $\mathbf{m}_i^{\mathrm{vae},\tau} = \alpha_{\tau}\mathbf{m}_i^{\mathrm{vae},0} + \sigma_{\tau}\boldsymbol{\epsilon}_i$。实现模式包括顺序携带（LVDM）、窗口队列（FIFO-Diffusion、PA-VDM）、检索式（LongLive-RAG）。它比 RGB 更紧凑、与 latent 生成器天然匹配，但编码器丢掉的信息不可恢复，且与具体 autoencoder 强耦合。

### 隐式状态记忆：三种神经状态

隐式记忆的状态演化写作 $\mathbf{M}^{\mathrm{imp}}_{n+1} = \mathcal{S}_{\mathrm{imp}}(\mathbf{M}^{\mathrm{imp}}_n, \mathbf{y}_n, \mathbf{C}_n, \mathbf{c})$，细分为三小类：

- **Attention-cache 状态** ：跨 AR 步骤被管理的持久 KV 状态 $\mathbf{M}^{\mathrm{KV}}_n = \{(K_i^{(\ell)}, V_i^{(\ell)}, p_i)\}$。保留结构上有 sink+local（LongLive、Rolling Forcing，师承 StreamingLLM）、近端-远端分层（RELIC、Context Forcing）、多级缓存（PackForcing）、选择性固化（Sparse Forcing、FadeMem）等。另一个越来越受关注的子问题是 **可寻址远端缓存** ：WorldKV 用相机/动作对应关系激活历史 KV；WorldTrace 则解决「存着但注意不到」的问题 —— 远端 KV 的时间偏移落在训练分布之外时会被重映射到分布内的虚拟位置。可见 **容量和可达性是两回事** 。
- **循环与状态空间状态** ：不为每个历史单元留独立条目，而是把历史不断折叠进一个演化状态 $\mathbf{s}_n$（LSTM、Mamba、线性注意力状态）。代表有 MALT（段级循环）、MemoryPack（全局摘要+局部上下文分离）、VideoSSM / Hybrid Forcing（局部精确注意力 + 远程循环摘要的混合）。优点是状态大小可与 rollout 长度解耦，适合流式生成；缺点是一旦折叠就无法单独回访，稀有信息易受干扰。
- **编码历史状态** ：由专门的历史编码器/压缩器产生的持久神经表示 $\mathbf{M}^{\mathrm{enc}}_n = E_{\mathrm{mem}}(\mathbf{y}_{\mathcal{I}_n}, \mathbf{c})$，从保留源结构的 TinyHistory 到高度压缩的 Infinite-World、Echo-Infinity，压缩越强，对「编码时保留了什么」的依赖就越重。

### 显式状态记忆：实体中心与空间几何

显式记忆被表示为一组带类型的状态变量 $s_{n,i} = (\rho_i, \kappa_i, v_{n,i})$，每个变量有声明好的世界语义。

**实体中心状态** 为每个被追踪的实体维护 $(\mathrm{id}_k, e_{n,k})$，字段可包括类型化属性、位姿、速度、交互/生命周期状态。低维的如 ActionParty（持久身份+紧凑位姿），丰富的如 A$^2$RD（类型化属性、关系、任务状态）。它把「世界状态演化」和「像素合成」解耦，支持遮挡期间的持续演化（LiveWorld、WorldDirector 维护画外物体的运动状态），但可靠性取决于状态空间设计和更新的正确性。

**空间与几何状态** 按索引域分成四类：相机位姿与轨迹（提供跨视角对齐的几何基准，如 WorldMem、AnchorWeave）；2D 场景表示（深度、光流、分割等视角对齐场，如 WorldWeaver、Geometry-as-Context）；3D 场景表示（点云、体素、surfel、Gaussian 的世界坐标整合，如 PERSIST、Voyager、Mem-World 的 4D surfel）；拓扑布局（节点+类型化边的非度量空间关系，如 MultiGen）。空间结构越强，跨视角一致性越好，但对标定、对应关系、状态融合精度的依赖也越重。

### 自适应参数化记忆

这类记忆把持久性放进参数： $\mathbf{M}^{\mathrm{par}}_n = \boldsymbol{\phi}^{\mathrm{mem}}_n$，通过适应过程 $\mathcal{A}$ 从证据 $\mathcal{D}_n$ 更新。注意不是任何参数都算参数化记忆 —— 预训练参数是通用知识；只有从经验中适应、跨步骤持久存在并影响后续生成的参数状态才算。

- **内部参数化记忆** ：快权重循环（LaCT 的大块 TTT、RAD-TTT）把历史逐步固化进序列层权重；ISPA 则用闭式最小二乘把被驱逐的历史 KV「吸收」进注意力投影权重。
- **模块化参数化记忆** ：结构上与主干分离的适应模块，如 SlowFast-VGen 推理时在线更新的时序 LoRA、HippoCampus 的在线快权重全局记忆、TTOM 的可存取/检索/删除的键控参数记忆库。模块化带来更好的生命周期控制，但内容依然不可解释，且多个参数记忆之间会引入检索、组合、干扰的新问题。

## 功能（Functions）：记忆到底要保住什么？

载体回答「存在哪」，功能回答「要保住什么」。综述定义 **记忆功能** 为：原始证据离开局部上下文后，持久状态所承担的具体生成职责。核心原则是 **选择性不变（selective invariance）** —— 记忆不是把世界冻住，而是保住该不变的、放行该变化的（位姿、光照、运动、世界状态都可以合法演变）。

五大保持功能与前述失效模式一一对应：

![记忆功能](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure_function.png)

> 图解：左侧是有界局部上下文下长程 AR 生成的各类失效模式；右侧将其映射到记忆系统必须履行的五大保持功能 —— 身份、空间、动态、语义、因果；底部的 Rollout Reliability（rollout 可靠性）是横向支撑项，用于稳定递归自生成循环，而不构成第六种保持功能。

![功能示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure_fuction_2.png)

> 图解：两段示例 rollout 展示五种功能在真实生成中的体现 —— 相机导航主要考验空间保持，未被观测的运动实体考验动态保持，物体操控引入因果依赖，多镜头叙事则对身份与语义保持提出更高要求。

### 身份保持（Identity Preservation）

关心的是「还是不是同一个实体」，而非所有可见属性是否不变。三条策略线：

- **参考锚定** ：保留历史身份证据作为长程锚，如 StreamingT2V 的首 chunk 外观特征、Gloria 的内容锚、Anchor Forcing 的锚记忆+三区 RoPE、Knot Forcing 的参考图像 KV + 跨 chunk 时间结。共同风险是锚会变陈旧，压制合法变化，需要配套的选择性更新/失效机制。
- **实体感知寻址** ：多实体共存时要解决「该回忆谁」的问题，如 VideoMemory 的动态视觉-语义记忆库、SlotMem 的角色槽、IAMFlow 的身份注册表。其特征性失败是 **绑定错误** ：把当前观测匹配到错误的记忆条目，导致身份互换、属性泄漏。
- **缺席期间的持久性（object permanence）** ：实体暂时不可见时仍应保持其表征，但不应简单回放最后可见状态 —— 缺席期间实体可能合理演变。

### 空间保持（Spatial Preservation）

任务从「回忆看到过什么」变成「从当前视角应该看到什么」。时间近邻是很差的相关性代理 —— 一条很老的观测可能与目标视角重叠，而最近几帧可能拍着不相干的区域。三条路线：

- **空间寻址检索** ：WorldMem 按位姿+时间索引、Context-as-Memory 检索共视视角、VMem 用 surfel 索引、Memory Forcing 把可见 3D 点映射回源帧、WorldKV 用相机/动作对应激活 KV。可靠性取决于对应精度 —— 位姿漂移会检索回空间不兼容的证据，产生鬼影和几何幻觉。
- **持久场景状态** ：把历史整合成世界坐标系下的点云/体素/占据场（SPMEM、EvoWorld、PERSIST、MoVerse）。支持导航和闭环，但位姿/融合错误会变成持久结构错误，动态物体可能被错误吸进静态地图。
- **隐式空间对齐** ：不显式重建世界模型，如 WorldPack 的几何打包、RELIC 的位姿感知压缩 KV、ViewRoPE 的几何感知旋转位置编码、WorldTrace 的远端状态重映射。避免了显式重建的代价，但空间对应不易检查。

### 动态保持（Dynamic Preservation）

关心状态「该如何继续演化」：运动冻结、速度漂移、相位错乱都是其失败表现。关键子问题：

- **状态与相位连续性** ：未来状态要与先前演化一致，而非退回最后一次观测。DFoT、VideoAR、MALT、各类 SSM、Grounded Forcing 都在不同载体上实现这一点，共同的权衡是时间压缩可能丢掉后期才关键的速度/相位线索。
- **画外演化（out-of-sight evolution）** ：被遮挡的物体不应冻结在最后可见状态。LiveWorld 在持久场景表示上推进未观测实体，WorldDirector 让语义运动状态独立于可见性传播。
- **自 rollout 下的稳定性** ：预测误差会改写后续运动推断所依赖的历史，Self Forcing、Causal Forcing 等训练策略以及 FIFO-Diffusion、FramePack 等历史管理设计是横向支撑 —— 记忆可能忠实地保留了一条已经错误的轨迹并加以强化。

### 语义保持（Semantic Preservation）

维护 rollout 中建立的高层承诺：目标、角色、关系、事件、未完成的任务。初始 prompt 不够用，因为很多承诺是 rollout 过程中才涌现的。要点包括：承诺与指称绑定（MemoryPack 的循环语义状态、SlotMemory 的对象中心 KV，绑定失败会出现角色互换）；事件与叙事状态（ReCA 的类型化叙事状态+关键帧、InfinityStory 的角色感知转场）；上下文切换下的选择性修订（SWIFT 的语义注入缓存、Visko Orbis 的有界多尺度记忆 —— 保留所有旧条件会让记忆陈旧，激进替换又会抹掉仍有效的承诺）。

### 因果保持（Causal Preservation）

最严格的一项：保持 **干预引起的状态改变** 。相机移动只改变观测，而开门、挪物建立了新状态，后续合成必须尊重。三个环节：干预-状态绑定（VRAG 的全局状态条件、ActWorld 的对象身份+事件更新 token，避免短暂但后果重大的转变被新近性压缩丢弃）；持久性与后果传播（PERSIST 的可编辑 3D 状态、WorldCraft 操控后刷新记忆；典型失败是 **先验回退** —— 证据消失后生成器退回视觉上更可能的「标准配置」）；有序修订与冲突解决（先开后关的门应保持关闭，需要覆盖、作废机制而非平等保留所有版本）。

此外五种功能在实践中是耦合的：身份与语义通过指称接地耦合，空间与动态在持久环境中相互约束，因果则同时对其他功能提出联合要求。

## 操作（Operations）：记忆如何工作？

持久状态要生效，必须靠操作。五大操作构成完整生命周期： **写入、读取、更新、管理、整合** 。

![记忆操作生命周期](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure_operation.png)

> 图解：上方是记忆的操作生命周期 —— 每一步生成前先从持久记忆中「读取」并「整合」相关信息；生成后将新证据「写入」「更新」并在预算约束下「管理」，形成供后续步骤使用的新记忆状态。下方展示了这些操作在视觉、隐式、显式、参数化四类载体上的代表性实现。

### 写入：准入与绑定

写入候选 $\mathbf{w}_n = \mathcal{W}_\alpha(\mathbf{y}_n, \mathbf{C}_n, \mathbf{c})$ 涉及两个决策： **准入** （哪些证据该持久化）和 **绑定** （存到哪个地址）。策略谱系：

- **穷举准入** （FIFO-Diffusion 入队一切）：不漏，但把选择推迟到了下游管理，冗余和自生成伪影照单全收。
- **锚点准入** （LongLive、FreqForcing）：稀疏保留长程参考，低成本但会陈旧。
- **效用选择** （Grounded Forcing、RECAP-Forcing 按显著性/新颖度打分）：更高效，但未来效用准入时只能部分观测 —— 一个不起眼的背景物体可能在镜头回访时变得至关重要。
- **转变触发写入** （ActWorld、SWIFT）：由动作发起、物体接触、镜头边界、prompt 切换等事件触发，适合「稀疏改变、长期后果」的场景，但依赖可靠的触发检测。
- **结构化绑定** （ActionParty、Memory Forcing、ReWorld）：$\mathbf{w}_n[k] \leftarrow \mathcal{W}_\alpha(\mathbf{y}_n, \mathbf{C}_n, \mathbf{c}; k)$，把证据写到实体 ID/坐标/地图元素下，索引错误会让正确证据污染后续读取。

写入要平衡三种风险： **遗漏、污染、错绑** 。

### 读取：找对证据

读取 $\mathbf{r}_n = \mathcal{R}_\beta(\mathbf{q}_n, \mathbf{M}_n)$ 决定哪些已存信息对当前步可用 —— 存着但找不到，等于没有。六类读取方式：

- **时间读取** ：滑窗、滚动 KV，高效但「近」不等于「相关」。
- **内容检索** ：按视觉/语义相似度选（Resampling Forcing、Focused Forcing），但相似性有歧义，可能召回指错实体的证据。
- **语义寻址** ：按实体 ID、角色、事件键访问（VideoMemory、SlotMemory），失败模式是绑定错误。
- **空间索引** ：用相机位姿、深度、目标视角做几何对应查询 $\mathbf{r}_n = \mathcal{R}_\beta(\pi_n, \mathbf{M}_n)$（WorldKV、DreamX-World），位姿漂移会引入持久几何不一致。
- **状态条件检索** ：按实体状态、交互历史、任务状态访问（ActWorld），依赖状态同步。
- **注意力路由** ：隐式记忆通过注意力暴露 $\mathbf{r}_n = \mathrm{Attn}(\mathbf{q}_n, \mathbf{K}_{\mathbf{M}_n}, \mathbf{V}_{\mathbf{M}_n})$（SparSTAR 只读得分最高的历史 KV 块），但远端状态可能「存着却注意不到」，需要 Wonder 的稀疏全保真保留或 WorldTrace 的位置重映射。

### 更新：稳定与可塑的平衡

更新算子 $\widetilde{\mathbf{M}}_{n+1} = \mathcal{U}_\phi(\mathbf{M}_n, \mathbf{w}_n, \mathbf{C}_n, \mathbf{c})$ 决定新证据如何改变已有状态。六种机制：

- **仅追加扩展** ：帧库、latent 队列，不破坏旧证据但不解决冲突，版本共存带来歧义。
- **循环固化** ：$\mathbf{s}_{n+1} = \mathcal{U}_\phi(\mathbf{s}_n, \mathbf{w}_n)$（AdaState、Ripple、S2DiT），规模友好但牺牲情节可分离性，错误转变会持续遗传。
- **缓存刷新与重映射** （Knot Forcing、UniSwap、Visko Orbis）：KV 状态与位置编码、激活统计耦合，刷新出错会让内容正确的证据变得不可用。
- **结构化状态更新** （WorldWeaver、PERSIST、Mem-World）：配准错误会把局部估计误差变成持久记忆错误。
- **世界状态转变** ：$\widetilde{\mathbf{M}}_{n+1} = \mathcal{U}_\phi(\mathbf{M}_n, \mathbf{w}_n, \mathbf{a}_n)$，无新观测时依赖旧状态+学到的动力学推进（WorldDirector、Programmable World Model）。
- **纠正性修订** （TokenTrim 检测漂移、移除不稳定 token 并重新生成）：难点是区分「真损坏」与「合法变化」，过度纠正会抹掉有效历史。

三类反复出现的失效：更新不足（留旧）、更新过度（删对）、更新错误（存错）。

### 管理：预算下保什么、怎么保

管理算子 $\mathbf{M}_{n+1} = \mathcal{G}_\gamma(\widetilde{\mathbf{M}}_{n+1}; B)$ 决定哪些信息以何种保真度存活：

- **滚动截断** ：FIFO 窗口，成本可控但把年龄当相关性代理。
- **锚点/多级保留** ：$\mathbf{M}_n = \mathbf{M}^{\mathrm{anchor}}_n \cup \mathbf{M}^{\mathrm{local}}_n$（LongLive、JoyStreamer-Flash），保护长程参考但可能保留陈旧约束。
- **效用感知保留** ：$\mathbf{M}_{n+1} = \operatorname{TopK}_{m} \, s_\gamma(m)$（Forcing-KV 按注意力头分工剪枝、DySink 动态选 sink），依赖不确定的未来效用估计。
- **压缩与固化** ：$\mathbf{M}^{\mathrm{comp}}_n = \mathcal{C}_\gamma(\mathbf{M}_n)$（Light Forcing、Ring Forcing、LongLive-2.0 的低比特 KV），延长有效视野但引入信息瓶颈 —— 注意这与循环更新不同：循环是把新证据融入状态，压缩是改变已有状态的表示精度。
- **有效性感知遗忘** ：$\mathbf{M}_{n+1} = \widetilde{\mathbf{M}}_{n+1} \setminus \mathcal{D}_\gamma(\cdot)$（StableWorld 主动删除几何退化的帧），过时证据留着比删掉更危险，但过早失效化会误杀长程证据。

### 整合：让记忆真正影响生成

读取决定哪些记忆可用，整合决定它通过哪条路径、以多大强度影响生成： $\mathbf{h}_n = \mathcal{I}_\eta(\mathbf{C}_n, \mathbf{r}_n, \mathbf{c})$。

- **上下文拼接** ：最简单，但对远端条目的有效注意力无保障，还吃掉本就紧张的上下文预算。
- **注意力条件化** ：$\mathbf{h}_n = \operatorname{CrossAttn}(\mathbf{Q}_n, \mathbf{K}_{\mathbf{r}_n}, \mathbf{V}_{\mathbf{r}_n})$，支持异构记忆，但依赖 query-key 表示对齐，大检索集会稀释注意力。
- **KV 状态注入** ：$\mathbf{h}_n = \mathrm{Attn}(\mathbf{Q}_n, [\mathbf{K}_{\mathrm{local}};\mathbf{K}_{\mathbf{r}_n}], [\mathbf{V}_{\mathrm{local}};\mathbf{V}_{\mathbf{r}_n}])$，保留细粒度视觉/运动信息，但历史 KV 与位置编码、激活统计强耦合（LoL 调整位置频率、Quantized Keys Steal Attention 修正量化 key 的 logit 偏差）。
- **自适应调制与门控** ：$\mathbf{h}_n = \lambda_n \odot \mathcal{I}_{\mathrm{mem}}(\mathbf{r}_n) + (1-\lambda_n) \odot \mathcal{I}_{\mathrm{local}}(\mathbf{C}_n)$，$\lambda_n$ 可依赖相关性、置信度、空间重叠等（Memorize When Needed、TetherMem 的注意力 logit 先验）。核心难点是校准：过强会施加陈旧约束，过弱则让正确检索的证据形同虚设。
- **几何对齐条件化** ：$\mathbf{r}^{\mathrm{view}}_n = \Pi(\mathbf{M}^{\mathrm{spatial}}_n, \pi_n)$，把点云/体素投影成视角对齐条件，空间接地强但几何误差直接传导。
- **结构化语义条件化** ：以结构化 token 暴露实体/关系/事件状态，保得住「现在是什么状态」但欠定「看起来什么样」，通常要与视觉/空间记忆配合。

## 学习（Learning）：记忆行为如何被训练出来？

记忆学习本质是 **闭环** 问题：模型生成的帧会写回记忆、重塑后续条件，预测误差和记忆状态误差会在长程上复合。闭环 rollout 的一步可以写成：

$$
\mathbf{r}_n = \mathcal{R}_\beta(\mathbf{q}_n, \mathbf{M}_n), \qquad \hat{\mathbf{y}}_n = G_\theta(\boldsymbol{\epsilon}_n; \mathcal{I}_\eta(\mathbf{C}_n, \mathbf{r}_n, \mathbf{a}_n, \mathbf{c})), \qquad \mathbf{M}_{n+1} = \mathcal{F}(\mathbf{M}_n, \hat{\mathbf{y}}_n, \mathbf{a}_n)
$$

![记忆学习闭环](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure_learning.png)

> 图解：记忆学习的闭环结构。生成的输出被写回记忆并作为未来生成的条件，预测误差会改变后续记忆状态并随时间累积 —— 这就是为什么单步生成质量再好，也不等于长程记忆可靠。

综述把记忆学习沿三个维度组织：学习目标、记忆状态分布、记忆感知学习。

### 学习目标：监督谁？

- **输出级预测监督** ：标准的去噪/流匹配目标

$$
\mathcal{L}_{\mathrm{pred}} = \mathbb{E}_{n,\sigma,\boldsymbol{\epsilon}} \left[ \left\| f_\theta(\mathbf{z}_n^\sigma, \sigma; \mathbf{C}_n, \mathbf{M}_n, \mathbf{c}) - \mathbf{f}^{*}(\mathbf{z}_n, \boldsymbol{\epsilon}, \sigma) \right\|_2^2 \right]
$$

按预测单元粒度可分 token 级（VideoGPT、Emu3、Loong）、帧级（RVD、DIAMOND、NOVA）、多尺度级（SAMPO、VideoAR 的 next-scale 式预测）、chunk 级（MAGI-1、InfinityStar）。单元越大，外层决策越少，但每步合成更复杂、误差以更粗粒度进入后续条件。输出级监督有三个固有盲区： **监督稀释** （损失被局部可观测内容主导）、 **监督延迟** （一次写入的后果几十步后才可见）、 **记忆状态欠约束** （不同记忆内容可能产生同样的即时输出）。
- **记忆状态监督** ：$\mathcal{L}_{\mathrm{mem}} = \mathbb{E}_n [d(P_\omega(\mathbf{M}_n), \mathbf{m}_n^{\mathrm{tgt}})]$，直接约束记忆编码的内容。监督目标可以是显式世界状态变量（PERSIST 的 3D 状态、PanoWorld 的全景深度），也可以是预训练模型的高维特征（GIM-World 用冻结 3D 基础模型的几何特征）；另一条路是状态条件生成 —— 把构建好的记忆作为条件输入，让生成损失本身检验记忆的可用性（Memory Forcing）。
- **记忆动态监督** ：约束多步状态演化

$$
\mathcal{L}_{\mathrm{dyn}} = \sum_{k=1}^{K} w_k \, d\!\left(\mathcal{F}^{(k)}(\mathbf{M}_n, \hat{\mathbf{y}}_{n:n+k-1}, \mathbf{a}_{n:n+k-1}), \mathbf{M}_{n+k}^{*}\right)
$$

长程结果约束如 Next Forcing（同一状态预测多个未来 chunk）、VIGOR（跨帧 3D 点重投影误差）、VideoRLVR（规则化奖励）；转移路径约束如 LIVE（前向 rollout 后反向重建初始状态的可恢复性目标）、Delta Forcing（teacher 与学生轨迹间的自适应信任域）。

### 记忆状态分布：训练时见过什么样的记忆？

这是 exposure bias 在记忆上的推广。Teacher forcing 用真实历史构建记忆，部署时记忆却由模型自己的输出递归构建，两者诱导出不同的状态分布：

$$
q_{\mathrm{TF},n} \neq q_{\theta,n}
$$

三个应对层次：

- **Teacher forcing** （ACDiT、Ca2-VDM、GPDiT）：真实历史条件下并行训练，稳定高效，可做 Causal Forcing 蒸馏的初始化，但完全不暴露模型自身的误差状态。
- **历史增强** ：对真实历史施加变换以拓宽分布 $q_{\mathrm{aug},n}(\mathbf{M}) = \int K_\psi(\mathbf{M}\mid\mathbf{M}') q_{\mathrm{TF},n}(\mathbf{M}') \,\mathrm{d}\mathbf{M}'$。包括时间无结构扰动（GameNGen 的随机噪声、Diffusion Forcing 的独立噪声水平）、时间结构化扰动（Rolling Diffusion 的渐进噪声、AR-Diffusion 的非递减损坏、SkyReels-V2）、模型导出扰动（Resampling Forcing 用在线模型完成部分去噪轨迹、Stable Video Infinity 的误差回放库）。但这些都只是 $q_{\theta,n}$ 的近似 —— 预设扰动复现不了身份漂移、几何畸变这类自反馈耦合错误。
- **自 rollout** （Self Forcing 是代表）：训练时就用 KV cache 自回归生成、把生成块喂回缓存、对整段序列做视频级监督，直接采样部署分布。代价是 $O(NS)$ 的串行求值和跨展开步的激活存储，实践上用少步生成器、截断反传来缓解。后续工作在两个轴上展开：rollout 组织（RAVEN 的重打包、Rolling Forcing 的滚动窗口、BAgger 的反向纠错轨迹）与 rollout 状态监督（Self Gradient Forcing 的免梯度 rollout + 并行上下文梯度、Self-Forcing++ 的短程双向 teacher 分布匹配、OPSD-V 的稠密速度目标）。

### 记忆感知学习：让训练见到部署时的接口

即使历史分布对齐了，训练和部署的 **记忆接口** 还可能不一致：部署时只有部分历史被保留、压缩或按需检索。记忆感知学习就是在训练时显式纳入部署期的容量、表示、访问约束，通用目标为：

$$
\mathcal{L}_{\mathrm{aware}} = \mathbb{E}_{(\mathbf{H}_n,\mathbf{y}_n,\mathbf{a}_n,\mathbf{c}) \sim p_{\mathrm{train}},\, B \sim p_B} \left[ \mathcal{L}_{\mathrm{pred}}(\theta; \mathbf{y}_n, \mathbf{h}_n) + \lambda \, \mathcal{L}_{\mathrm{mem}}(\mathbf{M}_n, \mathbf{H}_n) \right]
$$

四条路径：

- **学会保留** ：预定义策略下训练（LongLive 的 sink+短窗流式微调、Reward Forcing 的 EMA-Sink 吸收被逐 token）或联合学习保留策略（Sparse Forcing 学习保留显著 KV 块、PaFu-KV 从双向 teacher 蒸馏显著性头）。难点在于短期显著不等于长期有用。
- **学会压缩** ：显式历史压缩（FramePack 让总上下文长度近似恒定、PackForcing 对中段历史施加更强时空压缩、VideoMLA 的低秩 latent KV）与状态摘要压缩（TinyHistory 的轻量历史编码器、ARL$^2$ 的定长线性注意力状态、Echo-Infinity 的可学习 Memory Queries）。
- **学会检索** ：结构化规则（Context-as-Memory 的视场重叠过滤、COVRAG 的边际覆盖增益、MosaicMem 的 3D patch 对齐）或 learned 检索（LongLive-RAG 的离线检索编码器、MemLearner 的 query token、Ring Forcing 故意把目标证据放到远端历史里逼模型学会长程检索）。
- **学会适应** ：把经验写进可更新参数状态

$$
\Delta\theta_{u+1} = \mathcal{U}^{\mathrm{par}}_\phi(\Delta\theta_u, \mathbf{e}_u), \qquad \hat{\mathbf{y}}_n = G_{\theta \oplus \Delta\theta_{u(n)}}(\boldsymbol{\epsilon}_n; \mathbf{h}_n, \mathbf{a}_n)
$$

包括循环快权重适应（LaCT、RAD-TTT、HippoCampus）与局部化参数适应（SlowFast-VGen 的推理时 LoRA、ISPA 的闭式最小二乘写入）。参数化记忆不占上下文 token，但把成本和失效模式转移到了参数更新过程：干扰、遗忘、自我生成误差的固化。

## 评估（Evaluation）：如何证明模型真的「记得」？

这是全文方法论上最犀利的一节。核心论断： **视频质量好不等于有记忆** 。模型可以不靠历史也生成连贯结果 —— 目标可能还在当前窗口里、prompt 可能已经复述了答案、学到的先验可能直接猜对。

> **评估原则** ：一个能揭示记忆的评估，必须让先前历史成为必要条件、验证被查询行为本身可执行、并排除非记忆的解释。

### 基准的三个证据等级

综述按「证据强度」而非数据集规模给现有基准分级：

- **记忆导向基准** ：显式制造信息差（实体消失-重现、离开-回访、隐藏状态演化），如 MBench、MemoBench、StEvo-Bench、LiveBench、MIND、WBench、WorldRoamBench。注意证据强度可以在子集层面成立 —— 例如 WBench 只有 gated camera-return 子集满足标准。
- **邻近的理解侧记忆基准** （StreamMemBench、EGOSTREAM 等）：输出是答案而非生成视频，只作协议参考。
- **序列压力测试** （VBench-Long、StoryBench、InterVBench 等）：暴露漂移和误差累积，但目标始终可见时成功不能归因于记忆。

![评估协议](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/bench_fig.png)

> 图解：按「当前查询是否需要 rollout 历史」组织的评估协议谱系。可见一致性只是基线（目标仍然可见）；已确立的信息差协议包括实体重现、场景回访、画外状态演化；目标/叙事延续与延迟动作效应仍是开放方向，现有基准多只提供代理或协议组件。

### 记忆揭示协议与指标

协议负责「让历史变得必要」，指标负责「检验延迟查询是否答对」。同一个分数在不同协议下意义完全不同 —— 相邻帧的 DINO 相似度量的是可见一致性，而消失-重现之后的同一分数才能量身份保持。

- **实体重现** ：先验证真的重现了（检测/分割），再比身份属性（DINO/CLIP/ArcFace/VLM），顺序不能反，否则选中帧上的相似度会掩盖遗漏和替换。
- **场景回访** ：外观（PSNR/SSIM/LPIPS/DreamSim）与几何（位姿误差、深度/重投影、Chamfer 距离）要互补，二者缺一不可。
- **画外状态演化** ：目标从「重建旧图」变成「识别最新合法状态」，用状态分类器、进度标签、事件谓词、VQA rubric；前提控制是模型得先在可见条件下能生成该转变。
- **目标/叙事延续** ：目标完成度、矛盾检测、事件覆盖率/顺序/遗漏/重复，但只有在承诺从后续局部条件中移除后才算记忆证据。
- **延迟动作效应** ：先验证动作执行成功，再测延迟后果保持 —— 目前尚无标准化的延迟动作记忆指标，是明确的空白。
- **历史-查询分离曲线** ：按重现间隔、路径长度、隐藏时长、交互深度分层报告；退化斜率、首次失败时间、保持半衰期可作摘要但不能替代曲线本身（缓变漂移和突然崩溃可能有相同的端点分数）。

### 有效性控制与归因

高分可能来自 prompt 泄漏、可见线索或 learned prior；低分可能是执行失败、渲染失败或评估器失败。四类互补控制： **配对历史控制** （固定当前观测、只改相关历史，看输出是否跟着变 —— 最接近因果检验，但目前少有生成基准实现）； **协议前提检查** （回访前先确认离开成功、位姿闭合达标）； **组件级干预** （禁用/替换/腐坏记忆模块看行为是否变化，能定位机制但结论绑定具体架构）； **评估器可靠性** （VLM 裁判的时序采样覆盖率、与人类判断的校准）。

## 未来方向：还缺什么？

![未来路线图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/The-Past-Frames-the-Future-16-12-Memory-for-Autoregressive-Video-Generation-----A-Survey/figures/Figure_future.png)

> 图解：记忆增强 AR 视频生成的路线图。上方路径展示三次转变：从异构载体走向统一且资源感知的记忆、从门控更新走向可信的自 rollout 学习、从动作条件世界模型走向可交互、因果化、可分支的记忆；下方路径强调评估的平行演进 —— 从新兴的记忆导向基准走向保持力、归因、效率的标准化测量。

1. **统一且可组合的记忆** ：目标不是把四类载体压成一种表示，而是建立共享接口 —— 统一的实体标识、时间戳、坐标系、来源、置信度、有效期，支持跨载体一致性校验、局部修订、作废与重置；层级化组合让易失细节留在视觉/隐式记忆，稳定历史固化为紧凑结构化状态。
2. **资源感知记忆** ：把记忆分配当作前瞻性的效用问题而非回溯性的显著性决策 —— 低频但高影响的证据（早期身份线索、后果持久的干预）应被优先保护；存储、访问成本、检索延迟应与记忆性能一起报告。
3. **可信的记忆更新** ：区分「存储的证据」与「当前视为有效的状态」，证据携带来源与置信度，冲突按可靠性、时序、跨载体一致性裁决；修订应局部化、可回滚，并在闭环 rollout 中用矛盾证据、腐坏历史、延迟纠正来训练。
4. **自 rollout 学习** ：重点不是更长轨迹而是更好覆盖「既可能又关键」的诱导状态 —— rollout 深度课程、干净/增强/自生成历史的混合、失败轨迹回放；同时降低长程信用分配成本（段级 rollout、轨迹复用、选择性反传）。
5. **交互与因果记忆** ：把干预关联的转变作为一等公民记忆，记录作用者、受影响实体、前提、后果与作用域；支持受控修订、回滚与分支（反事实生成要求分支间状态隔离）；配套 act-wait-query 协议让「动作-延迟后果」可训可测。
6. **可比较的记忆评估协议** ：从孤立端点分数走向分阶段诊断画像 —— 区分「信息是否被保留」「状态是否正确」「需要时能否取出」「取出后是否真的影响了生成」；并覆盖矛盾处理、选择性遗忘、回滚等「记忆如何变化」的协议。

## 结语

这篇综述最重要的贡献，是把「记忆」从一堆散落在各家论文里的 trick（cache、bank、anchor、state……）提升为一个有统一定义、有生命周期、有训练范式、有评估准则的研究对象。它反复强调的观点值得记住： **有效记忆超越容量本身** —— 保留的状态必须准确、可达、且对后续生成有因果影响力。上下文加长、存储加大、视频质量总分提高，单独都不能证明记忆的存在；真正的问题是：正确的历史信息，是否在需要它的那一刻仍然可用且有效。

> 本文参考自 [The Past Frames the Future: Memory for Autoregressive Video Generation --- A Survey](https://arxiv.org/abs/2609.28466)，项目仓库：https://github.com/HaroldChen19/Awesome-AR-Video-Memory