# GAE：让感知与生成交叉在同一个"几何原生"潜空间里 —— 3D 一致世界生成的新范式

视频生成模型如今可以画出以假乱真的画面，但如果你把这些画面丢给一个 3D 重建模型去"验货"，往往会发现：深度在漂移、相机轨迹跑偏、点云对不上。来自 HKUST、腾讯 ARC Lab、HKU 和 UT Austin 的这篇论文 **GAE（Geometry-native Autoencoder）** 给出了一个很有意思的诊断：这不仅是建模能力的问题，更是一个 **表征（Representation）问题** —— 生成器在一个"以外观为中心"的潜空间里演化，而感知系统却在另一个"几何可读"的空间里工作。两者的空间不一致，一致性自然无从谈起。

GAE 的思路一句话概括： **把几何基础模型（DA3）的多层特征重参数化为一个紧凑的潜空间，让 RGB 外观和 3D 几何成为同一个潜变量的两种解码结果** ，然后用一个标准的 Flow Matching 模型在这个空间里做生成。在严格的控制变量实验中（生成器、训练配方、采样协议全部固定，只换潜空间），GAE 在 RealEstate10K 和 DL3DV 上将 FVD 分别降低了 12.7% 和 23.1%，相机轨迹误差直接减半。

![GAE Teaser](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/figures/gae_teaser.png)

> 图解：GAE 的整体效果展示。给定一张参考图、一条相机轨迹和可选文本，GAE 从 **同一个紧凑潜空间** 中联合生成 RGB 视图序列和 3D 几何。每个例子都配对了生成的 RGB 序列、深度序列，以及由解码几何直接拼出的 3D 点云 —— 注意这些点云没有经过任何外部重建或后处理融合，是生成状态的"原生"几何读出。

## 1. 问题：画面很真，场景却"站不住"

先看一个被论文反复点名的现象：几何基础模型（如 DUSt3R、VGGT、DA3）只需要几张真实视图，就能恢复出在统一 3D 坐标系下自洽的深度、相机位姿和点云。但视频生成模型产出的帧，经过同样的重建流程后，几何会漂移、相机轨迹会偏离指定的路径。

论文把这句话说得很漂亮： **"感知能读出来的东西，生成还写不出来。"** 如果视频生成要成为世界模型的基座，它生成的帧就必须描述一个 **持久的（persistent）场景** ，而不只是一堆各自漂亮的图片。

作者把这个需求一路追溯到上游 —— **生成器被训练去演化的那个潜空间本身** 。潜空间不只是压缩工具，它决定了哪些结构对生成器是"直接可见"的、哪些必须从外观中重新推断：

- **Pixel VAE** （如 SD-VAE）：保留外观，但深度、相机、跨视角关系完全不可读；
- **语义 RAE** （如 RAEv2，基于 DINOv3 特征）：语义邻域组织得好，但同样没有原生的相机/公共坐标系几何读出；
- 常见的补救办法是在外观潜空间旁边"外挂"几何 —— 相机控制、特征对齐、几何奖励后训练等。GAE 的主张恰恰相反： **让感知和生成共享一个几何原生的潜空间** ，把几何做成生成状态本身，而不是事后补丁。

## 2. 为什么几何特征不能"拿来就用"

既然几何基础模型的特征天然带有几何读出能力，直接在这些特征上训扩散模型不就行了？论文用 DA3（Depth Anything 3）作为受控案例，指出了两个现实障碍：

**第一，层级依赖。** 现代前馈式 3D 重建模型通常从多个网络深度的特征层级共同解码几何，而不是靠单一表征。浅层保留细节和跨视角对应，深层提供几何解码器需要的互补输入。这对感知是好的分工，对生成却很尴尬：建模全部四层需要一串层级级联的生成模型（GLD 就是这么做的，用级联 Flow 逐层生成）；只选一层又会丢掉几何解码器所期待的信息。

**第二，原始特征的"病态"统计性质。** 论文的诊断实验发现，DA3 单层特征虽然有 3072 个通道，有效秩（effective rank）却只有约 11，协方差条件数高达 $10^{8}$ 到 $10^{16}$。也就是说绝大多数通道近乎"死亡"，有用信息集中在极少数方向、且尺度差异巨大。几何信息确实在那里，但它的原始参数化方式与生成建模（需要平滑、良态的传输路径）严重不匹配。

这引出 GAE 对"好的生成重参数化"的三条原则：

1. **保留（Preserve）** 感知模型已经编码好的几何信息；
2. **统一（Unify）** 外观与几何于同一个紧凑表征；
3. **组织（Organize）** 这个表征，使其适合平滑的生成式传输。

## 3. 方法：GAE 两阶段框架

![GAE Pipeline](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/blog_figures/ngd_pipeline_overview.png)

> 图解：GAE 的两阶段流程。 **Stage 1（上半部分）** ：冻结的几何基础模型（DA3）输出四层级特征 $\mathbf{F}^{0:3}$，可训练的紧凑几何自编码器 $\mathrm{Enc}_{\phi}$ 将其压缩成紧凑几何潜变量 $\mathbf{z}$；随后分两路解码 —— 一路是学习出来的 RGB Head 直接渲染外观，另一路是特征解码器 $\mathrm{Dec}_{\phi}$ 重建完整特征层级，交给 **冻结的 DPT Head** 读出深度/射线/点云。Teacher Models（C-RADIO 与 DINOv2）通过 $\mathcal{L}_{\mathrm{repr}}$ 塑造潜空间的组织结构。 **Stage 2（下半部分）** ：冻结 codec，在标准化后的潜空间上训练条件 Flow Matching（DiT 架构），高斯噪声 $\bar{\mathbf{z}}$ 在参考图干净潜变量、相机射线、文本提示的控制下演化为生成潜变量 $\hat{\mathbf{z}}$，再一次性解码出多视角 RGB、深度和点云+位姿。

整体流程可以用下面这组式子概括：

$$
\underbrace{\mathbf{I}_{1:V} \xrightarrow{\;\mathcal{E},\,\mathcal{P}\;} \mathbf{X}_{1:V} \xrightarrow{\;\mathrm{Enc}_{\phi}\;} \mathbf{z}_{1:V} \xrightarrow{\;\mathrm{Dec}_{\phi}\;} \hat{\mathbf{X}}_{1:V}}_{\text{Stage 1: 训练 codec } \mathcal{A}_{\phi}},
\qquad
\underbrace{\bar{\mathbf{z}}^{(1)}_{1:V} \xrightarrow{\;\mathrm{Flow}_{\psi}(\cdot\mid\mathcal{C})\;} \hat{\bar{\mathbf{z}}}_{1:V} \xrightarrow{\;\mathrm{Std}^{-1}\;} \hat{\mathbf{z}}_{1:V}}_{\text{Stage 2: 训练流模型 } \mathbf{f}_{\psi}}
$$

其中 $\mathcal{E}$ 是冻结的 DA3 编码器，$\mathcal{P}$ 是固定的层级归一化与融合算子。Stage 2 从标准高斯噪声出发做 ODE 采样，条件集 $\mathcal{C}$ 包含文本、相机射线和参考视图的干净潜 token —— **只有目标视图的潜变量作为 ODE 状态演化，其余信号都是固定控制量** 。最终 RGB 和几何从同一个采样潜变量读出：

$$
\hat{\mathbf{I}}_{1:V} = \mathcal{D}_{\theta}^{\mathrm{rgb}}(\hat{\mathbf{z}}_{1:V}),
\qquad
\hat{\mathbf{G}}_{1:V} = \mathcal{H}_{\mathrm{DPT}}\!\left(\mathrm{Dec}_{\phi}(\hat{\mathbf{z}}_{1:V})\right)
$$

### 3.1 Stage 1：几何原生编解码器

#### 从特征层级到单一状态

给定 $V$ 个输入视图，冻结的 DA3 编码器产生四层级特征 $\mathbf{F}_v^{0:3}$，其中第 $\ell$ 层特征 $\mathbf{f}_{v,\ell} \in \mathbb{R}^{N \times C_{\ell}}$ 是视图 $v$ 在该层的 patch token。融合前先用训练集的固定逐通道统计量 $(\mathbf{m}_{\ell}, \mathbf{s}_{\ell})$ 对每层归一化，再把 token 重塑回空间网格、沿通道维拼接：

$$
\mathbf{X}_v = \mathcal{P}(\mathbf{F}_v^{0:3}) := \big\|_{\ell=0}^{3} \ \mathrm{reshape}\!\left(\frac{\mathbf{f}_{v,\ell}-\mathbf{m}_{\ell}}{\mathbf{s}_{\ell}+\epsilon}\right) \in \mathbb{R}^{C_x \times h_p \times w_p},
\qquad C_x = \sum_{\ell=0}^{3} C_{\ell}
$$

层级归一化这一步很关键：它防止高方差通道主导融合表征（回想一下原始特征 $10^{16}$ 量级的条件数）。

#### 紧凑几何 codec

Codec $\mathcal{A}_{\phi} = (\mathrm{Enc}_{\phi}, \mathrm{Dec}_{\phi})$ 把融合张量压缩成网格状潜变量 $\mathbf{z}_v \in \mathbb{R}^{C_z \times h_p \times w_p}$，并重建完整的四层特征层级：

$$
(\boldsymbol{\mu}_v, \log \boldsymbol{\sigma}_v^2) = \mathrm{Enc}_{\phi}(\mathbf{X}_v),
\qquad
\mathbf{z}_v = \boldsymbol{\mu}_v + \boldsymbol{\epsilon}_v \odot \exp\!\left(\tfrac{1}{2}\log \boldsymbol{\sigma}_v^2\right),
\qquad
\hat{\mathbf{F}}_v^{0:3} = \mathrm{Dec}_{\phi}(\mathbf{z}_v)
$$

注意 GAE **保留了 DA3 的 patch 网格，只压缩通道维** ：$C_z \in \{64, 128\}$，相比原始单层 3072 通道压缩了 24 倍以上。编解码器采用带空间自注意力的轻量卷积金字塔，既保留 patch 布局又允许长程混合。训练 codec 时采样 $\mathbf{z}_v$，而 Flow 训练与推理时确定性地使用后验均值 $\boldsymbol{\mu}_v$；一个很小的 KL 权重只起温和正则作用，不把瓶颈逼向强正则化的生成式 VAE 区域。

#### RGB 与几何联合解码：冻结 Head 的"审计"作用

重建的特征层级交给 **原始的、冻结的** DA3 DPT Head 读出深度、射线和点云，另有一个独立学习的 RGB Head 从同一潜变量渲染外观：

$$
\hat{\mathbf{I}}_v = \mathcal{D}^{\mathrm{rgb}}_{\theta}(\mathbf{z}_v),
\qquad
\hat{\mathbf{G}}_v = \mathcal{H}_{\mathrm{DPT}}\!\left(\mathrm{Dec}_{\phi}(\mathbf{z}_v)\right)
$$

这里最值得玩味的设计是 **保持 $\mathcal{H}_{\mathrm{DPT}}$ 冻结** 。为什么？因为冻结的几何 Head 相当于一个"审计员"：它不会去适应 codec 丢失的信息，任何几何信息的损失都会直接暴露在重建误差里。这保证了潜变量中的几何始终可通过 DA3 的原生接口读出。相比之下，原始单层特征的 baseline 还需要把生成状态重新喂回 backbone 传播才能补齐层级，GAE 则一步到位重建整个层级。

Codec 的总目标函数为：

$$
\mathcal{L}_{\mathrm{codec}} = \mathcal{L}_{\mathrm{feat}} + \lambda_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}} + \mathcal{L}_{\mathrm{rgb}} + \mathcal{L}_{\mathrm{geo}} + \mathcal{L}_{\mathrm{repr}}
$$

其中 $\mathcal{L}_{\mathrm{feat}}$ 是逐层特征重建的加权和，$\mathcal{L}_{\mathrm{rgb}}$ 组合像素与感知重建项，$\mathcal{L}_{\mathrm{geo}}$ 监督深度与射线（注意：监督目标是冻结 DA3 Head 作用于原始特征得到的 **伪标签** ，而非数据集真值深度），最后一项 $\mathcal{L}_{\mathrm{repr}}$ 负责塑造潜空间的内部组织方式，是下一节的主角。

### 3.2 为生成"塑形"潜空间：单靠重建远远不够

重建决定了瓶颈 **必须保留什么** ，但管不了信息 **如何组织** 。一个只做重建的 codec 可以完美恢复 RGB 和几何，却产出一个 Flow 模型极难学习的潜空间。GAE 用两个互补的表征目标来解决这一点，这也是论文方法部分最有洞察力的贡献。

**第一步：逐 token 语义对齐（$\mathcal{L}_{\mathrm{tok}}$）。** 将后验均值 token 投影后与冻结 C-RADIO 教师模型的同位置特征对齐：

$$
\mathcal{L}_{\mathrm{tok}} = \frac{1}{VN}\sum_{v,i} \left[1 - \left\langle \widehat{g_{\eta}(\boldsymbol{\mu}_{v,i})},\ \hat{\mathbf{c}}_{v,i} \right\rangle\right]
$$

其中 $g_{\eta}$ 把后验 token 映射到 C-RADIO 特征维度，帽子表示 $\ell_2$ 归一化。这一步改善了传输平滑度（$\rho$）和语义邻域一致性（LNC）。

**但它有个隐蔽的副作用** ：逐位置独立对齐会摧毁 token 之间的相对关系 —— 诊断指标 LDS、SRSS（刻画空间结构）急剧下降，跨视角对应也略微变弱。单独对齐每个 token，却把它们"队伍打散"了。

**第二步：关系结构蒸馏（$\mathcal{L}_{\mathrm{struct}}$）。** 用冻结的 DINOv2 教师提供空间位置间的两两相似度，直接在原始后验空间中匹配这些相似度：

$$
\mathcal{L}_{\mathrm{struct}} = \frac{1}{VN(N-1)} \sum_v \sum_{i \ne j} \left( \left\langle \hat{\boldsymbol{\mu}}_{v,i}, \hat{\boldsymbol{\mu}}_{v,j} \right\rangle - \left\langle \hat{\mathbf{d}}_{v,i}, \hat{\mathbf{d}}_{v,j} \right\rangle \right)^2
$$

这种做法不要求学生与教师通道维度一致，恰好约束了逐位置对齐所遗漏的邻域几何。两项合起来：

$$
\mathcal{L}_{\mathrm{repr}} = \lambda_{\mathrm{repa}} \left( \mathcal{L}_{\mathrm{tok}} + \lambda_{\mu} \mathcal{L}_{\mathrm{struct}} \right),
\qquad \lambda_{\mathrm{repa}} = 0.25,\quad \lambda_{\mu} = 8.0
$$

与 REPA（监督去噪器的中间表征）不同，GAE 的两个表征项都是在 **生成训练之前** 塑造 codec 潜空间本身；训练结束后教师模型和投影头直接丢弃。

这个"三步走"在 DA3-GIANT 上的诊断结果非常说明问题（$\rho$、$\kappa$ 越低越好，其余越高越好）：

| 表征目标 | $\rho$ (单视角) | $\kappa$ (单视角) | effrank | LDS | SRSS | xLNC$^{*}$ |
|---|---|---|---|---|---|---|
| 原始 DA3 L0 | 0.876 | $2.8\times10^{8}$ | 11.3 | 0.102 | 0.123 | 0.642 |
| 原始 DA3 L3 | 0.769 | $6.6\times10^{16}$ | 11.5 | 0.054 | 0.070 | 0.380 |
| 仅重建（无 $\mathcal{L}_{\mathrm{repr}}$） | 0.820 | **130** | **58.0** | 0.240 | 0.300 | 0.590 |
| + $\mathcal{L}_{\mathrm{tok}}$ | 0.700 | 170 | 53.0 | 0.020 | 0.030 | 0.580 |
| + $\mathcal{L}_{\mathrm{tok}}$ + $\mathcal{L}_{\mathrm{struct}}$ | **0.674** | 227 | 51.3 | **0.444** | **0.553** | **0.591** |

> 表解：融合+重建先把条件数从天文数字压到几百（良态化），但传输平滑度和语义邻域仍弱；加上 $\mathcal{L}_{\mathrm{tok}}$ 后 $\rho$ 改善，LDS/SRSS 却崩了；补上 $\mathcal{L}_{\mathrm{struct}}$ 后空间与跨视角结构恢复，同时保住了前两步的收益。xLNC$^{*}$ 是论文提出的 LNC 跨视角扩展：用稠密对应定义跨视角正样本，做机会水平归一化后的检索准确率。

训练完成后冻结 codec，取后验均值作为潜变量，并用训练集统计做逐通道标准化：

$$
\bar{\mathbf{z}}_v = \frac{\mathbf{z}_v - \mathbf{m}_z}{\mathbf{s}_z + \epsilon}
$$

### 3.3 Stage 2：条件 Flow Matching 与生成

冻结 codec 后，只在标准化后验均值上建模。给定目标潜变量 $\bar{\mathbf{z}}^{\mathrm{gt}}$ 和噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，采用线性插值路径 $\bar{\mathbf{z}}_t = (1-t)\bar{\mathbf{z}}^{\mathrm{gt}} + t\boldsymbol{\epsilon}$，目标速度为 $\mathbf{u}_t = \boldsymbol{\epsilon} - \bar{\mathbf{z}}^{\mathrm{gt}}$，训练条件 Transformer 最小化：

$$
\mathcal{L}_{\mathrm{flow}} = \mathbb{E}\!\left[ \rho(t) \left\| \hat{\mathbf{u}}_{\psi}(\bar{\mathbf{z}}_t, t, \mathcal{C}) - (\boldsymbol{\epsilon} - \bar{\mathbf{z}}^{\mathrm{gt}}) \right\|_2^2 \right]
$$

跟随 RAEv2 的做法，网络实际预测干净潜变量 $\hat{\bar{\mathbf{z}}}^{\mathrm{gt}} = \mathbf{f}_{\psi}(\bar{\mathbf{z}}_t, t, \mathcal{C})$，再换算成速度：

$$
\hat{\mathbf{u}}_t = \frac{\bar{\mathbf{z}}_t - \hat{\bar{\mathbf{z}}}^{\mathrm{gt}}}{\max(t, t_{\epsilon})},
\qquad t_{\epsilon} = 0.05
$$

单个 Transformer 联合建模所有目标视图，让跨视角交互直接发生在紧凑的几何原生状态里。

#### 条件设计：一切必须与推理时对齐

这一节体现了作者对 train/inference mismatch 的执念。核心原则是： **每个条件信号都必须用推理时真实可得的信息构造，且与被 Flow 演化的变量严格分离** 。

- **参考 token** ：几何编码器是集合条件的 —— 同一张参考图，与目标视图一起编码和单独编码得到的特征是不同的。如果把"全集编码"的参考潜变量放进训练，测试时根本构造不出来。GAE 的做法是：仅用 $K$ 张观测参考图自身作为上下文联合编码，得到干净潜 token $\bar{\mathbf{z}}^{\mathrm{ref}}$，赋时间步 $t=0$，前置拼接到噪声 token 序列里参与自注意力，但在预测头之前移除。这样所有 $V$ 个输出槽位（包括位于参考相机处的）都是普通的噪声流变量，无需在 ODE 中反复 clamp 参考槽，一个标准 Euler 采样器通吃条件/无条件采样。
- **相机射线** ：为每个潜格子构造世界坐标系下的 Plücker 射线 $(\mathbf{d}, \mathbf{m})$（$\mathbf{m} = \mathbf{o} \times \mathbf{d}$），并分解为方向 $\mathbf{d}$、归一化矩方向 $\hat{\mathbf{m}}$ 和对数尺度 $s = \log(\|\mathbf{m}\| + \epsilon)$。沿用 SCoPE 的做法，该嵌入调制 query/key 自注意力，让 token 两两交互显式依赖相机射线，而不是把位姿当成一个全局向量。对数尺度通道在有位姿元数据时保留了度量化的平移幅度。
- **文本** ：由冻结语言模型（Qwen3-0.6B）编码，经 cross-attention 提供全局语义上下文。

文本、相机、参考三类条件以概率 0.10 / 0.05 / 0.05（最终模型为 0.10 / 0.20 / 0.20）独立 dropout，于是 **同一套权重** 就能覆盖纯文本生成、相机控制视频、参考图条件的新视角合成等模式，dropout 分支还顺便提供了 classifier-free guidance（CFG）所需的无条件预测。

#### 训练与推理

Stage 1 在单视角+多视角混合数据上联合训练 codec 与 RGB 解码器（DA3 编码器与 DPT Head 全程冻结）；Stage 2 冻结全部 Stage 1 模块，在 T2I 与视角条件生成上训练 DiT 流模型（28 个宽度 768 的编码器 block + 6 个宽度 2048 的解码器 block，约 0.93B 参数）。推理时从 $t=1$ 积分到 $t=0$，反标准化后一次性解码 RGB 与几何 —— 没有层级级联，也没有外接的几何估计器。

## 4. 实验：只换潜空间，其余全固定

实验设计是本文的另一大亮点： **所有受控变体使用同一个 Flow 架构、同一份训练数据与预算、同一套条件与采样协议（9 视角 $252\times252$、1 张参考图、50 步 Euler、CFG=2，RealEstate10K 与 DL3DV 各 64 个留出场景），唯一改变的是潜空间** 。对比对象包括：两个 Pixel VAE（SD-VAE、WAN2.1 VAE）、语义 RAE（官方 RAEv2 + DINOv3-L）、原始几何特征（DA3-GIANT 的 L0 与 L3 层），以及 GAE-64 / GAE-128。灰色行的 GLD 与 Gen3R 是外部完整系统，只作参考不参与排名。

### 4.1 潜空间诊断

![Latent Quality Radar](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/blog_figures/output_quality_radar.png)

> 图解：潜表征质量（绿）、匹配 DiT 下的 RGB 生成质量（蓝）、3D 一致性（橙）的归一化雷达图，越靠外越好。可以看到没有任何 baseline 在所有轴上都强：Pixel VAE 良态但结构弱，语义 RAEv2 语义强但无几何读出，原始 DA3 特征几何原生但高维且病态；GAE 取得了最均衡的外轮廓。

关键数字：原始 DA3 单层 3072 通道、有效秩仅约 10、条件数 $10^{8}\sim10^{16}$；GAE-64/128 把通道压到 64/128，条件数降到 53~356，同时 $\rho$（传输模糊度）在所有几何原生表征中最低，LDS/SRSS（空间结构）大幅领先，xLNC$^{*}$ 0.591 接近原始 L0 的 0.642 —— 用 24 分之一的通道数保住了绝大部分跨视角对应。

### 4.2 Codec 重建质量

**RGB 重建** （PSNR/LPIPS 越高/低越好，rFID/rFVD 越低越好）：

| 潜空间 | 单视角 PSNR | 单视角 LPIPS | 多视角 PSNR | 多视角 rFVD |
|---|---|---|---|---|
| SD-VAE | 18.49 | 0.140 | 19.97 | 37.7 |
| WAN2.1 VAE | 18.34 | 0.125 | 19.83 | 46.7 |
| RAEv2 | 22.08 | 0.134 | 24.73 | 32.1 |
| 原始 DA3 L0 | 28.64 | 0.038 | 34.70 | 4.4 |
| **GAE-128** | **28.76** | **0.036** | **34.78** | **4.3** |
| GAE-64 | 27.30 | 0.061 | 32.37 | 9.1 |

GAE-128 在通道数少 24 倍的情况下，重建质量追平甚至略微超过最强的原始 L0 特征 —— 说明压缩没有丢掉外观细节。

**几何重建** （编码真实帧再解码几何，与 Pi3 在真实帧上的结果及真实轨迹对比）：GAE 两个变体在深度、点云、位姿指标上全面优于"L0 回传 backbone"的 baseline；ATE 低至 0.006~0.008（RealEstate10K），甚至低于 Pi3 在真实帧上的参考值 0.009，也显著优于外部系统 Gen3R（0.038）和 GLD（0.039）。

### 4.3 RGB 生成质量

| 潜空间 | RE10K FVD | RE10K PSNR | RE10K SSIM | DL3DV FVD | DL3DV PSNR |
|---|---|---|---|---|---|
| SD-VAE | 258.6 | 19.32 | 0.666 | 373.2 | 17.05 |
| WAN2.1 VAE | 362.9 | 16.52 | 0.575 | 596.7 | 14.31 |
| RAEv2 | 379.4 | 17.60 | 0.585 | 453.4 | 16.01 |
| 原始 DA3 L0 | 298.6 | 18.31 | 0.643 | 376.5 | 16.68 |
| 原始 DA3 L3 | 488.9 | 16.82 | 0.579 | 584.8 | 16.22 |
| GAE-128 | 233.4 | 19.54 | 0.701 | 345.2 | 17.39 |
| **GAE-64** | **225.7** | **20.02** | **0.711** | **287.0** | **18.00** |
| （参考）GLD | 445.1 | 14.91 | 0.518 | 587.4 | 14.51 |
| （参考）Gen3R | 269.7 | 17.79 | 0.621 | 580.5 | 15.62 |

GAE-64 相对最强的非 GAE 受控潜空间，FVD 在 RealEstate10K 降 12.7%、DL3DV 降 23.1%，且配对指标（LPIPS/PSNR/SSIM）全面领先。

![RGB Comparison](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/blog_figures/qual_rgb_re10k_compact.png)

> 图解：RealEstate10K 上两个场景的定性对比。每列是一种潜空间（含 GT 和外部系统 GLD/Gen3R），每行依次为参考帧、第 5 帧、第 9 帧目标视角，右下角数字是该场景两个目标视角的 PSNR。随着视角远离参考图，其它方法逐渐丢失物体身份与细节结构（如柜门、沙发的形态漂移），GAE-64 保持得最好，两个场景均取得最高 PSNR（18.0 / 18.2）。

### 4.4 3D 一致性：独立重建来"验货"

这一组实验完全绕开 GAE 自己的几何读出：用 **独立的 VGGT** 从生成的 RGB 序列重建相机与点云，再做 Sim(3) 对齐后计算 ATE（轨迹误差）、RPEt/RPEr（相对平移/旋转误差）和重投影误差；MEt3R 则直接度量视角对之间的对称特征不一致。

在 RealEstate10K 上，GAE-64 的 VGGT ATE 为 0.0034 —— 相比次优受控潜空间 SD-VAE 的 0.0072 **降低了 52.8%（近乎减半）** ；DL3DV 上降低 23.3%。MEt3R 方面 GAE-64 与 GAE-128 分别以 0.1208（RE10K）和 0.1347（DL3DV）拿下受控最优。重投影误差在两个数据集上同样最低（0.0068 / 0.0077）。外观收益并没有以牺牲相机控制或多视角一致性为代价。

![Independent 3D Check](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/blog_figures/compare.png)

> 图解：同一张参考图、同一条相机轨迹下三类潜空间的对比（上：外观原生 WAN2.1 VAE；中：语义原生 RAEv2；下：几何原生 GAE）。左侧为生成的首末帧及细节放大 —— 注意玻璃门上白色窗棂网格，前两种方法已经模糊错位，GAE 依然清晰。中间是独立 VGGT 从生成视频重建的点云，GAE 的窗棂结构锐利对齐。右侧是恢复的相机轨迹（虚线为目标轨迹，实线为恢复轨迹），GAE 的轨迹与指定路径几乎重合，而 WAN2.1 VAE 明显跑偏。

### 4.5 几何生成：从采样潜变量直接解码 3D

直接对 **采样得到的潜变量** 解码深度、射线与点云，与 Pi3（作用于真实帧）和真实轨迹对比：DL3DV 上 GAE-64 包揽全部六项几何指标最优；RealEstate10K 上 GAE-64 深度与位姿最优，GAE-128 的 Chamfer 与点云误差最低。由于参照系来自 Pi3 和数据集相机，这些收益不依赖 DA3 当"裁判"。

![Geometry Comparison](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/blog_figures/compare2.png)

> 图解：与外部系统 Gen3R、GLD 的多帧对比（RealEstate10K 两个场景）。每个场景展示早/中/晚期生成视图、生成的点云和恢复的位姿轨迹。GAE 紧贴指定相机轨迹、场景结构完整；Gen3R 与 GLD 则出现明显的位姿偏移和点云扭曲（如房间结构的塌陷与重影）。

### 4.6 长序列与文本生成能力展示

受控对比之外，论文还单独训练了一个最终模型（DA3-GIANT 时空 codec、$672\times378$ 分辨率、81 视角训练、多域数据混合、80 卡训练），用于能力展示：

![Long Rollout](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/blog_figures/showcase_geoft.png)

> 图解：81 视角长序列生成的展示（每行展示 5 帧），右侧是从同一批潜变量解码出的点云。注意点云的构造方式极其"朴素"：把 81 帧深度图按解码射线恢复的相机直接反投影拼接， **没有任何跨视角融合、外部重建或测试时优化** 。卧室、工作台、广场、车载场景的点云都保持了连贯的场景结构，这是几何原生状态威力的直观体现。

![T2I Samples](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-a-Geometry-Native-Latent-Space-for-3D-Consistent-World-Generation/scripts/figures/t2i_rgb_depth_caption_showcase_first2rows.jpg)

> 图解：同一个 Flow checkpoint 直接做文本条件生成。每对图像展示生成的 RGB 与从同一潜变量解码的深度。作者强调这只是能力验证而非 T2I 基准测试 —— 重点在于：文本→图像→几何，走的始终是同一条潜空间通道。

### 4.7 消融实验

在 RealEstate10K 上的消融（固定 Flow 架构与配方）验证了每个设计决策：

- **表征目标** ：在 DA3-GIANT 上加 $\mathcal{L}_{\mathrm{repr}}$，FVD 从 266.4 降到 233.4，VGGT ATE 从 0.0044 降到 0.0041；同宽度同目标下 GIANT backbone 优于 LARGE（外观与位姿更好），LARGE 仅在 MEt3R 上略占优 —— 表征塑形与 backbone 容量提供的是互补收益。
- **参考条件** ：把参考潜变量放进 ODE 状态（而非干净条件 token），参考-目标点云差距从 0.0318 恶化到 0.1016，FVD 从 225.7 升到 257.1 —— 证实了"推理对齐"条件设计的必要性。
- **位姿尺度** ：用每场景归一化位姿替代度量位姿，SE(3) ATE 从 0.0093 恶化到 0.0346，FVD 从 225.7 暴涨到 350.7 —— 度量化的 Plücker 射线保留了被归一化丢弃的平移尺度。
- **T2I 联合训练** ：去掉 T2I 共训后 FVD 从 225.7 恶化到 472.9 —— 单视角数据提供了更广阔的外观先验。

## 5. 与相关工作的位置关系

- **生成式表征空间** ：从 VAE/VQ-VAE 到 Latent Diffusion，紧凑连续码已是标准生成状态；RAE 系列改用冻结的预训练视觉表征作编码状态；REPA 对齐的是去噪器内部特征而非输出状态。这些工作都围绕外观或语义组织生成，GAE 则要求紧凑状态 **原生可被一个几何基础模型读取** 。
- **几何感知生成** ：相机控制视频、RGB+深度联合输出、几何特征对齐/奖励后训练等，本质上都是在外观主导的状态旁边"加"几何；GAE 反过来，让唯一的演化潜变量 **源自** 几何基础模型。
- **几何基础模型作生成状态** ：最接近的前作是 GLD（级联生成 DA3/VGGT 选定层级）和黎曼 Flow Matching（在乘积流形上联合演化 VGGT 四个归一化层级）。两者保留了几何读出，但都直接生成高维 backbone 特征；GAE 学习的是整个层级的 **单一紧凑欧氏重参数化** ，用标准 Flow 演化，再重建层级供冻结几何读出与学习的 RGB 解码。

## 6. 总结与思考

GAE 的贡献可以归纳为一条主线： **把几何从生成的"输出"或"条件"，提升为生成发生的"状态"本身** 。具体机制上有三个值得记住的设计：

1. **冻结 Head 审计** ：在冻结的几何编码器与冻结的几何解码头之间学习瓶颈，几何信息保留与否全程可观测；
2. **token 对齐 + 关系蒸馏的双层塑形** ：先发现逐 token 对齐会摧毁两两空间结构，再用 DINOv2 的成对相似度在原始后验空间中补回来 —— 这个"发现问题再针对性修复"的过程本身就很有方法论价值；
3. **推理对齐的条件接口** ：干净参考 token 置于 ODE 状态之外、度量 Plücker 射线调制自注意力、条件 dropout 一统多种生成模式。

作者也很诚实地划定了边界：潜变量仍是逐视图的（不是场景级潜变量）；几何进入状态并不意味着 3D 一致性"免费" —— 联合分布仍要生成器自己去学，几何只是变得显式、可直接监督、可原生解码；原始特征的病态谱是匹配单 Flow 配方下观察到的现象，而非"不可能定理"。

从更大的视角看，这项控制变量扎实的研究把一个容易被工程细节淹没的观点立住了： **潜空间的选择是几何一致性生成的第一等设计轴** 。当感知与生成共享同一个空间时，"感知能读的，生成也能写"这句话才算真正落地。

> 本文参考自 [Learning a Geometry-Native Latent Space for 3D-Consistent World Generation](https://arxiv.org/abs/2609.24981)