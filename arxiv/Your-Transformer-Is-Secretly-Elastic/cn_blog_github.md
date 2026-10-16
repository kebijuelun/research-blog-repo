# Your Transformer Is Secretly Elastic：无需重训，一个模型覆盖全部算力预算

这篇论文解决的是一个非常实际的问题：模型厂商只发布 S/B/L 等少数几个固定尺寸，但线上流量是波动的——高峰期希望模型快一点，空闲时又希望它准一点，固定尺寸永远不是最优解。作者的思路一句话概括： **把"剪枝"从硬裁剪松弛为可微分的软掩码，用梯度下降联合优化每一个 MLP 神经元和每一个 Attention Head 的重要性分数，且全程冻结模型权重、不做任何重训** 。最终得到的单一排序可以裁剪到任意稀疏度。最硬的结果：把 DINOv3 ViT-H+/16 剪掉 50%（删去 427M 参数），ImageNet-1k 线性探测精度只掉 0.8%；AugReg ViT-B/16 在 60% 稀疏度下吞吐提升至 1.80 倍，且整个弹性模型的构建只需要几分钟。

## 一、背景：为什么"固定尺寸"是个坏设计

预训练配方不断进步，模型已经膨胀到万亿参数级别，部署场景却千差万别：有的是离线批处理，有的是延迟敏感的单条请求。但模型家族只给几个固定档位，用户只能按硬件和延迟约束"押注"一个尺寸。流量波动时，选大了浪费，选小了掉点。

已有的出路各有硬伤：

- **结构化剪枝** ：每个剪出来的模型绑定一个固定预算。要服务 $k$ 个预算，就要生成、存储、切换 $k$ 份模型；检索场景下不同剪枝模型的 embedding 互不兼容，还得建 $k$ 份索引。
- **Elastic inference（弹性推理）** ：让单个模型在推理时自适应算力预算。但现有方法要么需要特定的嵌套架构并从头训练（如 MatFormer），要么需要大规模微调（如 SortedNet、Flextron）。对十亿参数级模型，微调成本高昂；更糟糕的是，模型的预训练数据分布往往未知，连"用什么数据微调"都很难选。

作者的直接前作 SnapViT 已经证明：预训练 ViT 可以 **不重训地弹性化** ——通过进化搜索为每个 MLP block 和每个 Attention Head 各学一个缩放因子，重排全局重要性。但为了控制黑盒搜索的维度，SnapViT **保留了每个 block 内部的神经元初始排序** ：如果一个重要的神经元初始分数低，想保住它就必须同时保住排在它前面的所有神经元。

这篇论文的核心贡献，就是把"每个 block / head 一个缩放因子"升级为" **每一个神经元、每一个 head 一个独立分数** "，并用可微分优化取代进化搜索，把被掩盖的"隐藏弹性"（hidden elasticity）释放出来。

## 二、方法：可微分的软剪枝（Differentiable Soft Pruning）

整体目标：把一个预训练 Transformer 变成一个弹性模型——推理时可以裁到任意算力预算，不需要重训。做法是排序网络的功能单元（MLP 神经元和 Attention Head），裁剪时从排序尾部依次删除，直到达到目标稀疏度。这与常规剪枝（锁定单一稀疏度）的本质区别在于： **一个排序天然实现所有稀疏度** 。

![主图：可微分软剪枝流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/main-figure.png)

> 图解：方法总览。初始重要性分数先被转换为可学习的 log-score，并按单元类型做中位数居中，对齐两者的参考点（不改变类型内部排序）。每一步优化中，联合排序为每个稀疏度目标 $b$ 确定阈值 $\kappa_b$ 和软掩码 $M_b$；同一批校准数据在冻结模型上分别按各掩码前向传播，损失跨目标平均后只更新 log-score。掩码梯度集中在各阈值附近，因此多个目标能把排序的不同区段暴露给有效梯度。优化完成后，排序可直接硬剪枝到任意推理预算，无需再优化。

### 2.1 初始重要性分数：Hessian 对角近似

单元排序需要逐单元的重要性分数，作者沿用经典做法：参数 $\theta_i$ 的重要性定义为移除它导致的损失增量，对收敛模型做二阶泰勒展开后归结为 Hessian。用对角近似并在 $N$ 个校准样本上以经验 Fisher 估计：

$$
\mathcal{H}_i \approx \frac{1}{N} \sum^{N}_{k=1} \left( \frac{\partial \mathcal{L}_k}{\partial \theta_i} \right)^2
$$

这里的 per-sample 损失 $\mathcal{L}_k$ 按模态选择：视觉编码器用自监督 DINO 目标（对监督和自监督模型都能产出梯度），文本编码器用对称化的 batch 内 InfoNCE 目标。

得到的是逐权重分数，而剪枝以整个单元为粒度，于是做聚合：MLP 神经元对其输入投影列与输出投影行的分数取平均；Attention Head 则对 value 投影切片取平均，得到初始分数 $s^m$ 和 $s^h$。

但这套初始分数是"弱基础"，有三个缺陷：

1. 对角近似忽略了单元之间的交互；
2. 代理损失未必与下游性能对齐；
3. 分数在不同 block、不同单元类型之间不可比。

这正是后面排序优化要逐一修复的问题。

### 2.2 分数参数化：把所有单元放到同一杆秤上

为 MLP 神经元和 Attention Head 分别定义可学习的 log-score $z^m$ 和 $z^h$：log 空间允许分数自由变化，同时指数化后恒为正。初始化时按类型做一次 **中位数居中** ：

$$
z^m = \log(s^m) - \mathrm{median}(\log(s^m)), \quad z^h = \log(s^h) - \mathrm{median}(\log(s^h))
$$

这一步把两类分数的中位数对齐到零，既不改变各自分布的离散程度，也不改变类型内部排序——只是把两类单元放到了同一个参考系上，让跨类型联合排序有意义。用中位数而非均值，是因为中位数对离群值更鲁棒。之后联合优化的 $z$ 诱导出覆盖全部单元的单一排序。

### 2.3 软掩码：把硬截断变成 Sigmoid

对稀疏度目标 $b$，硬剪枝是从排序顶端往下走，保留单元直到累计 FLOPs 达到稠密模型的 $(1-b)$。记最后被保留单元的分数为阈值 $\kappa_b$（以 stop-gradient 保持为常数）。关键一步是把硬选择 $\mathbf{1}[z \geq \kappa_b]$ 松弛为温度 $\tau$ 的 sigmoid 软掩码：

$$
M^m_b = \sigma \left( \frac{z^m - \kappa_b}{\tau} \right) \in (0, 1)^{L \times H}, \quad
M^h_b = \sigma \left( \frac{z^h - \kappa_b}{\tau} \right) \in (0, 1)^{L \times A}
$$

高于阈值的分数掩码值趋近 1，低于阈值的趋近 0；$\tau$ 控制过渡的锐度。每步优化都会重算 $\kappa_b$，使截断点随排序变化始终对齐目标稀疏度。软掩码在前向中缩放激活：

$$
\mathrm{MLP}_l = (\phi(xW_1) \odot M_{b,l}^m)W_2, \quad
\mathrm{Attn}_l = \left( \mathrm{Cat}\left( \mathrm{head}_1(x), ..., \mathrm{head}_A(x) \right) \odot M_{b,l}^h \right)W_o
$$

其中 $l$ 为 block 索引，每个 head 的掩码缩放其全部输出通道。

> 这个设计的聪明之处在于：它借用了可微分剪枝的经典工具（连续掩码），但用法完全不同——以往方法（L0 正则、Sheared LLaMA 等）是 **为单一稀疏度目标学一个掩码、且同时更新模型权重** ；本文是 **为所有稀疏度目标学一个排序、且权重全程冻结** 。前者产出"一个稀疏网络"，后者产出"一整个模型家族"。

### 2.4 优化目标：多目标联合，一次到位

弹性模型要覆盖很宽的稀疏度范围，因此目标函数是在一组稀疏度目标 $\mathcal{B}$ 上最小化掩码模型的损失：

$$
z^* = \arg\min_z \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \mathbb{E}_{(x,y) \sim D} \left[ \ell \left( f\left(x, M_b(z)\right), y \right) \right]
$$

$f(x, M_b(z))$ 是施加了目标 $b$ 掩码的网络，$D$ 为校准集。损失 $\ell$ 优先用模型原生目标（分类器用交叉熵），否则用软剪枝 embedding 与稠密 embedding 的余弦距离。掩码模型的损失反映了单元在剪枝下如何协同工作，捕捉了 Hessian 对角初始化忽略的交互；而逐单元独立分数给了排序完全的重排自由。

**优化细节** ：每步在 14 个稀疏度目标（5% 到 70%，步长 5%）上用同一批校准数据计算损失并平均。这里有个微妙但关键的观察：对单个目标而言，远离阈值的单元其 sigmoid 已饱和（掩码接近 0 或 1），几乎收不到梯度；只有阈值附近的单元对分数变化敏感。多个目标的阈值落在排序的不同位置， **一个单元对某个目标可能饱和，对另一个目标却恰好处于梯度敏感区** ——这就是"多目标联合优化优于单目标分别优化"的机理。最终 $s^* = \exp(z^*)$ 定义推理时的排序。与逐层均匀剪枝（SparseGPT）、局部搜索（FPTP）或分类型各自排序（SnapViT）不同，这里 **稀疏度往哪里分配完全由优化自己决定** 。

## 三、实验：27 个编码器上的全面验证

实验覆盖 27 个视觉与文本 Transformer 编码器（22M 到 8B 参数），任务包括图像分类、语义分割、文本 embedding 与检索。默认评估 10%–60% 六个稀疏度档位，以理论 GFLOPs 计量算力；为隔离"选哪些单元"的纯效果，主实验中所有剪枝方法都不做权重修正。

### 3.1 与剪枝基线的对比：深剪枝区间拉开差距

对比的 one-shot 基线：FPTP、SparseGPT、OPTIN、VBP、SnapViT（视觉）；SnapViT、SlimGPT（文本）。

![结果总览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/results-overview.png)

> 图解：横轴为计算量（GFLOPs），纵轴为性能，曲线标签为 FLOP 稀疏度。上排为五个 ViT-B/16 骨干在 ImageNet-1k 上的 k-NN top-1 精度；左下为 Pascal VOC 2012 上 Hummingbird 协议的 mIoU；右下为文本编码器的 MTEB 平均分。浅剪枝区间各基线尚有竞争力，但越过 30% 后急剧退化，而本文方法平滑下降。

**图像分类** ：30% 以内 SnapViT 在 DeiT/DeiT-III 上与本文差距不到 0.9%；但越过 30% 后所有基线陡降。50% 稀疏度时本文领先亚军 9.6%（DeiT）到 40.3%（DINOv3）；60% 时领先 15.3%（DeiT-III）到 28.4%（DINOv3）。差距在自监督和图文骨干上最夸张：40% 稀疏度下，本文在 DINOv3 上保住 69.4%、SigLIPv2 上 61.1%，而所有基线都跌破 30%。

**语义分割** （Pascal VOC 2012，Hummingbird 检索式协议）：AugReg 上本文从 30% 起领先，50% 时超出亚军 OPTIN 达 25.1% mIoU；DINOv3 上每个稀疏度都领先，30% 时仅掉 3.8%（SnapViT 掉 22.4%）。不过 40% 之后 DINOv3 上本文也会快速退化（50% 时掉 34.8%）。

**文本 embedding** （MTEB 41 任务均分）：本文在 BGE-L、E5-L、Qwen3 0.6B 上所有稀疏度都优于 SlimGPT，60% 时在 Qwen3 0.6B 上领先 38.0 分。SnapViT 浅剪枝时基本持平（最多反超 1.0 分），深剪枝时崩盘：60% 时 E5-L 用本文保住 57.3 分，SnapViT 只剩 30.2 分。双向编码器明显更耐剪：60% 时 E5-L 只掉 5.1 分，Qwen3 0.6B 掉 18.0 分。

### 3.2 规模效应：越大的模型越"弹性"

![规模效应](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/results-scaling.png)

> 图解：同族不同尺寸模型的性能-算力曲线。上排为 DeiT-III（ViT-S/16 到 ViT-H/14）与 DINOv3（ViT-B/16 与 ViT-H+/16）的 ImageNet-1k k-NN 精度；下排为 BGE 与 Qwen3 家族的 MTEB 均分。视觉模型随规模增大退化曲线明显变平；文本上规模的作用因家族而异，但本文相对基线的优势在两个家族中都随规模扩大。

**视觉** ：50% 稀疏度下，DeiT-III ViT-S/16 掉 33.2%，ViT-H/14 只掉 5.3%；DINOv3 ViT-B/16 掉 26.1%，ViT-H+/16 只掉 1.8%（且 40% 以内与稠密模型差距不到 0.4%）。60% 稀疏度下 DINOv3 ViT-H+/16 仍保有 79.4%，而所有基线跌破 20%。更有意思的是 DINO WebSSL 家族的发现： **剪枝后的大模型可以 Pareto 优于更小档位的稠密模型** ——剪枝的 3B、7B 分别 dominate 稠密的 2B、5B（见附录图）。

**文本** ：规模不如家族重要。60% 稀疏度时 BGE 的损失从 S 到 L 仅从 10.0 分收窄到 8.7 分；Qwen3 则是 0.6B 掉 18.0、4B 掉 19.5、8B 掉 15.7，无一致趋势。但本文对 SnapViT 的领先优势随规模稳定扩大：BGE 上从 2.5 分增至 13.3 分，Qwen3 上 4B/8B 分别领先 8.3 和 16.4 分。

### 3.3 弹性部署：索引建一次，查询弹性算

真实部署中更实用的场景是 **非对称编码** ：文档索引离线用稠密大模型建一次，在线查询根据当前负载弹性选择算力。

![弹性部署](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/results-elastic.png)

> 图解：索引由稠密模型编码一次，查询由剪枝模型编码时的性能-算力曲线。上排为 ImageNet-1k k-NN 精度，下排为 BEIR 平均 nDCG@10。基线在深剪枝下与稠密索引的兼容性崩塌，本文则平滑退化。

**图像** ：50% 稀疏度时最好的基线在 AugReg、DINOv3、SigLIPv2 上精度跌破 6%（VBP 因只剪 MLP 神经元甚至到不了高稀疏度）；本文平滑下降，领先亚军 13.8%（SigLIPv2）到 49.4%（DINOv3），40% 时在 DINOv3 上的领先幅度高达 66.2%。

**文本检索** ：20% 稀疏度最多掉 3.0 nDCG@10，40% 掉 10.9。50%/60% 区间本文在每个骨干上都优于两个基线，60% 时在 Qwen3 8B 上领先多达 36.5 分。双向的 E5-L 最稳：30% 只掉 1.5 分，60% 掉 9.3 分。

**实测加速** ：理论 GFLOPs 之外，作者在 A100（fp16、torch.compile、GPU 友好剪枝）上实测：AugReg ViT-B/16 在 60% 稀疏度、batch 128 下吞吐达稠密模型的 1.80 倍；单查询延迟加速 1.26 倍；预算切换平均开销仅 0.46 ms（不含首次编译）。

### 3.4 消融实验：三个关键设计都被验证

**（1）逐单元分数修复对角近似的缺陷** 。在相同优化协议、相同算力下比较不同粒度的参数化（AugReg ViT-B/16，50% 稀疏度，ImageNet-1k k-NN）：

| 优化粒度 | 参数量 | ImageNet-1k 精度 (%) |
| --- | ---: | ---: |
| 不优化 | 0 | 29.4 |
| Sublayer 级 | 24 | 29.4 |
| MLP Sublayer + Head（SnapViT 参数化） | 156 | 34.6 |
| 8 组神经元 + Head | 240 | 36.3 |
| 64 组神经元 + Head | 912 | 43.3 |
| 512 组神经元 + Head | 6,288 | 51.9 |
| **逐神经元 + 逐 Head（本文）** | 37,008 | **59.7** |

精度随粒度单调上升。值得注意的是，把 SnapViT 的参数化放到本文的梯度优化协议下，精度从 28.4% 提升到 34.6%——但仍落后逐单元优化 25.1%，说明 **粒度本身才是瓶颈，不只是优化器** 。

**（2）对初始分数的鲁棒性** 。比较随机分数、权重幅值、Wanda、DINO（默认）、幅值增强 SNIP 五种初始化在优化前后的表现（AugReg ViT-B/16 k-NN，%）：

| 初始化 | 20% Plain → Ours | 40% Plain → Ours | 60% Plain → Ours |
| --- | --- | --- | --- |
| Random | 58.2 → 76.8 (+18.6) | 14.7 → 63.2 (+48.5) | 7.9 → 35.5 (+27.6) |
| Magnitude | 7.3 → 77.0 (+69.7) | 4.2 → 63.7 (+59.5) | 2.5 → 33.8 (+31.3) |
| Wanda | 7.2 → 77.8 (+70.6) | 6.2 → 63.1 (+56.9) | 4.8 → 32.6 (+27.8) |
| SNIP Mag. | 78.2 → 79.7 (+1.5) | 37.8 → **69.4** (+31.6) | 12.4 → 43.3 (+30.9) |
| **DINO（默认）** | **78.6** → **80.1** (+1.5) | **42.5** → 69.1 (+26.6) | **17.0** → **45.2** (+28.2) |

优化能把 Wanda 从 7.2% 拉到 77.8%、把随机初始化从 58.2% 拉到 76.8%（20% 稀疏度下与 DINO 初始化只差 3% 左右），证明逐单元重排确实能"修复"糟糕的初始排序。但 60% 深剪枝时好初始化仍有优势（DINO 领先 Wanda 12.6%）。

**（3）多目标联合优化优于单目标分别优化** （AugReg ViT-B/16 k-NN，%）：

| 稀疏度 | 单目标优化 | 联合优化（Elastic） |
| --- | ---: | ---: |
| 20% / 28.2 G | 79.7 | **80.1** (+0.4) |
| 40% / 21.2 G | 67.2 | **69.1** (+1.9) |
| 60% / 14.2 G | 41.6 | **45.2** (+3.6) |

联合优化在所有稀疏度上都更好，且优势随深度增大（+0.4% → +3.6%）——印证了"多阈值让排序不同区段都暴露在有效梯度下"的假设，同时还免去了预算变化时的重新优化。

## 四、附录深挖：实现细节与更多证据

### 4.1 排序优化到底改变了什么

![稀疏度分配分析](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/sparsity/allocation-overlap-models-b50-compact.png)

> 图解：AugReg、DINOv3、SigLIPv2 ViT-B/16 在 50% 稀疏度下保留的 MLP 神经元与 Attention Head 分布。上排对比优化前后各 block 的分配与均匀剪枝；下排在优化的 block 级分配内，把保留单元分为"初始分数也会选中的"和"优化后新选中的"。即使 block 级配额几乎不变，新选中的单元也大量出现——说明优化不只是跨 block 重新分配容量，还在 block 内部重排了单元。

跨 block 的重分配因骨干而异：AugReg 变化温和，DINOv3 大幅重排（给最后一个 MLP block 回补容量），SigLIPv2 则保留了对靠后 MLP block 的激进剪枝。三个骨干上，block 内部的重排都清晰可见，尤以 MLP 神经元为甚。

### 4.2 更多骨干与线性探测

![DeiT-B/16 结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-probe-plus-transfer-deit-b.png)

> 图解：DeiT-B/16 在 ImageNet-1k 与迁移数据集上的 k-NN（上）与线性探测（下）精度-算力曲线。浅剪枝各基线有竞争力，深剪枝下本文在两种协议中都保住更高精度。

![AugReg 线性探测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/probe-linear-plus-transfer-augreg-b.png)

> 图解：AugReg ViT-B/16 线性分类精度曲线。相比 k-NN，线性探测在深剪枝下保留更多精度，但本文相对基线的定性优势一致。

![DINOv3/SigLIPv2 大模型线性探测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-model-plus-transfer-dinov3-h-siglip2-g-linear.png)

> 图解：DINOv3 ViT-H+/16（上）与 SigLIPv2 ViT-G/16（下）的线性分类结果。50% 稀疏度时 DINOv3 在 ImageNet-1k 上的线性精度几乎是一条平线：保住 84.4%（仅掉 0.8%），迁移平均 85.4%（掉 3.3%），同时删去 427M 参数；SigLIPv2 ViT-G/16 删去 573M 参数，掉 12.3%/17.2%。50% 稀疏度下本文在 ImageNet-1k 上领先亚军 SnapViT 41.2%（DINOv3）和 49.9%（SigLIPv2）。

### 4.3 各家族完整缩放曲线与 Pareto 前沿

![AugReg 缩放](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-model-plus-transfer-augreg.png)

> 图解：AugReg 家族（ViT-S/16、B/16、L/16）在 ImageNet-1k 与迁移数据集上的 k-NN 曲线，模型越大越耐剪。

![DeiT-III 缩放](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-model-plus-transfer-deit3.png)

> 图解：DeiT-III 家族（S/16 到 H/14）曲线。60% 稀疏度下 ViT-H/14 在 ImageNet-1k 仅掉 10.4%、迁移平均仅掉 2.73%，同时删去约 378M 参数。

![DINOv3 缩放](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-model-plus-transfer-dinov3.png)

> 图解：DINOv3 ViT-B/16 与 ViT-H+/16 曲线。ViT-H+/16 剪到 60% 稀疏度删去 512M 参数，k-NN 精度只掉 5.6%（ImageNet-1k）和 6.4%（迁移平均）。

![SigLIPv2 缩放](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-model-plus-transfer-siglip2.png)

> 图解：SigLIPv2 ViT-B/16 与 ViT-G/16 曲线。ViT-G/16 剪到 50% 删去 573M 参数，ImageNet-1k 掉 10.5 个百分点、迁移平均掉 12.9 个百分点。

![DINO WebSSL Pareto](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/pareto-dino-webssl-in1k.png)

> 图解：DINO WebSSL 六模型家族（300M–7B）的 ImageNet-1k k-NN 精度-算力曲线，横轴为 GFLOPs。剪枝后的 3B 和 7B 模型分别 Pareto 优于稠密的 2B 和 5B 锚点——"买大剪小"可能比"直接买小"更划算。

### 4.4 逐任务迁移表现：细粒度任务最敏感

![AugReg 逐任务](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-task-augreg-b-knn.png)

> 图解：AugReg ViT-B/16 在 ImageNet-1k 与九个迁移数据集上的逐任务曲线。50% 稀疏度下 ImageNet-1k 掉 24.3%，CIFAR-100、Flowers 102、Food 101 掉超 30%，但 Pets、EuroSAT、RESISC45 最多只掉 8.1%。细粒度任务最敏感；Pets 的好表现可能源于校准数据对齐（ImageNet-1k 含大量猫狗类）。唯一的例外是 EuroSAT，SnapViT 与 OPTIN 在那里更强。

![DINOv3-H 逐任务](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-task-dinov3-h-knn.png)

> 图解：DINOv3 ViT-H+/16 逐任务曲线，50% 稀疏度几乎无损：ImageNet-1k 掉 1.8%，迁移数据集掉 0.5%（Pets）到 6.2%（CIFAR-100），而 Flowers 102 甚至从 90.2% **提升** 到 97.9%。同稀疏度下领先 SnapViT 42.9%（ImageNet-1k），40% 起在每个数据集上都不输任何基线。

![SigLIPv2-G 逐任务](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/stacked-task-siglip2-g-knn.png)

> 图解：SigLIPv2 ViT-G/16 逐任务曲线，退化更不均匀：50% 时 ImageNet-1k 掉 10.5%，Aircraft 掉 39.6%，EuroSAT 只掉 0.8%。浅剪枝时 VBP/SnapViT 在 Aircraft 上领先，但 30% 起本文在每个数据集上全面领先。

### 4.5 语义分割细节

![AugReg Hummingbird](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/hummingbird-voc-augreg-b.png)

> 图解：AugReg ViT-B/16 在 Pascal VOC 2012 上的 in-context 语义分割 mIoU。本文在 50% 稀疏度时领先亚军 OPTIN 25.1%。

![DINOv3 Hummingbird](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/hummingbird-voc-dinov3-b.png)

> 图解：DINOv3 ViT-B/16 的 mIoU 曲线。40% 稀疏度（141.5 GFLOPs）时保住 70.3% mIoU，比亚军 VBP 高 39.8% 还少用 3.1 GFLOPs。本文无需任何任务特定校准就更好保住了 patch 级表征。

### 4.6 文本任务全表：九编码器、三种方法、七档稀疏度

MTEB(eng, v2) 均分全表（粗体为该档最优）：

| Backbone | 方法 | Dense | 10% | 20% | 30% | 40% | 50% | 60% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-S | SnapViT | 64.3 | 63.4 | **62.4** | **60.5** | 57.6 | 54.6 | 51.8 |
|  | SlimGPT |  | 63.2 | 61.2 | 58.4 | 54.7 | 48.9 | 41.8 |
|  | Ours |  | **63.7** | 62.3 | 60.3 | **58.4** | **56.7** | **54.3** |
| BGE-B | SnapViT | 65.2 | **65.0** | **64.2** | **63.2** | 60.1 | 55.6 | 49.9 |
|  | SlimGPT |  | 63.8 | 61.2 | 57.1 | 53.4 | 40.9 | 40.6 |
|  | Ours |  | 64.8 | **64.2** | **63.2** | **61.8** | **59.5** | **56.4** |
| BGE-L | SnapViT | 65.8 | 65.2 | 64.2 | 61.1 | 54.6 | 48.8 | 43.8 |
|  | SlimGPT |  | 59.1 | 51.8 | 50.1 | 49.6 | 47.1 | 40.0 |
|  | Ours |  | **65.5** | **64.8** | **63.6** | **61.2** | **59.5** | **57.1** |
| E5-B | SnapViT | 62.5 | 61.9 | 60.5 | 57.0 | 49.8 | 46.2 | 42.8 |
|  | SlimGPT |  | **62.4** | 59.9 | 54.1 | 55.6 | 51.9 | 48.0 |
|  | Ours |  | 62.1 | **62.0** | **60.8** | **59.8** | **57.7** | **55.6** |
| E5-L | SnapViT | 62.4 | 61.9 | 56.9 | 47.2 | 39.8 | 37.5 | 30.2 |
|  | SlimGPT |  | 54.7 | 55.6 | 54.6 | 52.0 | 46.1 | 43.5 |
|  | Ours |  | **62.4** | **61.7** | **61.1** | **60.3** | **59.1** | **57.3** |
| E5-Mistral 7B | SnapViT | 68.3 | 67.8 | 67.0 | **66.4** | **63.9** | 56.2 | 43.3 |
|  | SlimGPT |  | 67.5 | 65.6 | 62.1 | 56.3 | 48.9 | 36.7 |
|  | Ours |  | **68.1** | **67.3** | 65.6 | 62.7 | **59.4** | **56.8** |
| Qwen3 0.6B | SnapViT | 70.1 | **69.6** | 66.7 | **62.9** | 57.6 | 47.3 | 39.7 |
|  | SlimGPT |  | 50.9 | 37.4 | 21.0 | 18.3 | 14.1 | 14.1 |
|  | Ours |  | 69.5 | **66.8** | 61.9 | **58.2** | **54.4** | **52.1** |
| Qwen3 4B | SnapViT | 74.1 | **72.6** | **71.5** | **69.2** | **64.4** | 57.2 | 46.3 |
|  | SlimGPT |  | 68.7 | 59.8 | 44.4 | 34.0 | 22.2 | 14.6 |
|  | Ours |  | **72.6** | 70.8 | 68.0 | 63.6 | **58.6** | **54.6** |
| Qwen3 8B | SnapViT | 74.6 | 73.9 | **72.6** | **70.7** | 67.0 | 58.6 | 42.5 |
|  | SlimGPT |  | 71.8 | 67.4 | 61.2 | 50.3 | 39.4 | 26.1 |
|  | Ours |  | **74.2** | **72.6** | 70.3 | **67.3** | **63.8** | **58.9** |

规律一目了然： **浅剪枝几乎免费** （20% 时没有模型掉超 3.3 分，E5 系列掉不到 1 分）； **深剪枝下双向编码器远比 decoder 优雅** （60% 时 BGE/E5 掉 5.1–10.0 分，E5-Mistral/Qwen3 掉 11.5–19.5 分）。

把检索与非检索任务拆开看（表格见原文附录），检索对剪枝敏感得多：60% 稀疏度时 BGE-L 检索掉 19.6 分，非检索只掉 5.3 分；本文在 50%/60% 稀疏度下于每个编码器上都取得最高 nDCG@10。文本模型上"更大更耐剪"并不成立（Qwen3 4B 比 0.6B 掉得还多），规模红利因家族和任务组而异。

![文本 Pareto 前沿](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Your-Transformer-Is-Secretly-Elastic/media/text/pareto-mteb.png)

> 图解：BGE、E5、Qwen3 编码器的 MTEB 分数-GFLOPs（对数轴）曲线，虚线为 Pareto 前沿。稠密缩放对 BGE/E5 收益甚微（BGE-S 64.3 → BGE-L 65.8），Qwen3 从 0.6B 到 4B 提升明显（70.1 → 74.1）但 8B 只到 74.6。弹性剪枝把前沿推到稠密模型之外：BGE-B/L 在 10% 稀疏度内仍在前沿上，Qwen3 8B/4B 分别在 10%/20% 内，剪枝最小的 BGE-S 还能把前沿延伸到 5 GFLOPs 以下。

**与 EffiR 的对比** （BEIR-13 子集 nDCG@10，同一剪枝模型同时编码文档与查询）：对比两类基线——E5-Mistral 7B 直接丢层（不重训）、Mistral 7B 丢层/剪枝后检索微调。对不重训的丢层基线，本文在每一档都更高：4.3B 参数时 49.9 vs 24.7（领先 25.2 分）。对微调过的 EffiR 模型，本文在 5.7B 以上档位仍然更高；4.3B 时以 49.9 vs 55.3 落后 5.4 分——但本文 **完全没动权重** ，这个差距堪称惊人。

### 4.7 弹性检索全表

稠密模型建索引、剪枝模型编码查询时，九个文本编码器在 15 个 BEIR 数据集上的平均 nDCG@10（粗体为该档最优）：

| Backbone | 方法 | Dense | 10% | 20% | 30% | 40% | 50% | 60% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-S | SnapViT | 51.7 | 50.0 | 46.5 | 41.6 | 26.6 | 14.2 | 4.9 |
|  | SlimGPT |  | 48.8 | 41.6 | 28.5 | 9.8 | 2.4 | 0.2 |
|  | Ours |  | **51.4** | **48.3** | **44.5** | **40.0** | **36.7** | **32.0** |
| BGE-B | SnapViT | 53.2 | **52.9** | **51.4** | 49.6 | 45.3 | 29.7 | 12.6 |
|  | SlimGPT |  | 51.4 | 48.3 | 43.0 | 32.9 | 2.6 | 0.6 |
|  | Ours |  | 52.4 | 51.0 | **49.7** | **46.9** | **42.8** | **35.3** |
| BGE-L | SnapViT | 54.3 | 53.2 | **52.1** | 47.1 | 30.0 | 17.4 | 5.5 |
|  | SlimGPT |  | 47.5 | 34.7 | 31.3 | 26.7 | 19.4 | 8.1 |
|  | Ours |  | **53.6** | 52.0 | **48.9** | **43.4** | **38.4** | **33.6** |
| E5-B | SnapViT | 50.2 | 49.8 | 48.1 | 44.0 | 30.3 | 19.7 | 4.7 |
|  | SlimGPT |  | 49.8 | 46.4 | 35.1 | 41.3 | 34.4 | 17.4 |
|  | Ours |  | **50.0** | **49.9** | **49.2** | **47.5** | **45.3** | **41.8** |
| E5-L | SnapViT | 50.5 | **50.6** | 44.1 | 23.7 | 5.9 | 1.9 | 0.1 |
|  | SlimGPT |  | 43.6 | 41.1 | 37.8 | 33.3 | 24.3 | 16.0 |
|  | Ours |  | 50.3 | **49.8** | **49.0** | **47.5** | **45.4** | **41.2** |
| E5-Mistral 7B | SnapViT | 56.9 | 56.6 | **55.8** | **54.9** | **51.5** | 40.9 | 15.4 |
|  | SlimGPT |  | **56.8** | 55.7 | 52.6 | 40.4 | 29.2 | 13.2 |
|  | Ours |  | 56.7 | 55.7 | 53.7 | 49.7 | **41.7** | **33.7** |
| Qwen3 0.6B | SnapViT | 55.2 | 54.6 | 52.7 | **50.3** | 44.4 | 25.9 | 9.0 |
|  | SlimGPT |  | 36.9 | 8.2 | 1.2 | 0.1 | 0.1 | 0.1 |
|  | Ours |  | **54.7** | **53.2** | 50.1 | **47.2** | **41.0** | **34.1** |
| Qwen3 4B | SnapViT | 61.4 | **60.4** | **59.1** | **56.3** | 53.0 | 45.5 | 23.9 |
|  | SlimGPT |  | 58.5 | 50.3 | 23.9 | 13.2 | 3.5 | 0.2 |
|  | Ours |  | 59.9 | 58.4 | 56.2 | **53.3** | **47.8** | **41.5** |
| Qwen3 8B | SnapViT | 62.6 | 62.0 | 60.3 | 58.8 | 55.6 | 47.8 | 7.1 |
|  | SlimGPT |  | 60.9 | 57.6 | 49.6 | 21.0 | 7.8 | 2.5 |
|  | Ours |  | **62.2** | **61.2** | **59.2** | **56.2** | **52.5** | **43.6** |

本文在 BGE-S、E5-B 的所有档位上领先；其余骨干从 40% 起领先（E5-Mistral 7B 上 SnapViT 在 40% 内最多反超 1.8 分）；50% 和 60% 时本文在每个骨干上都最优。

### 4.8 权重修正：不修也能打，修了更强

主实验为隔离单元选择的效果而禁用权重修正，附录补齐了修正对比（AugReg ViT-B/16，ImageNet-1k k-NN）：

| 方法 | 10% Plain/Corr. | 20% | 30% | 40% | 50% | 60% |
| --- | --- | --- | --- | --- | --- | --- |
| SparseGPT | 78.8 / 82.4 | 64.2 / 79.3 | 36.5 / 72.2 | 20.3 / 59.9 | 14.1 / 42.4 | 10.2 / 29.1 |
| FPTP | 82.0 / 82.1 | 78.5 / 79.5 | 71.3 / 76.2 | 44.3 / 57.0 | 17.2 / 19.3 | 12.6 / 13.3 |
| SnapViT | 82.2 / **82.8** | 77.5 / **81.0** | 70.1 / **78.4** | 57.7 / **73.0** | 28.4 / 63.3 | 17.7 / 47.5 |
| **Ours** | **82.6** / **82.8** | **80.1** / **81.0** | **75.7** / 78.1 | **69.1** / 72.6 | **59.3** / **64.3** | **45.2** / **54.0** |

不修正时本文在每个档位都胜过所有未修正基线，也胜过修正后的 SparseGPT（全部档位）与 FPTP（除 30% 档），与修正后的 SnapViT 差距不超过 4.0%。修正对本文同样有效（60% 时 +8.8%），修正后反超修正 SnapViT 6.5%。在 DINOv3 ViT-H+/16 上，本文不修正就胜过修正后的两个基线（30% 起），60% 时领先修正 SnapViT 18.2%（79.4 vs 61.2）。文本侧：未修正的本文在 BGE-B、BGE-L、Qwen3 0.6B/8B 的所有档位上都胜过修正后的 SlimGPT；40%–60% 区间在九个编码器中的七个上更优。注意修正的代价是 **每个稀疏度档位要存一套独立权重** ，弹性部署的意义就打了折扣。

### 4.9 实现细节速览

- **初始分数校准** ：视觉用 DINO 目标（2 个 global crops + 10 个 local crops，无颜色增强，teacher 即冻结的自身，无 projection head/centering/sharpening），1000 张 ImageNet-1k 随机样本；文本用对称 InfoNCE（$\tau=0.02$），1000 对 MS MARCO (query, passage)，截断至 256 token。SwiGLU 模型只对两个输入投影聚合分数；GQA 模型剪 query head，用输出投影对应列打分。
- **排序优化** ：Adam（视觉 $\eta=0.1$、文本 $\eta=0.2$），300 步；sigmoid 温度视觉 $\tau=0.2$、文本 $\tau=0.1$；视觉校准最多 2000 张 ImageNet-1k，文本最多 1000 条 MS MARCO。大模型 batch 小，看不完整个校准集（如 ViT-G 只见 1200 张、Qwen3 8B 只见 600 条）。
- **优化损失** ：监督模型用交叉熵；SigLIPv2 用 sigmoid 图文对比损失（对全部 ImageNet-1k 类名的预计算文本 embedding）；DINOv3、DINO WebSSL 与文本编码器用剪枝/稠密特征的余弦距离。
- **FLOPs 记账** ：只计可剪单元的 FLOPs（MLP 的输入/输出投影；attention 的 QKVO 投影、注意力矩阵乘与 softmax），每 MAC 计 2 FLOPs。MLP 单神经元 $C_{\mathrm{MLP}} = 2JEN$（标准 MLP $J=2$，SwiGLU $J=3$）；标准 MHA 单 head $C_{\mathrm{MHA}} = 8EH_dN + (4H_d + 3)N^2$；GQA 按保留的 query/KV head 数精确计 $C_{\mathrm{GQA}} = 4EH_dN(A_q + A_{kv}) + (4H_d + 3)A_qN^2$，排序时给每个 query head 摊销成本。文本序列统一按 $N=256$ 计。
- **测试时剪枝** ：沿优化后的排序从顶端保留单元直到累计 FLOPs 达 $(1-b)$；保底约束是每个 block 至少保留 5% 的 MLP 神经元和 20% 的 attention head。

### 4.10 加速实测与工程细节

测量环境：单卡 NVIDIA A100 40GB，fp16，PyTorch 2.13 + timm，eager 与 torch.compile 两种执行模式。吞吐为 batch 128 下 200 次前向的平均，延迟为 500 次单独计时的中位数。

**GPU 友好分块对精度的影响** （AugReg ViT-B/16，把 MLP 神经元按 8/64 一块地剪）：

| GFLOPs | IN-1k A=1 | A=8 | A=64 | Transfer A=1 | A=8 | A=64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31.7 | **82.6** | 82.5 | 82.4 | **78.9** | **78.9** | 78.8 |
| 28.2 | **80.1** | 79.6 | 79.5 | **76.9** | 76.7 | 76.7 |
| 24.8 | **75.7** | 75.4 | 75.4 | **74.1** | 73.4 | 73.3 |
| 21.2 | **69.1** | **69.1** | **69.1** | 66.8 | **69.7** | 69.3 |
| 17.8 | **59.3** | 58.6 | 58.2 | 63.7 | **63.8** | 63.7 |
| 14.2 | **45.2** | 41.8 | 42.0 | **55.9** | 55.5 | 55.5 |

低中稀疏度几乎无代价，60% 时分块剪枝在 ImageNet-1k 上落后约 3.2–3.4 分。

**吞吐** （batch 128，相对稠密模型的加速比；eager 稠密 3572 samples/s，compiled 4143）：

| GFLOPs | Eager A=1 | A=8 | A=64 | Compiled A=1 | A=8 | A=64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31.7 | 0.75× | 1.03× | 1.06× | 1.03× | 1.07× | 1.08× |
| 28.2 | 0.70× | 1.10× | 1.14× | 1.11× | 1.16× | 1.17× |
| 24.8 | 0.91× | 1.19× | 1.23× | 1.24× | 1.27× | 1.28× |
| 21.2 | 1.03× | 1.30× | 1.34× | 1.35× | 1.39× | 1.42× |
| 17.8 | 1.16× | 1.44× | 1.47× | 1.53× | 1.56× | 1.59× |
| 14.2 | 1.27× | 1.59× | 1.61× | 1.74× | 1.77× | **1.80×** |

一个非常工程化的发现：eager 模式下不规整（ragged）的隐藏维度在浅剪枝时甚至比稠密模型还慢（20% 稀疏度只有 0.70×），分块剪枝可提速达 1.63 倍；torch.compile 基本抹平这个差距。

**延迟** （60% 稀疏度，相对稠密的加速比）：eager 模式 batch ≤ 16 时受 kernel launch 开销支配，剪枝无收益；batch 128 时 compiled + A=64 达 1.79×；batch 1 时用 reduce-overhead 编译模式（捕获 CUDA graph）才有收益（A=64 时 1.26×）。

**预算切换成本** ：保留一套稠密权重，剪枝模型表示为截断视图；因 QKV 融合投影无法连续切片，每个 block 用固定 buffer gather（约 85 MB 额外存储）。72 种切换的平均成本仅 **0.46 ms** ，不足 eager 模式 batch-1 前向耗时的 10%。新预算首次编译需 12–49 秒，可在启动时一次性付清。

### 4.11 运行耗时：分钟级，亚线性扩展

在单卡 RTX A6000 上产出全部六个稀疏度目标的总耗时：

| 方法 | ViT-S/16 | ViT-B/16 | ViT-L/16 | ViT-H/14 |
| --- | ---: | ---: | ---: | ---: |
| SparseGPT | 4.5 min | 5.0 min | 11 min | 20 min |
| FPTP | 5.6 min | 9.0 min | 19 min | 35.6 min |
| OPTIN | 12.5 min | 1.0 h | 8.0 h | 34.6 h |
| VBP | 2.6 h | 6.6 h | 20.6 h | 53.7 h |
| SnapViT | 2.2 min | 4.0 min | 11.2 min | 28.8 min |
| **Ours** | **8.7 min** | **13.4 min** | **24.5 min** | **40.7 min** |

本文从 ViT-S 到 ViT-H 参数量涨 29 倍，耗时只涨 4.7 倍（亚线性）。OPTIN 和 VBP 对 ViT-H/14 分别要 34.6 和 53.7 小时——是本文的 50 倍以上，而这还是假设 OPTIN 的 leave-one-out 搜索跨目标复用的乐观情形。

### 4.12 其余消融要点

- **任务对齐校准** ：用下游任务训练集做校准，27 个设置中 25 个提升，最高 +26.6%（Flowers 102 @ 40%）、+20.8%（Pets @ 60%）。
- **原生损失 vs 余弦对齐** ：有原生损失就用原生损失——AugReg 在 ImageNet-1k 上提升 0.9%（20%）到 8.9%（60%），SigLIPv2 提升 1.4% 到 10.1%，个别迁移集（CIFAR 100、EuroSAT）略有回退。
- **校准集规模** ：64 → 9600 唯一样本在 50% 稀疏度下提升 14.3 分；但默认 2000 样本再往上收益不稳定（40% 档 +2.6，60% 档反降 2.0）。
- **统一排序 vs 分类型阈值** ：跨 block 阈值一致优于逐 block 阈值（DINOv3 40% 档 +34.9%）；再统一神经元与 head 的收益因骨干而异（DINOv3 50% 档再 +8.7%，SigLIPv2 上略逊于跨 block 方案），但统一排序对用户只需一个稀疏度数字，省去手工分配表。
- **弹性插值** ：只在 10%–70%（步长 10%）七个锚点上优化，在每 5% 处评估，中间档性能平滑单调、无凹陷——排序确实覆盖连续预算区间。
- **温度扫描** ：$\tau \in \{0.01, ..., 5.0\}$ 中 $\tau=0.1$ 均值最高；过低（0.01）梯度太窄，过高（5.0）退化为不优化。sigmoid 要"足够锐利逼近硬剪枝，又足够平滑传导梯度"。
- **随机性** ：三个种子的标准差最多 0.65%（ImageNet-1k @ 60%），多数设置低于 0.56%，结果稳定。

## 五、总结与展望

- **核心方法** ：把剪枝松弛为 sigmoid 软掩码，在 14 个稀疏度目标上联合优化逐神经元、逐 head 的重要性分数，权重全程冻结、零重训。
- **一个排序服务所有预算** ：训练一次得到全局排序，推理时从尾部裁剪即可落到任意稀疏度，中间档位无需再优化。
- **深剪枝优势巨大** ：DINOv3 ViT-H+/16 剪 50%（-427M 参数）线性探测仅掉 0.8%；60% 稀疏度下全面碾压 one-shot 基线。
- **工程上可落地** ：A100 上 1.80× 吞吐、1.26× 单查询延迟加速，预算切换仅 0.46 ms；构建弹性模型只需几分钟且亚线性扩展。
- **部署形态有想象力** ：剪枝后的大模型可 Pareto 优于更小档稠密模型；稠密建索引 + 弹性查询的非对称检索可行。

局限与展望：decoder 型文本 embedder 在深剪枝下退化仍明显；尚未在扩散图像生成与自回归语言建模上验证；利用学到的排序做"弹性后训练"（elastic post-training）是值得探索的方向。这个工作最启发人的一点是：预训练 Transformer 内部的冗余远比静态重要性分数揭示的多，只要给排序足够的自由度去重排，"弹性"其实一直藏在权重里。

> 本文参考自 [Your Transformer Is Secretly Elastic](http://arxiv.org/abs/2610.20588v1)