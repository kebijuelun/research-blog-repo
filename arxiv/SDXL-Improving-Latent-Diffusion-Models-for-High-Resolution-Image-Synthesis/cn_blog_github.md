# SDXL：Stable Diffusion 的高分辨率系统性升级

文生图模型在 2022 年之后陷入了一个微妙的局面：闭源的 Midjourney、DALL-E 2 一路狂奔，而开源的 Stable Diffusion 虽然生态繁荣，生成质量却逐渐被拉开差距，尤其在构图完整性、细节清晰度和复杂提示词的跟随能力上明显落后。SDXL 这篇论文做的事情，就是把 Stable Diffusion 从头到脚做了一次系统性升级——用 3 倍大的 UNet、双文本编码器、两个巧妙的“微观条件化”技巧，再加一个专门的精修（refinement）模型。最硬的结果数字是：在用户偏好盲测中，SDXL + refiner 以 **48.44%** 的胜率碾压 SD 1.5 的 7.91% 和 SD 2.1 的 6.71%；在与 Midjourney v5.1 的 17153 次对比中，SDXL 也以 **54.9%** 的偏好率领先。更重要的是，代码和权重全部开源，把“接近闭源 SOTA”的能力交给了整个社区。

![SDXL 生成效果概览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/cherries/teaserblock001.jpg)

> 图解：SDXL 的生成效果合集。可以看到它对复杂提示词（多主体、特定风格、长文本描述）的跟随能力和画面质感，相比上一代 SD 有明显提升。

## 一、背景：扩散模型与 Stable Diffusion 的遗留问题

SDXL 是一个 LDM（Latent Diffusion Model，潜在扩散模型）：不直接在像素空间做扩散，而是先用一个 autoencoder 把图像压缩到一个低维 latent 空间，在 latent 空间里做加噪和去噪，最后解码回像素。这是 Stable Diffusion 系列的祖传配方，好处是计算量大幅降低。

先快速过一遍扩散模型的数学骨架（论文附录中的总结，帮助后续理解）。

记数据分布为 $p_{\text{data}}(\boldsymbol{x}_0)$，给数据加上方差为 $\sigma^2$ 的高斯噪声得到 $p(\boldsymbol{x}; \sigma)$。扩散模型采样时从大方差高斯噪声出发，逐步去噪回数据分布。这个去噪过程可以写成 Probability Flow ODE：

$$
d\boldsymbol{x} = -\dot{\sigma}(t)\,\sigma(t)\,\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}; \sigma(t))\, dt
$$

其中 $\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}; \sigma)$ 是 score function（分数函数），$\sigma(t)$ 是用户指定的噪声调度。训练则归结为用 denoising score matching 学一个去噪器 $D_\theta$：

$$
\mathbb{E}_{(\boldsymbol{x}_0, \boldsymbol{c}) \sim p_{\text{data}},\, (\sigma, \boldsymbol{n}) \sim p(\sigma, \boldsymbol{n})} \left[ \lambda_\sigma \| D_\theta(\boldsymbol{x}_0 + \boldsymbol{n}; \sigma, \boldsymbol{c}) - \boldsymbol{x}_0 \|_2^2 \right]
$$

$\boldsymbol{c}$ 是条件信号（文本 prompt），$\lambda_\sigma$ 是噪声级别的加权函数。SDXL 沿用离散时间框架，噪声级别取 1000 档，$\lambda_\sigma = \sigma^{-2}$，与前辈们保持一致。

采样时再用 classifier-free guidance 把条件预测和无条件预测混合：

$$
D^w(\boldsymbol{x}; \sigma, \boldsymbol{c}) = (1 + w)\, D(\boldsymbol{x}; \sigma, \boldsymbol{c}) - w\, D(\boldsymbol{x}; \sigma)
$$

$w$ 是 guidance strength。训练时以约 10% 的概率把条件换成 null embedding，让同一个网络同时学会“有条件”和“无条件”两种预测。

了解了底座之后，来看看 SDXL 到底改了什么。

## 二、架构与规模：3 倍大的 UNet + 双文本编码器

自 DDPM 起，卷积 UNet 一直是扩散模型的主流骨干。SDXL 顺着 ADM、LDM 的演进路线继续演进，但做了一个非常“工程直觉”的调整： **把 Transformer 计算的重心挪到 UNet 的低层特征上** 。

具体做法（对标 UViT 作者 Hoogeboom 等人的思路）：

- 最高分辨率的特征层 **不放** Transformer block（太贵）；
- 中间两层分别放 2 个和 **10 个** Transformer block（原始 SD 每个层级只放 1 个）；
- 直接砍掉 UNet 中 8 倍下采样的最低层。

文本编码方面，SDXL 同时使用 **OpenCLIP ViT-bigG** 和 **CLIP ViT-L** 两个文本编码器，把两者的倒数第二层输出沿 channel 维度拼接（借鉴 eDiff-I），context 维度从 768 拉大到 **2048**。此外还引入了 OpenCLIP 的 pooled text embedding 作为额外条件（借鉴 GLIDE）。

| 对比项 | SDXL | SD 1.4/1.5 | SD 2.0/2.1 |
| --- | --- | --- | --- |
| UNet 参数量 | 2.6B | 860M | 865M |
| Transformer blocks | [0, 2, 10] | [1, 1, 1, 1] | [1, 1, 1, 1] |
| Channel multiplier | [1, 2, 4] | [1, 2, 4, 4] | [1, 2, 4, 4] |
| 文本编码器 | CLIP ViT-L & OpenCLIP ViT-bigG | CLIP ViT-L | OpenCLIP ViT-H |
| Context 维度 | 2048 | 768 | 1024 |
| Pooled 文本嵌入 | OpenCLIP ViT-bigG | 无 | 无 |

> 表格说明：UNet 从约 860M 参数扩到 2.6B，增量主要来自更多的 attention block 和更大的 cross-attention 上下文；两个文本编码器合计还有 817M 参数。

笔者认为这个设计的聪明之处在于： **它没有盲目堆层数，而是把计算预算花在了“语义成型”最关键的层级上** ——低层特征决定构图和语义，高层特征只管纹理细节，Transformer 放在该放的地方。

## 三、微观条件化（Micro-Conditioning）：两个免费的训练技巧

模型变大只是故事的一半。SDXL 真正精妙的部分是两个不需要任何额外标注的 conditioning 技巧。先从一个老问题说起。

### 3.1 尺寸条件化：别再扔数据了

LDM 的两段式结构要求训练图像有一个最小尺寸。过去的做法要么是把小于最小分辨率的图全部丢弃（SD 1.4/1.5 丢掉所有边长小于 512 的图），要么把小图上采样凑数。前者浪费数据，后者引入模糊。

![预训练数据集的尺寸分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/size-dist.jpg)

> 图解：SDXL 预训练数据集的高度-宽度分布，颜色越深表示该尺寸区间的样本越多。虚线标出边长 256 像素的分界线——如果按老办法丢弃小图，39% 的训练数据会被扔掉。

SDXL 的方案非常直白： **把图像的原始分辨率当作条件喂给模型** 。即把缩放前的原始高宽

$$
\boldsymbol{c}_{\text{size}} = (h_{\text{original}}, w_{\text{original}})
$$

各自做 Fourier feature encoding，拼接后加到 UNet 的 timestep embedding 上。推理时用户可以设定“表观分辨率”，模型已经学会了把 $c_{\text{size}}$ 和分辨率相关的图像特征关联起来。

![不同尺寸条件下的生成效果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/sizecondv3/64_robot_2.jpg)

> 图解：同一随机种子、同一提示词下，把 size-conditioning 分别设为 $(64,64)$、$(128,128)$、$(256,256)$、$(512,512)$ 的生成对比（此处展示条件为 64 的样本）。条件尺寸越小，画面越模糊、伪影越多；设为 512 时质量明显最佳。说明模型确实“听懂”了分辨率条件，并且可以在推理时被利用来控制画质。

效果量化验证在 ImageNet 512×512 类条件生成上进行（三个模型对比：只用 512 以上大图训练的 CIN-512-only、全量数据不加条件的 CIN-nocond、全量数据加尺寸条件的 CIN-size-cond）：

| 模型 | FID-5k ↓ | IS-5k ↑ |
| --- | --- | --- |
| CIN-512-only（只用大图，剩 70k 张） | 43.84 | 110.64 |
| CIN-nocond（全量，无条件） | 39.76 | 211.50 |
| CIN-size-cond（全量 + 尺寸条件） | **36.53** | **215.34** |

> 表格说明：扔数据训出来的模型过拟合严重（FID 最差）；不加条件的全量训练会混入模糊样本模式；加上尺寸条件后两个指标同时最优。一个零成本的技巧，解决了数据利用率和画质的双重问题。

### 3.2 裁剪条件化：治好“断头猫”

看下面这个经典失败案例：SD 1.5 和 SD 2.1 生成的拿破仑猫，头经常被画面边缘切掉。

![与旧版 SD 的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/comp_old_model/sd1-5/catpoleon_row.jpg)

![与旧版 SD 的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/comp_old_model/sd2-1/catpoleon_row.jpg)

![与旧版 SD 的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/comp_old_model/sdxl/catpoleon_row.jpg)

> 图解：同一提示词「扮成拿破仑的猫宣传海报」下，SD 1.5（上）、SD 2.1（中）、SDXL（下）各 3 个随机样本。前两代模型生成的猫头部经常被裁掉，SDXL 则构图完整。另一组「喷火龙特写」对比见原文附录，结论一致。

病根在训练数据的预处理：为了凑齐 batch 里统一尺寸的 tensor，标准流程是先把图像短边缩放到目标尺寸，再沿长边 **随机裁剪** 。随机裁剪作为一种数据增强，其“物体可以被画面边缘切断”的分布特征泄漏进了生成结果。

SDXL 的对策同样是把裁剪参数变成条件：数据加载时均匀采样裁剪坐标 $c_{\text{top}}$ 和 $c_{\text{left}}$（从左上角沿高、宽方向裁掉的像素数），经 Fourier embedding 后拼接为

$$
\boldsymbol{c}_{\text{crop}} = (c_{\text{top}}, c_{\text{left}})
$$

作为额外条件输入。训练时随机采样，推理时固定设 $(0, 0)$——由于大规模数据集的图像平均而言是主体居中的，这样采样出来的图自然也是主体居中的。裁剪增强的好处保留了，坏处被隔离了。

两个条件可以方便地组合：沿 channel 维度拼接后再加到 timestep embedding。训练时的采样流程（论文 Algorithm 1）可以概括为：

1. 从数据集取图 $x$，记录原始宽高，得到 $\boldsymbol{c}_{\text{size}}$；
2. 将短边 resize 到目标尺寸；
3. 沿长边均匀采样裁剪起点，得到 $\boldsymbol{c}_{\text{crop}}$；
4. 裁剪后连同两个条件一起送入训练。

![裁剪条件的效果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/cropcondv2/0-0/0-0.jpg)

![裁剪条件的效果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/cropcondv2/512-512/512-512.jpg)

> 图解：固定种子，把 crop-conditioning 从 $(0,0)$ 调到 $(512,512)$，画面会模拟出对应程度的“裁切感”。推理时设 $(0,0)$ 即可得到构图完整的主体。值得一提的是，这个技巧不限于 LDM，任何扩散模型都能用；相比 NovelAI 的 data bucketing，它还能在线实现、无需离线预处理。

## 四、多宽高比训练（Multi-Aspect Training）

真实世界的图片有各种各样的宽高比，而文生图模型默认输出 512² 或 1024² 的正方形——在横屏 16:9 和竖屏内容泛滥的今天，这是个挺不自然的选择。

SDXL 的做法是业界熟悉的 **分桶（bucketing）** ：

- 按宽高比把数据分成若干桶，每桶像素总数尽量接近 $1024^2$，高宽取 64 的倍数；
- 每个训练 batch 内的图来自同一个桶，不同 step 之间轮换桶；
- 把桶的目标尺寸 $\boldsymbol{c}_{\text{ar}} = (h_{\text{tgt}}, w_{\text{tgt}})$ 也做成 Fourier 条件喂给模型。

多宽高比训练作为预训练之后的微调阶段，与前面的尺寸/裁剪条件通过 channel 拼接组合使用（裁剪条件此时只在 64 像素的桶边界内生效）。附录给出了完整的桶配置，从 512×2048（宽高比 0.25）到 2048×512（宽高比 4.0）共 40 种分辨率，覆盖竖屏、方图、宽幅全景。

## 五、更强的 Autoencoder：latent 空间的地基

LDM 的语义构图由扩散模型完成，但图像的局部高频细节上限由 autoencoder 的重建质量决定——地基不牢，上面盖什么都歪。

SDXL 从头训练了同款架构的 autoencoder，改动很朴素： **batch size 从 9 提到 256** ，并用 EMA（指数滑动平均）跟踪权重。结果在 COCO2017 验证集（256×256）上全面领先：

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | rFID ↓ |
| --- | --- | --- | --- | --- |
| SDXL-VAE | **24.7** | **0.73** | **0.88** | **4.4** |
| SD-VAE 1.x | 23.4 | 0.69 | 0.96 | 5.0 |
| SD-VAE 2.x | 24.5 | 0.71 | 0.92 | 4.7 |

> 表格说明：四项重建指标全胜。注意 SD 2.x 的 VAE 还是基于 1.x 微调来的，而 SDXL 的是从头训练。

## 六、整合训练流程与 Refinement 阶段

解决了数据利用、构图、宽高比和 latent 地基之后，最后的问题是怎么把这些串起来训练，以及生成质量还差一口气怎么办。

### 6.1 多阶段训练

最终 SDXL 的训练分三步走：

1. 在内部数据集上以 $256 \times 256$ 分辨率预训练 60 万步，batch size 2048，启用尺寸和裁剪条件；
2. 升到 $512 \times 512$ 继续训 20 万步；
3. 用多宽高比训练在约 $1024 \times 1024$ 像素面积的各种宽高比上微调，并加入 0.05 强度的 offset-noise（社区发现的技巧，让模型能生成纯黑纯白等大动态范围画面）。

扩散调度为 1000 步的离散时间方案。

### 6.2 Refinement 模型：一次 optional 的“磨皮精修”

作者发现 base 模型生成的图有时局部质量欠佳。解法是训练一个 **专门的精修模型** ：在同一个 latent 空间里、用高质量高分辨率数据训练、只负责前 200 个（低噪声）离散噪声级别的 LDM。

推理时，base 模型先生成 $128 \times 128$ 的 latent，然后用 SDEdit 的加噪-去噪流程，让 refiner 在同一 prompt 条件下对 latent 做一次轻度重构。

![SDXL 两阶段流水线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/sdxl_pipeline.jpg)

> 图解：SDXL 两阶段流水线。左：base 模型从噪声生成 128×128 的 latent；右：refinement 模型用 SDEdit 方式对该 latent 加噪再去噪，补出精细细节。两个模型共用同一个 autoencoder。

![Refiner 前后对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/refiner_magic/magic2_combined.jpeg)

> 图解：「被海洋淹没、长满丛林废墟的纽约远景」提示词下，不加 refiner（左）与加 refiner（右）的对比及局部放大。refiner 对细节背景和纹理的提升肉眼可见，尤其人脸和复杂纹理区域。

用户盲测（四选一：SDXL+refiner / SDXL base / SD 1.5 / SD 2.1）的结果：

| 模型 | 偏好胜率 |
| --- | --- |
| SDXL + refinement | **48.44%** |
| SDXL base | 36.93% |
| SD 1.5 | 7.91% |
| SD 2.1 | 6.71% |

> 表格说明：base 模型已经大幅领先两代前辈，加上 refiner 又拉高了 11 个百分点。注意这一步是可选的——代价是推理时要加载两个大模型。

## 七、实验对比：用户说好才是真的好

### 7.1 与旧版 Stable Diffusion 对比

![与旧版 SD 的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/comp_old_model/sdxl/dragon_row.jpg)

![与旧版 SD 的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/comp_old_model/sdxl/capybara_row.jpg)

![与旧版 SD 的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/comp_old_model/sdxl/cangaroo_row.jpg)

> 图解：更多 SDXL 样本（50 步 DDIM，cfg-scale 8.0）：喷火龙特写、体素水豚、举着“SDXL”牌子的袋鼠。无论是文字渲染（牌子上的字母）、材质表现还是构图完整性，都比 SD 1.5/2.1 高出一档。

### 7.2 与 Midjourney v5.1 的用户偏好对比

作者在 PartiPrompts（P2）基准上做了大规模用户研究：每个类别随机取 5 个 prompt，两个模型各生成 4 张 1024×1024 图，交给标注员按“提示词贴合度”投票，共 17153 次对比。

![SDXL vs Midjourney 总体偏好](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/mj_comp/total2.png)

> 图解：总体投票结果，SDXL v0.9 被偏好的比例为 **54.9%** ，领先当时最新的 Midjourney v5.1。

![按 P2 类别的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/mj_comp/categories2.png)

> 图解：按 PartiPrompts 的文本类别拆分（base 模型，不含 refiner）。SDXL 在 6 个类别中的 4 个领先 Midjourney，只在 2 个类别落后。

![按 P2 挑战维度的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/mj_comp/challenges2.png)

> 图解：按挑战维度拆分（含 refiner）。10 个挑战中，SDXL 在 7 个上优于或统计学持平 Midjourney。

此外附录还放出了与 DeepFloyd IF、DALL-E 2、Bing Image Creator、Midjourney v5.2 的定性对比（P2 prompt 随机抽取、固定 seed 3，避免 cherry-picking）：

![与各家模型的定性对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/SDXLcompsv5.jpeg)

> 图解：与闭源模型的同台对比。开源模型第一次在同一画面里不落下风。

### 7.3 FID 指标失灵了？

一个有意思的发现：尽管人类评审压倒性地偏好 SDXL，传统的 FID 和 CLIP score 却完全不认账。

![FID vs CLIP score](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/quant_eval/fidvsclip40-more.jpg)

![FID vs CLIP score](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/quant_eval/fidvsclip50-more.jpg)

> 图解：在 COCO 10k 图文对上，不同 cfg scale 下的 FID-CLIP 曲线。横纵轴分别为 CLIP score（文本对齐度）和 FID（分布距离）。SDXL 的 CLIP score 仅略好，FID 反而是三个模型中最差的——与人类评审的结论完全相反。

这与 Pick-a-Pic 论文的发现互相印证：COCO zero-shot FID 与视觉美感 **负相关** 。对文生图基础模型而言，美学、文本理解、构图这些维度已经超出了 FID 的度量能力，评测必须引入人类评审。

## 八、失败案例与局限

作者很坦诚地展示了 SDXL 的典型翻车场景：

![失败案例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/failure/leave_handpalm.jpg)

![失败案例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/failure/lion_giraffe.jpg)

![失败案例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/failure/piano.jpg)

![失败案例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/SDXL-Improving-Latent-Diffusion-Models-for-High-Resolution-Image-Synthesis/img/failure/truck.jpg)

> 图解：四类典型失败。左上「长出树叶的手掌」：手部结构依然出错，复杂空间关系处理不好；右上「电视里狮子抱长颈鹿」：多层级嵌套场景失败；左下「白色琴凳的三角钢琴」：属性绑定错误；右下「白色车门的红色皮卡」： **概念渗透** （concept bleeding）——颜色属性串到了错误的物体上。

论文分析 concept bleeding 的根源可能出在文本编码器：一是编码器把所有信息压成一个 token，属性-物体绑定容易丢；二是 CLIP 的对比损失依赖同 batch 内的负样本，绑定错误的负例难以覆盖。此外模型对长文本渲染仍有困难，大规模数据也可能带入社会偏见。

## 九、未来方向

- **单阶段化** ：两个大模型占显存、拖速度，需要同等质量的单阶段方案；
- **文本渲染** ：尝试 byte-level tokenizer 或继续 scale 模型；
- **架构** ：UViT/DiT 等纯 Transformer 架构初步试验没有即时收益，但值得更细致的超参研究；
- **蒸馏加速** ：通过 guidance 蒸馏、知识蒸馏、渐进蒸馏降低推理成本；
- **连续时间框架** ：EDM 框架采样更灵活，且不需要 offset-noise 这类噪声调度补丁。

## 十、总结

- **规模升级** ：UNet 扩到 2.6B，Transformer block 集中堆在低层特征，双文本编码器（CLIP ViT-L + OpenCLIP ViT-bigG）把 context 拉到 2048；
- **尺寸条件化** ：把原始分辨率做成条件，39% 本会被丢弃的数据得以保留，ImageNet FID 从 39.76 降到 36.53；
- **裁剪条件化** ：随机裁剪坐标作为条件、推理时归零，既保留数据增强又根治“断头构图”；
- **多宽高比分桶训练 + 更强 VAE + refiner 精修** ：系统工程层层加码，每一环都有明确动机；
- **人评验证** ：对 SD 1.5/2.1 胜率 48.44%，对 Midjourney v5.1 偏好率 54.9%；同时证明 FID/CLIP 对基础文生图模型已失真。

一句话展望：SDXL 把开源文生图拉回了与闭源 SOTA 同一牌桌，但双模型架构的推理成本、concept bleeding 和文本渲染仍是待解之题——这些也正是后来 SD3、DiT 系模型和蒸馏技术的主攻方向。

> 本文参考自 [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)