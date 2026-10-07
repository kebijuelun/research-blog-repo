# DreamBooth：3-5 张照片，让文生图模型记住"你的它"

想象一个场景：你想让 AI 画出"你自己家的狗"环游世界、登上巴黎展厅、变成绘本主角——但无论你给 DALL-E 2 或 Imagen 写多详细的提示词，它画出来的永远只是"某一只狗"，而不是"你的那一只"。这篇来自 Google Research 的 CVPR 2023 论文 **DreamBooth** 解决的就是这个"主体驱动生成"（subject-driven generation）问题。核心思路一句话： **用 3-5 张主体照片微调扩散模型，把该主体绑定到一个罕见 Token（唯一标识符）上，再用一个"先验保持损失"防止模型忘本** 。效果有多硬？在用户研究中，受试者对 DreamBooth 的主体保真度和提示词保真度偏好分别达到 **68% 和 81%** ，碾压同期工作 Textual Inversion 的 22% 和 12%；定量指标上 DINO 主体保真度 0.696，非常接近真实图片上限 0.774。

## 要解决的问题：大模型画得出"狗"，画不出"你的狗"

近几年 Imagen、DALL-E 2、Stable Diffusion 等大型文生图模型的生成质量已经令人惊叹，它们从海量图文对中学到了强大的语义先验——比如"狗"这个词能绑定到各种姿态、各种场景下的狗。但这种能力有一个盲区： **它们无法复现用户给定参考图中那个特定主体的外观** 。

原因在于文本的表达能力有上限。再细致的描述（"复古黄色闹钟、白色表盘、右下角有个黄色数字 3"）也只能圈定一个外观类别，无法锁定到具体个体。即使是用图像嵌入做条件的方法，也只能生成"内容上的变体"，无法精确重建主体。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/background_im_1.png)

> 图解：主体驱动生成任务的困难之处。左一是参考的真实闹钟；中间两列分别是 DALL-E 2 的图像引导生成和 Imagen 的文本引导生成（提示词已写得非常详细），可以看到黄色数字 3、表盘字体等关键细节全部丢失；最右是 DreamBooth 用简单提示词 "a [V] clock in the jungle" 生成的结果，关键视觉特征被完整保留，还自然地融入了丛林场景。

笔者认为，这个问题的本质是： **模型的"语言-视觉词典"里没有为你的主体准备的词条** 。DreamBooth 的思路就是给这本词典"加词条"，而不是在词典外围做提示词工程。

## 背景知识：文生图扩散模型

在讲方法之前，快速回顾一下扩散模型的训练框架。扩散模型是一类概率生成模型，通过逐步去噪从高斯噪声中恢复图像。一个条件扩散模型 $\hat{x}_\theta$ 用平方误差损失来训练，目标是还原加噪图像 $z_t = \alpha_t x + \sigma_t \epsilon$ 的原始干净图像：

$$
\mathbb{E}_{x, c, \epsilon, t}\left[ w_t \left\| \hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x \right\|_2^2 \right]
$$

其中 $x$ 是真实图像，$c$ 是条件向量（由文本提示词编码而来），$\epsilon \sim \mathcal{N}(0, I)$ 是噪声，$\alpha_t$、$\sigma_t$、$w_t$ 控制噪声调度与样本质量，均为扩散时间 $t \sim \mathcal{U}([0,1])$ 的函数。推理时从纯噪声出发，用 DDIM 或祖先采样器（ancestral sampler）迭代去噪即可得到图像。

文本条件方面，本文沿用 Imagen 的做法：用 SentencePiece 分词器 $f$ 把提示词 $P$ 切成 Token 序列，再由 T5-XXL 语言模型 $\Gamma$ 编码为条件嵌入 $c = \Gamma(f(P))$。这个细节后面很关键—— **标识符的选取正是围绕分词器的词表来设计的** 。

值得一提的是，方法同样适用于 Stable Diffusion 这类潜空间扩散模型：只训练 U-Net（可选连同文本编码器），解码器保持冻结。

## 方法：把主体"植入"模型的输出域

方法整体分三步：给主体分配一个罕见标识符 → 用 "a [V] [类名]" 格式的提示词微调模型 → 用先验保持损失防止语言漂移。下面我们逐个拆解，重点讲"为什么这么做"。

### 第一步：给主体起个"罕见代号"

目标是把一个新的（ **唯一标识符**，主体）键值对植入模型的"词典"。所有输入图片统一标注为 "a [identifier] [class noun]" 这种极简格式——例如 "a [V] dog"。这里 [V] 是唯一标识符，[class noun] 是粗粒度类名（如 dog、cat、watch）。

**为什么不用现成的英文单词（比如 "unique"、"special"）？** 因为这些词在模型里已有强语义先验，模型必须先"忘掉"原义再"重学"新义，事倍功半。

**那直接拼接随机字符（比如 "xxy5syt00"）行不行？** 也不行——分词器很可能把每个字母单独切分，而单个字母在扩散模型里同样有很强的先验，效果和用常见单词差不多。

作者的方案很巧妙： **在词表里反向查找稀有 Token** 。具体做法是：在 T5-XXL 分词器词表中做稀有 Token 查找，取 Token ID 落在 $\{5000, ..., 10000\}$ 区间、且对应不超过 3 个 Unicode 字符（不含空格）的 Token，均匀随机采样得到 Token 序列 $f(\hat{V})$（长度 $k = 1 \sim 3$ 效果都好），再用去分词器把它逆转回文本字符串 $\hat{V}$。这样得到的标识符在语言模型和扩散模型两侧都几乎没有先验，是一张干净的"白纸"。

**为什么一定要带类名？** 类名的作用是把模型已有的"类先验"（狗会跑、会坐、有各种姿态）拴到我们的特定主体上。有了类先验，模型才能生成训练照片里从未出现过的姿态和动作。消融实验证明（见下文表格）：用错类名会出现"圆柱形背包"这类怪异结果；不用类名则训练难以收敛。

### 第二步：防止模型"忘本"——先验保持损失

直接全量微调（这是达到最高主体保真度的必要选择，包括文本条件相关的层）会带来两个副作用：

- **语言漂移（Language Drift）**：微调后模型把类名和你的主体绑死了。输入 "a dog"，它生成的全是你的那一只狗——模型"忘记"了世界上还有别的狗。据作者所述，这是首次在扩散模型中发现并命名这一现象。
- **输出多样性下降**：模型过拟合到训练照片的姿态和视角，生成结果趋同。

作者的对策是一个 **自生的、类特异的先验保持损失**（class-specific prior preservation loss，简称 PPL）：在微调的同时，用冻结的预训练模型自己生成约 1000 张 "a [class noun]" 的样本 $x_{pr}$，让微调中的模型同时去拟合这些"自己人"样本：

$$
\mathbb{E}_{x, c, \epsilon, \epsilon', t}\left[ w_t \left\| \hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x \right\|_2^2 + \lambda w_{t'} \left\| \hat{x}_\theta(\alpha_{t'} x_{pr} + \sigma_{t'} \epsilon', c_{pr}) - x_{pr} \right\|_2^2 \right]
$$

第二项就是 PPL，$\lambda$ 控制其权重。这个设计的聪明之处在于 **监督信号来自模型自身** ——相当于让模型一边学新知识，一边"复习"自己原本对"狗"这个类的全部理解，不需要任何外部数据。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/prior_ablation_1.png)

> 图解：PPL 对生成多样性的作用。朴素微调会过拟合到输入照片的背景与姿态（生成图几乎复刻训练集构图）；加入 PPL 后，模型能在保持主体身份的同时生成更多样的姿态、动作与环境，过拟合明显缓解。

### 训练细节与超参

- 约 1000 次迭代，$\lambda = 1$，输入图片 3-5 张。
- 学习率：Imagen 用 $10^{-5}$，Stable Diffusion 用 $5 \times 10^{-6}$。
- 训练开销极小：Imagen 在一块 TPUv4 上约 5 分钟，Stable Diffusion 在一块 A100 上约 5 分钟。

这个成本意味着个性化定制真正做到了"平民化"——几分钟、几张照片，就能拥有一个专属生成模型。

## 实验验证

### 数据集与评测协议

作者构建了一个包含 30 个主体的数据集（21 个物体 + 9 个活体宠物），图片来自作者自行拍摄或 Unsplash；配套 25 条评测提示词（物体：20 条换场景 + 5 条属性修改；宠物：10 条换场景 + 10 条配饰 + 5 条属性修改）。每个主体每条提示词生成 4 张图，共 **3000 张** 评测图。数据集和评测协议已公开。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/dataset.png)

> 图解：数据集中 30 个主体的示例图，涵盖背包、毛绒玩具、猫狗、太阳镜、卡通形象等，物体与宠物两类均有覆盖。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/prompts.png)

> 图解：用于评测的全部提示词，上半部分对应物体类主体，下半部分对应活体和宠物类主体。

评测指标有两个维度：

- **主体保真度**：CLIP-I（生成图与真实图的 CLIP 嵌入余弦相似度）和作者主推的 **DINO** 指标（ViT-S/16 DINO 嵌入的余弦相似度）。
- **提示词保真度**：CLIP-T（提示词与图像 CLIP 嵌入的余弦相似度）。

为什么更信任 DINO？CLIP 是用图文对训练的，编码的是"文本里会提到的信息"，对区分"两只不同的黄色闹钟"这类细粒度差异并不敏感；而 DINO 是自监督训练的，目标本身就是"在数据增强不变的意义下区分不同图像"，天然对个体独有特征敏感。作者用数据验证了这一点：DINO 分数与人类偏好的 Pearson 相关系数为 0.32（CLIP-I 为 0.27），p 值低至 $9.44 \times 10^{-30}$。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/dino_metric.png)

> 图解：CLIP-I 与 DINO 指标的对比案例。四列分别是参考真实图、另一张真实图、DreamBooth 生成图、Textual Inversion 生成图。DreamBooth 的图明显更像参考主体，但其 CLIP-I 分数反而低于 Textual Inversion；DINO 分数则与人眼判断一致——DreamBooth 更高。

### 定量对比与用户研究

与同期工作 Textual Inversion 的定量对比：

| 方法 | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
| --- | --- | --- | --- |
| Real Images（上限参考） | 0.774 | 0.885 | N/A |
| DreamBooth (Imagen) | **0.696** | **0.812** | **0.306** |
| DreamBooth (Stable Diffusion) | 0.668 | 0.803 | 0.305 |
| Textual Inversion (Stable Diffusion) | 0.569 | 0.780 | 0.255 |

DreamBooth 在两项保真度上都大幅领先 Textual Inversion，Imagen 版本甚至逼近真实图片的主体保真度上限（0.696 vs 0.774）。

用户研究更有说服力：72 名用户、每人回答 25 道对比题、共 1800 份答案（每道题由 3 人多数投票）。结果如下：

| 方法 | 主体保真度偏好 ↑ | 提示词保真度偏好 ↑ |
| --- | --- | --- |
| DreamBooth (Stable Diffusion) | **68%** | **81%** |
| Textual Inversion (Stable Diffusion) | 22% | 12% |
| 无法判断 | 10% | 7% |

这张表也提醒我们：定量表里约 0.1 的 DINO 差距和 0.05 的 CLIP-T 差距，映射到人类感知上就是压倒性的偏好差异。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/textual_inversion.png)

> 图解：与 Textual Inversion 的定性对比（输入 4 张花瓶照片）。从上到下：DreamBooth (Imagen)、DreamBooth (Stable Diffusion)、Textual Inversion。四个提示词分别是雪地、海滩、丛林、埃菲尔铁塔背景四个场景。DreamBooth 在主体细节（花纹、瓶口形状）和场景贴合度上都明显更好。

作者还与"极致提示词工程"做了对比：针对一个特征独特的闹钟，给 DALL-E 2 和原版 Imagen 反复尝试后定下高度描述性的提示词，结果它们依然画不出表盘上独立的黄色数字 3，且场景词会"渗入"主体外观（比如场景提到蓝色布料，闹钟上的数字 3 就变成了蓝色）。DreamBooth 则稳定保留形状、字体和数字 3 等细节。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/comparison.png)

> 图解：与 DALL-E 2 / 原版 Imagen 加详细提示词的对比。即使提示词已精确到"右下角有黄色数字 3"，基线模型仍频繁出错，并出现场景向外观的"渗漏"；DreamBooth 在各场景下都保持了闹钟的关键特征。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/comparison_gal.png)

> 图解：使用 Textual Inversion 原文提供的主体、图片与提示词做的直接对比。DreamBooth 能生成语义正确的变体，同时对主体特征（如猫咪雕塑的细节纹路）的保持程度更高。

### 消融实验

**PPL 消融**。在 15 个主体上对比有无 PPL。作者定义 PRES 指标（随机类内样本与本主体的 DINO 相似度，越高说明先验坍缩越严重）和 DIV 指标（同主体同提示词生成图之间的 LPIPS 多样性）：

| 方法 | PRES ↓ | DIV ↑ | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
| --- | --- | --- | --- | --- | --- |
| DreamBooth w/ PPL | **0.493** | **0.391** | 0.684 | 0.815 | **0.308** |
| DreamBooth w/o PPL | 0.664 | 0.371 | **0.712** | **0.828** | 0.306 |

结论清晰：PPL 显著压制语言漂移（PRES 从 0.664 降到 0.493）、提升多样性，代价是主体保真度轻微下降——一个划算的交换。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/ablation_finetuning.png)

> 图解：PPL 对类先验的保护作用。原版模型输入 "a dog" 能生成多种多样的狗；不带 PPL 朴素微调后，输入 "a dog" 生成的全是训练的那只狗（语言漂移）；带 PPL 的模型既能用 "a [V] dog" 生成特定主体，又能用 "a dog" 生成多样的普通狗。

**类名消融**（5 个主体）：

| 设置 | DINO ↑ | CLIP-I ↑ |
| --- | --- | --- |
| 正确类名 | **0.744** | **0.853** |
| 不给类名 | 0.303 | 0.607 |
| 错误类名 | 0.454 | 0.728 |

不给类名时模型失去了类先验的依托，几乎学不动；给错类名则主体与先验互相打架（圆柱形背包就是这么来的）。这印证了"标识符 + 类名"设计的必要性。

## 应用场景：一个模型，N 种玩法

微调完成后，只需把 [V] 嵌入不同句式，就能解锁一系列此前"不可能的任务"。

**换场景（Recontextualization）**：提示词 "a [V] [类名] [场景描述]"，主体能以新姿态、新动作融入从未见过的场景，且接触、阴影、反射都很真实。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/gallery_1.png)

> 图解：多个主体在不同环境中的再情境化生成。注意场景与主体的交互（接触面、光影）非常自然，主体细节高度保留。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/gallery.png)

> 图解：背包、花瓶、茶壶三个主体的更多换场景样本，每张图下方标注了对应的条件提示词。

**新视角合成**：只见过猫的 4 张正面照，模型却能生成俯视、仰视、侧面、背面视角，连额头上复杂的毛发花纹都保持一致——这是类先验外推能力的直接体现。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/novel_views.png)

> 图解：文本引导的视角合成。从左到右依次是俯视、仰视、侧面、背面视角，姿势与训练照片不同，背景随视角变化合理改变，猫额头的复杂花纹得到保留。

**艺术风格演绎**：提示词 "a painting of a [V] [类名] in the style of [画家]"。与风格迁移不同，这里不是"保留结构换风格"，而是生成该画家风格下的全新构图与姿势（如米开朗基罗风格的狗是训练集中从未出现的姿势）。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/original_art.png)

> 图解：一只狗实例在多位著名画家风格下的艺术演绎，许多姿势在训练集中并不存在，且风格模仿相当到位。

**表情操控**：生成训练照片里没有的表情，从消极到积极、不同唤醒度都有，且狗的面部标志性特征（不对称的白色条纹）始终保留。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/expression.png)

> 图解：对狗实例的表情操控。各种表情均不出现在输入图像中，展示了模型的外推能力；注意狗脸上独特的白色条纹在所有生成图中保持一致。

**配饰穿戴**：提示词 "a [V] [类名] wearing [配饰]"，可以给松狮犬穿警服、厨师服、女巫装，配饰与身体的接触和遮挡关系处理得很真实。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/accessories.png)

> 图解：给狗穿戴各种配饰。主体身份在所有图中保持一致，服装与身体的贴合、关节活动都很自然。

**属性修改**：改颜色（"a [颜色] [V] car"），甚至跨物种融合（"a cross of a [V] dog and a [目标物种]"）——狗脸的独特特征在物种改变后依然保留并自然融合。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/property_mod.png)

> 图解：属性修改示例。第一行是汽车变色；第二行是特定松狮犬与不同物种的"混血"，狗的身份特征被保留并与目标物种融合。

**漫画生成**：论文给出了据其所知首个由生成模型产出的"角色一致"完整漫画——每一格都用描述性提示词生成，主角形象全程一致。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/comic.png)

> 图解：生成的完整漫画。角色在各格之间保持一致，每格由类似 "a [V] cartoon grabbing a fork and a knife saying 'time to eat'" 的描述性提示词生成。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/applications.png)

> 图解：新视角合成、艺术演绎与属性修改的综合展示，主体身份在各种强语义修改下依然可辨。

## 附录深挖：几个影响效果的工程细节

**多少张输入图最合适？** 作者对两个主体分别用 1-5 张图各训练 5 个模型：

| 主体保真度 (DINO) | 1 张 | 2 张 | 3 张 | 4 张 | 5 张 |
| --- | --- | --- | --- | --- | --- |
| Backpack | 0.494 | 0.515 | 0.596 | **0.604** | 0.597 |
| Dog | 0.798 | 0.851 | 0.871 | **0.876** | 0.864 |

| 提示词保真度 (CLIP-T) | 1 张 | 2 张 | 3 张 | 4 张 | 5 张 |
| --- | --- | --- | --- | --- | --- |
| Backpack | 0.798 | 0.851 | 0.871 | **0.876** | 0.864 |
| Dog | 0.646 | 0.683 | 0.734 | **0.740** | 0.730 |

结论是 4 张最优，实践中 3-5 张即可。常见主体（如柯基）1-2 张就能抓住外观，罕见物体（如特定背包）则需要更多样本。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/number_images.png)

> 图解：输入图片数量的影响。柯基 2 张图即可较好重建，而罕见的背包至少需要 3 张才能在多样场景中保住主体特征。

**超分辨率模型也要微调，且要降低噪声增强强度**。Imagen 的级联结构中，SR 模型负责照片级细节。如果 SR 模型不微调，会对主体的高频细节产生幻觉；但如果沿用 Imagen 原训练的噪声增强强度 $10^{-3}$，又会把高频纹理抹糊。作者把 $64 \times 64 \to 256 \times 256$ SR 模型微调时的噪声增强降到 $10^{-5}$，细节保留显著改善；$256 \times 256 \to 1024 \times 1024$ 一级则只对细节极丰富的主体有收益。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/low_level_noise.png)

> 图解：SR 模型微调消融。上：低噪声增强微调，细节清晰；中：沿用原噪声强度 $10^{-3}$，高频纹理模糊；下：不微调 SR 模型，主体表面出现幻觉纹理。

## 局限与失败模式

作者诚实地展示了三类失败情况：

- **罕见场景生成失败**：当提示的场景本身先验很弱，或主体与场景在训练数据中共现概率低时，环境生成会出错。
- **场景-外观纠缠**：场景描述会"污染"主体外观，比如蓝色布料场景让背包变色。
- **过拟合原图**：当提示词与训练照片的原环境接近时，模型倾向直接"复刻"训练图。

![Figure](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/DreamBooth-Fine-Tuning-Text-to-Image-Diffusion-Models-for-Subject-Driven-Generation/figures/limitations_1.png)

> 图解：三类失败模式。(a) 罕见上下文环境生成错误；(b) 场景与主体外观纠缠（背包颜色被场景改变）；(c) 提示词接近训练环境时过拟合、生成与训练集相似的图。

此外还有两点：不同主体的学习难度差异大（猫狗容易、罕见物体难）；生成图偶尔会出现幻觉出的主体特征，取决于基础模型先验的强弱和语义修改的复杂度。

## 总结

- **问题定位准**：大型文生图模型有"类"的语义先验，却没有"个体"的词条，DreamBooth 通过微调把特定主体绑定到唯一标识符上，补上了这块短板。
- **标识符设计巧**：在分词器词表中反查稀有 Token 再逆转回文本，避开语言和视觉两侧的既有先验。
- **PPL 是点睛之笔**：用模型自己生成的类内样本做监督，同时解决语言漂移和多样性坍缩，且完全自生、无需外部数据。
- **成本极低**：3-5 张图、约 1000 步迭代、单卡 5 分钟，让个性化文生图真正可用。
- **证据链完整**：新数据集 + DINO 新指标 + 定量对比 + 72 人用户研究，全面压制 Textual Inversion 和提示词工程基线。

展望来看，DreamBooth 开创了"个性化生成先验"这一方向，后续的 Textual Inversion 改进、LoRA 化微调、多主体组合生成等工作都建立在它的范式之上；但场景-外观纠缠与罕见主体的保真度问题，至今仍是开放挑战。

> 本文参考自 [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)