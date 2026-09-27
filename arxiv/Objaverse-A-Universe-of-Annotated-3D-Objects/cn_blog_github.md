# Objaverse：80 万 3D 物体的数据宇宙

2D 视觉的爆发靠的是 ImageNet、LAION 这样的大规模数据集，而 3D 领域至今最大的数据集 ShapeNet 只有 5 万个物体——这正是 3D 生成、具身智能等方向被“卡脖子”的根源。这篇 CVPR 2023 的工作推出了 **Objaverse 1.0**：一个包含 **818K 个高质量 3D 模型**、由 **160K 位艺术家** 创作的超大规模 3D 资产库，每个物体都配有标题、标签、描述等丰富标注。它不是纸上谈兵：作者用它训练 3D 生成模型，人类评测中生成多样性 **91% 碾压 ShapeNet**；用它做 2D 实例分割增强，LVIS 长尾类别 AP 再创新高；还首次把物体导航的目标类别从约 20 个扩到 **1.1K 个**，并构建了一个揭示 CLIP“视角偏见”的鲁棒性基准。

## 提出问题：3D 领域缺一个自己的 "LAION"

回顾近几年的 AI 突破，背后几乎都是数据规模的跃迁：Web 文本语料催生了 GPT-3，图文对数据催生了 CLIP 和 Stable Diffusion，YouTube 视频催生了视频理解模型。这些飞跃的共同路径，是从“人工精选的小数据集”走向“利用 Web 上的海量创作内容”。

反观 3D 领域，画风完全不同：

- 训练 3D 生成模型（如 GET3D）所用的 3D 资产，最多只有 **数千个**；
- 具身智能模拟器（AI2-THOR、Habitat 等）通常只有几十到一千个场景；
- 现有 3D 数据集普遍面临“规模、多样性、真实感”不可兼得的窘境。

笔者觉得，这篇论文的立论很直白但很有力：2D 和 3D 研究长期“各玩各的”，不是因为问题天然割裂，而是因为 3D 缺少大规模数据。一旦数据补上，很多跨模态的问题自然会被解锁。

下表是论文给出的 3D 数据集规模对比，差距一目了然：

| Dataset | 物体数 | 类别数 |
| --- | --- | --- |
| YCB | 77 | 5 |
| BigBIRD | 125 | -- |
| KIT | 219 | 145 |
| Pix3D | 395 | 9 |
| GSO | 1K | 17 |
| PhotoShape | 5K | 1 |
| ABO | 8K | 63 |
| 3D-Future | 10K | 34 |
| ShapeNet | 51K | 55 |
| **Objaverse** | **818K** | **21K** |

![Objaverse 与现有 3D 数据集对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/shapenet14.png)

> 图解：左图对比了 Objaverse 与 ShapeNet 在 Car、Bed、Vase、Bag 四个类别上的实例。ShapeNet 的模型全部来自面向建筑草图的 SketchUp 平台，风格趋同；Objaverse 来自各种 3D 创作平台，实例间差异明显大得多。右表是规模对比，Objaverse 的物体数是 ShapeNet 的 16 倍，类别覆盖（以 WordNet 实体估计）更是数量级的领先。

## 数据集解剖：Objaverse 里到底有什么

### 来源与基本盘

Objaverse 的物体全部来自 Sketchfab——一个允许用户上传、分享 3D 模型的在线平台。入选条件有两条：必须是可分发的 Creative Commons 许可；被平台标记为不良或成人内容的模型被排除在外。

![Objaverse 示例物体](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/annotations.png)

> 图解：Objaverse 中的示例实例。可以看到物体在语义上极其多样——从日常用品到奇幻生物——且每个资产都配有自然语言描述，这是以往 3D 数据集几乎没有的特性。

### 模型元数据：每个物体自带“简历”

每个物体都继承了创作者上传时填写的元数据：名称、固定类别、自由标签、自然语言描述，以及缩略图和统计信息。这意味着 Objaverse 不只是几何体的集合，而是一个 **3D-语言配对** 的数据库。

![元数据示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/metadata-3.png)

> 图解：一个物体的完整元数据页面，包括 3D 模型、缩略图、名称、描述、标签和类别。这些文本信息正是后续做 text-to-3D、语言引导导航等任务的基础。

### Objaverse-LVIS：47K 物体的精细分类子集

Sketchfab 自带的分类只有 18 个大类，对多数视觉任务来说太粗了。作者选用 LVIS 数据集的类别体系，人工构建了一个 47K 物体的子集 **Objaverse-LVIS**，每个物体被唯一分配到 1156 个 LVIS 类别之一。

标注流程是个“两阶段漏斗”：先用 CLIP 对缩略图做视觉分类预测，再结合元数据中术语的 GloVe 相似度，为每个类别圈出 500 个候选物体（共 250K 个）；然后交给 Amazon Mechanical Turk 的众包工人逐一核实。这个设计的聪明之处在于：CLIP 能捞回“长得像但元数据缺失”的物体，文本相似度能捞回“长得怪但名字写得准”的物体，两者互补，避免单一信号造成的系统性遗漏。

![类别标注界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/category_annotation.png)

> 图解：Objaverse-LVIS 的众包标注界面。工人每次看 9 个候选物体，勾选出属于目标类别的实例；除了缩略图，还能看到物体名称辅助判断。

### 超越静态物体：动画、角色、室内外场景

Objaverse 的“宇宙”感还体现在内容的广度上：

- **44K 个动画物体**：冰箱门开合、动物奔跑、钟表指针转动等，可支撑时序 3D 学习（如动态 NeRF、文本生成动画）；
- **63K+ 绑定骨骼的角色**：带骨骼映射，可直接用于动画和渲染；
- **可拆解的关节物体**：很多艺术家上传的模型天然分部件，比如椅子可以拆成靠背、轮子、椅腿，利好机器人抓取和部件级分割研究；
- **大量室外扫描**：包括纽约天际线这样的城市级扫描模型；
- **16K+ 室内场景（Objaverse-Interiors）**：多楼层、多房间、物体密集。作为对比，现有手工搭建的交互式具身 AI 场景总共只有约 400 个——Objaverse 直接把这个数字提升了两个数量级；
- **风格跨度极大**：从哥特式、维多利亚式到卡通、抽象，从 3D 扫描到 PBR 写实渲染。同一类别（比如椅子）存在大量风格变体，这对训练鲁棒视觉模型至关重要。

![视觉多样性示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/fig2.png)

> 图解：Objaverse 的视觉多样性一览，涵盖动画物体、绑定骨骼的角色、可拆部件的模型、室外环境、室内场景，以及多种视觉风格下的椅子（哥特、现代、维多利亚、卡通、抽象）。

### 统计画像

Objaverse 1.0 的核心数字：**818K 个物体**、**160K 位艺术家**、**235 万+ 个标签**（其中 17 万+ 是唯一的）、物体上传时间跨度为 2012–2022 年（仅 2021 年就新增 20 万+）。

![数据集统计](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/categories.png)

> 图解：Objaverse 元数据中 18 个高层类别的分布。头部类别的占比相当均匀，长尾类别数量较少——整体分布比想象中均衡。

![标签词云](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/tag_cloud.png)

> 图解：Objaverse 热门标签词云（按频率对数缩放）。"furniture" "car" 这类高频词符合现实世界的常识，但 "sword"（剑）的高频出现则明显偏离现实分布——艺术家爱造剑，这类有趣的分布偏移在使用数据时值得留意。

一个有意思的问题：80 万物体到底覆盖了世界上多少“种”东西？作者用 CLIP ViT-B/32 做了个估算：对每个物体的缩略图提取图像 Embedding，与所有 WordNet 实体的文本 Embedding 算余弦相似度，取最近者作为该物体的实体。文本端采用 CuPL 风格的模板 `"a {entity} is a {definition}"`，利用 WordNet 的释义消歧——例如 "bat" 既可以是“蝙蝠（会回声定位的哺乳动物）”也可以是“球棒（击球用的棍）”。最终估算覆盖约 **20.8K 个 WordNet 实体**，这个数字就是上表中 "# Classes = 21K" 的来源。

## 应用一：3D 生成建模——多样性 91% 碾压 ShapeNet

数据集好不好，拉出来练一练才知道。作者接下来的四个应用实验，分别回答“3D 生成能否受益”“2D 任务能否受益”“具身智能能否受益”“能挖出什么新发现”四个问题。

第一个实验选了当时的 SOTA 3D 生成模型 GET3D，在三个类别上训练：Shoe（143 个物体）、Bag（816 个）、Fruit&Veg（571 个，含 116 个苹果、92 个蘑菇、68 根香蕉等 9 个品种）。对照组是用 ShapeNet 全部 83 个 Bag 训练的同款模型。

![Bag 生成对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/shapenet-single-col.png)

> 图解：分别用 Objaverse（上）和 ShapeNet（下）训练的 GET3D 生成的“包”。Objaverse 版本在造型、颜色、风格上明显更多样，ShapeNet 版本则高度雷同。

![Shoe 与果蔬生成](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/fv2.png)

> 图解：Objaverse 训练的模型生成的鞋子和水果蔬菜样本。尤其果蔬模型（9 个品种混合训练）质量最高，预示着多类别联合训练 + 文本条件的 text-to-3D 大有可为。

![果蔬插值](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/fv1.png)

> 图解：在果蔬模型的隐空间中做插值，一个南瓜平滑地变形为蘑菇——说明模型学到的隐空间是连续且有语义的。

定量验证也很直接：把两个模型各自随机生成的 9 个包摆在一起，让众包工人判断哪组更多样。结果 Objaverse 版本以 **91%** 的压倒性比例被认为更多样。

![多样性评测界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/diversity_task_inteface.png)

> 图解：多样性对比评测界面。左右各展示一组 9 个生成样本（随机排列左右位置），工人选择外观变化更丰富的一组。

## 应用二：CP3D——用 3D 资产给 2D 分割“加餐”

3D 数据帮 3D 任务顺理成章，但能否反哺 2D？第二个实验给出了肯定答案。

目标任务是 LVIS 大规模实例分割：1230 个类别、164K 张图，难点在于类别分布的长尾——大量类别在整个数据集里平均只有 9 个实例。实例分割标注（勾勒轮廓）极其昂贵，而 3D 资产的标注几乎是免费的：渲染一下就同时得到图像和精确 mask。

作者提出 **3DCP（3D Copy-Paste）**，在经典 Copy-Paste 增强的基础上把 2D 贴纸换成 3D 渲染图：

- 对 Objaverse-LVIS 中每个物体渲染 5 个不同视角并缓存；
- 训练时以 0.5 的概率选中一张图做增强，随机挑 1–3 个 3D 物体的渲染图贴上去，随机缩放、平移；
- 同时把物体的 mask 加入训练标注。

![3DCP 示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/cp3d.png)

> 图解：3DCP 增强示意。把 3D 物体从多个视角渲染出来，粘贴到 LVIS 训练图像上，同时获得像素级精确的 mask——相当于“免费”获得长尾类别的训练样本。

模型侧直接复用 GOL 的 ResNet-50 Mask-RCNN 预训练权重（其用 Gumbel 激活 $\eta(q) = \exp(-\exp(-q))$ 替代 softmax），以 batch size 64、学习率 0.002 微调 24 个 epoch，不改任何架构。

结果如下表，GOL + 3DCP 全面超越原版 GOL，刷新 LVIS 上的 SOTA：

| Method | AP | APr | APc | APf |
| --- | --- | --- | --- | --- |
| RFS | 23.7 | 13.3 | 23.0 | 29.0 |
| EQLv2 | 25.5 | 17.7 | 24.3 | 30.2 |
| LOCE | 26.6 | 18.5 | 26.2 | 30.7 |
| NorCal w/ RFS | 25.2 | 19.3 | 24.2 | 29.0 |
| Seesaw | 26.4 | 19.5 | 26.1 | 29.7 |
| GOL | 27.7 | 21.4 | 27.7 | 30.4 |
| **GOL + 3DCP** | **28.3** | **21.8** | **28.3** | **31.1** |

（APr/APc/APf 分别衡量稀有、常见、高频类别的 AP。）附录中的检测指标同样亮眼：bbox AP 从 27.5 升到 28.9，其中稀有类别 APr 从 19.8 涨到 21.8，**一口气提升 2 个点**。笔者认为这个实验的分量在于：它证明了 3D 资产不是 3D 社区的“自留地”，而是可以低成本反哺成熟 2D 任务的通用资源。

## 应用三：开放词汇物体导航——目标类别 50 倍扩容

第三个应用把 Objaverse 搬进了具身智能。传统 ObjectNav 任务中，智能体只需导航到约 20 种目标（给一个类别标签），而模拟器里的物体总共也就 2K 个、百余类。Objaverse 让作者能一步到位提出 **开放词汇 ObjectNav**：目标用任意自然语言描述（如 "a victorian-monobike motorcycle"、"a unicorn pony"），目标类型扩到 1.1K 类，模拟环境中的可用物体从 2K 暴涨到 36K。

### 把 Objaverse 装进模拟器：三个工程细节

数据集直接塞进模拟器是行不通的，作者的预处理三板斧值得记录：

1. **摆放约束标注**：为每个 LVIS 类别标注物体通常出现在地板、台面还是墙面；地板物体再细分“放中间”（如篮球）还是“贴边”（如马桶、冰箱）；同时自动检测物体 mesh 顶部的平坦区域，用于承托台面物体。飞机这类不该进屋的物体被直接过滤。
2. **尺寸矫正**：Sketchfab 上的模型尺寸常常离谱（植物比塔还高）。作者为每个类别标注最大包围盒边长（如书架 2 米、叉子 0.18 米），超限的物体按比例缩放。例如一个 20m×6m×3m 的“书架”，缩放因子为 $\max(20, 6, 3) / 2 = 5$。
3. **运行时加载与压缩**：以往 AI2-THOR 要求所有物体打包进 Unity build，面对几万物体不可行。作者为 AI2-THOR 增加了运行时动态加载能力，并用 Blender 做预处理：合并 mesh、顶点数压到 5K 以内、烘焙成单张纹理、用 V-HACD 生成碰撞体以支持刚体交互。

### 任务设置与模型

作者用 ProcTHOR 程序化生成 **10,080 栋房屋**（每栋至多 3 个房间），全部用 Objaverse-LVIS 物体填充（门窗等结构件沿用 ProcTHOR 原件）。训练目标覆盖 **262 个类别、9,421 个不同物体**。测试则在 151 栋没见过的房屋里进行，30 个测试类别各采 150 个 episode，共 4,500 个。

智能体是模拟的 LoCoBot，动作空间只有 6 个：MoveAhead、RotateLeft、RotateRight、End、LookUp、LookDown。模型沿用 EmbCLIP 架构，但把原先“学出来的目标类别 Embedding”换成了 **CLIP 文本分支输出的线性投影**——这正是“开放词汇”的关键改动，目标描述不再受限于固定类别集合。同时视觉与目标两种模态的内部表征维度从 32-D 提升到 256-D，以容纳更丰富的目标信息。视觉输入由冻结的 CLIP ResNet-50 编码，整个策略用 DD-PPO 训练（28 张 GPU、Adam、学习率 $3 \times 10^{-4}$、折扣因子 $\gamma = 0.99$、GAE $\lambda = 0.95$）。

![开放词汇导航模型](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/openvocab_model_with_actions.png)

> 图解：开放词汇 ObjectNav 模型结构。RNN 策略网络接收视觉编码、目标文本编码、上一时刻的隐状态和动作，输出下一个动作。目标描述经由 CLIP 文本分支编码，因此任意自然语言描述都能作为目标。

### 结果

仅训练 **18M 步**，策略就取得 **19.9%** 的成功率（随机策略为 5.1%）；延长训练到约 **460M 步** 后，成功率进一步提升至 **33.0%**。考虑到目标类别从 20 扩到 1.1K、房屋和目标组合全是未见过的，这个成绩相当能打——而且明确了“继续加训练量就能继续涨”的 scaling 信号。

## 应用四：视角鲁棒性基准——SOTA 模型的“阿喀琉斯之踵”

最后一个应用反过来用 Objaverse 给现有模型“挑刺”。ImageNet 等 2D 数据集存在根深蒂固的偏见：物体几乎都是从正面标准视角拍摄的（没人会趴在地上从电视机背后拍照）。Alcorn 等人的研究早已发现，现代视觉系统一旦偏离标准姿态，性能就会暴跌——这对自动驾驶等安全攸关的场景是实打实的隐患。

过去这类评测受限于真实拍摄的代价，只能小规模进行。而有了 3D 资产，**任意视角渲染成了免费操作**：作者对 Objaverse-LVIS 中每个物体从 12 个随机朝向渲染图像（背景填充为 ImageNet 的均值 RGB），构建了一个长尾类别的视角鲁棒性基准，然后零样本评测多个 CLIP 模型（类别限定在约 1200 个 LVIS 类）。

![随机视角渲染示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Objaverse-A-Universe-of-Annotated-3D-Objects/assets/robustness4.png)

> 图解：从随机朝向渲染的物体示例及 CLIP ViT-B/32 的零样本分类结果。非标准视角下，分类错误明显增多。

评测指标除了常规的 Top-1/Top-5（Random Rotation），还设计了一个诊断性指标 **Any Rotation**：只要模型在 12 个视角中至少 1 次分类正确就算对——它近似代表“物体处于标准姿态时”的模型水平。两者之差 $\Delta$Top-1 就是视角敏感度的度量。

| Model | Random Top-1 | Random Top-5 | Any Top-1 | Any Top-5 | $\Delta$Top-1 |
| --- | --- | --- | --- | --- | --- |
| OpenAI-400M RN50 | 21.4% | 45.0% | 43.9% | 70.8% | 22.5% |
| OpenAI-400M ViT-L/14 | 29.1% | 54.5% | **52.3%** | 77.2% | 23.2% |
| LAION-400M ViT-B/32 | 24.1% | 48.5% | 46.9% | 74.2% | 22.8% |
| LAION-400M ViT-L/14 | 30.6% | 56.8% | 50.5% | 77.0% | 19.9% |
| LAION-2B ViT-B/32 | 27.0% | 51.8% | 50.3% | 76.1% | 23.3% |
| LAION-2B ViT-L/14 | **32.9%** | **59.2%** | 52.1% | **78.0%** | 19.2% |
| LAION-2B ViT-H/14 | 32.3% | 58.8% | 50.1% | 77.3% | **17.8%** |

结论相当扎心：所有模型的 $\Delta$Top-1 都在 **18–23 个百分点**，也就是说同一物体换个朝向，正确率直接腰斩。模型规模变大（ViT-H/14 的 17.8%）能缓解但不能消除这个问题。这个基准只有在 Objaverse 这种规模与多样性的 3D 数据上才建得起来——真实世界拍不出来，2D 图像也生成不出来。

## 总结

- **数据规模断层补齐**：Objaverse 1.0 提供 818K 个 3D 物体、160K 位创作者、235 万+ 标签，类别覆盖约 21K 个 WordNet 实体，比 ShapeNet 大 16 倍；
- **标注体系完整**：自带名称/描述/标签元数据，外加 47K 物体、1156 类的 Objaverse-LVIS 人工精标子集，还有 44K 动画物体和 16K+ 室内场景；
- **四大应用验证价值**：GET3D 生成多样性 91% 胜过 ShapeNet；CP3D 把 LVIS 分割推向新 SOTA（AP 28.3）；ObjectNav 目标类别扩容 50 倍至 1.1K，成功率 19.9%（随机 5.1%）；视角鲁棒性基准揭示 CLIP 模型换视角即掉 20 个点；
- **方法论启示**：2D 的“大数据飞轮”逻辑正在 3D 复现——先有 LAION 级别的数据，才有后续的模型爆发。

展望来看，Objaverse 是 1.0 且仍在增长，真正的想象空间在于它开启的方向：大规模 text-to-3D 生成、开放世界具身智能、以及 2D/3D 联合的鲁棒表征学习——这篇论文展示的只是冰山一角。

> 本文参考自 [Objaverse: A Universe of Annotated 3D Objects](https://arxiv.org/abs/2212.08051)