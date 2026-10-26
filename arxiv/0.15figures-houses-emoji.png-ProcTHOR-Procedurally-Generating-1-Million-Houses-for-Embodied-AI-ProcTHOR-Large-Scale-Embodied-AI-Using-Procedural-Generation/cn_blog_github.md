# ProcTHOR：1 小时生成万座可交互房屋

具身智能 agent 常在几十到几百个训练场景里反复练习，结果是记住了房间，却没有学会泛化到陌生房屋。场景不足因此成为导航、操作等任务扩展训练规模的关键瓶颈。ProcTHOR 的思路是把房屋生成拆成户型、结构、光照、物体、材质和状态等条件采样步骤，在 AI2-THOR 中自动搭建可交互、可物理模拟的房屋。作者在 4 张 NVIDIA RTX A5000 GPU 上用 **1 小时生成 10,000 座房屋**；只用 RGB 和简单的 CNN+GRU 模型预训练后，RoboTHOR ObjectNav 微调结果的 SPL 比此前 SOTA 提高 **8.8 个百分点**，并在论文统计时点的 6 项基准上取得领先结果。

![ProcTHOR 生成的多样可交互房屋](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/procthor-cover.jpg)

> 图解：这些房屋不是同一户型换皮，而是在房间结构、家具组合、材质和光照上共同变化。ProcTHOR 的目标正是把这种变化转化为可规模化生成的训练数据。

## 为什么具身智能需要“无限房屋”

视觉和语言模型已经证明了大规模、任务无关数据的价值，但具身智能面对的不只是识别一张图。Agent 要连续观察、移动、规划并改变环境，任务链条更长，也更容易过拟合有限场景。

过去的场景主要来自两条路线：

- **人工建模**：AI2-THOR、RoboTHOR 等环境交互性强，但 3D 美术师需要逐房设计结构、摆放家具、配置材质和灯光，扩展成本高。
- **真实扫描**：HM3D 等数据集外观接近现实，却要实地扫描、拼接和清理；静态扫描通常不能让 agent 拿起物体、开关灯具或改变物体状态。

这形成了一个规模与交互性的取舍：真实扫描容易扩到较多房屋，却难支持操作；人工场景适合交互，却难以扩到成千上万。ProcTHOR 选择程序化生成，就是希望同时保住两边。

与同样使用程序生成的 Megaverse 相比，ProcTHOR 更强调住宅外观、物理和物体交互的复杂度；与生成户型或室内图像的图形学方法相比，它更关心生成结果能否真正用于 embodied task，而不只是“看起来合理”。

## ProcTHOR 总览：一条带约束的房屋流水线

ProcTHOR 继承 AI2-THOR 的资产库、机器人 agent 和物理模拟能力。输入可以只是“1 间卧室连接 1 间浴室，再加厨房和客厅”这样的房间规格，系统随后逐阶段采样完整住宅。

![ProcTHOR 多阶段房屋生成流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/compressions/generation-3-vector.png)

> 图解：流程从房间规格开始，依次采样内部边界与户型、创建墙体结构、加入灯光、采样门和结构材质，再放置大型物体、墙面物体及台面小物体。前一阶段的结果会成为后一阶段的条件，因此不是把所有元素独立随机撒进场景。

这套设计可以概括为五个关键词：**多样性、交互性、可定制性、规模、效率**。其中最重要的不是随机化本身，而是随机化始终受语义和物理约束。

### 户型与资产：随机，但不违背常识

外轮廓从矩形开始，通过切角形成复杂多边形；内部再用递归布局算法划分房间，并按用户约束连接。于是卧室可以邻近浴室，浴室通常只有一个入口，而不是任意房间两两开门。

![ProcTHOR 生成的多样户型](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/floorplans4_1.png)

> 图解：不同颜色代表卧室、浴室、厨房和客厅。相同类型的房间可以组成大量不同户型，房屋规模也从少量房间扩展到复杂的多房间结构。

资产库包含 **108 个类别、1,633 个可交互实例**。多数资产来自 AI2-THOR，作者还补充了 21 扇窗、20 扇门和 33 个台面模型，使这些原本固定在房间结构里的元素也能独立采样。

![ProcTHOR 资产库中的物体实例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/assets2.jpg)

> 图解：同一物体类别拥有多种外形实例，而不是每把椅子、每个沙发都共用同一模型。实例还按 train/validation/test 划分，从资产层面减少评估时的视觉泄漏。

材质层提供 **40 种墙面纯色、122 种墙面纹理和 55 种地板材质**。随机化限制在语义相近的类别内部，避免苹果被换成橙子的外观，或墙面出现不合住宅常识的高饱和颜色。

![房屋结构与物体的材质增强](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/demo3.jpg)

> 图解：墙体、地板和物体可以改变颜色或材质。它增加的是视觉分布的覆盖范围，而不是改变物体类别和交互规则。

### 摆放、光照与状态：让“不同”仍然可执行

物体摆放分为地面、墙面和承载表面三层。沙发可以出现在客厅而非浴室，冰箱应靠墙而不是背对可使用空间，杯子则应放在台面等承载物上。对于餐桌与椅子这类天然共现的组合，作者使用 Semantic Asset Group（SAG，语义资产组）进行依赖采样。

![同一房间布局下的四种物体摆放](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/pos-placement-2.jpg)

> 图解：四个例子保持房间布局不变，只改变物体和摆放位置。这说明多样性不仅来自“换房子”，也来自同一空间中的不同功能布置。

光照由一盏模拟太阳的方向光和多盏模拟灯泡的点光源组成。系统改变光源颜色、强度和位置，并结合不同时段的 skybox，制造清晨、黄昏和夜晚等视觉条件。

![同一场景在清晨、黄昏和夜晚的光照](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/lighting-small2.jpg)

> 图解：同一房屋在三种光照下呈现明显不同的亮度与色温。对仅使用 RGB 的策略而言，这类变化迫使模型学习比固定纹理和固定曝光更稳健的特征。

交互性是 ProcTHOR 与静态扫描场景的根本区别。物体能够被拿起和移动，灯具能开关，笔记本电脑能开合，带机械臂的 agent 还能操作物体并与其他 agent 交互。

![物体状态变化、机械臂操作与多 agent 交互](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/3PanelProcTHOR_03.jpg)

> 图解：左侧面板展示笔记本电脑、灯具等状态变化，中间和右侧展示机械臂操作及多 agent 场景。环境不只是视觉背景，而是会被动作改变的动态系统。

房间、资产、材质和光照规格都可替换，因此生成器并不只限于普通住宅。

![ProcTHOR 定制的教室、图书馆和办公室](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/customizability.jpg)

> 图解：通过更换房间与资产规格，同一生成框架可以构造教室、图书馆和办公室。这说明它更接近环境生成平台，而不是固定的房屋数据集。

解决了“怎样快速造出不同房屋”之后，真正困难的问题变成：怎样保证这些房屋不是随机杂物间，而是可导航、可交互、符合居住常识的训练环境？

## 方法深拆：程序化生成中的关键约束

### 用树状 Room Spec 控制房屋分布

Room Spec 是一棵树，叶节点表示房间，内部节点组织房间组。每个节点可以带 growth weight，近似控制同父节点下各区域的比例；树结构还约束哪些房间之间必须存在通路。

![四房间房屋的 Room Spec 树](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/spec1.png)

> 图解：四房间示例把卧室与浴室组成一个子树，把厨房与客厅组成另一个子树。子树内部和两子树之间都要连通，同时还能避免浴室再开一扇通向公共区域的门。

若希望卧室占“卧室+浴室”区域约 60%，可以把二者 growth weight 设为 3 和 2。ProcTHOR-10K 只使用 **16 套 Room Spec**，每套覆盖 1—10 个房间，再加权采样，便能控制小型单间、公寓和大型住宅在数据集中的比例。

![五房间与七房间 Room Spec 示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/spec2.png)

> 图解：五房间规格沿用相同的层级组织方式，通过增加节点控制更复杂的住宅。

![七房间房屋的层级房间规格](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/spec3.png)

> 图解：七房间规格进一步说明，少量可复用模板并不等同于少量固定户型；模板规定的是语义约束，具体边界和布局仍由采样产生。

### 先切外轮廓，再递归分房

设房间数为 $n_r$，初始边界的两个整数维度 $x_s,z_s$ 通常独立采样自：

$$
x_s,z_s\sim U\left(\max\left(\ell_{\min},\mu_a\sqrt{n_r}-\frac{\mu_a}{2}\right),\mu_a\sqrt{n_r}+\frac{\mu_a}{2}\right)
$$

其中 $\ell_{\min}=2$，$\mu_a=3$。这让初始面积大致随房间数增长，而具体 Room Spec 仍可覆盖该分布。

![矩形边界经过三次切角形成多边形](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/boundaries.png)

> 图解：左侧是初始矩形，右侧是在角落做三次矩形切割后的内部边界。整数网格使边界可用布尔矩阵紧凑表示，也方便后续递归划分。

切角数量 $n_c$ 不是固定值，而是：

$$
n_c\sim\left\lfloor10\cdot\operatorname{Beta}\left(\frac{n_r}{2},6\right)+\frac12\right\rfloor
$$

![不同房间数对应的切角次数概率分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/cuts-2.png)

> 图解：横轴是房间数，纵轴是概率，每组不同深浅的柱表示 0—10 次切角。房间越多，概率质量总体移向更多切角，大房屋因而更常出现复杂轮廓；切角上限为 10。

有了边界后，系统按 Room Spec 的子树递归划分区域，再把整个户型缩放 $s\sim U(1.6,2.2)$。放大能给 agent 留出活动空间，上限 2.2 则避免房屋显得过于空旷。

![按 Room Spec 递归划分房间](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/gen-spec.png)

> 图解：内部边界先被分成“卧室与浴室”和“厨房与客厅”两个区域，再各自细分为具体房间。递归过程同时服务于面积比例与连通约束。

### 门的连接也要符合居住逻辑

相邻房间有三种连接方式：带门扇的 doorway、无门扇的 doorframe，以及完全开放的 open-room。开放连接和门框连接只允许出现在厨房与客厅之间，避免浴室直接暴露在另一房间中。

![带门扇的房间连接](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/doorway.jpg)

> 图解：Doorway 有可开关的门扇，门的位置与开启方向会采样；若两扇门开启后碰撞，系统用 rejection sampling 重新调整。

![无门扇的门框连接](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/doorframe.jpg)

> 图解：Doorframe 保留门框但没有门扇，适合需要连续通行的空间。

![厨房与客厅的开放式连接](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/open-room.jpg)

> 图解：Open-room 取消完整隔墙，仅用于厨房和客厅。厨房—客厅之间采样 doorway、doorframe、open-room 的概率分别为 0.25、0.375、0.375。

结构材质和高度也不是均匀乱选。墙面以 0.5 概率使用纯色，整栋房屋以 0.35 概率让所有房间共用墙材，地板统一材质的概率为 0.15。天花板高度则为：

$$
c_h=h_{\min}+(h_{\max}-h_{\min})\operatorname{Beta}(1.25,5.5),\quad h_{\min}=2.5,\ h_{\max}=7
$$

![ProcTHOR 天花板高度分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/ceiling-distribution-function.png)

> 图解：横轴为高度（米），纵轴为密度。分布在约 2.8 米附近达到峰值，同时保留向更高天花板延伸的长尾，而不是在 2.5—7 米之间均匀采样。

每个房间在地面多边形质心附近放一盏点光源，台灯和落地灯还自带光源。Skybox 覆盖正午、golden hour 和 blue hour，分别有 16、5、1 个候选，来自西雅图和旧金山拍摄的 360° 照片。

![正午、金色时刻与蓝调时刻的天空盒](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/skyboxes/midday-small.jpg)

> 图解：Midday skybox 提供明亮的日间室外光与窗外内容。

![金色时刻天空盒](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/skyboxes/golden-hour-small.jpg)

> 图解：Golden hour 改变环境光的强度、色相和方向，画面呈更暖的日间光照。

![蓝调时刻天空盒](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/skyboxes/blue-hour-small.jpg)

> 图解：Blue hour 进一步转向昏暗、偏冷的光照，配合室内点光源形成不同视觉条件。

### SAG：把“桌椅关系”写进采样器

单独随机一张桌子和四把椅子，很难自然得到一套餐桌。SAG 把共现物体组织成父子结构，每个子物体通过父物体的 anchor 与自身的 pivot 定位，并可附加偏移和旋转随机性。

![由餐桌父节点与椅子子节点构成的 SAG](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/asset-group-3.png)

> 图解：两个椅子 sampler 依附于餐桌 sampler，并锚定在桌边。换成更大或更长的桌子时，椅子会随父物体边界重新定位，而不是死守全局坐标。

这一设计的聪明之处在于，它把“相对语义”与“具体实例”分离：桌椅关系保持不变，桌和椅的模型仍可独立随机。若电视柜、电视、沙发和扶手椅各有 30 个候选实例，一个 SAG 就有 $30^4=810{,}000$ 种实例组合。

相对定位不能完全消除网格穿插，因此还要做碰撞检查。

![因椅子与餐桌碰撞而被拒绝的 SAG](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/rejected-sag.png)

> 图解：椅子与较矮餐桌发生穿插，该实例组合会被拒绝。

![通过碰撞检查后接受的 SAG](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/accepted-sag.png)

> 图解：系统重新采样，直到组内物体不再碰撞才接受。ProcTHOR-10K 使用 18 个 SAG，正文估计可形成约 2,000 万种组合，统计分析中给出的精确值为 1,930 万种。

### 地面、墙面、台面：三层摆放算法

地面摆放先把剩余空间多边形切成矩形。每次以 0.8 概率选最大矩形，否则按面积加权选择其他矩形；再筛掉语义不符、数据划分不符或尺寸放不下的物体。

![地面物体的迭代矩形摆放算法](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/placement.png)

> 图解：系统从空闲区域的角点画出横纵分割线，枚举矩形并采样一个位置。蓝色区域是带安全边距的物体俯视包围盒，放入后从空闲多边形中扣除，再进入下一轮。

对宽 $o_w$、深 $o_h$ 的物体和目标矩形 $r_w\times r_h$，至少一种旋转需满足：

$$
(o_h+w_{\mathrm{pad}}\le r_w\ \land\ o_w+w_{\mathrm{pad}}\le r_h)\ \lor\
(o_h+w_{\mathrm{pad}}\le r_h\ \land\ o_w+w_{\mathrm{pad}}\le r_w)
$$

房间中央物体四周各留 0.35 米边距；墙边和角落物体只在正面留 0.5 米，以便其他物体紧邻摆放，同时保证柜门、冰箱门仍能使用。墙边物体背面朝墙，角落物体有两种合法朝向，中央物体可按 90° 旋转。

每轮地面物体数 $r_i$ 也偏向较充实但不绝对固定的房间：取 1、4、5、6、7 的概率分别为 $1/200$、$2/200$、$4/200$、$20/200$、$173/200$。SAG 内含多个物体，所以实际物体总数可以超过 $r_i$。

地面完成后才处理墙面。窗户只能开在外墙上；厨房、客厅和卧室采样 0、1、2 扇窗的概率分别为 0.125、0.375、0.5。挂画可以出现在任何房间，还能放在高度低于 1.15 米的矮柜上方；每间最多一台挂墙电视，并避开已有电视的房间。

![墙面窗户、挂画与电视的放置示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/wall-objects/wall-window.jpg)

> 图解：窗户的位置按剩余墙段长度加权采样，太短的墙段直接排除，已占用的墙段不会重复利用。

![墙面挂画示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/wall-objects/wall-painting.jpg)

> 图解：挂画放置后只扣除自身宽度，因此同一面墙能容纳多幅画；垂直位置偏向可用高度区间的中部。

![挂墙电视示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/wall-objects/wall-tv.jpg)

> 图解：只有可挂墙且没有外凸底座的电视实例才参与采样，客厅、厨房、卧室的尝试概率分别为 0.8、0.25、0.4。

最后是台面小物体。作者从人工设计的 AI2-iTHOR 和 RoboTHOR 场景统计“某类物体出现在某类承载面上”的频率 $p_{\mathrm{spawn}}$，再加入三级偏置：

$$
p=p_{\mathrm{spawn}}+b_{\mathrm{house}}+b_{\mathrm{receptacle}}+b_{\mathrm{object}}
$$

房屋级偏置让不同住宅呈现不同整洁程度：

$$
b_{\mathrm{house}}\sim0.4\operatorname{Beta}(3.5,1.9)-0.3
$$

![用于控制台面物品多少的房屋偏置分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/bias-dist-2.png)

> 图解：横轴是房屋偏置，纵轴是密度，范围约为 -0.3 到 0.1，峰值靠近 0。偏置越低，台面物体越少，房屋越整洁；它让“凌乱程度”成为整栋房子共享的风格，而非每个台面独立决定。

承载面默认偏置为 0.2，置物架为 0.4，扶手椅和普通椅子为 0；部分目标物体还有独立偏置。每次尝试最多采样 5 个姿态，同一承载面上同类物体最多 3 个，并继续做碰撞检查。

### 训练时随机化，生成后做可达性验证

颜色与材质增强发生在训练 episode 加载时：花瓶、雕像和瓶子各有 0.8 概率随机 RGB 颜色，整个场景也有 0.8 概率随机物体材质，但墙、地板和天花板保持不变。

![花瓶颜色随机化与场景材质随机化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/color-randomization/default-vase.jpg)

> 图解：最左侧为花瓶原始外观，同组另外三种随机色保持物体类别和形状不变，只扩大颜色分布。

![材质随机化后的室内场景](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/procthor-material-randomization/o1.jpg)

> 图解：材质随机化只作用于物体，房屋结构材质不变，避免视觉增强破坏原本的房间风格约束。

状态随机化覆盖灯具开关、床铺整洁/凌乱以及盒子和笔记本电脑的开合程度。它让策略不能把“物体长什么样”误解为固定状态。

![笔记本电脑开合、床铺整洁与灯具开关状态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/open1.jpg)

> 图解：笔记本电脑可在不同开合程度下出现，同类状态变化还覆盖床铺和灯具，增加操作任务的初始条件多样性。

房屋生成后，验证器在 $0.25\times0.25$ 米网格上做 BFS（广度优先搜索）。每个房间至少要有 5 个无需移动物体即可到达的网格位置；失败则沿用同一 Room Spec 重新采样，避免验证过程改变房屋类型分布。

## 从生成器到数据集：ProcTHOR-10K 与 ArchitecTHOR

### 一万座房屋有多大、跑得多快

ProcTHOR-10K 含 10,000 座训练房屋，另有 1,000 座验证房和 1,000 座测试房。所有房屋均通过上述可导航验证，每栋用 JSON 描述，运行时再由 AI2-THOR 加载，不必把完整场景逐一固化为大型独立文件。

与当时环境相比，规模差异很直观：

| 环境 | 场景规模 | 主要特征 |
|---|---:|---|
| AI2-iTHOR | 120 | 人工设计、可交互 |
| RoboTHOR | 89 | 人工设计、可交互 |
| iGibson | 15 | 写实场景，交互有限 |
| HM3D | 1,000 | 真实扫描、静态非交互 |
| Habitat 2.0 | 105 种布局 | 支持交互，但布局规模有限 |
| ProcTHOR-10K | 10,000 | 程序生成、可交互、支持物理模拟 |

数据使用 16 种房间规格，覆盖 1—10 个房间，房间内出现 95 类物体。下面的三幅正文统计图分别描述物体数、面积和房间数。

![每个房间内物体数量的频率分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/objects_per_room.png)

> 图解：横轴是每间房的物体数量，纵轴是房间频数。多数房间约有 7—28 个物体，峰值位于 17—22 个附近，说明场景并非只有几件家具的空壳。

![不同房间规模房屋的面积密度分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/house_area_distribution.png)

> 图解：横轴为房屋面积（平方米），纵轴为密度；蓝、橙、绿分别对应 1—3、4—6、7—10 个房间。三类分布峰值依次后移，大户型还延伸到数百平方米。

![ProcTHOR-10K 的房屋房间数分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/num_rooms.png)

> 图解：横轴为每栋房屋的房间数，纵轴为房屋数量。1—4 房占多数，同时保留 5—10 房的长尾，覆盖小房间到大型住宅。

![ProcTHOR-10K 的俯视图与第一人称视图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/sec4-4.jpg)

> 图解：俯视结构展示户型和物体布局，第一人称画面展示 agent 实际看到的场景。两类视图共同说明，统计规模最终落实为可进入、可观察的三维环境。

生成 10K 房屋只需本地工作站的 4 张 RTX A5000 GPU 运行 1 小时。但训练还需要高频仿真，论文因此在 8 张 Quadro RTX 8000 上测试渲染；单 GPU 使用 15 个进程，8 GPU 使用 120 个进程。小房屋指 1—3 房，大房屋指 7—10 房，每步渲染一张 $3\times224\times224$ RGB 图。

| 计算配置 | 导航 FPS（小） | 导航 FPS（大） | 独立交互 FPS（小） | 独立交互 FPS（大） | 环境查询 FPS（小） | 环境查询 FPS（大） |
|---|---:|---:|---:|---:|---:|---:|
| 8 GPUs | 8,599±359 | 3,208±127 | 6,488±250 | 2,861±107 | 480,205±19,684 | 433,587±18,729 |
| 1 GPU | 1,427±74 | 6,280±40 | 1,265±71 | 597±37 | 160,622±2,846 | 157,567±2,689 |
| 1 Process | 240±69 | 115±19 | 180±42 | 93±15 | 14,825±199 | 14,916±186 |

导航测试移动和旋转，交互测试推物体，查询测试读取不常每步返回的元数据。附录中的对照显示，ProcTHOR 的帧率总体与 AI2-iTHOR、RoboTHOR 同量级，但大房屋交互明显比小房屋慢；因此“生成便宜”和“训练毫无成本”不能混为一谈。

### ArchitecTHOR：保留人工高质量测试集

完全依赖程序生成也会带来评估盲点：训练与测试若共享同一套生成规则，模型可能只是适应生成器。作者因此让 3D 美术师手工制作 ArchitecTHOR，包含 5 座验证房和 5 座测试房，仅用于评估。

![ArchitecTHOR 五座人工验证房屋的俯视图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/architecthor-2.jpg)

> 图解：这些房屋比 AI2-iTHOR 和 RoboTHOR 的典型场景更大，结构、厨房台面、淋浴间和大量橱柜经过专门建模，用作更逼真的测试环境。

![ArchitecTHOR 人工厨房与浴室内部](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/design-3.jpg)

> 图解：人工资产带来更细致的空间与材质设计，为 ProcTHOR 预训练模型提供不共享生成过程的外观和布局分布。

![不同数据集的可导航面积箱线图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/navigable-area.png)

> 图解：纵轴为可导航面积（平方米）。ArchitecTHOR 验证房的面积整体明显高于 AI2-iTHOR 与 RoboTHOR 训练场景；ProcTHOR-10K 分布更宽，覆盖小场景到接近 ArchitecTHOR 的大场景。

10 座 ArchitecTHOR 房屋累计耗费约 **320 小时**专业美术工作，其中厨房资产约 125 小时、房屋建模约 60 小时。这与 10K 程序房屋的 1 小时形成鲜明对照，也说明人工集更适合做小规模高质量评估，而非承担大规模预训练。

![ArchitecTHOR 各环节累计制作时间](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/time-breakdown.png)

> 图解：横轴为小时，纵向列出制作环节。厨房资产耗时最多，其后是房屋建模、Unity 设置、纹理、浴室资产、物体摆放和灯光，总计约 320 小时。

虽然论文实验只用 RGB，平台还能输出深度、实例分割、语义分割、边界框和表面法线。

![ProcTHOR 的 RGB 输入模态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/rgb.jpg)

> 图解：RGB 是本文策略唯一使用的图像输入。

![深度模态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/depth.jpg)

> 图解：Depth 提供距离信息，但主实验刻意不使用，以检验大规模 RGB 训练本身的效果。

![实例分割模态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/instance-segmentation.jpg)

> 图解：Instance Segmentation 区分不同物体实例，可用于构造感知监督或分析中间表征。

![语义分割模态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/class-segmentation.jpg)

> 图解：Semantic Segmentation 按类别标记像素，平台还可通过 Unity 后端扩展更多模态。

![边界框标注模态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/bounding-box.jpg)

> 图解：Bounding Box 给出物体位置标注，可服务于检测等任务。

![表面法线模态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/0.15figures-houses-emoji.png-ProcTHOR-Procedurally-Generating-1-Million-Houses-for-Embodied-AI-ProcTHOR-Large-Scale-Embodied-AI-Using-Procedural-Generation/figures/normals.jpg)

> 图解：Surface Normals 描述表面朝向，说明该平台不只服务导航，也能支持三维视觉研究。

## 实验：简单模型能否吃到“规模红利”

### 任务、模型与评价指标

作者选择两类共 6 项基准：

- **ObjectNav**：给 agent 一个目标类别，如床或电视，让它导航到目标附近；覆盖 RoboTHOR、HM3D-Semantics、AI2-iTHOR 和 ArchitecTHOR。
- **ArmPointNav**：根据相对三维坐标，用机械臂把目标物体从起点搬到终点。
- **1-phase Room Rearrangement**：通过移动物体或改变状态，把当前房间恢复为目标场景。

ObjectNav 成功必须同时满足：agent 主动执行 `Done`；目标距相机不超过 1 米；目标在最终帧真实可见。SR（Success Rate）只衡量是否成功，SPL（Success weighted by Path Length）还惩罚绕路；若房内有多个同类目标，最短路径取到任一可达目标的最小值。

主模型刻意保持简单：CNN 编码当前图像，GRU 保存时间信息。ObjectNav 和 Rearrangement 使用 CLIP 视觉编码器，ArmPointNav 使用 3 层卷积网络；所有核心结果只输入 RGB，不使用深度、显式地图或人工任务监督。

### ObjectNav 的策略网络与奖励

标准 ObjectNav 模型接收 $3\times224\times224$ 图像，经冻结的 RN50 CLIP-ResNet 得到 $2048\times7\times7$ 特征，再用两层 $1\times1$ 卷积压缩到 $32\times7\times7$。目标类别被嵌入并复制到相同空间尺寸，与视觉特征拼接、压缩和展平；随后拼接上一动作，输入隐藏状态为 512 的单层 GRU。Actor 输出 6 个动作的 softmax 分布，Critic 输出状态价值。

动作包括前进、左转、右转、抬头、低头和结束。前进距离采样自均值 0.25 米、标准差 0.01 米的正态分布，旋转角均值为 30°、标准差为 0.5°，碰撞时前进失败。

训练采用 DD-PPO，episode 最长 500 步，最大化：

$$
\sum_{t=0}^{H}\gamma^t r_t,\qquad \gamma=0.99
$$

单步奖励为：

$$
r_t=\max\left(0,\min\Delta_{0:t-1}-\Delta_t\right)+s_t-\rho
$$

其中第一项奖励接近历史最近目标距离，成功奖励 $s_t=10$，每步惩罚 $\rho=0.01$。GAE 参数为 0.95，PPO clip 为 0.1，价值与熵损失系数分别为 0.5 和 0.01，学习率为 $3\times10^{-4}$。

预训练使用全部 10K 房屋和 16 类目标，共训练 4.23 亿步；2 亿步时已达到峰值表现的 90%。为避免抽到“冰箱里永远看不见的苹果”这种不可完成任务，系统先在网格上做 BFS，再从最近的 6 个可达位置向目标最多 6 个可见点做 raycast，只保留无需操作即可看见并接近的目标。

目标采样采用 $\epsilon=0.2$ 的 $\epsilon$-greedy：20% 随机选可达类别，80% 选训练中采样次数最少的类别，缓解床远多于棒球棒造成的类别不均衡。

### 六个基准的完整结果

下表保留正文全部结果。“0-shot”表示完全不使用该下游基准的训练场景；“fine-tune”表示先做 ProcTHOR 预训练，再在下游训练数据上微调。Habitat 的 Large 版本使用 RN50x16 CLIP、1024 维 GRU 隐藏状态和 $512\times384$ 输入，并在 4—10 房的 ProcTHOR-Large-10K 上预训练。

| 任务 | 基准 | 方法 | 成功率 | 第二指标 |
|---|---|---|---:|---:|
| ObjectNav | RoboTHOR Challenge | EmbCLIP（RoboTHOR 训练） | 47.0% | SPL 0.200 |
| ObjectNav | RoboTHOR Challenge | ProcTHOR 0-shot | 55.0% | SPL 0.237 |
| ObjectNav | RoboTHOR Challenge | ProcTHOR + fine-tune | **65.2%** | **SPL 0.288** |
| ObjectNav | Habitat 2022 / HM3D-Semantics | MLNLC | 52.0% | SPL 0.280 |
| ObjectNav | Habitat 2022 / HM3D-Semantics | FusionNav（AIRI） | 54.0% | SPL 0.270 |
| ObjectNav | Habitat 2022 / HM3D-Semantics | ProcTHOR 0-shot | 9.0% | SPL 0.055 |
| ObjectNav | Habitat 2022 / HM3D-Semantics | ProcTHOR + fine-tune | 53.0% | SPL 0.270 |
| ObjectNav | Habitat 2022 / HM3D-Semantics | ProcTHOR + Large 0-shot | 13.2% | SPL 0.077 |
| ObjectNav | Habitat 2022 / HM3D-Semantics | ProcTHOR + Large + fine-tune | **54.4%** | **SPL 0.318** |
| ObjectNav | AI2-iTHOR | EmbCLIP（AI2-iTHOR 训练） | 68.4% | SPL 0.516 |
| ObjectNav | AI2-iTHOR | ProcTHOR 0-shot | 75.7% | **SPL 0.644** |
| ObjectNav | AI2-iTHOR | ProcTHOR + fine-tune | **77.5%** | SPL 0.621 |
| ObjectNav | ArchitecTHOR | EmbCLIP（AI2-iTHOR 训练） | 18.5% | SPL 0.118 |
| ObjectNav | ArchitecTHOR | ProcTHOR 0-shot | **31.4%** | **SPL 0.195** |
| Rearrangement | AI2-THOR 2022，1-phase | EmbCLIP | 7.10% | % Fixed Strict 0.190 |
| Rearrangement | AI2-THOR 2022，1-phase | ProcTHOR 0-shot | 3.80% | % Fixed Strict 0.156 |
| Rearrangement | AI2-THOR 2022，1-phase | ProcTHOR + fine-tune | **7.40%** | **% Fixed Strict 0.245** |
| ArmPointNav | ManipulaTHOR | iTHOR-SimpleConv | 29.2% | PickUp SR 73.4% |
| ArmPointNav | ManipulaTHOR | ProcTHOR 0-shot | **37.9%** | **PickUp SR 74.8%** |

最有说服力的不是“所有零样本都赢”，而是不同迁移难度下呈现出的层次：

- RoboTHOR 上，0-shot 的成功率已从 47.0% 提到 55.0%；微调后 SPL 从 0.200 到 0.288，即提高 8.8 个百分点。
- AI2-iTHOR 上，0-shot 成功率达到 75.7%，高于在目标环境训练的 68.4%；微调后为 77.5%。
- ArchitecTHOR 完全不提供训练房，成功率从 18.5% 提到 31.4%，说明程序场景能迁移到独立人工大宅。
- ArmPointNav 只需 0-shot，成功率从 29.2% 提到 37.9%；作者尝试微调，但没有超过零样本模型。
- HM3D 的真实扫描外观差异最大，0-shot 成功率只有 9.0% 或 13.2%，并不好；优势主要在经过目标域微调后体现。
- Rearrangement 的 0-shot 同样弱于基线，微调才把 % Fixed Strict 从 0.190 提到 0.245。

截至 2022 年 6 月 14 日 10:00 PT，模型在 Habitat 2022 ObjectNav、AI2-THOR Rearrangement 2022 和 RoboTHOR ObjectNav 三个公开榜单排名第一。这里的意义在于：收益并不依赖更复杂的地图或视觉里程计模块，但也不能据此宣称 ProcTHOR 已经消除 Sim2Sim 外观差距。

### 规模消融：多数指标随房屋数增长

为隔离数据规模，作者分别用 10、100、1,000、10,000 座房屋训练，并关闭材质增强；每个模型都训练到训练集 80% 成功率。这样可以比较“同样会背训练环境”的模型，在陌生环境中的表现。

| 训练房屋数 | ArchitecTHOR SPL / SR | RoboTHOR 0-shot SPL / SR | HM3D 0-shot SPL / SR | AI2-iTHOR 0-shot SPL / SR |
|---:|---:|---:|---:|---:|
| 10 | 0.077 / 11.3% | 0.040 / 8.53% | 0.007 / 1.60% | 0.249 / 28.7% |
| 100 | 0.102 / 18.6% | 0.076 / 20.9% | **0.050 / 10.4% SR** | 0.352 / 42.0% |
| 1,000 | 0.122 / 17.2% | 0.157 / 33.1% | 0.027 / 4.65% | 0.456 / 53.0% |
| 10,000 | **0.185 / 27.0%** | **0.210 / 44.5%** | **0.060 / 9.70%** | **0.554 / 64.9%** |

总体趋势清楚：从 10 到 10K，RoboTHOR 成功率从 8.53% 增至 44.5%，AI2-iTHOR 从 28.7% 增至 64.9%。但并非每个中间点都单调，ArchitecTHOR 的 SR 在 100 到 1,000 间从 18.6% 降到 17.2%，HM3D 也有明显波动。更准确的结论是**规模带来强劲总体收益，而不是数据翻倍必然提升每个指标**。

### 三类任务的复现配置

ObjectNav 之外，两项操作任务采用不同数据和训练策略：

| 项目 | ObjectNav | ArmPointNav | Rearrangement |
|---|---|---|---|
| ProcTHOR 训练数据 | 10K 房屋、16 类目标 | 7,000 房屋、58 类可拿取物体 | 2,500 个一/二房间房屋 |
| 训练方式 | DD-PPO | 强化学习，GRU 策略 | Teacher Forcing + DAgger + Behavior Cloning |
| 主要训练量 | 4.23 亿步 | 1 亿帧 | 20 万 teacher-forcing 步、200 万 DAgger 步、约 1.8 亿 BC 步 |
| 预训练资源与时间 | 3 台 AWS g4dn.12xlarge，约 5 天 | 4 台同型机器，16 GPU、192 CPU 核，3 天 | 6 台同型机器，24 GPU、288 vCPU，4 天 |
| 下游微调 | 按基准调整目标类别与训练场景 | 微调未超过 0-shot | 4,000 个 AI2-iTHOR episode，分三阶段共 1,500 万步 |

ArmPointNav 把图像编码为 512 维向量，再把目标相对三维坐标经三层线性层编码为 512 维，共同输入 GRU，episode 上限 200 步。评估覆盖 AI2-iTHOR 的 20 个房间、28 类物体和 528 个任务。

Rearrangement 每个 episode 打乱 1—5 个物体，既包括位置变化，也包括最多一个物体的开合状态变化。预训练在 240 个并行仿真上运行；微调使用 80 个单房间场景的 4,000 个 episode，在 8 张 Titan X 上约需 16 小时。

## 数据使用、局限与影响

ProcTHOR 的房屋实例是数千行 JSON，不附人工任务标签，各房屋独立生成。论文数据说明中区分了实验所用的 10K 训练集与计划发布的 **100 万座静态房屋 JSON**；代码、资产和材质库按 Apache 2.0 许可发布。房屋生成不依赖个人数据，可用于商业和非商业研究。

除了导航与操作，作者还列出多 agent、视觉语言导航、音视频导航、Sim2Real、物体检测、分割、深度与光流估计、NeRF、姿态估计等潜在用途。通过替换规格，框架也可能扩展到工厂、办公室、商店乃至更大规模的虚拟空间。

局限同样需要明确：

- ProcTHOR-10K **只有单层房屋**，无法覆盖楼梯、多楼层路径和跨层操作。
- 资产仍受 108 类、1,633 个实例限制，作者希望继续引入 ABO、PartNet、ShapeNet、Google Scanned Objects 和 CO3D 等资源。
- 物体合理性大量依靠人工类别标注与经验参数，生成规则本身可能成为模型偏差。
- HM3D 和 Rearrangement 的较弱 0-shot 表明，程序场景规模不能自动解决外观和任务分布差异。
- 大规模预训练仍需数天和多 GPU，低生成成本不等于低训练成本。

家庭机器人若能因此获得更强泛化能力，可能帮助烹饪、清洁、照护和行动不便人群；但陪伴依赖、服务可及性不均、家庭隐私、监控与恶意使用也会同步放大。数据规模扩大的是能力，不会自动给出这些社会问题的答案。

## 总结

- ProcTHOR 用多阶段条件采样生成户型、结构、光照、资产与状态，把场景规模和可交互性放到同一框架中。
- Room Spec、SAG、碰撞检查、语义标注和 BFS 验证共同保证“随机生成”仍然符合常识并可执行任务。
- 10K 房屋在 4 张 RTX A5000 上 1 小时生成，ArchitecTHOR 则以 320 小时人工制作提供独立高质量测试。
- 简单 RGB 模型在 6 项基准上取得论文统计时点的领先结果，规模消融也显示 10 到 10K 房屋带来显著总体收益。
- 零样本迁移并非全面成功，单层限制、资产规模、生成偏置和训练算力仍是后续工作需要面对的问题。

ProcTHOR 开启的方向，是把具身智能的数据瓶颈从“还能找到多少现成场景”转向“怎样设计更好的生成约束与训练分布”。

> 本文参考自 [ProcTHOR: Large-Scale Embodied AI Using Procedural Generation](https://arxiv.org/abs/2206.06994)