# OpenTac：机器人皮肤按需生成

给机器人手指、手掌或足底装触觉，往往要先迁就传感器的固定形状；想贴合新结构，又得重新设计电路、布线和机械外壳。这不仅拖慢机器人本体迭代，也让仿真、采集和控制代码随着传感器形状反复重写。OpenTac 的思路是：用可变形凸点把三维力“翻译”成压阻阵列的共模与差模信号，再以一份共享的传感器描述文件贯通自动设计、仿真、跨传感器迁移和自适应采样。作者实际制作并验证了 31 块不同外形的传感器，全部通过设计规则检查且均能工作；在 $32\times32$ 皮肤上，稀疏接触采样从全阵列扫描的 67 Hz 提升至 1286 Hz，达到 **19.2 倍**。

![OpenTac 全栈触觉生态与四类机器人部署](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac_overview.png)

> 图解：A 展示可贴合手指、夹爪、手掌和足底的定制皮肤；B 是 YAM、Dexmate Vega-1、UR5 与 SO-101 四类真实机器人部署；C 给出完整链路：文本、2D 轮廓或 3D 模型经 OpenTac-CAD 生成柔性 PCB、凸点外壳和读数板文件，再由共享 manifest 驱动仿真、跨传感器映射与高速采样。

## 一、触觉落地的瓶颈，不只是“造一个传感器”

机器人要可靠地插接、抓取和滚动物体，仅靠关节位置或腕部力传感器往往不够。真正决定接触状态的，是物体在皮肤表面哪里压、往哪个方向滑，以及法向力和切向力如何变化。

**共形触觉皮肤**把分布式感知单元集成到薄而柔的表面中，像给机器人贴上一层可弯曲的皮肤。但现有方案常把几个难题留给了使用者：

- **外形定制难**：AnySkin 可更换感知表面，但电子布局固定；eFlesh 能自动生成可变形结构，却仍需另配磁强计电子系统。
- **感知维度受限**：FlexiTac 等低成本皮肤主要提供法向力，难以分辨“压住了”与“正在侧向滑动”。
- **规模与速度冲突**：常见触觉传感器采样率约为 30–100 Hz；阵列越大，逐点扫描越慢，毫秒级撞击容易被漏掉。
- **软件接口随硬件返工**：不规则皮肤没有整齐的二维网格，采集、仿真和下游策略若默认矩形阵列，就需要逐款适配。

从传感器对比可以看到，OpenTac 并不是单纯追求最低成本或最高帧率，而是把剪切力感知、全传感器自动设计和自适应读取放在同一个开放系统中。

| 传感器 | 剪切力 | 形态 | 自动设计范围 | 速率（Hz） | 估算成本（美元） |
|---|---|---|---|---:|---:|
| GelSight | 支持 | 指尖 | 不支持 | 30 | 65–400 |
| DIGIT | 支持（借助标记） | 指尖 | 不支持 | 60 | 55–103 |
| GelSlim 4.0 | 支持 | 指尖 | 不支持 | 30–60 | 73–123 |
| TacTip | 支持 | 指尖 | 不支持 | 30–120 | 67–169 |
| uSkin | 支持 | 贴片 | 不支持 | 100 | 10–25 |
| ReSkin | 支持 | 皮肤 | 不支持 | 400 | 5.5–10.5 |
| AnySkin | 支持 | 皮肤 | 指尖与模具 | 100 | 2–10 |
| eFlesh | 支持 | 3D 本体 | 传感表面 | 100 | 25–50 |
| iCub skin | 不支持 | 皮肤 | 不支持 | 25 | 2.5–7 |
| FlexiTac | 不支持 | 皮肤 | 不支持 | 100 | 约 1.95* |
| Sun 等 | 支持 | 皮肤 | 不支持 | 未给出 | 2–10 |
| **OpenTac** | **支持** | **皮肤** | **完整传感器** | **67–1286（自适应）** | **约 1.95*** |

> 表中成本按原论文口径列出：按传感器单元计，皮肤的成本也可按平方厘米计；星号表示仅传感器本身，不含随形态变化的读数板。OpenTac 的 1286 Hz 对应稀疏接触下的部分采样，不能与所有条件下的全阵列采样率简单等同。

相关工作的差异也体现在软件层。TACTO、Taxim 主要模拟视觉触觉传感器的成像过程，DiffTactile、TacEx、Taccel 等进一步建模表面变形；OpenTac-Sim 则直接读取编译后的传感器几何与凸点布局。已有跨传感器方法常依赖配对数据学习统一表征，OpenTac-Transfer 选择物理坐标与局部坐标系做显式映射，不要求成对的跨传感器数据。

![OpenTac 在多种皮肤上记录的法向力、剪切力与旋转](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/tactile_examples_1x4.png)

> 图解：热图颜色表示法向载荷，青色箭头表示各凸点的剪切方向，白色弧线表示旋转估计。从左到右依次为 $64\times64$ 皮肤上的圆顶与剪钳扭转、$9\times9$ 指尖的拇指扭转、UMI 夹爪两片 $29\times6$ 皮肤上的罐体与球体摇动，以及螺丝刀上下移动造成的外部接触。这里的关键不是画面更丰富，而是不同外形都能输出统一的法向与切向信息。

明确了这些断层之后，OpenTac 首先要回答的是：怎样用简单、可量产的结构，同时测出法向力与剪切力？

## 二、核心传感器：让凸点把三维力写成压力图案

### 2.1 三层结构与共模—差模解码

OpenTac 采用**压阻式传感**：压力改变材料的局部电阻，再由电路转成电压。这一路线材料便宜、读数电路简单，也兼容标准柔性 PCB（fPCB）工艺。传感器从下到上只有三类核心部件：

1. **双层柔性 PCB**：每个行—列交叉处的叉指铜电极构成一个 taxel，即触觉像素。
2. **压阻薄膜**：受压后局部电阻下降，提供与压力相关的模拟信号。
3. **可变形凸点盖**：每个弹性凸点覆盖一小组 taxel，例如 $2\times2$，把外部三维载荷转成组内压力分布。

可以把凸点理解成一块微型“杠杆橡皮”：竖直下压时，四个触点受力较均匀；横向推挤时，凸点倾斜，一侧更紧、另一侧更松。因此，**总和**反映法向载荷，**方向性差分**反映剪切载荷。

对 $2\times2$ 配置，记扣除基线后的信号为 $z_{uv}$，其中 $u,v\in\{-1,+1\}$ 表示 taxel 相对凸点中心的位置方向，则三维触觉读数为：

$$
h_z=\sum_{u,v}z_{uv},\qquad
h_x=\sum_{u,v}u z_{uv},\qquad
h_y=\sum_{u,v}v z_{uv}
$$

其中 $h_z$ 是共模响应，$h_x,h_y$ 是差分响应。再通过可选的仿射标定，就能把 $[h_x,h_y,h_z]$ 映射为力单位下的 $[F_x,F_y,F_z]$。每种不规则阵列的分组与符号都写入 manifest，避免把 $2\times2$ 的假设硬编码进所有皮肤。

这个设计的聪明之处在于，它没有为每个方向增加一套独立传感器，而是复用同一压阻阵列。这样，密度、电极和机械凸点都可以调整，装配与读数流程却基本不变。

### 2.2 材料与几何如何改变量程

作者用方形阵列和载荷传感器逐项改变材料、电极间隙及 taxel 间距。下图（a）至（c）的横轴均为外加力，纵轴为每 taxel 的输出计数；实线为各力区间中位数，阴影为四分位范围，菱形表示达到饱和判据的位置，虚线为 12 位 ADC 上限。

![压阻材料、电极几何与三轴力标定结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/calibration.png)

> 图解：（a）在 2.0 mm 间距、0.2 mm 电极间隙下比较 Chibitronics、LINQSTAT 和 Yunjia；（b）比较 0.1–0.5 mm 电极间隙；（c）比较 1.5、2.0、4.0 mm taxel 间距；（d）横轴是 OptoForce 参考力，纵轴是 OpenTac 估计力，虚线为 $y=x$，三轴结果均靠近对角线。

材料选择直接决定灵敏度与饱和点。Chibitronics 在低力下就趋于饱和；LINQSTAT 在测试到 60 N 时仍未触发饱和判据；Yunjia 的整体 ADC 响应最低。电极间隙影响相对温和，而增大 taxel 间距会增强单点信号，也同时改变空间分辨率和量程。

一个反直觉的例子是：Chibitronics 配合 2 mm 间距时可覆盖约 60 N；间距增至 4 mm 后反而约在 20 N 提前饱和。也就是说，**信号更大并不等于量程更宽**，传感器几何必须围绕目标力区间联合设计。

默认三轴标定配置为 $4\times4$ 凸点阵列、LINQSTAT 薄膜、4.0 mm taxel 间距和 0.2 mm 电极间隙。相对 OptoForce 的样本内结果如下：

| 力分量 | $R^2$ | RMSE |
|---|---:|---:|
| $F_x$ | 0.95 | 0.11 N |
| $F_y$ | 0.93 | 0.16 N |
| $F_z$ | 0.96 | 0.15 N |

这些数字证明共模—差模解码确实能恢复三轴力，但它们是**样本内标定结果**，不应直接外推为所有形状、材料和接触条件下的统一精度。

有了可调的传感原理，下一步是把“调参数”变成一条从机器人外形直达制造文件的自动流水线。

## 三、OpenTac-CAD：从轮廓、曲面或一句话到柔性 PCB

### 3.1 编译的不是外形，而是完整传感器

OpenTac-CAD 把传感器设计视作确定性编译过程。输入可以是手绘 2D 区域、DXF 矢量图，也可以是 STL 机器人网格上选定的表面；输出则包括柔性 PCB、凸点外壳、连接与布线信息，以及机器可读的 manifest。

对 3D 表面，系统沿共享边逐三角形展开。圆柱、圆锥和分片平面等**可展曲面**能够无畸变摊平；无法摊平的双曲面区域会被检测并报告，而不是悄悄生成失真的板形。柔性 PCB 可弯曲但不可拉伸，这个限制必须在输入阶段就处理。

随后，编译器完成三类工作：

- **排布 taxel**：默认尺寸多为 4 mm、中心间距 4.2 mm；实验中制作过 1–4 mm 的 taxel。用户还能设置边缘留白、尾部形状、连接器及电气 clearance（安全间距）。
- **规划行列读数**：$R\times C$ 矩阵只需 $R+C$ 条通道，而非每个 taxel 一条线。60 针读数板支持至多 30 行、30 列；紧凑的 30 针板对支持至多 15 行、15 列，两者共用固件与 USB 接口。
- **沿轮廓布线**：行、列位于柔性板两面，每面都是单层平面布线问题。系统按跨度从短到长处理，沿阵列及既有铜线的安全偏置轮廓走线，绕开凹部并检查间距，从而避免昂贵的通用布线搜索。

读数电路的一侧通过移位寄存器驱动线路，另一侧经多路复用器和反相放大器采样；虚拟地保持被测线路电位，以抑制串扰。不同皮肤因此可以复用同一读数板，而不必重新写一套底层驱动。

![OpenTac-CAD 从机器人表面生成传感器及其性能](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/cad2sensor_wide.png)

> 图解：A 中橙色区域是 YUBI、UMI、LEAP Hand、Robotiq 2F-85 与 Unitree B2 上选定的感知面，下方为生成的柔性 PCB；B 展示凸点盖、压阻膜、柔性 PCB 和底板的堆叠，以及可调电极参数和实际转角布线；C 左图横轴为感知区域覆盖率，比较 OpenTac 与同间距最大内接矩形阵列，右图横轴为 taxel 数、纵轴为生成时间，31 块板均为零 DRC（设计规则检查）违规。

### 3.2 manifest：全生态真正共享的“接口协议”

manifest 可以理解为传感器说明书与软件接口的合体。对每个 taxel $i$，它记录物理中心 $\mathbf{p}_i$、行列扫描索引 $(r_i,c_i)$ 与电极 footprint（占板形状），同时保存感知边界、制造轮廓、连接器引脚映射、生成参数及凸点机械分组 $b(i)$。

这里最关键的设计，是把**物理几何**与**电扫描顺序**分开。不规则皮肤为了少交叉、少占空间，布线顺序未必等于空间排列；如果软件把扫描序号误当作网格坐标，接触图案就会错位。manifest 让仿真、迁移与采样都依据真实坐标工作。

![OpenTac-CAD 制作的 31 块传感器与两种读数板](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac-sensors-readout.png)

> 图解：A 是已制作并测试的 31 块柔性 PCB，覆盖细长、凹形、近似矩形与异形轮廓；B 展示 60 针与 30 针读数板，板上集成滤波、读数电路、微控制器和 IMU（惯性测量单元）。右侧多模态读数中，横轴为相对当前的时间，转向图纵轴为相对起点的角度，振动图纵轴为去重力加速度（g），底部为同步触觉帧。

### 3.3 自动设计真的节省时间吗？

作者在 Apple M4 Pro 笔记本上对每款设计运行三次。31 块已编译并制造的板全部零 DRC 违规，所有已制造板均能工作。五个代表性机器人部件含 153–467 个 taxel，间距为 2–4 mm，只需 35–46 条电气通道。

| 目标区域 | OpenTac-CAD 覆盖率 | 最大内接矩形阵列 | 提升 |
|---|---:|---:|---:|
| YUBI 指垫 | 75% | 57% | 18 个百分点 |
| UMI 软夹爪手指 | 90% | 90% | 0 |
| LEAP Hand 手掌 | 89% | 79% | 10 个百分点 |
| Robotiq 2F-85 指垫 | 87% | 87% | 0 |
| Unitree B2 足底 | 93% | 93% | 0 |

结论很清晰：接近矩形的区域，两种方法相同；tapered（渐缩）或不规则区域才体现形状感知排布的价值。五款设计从轮廓到制造文件只需 **0.7–2.6 秒**。

自然语言入口则由 Claude Opus 5 视觉语言模型（VLM）代理提供。模型可搜索尺寸、提出几何假设并选择 taxel 与板形，再反复调用编译器扫参；真正的电极规则、布线、连接和 DRC 仍由确定性工具完成，而不是让模型直接“画电路”。

![从一句话生成多种触觉皮肤的自动设计实例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/cad_agentic.png)

> 图解：A 展示 Allegro Hand 指尖设计中的决策记录：联网查得尺寸、假设指尖几何，并因连接器宽度把间距从 2.0 mm 调到 2.2 mm；B 包括 Unitree G1 足底、Shadow Dexterous Hand 手掌、自行车座、地图轮廓、壁虎和星形等设计，标签给出 taxel 数、行列网络数和板尺寸。九次代理会话均生成零 DRC 布局，含 76–254 个 taxel，中位耗时 6.5 分钟；这些例子只验证自动设计，未声称全部制造。

笔者认为，这里比“一句话生成”更重要的是**模型与编译器的分工**：语言模型处理模糊需求，编译器守住几何与电气约束，设计假设也被明确记录。

硬件能自动做出来之后，下一个问题是：能否不先制造每一块板，就在仿真中使用同样的传感器？

## 四、OpenTac-Sim：同一份描述生成可交互的触觉传感器

### 4.1 凸点皮肤用 FEM，平滑皮肤用泡沫近似

OpenTac-Sim 基于 MuJoCo。带凸点的三轴传感器采用有限元方法（FEM）：每个触觉 pad 被转成四面体网格，作为线弹性体模拟，底部固定在传感器基板或夹爪上；原始表面网格保留用于碰撞计算。接触发生后，系统在凸点局部坐标系计算法向和切向力，再插值到 taxel 分辨率。

为了逼近真实读数链路，仿真还计算 pad 对基板的弹性力，将其映射到电极 footprint，再使用共模—差模公式得到读数。自动标定会扣除 pad 自身等效质量在重力和机械臂加速度下产生的载荷。

![OpenTac-Sim 的 FEM 模型与多机器人任务](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac-sim-pagewide.png)

> 图解：A 从“按压并拖动”的表面接触出发，经四面体网格变形得到法向与剪切读数；B 展示 SO-101 抓罐、ALOHA 双臂插接、UR5 开抽屉和 SO-101 整理入篮。各热图对应不同手指或夹爪的局部力，说明仿真输出由 manifest 决定，而非固定矩形模板。

不带凸点、只测法向力的平滑皮肤使用更轻的 Winkler 泡沫层近似：把泡沫看成许多独立小弹簧，压力随压入深度增加。其压力关系为：

$$
p=k(\delta-z)_+
$$

其中 $\delta$ 为压入量，$z$ 为表面位置，$(\cdot)_+$ 表示只保留正值。系统求解 $\delta$ 以匹配施加的法向力，再用宽度为 $\sigma$ 的高斯核平滑压力场，积分到 manifest 中的 taxel，最后施加目标标定。

当前版本只对**刚性物体**的传感器—物体交互提供稳定模拟，因此它不是任意软体接触问题的通用求解器。

### 4.2 仿真与真实读数是否对齐？

作者让力控 UR5e 末端以 1–5 N、每次增加 0.5 N 的序列按压凸点，再在仿真中重复相同载荷。真实传感器先标定到力单位，用于比较读数与接触深度。

![OpenTac-Sim 与真实传感器在读数和接触深度上的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac-sim-vs-real-plot.png)

> 图解：左图横轴为外力（N），纵轴为读数计数；右图横轴为相对首次接触的压入深度（mm），纵轴为力（N）。蓝线是真实数据，绿线是仿真，阴影显示真实测量波动。两者均值趋势接近，但真实噪声随信号增大而增大，说明仿真抓住主要力—变形关系，却没有完整复制真实噪声。

作者还在仿真中训练 SO-101 插销策略，验证其不仅能画触觉热图，还能接入操作策略学习流程。

![SO-101 在仿真中利用可变形触觉 pad 学习插销](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac-sim-policy-learning.png)

> 图解：蓝球为 100 个训练起始位姿，橙色方块为 50 个留出评估位姿；紫线是专家轨迹，半透明夹爪表示不同时刻，灰色虚线是另一条示范。右侧放大插入过程中固定侧与运动侧 pad 的切向位移，箭头表示各凸点剪切。该图提供的是可行性示例，不等同于 sim-to-real 策略迁移成功率。

解决了单款传感器的仿真，还要处理实验室里更常见的情况：新旧皮肤形状和分辨率不同，采集到的数据能否互换？

## 五、OpenTac-Transfer：沿物理坐标迁移接触图案

OpenTac-Transfer 不做黑箱特征对齐，而是把源皮肤的标定力场“铺”到目标皮肤的真实几何上。设源 taxel 单元为 $A_i$、目标单元为 $B_j$，接触坐标系对齐为 $T$，源标定为 $C_s$。重叠面积决定映射权重：

$$
M_{ji}=\frac{|T(A_i)\cap B_j|}{|A_i|},\qquad
\hat{F}_j=\sum_i M_{ji}C_s(s_i)
$$

随后通过目标标定的逆函数得到预测计数，并加入目标的噪声底与饱和限制：

$$
\hat{s}_j=C_t^{-1}(\hat{F}_j)
$$

当目标皮肤覆盖完整接触区域时，该映射保持总力。它只需要每款传感器的仿射标定和已知接触位姿，不需要采集一一对应的跨皮肤数据；这也是它与依赖配对数据的表征学习路线的根本差别。

实验使用四款无凸点 LINQSTAT 皮肤：2 mm 间距的 Triangle（409 taxels）和 AgileX（502），以及 4 mm 间距的 UMI（174）和 Robotiq（96）。五种物体在 95 个位置按压，接触力覆盖 1–8 N，所有预测器都获得接触位姿。

评价指标是接触形状的 F1：预测与测量场先做一次间距尺度的模糊，再以各自峰值的 25% 为阈值提取活跃 taxel。对比对象包括基于 manifest 的原始计数面积平均、无 manifest 的双线性缩放，以及前述泡沫模型。

| 方法或评估范围 | 中位形状 F1 |
|---|---:|
| OpenTac-Transfer，全部 5290 次预测 | **0.750** |
| resize + manifest | 0.714 |
| resize、无 manifest | 0.634 |
| 两款皮肤共同覆盖的区域 | **0.788** |
| 同一皮肤重复测量参考 | 0.800 |
| 泡沫模型，三款未参与拟合的皮肤 | 0.727 |

![不同皮肤之间的接触形状迁移与泡沫模型预测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/transfer.png)

> 图解：（a）按“目标 $\leftarrow$ 源”排列至少 4 N 按压的中位表现，比例尺为 10 mm；（b）纵轴为不同物体的形状 F1，灰线是同皮肤重复性参考；（c）横轴为标定接触力、纵轴为 F1；（d）热图横轴为目标皮肤、纵轴为源皮肤，颜色表示共同覆盖区域上的迁移 F1。4 mm 目标皮肤达到 0.84–0.88，高于 2 mm 目标的 0.67–0.78。

共同覆盖区域的 0.788 已接近同皮肤重复测量的 0.800，说明显式几何映射是有效的。不过，较粗分辨率天然更容易获得形状重叠高分；更重要的是，这些结果只覆盖**已知位姿下的法向接触形状**，尚未证明力幅值精度或 sim-to-real 策略迁移。

当皮肤面积和分辨率继续扩大，逐点读取又成为新瓶颈。OpenTac 的最后一个系统模块因此转向采样本身。

## 六、OpenTac-Xampling：少读一些点，抓住毫秒级接触

传统 raster scanning（栅格扫描）像逐行读完一整页文字：$N$ 个 taxel 每帧就读 $N$ 次，即使实际接触只占几个点。阵列越大，帧率越低；持续不足 10 ms 的撞击可能恰好落在两次读取之间。

OpenTac-Xampling 利用接触通常**稀疏、局部且时间连续**的特点，将每帧预算限制为 $M\ll N$，采用二叉四叉树自适应子采样：

- 约四分之一预算在全有效 mask 中搜索，避免只看上一帧而漏掉新接触；
- 其余预算跟踪上一帧的活跃区域，把分辨率集中到正在发生的接触上；
- manifest 提供有效区域和邻域，因此矩形、凹形、圆形和夹爪垫无需手写扫描表；
- 若所有 taxel 同时接触，方法会退化到与栅格扫描相同的复杂度，而不会凭空保持高帧率。

在 $32\times32$、共 1024 taxels 的皮肤上，全扫描为 67 帧/秒；随着 $M$ 减小，帧率单调上升，在 $M=16$ 时达到 1286 帧/秒。

![自适应部分扫描的速度、网球撞击检测和异形皮肤采样](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/xampling_wide.png)

> 图解：（a）横轴为每帧采样预算 $M$，纵轴为帧率，橙线为 67 fps 全扫描；（b）纵轴为每次网球撞击被捕获的帧数，空心点为单次下落、实心点为均值，虚线表示约 3 ms 接触在当前帧率下覆盖的帧数，上方为检测次数；（c）为 $M=128$ 的代表撞击，颜色为高于基线的 ADC 计数；（d）为 $M=64$ 时四种皮肤的采样分布，蓝色表示未采样点。

网球从 30 cm 高度落下，信号超过 ADC 满量程 50% 才计为捕获帧。全扫描在六次撞击中只检测到两次，而且各只有一帧；$M=16$ 至 256 时，每种设置的三次测试撞击均被检测到，最快设置能在一个约 3 ms 事件中记录多帧；$M=512$ 时检测到三次中的两次，逐渐接近全扫描极限。

同一固件在 $M=64$ 下，对 1024-taxel 方形皮肤每帧只读 6%，对 102-taxel 夹爪垫读 63%。这体现了统一 manifest 的便利，也提醒读者：**部分采样用空间完整性换时间分辨率**，接触区域越大，需要的预算越高。

至此，OpenTac 的设计、仿真、迁移和读取形成闭环。但这些模块最终仍要回答一个应用问题：剪切力信息真的能让机器人做得更好吗？

## 七、真实机器人验证：剪切力在接触任务中补上了什么

作者在四类真实机器人上部署 OpenTac，覆盖无视觉策略学习、富接触反馈控制与交互式阻抗控制。

![OpenTac 在 SO-101、YAM、UR5 与 Dexmate 上的部署](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac-robot-deployments.png)

> 图解：A 为 SO-101 插头插入，热图显示法向力、青色箭头显示剪切；B 为双臂 YAM 钥匙旋转插入，右侧是两指触觉图；C 为 UR5 无球位置反馈的边缘滚球，轨迹图横轴为时间、纵轴为区域比例，绿色、黄色、红色分别对应 Valid、Borderline、Invalid；D 为 Dexmate 前臂感知推压方向并做阻抗响应，右侧色条为实时 ADC 计数。

### 7.1 策略学习：不是“有触觉”就够了，而是要有剪切

SO-101 上评估两项任务：peg-in-hole（插销入孔）和 plug-in-socket（插头入插座）。后者要求电触点亚厘米级对齐，单靠是否受压很难区分“已经对准”与“顶在插座周围”。

实验比较 FlexiTac、只测法向力的 OpenTac-1D，以及同时测法向与剪切的 OpenTac-3D。两款 OpenTac 使用相同皮肤和读数板，仅覆盖层不同。每种传感器分别训练：

- **Pose**：仅输入关节角；
- **Full**：输入关节角与触觉；
- 策略采用 ACT（Action Chunking Transformer，动作分块 Transformer）；
- 每项使用 100 条示范、评估 50 次，全程不使用视觉。

为每种传感器单独设置本体感觉基线很有必要，因为凸点不仅增加剪切感知，也会改变夹持摩擦与示范分布。

![SO-101 与双臂 YAM 上不同触觉输入的策略成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac-policy-learning-quantitative.png)

> 图解：纵轴均为成功率；A 中斜线柱形为 Pose，实心为 Full，标签给出均值和相对提升，字母标注用于统计分组，不共享字母表示在超过 90% 置信区间下存在显著差异；B 为 25 次钥匙插入评估。图中 OpenTac-3D 标为 61%，而论文正文写为 62%，下表按图列出。

| 机器人与任务 | 传感器 | Pose／本体感觉 | Full／触觉 | 变化 |
|---|---|---:|---:|---:|
| SO-101 插销入孔 | FlexiTac | 50% | 58% | +8 个百分点 |
| SO-101 插销入孔 | OpenTac-1D | 52% | 60% | +8 个百分点 |
| SO-101 插销入孔 | OpenTac-3D | 66% | **82%** | **+16 个百分点** |
| SO-101 插头入插座 | FlexiTac | 32% | 22% | −10 个百分点 |
| SO-101 插头入插座 | OpenTac-1D | 40% | 46% | +6 个百分点 |
| SO-101 插头入插座 | OpenTac-3D | 54% | **70%** | **+16 个百分点** |
| 双臂 YAM 钥匙入锁 | 仅本体感觉 | 18% | — | — |
| 双臂 YAM 钥匙入锁 | OpenTac-1D | — | 43% | — |
| 双臂 YAM 钥匙入锁 | OpenTac-3D | — | **61%（图示）** | 较 1D 高 18 个百分点 |

最值得注意的不是所有“Full”都变好，而是**法向触觉并未稳定超过各自基线，三轴触觉在三项任务中都取得最高观察值**。插头悬空与插入时，法向热图可能几乎相同，剪切估计却明显变化；钥匙是否进入槽内、卡在边缘或尚未接触，也主要体现在切向载荷中。

双臂 YAM 的钥匙机构要求一只手持锁，另一只手旋转并插入带凸槽的钥匙，只在特定角度匹配。仅本体感觉成功率为 18%，OpenTac-1D 为 43%，OpenTac-3D 图示为 61%。这组结果支持“剪切提供关键接触线索”，但任务与传感器覆盖层同时变化，仍不宜把它解释为所有操作任务的普遍增益。

### 7.2 反馈控制：在滑动发生前调整动作

除了训练策略，作者还直接测试剪切反馈控制器。

**边缘滚球。** UR5 末端安装 $32\times32$ OpenTac，让球沿末端外缘滚动，不使用球位置反馈，而是调节到沿边缘的目标剪切向量。与仅用法向力相比，剪切控制能更可靠地把球从 Borderline 区域带回 Valid 区域。

**受冲击的双臂抓取。** Dexmate Vega-1 Pro 用两个 OpenTac 探头夹住纸杯，各条件采用相同轨迹、控制增益和 2.5 N 标称夹持力。剪切控制器在滑落前提高目标夹持力；每种条件测试 20 次，Held 表示保持到 40 秒的次数，Slip 为纸杯净位移的均值与标准差。

| 负载条件 | 反馈方式 | Held（次，越高越好） | Slip（cm，越低越好） |
|---|---|---:|---:|
| 无负载 | 仅本体感觉 | 3 | 掉落，未定义 |
| 无负载 | 腕部 F/T 传感器 | 20 | $0.15\pm0.08$ |
| 无负载 | OpenTac-1D | 20 | $0.2\pm0.1$ |
| 无负载 | OpenTac-3D | 20 | **$0.11\pm0.06$** |
| 有负载 | 仅本体感觉 | 0 | 掉落，未定义 |
| 有负载 | 腕部 F/T 传感器 | 9 | $5.5\pm1.3$ |
| 有负载 | OpenTac-1D | 9 | $5.8\pm2.1$ |
| 有负载 | OpenTac-3D | **16** | **$2.6\pm1.2$** |

有负载时，OpenTac-3D 保持成功 16/20 次，高于腕部 F/T 和 OpenTac-1D 的 9/20 次，滑移也更小。这说明分布在被夹持表面的剪切信号，比单纯腕部合力或法向压力更早提供了失稳线索。

不过，传感阵列要进入长期使用，除了“测得准”和“控制得好”，还必须经受反复载荷与漂移的考验。

## 八、耐久性、局限与总结

作者用 UR5e 末端压头连续按压同一个凸点 10 小时，每约 30 分钟施加 3–5 N、步长 0.5 N 的力并记录读数。

![OpenTac 单凸点十小时连续按压测试](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenTac-An-Open-Ecosystem-for-Customizable-Scalable-and-Transferable-3D-Tactile-Sensing/figures/opentac-endurance-test.png)

> 图解：横轴为测试时间（小时），纵轴为触觉计数总和，五条曲线对应 3、3.5、4、4.5、5 N。最初约 2 小时读数逐渐上升，之后总体趋稳；整个 10 小时内，不同力对应的相对大小仍能区分。

这意味着实际部署最好在每次使用前设置 warmup（预热）阶段，或优先使用相对测量而不是把绝对 ADC 计数当作长期不变的真值。10 小时测试证明了相对可分性，并不等同于消除了漂移、迟滞和温度依赖。

OpenTac 的边界同样清楚：

- 柔性 PCB **不可拉伸**，网格设计只适用于能展开成可展面片的表面；
- 压阻材料存在迟滞、漂移和温度敏感性，长期绝对力测量仍需校准策略；
- 自适应采样用空间完整性换更新率，大接触区域需要更大的采样预算；
- 仿真信号主要是变形代理，当前稳定交互限于刚体，真实信号噪声也未被完整复现；
- Transfer 验证的是位姿已知的法向接触形状，不是力幅值精度或 sim-to-real 策略性能。

最后把这项工作收束为五点：

- **传感层**：用凸点下 taxel 的共模与差模响应，同时估计法向力和剪切力，默认标定三轴 $R^2$ 为 0.93–0.96。
- **设计层**：OpenTac-CAD 从 2D、可展 3D 表面或文本生成完整制造文件，31 块实制板零 DRC 违规且全部工作。
- **系统层**：共享 manifest 解耦物理几何与电扫描顺序，使同一套仿真、迁移和采样工具适配异形皮肤。
- **性能层**：自适应部分采样在 $M=16$ 时达到 1286 Hz，是同一 $32\times32$ 皮肤全扫描的 19.2 倍。
- **应用层**：三轴触觉在无视觉插接与受冲击抓取中优于对应法向方案，说明“剪切”比单纯增加触觉通道更关键。

未来最值得推进的方向，是把可拉伸基底、更完整的真实噪声与软体接触建模，以及力值和策略层的跨传感器验证接入同一开放生态；论文已在 [OpenTac 项目页](https://opentac.github.io/) 开放硬件、固件、软件和数据，为这些扩展提供了可复现起点。

> 本文参考自 [OpenTac: An Open Ecosystem for Customizable, Scalable, and Transferable 3D Tactile Sensing](http://arxiv.org/abs/2610.31661v1)