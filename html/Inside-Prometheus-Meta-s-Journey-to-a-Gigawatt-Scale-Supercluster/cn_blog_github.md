# Prometheus：Meta 如何把一座吉瓦级超算集群的中断率压低 99%

一个训练任务跑在几万张 GPU 上，任何一台主机出问题，整个任务都可能被拖慢甚至中止——这是大模型训练基础设施最核心、也最不性感的难题。Meta 最近披露了它在俄亥俄州新奥尔巴尼（New Albany）建设的吉瓦级超算集群 Prometheus 的工程细节。一期工程已经完成，并训练出了 Muse 系列基础模型（包括面向个人和专业 agentic 任务的 Muse Wonder）：预训练任务的 goodput（有效产出率）超过 85%，强化学习任务超过 90%；训练期间的整体平均无中断时间（MTBI）约 4.7 万 GPU 小时，后续固件修复又把它推到 5.4 万 GPU 小时以上。

这篇工程博客的价值不在于宣布又一个"百万 GPU 集群"——虽然 Prometheus 满配后确实将包含约 100 万张分布式 GPU 同步工作——而在于它罕见地公开了一个超大规模集群从"天天掉"到"接近工业级稳定"的完整过程：2025 年年中峰值期每天每 8000 张 GPU 发生近千次中断，一年后下降了 99%。下面按硬件、网络、可靠性和供电散热四条线展开。

## 从传统数据中心到超算集群

两年前，新奥尔巴尼还只是一个跑 Web 流量的常规数据中心站点。Meta 从硬件、网络和建筑设计入手，分期把它改造成训练前沿模型的基地。目前 Prometheus 由多栋互联建筑中的几十万张 GPU 组成，涵盖传统数据中心楼、AI 优化楼和快速部署结构，靠数英里的光纤连接成"一台机器"。这种容量让 Meta 得以追求以前不现实的更大模型和更长上下文窗口。

## 三代 NVIDIA 硬件平台

Prometheus 目前部署两代 NVIDIA GPU 平台，第三代 Vera Rubin NVL72 即将到来。三代都基于 NVL72 机架架构：72 张 GPU 在同一网络 fabric 内作为单一加速器协同工作。

### Catalina：GB200 NVL72

第一代平台 Catalina 采用 GB200 NVL72 的 Ariel 变体——双机架配置，每机架 36 张 GPU 通过 NVLink 互联成一个统一的 72 GPU 域。相比 Grand Teton H100，它在原始算力和显存容量上都有提升，并基于 Meta 贡献给开放计算项目（OCP）的 Open Rack V3 高功率机架。

Ariel 配置里每张 Blackwell GPU 搭配一颗 Grace CPU 和两块 ConnectX-7 400G 网卡，CPU 与 GPU 比例 1:1。这个选择背后有个值得注意的判断：GB200 NVL72 相对 H100 几乎把每个维度都扩展了——FLOPS、HBM 带宽、scale-up 带宽、scale-up 域大小——唯独每张 GPU 的 scale-out 带宽仍是 400G。团队预判这种不均衡会损失性能，于是选择了每 GPU 800G scale-out 的配置。这让 Meta 比行业更早部署这个规格，也为后续扩展保留了规格一致性。

Catalina GB200 NVL72 的关键指标：

- 72 张 Blackwell GPU，共 13.4 TB HBM3e（每卡 192 GB）
- FP8 训练峰值算力 720 PFLOPS（FP4 下 1,440 PFLOPS）
- 域内 NVLink 峰值带宽 130 TB/s
- 聚合显存带宽峰值 576 TB/s
- 双机架总功耗约 120 kW

### Clemente：GB300 NVL72

第二代平台 Clemente 升级到 GB300 NVL72，在 GPU 和平台两个层面改进 Catalina。GB300 把每卡 HBM 容量提升 1.5 倍（192 GB → 288 GB），dense FP4 吞吐同比例提升，attention（softmax）加速能力约翻倍。平台层面，ConnectX-8 SuperNIC 原生提供每 GPU 800G，Ariel 上提前部署的能力变成了基线；配合 2:1 的 GPU-CPU 比例，完整的 72 GPU 域被塞进单机架，物理部署大幅简化，线缆减少、可维护性明显提升。散热上既可用模块化风辅助液冷（AALC），也可在新建数据中心使用设施级液冷（FLC）。

Clemente GB300 NVL72 的关键指标：

- 72 张 Blackwell Ultra GPU，共 20.7 TB HBM3e（每卡 288 GB，比 GB200 多 50%）
- FP4 峰值算力超过 1 exaFLOP
- NVLink 峰值带宽 130 TB/s（每 GPU 双向 1.8 TB/s）
- 机架功耗约 135 kW
- 相比 GB200 NVL72 性能提升 5-10%

## 把集群连成一台机器的网络

传统 Web 服务的数据中心网络承载的是服务间的请求响应，实例之间通常不需要高带宽、低延迟、紧同步。大规模 AI 训练三者都要，所以网络本身成了 AI 负载的关键部分。Meta 几年前在标准数据中心网络之外加了第二张专用以太网——Back-End（BE）网络，用于楼内和跨楼的 GPU 直接通信。

大规模预训练会把任务划分成数据并行的模型副本，网络使用因此天然分层：延迟敏感、带宽敏感的操作集中在副本内部，对延迟更宽容、带宽需求更低的流量安排在副本之间。Meta 的网络设计就匹配这个层次——楼内是低延迟高带宽 fabric，楼间用聚合网络连接。团队还与 Meta Superintelligence Labs 紧密协作，优化并行设计、流量模式和通信算法，尽量不暴露网络通信的开销，让 GPU 保持忙碌而不是等网络。这些概念先在楼级 2.4 万卡集群、再到区域级 10 万卡以上集群中验证过，Prometheus 把它们推向新的规模和距离。

### 楼内 fabric

每栋楼的以太网 fabric 通过多层拓扑互联最多 20,736 张 GPU。最初采用 Disaggregated Scheduled Fabric（DSF），用来缓解训练集合通信的拥塞——这类流量通常是少量大体积长生命周期的流，传统负载均衡下容易碰撞；DSF 通过子报文级多路径路由、fabric 内报文重排、虚拟输出队列和基于信用的端到端流控来解决。随着以太网卡技术进步（端点可以做报文重排），Meta 升级到 Non-Scheduled Fabric（NSF），采用浅缓冲的 51.2T 交换机，比如支持动态负载均衡的 Minipack3，以及基于 NVIDIA Spectrum-4 ASIC、配合 ConnectX 网卡自适应路由的 Minipack3N。DSF 和 NSF 都跑在 Meta 自研的 FBOSS 网络栈上，这让团队能快速迭代优化，也能在大规模运维中获得足够的可观测性。

### 跨楼聚合：BAG 网络

Backend Aggregation（BAG）网络是多层 hub-and-spoke 结构，连接 Prometheus 内各楼的 fabric，跨度从 2 公里到超过 100 公里，采用 Arista 7800 系列机箱和深缓冲的 Broadcom Jericho3 芯片，每对区域之间提供数 Pbps 级带宽。

![Prometheus 园区建筑间互联示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Inside-Prometheus-Meta-s-Journey-to-a-Gigawatt-Scale-Supercluster/images/Meta-Prometheus-BAG-3.png)

> 图解：Prometheus 园区鸟瞰图，蓝色节点和连线标出各建筑之间的互联关系，标注的距离从 321m、1500m 到 3800m、5400m 不等——这就是 BAG 网络要覆盖的物理尺度。

![BAG 区域互联拓扑](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Inside-Prometheus-Meta-s-Journey-to-a-Gigawatt-Scale-Supercluster/images/Meta-Prometheus-BAG-1.png)

> 图解：BAG 的区域级拓扑。每个 Region（A-D、W-Z）包含约 8 万张 GPU（4 个 2 万卡单元），区域内各 Region 汇聚到本侧的 BAG 节点（间距约 5km），两侧 BAG 之间再经约 10km 链路互联。这种 hub-and-spoke 结构把几十万张 GPU 拼成一个逻辑集群。

在这样的网络上跑预训练需要专门优化——尤其要考虑这种距离下光速带来的物理延迟下限，Meta 与 Superintelligence Labs 联合做了针对性调优，确保训练不被网络拖慢。

![Prometheus 网络距离层级](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Inside-Prometheus-Meta-s-Journey-to-a-Gigawatt-Scale-Supercluster/images/Meta-Prometheus-BAG-2.png)

> 图解：Prometheus 网络的距离层级示意：从建筑内的短距互联，到 5km、10km 的园区级链路，最外圈达到 100km 量级。距离每上一个台阶，对光模块和延迟优化的要求就完全不同。

整个 Prometheus 网络的复杂度体现在数字上：数百万英里的光纤束、数万台交换机、数百万只光模块——从楼内低成本短距的 2×400G FR4 LITE 可插拔模块，到超过约 10 公里链路使用的 800G-ZR 相干光模块。

## 可靠性：把中断率压低 99% 的工程

在 Prometheus 的规模下，任何单点故障都可能中止整个训练任务、浪费宝贵的算力。这要求一种超出传统供应商关系的联合工程模式：NVIDIA 团队深度进驻 Meta 数据中心，双方共享实时遥测，联合召开故障分析会，把每一次系统停机按根因拆解。归因明确带来责任明确：Meta 优化调度和基础设施，NVIDIA 处理固件、信号诊断和内存管理，两边都不扯皮，修复更快。

### 72 GPU 故障域：性能与风险的交换

从 H100 时代的 NVL8 到 NVL72，基础设施韧性的玩法被根本改变。scale-up 域——通过 NVLink 紧耦合做张量并行的 GPU 组——从 8 张扩到 72 张，故障爆炸半径扩大 9 倍。大域带来带宽优势，但也让 72 GPU 机架成为容量规划的基本分配单位。

训练并行在 2 的幂次分组上效率最高，所以每个 NVL72 域实际跑 64 张 GPU，剩下 8 张作为机内备件：GPU 故障时备件立刻顶上，任务从上一个 checkpoint 重启而不是等硬件修好，机架继续留在任务里。机内冗余把有效机架故障率压到"整 pod 失效"的下限；对更罕见的整机架故障，再额外保留约 2% 的备用机架。以一个 32,768 张活跃 GPU 的训练任务为例，需要配置 523 个 NVL72 机架、共 37,656 张 GPU，比活跃占用高约 13%——Meta 特别说明，其中 11% 来自 64/72 切片的几何约束，而不是纯粹的可靠性税。

物理复杂度进一步放大可靠性挑战：单个 NVL72 域包含 5,184 条调谐差分对信号链路，每条跑 200 Gbps，穿过 GPU SerDes、计算托盘 PCB、线缆盒、交换机托盘接口等多个潜在失效点。在这个密度下维持每通道 200 Gbps 的信号完整性，是前所未有的工程难题。

### 故障恢复的两层机制

按故障爆炸范围分两类处理：

- **Parallelism-Aware Fault Tolerance（PAFT）**：故障能隔离在单个数据并行副本内时，把故障副本摘出，其余集群重复该步继续训练，受损切片在备用主机上重启并重新同步——一次潜在的全系统停机被降级为聚合吞吐的短暂小幅下降。
- **Checkpointing**：爆炸范围大于单个副本时，整个任务回滚到上一个 checkpoint。分布式 checkpoint 架构把模型状态（含梯度和优化器数据）异步保存到多层存储，用 delta checkpointing 和内存暂存把保存开销对训练吞吐的影响降到最低。

这两层机制配合机内备用 GPU 托盘和备用机架策略，压缩了更换硬件的时间和损失的工作量。

### 中断率曲线：从每天近千次到接近目标

![GB200/GB300 中断率趋势](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Inside-Prometheus-Meta-s-Journey-to-a-Gigawatt-Scale-Supercluster/images/Meta-Prometheus-GB200-GB200-interruptions.png)

> 图解：2025 年 6 月至 2026 年 9 月 GB200（自建，蓝线）、GB300（自建，橙线）及云端（紫线）的日均中断次数（每 8000 张 GPU）。GB200 在 2025 年中峰值接近 1000 次，经过图中标注的 8 轮固件与配置修复（NVLink 配置变更、固件分批推送、显存修复、链路恢复检查等）一路下降；GB300 上线初期中断率更高，随后收敛；粉色虚线是目标：每天每 8000 张 GPU 仅 2 次中断。

2025 年年中 Prometheus 初期，中断率峰值接近每天每 8000 张 GPU 1000 次。经过一年的跨职能工程，中断率下降 99%，集群现在能并行支撑多个数万卡训练任务。Meta 的做法是把每个中断来源做细粒度分解，再对每类故障部署针对性方案：

- **NVLink 信号完整性**：GB200 部署初期最主要的瓶颈。超过 60 万条 NVLink 连接，哪怕轻微的信号劣化都会频繁触发 NVLink 恢复周期。Meta 上了持续遥测和自动标记，用 L11 诊断烧机在投产前隔离弱链路；分类工具区分瞬时故障和持续性信号完整性故障，指导定点维修和换件，修好的系统验证后才回生产池。
- **内存可靠性（HBM 与 LPDDR）**：随 GB300 显存容量上量，内存故障成为主要中断源。Meta 与 NVIDIA 用全机队遥测和错误分析识别出反复出现的故障特征，通过 HBM 固件参数调优、内存擦洗（scrubbing）、改进的 LPDDR 纠错算法、以及基于模式的预测性检测（在劣化器件影响任务前提前退役）来缓解，目前正在全机队铺开并量化效果。
- **NVLink 交换机与 fabric 健康**：单个交换机故障可能瘫痪整个 NVL72 域。Meta 以毫秒级分辨率监控 fabric 健康，自动恢复路径隔离故障组件、在剩余基础设施上重均衡流量，不打断任务。
- **后端网络事件**：在 BAG super-spine 层，链路事件通过基于硬件的故障检测透明处理；平面拓扑确保单根光纤或端口故障被绕开，不中断同步的 GPU 流量突发。

支撑持续改进的是运营体系：新奥尔巴尼现场常驻的联合分诊团队、2026 年 3 月一次跨栈同步修复的稳定性冲刺、以及归因驱动的问责框架——每次维修都必须经过强制隔离验证才能回到算力池。Meta 的可靠性北极星是每天每 8000 张 GPU 2 次中断（约 10 万 GPU 小时 MTBI）。

## 机队级硬件成熟化

部署几十万张第一代加速器，会暴露出低产量下隐藏的质量问题。GB200 的部署把"硬件成熟化"变成了 Meta 及其合作伙伴的一门主修工程学科，具体包括：

- **快速固件部署**：自动化的固件包下发，几小时内把修复推到 10 万张以上 GPU；单 pod 升级耗时的 P90 从早期硬件导入阶段的 7 个多小时降到 2 小时。
- **机队健康**：关联多机架系统的遥测，给每次任务中断分类，指导软件修复、硬件维修和更换决策；MTBI 与可用容量、有效训练吞吐一起跟踪。
- **跨伙伴联合认证**：与多个制造伙伴并行验证，靠生产机队的持续反馈保证质量一致。
- **自动化信号筛查**：60 万条 NVLink 路径在任何生产流量上路前都经过诊断压力测试——这个规模下再小的缺陷率也得主动找出来。
- **供电压力测试**：研究 PSU 模块和电压调节器在单卡 1400W 负载下的相互作用，暴露只在大批量下出现的失效模式。
- **精密液冷**：对冷板和歧管装配执行严格质量门，维持结温、防止漏液。
- **机队维护与健康检查**：主动备件、快速处置、修复优先级排序，以及入场烧机和任务前健康检查，筛出健康节点进大训练任务。

从 Blackwell 到 Vera Rubin，每一代新平台都吸收这些一线运维经验，逐步收紧筛查和设计规范。

## 供电瞬变与散热

几万张 GPU 同步执行计算密集阶段和通信周期时，会触发大幅功率瞬变——集群内 30-40% 的负载突变；启动或暂停一个大训练任务，几秒钟内负载变动可超过几十兆瓦。Meta 的应对是 Dynamo Dimmer，一个集群级电源管理框架：接近电气上限时不硬停任务，而是渐进调整 GPU 功率上限，让训练不中断地继续。配套的还有机架级电池备份（BBU）应对瞬时尖峰、电容托盘（CBU）削峰、设施级电池储能系统（BESS），以及与电力公司的深度协同。

散热方面，Prometheus 混合使用多代冷却方案：模块化风辅助液冷（AALC）用于自由空气冷却楼和快速部署结构，新建的 AI 优化楼则采用设施级液冷（FLC）以获得最大功率密度。除了自有建筑，Meta 还整合战略伙伴的租赁容量来加快部署；工业级快速部署结构则把"生产级算力上线"的工期压到了新水平。

能效上，新奥尔巴尼站点与当地公用事业合作、支付全额能源成本，并匹配新增清洁可再生能源。闭环液冷是 Meta 新一代 AI 数据中心的首选，直接到芯片的闭环液冷还能让单机架塞进更多 GPU，液冷网络机架 IcePack 等设计也通过 OCP 开放给了行业。

## 下一步：Vera Rubin 与 Hyperion

Prometheus 将继续用 GB300 NVL72 和 Vera Rubin NVL72 混合扩容。Vera Rubin 将直接接入 Prometheus 现有网络 fabric，以 Meta 即将推出的 San Miguel 平台部署。

![数据中心内的机架实拍](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Inside-Prometheus-Meta-s-Journey-to-a-Gigawatt-Scale-Supercluster/images/Vera-Rubin-Prometheus.jpg)

> 图解：Meta 数据中心内的机架实拍，可见密集的 NVLink 线缆（紫色）与光纤布线——这种物理密度正是前文信号完整性与散热挑战的直观来源。

San Miguel Vera Rubin NVL72 的规划指标：

- 72 张 Vera Rubin GPU，每卡 288 GB HBM4、峰值带宽 19.2 TB/s
- 每机架 FP4 训练峰值 2.52 EFLOPS、FP8 峰值 1.26 EFLOPS
- NVLink 峰值带宽 216 TB/s（每 GPU 双向 3 TB/s）
- 聚合显存带宽峰值约 1,500 TB/s
- 全液冷计算托盘，单机架功耗超过 170 kW
- 每瓦性能比 GB300 NVL72 提升 10-20%

Prometheus 积累的经验也将成为后续超算集群的蓝图：Meta 在路易斯安那州 Richland Parish 规划的下一代集群 Hyperion 从头按这些经验设计——闭环液冷、与公用事业伙伴 Entergy 的全额成本电网接入，以及新一代网络设计。

## 结语

Prometheus 这篇披露最有信息量的部分，不是百万 GPU 的愿景数字，而是一条可复用的工程路径：大规模集群的稳定性不是买来的，是靠归因到每一类故障源、再逐一工程化解决出来的——从 60 万条 NVLink 链路的信号筛查，到把每天近千次中断压到个位数的固件迭代节奏。对关注大模型基础设施的读者，这份细节比任何参数表都更能说明前沿训练的真实瓶颈在哪里。需要注意的是，文中所有数字都来自 Meta 官方口径，goodput、MTBI 等指标的定义与测试条件并未完全公开，跨集群对比时要谨慎；Vera Rubin 的指标还是规划值，实际表现要等落地验证。

> 本文参考自 [Inside Prometheus: Meta's Journey to a Gigawatt-Scale Supercluster](https://engineering.fb.com/2026/10/19/data-center-engineering/inside-prometheus-meta-gigawatt-scale-supercluster/)