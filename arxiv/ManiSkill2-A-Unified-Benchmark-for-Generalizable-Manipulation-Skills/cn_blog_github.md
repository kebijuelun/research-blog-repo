# ManiSkill2：单卡采样达 2532 FPS

机器人操作研究长期缺少一个既能比较不同算法、又能真正考察物体几何与拓扑泛化的统一基准，这让“在熟悉物体上成功”很容易被误当成“学会了操作”。ManiSkill2 用 20 个任务族、2000+ 物体模型和 4M+ 示范帧，把刚体与软体、固定与移动底座、单臂与双臂操作放进同一套动态仿真、接口和评测协议中。它进一步用双向刚体—MPM 耦合、闭环示范动作空间转换、异步渲染和 Render Server，同时解决任务真实性、示范复用与视觉 RL 采样效率问题。最硬的结果有两个：在 RTX Titan 与 16 个 CPU 逻辑核的 PickCube 测试中，CNN 策略采样最高达到 **2532±63 FPS**；PickCube 策略从仿真中的 **91.0%** 成功率直接迁移到真实机器人后，仍取得 **60.0%** 的成功率。

![ManiSkill2 的刚体、软体、固定臂与移动操作任务](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/teaser.png)

> 图解：上排是固定底座的刚体操作，中排展示推椅子、移桶、开关柜门和抽屉等单臂或双臂任务，下排展示挖掘、倒水、悬挂面条等软体操作。它们共享统一接口，但机器人形态、物体动力学和成功条件并不相同，这正是“通用操作”与“单一任务脚本”之间的差别。

## 一、旧基准为什么难以衡量“可泛化操作”

人学会抓一类物体后，通常能处理没见过的新实例和新初始位姿。论文把这类能力定义为**可泛化操作技能**：面对不同外观、几何、拓扑和初始配置，仍能完成交互；这些技能还可以组合成更长程的日常任务。

要衡量它，基准不能只提供若干能运行的环境，还必须同时满足三个条件：

- **变化发生在物体层面**：不仅随机初始位置，还要更换形状、结构和拓扑。
- **交互来自完整动力学**：抓取、插入和软体形变都应由物理过程决定，而非把物体吸附到夹爪上。
- **算法能在同一协议下比较**：Sense-Plan-Act、Imitation Learning（IL）和 Reinforcement Learning（RL）应能共享任务、观测、控制器与训练/测试划分。

此前不少基准为了降低构建难度，采用抽象抓取、少量物体变化或单一操作类型。研究者为了适配自己的算法又各自修改环境，结果看似使用同一基准，实际评测条件已经不同。ManiSkill2 的核心贡献，首先是把这些被拆散的条件重新统一起来。

### 与其他操作基准的差异

下表按论文原表保留各基准的主要定位。这里的“抽象抓取”不表示底层引擎无法实现物理抓取，而是表示该基准在算法评测中使用了高层抽象。

| 对比项 | ManiSkill2 | BEHAVIOR-1K | Habitat 2.0 | IsaacGym | ManipulaThor | MetaWorld | RoboSuite | RLBench | TDW |
|---|---|---|---|---|---|---|---|---|---|
| 抓取实现 | Physical | Abstract | Abstract | Physical | Abstract | Physical | Physical | Abstract | Abstract |
| 示范轨迹数 | **>30k** | — | — | — | — | Procedural | 约 2000 | Procedural | — |
| 多控制器支持 | **是** | 未知 | **是** | 否 | 否 | 否 | **是** | **是** | 否 |
| 视觉 RL/IL Baseline | **完整** | 有限 | **完整** | 否 | **完整** | 否 | 否 | 否 | 否 |
| 物体模型数 | **>2144** | 3324 | YCB | — | 150 | 80 | 10 | 28 | 112 |
| 场景数 | — | **306** | 105 | — | 30 | — | — | — | 105 |
| 光线追踪 | Kuafu | Omniverse | — | — | — | — | NVISII | — | Unity |
| Domain Randomization | 部分支持 | 未知 | 否 | **是** | 否 | 否 | **是** | **是** | 否 |
| 刚体仿真 | SAPIEN | Omniverse | Bullet | PhysX 5 | Unity | MuJoCo | MuJoCo | V-REP | Unity |
| 软体仿真 | Warp-MPM | Omniverse | — | — | — | — | — | — | — |

这张表也提醒我们：ManiSkill2 并非在所有维度都占优。它强调低层、短视界、全物理操作和大规模示范，而 BEHAVIOR-1K、Habitat 2.0 更强调场景级变化；它的 Domain Randomization 也只覆盖部分任务。统一并不意味着包办所有机器人问题。

## 二、20 个任务族怎样覆盖操作难点

明确评测目标之后，下一步是把“泛化”落实为可执行任务。ManiSkill2 的一个任务族包含目标相同、资产或初始状态不同的多个变体；20 个任务族不是 20 条固定轨迹，而是 20 组可系统采样的任务分布。

### 刚体操作：从抓放到毫米级插入

固定臂 6-DoF 抓放包括 PickCube、StackCube、PickSingleYCB、PickSingleEGAD 和 PickClutterYCB。目标位置在 $30\times50\times50\,\mathrm{cm}^3$ 的大工作空间中随机选择，因此“感知模型给出一个好抓姿”并不等于“机器人能到达并执行这个抓姿”。

精密装配包括 PegInsertionSide、PlugCharger 和 AssemblingKits。三者分别使用 3 mm、0.5 mm 和可配置的间隙（例如 0.8 mm），并要求真实插入，而不是只比较末端位置误差。PegInsertionSide 必须插入至少一半，PlugCharger 必须完全插入，AssemblingKits 则要求物体真正嵌入对应槽位。

关节物体（Articulated Object）操作继承并扩展了 ManiSkill1：OpenCabinetDoor、OpenCabinetDrawer、PushChair、MoveBucket 使用移动机器人，TurnFaucet 使用固定臂。最后的 AvoidObstacles 要求固定臂在密集障碍中主动感知并到达目标位姿。

### 软体操作：难点是“动作改变了多少物质”

Fill、Hang、Excavate、Pour、Pinch、Write 六个任务分别覆盖转移黏土、挂面条、挖取定量黏土、倒水、捏塑和写字。它们考察的不仅是末端轨迹，还包括动作对位移量、流量和形变的影响。

例如，Fill 只要求目标烧杯中的黏土超过 90%，而 Pour 要求最终液面与红线相差不超过 4 mm。两者看似都在“把东西倒进容器”，后者却要求对瓶子倾角和液体流量进行更精细的闭环控制。

### 任务、成功条件与示范规模

论文默认使用两阶段评测，最终结果取两阶段平均值。下表汇总 20 个任务族的关键定义；示范数量是论文发布时提供的数据，不等同于每个任务的评测 episode 数。

| 任务族 | 目标与成功条件要点 | 示范数据 |
|---|---|---|
| PickCube | 立方体到达目标 2.5 cm 内，机器人静止 | 1000 条成功轨迹 |
| StackCube | 红立方体稳定放在绿立方体上，且不再被抓取 | 1000 条成功轨迹 |
| PickSingleYCB | 指定 YCB 物体到达目标 2.5 cm 内 | 74 个物体各 100 条 |
| PickSingleEGAD | EGAD 物体到达目标 2.5 cm 内，颜色随机 | 1600 个训练物体通常各 5 条 |
| PickClutterYCB | 从 4–8 个 YCB 物体中抓出目标并放置 | 共 4986 条 |
| AssemblingKits | 物体完全入槽；中心高度差 < 3 mm、旋转误差 < 4°、位置误差 < 2 cm | 共 1720 条，覆盖 300+ 板配置与 20 个训练形状 |
| PegInsertionSide | 至少一半插销进入横向孔 | 1000 条成功轨迹 |
| PlugCharger | 充电器完全插入墙面插座 | 1000 条成功轨迹 |
| AvoidObstacles | 穿过密集障碍，末端位置误差 < 2.5 cm、姿态误差 < 15° | 1976 条 |
| TurnFaucet | 将指定水龙头把手转过目标角度 | 多数模型每个 100 条；约 15 个难例由 MPC-CEM 生成 |
| Fill | 烧杯内黏土 > 90%，软体速度 < 0.05 | 200 条成功轨迹 |
| Hang | 面条跨过杆、两端分居两侧、不触地，夹爪打开且软体稳定 | 200 条成功轨迹 |
| Excavate | 挖起指定范围内的黏土量并达到高度，落地粒子 < 20 | 200 条成功轨迹 |
| Pour | 液面距红线 < 4 mm，洒出粒子 < 100，瓶子最终回正 | 200 条成功轨迹 |
| Pinch | 当前形状到目标形状的 Chamfer Distance 小于初始距离的 0.3 倍 | 1556 条成功轨迹 |
| Write | 当前图案与目标字符的交并比（IoU）> 0.8，目标来自 50+ 字符 | 200 条成功轨迹 |
| OpenCabinetDrawer | 指定抽屉打开到最大行程的 90% 以上并静止 | 25 个训练柜体，每个目标抽屉 300 条 |
| OpenCabinetDoor | 指定柜门打开到最大行程的 90% 以上并静止 | 42 个训练柜体，每个目标柜门 300 条 |
| PushChair | 双臂移动机器人将转椅推至目标 15 cm 内，静止且不倾倒 | 26 个训练椅子，每个 300 条 |
| MoveBucket | 将装有球的桶竖直放到平台上，静止且球不洒出 | 29 个训练桶，每个 300 条 |

AssemblingKits 的测试资产不是训练形状的简单复刻，而是相似但未见过的新形状。

![AssemblingKits 的测试板与测试形状](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/assembling_kits_eval.png)

> 图解：板上五个几何轮廓对应不同的待插入物体。成功不是把物体放到轮廓附近，而是同时满足高度、旋转和平移约束，使物体在真实碰撞与接触下嵌入槽位。

### 观测模式与评测划分

ManiSkill2 支持六类观测：

1. `state_dict`：机器人本体状态、物体真值状态和任务目标，不含视觉。
2. `state`：上述字典的扁平化版本。
3. `rgbd`：所有相机的 RGBD、机器人本体状态和可用的任务目标。
4. `rgbd_robot_seg`：在 RGBD 上增加机器人各连杆的分割掩码。
5. `pointcloud`：多相机融合点云、机器人本体状态和任务目标。
6. `pointcloud_robot_seg`：在点云模式上增加机器人连杆分割。

默认 RGBD 和点云模式**不包含物体真值分割**。例如 TurnFaucet 用把手初始位置指定目标，而不是依赖测试时难以获得的特权掩码。对含资产变化的任务，论文分别报告训练物体与 held-out 物体结果，避免只用训练资产上的高分代表泛化能力。

## 三、环境不是一次建模，而是验证驱动的迭代

任务数量多并不自动意味着质量高。物理参数、碰撞网格、成功判据或相机位置中的一个小错误，都可能让环境“看起来合理、实际上不可解”。ManiSkill2 因此把示范生成同时当作环境验证手段。

![ManiSkill2 的验证驱动环境构建流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/build_pipeline_v3.png)

> 图解：流程分为任务创建、奖励设计和观测配置三阶段。任务创建负责资产、机器人、初始状态与成功指标；奖励设计为 MPC/RL 提供 shaped reward；观测配置调整相机和状态信息。随后通过 TAMP、MPC 或 RL 生成示范并验证环境，失败会反向推动任务与资产修正。

三种验证方式各有边界：

| 方法 | 优势 | 局限与适用场景 |
|---|---|---|
| TAMP（任务与运动规划） | 不需要手工 shaped reward，适合固定臂抓放与装配 | 面对推椅子等欠驱动系统较困难 |
| MPC（模型预测控制） | 有好奖励时无需训练和观测即可搜索难任务解 | 很难为大量物体设计统一奖励 |
| RL | 推理阶段更易扩展，能直接优化交互策略 | 需要额外训练和 Hyperparameter 调优 |

这个设计的聪明之处在于：**示范不是环境完成后的附属品，而是环境验收的一部分**。如果 TAMP、MPC 和 RL 都无法在合理条件下完成任务，问题可能出在环境，而不只是算法。

统一评测还延伸到了云端。用户以 Docker 镜像提交方法，只需提供“接收观测、返回动作”的函数，并可配置 RGBD/点云与控制器。服务器拉取镜像、运行评测、写入数据库并展示公共榜单，使规划器、启发式方法和端到端 RL 能在同一协议下比较。

## 四、多控制器与示范转换：让数据不被动作空间绑死

任务与评测统一之后，还有一个容易被忽略的问题：同一个“抓起物体”的动作，在关节空间和末端空间中是两组完全不同的数字。如果示范由关节位置控制器生成，而学习算法需要末端位姿增量，直接训练就会遇到动作空间不匹配。

### 控制器到底做了什么

Controller 是策略与电机之间的接口：策略输出目标位置、速度或位姿，控制器再将其转成关节力矩。以 PD Controller 为例：

$$
\tau(t)=K_p\left(\bar{q}(t)-q(t)\right)+K_d\left(\bar{\dot q}(t)-\dot q(t)\right)
$$

其中 $q$ 与 $\dot q$ 是当前关节位置和速度，$\bar q$ 与 $\bar{\dot q}$ 是目标值，$K_p$、$K_d$ 分别是刚度和阻尼。Augmented PD Controller 还会补偿重力等被动力。

ManiSkill2 默认机器人是 7-DoF Franka Emika Panda，大多数学习用控制器的动作归一化到 $[-1,1]$。它支持底座、机械臂和夹爪分别选择控制器，而不是给整台机器人强制使用同一种动作参数化。

| 控制器 | 维度 | 动作含义 |
|---|---:|---|
| `arm_pd_joint_pos` | 7，未归一化 | 绝对目标关节位置，适合运动规划 |
| `arm_pd_joint_delta_pos` | 7 | 相对当前关节位置的增量 |
| `arm_pd_joint_target_delta_pos` | 7 | 相对上一目标关节位置的增量 |
| `arm_pd_ee_delta_pos` | 3 | 相对当前末端位置的位置增量，内部用 IK 求关节目标 |
| `arm_pd_ee_target_delta_pos` | 3 | 相对上一目标末端位置的增量 |
| `arm_pd_ee_delta_pose` | 6 | 相对当前末端的平移与紧凑轴角旋转 |
| `arm_pd_ee_target_delta_pose` | 6 | 相对上一目标末端位姿的增量 |
| `arm_pd_joint_vel` | 7 | 目标关节速度，$K_p=0$ |
| `arm_pd_joint_pos_vel` | 14 | 同时给目标关节位置与速度 |
| `arm_pd_joint_delta_pos_vel` | 14 | 上述位置—速度控制器的增量形式 |
| Gripper | 1 | 两指共享同一目标关节位置 |

“target”与“非 target”系列的区别也很重要：前者相对**上一目标状态**累积，后者相对**当前实际状态**计算。实际机器人存在跟踪误差，两者并不等价。

### 闭环转换为什么比开环重放更可靠

设源环境用关节位置控制器，目标环境用末端位姿增量控制器。$a_{\mathrm{src}}(t)$ 是源动作，$\bar T_{\mathrm{src}}(t)$ 是它对应的期望末端位姿，$T_{\mathrm{tgt}}(t)$ 是目标环境当前实际末端位姿，$FK(\cdot)$ 是前向运动学。转换公式为：

$$
a_{\mathrm{tgt}}(t)=FK\left(a_{\mathrm{src}}(t)\right)T_{\mathrm{tgt}}^{-1}(t)
$$

也就是先把源关节目标映射到末端目标，再用目标环境的真实当前位姿计算相对变换。系统必须实例化目标环境并逐步执行，因此是**闭环转换**。

若开环地使用源环境当前位姿 $T_{\mathrm{src}}^{-1}(t)$，源、目标轨迹之间的小误差会逐步积累；闭环方式则每一步都根据目标环境的实际状态纠偏。笔者认为，这一设计把“动作重放”变成了“在目标动力学中跟踪同一任务意图”，才是它能跨控制器复用示范的关键。

在 PickSingleYCB、AssemblingKits、Write、TurnFaucet 上各随机选 100 条示范，从 `arm_pd_joint_pos` 转到 `arm_pd_ee_delta_pose` 后，成功率分别为 **99%、98%、100%、80%**。TurnFaucet 较低，是因为原策略用夹爪推把手，接触丰富且不一致，对执行误差尤其敏感。

## 五、实时软体仿真：让刚体机器人与 MPM 真正互相施力

控制器解决了“怎样下命令”，软体环境还要解决“命令怎样改变物质”。传统 FEM（有限元法）适合绳、布和弹性体，却不擅长挖面粉、切面团这类大形变与拓扑变化；NVIDIA Flex 能处理大形变，但对黏土这类弹塑性材料不够真实。PlasticineLab 使用 MPM，却缺少与刚体机器人的双向动力学耦合。

### MPM 与双向耦合的直觉

MPM（Material Point Method，物质点法）可以直觉地理解为：物质点负责携带质量、速度和材料状态，背景网格负责高效计算力学更新，最后再把速度传回粒子。它兼具粒子表达大形变的灵活性和网格求解的效率。

ManiSkill2 的耦合不是让软体“碰到机器人后单方面变形”，而是双向交换：

1. 初始化时，把刚体碰撞形状复制到软体场景。基础几何用解析 Signed Distance Function（SDF，有符号距离函数），Mesh 转为存于 3D CUDA Texture 的 SDF Volume。
2. 每个刚体步后，把刚体位姿和速度传给 MPM Simulator。
3. 在 MPM 粒子位置评估 SDF，对穿透施加惩罚力（penalty force），同时累计刚体受到的等效力与力矩。
4. 软体子步结束后，将累计力与力矩施加回 SAPIEN 刚体。

与 PlasticineLab 在 MPM 网格节点上评估 SDF 不同，ManiSkill2 选择在粒子上施力；作者观察到这样能减少穿透等伪影。由于接口只依赖位姿、速度、力与力矩，它原则上可以耦合其他刚体 Simulator，而不是绑定某一套机器人实现。

一个环境步可概括为：

```text
执行控制器 → 刚体步进 → 同步刚体位姿 → 多个 MPM 子步
→ 粒子到网格 → 网格速度更新 → 网格到粒子 → 粒子积分
→ 回传力与力矩 → 同步并渲染
```

### 为什么能实时：编译、传输与渲染一起优化

性能并非只来自“用了 GPU”。论文做了四层优化：

- 用 NVIDIA Warp 将 Python 代码即时编译（JIT）为原生 C++/CUDA，并直接使用 CUDA 实现高效求解器。
- 扩展 Warp 的主机—设备通信，减少 CPU/GPU 间位姿与力传输瓶颈。
- 使用 Warp 编译缓存，缩短环境编译和启动时间。
- 为视觉学习实现屏幕空间粒子渲染，避免每个粒子都先转为高成本三角网格。

渲染采用屏幕空间点溅射（screen-space splatting）：把粒子画成球，用双边滤波平滑深度缓冲，再在平滑后的深度上计算法线和光照。粒子位置从 Simulation 到 Renderer 只需一次 GPU-GPU Copy。屏幕空间滤波可能带来跨视角不一致，作者按像素到相机的距离缩放滤波器，在实践中取得较稳定的效果。

![未使用双边滤波的软体粒子渲染](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/before_smoothing.jpg)

> 图解：未滤波时，黏土表面清楚暴露出一颗颗球状粒子，深度与法线不连续，视觉输入带有明显颗粒伪影。

![使用双边滤波后的软体粒子渲染](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/after_smoothing.jpg)

> 图解：滤波后粒子之间的突变被压平，表面更接近连续黏土，同时尽量保留容器边界。这一处理服务于视觉策略输入，而不是改变底层 MPM 动力学。

论文使用的关键 MPM 参数如下，全部采用 SI 单位：

| Grid Length | Particle Volume | Density | Young's Modulus | Poisson Ratio | Yield Stress |
|---|---|---|---|---|---|
| 0.005–0.015 | $6.2\times10^{-8}$ / $1.2\times10^{-7}$ | 300–3000 | $10^4$ / $3\times10^5$ | 0.3 | $2\times10^3$ / $10^4$ |

带渲染的单个软体环境约 **17–18 FPS**；一张 RTX Titan 上并行 16 个环境可达 **80–84 FPS**，约为实时的 4 倍。收益主要来自 CPU 刚体与 GPU 软体顺序执行，以及多进程更充分地提交 CUDA Kernel；作者也将向量化软体仿真列为后续优化方向。

## 六、视觉 RL 提速：异步渲染只是第一步，共享显存同样关键

刚体和软体任务都具备后，视觉 RL 仍会被样本采集成本卡住。ManiSkill2 保留 Python 环境脚本和 SAPIEN，用多进程并行，而不是要求研究者用 C++ 重写任务；优化目标有两个：提高所有环境的总 FPS，并降低 GPU Memory 占用。

### 九步串行流水线里，浪费藏在哪里

典型多进程视觉 RL 采样依次执行：物理仿真 → 提交渲染 → 等待 GPU → 图像拷回 CPU → 计算奖励与其他状态 → 图像传到主进程并同步 → 再上传 GPU → Policy 前向 → 动作传回 Worker。

其中有两类明显浪费：

- 等待 GPU 渲染时 CPU 空闲，但奖励计算通常不依赖图像。
- 图像经历 GPU → CPU → 主进程 → GPU 的重复搬运；每个进程还各自保存 Mesh、Texture 等渲染资源。

**异步渲染（Asynchronous Rendering）**在提交 Draw Call 后立即计算奖励，不再空等图像。**Render Server** 则在主进程用线程池统一处理各 Worker 的渲染请求，让图像留在 GPU，并由中央资源管理器（Resource Manager）保证同一份模型、图片和纹理只加载一次。

![串行采样与异步 Render Server 采样流水线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/system_pipeline.png)

> 图解：（a）串行方案中，Worker 等待 GPU，并在 CPU/GPU 间反复复制图像；（b）ManiSkill2 将渲染请求交给主进程 Render Server，Worker 可继续计算奖励，图像不必回传 CPU，多环境还共享 GPU 渲染资源。Server 使用 gRPC、Protocol Buffers 与 HTTP/2，可通过网络或 Unix Socket 通信。

笔者认为，Render Server 的价值不只是“快”。它还让多进程 Vulkan 渲染回到单进程多线程的开发与性能分析模式，规避了 Linux 上 Nsight Systems 难以分析多进程 Vulkan 的问题，并为跨机器仿真与渲染留下网络接口。

### 吞吐量与显存实测

测试统一使用 PickCube、128×128 图像、500 Hz 物理仿真与 20 Hz 控制频率。硬件预算为 Intel i9-9960X 的 16 个逻辑核、128 GB 内存和一张 24 GB RTX Titan；每个控制步含 25 个物理步。不同框架的仿真与渲染保真度无法完全一致，因此数字应理解为该设置下的工程对比，而非绝对物理性能排名。

**采样吞吐量：**

| 指标 | ManiSkill2 Server | ManiSkill2 Sync | Habitat 2.0 | RoboSuite 1.3 | Isaac Gym |
|---|---:|---:|---:|---:|---:|
| Total FPS，随机动作 | **2487±24** | 942±19 | 1275±10 | 924±3 | 865±35 |
| Total FPS，Nature CNN | **2532±63** | 931±4 | 1224±13 | 894±15 | 835±5 |
| 最优并行环境数 | 64 | 32 | 64 | 32 | 512 |

**每个场景含 74 个 YCB 物体时的 GPU 显存：**

| 并行环境数 | ManiSkill2 Server | Habitat 2.0 |
|---:|---:|---:|
| 4 | 4.9 GB | 6.4 GB |
| 8 | 5.1 GB | 12.9 GB |
| 64 | 5.8 GB | OOM |

ManiSkill2 从 4 个环境扩到 64 个，显存只增加 0.9 GB；Habitat 即使启用了纹理压缩，占用仍随环境数近似线性增长。省下的显存可以留给更大的网络或 Batch，而不是重复保存同一份纹理。

![使用 Nature CNN 时不同并行环境数的采样速度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/CNNE.png)

> 图解：横轴是并行环境数，采用对数式刻度；纵轴是所有环境合计的 Overall FPS。蓝线 ManiSkill2-Server 在 64 个环境附近达到约 2500 FPS；绿线仅启用异步渲染也已超过其他框架。部分曲线在高并行度处中断，源于性能下降、显存溢出或崩溃，不能将缺失段外推为持续扩展。

![使用随机动作时不同并行环境数的采样速度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/RE.png)

> 图解：横纵轴含义与上图一致，但动作不经过 CNN。ManiSkill2-Server 同样在 64 个环境附近接近 2500 FPS，高并行度后略有回落；这说明 Render Server 的收益并非只在 Policy 前向较慢时出现。

一个反直觉现象是，ManiSkill2 在并行环境数超过 CPU 核数时表现最好。作者推测，当请求的计算量超过可用资源，操作系统或驱动调度反而能更充分地交错 CPU 与 GPU 工作。

## 七、基准是否足够难：三类算法给出的答案

系统速度快、任务数量多，只能说明“能用”。基准是否有研究价值，还要看它能否区分算法能力，并暴露尚未解决的问题。论文分别测试经典 Sense-Plan-Act（SPA）、Behavior Cloning（BC）和示范增强 RL。

### SPA：感知高分不等于物理成功

SPA 像一条“看清世界—规划动作—执行计划”的流水线。它通常开环运行，感知与规划分别优化，因此上游预测的微小偏差可能在下游被放大。

**Contact-GraspNet + Motion Planning** 在 PickSingleYCB 上先从部分点云预测候选抓姿及置信度，再按分数从高到低尝试运动规划，找到可执行抓姿后抓取并移动到目标。74 个物体各测试 5 次，物体中心到达目标 2.5 cm 内即成功，最终成功率为 **43.24%**。

失败分为两类：27.03% 的试验中预测抓姿置信度低，尤其常见于勺、叉等细小物体；29.73% 的试验中抓姿置信度高，但抓取质量差或机器人不可达。

![Contact-GraspNet 成功轨迹：初始状态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/correct_grasp_0.png)

> 图解：机械臂、罐体和绿色目标位置构成初始场景，后续需要同时满足抓姿质量与机械臂运动学约束。

![Contact-GraspNet 成功轨迹：接近抓姿](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/correct_grasp_1.png)

> 图解：夹爪从上方接近预测抓姿，说明该候选不仅感知分数高，也通过了运动规划。

![Contact-GraspNet 成功轨迹：闭合夹爪](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/correct_grasp_2.png)

> 图解：夹爪在罐体上方闭合并形成稳定抓取，随后才能进入搬运阶段。

![Contact-GraspNet 成功轨迹：移向目标](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/correct_grasp_3.png)

> 图解：机械臂将罐体移向绿色目标位置，完成“感知—规划—抓取—放置”的完整任务链条。

![失败示例：抓姿旋转预测错误](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/incorrect_grasp_pred_0.png)

> 图解：错误的旋转预测让夹爪无法形成有效抓取，表明点云抓姿估计本身就可能是瓶颈。

![失败示例：夹爪未充分接近物体](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/incorrect_grasp_pred_1.png)

> 图解：旋转方向合理，但夹爪与物体距离不合适；姿态预测的单个分量正确，并不足以保证成功。

![失败示例：摩擦与几何导致滑脱](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/incorrect_grasp_pred_2.png)

> 图解：看似合理的抓姿在夹爪闭合时会因瓶身几何和摩擦而滑脱，全动力学仿真因此暴露了抽象抓取会隐藏的问题。

![失败示例：运动学不可达](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/cgn/incorrect_grasp_pred_3.png)

> 图解：抓姿从物体几何看合理，机械臂却无法通过运动规划到达。这正是大工作空间随机目标对 SPA 流水线的额外约束。

**Transporter Networks（TPN）** 在 AssemblingKits 上暴露了另一个问题：指标松紧会彻底改变结论。作者使用官方代码，以底座相机和手部相机生成融合点云，再渲染俯视正交图训练 TPN；旋转预测的离散区间数从 36 增至 144，并分别测试两指夹爪与吸盘。下表中 Pos 和 Rot 分别表示位置与旋转误差阈值。

| 资产划分 | Gripper | 本文真实入槽成功率 | 原论文位姿成功率 | Pos <5 mm | Pos <2.5 mm | Pos <1 mm | Rot <4° | Rot <1° | Rot <0.25° |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 训练资产 | 两指夹爪 | 0.18 | 0.98 | 0.98 | 0.80 | 0.30 | 0.96 | 0.48 | 0.22 |
| 训练资产 | 吸盘 | 0.15 | 0.93 | 0.89 | 0.66 | 0.18 | 0.91 | 0.48 | 0.22 |
| 测试资产 | 两指夹爪 | 0.18 | 0.99 | 0.99 | 0.82 | 0.31 | 0.96 | 0.48 | 0.24 |
| 测试资产 | 吸盘 | 0.14 | 0.96 | 0.92 | 0.66 | 0.17 | 0.94 | 0.52 | 0.31 |

原指标只要求距目标 1 cm、15° 内，测试资产两指夹爪因此有 99% 成功率；换成真正入槽后只剩 18%。这个巨大落差不是模型突然失效，而是**宽松位姿指标无法代表毫米级接触任务已经完成**。

### BC：会模仿轨迹，不等于会估计物质变化

BC 使用 50k Gradient Steps、Batch Size 256。RGBD 策略以 IMPALA 为视觉 Backbone，拼接底座与手部相机的 128×128 图像；点云策略使用 PointNet，将多相机点云融合到机器人基座系，去地面并随机采样 1200 点。刚体任务进一步把点云、本体状态和目标信息变换到末端坐标系；PickCube 与 PickSingleYCB 还加入目标附近的 50 个绿色点作为视觉提示。

刚体示范通常使用 1000 条轨迹，多资产任务使用 PickSingleYCB 7300、TurnFaucet 4500、AssemblingKits 1700 条；软体任务通常使用 200 条，Pinch 使用 1550 条。转换后只保留成功轨迹。

| 观测 | PickCube | StackCube | Fill | Hang | Excavate | Pour | Pinch | Write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Point Cloud | 0.22±0.06 | 0.04±0.02 | 0.45±0.04 | 0.35±0.15 | 0.08±0.04 | 0.02±0.02 | 0.00±0.00 | 0.00±0.00 |
| RGBD | 0.01±0.02 | 0.00±0.00 | 0.62±0.07 | 0.20±0.12 | 0.21±0.04 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |

表中为 5 次实验的成功率均值与标准差，未列出的刚体装配任务成功率均为 0。Fill 和 Hang 相对容易，是因为成功条件允许较大的物质量或形变误差；Excavate 要估计挖取量，Pour 要控制精确液面，Pinch 与 Write 要利用目标形状引导细粒度形变，BC 就很难只靠轨迹模仿完成。

![Pinch 与 Write 上的 Behavior Cloning 失败示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/soft_bc.png)

> 图解：BC 已学到捏压和描绘的基本动作，也能朝目标取得部分进展，但最终形状与目标仍相差较远。失败并非“完全不动”，而是无法准确估计动作对软体形变的细粒度影响。

### DAPG+PPO：抓放已有高分，精密装配仍接近零

示范增强 RL 将近端策略优化（PPO）的在线 Rollout 目标与示范增强策略梯度（DAPG）的示范目标结合。令 $\rho_\pi$ 为在线策略转移分布，$\rho_D$ 为示范转移分布，$r_t(\theta)=\pi_\theta(a\mid s)/\pi_{\theta_{\mathrm{old}}}(a\mid s)$，核心策略损失为：

$$
\mathcal L_{\rho}^{\mathrm{CLIP}}(\theta)=-\mathbb E_{(s,a)\sim\rho}\left[\min\left(r_t(\theta)\hat A(s,a),\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A(s,a)\right)\right]
$$

$$
\mathcal L_{\mathrm{DAPG+PPO}}(\theta)=\mathcal L_{\rho_\pi}^{\mathrm{CLIP}}(\theta)-\omega\,\mathbb E_{(s,a)\sim\rho_D}\left[\pi_\theta(a\mid s)\right]
$$

示范权重随 PPO 的 Epoch 数 $N$ 衰减：$\omega=0.1\cdot0.995^N$。这使训练初期更多借助示范，后期逐渐转向在线交互。所有 Agent 从头训练 25M Time Steps；受算力限制，论文只报告刚体环境的 RL 结果。

| 观测 | PickCube | StackCube | PickSingleYCB | PegInsertionSide | PlugCharger | AssemblingKits | TurnFaucet | AvoidObstacles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Point Cloud | **0.94±0.03** | **0.95±0.02** | **0.51±0.05** | 0.01±0.01 | 0.01±0.02 | 0.00±0.00 | 0.04±0.03 | 0.00±0.00 |
| RGBD | 0.91±0.05 | 0.87±0.04 | 0.18±0.07 | 0.01±0.01 | 0.01±0.01 | 0.00±0.00 | 0.03±0.03 | 0.00±0.00 |

点云策略在 PickSingleYCB 训练物体上达到 0.51，超过 CGN+Motion Planning 的 0.4324，说明单一在线策略有潜力跨多样几何学习抓放。但精密装配、TurnFaucet 和 AvoidObstacles 仍接近零，高分没有扩散到所有操作类型。

真正考验泛化的是 held-out 物体。下表显示，训练资产上的提升不能直接外推到新资产：

| 观测 | PickSingleYCB | AssemblingKits | TurnFaucet | AvoidObstacles |
|---|---:|---:|---:|---:|
| Point Cloud | 0.36±0.06 | 0.00±0.00 | 0.01±0.01 | 0.00±0.00 |
| RGBD | 0.08±0.03 | 0.00±0.00 | 0.01±0.01 | 0.00±0.00 |

### 消融：控制器、坐标系与视觉提示都不是细节

PickSingleYCB 点云 DAPG+PPO 的消融进一步解释了点云策略为何有效：

| 设置 | 成功率 |
|---|---:|
| 原始：末端位姿增量 + 末端坐标系 + 目标视觉提示 | **0.51±0.05** |
| 改为关节位置增量控制器 | 0.22±0.18 |
| 改用机器人基座坐标系表示点云 | 0.00±0.00 |
| 移除目标视觉提示 | 0.16±0.07 |

同一任务仅更换控制器，成功率就从 0.51 降到 0.22；换错点云参考系更是降到 0。由此回看多控制器示范转换，它不是附加的便利功能，而是让研究者能系统选择学习动作空间、又不浪费既有示范的基础设施。

装配任务还进行了“默认间隙 1×”与“十倍间隙 10×”对照。1× 分别对应 PegInsertionSide 的 3 mm 和 PlugCharger 的 0.5 mm。

| 算法 | 观测 | PegInsertionSide 1× | PegInsertionSide 10× | PlugCharger 1× | PlugCharger 10× |
|---|---|---:|---:|---:|---:|
| BC | Point Cloud | 0.00±0.00 | 0.01±0.01 | 0.00±0.00 | 0.01±0.01 |
| BC | RGBD | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| DAPG+PPO | Point Cloud | 0.01±0.02 | **0.74±0.10** | 0.01±0.01 | 0.02±0.02 |
| DAPG+PPO | RGBD | 0.01±0.01 | 0.05±0.03 | 0.01±0.01 | 0.29±0.07 |

放宽间隙后，部分成功率大幅提升，证明毫米精度确实是难点来源之一；但 RGBD PegInsertionSide 和 Point Cloud PlugCharger 依然很低，说明目标槽遮挡、接触探索和视觉表达等问题不会因间隙变大就自动消失。

### 复现实验所需的主要 Hyperparameters

RGBD 的 IMPALA Backbone 类似 ResNet10，各层 Hidden Size 为 64，去掉 Normalization，首层卷积改为 Kernel 4、Stride 4，最终投影为 384 维再拼接状态向量。PointNet 在 Max Pooling 前使用 [64,128,512] 隐层，中间层输出加 Layer Normalization。BC 与 DAPG+PPO 都用 Adam，Learning Rate 为 $3\times10^{-4}$；点云策略输出头前的 Linear Layer 初始化为零，以稳定训练。

| Hyperparameter | Value |
|---|---|
| Discount $\gamma$ | 0.95 |
| GAE $\lambda$ | 0.95 |
| PPO Clip Range | 0.2 |
| Max KL | 0.1 |
| Gradient Norm Clipping | 0.5 |
| Entropy | 0.0 |
| 每次 PPO Step 样本数 | 20000 |
| 每个 Minibatch 样本数 | 300 |
| Critic Warmup Epochs | 4 |
| PPO Update Epochs | 2 |
| Critic Loss Coefficient | 0.5 |
| Demonstration Loss Coefficient | $0.1\cdot0.995^N$ |
| 每轮 PPO 更新后重算 Advantage | True |
| Reward Normalization | True |
| Advantage Normalization | True |
| 只用 Critic Loss 更新视觉 Backbone | True |
| Rollout 成功后立即 Reset 环境 | False |

## 八、Sim2Real：可迁移，但差距仍然清楚

仿真实验证明基准能区分算法，最后还要问：这些动力学与视觉观测是否能连接真实世界？论文没有做大规模现实 Benchmark，而是给出两个直接验证。

### PickCube：91% 到 60%

真实平台由 Intel RealSense D415、7-DoF ROKAE xMate3Pro、Robotiq 2F-140 夹爪和目标立方体组成。作者标定相机内参与相对机器人基座的位姿，在仿真中对齐真实场景，再训练 10M Time Steps 的点云策略。输入不含颜色，只含深度反投影点云、关节位置和末端位姿。

策略在仿真中的成功率为 **91.0%**，不做现实微调，直接以不同初始状态测试 50 次，现实成功率为 **60.0%**。训练只使用高斯噪声和随机像素丢弃做 Depth Data Augmentation，作者因此将下降主要归因于深度图的 Domain Gap。

![PickCube 的仿真与真实机器人设置](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/pickcuble_sim2real-cropped.png)

> 图解：左右对照仿真与真实平台的相机、机械臂、夹爪和立方体布局。实验重点是同一训练策略直接部署，而不是在现实环境继续训练；91% 与 60% 的差距也应被理解为仍需缩小，而非已经消除的 Sim2Real Gap。

### Pinch：同一动作序列能否得到相似形变

软体迁移实验不评估学习策略，而是在仿真与真实橡皮泥上执行同一套由 Motion Planning 生成的动作序列，再比较最终形状。

![Pinch 在仿真与真实世界中的形变对照](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/ManiSkill2-A-Unified-Benchmark-for-Generalizable-Manipulation-Skills/figures/sim2real_deformable.png)

> 图解：从初始状态出发，依次形成字母“M”和“S”的形状。真实橡皮泥在多次抓取后的形变与双向刚体—MPM 仿真大体一致，说明耦合 Simulator 能合理再现这类弹塑性交互，但这仍是定性结果，不等于所有软体材料都已精确标定。

## 九、结论：ManiSkill2 真正统一了什么，又还缺什么

- **任务与协议统一**：20 个任务族、2000+ 物体、4M+ 示范帧覆盖刚体/软体、固定/移动、单臂/双臂，并统一 Gym 接口、观测与两阶段评测。
- **示范与控制器解耦**：闭环动作空间转换在四个代表任务上取得 99%、98%、100%、80% 的成功率，控制器消融也证明动作空间会实质影响学习。
- **刚体—软体双向交互**：Warp-MPM 与 SAPIEN 双向耦合，配合粒子渲染，在单卡 16 个软体环境上达到 80–84 FPS。
- **视觉采样兼顾速度与显存**：异步渲染与 Render Server 在指定硬件上取得最高 2532±63 FPS，64 个含 74 个 YCB 物体的环境仅占 5.8 GB 显存。
- **基准暴露了真实难点**：BC 与 DAPG+PPO 在精密装配、held-out 资产和精细软体控制上仍大量失败，宽松位姿指标会显著高估装配能力。

ManiSkill2 的局限同样明确：照片级真实感会牺牲速度，Domain Randomization 只覆盖部分任务，场景级变化和布料等可变形物体仍不足；因此它更适合作为低层、短视界、全物理操作的高效试验场，而场景级长程任务、向量化软体仿真和更可靠的 Sim2Real 仍是下一步方向。项目代码、环境、Simulator、Renderer、示范生成与学习框架均已开源，资产生成和云评测服务相关代码则因竞赛公平性未发布。

> 本文参考自 [ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills](https://arxiv.org/abs/2302.04659)