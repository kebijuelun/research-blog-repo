# Physical RSI 1.0：递归自我改进登顶 RoboDojo

具身智能（Embodied AI）有个长期痛点：机器人策略一旦训练完成，能力就基本冻结，遇到新任务只能重新采数据、重新训练。这篇工作提出的 **Physical RSI 1.0** 给出了另一条路——让机器人利用自己在物理世界中获得的成功与失败反馈，递归地重写自己的「脚手架」（Harness）代码，实现技能的自我进化。核心思路一句话：冻结的多模态大模型充当 System 2 负责理解、决策与改写，可编辑的代码策略充当 System 1 负责高频执行，二者原生协同进化。结果相当硬：截至 2026 年 10 月 8 日，Physical RSI 在极具挑战的 **RoboDojo-Sim 榜单上排名第一** 。

![figure-1.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-1.svg)

> 图解：RoboDojo 基准涵盖多样化的操作任务，作者将其称为具身智能的「珠穆朗玛峰」。Physical RSI 1.0 登顶 RoboDojo-Sim 排行榜，相当于为登顶珠峰开辟了一条新路线——不靠更大的训练数据，而靠系统自身的递归进化。

## 提出问题：System 1 与 System 2 的割裂

具身智能现有两条主流路线，各自有一个致命短板。

**路线一：System 2 直接干活。** 让 GPT 这类大模型直接输出低层动作 $a_1, a_2, a_3, \dots$。大模型推理频率只有约 0.1–1 Hz，而物理世界的控制需要 20–50 Hz——让「大脑」去干「小脑」的活，节奏完全跟不上。

![figure-2.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-2.svg)

> 图解：System 2（GPT）直接接管低层控制。上方是「观测 + 目标」输入，下方输出逐帧低层动作到物理世界。0.1–1 Hz 的推理频率面对高频物理交互，如同让一个深思熟虑的棋手去接手乒乓球对打。

**路线二：System 1 单打独斗。** VLA（Vision-Language-Action）或 WAM（World-Action Model）这类端到端策略可以跑到 20–50 Hz，控制频率够了，但推理能力和泛化能力有限——它们本质上是「肌肉记忆」，遇到需要多步推理、记忆或符号计算的任务就会露馅。

![figure-3.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-3.svg)

> 图解：纯 System 1 方案。VLA / WAM 接收观测后以 20–50 Hz 输出动作，频率达标，但短板明确标注在图下方：有限的推理与泛化能力。

两条路一个「想得太慢」，一个「想得太少」。问题自然就来了：能不能让两者各司其职，并且让这种分工本身也能不断进化？

## 核心思路：System 1–2 原生协同进化

Physical RSI 的答案是 **Native System 1–2 co-evolution（System 1–2 原生协同进化）** ：多模态基础模型作为 System 2，以 0.1–1 Hz 的频率做规划；System 1 则以 20–50 Hz 的频率执行，其内部是 **Skills（VLA + 程序代码）+ Tools + Physical API** 的组合。

关键在于右侧那个回路：System 2 会 **反思（Reflecting）** System 1 的执行表现，并改写 System 1 的代码，让执行体从 v1 进化到 v2、v3…… 这就是「协同进化」的含义——大脑不仅指挥小脑，还负责给小脑「升级固件」。

![figure-4.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-4.svg)

> 图解：System 1–2 协同架构。左半部分：System 2（多模态基础模型）接收「观测 + 目标」做低频规划；右半部分：System 1 高频执行，内部由技能（VLA + 程序）、工具和物理 API 构成。右侧的 Reflecting 回路是全文核心：System 2 观察执行结果并改写 System 1，版本号 v1 → v2 → … 不断滚动。

笔者认为这个设计聪明的地方在于：它把「学习能力」从模型权重里挪到了 **代码** 里。改代码不需要梯度、不需要重新训练，一次反思就是一次进化，成本极低、可读性极高。

## 主算法：具身自我 Harness（Embodied Self-Harness）

明确了分工之后，下一个问题是：进化的具体机制是什么？作者把它形式化为一个递归过程。

在第 $k$ 代，完整的智能体是基础模型 $F$ 通过 Harness $H_k$ 运作：

$$
A_k = \mathrm{Agent}(F, H_k), \qquad H_{k+1} = \mathrm{Improve}_{A_k}(H_k, \tau_k)
$$

其中各符号含义：

- $F$：System 2，负责理解、决策与改写的基础模型；
- $H_k$：智能体可编辑的 Harness——为 System 1 执行服务的路由、技能、工具与控制代码；
- $\tau_k$：具身反馈，包括观测、动作、成功与失败记录；
- $\mathrm{Improve}$ 算子由四步构成： **vary（变异）→ evaluate（评估）→ select（选择）→ inherit（继承）** 。

用一句话概括： **用你的 Harness，重写它，再带着它上场。**

```python
H[0] = initial_harness()                    # Start
for k in range(budget):
    A[k] = Agent(F, H[k])                   # S2 + H
    plan = A[k].understand(task)            # S2
    τ[k] = H[k].execute(plan)               # S1
    variants = A[k].rewrite_many(H[k], τ[k]) # S2 → H
    pool = [Agent(F, h) for h in variants]  # Candidates
    survivor = select_on_eval([A[k], *pool]) # Selection
    H[k+1] = survivor.harness               # Inherit
```

![figure-9.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-9.svg)

> 图解：自我 Harness 的单代循环。System 2 检查（inspect）当前 Harness 并改写（rewrite）出 $H_{k+1}$，候选 Harness 被调用（invoke）后在环境中跑出观测 $o_{t+1}$，评估产生的成功/失败证据 $E^+, E^-$ 回流给 System 2，驱动下一轮改写。

### 达尔文式自我 Harness

这套机制的本质是 **达尔文式进化** ：一个智能体重写自己的 Harness，派生出多个候选智能体；由具身表现做自然选择，最适应的 Harness 存活到下一代。

**自我重写 → 候选智能体 → 适者生存 → 循环往复。**

作者引用了爱因斯坦的话作为设计信条：「Everything should be made as simple as possible, but no simpler.（一切应尽可能简单，但不能更简单。）」这句话也贯穿了后文的极简实现。

## 极简实现：两条公式跑通全流程

原理讲完，落地可以非常克制。整个 Physical RSI 1.0 只用两个函数：

**System 2 负责改写：**

$$
H_{k+1} = F(H_k, E_k^+, E_k^-)
$$

**System 1 负责执行：**

$$
(a_t, m_{t+1}) = H_k(o_t, g, m_t; S, T)
$$

符号约定：$F$ 是多模态智能体，$H$ 是控制代码，$E^+ / E^-$ 是成功/失败证据，$S / T$ 是技能/工具，$o / g / m / a$ 分别是观测、目标、episode 状态与动作，$k$ 与 $t$ 区分改写步与控制步。

![figure-10.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-10.svg)

> 图解：Physical RSI 1.0 的极简实现全景。System 2 的「改写」以代（$k$）为单位缓慢迭代，System 1 的「执行」以控制步（$t$）为单位高速运转，两者通过证据 $E^+/E^-$ 和 Harness 代码耦合。

没有新网络结构，没有新损失函数——笔者认为这正是「1.0 版本」的姿态：先证明「递归自我 Harness」这条路线本身能 Scale，工程上足够简单才有人愿意复现、才谈得上成为 baseline。

## 自我进化的四种形态

Harness 能改写的结构不止一种。作者用四个 RoboDojo 任务展示了自我进化的多样性： **表征、状态、规划、控制** ，每一层都可以成为 $\Delta H$（Harness 的修订量）的落点。

### 表征层进化：数字排序任务

任务要求识别散落数字并按大小摆入槽位。底层电机策略 $\pi_{0.5}$ 会在干扰物之间盲目抓取，无法完成整行数字的摆放。

进化后的 Harness 增加了 **数字定位、排序与槽位分配** 的表征结构：

$$
\sigma = \mathrm{argsort}(d, \downarrow), \qquad p^{*}_i = p_0 + i \, \Delta x \, \hat{e}_x
$$

其中 $d$ 是数字取值，$p^{*}_i$ 是第 $i$ 个目标槽位。识别数字 → 按值排序 → 映射到摆放槽位，三步表征把「瞎抓」变成了「有计划的排列」。

![figure-5.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-5.svg)

> 图解：数字排序任务的几何示意与独立 rollout 视频。$\Delta H$ 处标注了本次 Harness 修订：新增数字定位、排序与槽位分配模块。

### 状态层进化：遮挡记忆任务

任务中杯子会盖住积木，之后要求揭开指定顺序的杯子。失败案例里，最后一次抓杯失误，蓝色积木始终被盖着——因为系统「忘记」了哪个杯子盖着哪个积木。

进化后的 Harness 学会 **跨遮挡保持物体–杯子的对应关系** ：

$$
o_0 \rightarrow m = \{ c_i \mapsto p_i \}, \qquad p_{\text{target}} = m(c_{\text{goal}})
$$

其中 $c_i$ 是物体身份，$p_i$ 是存储的位置。先记住物体与杯子的位置映射，再按请求的序列逐一揭开。

![figure-6.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-6.svg)

> 图解：遮挡记忆任务。$\Delta H$ 处标注：跨遮挡保留物体–杯子对应关系。这本质上是给 Harness 加了一个「显式记忆字典」，而不是指望策略网络自己记住。

### 规划层进化：交换两个 T 形块

任务要求交换两个 T 形块并保持目标朝向。朴素做法两块都动了，但最终朝向对不上。

进化后的 Harness **插入了一个缓冲位姿（buffer pose）阶段** ，三段转运、全程携带朝向信息：

$$
T_A \rightarrow T_{\text{buf}}; \qquad T_B \rightarrow T_A; \qquad T_{\text{buf}} \rightarrow T_B
$$

其中 $T = (R, p)$ 表示朝向与位置，$T_{\text{buf}}$ 是空出的缓冲位。这是经典的「交换两个变量需要临时变量」——规划结构本身被 Harness 悟了出来。

![figure-7.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-7.svg)

> 图解：T 形块交换任务。$\Delta H$ 处标注：插入缓冲位姿阶段并在每次搬运中保持目标朝向。借缓冲位完成三步转运，是规划层面的结构性修订。

### 控制层进化：钥匙插锁

插入类任务对误差极敏感。进化后的 Harness 把 **接近（approach）、对准（align）、插入（insert）拆成三个控制阶段** ，每阶段有明确的容差判据：

$$
T_{\text{key}} \rightarrow T_{\text{lock}}: \quad \|\Delta p\| < \varepsilon_p, \quad \theta(\Delta R) < \varepsilon_r
$$

其中 $\Delta p$ 是位置误差，$\Delta R$ 是朝向误差，$\varepsilon$ 是容差。

![figure-8.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-8.svg)

> 图解：钥匙插入任务的腕部视角与头部视角（同一成功 episode 的两个机位）。腕部视角呈现钥匙–锁孔的局部几何关系，头部视角呈现接近、对准、插入的完整流程。$\Delta H$ 处标注：将控制拆分为三个阶段。

四个案例放在一起看，进化的「粒度」从感知表征一直覆盖到控制律——这正是「Harness」这个抽象的价值：它是代码，代码的哪一层都能改。

## Harness 记忆：技能常驻，状态刷新

解决了「怎么改」之后，还有一个工程细节：跨 episode 什么该保留、什么该重置？Physical RSI 的原则是 **保留技能，刷新状态（Keep the skill, refresh the state）** 。

以叠衣服为例，一次「反思 + 改写」后沉淀下来的代码策略技能会持久保存：

```python
def fold_sleeve(arm, pick, place):
    arm.move_above(pick)
    home = arm.joints().copy()
    arm.move_to(pick)
    arm.close_gripper()
    arm.move_above(place)
    descent = [arm.joints().copy()]
    for q in arm.lower_to(place):
        descent.append(q.copy())
    arm.open_gripper()
    for q in reversed(descent):
        arm.move_joints(q, step=0.12)
    arm.move_joints(home, step=0.12)
```

而 episode 状态（本次的具体位姿、中间记录）在再次运行时重置。上面这段 `fold_sleeve` 就是一次成功进化留下的「肌肉记忆」：记录下降轨迹，松手后沿原轨迹反向撤出，最后归位——一个纯代码实现、可审查、可复用的折叠技能。

![figure-11.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-11.svg)

> 图解：Harness 记忆机制示意。技能代码（code-policy skills）跨 episode 持久存在，episode 状态则在每次运行时重置。记忆留在「怎么做」，不留在「这次做了什么」。

## 技能继承与组合：从单体技能到能力网络

单个技能沉淀下来之后，更进一步的问题是：技能之间能否互相成就？Physical RSI 展示了两种机制。

### 技能继承：跨任务搬运视觉定位能力

「通用抓取（general pickup）」与「插充电器（plug in charger）」是两个任务，但后者可以直接继承前者的像素–世界定位模块 $\phi$：

$$
\phi_B \equiv \phi_A, \qquad \pi_B = H_B[\phi_A, \sigma_{\text{grasp}}, \sigma_{\text{align}}, \sigma_{\text{insert}}]
$$

$A$ 是抓取任务，$B$ 是充电器插入任务，$\phi$ 是像素到世界的定位，$\sigma$ 是电机技能。继承的方式朴素得可爱—— **直接 import 兄弟模块** ：

```python
from ..general_pickup.rgb_perception import (
    pixel_to_world_on_height as ground,
)
x, y = ground(pixel, world_z=surface_z)
```

笔者认为这个细节最能体现代码策略的红利：技能复用就是软件工程里的模块导入，不需要蒸馏、不需要微调。

### 技能组合：解方程并摆放数字

任务「解方程 $8 \times \square = 0$ 并把缺失数字放到槽位」需要把读取、求解、抓取、移动、摆放串成一条链：

$$
\pi_{\text{eq}} = \sigma_{\text{place}} \circ \sigma_{\text{move}} \circ \sigma_{\text{pick}} \circ f \circ \phi
$$

$\phi$ 负责「读」，$f$ 负责「解」，$\sigma$ 是电机技能，$\circ$ 是在任务状态上的顺序组合。代码策略草图如下：

```python
scene = read_equation(observation)
digit, slot = solve(scene)          # 8 × ? = 0 → 0
held = pick(scene.object(digit))
assert attached(held, observe())
move(held, above(slot))
if aligned(held, slot, observe()):
    place(held, slot)
```

注意其中的 `assert` 和 `if`——代码策略天然支持在关键节点插入运行时校验，这是神经网络策略很难给的「安全带」。

## 实验验证：RoboDojo 榜首的含金量

方法讲完，看成绩。以下结果均来自 RoboDojo 官方排行榜（2026 年 10 月 3 日快照，10 月 8 日核验），Physical RSI 使用自家电机策略工具 $\pi_{0.5}$，对比对象包括 VLA 路线的 DM0.5 与 WAM 路线的 Liber-0 Lite。

![figure-12.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-12.svg)

> 图解：类别维度对比（官方榜单 2026 年 10 月 3 日快照）。Physical RSI（搭载 $\pi_{0.5}$ 电机策略工具）与 VLA 模型 DM0.5、WAM 模型 Liber-0 Lite 在各类任务上的整体表现对比。

![figure-13.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-13.svg)

> 图解：任务聚焦视图与任务级官方结果分布，覆盖 52 个「任务 × 布局」设置，按 Generalization（泛化）、Long horizon（长程）、Memory（记忆）、Open（开放）、Precision（精细）五类组织。

52 个设置的完整任务级结果如下（SR 为成功率，Score 为榜单得分；标注「不完整 rollout」的条目官方数据尚未跑完，供参考）：

| 任务 | 类别 | SR | Score | 备注 |
| --- | --- | --- | --- | --- |
| arrange largest number | Generalization | 1.33% | 3.87 | |
| arrange largest number random | Generalization | 21.33% | 32.47 | 随机布局 |
| fold clothes | Generalization | 41.33% | 48.80 | |
| fold clothes random | Generalization | 34.67% | 43.47 | 随机布局 |
| hang mugs | Generalization | 0% | 8.80 | |
| make toast | Generalization | 0% | 6.00 | |
| pack objects into box | Generalization | 5.33% | 26.27 | |
| pack objects into box random | Generalization | 0% | 15.07 | 随机布局 |
| pour liquid into cup | Generalization | 93.33% | 93.33 | |
| pour liquid into cup random | Generalization | 9.33% | 9.33 | 随机布局 |
| push T | Generalization | 65.33% | 65.33 | |
| sort nesting dolls by size | Generalization | 16% | 16.00 | |
| stack blocks | Generalization | 9.33% | 18.93 | |
| stack bowls | Generalization | 74.67% | 78.47 | |
| stack bowls random | Generalization | 0% | 10.00 | 随机布局 |
| store laptop and headphones | Generalization | 1.33% | 13.07 | |
| store laptop and headphones random | Generalization | 1.33% | 13.07 | 随机布局 |
| sweep blocks | Generalization | 0% | 0.00 | |
| classify objects | Long horizon | 40% | 55.83 | |
| fill pen holder | Long horizon | 3.33% | 21.87 | |
| make kong | Long horizon | 82% | 82.00 | |
| organize table | Long horizon | 4% | 27.33 | |
| play tic tac toe | Long horizon | 94% | 98.33 | |
| put bottles into dustbin | Long horizon | 75.33% | 83.70 | |
| cover blocks | Memory | 100% | 100.00 | |
| match and pick from conveyor | Memory | 7.33% | 7.33 | |
| press by number | Memory | 79.33% | 79.33 | |
| swap T | Memory | 92.67% | 92.67 | |
| swap blocks | Memory | 0% | 0.00 | |
| align blocks | Open | 86% | 86.00 | |
| classify objects by language | Open | 6% | 20.93 | |
| general pickup | Open | 47.33% | 47.33 | |
| solve equation | Open | 60% | 60.00 | |
| build tower | Precision | 28.67% | 37.13 | |
| deposit coin | Precision | 56% | 62.27 | |
| fasten screws | Precision | 0% | 7.60 | |
| insert key | Precision | 39.33% | 48.43 | |
| insert tubes | Precision | 66% | 78.93 | |
| plug in charger | Precision | 30.67% | 30.67 | |
| pour balls into vase | Precision | 38.67% | 38.67 | |
| hang mugs random | Generalization | 0% | 2.00 | 随机布局，不完整 rollout |
| make toast random | Generalization | 0% | 4.67 | 随机布局，不完整 rollout |
| sort nesting dolls by size random | Generalization | 0% | 0.00 | 随机布局，不完整 rollout |
| stack blocks random | Generalization | 0% | 2.40 | 随机布局，不完整 rollout |
| fill egg holder | Long horizon | 0% | 1.90 | 不完整 rollout |
| play stacking toy | Long horizon | 0% | 0.00 | 不完整 rollout |
| imitate sorting sequence | Memory | 0% | 1.10 | 不完整 rollout |
| pick from conveyor by image | Open | 0% | 0.00 | 不完整 rollout |
| pour by language | Open | 0% | 10.13 | 不完整 rollout |
| stack blocks by language | Open | 0% | 0.00 | 不完整 rollout |
| store tools in toolbox | Open | 0% | 6.67 | 不完整 rollout |
| play xylophone | Precision | 0.67% | 0.67 | 不完整 rollout |

几个值得注意的数字：

- **记忆类任务全面爆发** ：cover blocks 拿到 **100% SR / 100 分** ，swap T 92.67%，press by number 79.33%——这正是 Harness 显式状态管理的直接红利；
- **长程任务表现突出** ：play tic tac toe 94% SR / 98.33 分，make kong 82%，put bottles into dustbin 75.33%；
- **短板同样诚实** ：sweep blocks、swap blocks、fasten screws 等任务仍是 0%，随机布局下多项成绩大幅下滑（如 pour liquid 93.33% → 9.33%），说明泛化与精细操作仍是硬骨头。

笔者认为这种「不遮丑」的完整榜单反而增强了说服力：一个还在进化的系统，留下 0% 的任务正是下一代 Harness 的进化空间。

## 具身技能库：进化的化石记录

所有进化成果最终沉淀为一个可浏览的技能库： **212 段录制剪辑，覆盖 40 个任务** ，每段剪辑都可回溯到具体的任务、相机、episode 与时间窗口。

![figure-14.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/PhysicalRSI1.0-Recursive-Self-Harnessfor-Scaling-Embodied-Skills/images/figure-14.svg)

> 图解：技能库图谱示意。一个点对应一段录制剪辑，一条路径对应一个任务序列；S 表示技能库规模，L 表示序列长度。每个方块都是一段代码策略技能的时间窗视频，可按任务浏览、按技能跳转。

这个技能库有点像生物进化的「化石层」：每一代 Harness 的修订都留下了可检查、可回放、可复用的代码与影像证据。

## 总结与展望

- **核心思想** ：System 2（冻结的多模态基础模型）+ System 1（可编辑代码策略）原生协同，System 2 通过反思执行反馈递归改写 System 1 的 Harness；
- **进化机制** ：达尔文式自我 Harness——自我重写产生候选智能体，具身表现做选择，适者 Harness 继承到下一代（vary → evaluate → select → inherit）；
- **实现极简** ：System 2 改写与 System 1 执行各一条公式，无新网络结构、无重训练，技能以代码形式沉淀、继承与组合；
- **实验结果** ：截至 2026 年 10 月 8 日登顶 RoboDojo-Sim 榜首，记忆类任务 cover blocks 达 100% SR，并开源了 212 段剪辑、40 个任务的技能库。

**展望** ：Physical RSI 1.0 证明了「把学习能力写进代码而非权重」这条路线的可行性，但它目前仍依赖仿真环境的反馈闭环，随机布局与精细操作上的低成功率表明 Harness 的搜索效率与物理真实性迁移还有很长的路。它开启的方向足够诱人：当机器人能像程序员重构代码一样重构自己的行为，「自我改进」或许才真正开始 Scale。

> 本文参考自 [Physical RSI 1.0: Recursive Self-Harness for Scaling Embodied Skills](https://mmlab.hk/research/PhysicalRSI)