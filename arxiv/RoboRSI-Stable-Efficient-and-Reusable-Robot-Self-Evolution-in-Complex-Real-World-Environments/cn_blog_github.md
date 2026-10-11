# RoboRSI 论文解读：让机器人在真实家庭环境中稳定、高效、可复用地"自我进化"

![RoboRSI at a glance](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/2027_ICLR_Teaser1.0.png)

> 图解：RoboRSI 总览。中间是多智能体自我改进闭环，左侧是真实移动操作平台上的实物清理演示与复合技能带来的效率提升，右侧是四个仿真基准上的结果。

这篇文章要回答一个很现实的问题：让大模型写代码驱动机器人（Code as Policy）已经能做不少事了，但机器人干一次活积累的"经验"怎么才能变成后续任务可以复用的能力，而不是每次失败后打一堆散乱的补丁？RoboRSI 的核心思路是把执行经验组织在一棵自顶向下的技能层级树（Top-Down Skill Refinement, TSR）上，每次失败先定位"最早出错的节点"，只修改它负责的那个分支，并且经过验证后才允许发布复用。结果很硬：在 LIBERO、LIBERO-PRO、LIBERO-Plus、RoboTwin 四个基准上全部拿到最高成功率，比最强 baseline 高出 2.7 到 11.0 个百分点；在真机上完成了 104 轮、累计 24 小时的移动操作自进化，并在两个场景里无人工干预地完成了 5+4 个物体的清理任务。

## 背景：执行经验为什么总是"用不起来"

先建立一个直觉：现在的 coding agent 机器人，更像一个记忆力为零、但每次会写复盘笔记的新员工——笔记越写越多，却没人告诉他哪条经验属于哪项职责，下次换个场景又从头犯错。

作者把问题拆得很清楚。Code as Policies 开创性地让语言模型把感知与控制 API 组合成可执行程序；CaP-X 研究了程序的抽象层级；ASPIRE、ETA/OpenETA、ENPIRE 等工作引入了执行反馈，失败后修程序、存进技能库。但 **经验要变成可复用能力，必须挂回赋予它意义的任务结构上** 。缺了这个结构，会出现三个典型毛病：

- 上下文被完整轨迹塞满，Agent 在长记录里迷失；
- 局部修补会偏离整体目标——比如一个失败其实源于感知或导航，补丁却打在了释放动作上（因为失败是在那里被观察到的）；
- 目标只隐含在最新的 prompt 里，于是每次编辑都在为过拟合当前场景服务，人也得不断回来读日志、判断每个补丁该归谁管。

作者的洞察是：这个结构必须同时服务两类用户—— **对人友好** （Human-friendly steering），让人可以在"服务"层面表达目标、领域知识、约束和安全判断； **对 Agent 友好** （Agent-friendly structure），给 Agent 任务相关的上下文、显式接口和有边界的职责。RoboRSI 就是围绕这两个互补抽象搭起来的多智能体系统。

## 方法：TSR + 四个 Agent 的执行-改进闭环

### 技能层级：四类节点

RoboRSI 把每次执行转化为一个不断生长的技能库，层级里有四种节点：

- **任务族（Task family）** ：例如"家庭清理"，把共享目标和约束的任务归为一组；
- **原子任务（Atomic task）** ：有明确可观察结果的局部目标，例如"把物体放进容器"，每个原子任务由带 **后置条件（postcondition）** 的原子技能完成；
- **基础技能（Base skill）** ：感知与控制的原语，以工具形式暴露；
- **复合技能（Compound skill）** ：稳定的技能序列固化成参数化的代码策略，作为单一单元被调用。

每个技能都声明自己的输入、输出和职责，而且可以被多个分支共享——这一点后面会看到威力。

![RoboRSI framework](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/2027_ICLR_Framework2.0.png)

> 图解：一个开发轮次的完整流程。Manager 组织任务的技能层级，Planner 组合执行计划，Engineer 逐层执行到基础技能；Reviewer 根据观察和工具轨迹诊断失败、定位最早偏离点，把问题发回做 TSR 引导的修订；验证通过的更新进入下一轮迭代，稳定的分支固化为参数化技能。人在第一步设定目标，并在需要纠正时介入。

### 多智能体分工

四个 Agent 各司其职，每个只看到自己职责相关的层级局部，因此上下文短、每个改动都可追溯：

- **Manager** ：把目标分解为原子任务，维护技能版本，审查每份修订提案，决定它能否被接纳、允许改哪个分支、是否发布；
- **Planner** ：接收一个原子任务、当前观察和相关技能接口，返回可执行计划；
- **Engineer** ：通过工具接口执行计划，并实现缺失的技能；
- **Reviewer** ：从观察和工具轨迹判断原子任务是否达到预期结果，失败时找出执行最早偏离的位置，为责任分支写修订补丁提交给 Manager。

### TSR：把每个失败归因到"最早出错节点"

TSR 的形式化很干净。第 $t$ 轮的已发布层级记为图 $G_t=(V_t,E_t)$，每个节点 $v$ 带实现 $c_v$ 和声明的后置条件 $\phi_v$。执行任务产生观察与工具调用记录 $\tau_t$，沿任务分支按返回顺序列出执行过的节点 $\pi_t=(v_1,\ldots,v_m)$。Reviewer 沿路径评估后置条件，返回 **最早失败的节点** ：

$$
v_t^\star=v_{k^\star},\qquad
k^\star=\min\{k:\phi_{v_k}(\tau_t)=0\}
$$

修订范围 $S_t$ 是 $v_t^\star$ 拥有的子层级，$\partial S_t$ 是它对外部的接口。一轮修订就是：

$$
\Delta_t=\mathcal{R}\bigl(G_t|_{S_t},\partial S_t,\tau_t\bigr),\qquad
G_{t+1}=G_t\oplus\mathcal{V}(\Delta_t;\mathcal{H}_t)
$$

其中 $\mathcal{R}$ 是 Reviewer 的补丁提案，只能改动 $S_t$ 内的节点；如果接口本身必须改，先扩大范围把调用方纳入。$\mathcal{H}_t$ 收集了所有调用图可达 $S_t$ 的任务的近期成功与失败执行；验证算子 $\mathcal{V}$ 只有在补丁通过审查、功能测试和 $\mathcal{H}_t$ 回放且无回归时才放行，否则返回空补丁。

这个设计的聪明之处在于： **补丁打在失败的起源处，而不是失败被观察到的地方** 。失败往往在被发现前好几步就已发生，按观察点打补丁是治标，TSR 按最早失败节点打补丁是治本。

论文还给了一个依赖性分析：设任务 $q$ 的入口节点为 $r_q$，受影响任务集合 $\mathcal{A}_t=\{q:\operatorname{Reach}_{G_t}(r_q)\cap S_t\neq\emptyset\}$，$J_q(G)$ 为图 $G$ 下任务 $q$ 的期望失败率，则整体失败率变化只发生在受影响任务上：

$$
J(G_{t+1})-J(G_t)=\sum_{q\in\mathcal{A}_t}p(q)\bigl[J_q(G_{t+1})-J_q(G_t)\bigr],\qquad
\left|J(G_{t+1})-J(G_t)\right|\leq\sum_{q\in\mathcal{A}_t}p(q)
$$

换句话说，改动的影响上界正比于受影响任务的概率质量——任务专属分支影响小，被广泛共享的基础技能影响大。这直接解释了为什么回归测试要按 **依赖覆盖率** 来选，而不是按改了多少函数。

### 三条设计原则与复合技能固化

TSR 背后有三条原则：

- **职责与接口信任** ：每个技能通过显式接口报告结果，调用方依赖它的后置条件而不是重复检查，诊断沿调用路径找到责任组件；
- **参数化与历史引导的修订** ：物体类别、条件、目标关系、执行顺序都变成参数，场景几何来自当前观察，每个改动都要对照该技能近期的成功使用记录；
- **持久化人类知识** ：人的纠正变成运行时检查、测试或技能上的维护指引，而不是重复出现的 prompt。

当一个分支稳定后（至少 3 次成功执行、工具序列的最长公共子序列共享 60% 以上、有至少 3 步的公共骨架），系统会提议把它固化为复合技能：代码保留调用骨架和检查点，但所有参数在运行时从感知重新计算， **绝不存储历史坐标** 。

## 真机实验：104 轮、24 小时的移动操作自进化

### 实验设置

真机案例是家庭地面清理：一台 WheelSingleArm M1 移动操作机器人（Athena Pro Max 底盘 + RealMan 机械臂 + Zhiyuan 夹爪，腕部相机 + 底盘相机提供 RGB-D），任务是搜索物体、接近抓取、运到指定容器。开发期共 104 次运行、累计 24 小时；每次运行前场景恢复到统一基础布局，但物体和容器部分随机化；技能版本在单次运行内冻结，轮次之间才允许修订。

![Placement progress in physical runs](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/physical_four_item_line_paper.png)

> 图解：从 104 次开发运行中选出的 10 次运行的放置进展。纵轴是该次运行前四个物品中被现场确认放入指定容器的数量（四个是统一显示标尺，各次任务清单大小不一），横轴按所选运行顺序等距排列，刻度标注开发轮次编号，标记颜色表示经审计的任务结局。可以看到进展不是单调的：在已出现完整完成轮次之后，仍有一轮只放置了一个物体——这正是"开发过程"而非"固定策略重复试验"的体现。

### 双场景演示

扩展执行演示更说明问题：机器人在第一个场景放置了全部 5 个指定物体，第二个场景放置了全部 4 个，每次放置都由现场人员确认。两个场景之间，操作员重新摆放了物体和容器，但 **技能代码、prompt 和策略配置完全没动** ，执行过程无需任何人工纠正或重启。

![Two-scene physical mobile cleanup](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/2027_ICLR_RealRobot2.0.png)

> 图解：双场景移动清理的关键帧：准备、接近、抓取、放置和最终场景。回转箭头表示对多个物体的重复处理循环。场景间物体和容器被重新布置，Agent 配置保持不变。

### 技能树的生长

技能树快照展示了能力如何被组织和复用：层级在 *Place Object* 下引入 *Arm Explore* 以获得目标容器的视野；后续修订在 *Move To Object* 下添加 *Grasp Plan Check*（运动前检查候选抓取），并让 *Arm Explore* 同时服务接近和放置两个分支。这些改动针对清理流程中两个反复出现的需求：运动前验证抓取、把放置动作锚定在对目的地的观察上。

![TSR during physical cleanup](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/2027_ICLR_TSR.png)

![Physical TSR case](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/physical_tsr_case.png)

> 图解：(a) 物理任务开发过程中四个代表性的任务-技能结构快照，彩色块标记各快照点的技能操作，展示从初始任务提案到成熟技能库的演进。(b) 物理机器人上的两次 TSR 修订：第 25 轮（上）机器人拿着包装袋站在正确的垃圾桶前却不松手（i），其实手臂已到达观察位姿（ii），是运动调用误报了容差超标导致 fallback 耗尽预算；TSR 把失败归因到观察运动节点，只修订 Place Object 分支，下一次放置成功（iii）。第 33 轮（下）目标桶在底盘视角中只出现在图像边缘（iv），但同一扫描的手持视角中完整可见（v）；修订让身份判定归生产者所有并偏好完整视图。

### 两次物理修订的细节

**第 25 轮（放置观察与释放）** ：机器人拿着食品包装袋站在正确的带内衬垃圾桶前，观察移动却报告了 0.052 m 和 17.1° 的容差超标——尽管关节状态、末端状态和第三人称视频都显示手臂已到请求位姿。fallback 不断重复开盖估计、以无效的自检目标调用 Arm Explore，耗尽 28 次合并工具调用；运行时内存里其实存着完整的三步释放序列，模型却传了个不完整副本给 `place_object`。修订内容：(i) 失败的观察移动获得一次只读位姿复检，物理上已到达的位姿直接继续；(ii) 未到达的位姿向 Planner 返回不可释放检查点，禁用 Engineer fallback；(iii) 容器上方的探测与释放必须来自当前观察的开盖几何；(iv) `place_object` 用运行时内存中的完整序列替换模型写的不完整序列。验证通过 5 个新回归测试、172 个放置测试、863 个 manager 测试和 97 个真机接口测试。

**第 33 轮（目的地身份与导航交接）** ：扫描记录齐全、第一件物品释放成功，但运行约 25 分钟后结束、第二件仍握在手里。原因链很微妙：库存生产者选了一张边缘裁切的桶视图（同一次扫描里有完整视图），消费者又把蓝色占比当成强制检查拒绝了生产者认证的目的地；更早一次导航未完成且没有导航后图像，被路由到 Engineer fallback 后反复查询状态耗尽了预算。修订：库存选择偏好完整非边缘视图；查询颜色降级为建议性检查（语义身份归生产者）；不完整导航返回带类型的检查点给 Planner 而非进入 fallback。离线回放验证通过，28 个专项测试、866 个 manager 测试、97 个真机接口测试全部通过。

### 需要人类知识的纠正

104 轮中 53 个已分类修订触发里，只有 4 个需要系统自己拿不到的知识：确认误放会污染场景、必须由人重置（第 23 轮）；抓取接近需遵守 0.02 m/s 默认线速度安全限制（第 31 轮）；包装袋落在桶边应判为失败并需人工重置（第 37 轮）；持物保护必须继承审查过的夹持宽度与力度、持物丢失时冻结运动并请求确认（第 99 轮）。每条纠正都被固化为可复用的契约、检查点或测试——人只需说一次。

![Physical corrections requiring human knowledge](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/physical_human_knowledge.png)

> 图解：四个需要人类知识的物理纠正案例：(a) 确认误放后的场景重置；(b) 操作员指定的安全抓取速度；(c) 包装袋被释放到垃圾桶外；(d) 任务终止后的持物保护。

## 仿真主结果：四个基准全面领先

### 评估设置：自我改进本身就是被测能力

这里有个关键设计：在 LIBERO、LIBERO-PRO、RoboTwin 上，RoboRSI 从初始技能库出发， **在评估任务上边评边改** ，而 baseline 没有任何自我改进机制——作者明确说"在任务上改进本身就是 RoboRSI 贡献的能力"。LIBERO-Plus 则相反：在 LIBERO 上学到的技能库被冻结，直接迁移到扰动实例上测泛化。所有方法使用同一 backbone（GPT-5.6-SOL）、同一工具接口、同样的每回合交互预算和模拟器判定。整个主比较包含 5,974 个评估 episode。

| 基准 | 方法 | 成功率 | 任务覆盖率 |
|---|---|---|---|
| LIBERO | CaP-X | 36/150 (24.0%) | 16/30 (53.3%) |
| | Maestro | 62/150 (41.3%) | 23/30 (76.7%) |
| | OpenETA | 76/150 (50.7%) | 23/30 (76.7%) |
| | **RoboRSI** | **84/150 (56.0%)** | **24/30 (80.0%)** |
| LIBERO-PRO | CaP-X | 111/600 (18.5%) | 46/120 (38.3%) |
| | Maestro | 198/600 (33.0%) | 77/120 (64.2%) |
| | OpenETA | 231/600 (38.5%) | 80/120 (66.7%) |
| | **RoboRSI** | **297/600 (49.5%)** | **102/120 (85.0%)** |
| LIBERO-Plus | CaP-X | 130/840 (15.5%) | 21/30 (70.0%) |
| | OpenETA | 306/840 (36.4%) | 27/30 (90.0%) |
| | **RoboRSI** | **354/840 (42.1%)** | **28/30 (93.3%)** |
| RoboTwin | CaP-X | 3/150 (2.0%) | 3/50 (6.0%) |
| | OpenETA | 32/150 (21.3%) | 17/50 (34.0%) |
| | **RoboRSI** | **37/154 (24.0%)** | **26/50 (52.0%)** |

最大优势在 LIBERO-PRO：成功率超 OpenETA 11.0 个百分点，覆盖率超 18.3 个百分点——优势不只是平均成功率，而是扩展到了更多被解决的任务。LIBERO 各套件细分（每套件 10 任务、每方法 50 episode）显示 Spatial 上超 14 分、Object 上超 8 分：

| 方法 | Spatial | Object | Goal | 合计 |
|---|---|---|---|---|
| CaP-X | 12/50 | 15/50 | 9/50 | 36/150 |
| Maestro | 22/50 | 24/50 | 16/50 | 62/150 |
| OpenETA | 23/50 | 29/50 | **24/50** | 76/150 |
| **RoboRSI** | **30/50** | **33/50** | 21/50 | **84/150** |

LIBERO-PRO 按套件与扰动类型拆解后，RoboRSI 在每一列都是最优，其中 Object 套件 +16.5 分、物体扰动下 +15.3 分——这些恰好是"需要在当前场景重新识别目标"的情况，正是后置条件验证和技能修订的主场：

| 方法 | Spatial | Object | Goal | 语言扰动 | 物体扰动 | 位置扰动 | 任务扰动 | 合计 |
|---|---|---|---|---|---|---|---|---|
| CaP-X | 24.5 | 13.5 | 17.5 | 21.3 | 16.7 | 17.3 | 18.7 | 18.5 |
| Maestro | 39.5 | 38.0 | 21.5 | 34.7 | 30.0 | 32.0 | 35.3 | 33.0 |
| OpenETA | 50.0 | 36.5 | 29.0 | 42.7 | 33.3 | 36.0 | 42.0 | 38.5 |
| **RoboRSI** | **57.0** (+7.0) | **54.5** (+16.5) | **37.0** (+8.0) | **52.0** (+9.3) | **48.7** (+15.3) | **45.3** (+9.3) | **52.0** (+10.0) | **49.5** (+11.0) |

### 一天无人值守的自迭代

作者在 120 个 LIBERO 任务目录和 120 个 LIBERO-PRO 任务上让 RoboRSI 无人干预地自我迭代一天：每轮当前技能库在尚未解决的任务上运行，通过验证的修订发布给下一轮。五轮之内，LIBERO 任务覆盖从 32 涨到 71，LIBERO-PRO 从 43 涨到 80，LIBERO 的五轮只花了 7.0 小时有效执行时间。

![One-day self-iteration](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/evolution_cost.png)

> 图解：一天自迭代中每轮之后"至少解决过一次"的任务累计数（按首次成功计入）。曲线持续上升且尚未饱和，说明自我迭代的红利还能继续挖。

### 失败分析：baseline 为什么输

诊断结果相当有说服力：OpenETA 和 Maestro 在 LIBERO-PRO 上约三分之二的失败（66% 和 64%）、OpenETA 在 LIBERO-Plus 上 66% 的失败，都是 **提前宣告完成** ——工具报告了"已释放"，Agent 就宣布任务完成，而模拟器判定未完成。RoboRSI 的失败中这一比例只有 10% 和 14%。这正是"每个原子任务完成后必须用新鲜观察验证后置条件"这一设计的直接效果。RoboRSI 剩余的失败大多是恢复过程中预算耗尽，而不是谎报成功。在执行失败发生的 episode 里，RoboRSI 仍有 67.0%（LIBERO-PRO）和 63.2%（LIBERO-Plus）最终成功，OpenETA 只有 55.0% 和 51.6%。

![Failure analysis](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/failure_analysis.png)

> 图解：(a) LIBERO-PRO 与 LIBERO-Plus 上失败 episode 的构成，$n$ 为失败数，标注了"提前完成"的占比——baseline 的失败大头是它，RoboRSI 把它压到了一成多。(b) LIBERO-Plus 按扰动类型（每类 120 实例）的成功率，数字是 RoboRSI 相对 OpenETA 的百分点差：背景 +13、语言 +12、机器人初始状态 +9、噪声 +8、光照 +7。(c) 在发生过执行失败的 (任务, 种子) 对中，最终仍成功的比例——RoboRSI 的恢复能力更强。

## 案例研究：同样的开局，不同的结局

作者挑了三个 RoboRSI 成功而 baseline 失败的 episode（相同初始状态），非常有戏剧性地展示了机制差异：

- **LIBERO-Plus 光照扰动** ：把奶油奶酪包放进篮子。OpenETA 的组合 pick-and-place 工具返回"已在目的地释放"，Agent 立刻宣布成功；CaP-X 的放置也报告了释放但容器内检查未验证——两者都失败。RoboRSI（用冻结的源技能库）收到"夹爪未打开"的放置结果，没有结束，而是检查自己是否还拿着包裹、从头部和顶部视角重新观察篮子、再次放置，成功。
- **RoboTwin 双臂任务** ：两只手臂分别抓罐子放进塑料盒。CaP-X 反复尝试让左臂越过中线去够盒子（不可行）；OpenETA 在夹爪报告释放后就停了。RoboRSI 第一次左臂放置失败后，保持双罐在手、验证两个抓取和负重手臂间的间隙，然后一次一侧地释放。
- **LIBERO-PRO 多步任务** ：打开顶层抽屉放碗。CaP-X 对抽屉本体而非把手调用拉拽技能被拒绝；OpenETA 反复无法接近把手又抓碗失败。RoboRSI 也有一次拉拽和一次放置被拒，但持续观察与行动直到成功。

共性一句话： **baseline 败在信任工具返回值，RoboRSI 的技能会暴露失败结果、Agent 在继续前验证后置条件** 。

![Simulation cases](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/sim_cases.png)

> 图解：三个仿真案例。图像为 RoboRSI episode 的首末帧，文字总结各方法决定性的工具调用——可以看到 baseline 都在某次"假成功"的工具返回后停止，而 RoboRSI 继续验证并恢复。

### 一次修订，处处受益

共享技能的威力在复用案例里体现得淋漓尽致：基础技能 `grasp_object` 只在 **一个** 开发任务上被修订——Manager 的补丁要求每次重试都从新鲜观察中验证目标身份，并禁止重放已失败过的抓取点。修订通过发布门后， **其他任务上的 42 个成功 episode 调用了这个修订版技能** ，包括 Goal、Object、Spatial 三个套件里在先前冻结评估中失败过的任务。因为技能通过层级共享，一个任务上的修复自动流向所有调用它的分支。

![Reuse of a revised skill](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/skill_reuse.png)

> 图解：修订版 `grasp_object` 在三个开发集外任务上的调用。三列分别是首帧、`grasp_object` 调用后的状态、最终成功状态。这三个任务在冻结评估中都失败过，共享修订让它们无需任何额外开发即被解决。

## 技能代码的微观解剖

附录里的代码级案例把 TSR 的"归因-局部修订-验证-发布"流程讲得最透。

### 案例一：删除一次"过早拒绝"

在 `click_bell` 任务上，原始技能观察到铃铛包围框 $[249,185,306,240]$（320×240 图像），下边缘超过了硬编码阈值 237，目标关联检查在发出点击前就拒绝了目标——任务失败。责任节点在 `get_object_bbox` 之后的关联检查，按铃目标和工具接口本身不需要改。Manager 收到失败轨迹后生成的修订允许紧凑物体有界的下边缘截断，同时保留尺寸限制、拒绝大面积截断。修订版在同一初始状态下接受目标、执行 tap，模拟器判定成功（1/3 → 3/3，满足至少 2/3 的预声明发布门槛）。有意思的是：把旧代码在新观察上重放，仍然因为 $239>237$ 拒绝；新代码能完整重放全部 22 次公开调用。

![TSR execution repair](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/tsr_execution_repair.png)

> 图解：按铃案例的局部技能修订。左：原始目标关联检查在动作前拒绝了近边界的包围框；中：修订版检查接受紧凑目标；右：成功修订运行后的观察。方框为检测到的目标，任务结局由模拟器独立判定。

### 案例二：复合技能的验证、发布与调用

后续的 `exact_waypoint_feature_tap` 技能是可执行能力积累的完整样本：描述符暴露四个输入（手臂、物体描述、功能特征描述、相机），实现负责取图、定位物体与功能点、恢复深度、检查运动端点、执行 tap、检查回撤与终止状态。它通过了 2/3 模拟器开发检查后被发布为版本化技能——代码保留流程骨架，但物体关联和动作目标每次调用时重新计算。

![Skill publication case](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/skill_publication_case.png)

> 图解：已发布 tap 技能在一次成功 `click_bell` 开发试验中的观察：初始视野、特征定位、执行后视野（320×240 像素）。结局由模拟器评分。

### 案例三：审查驱动的终止检查修订

一个早期候选在 Manager 审查阶段就被拒了：它的恢复表达式存在短路——当运动或位姿检查失败时，后续的占用和间隙查询可能被跳过；还可能在占用未确认时用"空手假设"查询间隙。修订版在每次恢复尝试后评估终止占用状态，仅在双臂确认空手时才检查间隙，并用这些记录在案的检查判断恢复是否成立。这个版本通过了独立审查和 2/3 模拟器检查后发布，而旧候选止步于代码审查、一次模拟器运行都没有。 **审查环节真的拦住了有逻辑缺陷的代码** ——这正是把 Reviewer 从执行者中分离出来的价值。

## 消融实验

### 角色分离：诊断必须与执行解耦

在 50 个 RoboTwin 任务上，单 Agent（一个 Engineer 自己规划、执行、评判）只解决 9 个任务，多 Agent（Planner + Engineer + Reviewer）解决 36 个：

| 条件 | 角色 | 解决任务数 |
|---|---|---|
| 单 Agent | Engineer | 9/50 (18.0%) |
| **多 Agent** | Planner, Engineer, Reviewer | **36/50 (72.0%)** |

让执行者评判自己的结果，等于让运动员兼任裁判。诊断独立后，每个失败在下一次尝试前都有一次独立复盘。

### TSR vs 平铺修订

同一份失败轨迹（`libero_goal/8` 上失败的 `libero_pick_place`）、相同的起点代码、相同的角色配置：TSR 先把失败归因到两个节点——原子计划把抓取时设置的"不可重抓"标志在已验证释放后仍当永久值、基础持握检查把固定抓取块当作运输过程中的不变量——只改这两处，两个开发种子全过， **发布** ；平铺修订直接重写（包括放置策略），两个候选都只过 1/2 个种子（其中一个还出现"物体不在目标上却宣告成功"）， **均不能发布** 。TSR 改动 3 个文件 50 行，平铺候选分别改了 4 个文件 104 行和 1 个文件 63 行——改得更少、改得更准、改得更稳。

### 代码固化：成功率 +7.5 分，成本降三成

在 120 个 LIBERO-Plus 任务、600 对匹配 episode 上，复合技能可用（Code on）对比被扣掉（Code off）：成功率从 21.5% 提到 29.0%（配对增益 7.5 个百分点，95% CI $[3.7,11.5]$，McNemar $p<10^{-4}$），至少解决一次的任务从 58 涨到 71；中位 token 消耗、模型调用数、墙钟时间分别下降 29.4%、27.2%、17.0%。

原因很本质：没有固化代码时，Agent 每个 episode 都要重新"发明"整条工具序列，每一步都是选错工具、传错参数、漏掉检查的机会，误差在长序列上累积；复合技能把验证过的调用顺序和中间检查固定下来，Agent 只需要决定调哪个技能、传什么参数。墙钟时间降幅小于 token 降幅，是因为模拟器里执行的运动没变，省下的只是运动之间的推理。

![Effect of code consolidation](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/ablation_code.png)

> 图解：(a) 600 对匹配 episode 的任务成功率，Code on 显著更高；(b) 118 任务效率样本上的中位 token、模型调用、时间成本（以 Code off 归一化），三项均明显下降。

### Backbone 模型

GPT-5.6-SOL、GPT-5.5、GPT-5.4 三个 backbone 对比（相同的初始技能、任务分布与交互预算）：GPT-5.5 在 LIBERO-PRO 上成功率最高（48.6%），GPT-5.6-SOL 在 RoboTwin 上最好。三者的模型调用数中位数几乎相同（每 episode 115~124 次），差异在于 **多少 episode 耗尽预算** ：GPT-5.4 有 101 个、GPT-5.5 有 80 个预算耗尽的 episode。更强的 backbone 不是靠想得更久赢，而是更常在预算内到达已验证的结局。排名随基准变化而技能库不变，说明 backbone 选择与基准之间存在交互，不存在全程最优的单一模型。

| Backbone | LIBERO-PRO 总体 | Spatial | Object | Goal | RoboTwin | 模型调用（中位） | 预算耗尽 episode |
|---|---|---|---|---|---|---|---|
| GPT-5.6-SOL | 38.3 | 38.3 | 48.3 | 28.3 | **24/150** | 115 | 79 |
| GPT-5.5 | **48.6** | 53.3 | **55.0** | **37.5** | 19/150 | 122 | 80 |
| GPT-5.4 | 40.3 | **55.8** | 39.2 | 25.8 | 21/150 | 124 | 101 |

### 从执行数据学策略：闭环的最后一环

成功的 RoboRSI 执行还能作为示范训练紧凑的视觉运动策略，再以技能身份回到同一层级——这让"代码技能"和"学习策略"在同一棵树上共存。作者用 10 次 `click_bell`、7 次 `click_alarmclock` 成功执行训练 ACT 策略：绝对关节目标下一个任务都解不了；改成 **chunk 相对目标** （相对每个动作块起点的手臂构型）后，铃铛任务 4/10、闹钟任务训练布局 6/7、未提供训练数据的采集布局 2/3。训练观察上的关节指令误差下降 57%（0.0086 → 0.0037 rad）。示范极少时，绝对目标要求策略每步回归整个手臂构型，小回归误差就会把末端带离按压任务的小接触区域；相对目标只编码"从当前构型怎么动"，被示范约束得紧得多。

![ACT policies from executions](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/RoboRSI-Stable-Efficient-and-Reusable-Robot-Self-Evolution-in-Complex-Real-World-Environments/figures/act_results.png)

> 图解：相同示范、架构与训练预算下，绝对与 chunk 相对关节目标的 ACT 策略成功率对比。相对表示在两个任务上都大幅占优。

## 总结与展望

- **问题定位准** ：执行经验只有挂回任务结构才能复用——修复要归因到负责的能力、有执行证据支持、验证后才许复用；
- **TSR 是核心机制** ：后置条件沿调用路径验证、归因到最早失败节点、修订限定在责任子层级、发布前要过审查+功能测试+历史回放三重门；
- **多智能体分工有效** ：诊断与执行解耦，把 RoboTwin 解决任务数从 9/50 拉到 36/50；
- **结果全面且诚实** ：四个基准全部最高（+2.7 ~ +11.0 分），真机 104 轮 24 小时开发，失败分析显示 baseline 三分之二的失败是"提前宣告完成"，RoboRSI 只有一成；
- **复用性得到实证** ：单个任务上修订的 `grasp_object` 被 42 个其他任务的成功 episode 调用，复合技能固化同时提升成功率 7.5 分、降本约三成。

局限与方向也清晰：作者明确计划把仿真中积累的经验迁移到物理机器人、并跨机器人与平台共享技能；而执行轨迹训练出的神经策略能否作为"技能"与代码策略在同一层级中规模化共存，是这条路线最值得关注的下一步。

> 本文参考自 [RoboRSI: Stable, Efficient, and Reusable Robot Self-Evolution in Complex Real-World Environments](http://arxiv.org/abs/2610.12424v1)