# 96 台机器人"抱团学习"：不再依赖人类专家纠错，机器人车队靠集体经验把成功率刷到 99%

机器人在真实部署中每天都会产生海量执行数据，但传统的"人类纠错"式学习要求遥操作员全程盯梢、随时接管，机器人越多，人越不够用。这篇来自清华大学、智元机器人（AgiBot）和无问芯穹（Infinigence AI）的联合工作提出了 **Collective Experience Learning (CEL，集体经验学习)** 范式：让整个机器人车队共享执行经验，共同打磨同一个策略。它的算法实现 **FERT** 用"失败片段"去检索车队里相似的成功片段作为监督信号，全程不再需要人类专家补演示、补纠正；系统实现 **FATS** 则保证 96 台机器人在不断出故障的情况下异步、不中断地学习。最硬核的结果是：在 5 个真实操作任务上，最终策略在未参与训练的机器人上达到 **93.3%–99.2%** 的成功率（每任务 120 次测试）；在多阶段任务 Phone Packaging 上，96 台机器人集体学习将成功率从 79.2% 一路推到 **98.3%**，甚至超过了人类遥操作员的平均水平。

## 一、要解决的问题：车队在扩张，人类专家却不够用了

先建立直觉。VLA（Vision-Language-Action）大模型给了机器人一个不错的初始能力，但初始能力不等于可靠执行——物体形态差异、接触状态变化、多阶段任务中的误差累积，都会让策略在真实环境里频频翻车。好在机器人每次执行本身就在产生经验，这些经验天然可以用来继续改进策略。

问题在于：现有的"从部署中学习"方法几乎都要求人类持续提供**动作级监督**。DAgger、HG-DAgger、HIL-SERL 这类方法需要遥操作员盯着机器人执行，发现失败就接管示范正确动作。一台机器人配一个操作员尚且可行，当车队扩到几十上百台时，人的监控和纠错能力根本无法同比例扩张——这成了整个范式的天花板。

这篇文章的核心洞见是：**车队的失败和成功本身就是最好的教材，关键在于怎么把两者配对使用**。他们提出的 CEL 范式把"分布式执行"和"共享策略改进"接成闭环：机器人执行产生经验 → 经验汇聚成集体经验池 → 更新共享策略 → 新策略下发回车队 → 产生新经验。在这个闭环里，人类只做两件轻量的事：给每个 episode 打一个"成功/失败"的二值标签，以及复位环境；动作监督全部来自机器人自己的成功执行。

![经验来源的三种范式对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/algorithm/CEL.png)

> 图解：三种机器人策略学习的经验来源。最左是纯人类演示（SFT 式），中间是专家纠正式学习（DAgger 式），最右是本文的集体经验学习（CEL）。从左到右，对专家动作监督的依赖逐步降低：CEL 中，人类标注 episode 成败结果（或由学习到的奖励模型代劳），而用于后续学习的动作全部来自机器人自己的成功执行。失败的执行不充当模仿对象，而是作为"检索词"去成功经验池里找答案。

这个设计的聪明之处在于它重新定义了"失败"的价值——以往失败轨迹要么被丢弃、要么只能当负样本，而在这里失败变成了**定位策略薄弱环节的路标**。

## 二、预备知识：问题形式化

先把问题钉死。作者使用 $\pi_0$（一个基于 flow matching 的 VLA 模型）作为基础策略，策略根据视觉观测、本体感觉和语言指令生成连续动作块：

$$
\pi_\theta(A_t \mid o_t, g), \qquad A_t = (a_t, \ldots, a_{t+H-1})
$$

其中 $o_t$ 是 $t$ 时刻的观测（多视角图像 + 本体感觉），$g$ 是任务指令，$A_t$ 是预测时域为 $H$ 的动作块（本文 $H=30$、动作维度 $d=16$），$\theta$ 为策略参数。

每次执行产生一条轨迹 $\tau = \{(o_t, a_t)\}_{t=0}^{T-1}$，并附带一个二值结果标签 $y(\tau) \in \{0,1\}$ 表示任务成败。学习目标非常直白——最大化任务成功概率：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ y(\tau) \right]
$$

注意这个设定里没有任何奖励函数、价值函数或优势估计，唯一的反馈就是那个 0/1 标签。整个 CEL 的难点由此而来：**二值结果只告诉你"成没成"，并不告诉你"哪个动作错了、该怎么改"**。接下来看 FERT 如何绕过这个信息缺口。

## 三、FERT 算法：失败当"检索词"，成功当"教科书"

FERT（Failure-guided Experience Retrieval fine-Tuning，失败引导的经验检索微调）是 CEL 的算法实现。一句话概括：**成功轨迹提供要模仿的动作，失败轨迹决定学习该往哪儿使劲**。整个流程分四步：异步经验收集、经验编码、失败引导检索、优先级微调。

![FERT 算法流程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/algorithm/FERT.png)

> 图解：FERT 的完整闭环。机器人异步自主执行并收集带二值成败标签的经验；观测被编码为视觉 + 本体感觉联合特征；失败片段作为查询，从车队的成功轨迹库中检索相似片段；被检索到的成功片段获得更高的采样优先级，用于策略微调；更新后的策略再下发回车队继续产生新经验。

### 第一步：异步经验收集

车队持续自主执行，每条完成的轨迹拿到一个二值成败标签。成功轨迹进入**集体经验缓冲区** $\mathcal{D}^{+}$（一个滚动窗口的内存 buffer），充当后续微调的动作目标来源；失败轨迹的动作**不**被模仿，只被临时用作检索查询。这个"成败分工"是 FERT 的骨架：用失败引导学习方向，又避免了模仿失败动作。收集与训练完全异步进行，机器人可以各跑各的节奏。

### 第二步：经验编码

检索需要一个既能刻画场景、又能刻画机器人姿态的表征。作者把冻结的视觉塔（VLA 中 vision tower 的冻结副本，保证训练过程中表征空间不漂移）输出的 $16 \times 16$ 视觉 token 网格做 $4 \times 4$ 平均池化，保留粗粒度空间信息，再与归一化关节状态拼接：

$$
z_t = \left[ \frac{h_t}{\max(\|h_t\|_2, \epsilon)} ; \frac{s_t}{\max(\|s_t\|_2, \epsilon)} \right]
$$

其中 $h_t$ 是多相机视觉特征按固定顺序拼接的结果，$s_t$ 是归一化关节状态。视觉和状态分别归一化，保证两者在可比尺度上。成功轨迹的嵌入在入库时就算好并缓存，失败轨迹只需编码查询片段。

### 第三步：失败引导检索

二值失败标签无法精确指出失败从哪一帧开始，所以 FERT 取失败终点**之前**的一小段观测序列作为查询：

$$
e = T - \delta, \qquad b = \max(0, e - L), \qquad Q_\tau = (o_b, \ldots, o_{e-1})
$$

本文取 $L = 90$ 帧、$\delta = 10$ 帧，即 30Hz 控制频率下最多 3 秒的上下文，且避开终点前约 1/3 秒（终点附近的画面往往已经是"失败之后"的状态）。这个偏移量设计很细节——它排除了失败已经发生后产生的误导性画面。

检索采用带斜率约束的子序列动态时间规整（S-DTW）：对每条成功轨迹计算查询帧与候选帧的距离矩阵 $d_{i,j} = \|z_i^{\tau_f} - z_j^{\tau_s}\|_2$，再用动态规划找最小代价对齐：

$$
D_{i,j}^{\tau_s} =
\begin{cases}
d_{0,j}^{\tau_s}, & i = 0,\\[4pt]
d_{i,j}^{\tau_s} + \min\left\{ D_{i-1,j-1}^{\tau_s},\; D_{i-1,j-2}^{\tau_s} \right\}, & i > 0,
\end{cases}
$$

约束 $p_i - p_{i-1} \in \{1, 2\}$ 允许查询在更长的成功序列内做有界的时序拉伸，同时保持时间顺序。每条成功轨迹保留代价最低的一段，然后全局选出代价最小的 $K = 10$ 段，并向前后各扩展 $h_{\mathrm{pre}} = h_{\mathrm{post}} = 30$ 帧，把匹配片段前后的动作上下文也纳入学习材料。后面消融实验会证明这个"扩展"极其重要。为了不让检索拖慢训练，查询和候选序列都先做 3 倍下采样（理论加速 9 倍，实测约 6 倍）。

### 第四步：优先级微调

检索到的片段转化为采样优先级。对成功动作块 $A_i$，其优先级按检索事件 $n$ 更新为：

$$
w_i^{(n)} = \rho \, w_i^{(n-1)} + \sum_{R \in R_n} \frac{|\mathrm{idx}(A_i) \cap R|}{H}
$$

其中 $\rho \in [0,1]$ 是历史优先级的保留因子（本文取 0.88），重叠比例让完全落入检索区间的块获得完整增量、跨界块获得部分增量。**反复出现的失败场景会不断强化对应成功动作的优先级**，而指数衰减让陈旧检索事件的影响逐渐消退。训练时按优先级比例采样：

$$
P(i) = \frac{w_i}{\sum_{j \in \mathcal{D}^{+}} w_j}, \qquad
\widehat{\mathcal{L}}_{\mathrm{FERT}}(\theta; \mathcal{B}) = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \log \pi_\theta(A_i \mid o_i, g_i)
$$

由于 $\pi_0$ 是 flow-based 策略，直接算对数似然需要追踪生成流的密度变化，代价太高。作者改用条件 flow matching 损失作为似然拟合的可行替代——通过把速度预测重写为噪声预测，可以证明加权去噪目标等价于最大化带噪动作对数似然的变分下界（ELBO），因此用 flow matching 损失拟合优先级采样出的成功块在原理上是站得住的。

关键超参数如下（每次 FERT 运行约 7–8 小时在线训练）：

| 参数 | 取值 |
| --- | --- |
| 最大学习率 | $2.5 \times 10^{-5}$ |
| 全局 batch size | 1024 |
| 学习率调度 | Cosine（80 步 warmup，10,000 步 horizon） |
| 梯度裁剪 | 4.0 |
| 每个训练 worker 的滚动 buffer | 350,000 帧 |
| 失败查询长度 $L$ | 90 帧 |
| 失败端点偏移 $\delta$ | 10 帧 |
| 最优检索段数 $K$ | 10 |
| 检索段扩展 $h_{\mathrm{pre}}, h_{\mathrm{post}}$ | 各 30 帧 |
| 优先级保留因子 $\rho$ | 0.88 |
| 初始采样优先级 $w_{\mathrm{init}}$ | 0.5 |
| 下采样比 | 3 |
| 权重同步间隔 | 500 个优化步 |
| 训练 worker 与机器人数量比 | 0.25 |

### 策略从哪来：SFT + 多机 HG-DAgger 初始化

FERT 不是从零开始。每个任务先用人类演示做 SFT，再跑多机器人版 HG-DAgger（记为 (M)HG-DAgger）：多台机器人并行执行当前策略，人类操作员在需要时接管纠偏，只有"完全由人类操作的动作块"才作为监督目标（还会丢掉每段接管开头的若干帧，滤掉人机切换瞬间的犹豫动作）。训练用条件 flow matching 目标：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = \mathbb{E}_{(o_t, g, A_t) \sim \mathcal{D}_{\mathrm{demo}},\, \alpha,\, \varepsilon_{\mathrm{noise}}} \left[ \frac{1}{Hd} \left\| v_\theta(A_t^\alpha, \alpha \mid o_t, g) - (\varepsilon_{\mathrm{noise}} - A_t) \right\|_F^2 \right]
$$

其中 $A_t^\alpha = (1-\alpha) A_t + \alpha \varepsilon_{\mathrm{noise}}$ 是插值动作块。作者还发现训练 batch 里有约 5–10% 的高损失长尾样本，于是加了 batch 级损失过滤：以分位数与绝对下限的较大者为阈值，只在保留样本上回传梯度。FERT、RECAP、RLT 三个方法都从同一个 (M)HG-DAgger checkpoint 出发，保证比较公平。

![多机器人 HG-DAgger](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/algorithm/HGDAgger.png)

> 图解：多机 HG-DAgger 初始化流程。人类操作员在机器人执行出错时接管，成功轨迹连同"每一帧是人操作还是策略操作"的指示位一起存储，只有合格的干预动作块被用于监督式策略更新。

### 理论分析：为什么"检索加权"真的有用

附录中作者给出了一套局部理论保证，核心结论很干净。记 $\mathbf{v} = \nabla J(\theta_0)$，样本的一阶贡献为 $u = \mathbf{v}^{\top} \nabla_\theta \log \pi_{\theta_0}(A \mid o)$，定义两个对齐量：$\mu = \mathbb{E}_{\mathcal{D}^{+}}[u]$（成功回放与回报目标的对齐度）和 $\kappa = \mathrm{Cov}_{\mathcal{D}^{+}}(\omega, u)$（检索权重引入的额外对齐）。

- **引理 1（成功回放对齐）**：在当前策略成功回放的近似下，均匀回放的平均梯度恰好是 $\mathbf{G}_{\mathrm{unif}} = \mathbf{v} / (T_c J(\theta_0))$，于是 $\mu = \|\mathbf{v}\|_2^2 / (T_c J(\theta_0)) > 0$。也就是说，**只要策略还有非零成功率，拟合自己的成功经验本身就沿着提升期望回报的方向**。
- **引理 2（检索对齐）**：在一个分阶段任务模型中（各阶段独立、局部失败率 $\phi_k$），可以推出阶段 $k$ 成功经验的一阶贡献正比于 $\phi_k^2$——**越容易失败的阶段，其成功经验越"值钱"**。因此只要检索把更多权重分给失败率更高的阶段，就有 $\kappa > 0$。
- **定理 1（局部改进保证）**：在足够小的步长下，$J(\theta_{\mathrm{FERT}}) > J(\theta_{\mathrm{unif}}) > J(\theta_0)$，且改进量有下界：

$$
J(\theta_{\mathrm{FERT}}) - J(\theta_0) \geq \eta \underbrace{\mu}_{\text{成功回放}} + \eta \underbrace{\kappa}_{\text{检索贡献}} - \underbrace{\frac{\lambda_J \eta^2}{2} \|\mathbf{G}_{\mathrm{FERT}}\|_2^2}_{\text{有限步长惩罚}}
$$

证明思路是用 $\nabla J$ 的 $\lambda_J$-Lipschitz 性质把有限步长的高阶余项压进 $\frac{\lambda_J}{2}\|\Delta\|_2^2$，再分别对均匀回放和 FERT 更新套用微积分基本定理作差。这个分解很直观：FERT 的收益 = 成功回放的基础收益 $\eta\mu$ + 检索的定向加成 $\eta\kappa$ − 有限步长的代价。当然，这是单步、局部、带假设的保证，长期多步的实际收益要靠后面的消融实验验证。

算法有了，下一个问题很现实：96 台物理机器人 7×24 小时地跑，会断电、会断网、会撞关节限位，怎么让学习 loop 不中断？这就是系统层要回答的问题。

## 四、FATS 系统：让 96 台机器人"边坏边学"

FATS（Fault-tolerant Asynchronous Training System，容错异步训练系统）是 CEL 的系统实现，部署规模达到 96 台机器人 + 6 台云端服务器（共 48 张 NVIDIA A800），机器人站点与云端之间只有一条 30 Gbps 专线。它要应对两个本质矛盾：

| 维度 | 交互流量（Interaction） | 经验流量（Experience） |
| --- | --- | --- |
| 数据单元 | 一次观测 + 一个动作块 | 组装成完整 episode 的各轨迹字段 |
| 数据大小 | 每次模型输入约 0.45 MB 图像 | 一个 Phone Packaging episode 约 1.58 GB |
| 时序要求 | 必须在机器人下一次交互前返回 | 训练前到齐即可，不必实时 |
| 传输路径 | 机器人观测 → 云端推理 → 动作返回机器人 | 云端组装，复用推理数据 + 边缘补充字段 |

第一个矛盾：交互要"快"，经验要"全"，两者共享同一条上行链路，不能让训练数据传输卡住机器人控制。第二个矛盾：物理机器人远比 GPU 脆弱——复位、亏电、硬件故障随时发生（96 机运行中 22 台出过故障，平均每 22 分钟就有一台中断），而故障恢复必须保住"机器人 ↔ 物理工位"的绑定关系，训练侧的优化器状态又极其昂贵不能丢。

FATS 的答案是数据平面 + 控制平面的双平面架构。

![FATS 系统架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/infra/fats_concept.png)

> 图解：FATS 总览。a）SwarmLink 数据平面连接边缘机器人与云端共享推理，并把流式上传的轨迹字段组装成训练样本；HotSwarm 控制平面管理成员身份、检测故障并原地恢复 worker。b）同步收集中每次策略更新都要等最慢的机器人，快的机器人干等；异步收集中各机器人独立推进。c）HotSwarm 原地重启故障机器人的 worker，其余机器人照常运行，为故障 worker 算出的旧动作（#1）被丢弃。d）推理数据与执行记录在云端汇合，episode 在字段与结果到齐后才释放给训练。e）96 机运行实测 + 模型推演：局部恢复下可用于采集的机器人时间不随车队规模恶化，而"每次故障全系统重启"的方案在车队翻倍时损失占比也大致翻倍。

### SwarmLink 数据平面

核心是**分布式 channel**——一种带命名的生产者-消费者队列，worker 之间无需固定点对点连接，支持异步操作让通信与计算重叠。三个关键设计：

- **共享推理池**：云端推理 worker 服务公共请求池而非绑定固定机器人，暂停的机器人不会拖住 GPU；batch 凑满目标数或最老请求超时（取先到者）即派发，避免等慢机器人。
- **轨迹流式组装**：episode 数据在执行过程中就持续上传，推理时收过的观测直接复用不再重传；trajectory collector 按 worker/episode/step 标识拼接字段，数据与结果到齐才释放给训练。
- **流量优先级**：交互流量优先于批量经验数据，后者按有界分块传输并施加背压，保证训练上传永远不会卡死机器人控制回路。

### HotSwarm 控制平面

HotSwarm 的关键抽象是区分 **station（工位）**和 **worker（进程）**：工位是绑定物理节点/机器人的固定位置，整个训练期间都在；worker 是占据工位的进程，挂了就换。机器人掉电重启、下线维护、新节点加入集群，统一走"工位上 admit/retire/再 admit"这一套机制。

恢复策略按 worker 类型分层：环境 worker（连着物理机器人、故障最多、状态最廉价）无限次原地重启，代价最多一个 episode；推理 worker 重启后重新加载当前策略；训练 worker 持有数小时积累的优化器状态，靠周期 checkpoint 从最近点恢复，期间存活的推理 worker 继续服务、经验在 channel 里排队等待。

故障检测上有个干脆的决定：**宁可错杀不可等待**——无法从外部区分"进程崩了"和"进程只是慢"，HotSwarm 直接终止可疑进程，把每次故障变成确定性事件，代价最多是一条 episode。

故障 worker 留下的"遗物"如果不清干净，会静默污染训练数据，这是分布式系统里最容易被忽视的坑：

| 残留物 | 不校验身份的危害 | HotSwarm 的应对 |
| --- | --- | --- |
| 成员记录 | 对端向死进程建连接并无限等待 | 拒绝并等待新记录 |
| 在途控制消息 | 迟到的故障报告误杀新 worker | 忽略 |
| 对端持有的连接 | 新 worker 加入过期的连接组 | 在新 epoch 下重建 |
| 管道中的数据 | 训练把动作配错观测，静默学坏 | 丢弃并重新接收 |
| 排队的传输 | 传输跑在已销毁的连接上 | 在新连接上按原序重放 |

最危险的是倒数第二行：为故障 worker 观测算出的动作若被新 worker 执行，机器人会按一个已经不存在的状态行动；若被录进 episode，训练就会静默学到"动作-观测"错位的样本，而且不会报任何错。SwarmLink 的做法是把动作只投递给发出该观测的 worker，新 worker 丢弃一切带前任身份的回复。

一次机器人故障的完整跨平面流程是：HotSwarm 检测到故障 → 终止残留进程、注销 worker → SwarmLink 照常服务其他机器人 → 故障 worker 的半截 episode（含已上传部分）整体丢弃 → 训练消费已排队的完整经验继续 → 新 worker 同工位启动、新 epoch 重建连接、重放中断调用 → 机器人复位后开启新 episode。实测案例中，20 号机器人 11:24 丢掉移动底盘惯性传感器链路，11:37 原地重启回归，其余车队全程未停。

### 部署与恢复成本模型

实际部署：站点支持 100 台机器人（最大训练用 96 台），云端 6 台服务器共 48 张 A800，站点内两台一级 + 三台二级交换机、互联双 100 Gbps 光纤，边云之间 30 Gbps 专线。FATS 构建在 RLinf（基于 Ray）之上。

![硬件与网络部署](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/infra/hardware-link.png)

> 图解：FATS 的机器人站点网络与云端资源。站点侧最多 100 台机器人，云端 48 张 A800 GPU 承担推理与策略优化，两侧通过一条 30 Gbps 专线连接。

"局部恢复"相比"每次故障全系统重启"值多少？作者用一个简单模型量化：设 $N$ 台机器人、单机故障率 $\lambda$、全系统重启耗时 $t_{\mathrm{restart}}$，则全系统重启方案下可采集时间占比为

$$
A_{\mathrm{sys}} = \frac{1}{1 + N \lambda \, t_{\mathrm{restart}}}
$$

而只重启故障机器人的方案为 $A_{\mathrm{robot}} = 1 - \sum_i \Delta t_i / (N T_{\mathrm{run}})$，在单机故障率与停机时长固定的假设下与车队规模无关。代入 96 机实测参数（$\lambda = 0.0288$ 次/机器人·小时）：全系统重启在 $t_{\mathrm{restart}} = 2/5/10$ 分钟时分别只剩 91.6%/81.3%/68.5% 的采集时间；若车队扩到 1024 台，则暴跌到 50.4%/28.9%/16.9%，而局部恢复始终保持 98.7%。**车队越大，容错架构的收益越悬殊**——这是"异步 + 局部恢复"最硬的量化论据。

## 五、实验验证

解决了算法和系统，接下来是重头戏：真实机器人上的全面验证。

### 实验设置

五个真实桌面操作任务，覆盖柔性物体、精密对位、工具使用和多阶段执行，全部使用双臂 AgiBot G2 机器人（1 个头相机 + 2 个腕部相机 + 本体感觉，动作空间为双臂 delta 关节位置）：

| 任务 | 语言指令 | 难点 |
| --- | --- | --- |
| Flower Arrangement | "Put the flowers into the vase." | 柔性花茎会晃动，需平稳搬运 + 精准插瓶 |
| Clothes Folding | "Fold the cloth." | 可变形织物，约 40 秒连续操作 |
| Phone Charging | "Charge the phone." | 插头插排插、拔 USB-C，高精度对位 |
| Stool Assembly | "Assemble the stool with a screwdriver." | 小螺丝取放 + 螺丝刀工具使用 |
| Phone Packaging | "Pack the phone." | 10 个步骤、2–3 分钟多阶段执行，误差逐级累积 |

![五个任务概览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/tasks/all.png)

> 图解：五个操作任务的实拍场景，自上而下展示各任务的执行流程：插花、叠衣服、手机充电、凳子装配（含螺丝刀拧螺丝）、手机装盒（放说明书、数据线、充电器、手机壳、手机并合盖）。

评测协议值得一提：**所有策略都在两台完全未参与训练数据收集的同型号机器人上测试**，每任务每台 60 次、共 120 次自主执行（无人工干预），成功标准严格到"空抓三次即判失败""花瓶位移超过 2cm 即判失败"这种程度。这测的是经验能否跨机器人个体迁移——同型号机器人之间也存在硬件个体差异，能迁移才说明车队学到的策略具有真正的通用性。对比基线有两个：RLT（单机器人在线残差 RL，用稀疏奖励 + 人类纠正）和 RECAP（多机数据离线 RL，用价值函数估计优势 + 人类纠正），两者在后续学习中**都继续使用人类纠正动作**；另外还以 6 名人类遥操作员（每人每任务 20 次）作为参照。

### 主结果：集体经验把失败率压缩了 86%

![五个任务的主结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/all_tasks_success_by_method_grid.png)

> 图解：五个任务上人类遥操作、SFT、(M)HG-DAgger、RLT、RECAP 与 FERT 的成功率对比。FERT 报告各任务最大训练车队规模下的 checkpoint（插花/叠衣/充电为 16 机，装凳为 32 机，装盒为 96 机），误差棒为 95% Wilson 置信区间。可以看到 FERT（粉色）在所有任务上全面领先，包括超过人类遥操作基线。

数字拆开看更震撼：

| 任务 | (M)HG-DAgger 失败率 (%) ↓ | FERT 失败率 (%) ↓ | 相对失败率下降 (%) ↑ |
| --- | --- | --- | --- |
| Flower Arrangement | 1.67 | **0.83** | 50.0 |
| Clothes Folding | 14.17 | **1.67** | 88.2 |
| Phone Charging | 40.00 | **2.50** | 93.8 |
| Stool Assembly | 18.33 | **6.67** | 63.6 |
| Phone Packaging | 20.83 | **1.67** | 92.0 |
| **五任务平均** | 19.00 | **2.67** | **86.0** |

分阶段看：(M)HG-DAgger 初始化把五任务平均成功率从 SFT 的 36.3% 拉到 81.0%；在此之上，**FERT 不再使用任何新的人类演示或纠正动作**，纯靠车队自主经验把平均成功率进一步推到 97.3%（+16.3 个百分点），相当于把平均失败率相对压缩了 86%。Phone Charging 从 60.0% 到 97.5%（失败率相对降 93.8%），Phone Packaging 从 79.2% 到 98.3%（相对降 92.0%）。

与基线相比，FERT 在全部五个任务上均超过 RLT 和 RECAP，平均分别领先 12.3 和 11.5 个百分点——而这两个基线还额外消耗着人类纠正动作，FERT 一分人力的纠正动作都没用，还省掉了价值函数训练。更夸张的是，FERT 在五个任务上全部超过人类遥操作员的平均表现，平均领先 16.7 个百分点（人类平均：插花 90.8%、叠衣 82.5%、充电 71.7%、装凳 83.3%、装盒 75.0%；充电和装盒上分别领先 25.8 和 23.3 个百分点）。人类的短板恰好出在长时程多阶段任务上——操作员个体在 Phone Packaging 上的成功率从 25% 到 95% 不等，波动极大，而车队策略稳定得多。

五个任务、各车队规模下的完整成功率（两台 held-out 机器人各 60 次，Avg 为 120 次汇总）：

**Flower Arrangement（FERT 最大车队 16 机）**

| 机器人 | SFT | HG-DAgger(1) | HG-DAgger(4) | HG-DAgger(8) | RLT | RECAP | FERT(4) | FERT(8) | FERT(16) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Robot 1 | 23/60 | 47/60 | 53/60 | 60/60 | 56/60 | 58/60 | 59/60 | 60/60 | 60/60 |
| Robot 2 | 20/60 | 56/60 | 56/60 | 58/60 | 58/60 | 56/60 | 57/60 | 59/60 | 59/60 |
| Avg. (%) | 35.83 | 85.83 | 90.83 | 98.33 | 95.00 | 95.00 | 96.67 | **99.17** | **99.17** |

**Clothes Folding（FERT 最大车队 16 机）**

| 机器人 | SFT | HG-DAgger(1) | HG-DAgger(4) | HG-DAgger(8) | RLT | RECAP | FERT(4) | FERT(8) | FERT(16) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Robot 1 | 44/60 | 48/60 | 50/60 | 51/60 | 50/60 | 54/60 | 49/60 | 56/60 | 58/60 |
| Robot 2 | 34/60 | 44/60 | 47/60 | 52/60 | 56/60 | 56/60 | 51/60 | 57/60 | 60/60 |
| Avg. (%) | 65.00 | 76.67 | 80.83 | 85.83 | 88.33 | 91.67 | 83.33 | 94.17 | **98.33** |

**Phone Charging（FERT 最大车队 16 机）**

| 机器人 | SFT | HG-D(1) | HG-D(4) | HG-D(8) | HG-D(Mixed) | RLT | RECAP | FERT(2) | FERT(4) | FERT(8) | FERT(16) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Robot 1 | 7/60 | 1/60 | 39/60 | 43/60 | 41/60 | 46/60 | 45/60 | 20/60 | 52/60 | 58/60 | 58/60 |
| Robot 2 | 12/60 | 0/60 | 32/60 | 28/60 | 31/60 | 45/60 | 54/60 | 28/60 | 56/60 | 56/60 | 59/60 |
| Avg. (%) | 15.83 | 0.83 | 59.17 | 59.17 | 60.00 | 75.83 | 82.50 | 40.00 | 90.00 | 95.00 | **97.50** |

**Stool Assembly（FERT 最大车队 32 机）**

| 机器人 | SFT | HG-D(1) | HG-D(4) | HG-D(8) | HG-D(16) | HG-D(Mixed) | RLT | RECAP | FERT(8) | FERT(16) | FERT(32) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Robot 1 | 41/60 | 2/60 | 50/60 | 52/60 | 39/60 | 50/60 | 48/60 | 53/60 | 41/60 | 57/60 | 57/60 |
| Robot 2 | 35/60 | 0/60 | 46/60 | 44/60 | 37/60 | 48/60 | 49/60 | 49/60 | 49/60 | 54/60 | 55/60 |
| Avg. (%) | 63.33 | 1.67 | 80.00 | 80.00 | 63.33 | 81.67 | 80.83 | 85.00 | 75.00 | 92.50 | **93.33** |

**Phone Packaging（FERT 最大车队 96 机）**

| 机器人 | SFT | HG-D(1) | HG-D(4) | HG-D(8) | HG-D(16) | HG-D(Mixed) | RLT | RECAP | FERT(4) | FERT(8) | FERT(16) | FERT(32) | FERT(64) | FERT(96) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Robot 1 | 0/60 | 0/60 | 14/60 | 37/60 | 42/60 | 47/60 | 54/60 | 47/60 | 32/60 | 45/60 | 53/60 | 56/60 | 57/60 | 59/60 |
| Robot 2 | 2/60 | 2/60 | 25/60 | 41/60 | 39/60 | 48/60 | 48/60 | 43/60 | 29/60 | 40/60 | 51/60 | 55/60 | 57/60 | 59/60 |
| Avg. (%) | 1.67 | 1.67 | 32.50 | 65.00 | 67.50 | 79.17 | 85.00 | 75.00 | 50.83 | 70.83 | 86.67 | 92.50 | 95.00 | **98.33** |

一个有趣的细节：单机 HG-DAgger 在 Phone Charging、Stool Assembly、Phone Packaging 上反而比 SFT 还差（最低仅 0.83%）——过拟合到单个机器人的环境和单个操作员的习惯，换个机器人就崩。这从反面印证了"集体"的必要性。

### 规模效应：机器人越多，学得越好

![成功率随车队规模变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/all_tasks_success_by_robots_combined.png)

> 图解：五个任务上成功率随训练机器人数量变化的曲线。蓝色为 HG-DAgger（从 SFT 出发），粉色为 FERT（从 (M)HG-DAgger checkpoint 出发），彩色虚线为两种初始策略的成功率。每个任务内单机器人采集时间固定，因此车队越大，总经验越多。

固定每台机器人的采集时间，车队规模翻倍意味着总经验量翻倍。结果呈现两种截然不同的规律：

- **HG-DAgger 会饱和甚至退化**：在 Phone Charging 和 Phone Packaging 上，机器人增加到一定数量后成功率不再上涨；Stool Assembly 上 16 机反而比 8 机差。作者推测原因是人类的监控和干预能力跟不上车队扩张——这恰恰证明了"依赖人类纠错"的路线存在规模天花板。
- **FERT 单调向好**：五个任务上成功率总体随车队规模持续增长。Phone Packaging 从 16 机的 86.7% 涨到 96 机的 98.3%（+11.6 个百分点）。不过小车队可能不够用——8 机训练在 Stool Assembly 和 Phone Packaging 上甚至低于初始策略，原因是成功经验不够丰富、覆盖不了策略遇到的各类失败场景，检索自然无从发力。这说明 FERT 的经验池需要一定的"规模门槛"才能转起来。

### 消融实验：失败引导检索贡献有多大

在 16 机 Phone Packaging 上做的消融（完整 FERT 为 86.7%）：

![FERT 消融实验](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/ablation/rft_ablation_fr.png)

> 图解（a）：去掉失败引导检索（w/o FR，即从经验池均匀采样）后，成功率从 86.7% 暴跌到 64.2%，掉了 22.5 个百分点。这是对"失败引导的经验选择"机制最直接的证据。

![权重同步间隔消融](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/ablation/rft_ablation_interval.png)

> 图解（b）：策略同步间隔（横轴，优化步数）的影响。每 500 步同步最优（86.7%）；拉长到 1500 步降到 80.0%（新策略下发太慢），缩短到 100 步降到 75.0%（每个策略版本积累的经验太少）。中等频率最优。

![检索段扩展消融](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/ablation/rft_ablation_extend.png)

> 图解（c）：检索段前后扩展帧数的影响。去掉扩展（$h_{\mathrm{pre}} = h_{\mathrm{post}} = 0$）后成功率从 86.7% 跌到 57.5%，掉了 29.2 个百分点——**匹配片段前后的动作上下文（怎么进入这个状态、之后怎么收尾）几乎和匹配片段本身一样重要**。

![优先级保留因子消融](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/ablation/rft_ablation_decay.png)

> 图解（d）：优先级保留因子 $\rho$ 的敏感性。默认 0.88 最优（86.7%），0.8（遗忘太快，70.8%）和 0.95（遗忘太慢，82.5%）都更差，呈现"既要记住历史检索信息、又要响应近期失败"的中间最优。

完整消融数据（两台 held-out 机器人各 60 次）：

| 配置 | Robot 1 | Robot 2 | Avg. (%) |
| --- | --- | --- | --- |
| FERT 完整配置（16 机） | 53/60 | 51/60 | **86.67** |
| w/o FR（去掉失败引导检索） | 33/60 | 44/60 | 64.17 |
| No extension（无扩展） | 34/60 | 35/60 | 57.50 |
| Retention $\rho = 0.8$ | 44/60 | 41/60 | 70.83 |
| Retention $\rho = 0.95$ | 50/60 | 49/60 | 82.50 |
| Sync interval 100 | 46/60 | 44/60 | 75.00 |
| Sync interval 1500 | 48/60 | 48/60 | 80.00 |

### 系统实测：96 台机器人的一天

最后看 FATS 在真实 96 机训练中（10:37–18:30，含 12:00–13:00 午休，有效运行 6.88 小时）的表现。整个运行共收集 **9,834 条 episode**，午休外平均 65 台机器人同时在跑，每分钟完成 24 条 episode。

![96 机运行 trace](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/fats_trace.png)

> 图解：a）每 5 分钟完成的 episode 数，曲线上方的标记为故障事件（高度正比于受影响机器人数量），阴影为午休。b）30 起故障按起因（计算/网络、硬件、机器人未就绪）与时长分布，圆圈表示重启恢复、方块表示恢复受阻。c）把实测 episode/暂停序列回放代入其他设计后的相对采集量：全系统重启（2/5/10 分钟）只剩 92%/86%/74%，同步轮次制只剩 33%。d）各机器人完成 episode 数分布（45–222 条不等）。

三个关键发现：

1. **故障被完全局部化**。30 起故障波及 22 台机器人（19 起计算/网络故障、9 起硬件故障、2 起未就绪），聚成 19 个事件即平均 22 分钟一起。取离午休和运行端点足够远的 14 个事件，故障后 10 分钟车队的 episode 完成量中位数是故障前的 **99%**，最差也有 88%。30 起中 23 起恢复采集，从故障结束到下一条 episode 的中位间隔仅 12 秒。全部故障合计只损失 8.3 个机器人·小时，占 660 机器人·小时的 **1.3%**。
2. **异步采集的收益主要来自消除"等待"而非容错**。把实测轨迹回放进"同步轮次"设计（每次更新等所有机器人各交一条 episode）：只能收到 33% 的 episode。把所有与故障重叠的暂停换成中位暂停也只升到 42%，把全部暂停都换掉也才 78%——剩下 22% 的损失纯粹来自机器人之间天然的速度差异。同步设计的代价主要不是故障，而是"等最慢的人"。
3. **通信足够快**。交互通道上，Env-to-Rollout（观测上行）95.18% 的操作 20ms 内完成、99% 以上 50ms 内；Rollout-to-Env（动作下发）83.15% 落在 200–250ms，超过 1s 的不到 1%；Env-to-Train（经验传输）99% 以上两分钟内完成。毫秒级的交互与分钟级的经验传输差着四个数量级，这正是把两条路径解耦设计的依据。

![通道通信耗时分布](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/channel_time_20260813_bars.png)

> 图解：三类通道操作的耗时分布直方图（Env-to-Rollout 观测上行、Rollout-to-Env 动作下发、Env-to-Train 经验上传），柱子为各耗时区间内的操作占比与数量。交互通道集中在毫秒级，经验通道以分钟计，量级差异悬殊。

![逐机器人 episode 时间线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/episode_intervals.png)

> 图解：96 机运行中每台机器人的逐条 episode 时间线，成功、失败与系统错误分别着色，空白段为环境复位与操作员休息。可以看到机器人个体差异很大：3、4 号机器人几乎条条成功，6、7 号则大量失败——这种多样性恰恰是集体经验池的价值来源。

![活跃机器人数量变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/active_robots_episodes.png)

> 图解：运行期间正在执行动作的机器人数量随时间变化。浅色为原始计数，深色为高斯平滑（$\sigma = 60$ s）。午休时段全体停摆，其余时间平均约 65 台活跃。

![故障类型与发生时间](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/error_overview_refined.png)

> 图解：30 起故障的发生时间（横轴）、涉及的机器人 ID（纵轴）与故障类型（颜色区分）。注意 17:17 附近 44–47 号机器人因宿主机内存耗尽几乎同时掉线——这类"一坏坏一片"的关联故障被聚类为单个故障事件处理。

### 检索效果可视化：失败到底"检索"回了什么

附录给了每个任务的检索实例（红底为失败上下文，绿底为检索到的成功片段），非常直观：

![Flower Arrangement 检索示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/FR_visualize/flower.png)

> 图解：第一行是机器人插第二朵花失败的片段，下面三行是检索到的三条成功片段——全部精准命中"插第二朵花"这一关键步骤。

![Clothes Folding 检索示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/FR_visualize/clothes.png)

> 图解：失败片段中机器人右手没能抓住 T 恤右角对折；检索结果跨越不同颜色的衣服，准确匹配到同一个关键步骤，说明视觉 + 状态的联合表征抓的是"步骤语义"而非表面颜色。

![Phone Charging 检索示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/FR_visualize/charger.png)

> 图解：Phone Charging 中最易失败的步骤——把充电插头插入排插，需要高精度对位；检索到的成功片段均对应该步骤。

![Stool Assembly 检索示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/FR_visualize/tooluse.png)

> 图解：将螺丝放入螺丝孔的失败尝试，以及检索到的对应成功片段。

![Phone Packaging 检索示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Learning-from-Fleet-Experience-A-Collective-Learning-Paradigm-for-Robots/figs/exp/FR_visualize/phonepack.png)

> 图解：合盖阶段（视线受限下的高精度操作）。上方两行中左臂反复调整盖子的角度但盖子仍然翻倒；下方三行是检索到的成功操作。

这些案例说明 S-DTW + 联合表征确实能把"我在这儿失败了"翻译成"看看别人在这儿是怎么成功的"，这正是 FERT 区别于普通 prioritized replay 的核心机制——它不需要学价值函数，查询直接来自车队自己的失败现场。

## 六、总结与展望

回顾这篇工作的核心要点：

- **范式层面**：提出 CEL（集体经验学习），把车队部署与策略改进接成闭环，机器人扩张不再要求人类专家监督同比例扩张。
- **算法层面**：FERT 用失败片段检索相关成功片段并提升其采样优先级，动作监督全部来自机器人自主成功执行；理论上证明了成功回放对齐（$\mu > 0$）、检索对齐（$\kappa > 0$）与局部改进保证 $J(\theta_{\mathrm{FERT}}) > J(\theta_{\mathrm{unif}}) > J(\theta_0)$。
- **系统层面**：FATS 用 SwarmLink（异步数据平面）+ HotSwarm（原地恢复控制平面）支撑 96 台机器人容错学习，故障损失仅 1.3% 的机器人时间；模型推演显示局部恢复的收益随车队规模放大。
- **实验层面**：5 个真实任务 93.3%–99.2% 成功率，平均失败率相对压缩 86%，超过人类遥操作平均水平 16.7 个百分点；消融证明失败引导检索贡献 22.5 个百分点；成功率随车队规模单调提升（Phone Packaging 16→96 机：86.7%→98.3%）。

局限与方向也很明确：目前只在单一机器人平台、五个任务上验证了任务专用策略，跨任务、跨本体（heterogeneous fleet）的集体学习、以及更长时间尺度上的策略适应，都是下一步。更长远看，CEL 指向一种新的机器人部署模式：车队交付的不只是劳动力，还有不断反哺集体能力的经验——机器人越多，整个车队越聪明，而人类专家只需要站在起点，不必一路陪跑。

> 本文参考自 [Learning from Fleet Experience: A Collective Learning Paradigm for Robots](https://arxiv.org/abs/2610.25736)