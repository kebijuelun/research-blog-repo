# Gemini 4 Argon：物理推理走进机器人研发

让机器人踩下脚踏垃圾桶、再把汽水罐投入桶中，难点不只是认出物体，而是同时处理三足支撑、脚踏位置、手部与桶口对齐以及接触后的状态变化；任何一环出错，任务都可能失败。这类问题卡住的，正是机器人从固定脚本走向新任务的能力。Google 在 2026 年 10 月 13 日发布的 Gemini 4 Argon，尝试用一个通用物理推理模型统一处理多视角视频、3D 几何、接触力学和本体感知数据，并生成控制代码与 MuJoCo 仿真。围绕这一思路，Google 展示了机器人控制、Real2Sim、物理理解、研究与自我改进四类工作流。官方结果显示，Argon 在 ERQA 纯推理评测中取得 87.0%；另一项仿真数据微调实验中，真实机器人成功率由 56% 提升至 75%，完成时间由 35 s 降至 24 s。

![Gemini 4 Argon 机器人主题主视觉](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/argon__robotics__hero.original.png)

> 图解：Google 为 Argon 发布的机器人主视觉，图中是抽象机械臂，不对应具体实验设置。模型能力需要结合下方的基准条件和演示流程判断。

## ERQA 的 87.0%，能说明什么

Embodied Reasoning Question Answering（ERQA）考察模型理解并解决 3D 环境中的物理与空间问题。图中评测关闭工具调用，样本数为 400，因此它更接近对模型自身推理能力的测试，而不是让模型借助外部程序完成问答。

![Gemini 4 Argon 在 ERQA 基准上的对比结果](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/gemini-4-argon__benchmark__robotics.original.jpg)

> 图解：纵轴为 ERQA 得分，越高越好；测试条件为 Tool off、N=400。Gemini 4 Argon 得分 87.0%，图中其余模型依次为 Claude Opus 5.5 的 82.3%、GPT-6 Astra 的 81.0% 和 Claude Fable 5.1 的 74.0%。

这个结果支持的是“物理与空间问答能力更强”，不能直接换算成真实机器人的任务成功率。真实控制还受相机标定、接触不确定性、执行器误差和安全约束影响。Argon 演示真正有价值的地方，在于它把这些外部环节也纳入了工作流，而不只回答一道关于物理世界的题。

## 控制机器人：把规划拆成可校验的动作

### Spot：先在 MuJoCo 练习，再踩真实踏板

第一个案例使用 Boston Dynamics Spot，硬件包含 12 自由度腿组和 6 自由度机械臂。任务是学习用脚踩开脚踏垃圾桶并丢弃汽水罐。Argon 先建立带关节的 MuJoCo 垃圾桶数字孪生，在仿真中练习三腿支撑平衡，再到真实硬件上执行踩踏板和投放动作。

提示词中的约束比“把罐子扔进桶里”更关键：踩下踏板前，要用前视相机确认右前脚目标位于踏板中央；释放物体前，还要由 `foot-verifier` 和 `hand-verifier` 两个验证代理分别检查脚踏动作与手部在桶内的 3D 对齐。演示流程还包含腕部相机检查和手部居中确认。换句话说，模型不是只生成一段运动代码，而是把“观察—练习—验证—执行”组织成了闭环。

### 零样本机械臂：坐标网格与动作原语共同降低难度

第二个案例要求简单机械臂把倒下的杯子扶正，再把皱餐巾放入杯中。这里的“零样本”并不等于完全没有工程辅助。Argon 使用经过标定的腕部相机，画面叠加 2 cm 世界坐标网格、当前夹持中心、夹爪闭合线和物体长轴方向，再通过 `look_at`、`grasp`、`place_until_contact`、`move_until_contact` 等动作原语分多轮完成任务。

系统允许按任务难度选择调用粒度：粗粒度执行为 2～4 次模型调用，抓取前悬停验证为 4～7 次，精细视觉伺服为 8～20 次。每轮动作后都要检查更新后的腕部图像；若状态为 `grasp_failed`，就重新观察并恢复，而不是沿错误轨迹继续执行。这个案例的聪明之处，是把连续控制变成一系列可观察、可纠正的小决策，但它也明确依赖相机标定和预先封装好的机器人 API。

### Omni：在视频世界模型中反复修正动作提示

第三个案例发生在 Gemini Omni 视频世界模型中，不应与真实机器人控制混为一谈。Argon 要把散落的 G、R、A、N、O 五块字母牌整理成 A-R-G-O-N，通过 33 步视频 rollout 逐步转向目标状态。它不是一次性写好完整提示，而是检查中间关键帧，再生成下一组带时间戳的动作提示。

提示策略也很具体：每次只描述一个简单、单方向动作；不操作的手要收回，避免遮挡工作区；物体移动后，明确说明哪些区域已经腾空。Argon 在这里更像一位根据样片不断修改分镜的导演，核心能力是视觉反馈下的序列修正，而非直接输出关节力矩。

闭环控制仍然需要一个可练习、可验证的环境。接下来更棘手的问题是：怎样从真实记录中低成本地得到这样的环境？

## Real2Sim：把真实观测变成可交互的仿真

Real2Sim 指从真实图像、视频或机器人日志重建仿真环境，再把仿真用于练习、验证或数据生成。难点不只是把画面做得像，还要让物体在接触时表现得像。Argon 的三组案例分别使用机器人视角、人类第一视角和手绘草图作为起点。

### 机器人视频：同时对齐画面与接触力

在机器人遥操作案例中，Argon 根据真实 episode 构建 MuJoCo 场景，获取或生成物体资产，并迭代调整场景。视觉目标是让 `head_left`、`left_wrist` 和 `right_wrist` 三个相机的仿真渲染对齐真实观测；物理目标则要求 Sharpa 指尖传感器坐标系中的接触力和力矩，在时间、大小与方向上匹配日志中的 `tactile_force`。

可调整项包括几何、碰撞形状、质量、惯量、摩擦、关节和初始位姿。离线规划的机器人命令只能在真实指尖接近物体的帧中修改，且相对原遥操作轨迹的偏差不得超过臂关节 0.55 rad、手关节 0.65 rad，其余帧保持原轨迹不变。机器人关节通过关节空间比例—微分（PD）控制跟踪记录状态。

更重要的是原文列出的“禁止仿真作弊”约束：物体只能由 `mj_step` 中的真实接触推动；开始仿真后不能修改模型参数、施加外力或力矩、添加焊接约束、直接设置或回滚物体状态、关闭碰撞，也不能把日志中的触觉读数冒充求解器输出。最终的可交互 HTML 查看器同步展示 3D 场景、接触点、力箭头、三路真实与仿真画面以及指尖力曲线。这样的双重校验很关键，因为视觉相似的场景仍可能拥有错误的物理性质。

### 人类视频：把手部轨迹迁移到五指灵巧手

在人类第一视角视频中，Argon 先识别与操作有关的物体，再利用相机内参和外参恢复桌面 3D 坐标。对于碗、凹槽托盘等凹形容器，简单的包围盒无法表达内部空间，因此场景使用复合凸分解网格，让物体能够真正放入容器，而不是停在虚假的封闭表面上。

随后，人类手腕和手指关键点被重定向到 76 自由度双臂人形机器人，其中每只五指手有 22 个自由度。仿真以 500 Hz 运行，即 `dt = 0.002 s`，且不使用运动学焊接约束。验证条件包括拇指与对指形成力封闭、物体干净抬升超过 2 cm，以及最终稳定放置。这里检验的不只是“手的位置像不像人”，还包括接触下能否真的抓起和放下物体。

### 手绘草图：从空间意图生成全身投掷轨迹

第三个起点更粗略：一张手绘草图，内容是 Apollo 2 人形机器人沿抛物线投球，击倒桌上按 3-2-1 堆叠的 6 个箱子。Argon 根据草图构建 GPU 加速的 `mujoco_warp` 场景，并生成 76 自由度全身与五指投掷轨迹；模型由 32 自由度身体和两只 22 自由度灵巧手组成，最后导出 Blender 4.5 Cycles 渲染。

投掷轨迹设为 50 Hz、140 步，共 2.80 s，分为稳定站立与五指抓握、躯干和肩部蓄力并配合踝髋质心补偿、向前加速，以及在手部速度峰值附近同步张手释放四个阶段。物理验证要求骨盆漂移小于 3 cm，并击倒全部 6 个目标；原文描述的演示完成了 6/6 击倒。这个案例把草图中的空间布局转成了可检验的运动和接触过程，但演示结果仍限于给定场景，不能据此推断任意草图都能得到可靠仿真。

能够构建环境之后，机器人学习还面临数据问题：一段视频里哪些动作在推进任务，哪些只是停顿、滑脱和重复尝试？

## 看懂动作质量：从视频标注到价值函数

### 折纸视频：四层事后标注区分进展与返工

Argon 可以从视频中生成细粒度分割和文字标签，并识别停顿、不稳定抓取和笨拙操作。Google 展示了一段 136 s 的双臂 ALOHA 2 折纸视频，任务是把方形纸折成狐狸。Argon 输出时间对齐的四层事后标注：G1 描述整段任务目标，G2 包含 6 个连续语义子任务，G3 细分为 17 个接触级原子动作，最后再由质量控制层重新加权。

G3 的原子动作通常持续 3～15 s，对齐抓握闭合、折痕扫动、释放或重试。对于没有推进任务的滑脱、分层和重试循环，进度曲线保持不增长，并在行为克隆训练中将对应片段降权至 0.15x～0.25x。这样做避免了“视频时长相同，训练价值就相同”的假设，把失败恢复与有效进展区分开来。

### 十二球称量：视觉读数驱动物理搜索

“找异常球”案例把经典天平谜题变成机器人任务。场景中有 12 个外观相同的台球，其中 11 个为 150 g，异常球为 270 g 或 30 g；两台 7 自由度 Franka FR3 机械臂使用 Roberval 双盘天平，另有红色重球托盘和蓝色轻球托盘。只有在不超过 3 次称量内找到异常球并放入正确托盘，任务才获得 1.00 奖励。

求解策略只使用双目 RGB 图像和机器人本体感知，不开放特权任务状态。它需要分割球、天平盘和托盘，三角化 3D 坐标，再通过双目垂直像素差读取天平倾斜；在狭窄天平盘内抓放时，还要将手腕偏航旋转 90° 以避开边框。原文演示在 3 次称量内确认了较重的异常球。这个案例同时考验组合搜索、视觉测量和操作约束，而不只是识别哪个球更重。

### 价值函数：把任务完成度画成连续曲线

Google 还将 Argon 用作机器人任务的通用价值函数。这里的“价值”可以理解为对任务进展的连续估计，而不是只在结束时给出成功或失败。面对一段来自人类 UMI 数据集的 60 s 双臂圆环分类操作，Argon 同步检查头部、左夹爪和右夹爪三路视频，并重点观察接触状态切换附近的内容。

输出包含密集的任务进度曲线，以及 28 个带活动标签的轨迹片段；各片段分别呈正、零或负进度斜率。相比单一成功标签，这种曲线能够暴露倒退、停顿和恢复过程。不过，该案例展示的是进度预测与分解结果，原文没有继续给出用它训练策略后的提升数据。

理解了动作质量，下一步自然是让模型参与数据和策略本身的改进，而不只承担分析工作。

## 进入研发循环：生成数据、硬件迭代与论文复现

### 仿真生成数据：真实成功率由 56% 升至 75%

在这一组结果中，输入只有一张真实桌面的俯视 RGB 照片，工作区包含两台 7 自由度 Franka FR3 机械臂、4 个 RGB 相机和 9 个派对道具。Argon 自动重建网格与碰撞体，甚至为软帽建立弹簧铰接的变形片，然后定义 77 个单臂和双臂任务，涵盖抓放、扶正、定向推动、空中交接、双臂铲推和流水线清桌等类型。

接着，Argon 生成闭环逆运动学轨迹，在 ±10 cm 位姿随机化下验证 `reward == 1.0`，导出 4,500 多条域随机化 episode，用于微调视觉—语言—动作（VLA）模型。Google 报告，在真实物理机器人评测基准上，成功率由 56% 提升到 75%，增加 19 个百分点；完成时间由 35 s 缩短到 24 s。

这是原文中较少见的“仿真生成数据—真实硬件结果”完整证据链，说明 Argon 的价值不止在演示控制。不过，正文没有给出评测任务数、重复次数或方差，因此这组数字应视为特定基准结果，不能直接外推到其他机器人或开放环境。

### SO-101：根据日志和录像迭代真实硬件

在一个 6 自由度 SO-101 机械臂上，Argon 通过试错和自我反思改进乐高积木抓放。流程先要求在不移动硬件的情况下识别物体、输出规划表和标注图，再让夹爪悬停于积木上方，用腕部相机与外部视角核对位置。进入真实执行后，失败时检查日志和双视角录像，定位原因，先做 dry-run 验证修复，再重试；成功后还要在安全限制内至少提速 2 倍。

三轮记录给出了具体变化：第 1 轮为 58.9 s；第 2 轮缩短到 22.9 s，但投放时擦到木盒边缘；第 3 轮为 17.4 s，并完成居中投放。这里既有速度优化，也有放置质量改进。不过，三轮迭代展示的是单次任务的改进过程，并不等同于跨位置、跨物体的成功率统计。

### 论文复现：把控制器移植进 MuJoCo WebAssembly

Argon 还被用于机器人论文复现。每篇论文只给一个主提示，模型读取论文及可选的作者 checkpoint 或 URDF 资产，把物理模型和闭环控制器移植为自包含的 MuJoCo WebAssembly 浏览器实验室。原文列出的技术跨度包括 DeepLoco 的 7 网络分层运动控制、HUSKY 的 50 Hz PPO actor，以及 Whole-Body MPC 中运行在 Web Worker 的 iLQR。

复现要求匹配论文的物理时间步、执行器增益 `kp/kv`、力矩上限、运动学约束和控制频率，并生成带相机预设、实时遥测和扰动注入的模拟器及交互式视觉文章。Google 展示了 Robotic Soccer OP3 1v1、DeepLoco、ANYmal C 大规模并行深度强化学习、Whole-Body MPC Go2 和 HUSKY G1 滑板五个案例，并称 Argon 共复现了 68 个交互式浏览器沙盒。

自动验收使用 Playwright 驱动无头 Chromium，要求运行 300 多帧期间控制台零错误且任务进度非零。这能证明浏览器实验室可运行并产生行为，却不能单独证明论文中的全部指标已被严格复现；要确认科学等价性，仍需对照原论文的测试协议和定量结果。

## 开发者现在能怎么用

Google 已开放三个入口，适用条件并不相同：

- **Google AI Studio** ：面向 Google AI Ultra 订阅用户。
- **Gemini API** ：面向 AI Studio 中 Gemini API 的付费层级，在 Interactions API 中使用 `gemini-4-argon`，并在发布当天获得 `google-genai` SDK 支持。
- **Google Antigravity** ：面向 Google AI Ultra 订阅用户，可将 Gemini 4 Argon 设为活动模型，用于复杂规划、`/goal` 运行和多代理编排。

## 结语：一条完整链路已经出现，可靠性仍需单独验证

Google 将 Argon 定位为迈向物理通用人工智能（physical AGI）的重要一步。就现有材料而言，更稳妥的判断是：它展示了通用物理推理模型贯穿机器人研发链路的可能性，尤其是把视觉理解、代码生成、仿真校验和硬件反馈连起来的方式；ERQA 的 87.0% 支持其物理问答能力，真实机器人 56% 到 75% 的提升支持特定数据生成流程的有效性，但两者都不能替代广泛硬件上的可靠性测试。

原文主要以精选演示、提示词和少量汇总指标呈现能力，没有系统公开跨任务成功率、失败分布、统计波动、运行成本与安全边界。机器人学习、仿真重建和 VLA 数据工程的开发者值得重点关注，但更合理的试用方式，是选择一个已有真实基线、能够闭环校验的小任务，同时记录成功率、完成时间和失败类型，再判断它是否真正减少了研发工作，而不是把单个演示直接外推到开放环境。

## 原文图表补充

![Google for Developers](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/g-dev.svg)

![figure-2.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-2.svg)

- Facebook

![figure-3.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-3.svg)

- Twitter

![figure-4.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-4.svg)

- LinkedIn

![figure-5.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-5.svg)

- Mail

![figure-6.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-6.svg)

-

![figure-9.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-9.svg)

Previous

![figure-10.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-10.svg)

Related Posts

![Three layers, one repo: a tour of google/adk-recipes](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/1_1.2e16d0ba.fill-800x400.png)

AI Cloud Announcements Solutions

![figure-12.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-12.svg)

![Evaluating AI agents: moving past the vibe check](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/evaluating-ai-agents-banner-1600x9.2e16d0ba.fill-800x400.png)

AI Cloud Best Practices Industry Trends

![figure-14.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-14.svg)

![Accelerating Spatio-Temporal Attention for Video Diffusion on TPUs](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/header_2_UVrdPMU.2e16d0ba.fill-800x400.jpg)

AI Cloud Tutorials Case Studies

![figure-16.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-16.svg)

![ML Drift: Next-Gen GPU AI/ML Inference at the Edge](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/Gemini_Generated_Image_c5igf5c5igf.2e16d0ba.fill-800x400.jpg)

Mobile Web Case Studies Announcements

![figure-18.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Gemini-4-Argon-for-Robotics--Google-Developers-Blog/images/figure-18.svg)

> 本文参考自 [Gemini 4 Argon for Robotics](https://deepmind.google/blog/gemini-4-argon-for-robotics/)