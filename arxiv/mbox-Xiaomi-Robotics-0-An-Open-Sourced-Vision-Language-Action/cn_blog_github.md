# Xiaomi-Robotics-0：开源 VLA 的高性能与实时执行实践

## 一句话看点
这篇工作把 VLM 的视觉语义能力和 DiT 的动作生成能力“拼”成一个端到端 VLA，并通过  **训练配方 + 异步执行策略**  同时解决性能和实时性。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/mbox-Xiaomi-Robotics-0-An-Open-Sourced-Vision-Language-Action/figures/fig1.png)
> 图解：整体概览图展示了模型在三类仿真基准上的 SoTA 表现、在双臂真实机器人任务上的高吞吐，以及在多项 VLM 基准上与底座模型的能力对齐。核心信息是：性能不牺牲、实时执行可落地。

## 背景与问题
VLA 模型把图像、语言指令和机器人状态直接映射到动作，是当下机器人策略学习的主流范式。但一个老问题一直存在：  
**模型参数大 → 推理慢 → 动作衔接断裂 → 真实机器人出现卡顿和抖动** 。  
这篇工作聚焦的不是“再提一个新架构”，而是解决  **高性能 + 实时执行**  之间的冲突。

## 核心思路总览
作者把整体方案拆成三块，逻辑很清晰：

1. **强能力来源** ：VLM 负责视觉语义，DiT 负责动作分布建模。  
2. **训练时保留 VLM 能力** ：用 VL 数据 + 机器人轨迹做混合训练，避免 VLM 被动作训练“冲掉”。  
3. **部署时保证动作连续性** ：通过异步执行 + 动作前缀 + 注意力结构修正，避免动作块之间断裂与“偷懒”。

## 模型与数据：VLM + DiT 的混合式组合
### 数据组成
训练用到两类数据：

- 机器人轨迹：来自 DROID、MolmoAct 等开源数据集 + 两项自采集任务（Lego Disassembly 与 Towel Folding），总量约  **200M timesteps** 。  
- 视觉语言数据：超过  **80M**  样本，覆盖 grounding、VQA、caption、embodied reasoning 等任务。

核心目的是：  
**机器人轨迹给动作能力，VL 数据保语义能力** 。

### 模型结构
- VLM：Qwen3-VL-4B-Instruct  
- 动作生成：DiT（Diffusion Transformer）  
- 总参数量：约  **4.7B**

VLM 产出多模态语义特征，DiT 在此基础上生成动作 chunk。

## 训练策略：两阶段预训练 + 针对异步的后训练
### 预训练 Step 1：让 VLM 会“想动作”
使用 Choice Policies 思路，VLM 同时预测多个动作候选与分数，用 winner-takes-all 更新。  
这一步把动作预测融入 VLM，但不会完全改变 VLM 的语义能力。

### 预训练 Step 2：固定 VLM，训练 DiT
VLM 冻结，DiT 用 flow-matching 学动作分布：

$$
L(\theta) = \left\|\mathbf{v}_{\theta}(\mathbf{o}_{t}, l, \mathbf{s}_{t}, \tilde{\mathbf{a}}_{t:t+T}^{\tau}, \tau) - \mathbf{u}(\tilde{\mathbf{a}}_{t:t+T}^{\tau}, \mathbf{a}_{t:t+T}, \tau)\right\|_{2}^{2}
$$

其中：
- $\tilde{\mathbf{a}}_{t:t+T}^{\tau} = \tau \mathbf{a}_{t:t+T} + (1 - \tau)\boldsymbol{\epsilon}$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

这样做的意义是：  
**让动作生成在连续空间内更平滑，同时不损伤 VLM 的语义能力** 。

### 后训练：为异步执行做适配
异步执行的关键问题是：  
**模型可以“偷懒”复制前缀动作，而忽视视觉与语言输入。**

解决办法有三点：

- **动作前缀** ：条件化前一个动作块，保证连续性  
- **RoPE 位置偏移** ：区分“干净动作前缀”和“带噪预测动作”  
- **Λ 形注意力** ：后面动作不能过度看前缀，只能依赖视觉与语言输入

本质是：  
**在保证连续性的同时，强制模型保持“对视觉语言的反应性”。**

## 部署策略：异步执行的时间对齐
异步执行核心是  **“推理在跑，机器人也不停”** 。

执行流程：

1. 当前动作块先执行 $T_e$ 步  
2. 推理下一块时，继续执行当前块剩余动作  
3. 新块从 $\Delta t_{\mathrm{inf}}$ 时刻无缝接入  

关键约束：

- 设定 $\Delta t_{c} \ge \Delta t_{\mathrm{inf}}$  
- 确保动作前缀覆盖整个推理窗口  

在 RTX 4090 上推理延迟约  **80ms**  ，动作执行维持 30Hz 的统一时间线。

## 实验结果：三个仿真基准 + 两个真实机器人任务
### 仿真基准
1.  **LIBERO**  ：平均成功率  **98.7%**  
2.  **CALVIN**  ：  
  - ABCD→D 平均 5 连任务长度  **4.80**  
  - ABC→D 平均 5 连任务长度  **4.75**  
3.  **SimplerEnv**  ：  
  - Google Robot Visual Matching  **85.5%**  
  - Google Robot Variant Aggregation  **74.7%**  
  - WidowX  **79.2%**

这些数字说明：  
**性能已经达到 SoTA，并且是跨环境稳定提升** 。

### 真实机器人任务
1.  **Lego Disassembly**  ：  
  - 吞吐高于同期异步方案  
  - 动作更连贯  
2.  **Towel Folding**  ：  
  - 1.2 pcs/min，高于 1 pcs/min 基线  

结论：  
**异步执行不是牺牲性能，而是提升效率的关键因素** 。

## 视觉语言能力是否被保住？
作者用 10 个 VL 基准验证，包括 MMBench、SEED、POPE、MME 等。  
结果显示：

- 训练了 VL 数据的版本能力保持良好  
- 不加 VL 数据的版本几乎全灭（灾难性遗忘）

因此可以确认：  
**联合训练是保住 VLM 能力的必要条件** 。

## 关键创新点总结
- **训练配方层面** ：VL 数据与机器人轨迹混训，避免语义能力遗忘  
- **模型层面** ：VLM + DiT 的清晰分工  
- **执行层面** ：异步执行 + Λ 形注意力，解决实时衔接和“动作偷懒”问题  

这让 VLA 迈向一个更实际的方向：  
**“可落地、可长时间运行的真实机器人策略”。**

> 本文参考自 [Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution](https://arxiv.org/abs/2602.12684)