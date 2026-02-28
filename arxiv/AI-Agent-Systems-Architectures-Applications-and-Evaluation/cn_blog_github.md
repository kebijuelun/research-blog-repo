# AI Agent Systems：架构、应用与评估全景解读

## 一句话总览
这是一篇面向 AI Agent 的系统性综述：从 **架构与控制循环**、**学习机制**、**系统工程与基础模型**，到 **应用版图与评估体系**，再到 **未来研究方向**，给出了完整的“从模型到系统、从能力到可用性”的框架。

## 1. 为什么需要 AI Agent
传统对话式模型的瓶颈是：只能“回答”，很难“完成任务”。真实任务需要 **多步决策**、**工具调用**、**状态维护**、**失败恢复**、**安全约束**。AI Agent 的核心就是把 LLM/VLM 变成一个 **可执行的控制器**，让它能在真实环境里“做事”。

## 2. Agent 的核心执行循环
AI Agent 可以被理解为：一个带工具、带记忆、带验证器的闭环控制系统。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section1.png)
> 图解：Agent 的执行循环通常包含 **观察 → 规划 → 工具调用 → 验证 → 记忆更新** 的闭环流程。横向代表时序执行，纵向是模型与环境之间的交互路径。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section2.png)
> 图解：Agent-centric 范式强调模型必须嵌入到 **工具-环境交互闭环** 中，而非仅输出文本。图中展示了决策与环境反馈的循环依赖关系。

## 3. Agent Transformer：统一抽象
论文提出一个清晰的抽象：Agent Transformer 由五个部件组成：
- `πθ`：策略核心（LLM/VLM）
- `M`：记忆系统
- `T`：工具集
- `V`：验证器/批评器
- `E`：环境

核心公式如下：

$$
\mathcal{A} = (\pi_\theta,\mathcal{M},\mathcal{T},\mathcal{V},\mathcal{E})
$$

执行循环简化为：

$$
o_t \leftarrow \mathrm{Obs}(\mathcal{E}_t), \quad
m_t \leftarrow \mathrm{Retrieve}(\mathcal{M}_t, o_t)
$$

$$
\tilde{a}_t \sim \pi_\theta(\cdot|o_t,m_t), \quad
\hat{a}_t \leftarrow \mathrm{Validate}(\mathcal{V}, \tilde{a}_t)
$$

$$
\mathcal{E}_{t+1} \leftarrow \mathrm{Exec}(\mathcal{E}_t,\mathcal{T},\hat{a}_t), \quad
\mathcal{M}_{t+1} \leftarrow \mathrm{Update}(\mathcal{M}_t,o_t,\hat{a}_t,\mathcal{E}_{t+1})
$$

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section3.png)
> 图解：Agent Transformer 把 **工具接口、记忆、验证器** 显式化，让“推理”变成可审计的交互轨迹。

**关键创新**：Agent 不再是文本生成器，而是 **风险感知、预算约束的控制器**。高风险动作必须触发更强的验证或人类确认。

## 4. 学习与优化：Agent 的能力从哪里来
Agent 学习不是单一算法，而是多个层面协同。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section4.png)
> 图解：Agent 学习堆栈由三层组成：**学习机制**、**系统工程**、**基础模型适配**。

### 4.1 强化学习（RL）
RL 适合长任务决策，但工具环境成本高、奖励稀疏，因此更常见的是 **离线 RL** 与 **安全 RL**。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section5.png)
> 图解：RL 在 Agent 中通常作为底层控制器或高风险动作优化器。

### 4.2 模仿学习（IL）
大量 Agent 实际靠 **专家轨迹训练**，结构化工具调用轨迹成为监督信号。

![Figure 6](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section6.png)
> 图解：IL 强调从演示轨迹中学习工具使用与决策流程，而不是直接学答案。

### 4.3 传统 RGB 组件（Rule/Graph/Behavior Tree）
在很多工程系统里，规则和流程仍是核心安全底座。

![Figure 7](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section7.png)
> 图解：规则系统与行为树提供可预测、可验证的执行路径，是 Agent 的安全保障。

### 4.4 In-context Learning
通过 Prompt 和示例，快速教会模型如何使用工具、遵守格式。

![Figure 8](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section8.png)
> 图解：In-context learning 将“工具协议”以示例方式植入上下文。

### 4.5 系统层优化
核心是三角权衡：**可靠性、延迟、成本**。

![Figure 9](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section31.png)
> 图解：Agent 系统优化本质是对成本、可靠性与延迟的多目标权衡。

## 5. 系统工程层面的关键模块
Agent 的稳定性不来自模型，而来自系统工程。

### 5.1 组件化设计
- LLM 核心
- 记忆与检索
- 工具路由
- 验证器/批评器
- 多智能体协作

### 5.2 基础设施
- 工具沙箱与权限控制
- Schema 校验
- 审计日志
- 缓存与可观测性

![Figure 10](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section10.png)
> 图解：基础设施层保障 Agent 可部署、可审计、可回放。

## 6. Agent Foundation Models：模型侧的适配
预训练 + 微调决定了 Agent 的 **接口意识** 与 **工具纪律**。

![Figure 11](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section11.png)
> 图解：Agentic Foundation Models 需要针对工具调用与规划进行显式适配。

## 7. 应用版图与分类
论文提出了从 **交互形态**、**生成目标**、**推理底座** 三个维度的分类法。

### 7.1 Generalist Agents
广泛任务型 Agent：编码、企业流程、浏览、分析等。

![Figure 12](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section13.png)
> 图解：通用 Agent 的核心挑战是长任务可靠性与工具稳定性。

### 7.2 Embodied Agents
物理世界 Agent（机器人、智能设备）。

![Figure 13](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section14.png)
> 图解：人机协作中的共享自治，需要强验证与风险控制。

### 7.3 Generative Agents
生成式 Agent：社会模拟、内容生成、叙事互动。

![Figure 14](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section16.png)
> 图解：生成式 Agent 的核心难点是长期一致性与状态记忆。

### 7.4 AR/VR & Mixed Reality
低延迟、多模态、空间对齐。

![Figure 15](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section17.png)
> 图解：AR/VR Agent 需要低延迟多模态感知与可控动作输出。

### 7.5 情感与社会推理
情感 Agent 必须兼顾安全与人格一致性。

![Figure 16](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section19.png)
> 图解：情感 Agent 的风险来自人格漂移与诱导性对话。

### 7.6 Neuro-Symbolic Agents
结合符号系统提高可验证性。

![Figure 17](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section20.png)
> 图解：神经-符号 Agent 用符号工具做可控执行，用 LLM 做规划与解释。

## 8. 应用任务版图
Agent 的应用已经扩展到 **软件工程**、**企业流程**、**浏览器操作**、**多模态助手**、**游戏与机器人**。

![Figure 18](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section22.png)
> 图解：Agent 的应用覆盖从网页执行到企业流程、到机器人控制的完整版图。

![Figure 19](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section24.png)
> 图解：企业 Agent 强调权限控制、合规审计与工具链治理。

![Figure 20](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section25.png)
> 图解：Web/GUI Agent 必须应对 UI 变化、弹窗、A/B 测试等动态干扰。

![Figure 21](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section26.png)
> 图解：多模态实时助手需要 OCR、ASR、视觉检测等工具协作。

![Figure 22](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section27.png)
> 图解：游戏分析型 Agent 强调“可追溯的工具执行链”。

![Figure 23](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section28.png)
> 图解：机器人 Agent 的核心是 **高层规划 + 低层控制** 的层级结构。

![Figure 24](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/AI-Agent-Systems-Architectures-Applications-and-Evaluation/section29.png)
> 图解：图像语言 Agent 的关键是把视觉证据拆成可验证的中间结果。

## 9. 评估体系：不只是“成功率”
论文提出完整的多维评估体系。

### 9.1 任务成功率

$$
\mathrm{SuccessRate}=\frac{1}{N}\sum_{i=1}^{N} s_i
$$

### 9.2 成本与效率

$$
\mathrm{Tokens}_i = x_i + y_i, \quad
\mathrm{Cost}_i = p_{\mathrm{in}}x_i + p_{\mathrm{out}}y_i
$$

### 9.3 工具正确性

$$
\mathrm{ToolSelAcc}=\frac{\sum_{i,j} \mathbf{1}\{\hat{\ell}_{i,j}=\ell_{i,j}\}}{\sum_i T_i}
$$

### 9.4 轨迹质量与鲁棒性

$$
\mathrm{LoopRate}_i = 1-\frac{\mathrm{uniq}(\tau_i)}{T_i}
$$

### 9.5 安全与合规

$$
\mathrm{ViolationRate}=\frac{1}{N}\sum_{i=1}^N q_i
$$

**关键观点**：Agent 的评估一定是系统级的，必须考虑 **工具调用、成本、延迟、鲁棒性、安全性**。

## 10. 未来研究方向
### 10.1 可验证的工具执行
需要把工具调用变成 **可审核的合同**，建立前置/后置条件与证明链。

### 10.2 长期记忆与安全
记忆是能力也是攻击面，必须解决 **写入策略、追溯性、污染防护**。

### 10.3 预算化规划
Agent 需要根据风险与预算动态分配计算资源。

### 10.4 可复现评估
需要标准化日志、环境版本、随机种子，避免不可复现。

### 10.5 多智能体治理
协作需要角色分工、权限边界与冲突解决机制。

## 11. 总结
这篇综述的最大贡献是：把 AI Agent 从“模型能力”重新定义为“系统能力”。可靠性、可控性、成本、安全，都不是模型本身决定的，而是 **架构 + 工具 + 评估 + 治理** 共同决定的。

> 本文参考自《AI Agent Systems: Architectures, Applications, and Evaluation》