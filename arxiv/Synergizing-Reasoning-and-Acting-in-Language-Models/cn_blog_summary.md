# ReAct 论文精华：让大模型「边思考边行动」

ReAct（Reasoning + Acting，ICLR 2023，Princeton + Google Brain）提出了一个简单却深刻的视角转换：**把「思考」看作一种不改变环境、只更新上下文的特殊动作**。形式化地说，将 agent 的动作空间扩充为 $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$（$\mathcal{L}$ 为语言空间），推理轨迹（thought）与任务动作（action）便能在同一个自回归生成过程中自然交错——reason to act（用推理指导行动），act to reason（用行动获取知识）。

此前推理与行动是两条平行线：Chain-of-Thought 纯靠内部知识「闭门造车」，极易幻觉；WebGPT、SayCan 等行动类方法则缺乏高层抽象推理，且需要昂贵的模仿学习或 RL 训练。ReAct 仅用**冻结的 PaLM-540B + 1~6 个手工编写的 few-shot「思考-动作-观测」交错轨迹**，无需任何训练即可工作。thought 承担多种职能：分解目标、注入常识、从观测提取关键信息、跟踪进度、处理异常、合成答案；知识密集任务中思考与动作密集交替，决策任务中则稀疏出现。

**知识密集任务（HotpotQA / FEVER）**：配合一个仅含 search / lookup / finish 三个动作的极简 Wikipedia API，ReAct 稳定超过纯行动的 Act（FEVER 60.9 vs. 58.9），并显著优于纯推理的 CoT（60.9 vs. 56.3）。人工分析 200 条轨迹发现清晰 trade-off：幻觉占 CoT 失败的 56%，而 ReAct 失败中幻觉率为 0%——外部检索让轨迹更 grounded；但交错结构也限制了推理灵活性（reasoning error 47% vs. CoT 16%），且 23% 失败源于搜索无效。最佳结果是 ReAct 与 CoT-SC 的启发式组合（HotpotQA EM 35.1，FEVER Acc 64.6）。微调实验更有说服力：用 ReAct 自生成的 3,000 条正确轨迹微调 PaLM-8B/62B 后，小模型 ReAct 逆袭超过所有 540B prompting 方法——因为微调 ReAct 教的是「如何检索和使用知识」这一可迁移技能，而非背诵知识。

**决策任务（ALFWorld / WebShop）**：在文字家务游戏 ALFWorld 上，ReAct 最佳 trial 达 71% 平均成功率，远超 Act 的 45% 和用 10^5 条专家轨迹训练的 BUTLER 的 37%，对 Act 平均相对提升 62%。在 118 万真实商品的 WebShop 上，ReAct 成功率 40.0%，比 Act 绝对提升 10 个百分点，也超过 IL（29.1%）和 IL+RL（28.7%），但与人类专家（59.6%）仍有差距。对比 Inner Monologue 的消融（ReAct-IM 仅 53%）证明：灵活、稀疏、多样的内部推理才是关键，而非简单复述环境反馈。

局限包括上下文长度瓶颈、推理灵活性下降易陷入重复循环、依赖检索质量。ReAct 的贡献不在复杂技术，而在这一范式带来的可解释、可诊断、甚至可被人类实时「编辑思考」纠正的决策轨迹——今天各类 LLM Agent 框架中习以为常的 Thought-Action-Observation 循环，正是从这里开始的。

> 本文参考自 [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)