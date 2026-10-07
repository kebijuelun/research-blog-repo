# Mistral Large 4 精华总结：万亿开源巨兽，网络安全全场第一

Mistral 发布公测版 Mistral Large 4（ML4，昵称「Le Chonk」），目标是用开源权重模型在编程、Agent 和企业级场景正面硬刚闭源旗舰。最亮的两个数字：安全漏洞复现与修复测试 **82%** 全场最高分，Cybench 网络安全挑战赛解题率 **93%**。

## 它是什么

- **规模**：总参数 1 万亿、激活 490 亿的 MoE，原生多模态，覆盖 160+ 种语言；
- **获取**：API 已在 Mistral Studio 公测，权重本月底开源；
- **主权叙事**：在欧洲自有数据中心用 3800 张 Grace Blackwell GPU 从零训练，训练到推理全链路可在欧洲法律框架内闭环——对安全团队而言，自部署意味着不会被闭源厂商的「拒答」在事件处置中途卡脖子。

## 五大战场的关键数字

- **网络安全**（最大亮点）：漏洞复现+打补丁 82%，为所有参评模型最高——多个顶级闭源模型（含 Claude Opus 5.5、GPT-6 Astra）因安全过滤器拒绝执行而得分接近零；Cybench 解题率 93%；Artificial Analysis Cyber Index 全球前五。
- **Agent 编程**：DeepSWE 61.7%、SWE-Atlas-QnA 59.4%、Terminal-Bench 4 28.3%，合成 Coding Agent Index **49.8%**，领先 DeepSeek V4 Pro 与 Qwen3.8 Max；Surge AI 人工盲评 3.74 分排第二，仅次于 Claude Opus 5（4.22）——官方坦然公布差距。
- **Agent 工作流**：AutomationBench（657 条 Gmail/Slack/Salesforce 真实流程）**59.9%**；AA-Briefcase 长周期知识工作 1393 Elo。
- **多模态**：Dense 200 视觉定位 **42% 对 41%** 险胜闭源 GPT-6-Astra，「开源超闭源旗舰」具标志性意义。
- **科学与数学**：SciCode-Verified 开源 SOTA，可一次性生成完整 Hartree–Fock 模拟；内部 STEM 人工评估优于 GLM-5.3。

## 知识工作与安全性

- vals.ai 第三方评测：法律、金融任务双双超过 GPT-6-Astra，HarveyAI 法律 Agent 基准开源第一；
- 安全性不降反升：Lakera B3 基准抵御 **93.3%** 攻击，KORA 得分 1.691/2（开源最高），恶意网络请求平均拒答率开源模型中最高——能打的矛配了坚固的盾。

## 幕后：工业化 RL 流水线

能力跃升的核心是大规模强化学习，三个关键设计：

- **可组合环境接口**：单次训练混合对话、科学解题、安全对齐、长周期工具调用等多类任务；
- **可组合验证器**：奖励模型、单元测试、LLM 裁判按需组合；
- **异步大规模 rollout**：3000 卡集群日产约 **330 亿 token**（约 160 亿可训练），支持百万 token 级长轨迹，且 reward 曲线未见饱和。

本质是把 RL 从「一任务一工程」的手工作坊推向工业化流水线，同时喂饱编程、安全、数学多条能力线。

## 价格与展望

输入 $1.36 / 输出 $4.18 每百万 token。ML4 只是 Mistral 30 亿欧元 D 轮路线图的第一站，权重、架构细节和后训练方法学本月底陆续放出；唯一悬念是公测表现能否经得起社区独立复测。

> 本文参考自 [Introducing Mistral Large 4 | Mistral](https://mistral.ai/news/mistral-large-4/)