# Mistral 发布 1 万亿参数「Le Chonk」：欧洲最强开源权重模型

欧洲终于有了一个能站上全球第一梯队的开源权重模型。2026 年 10 月 6 日，Mistral AI 发布旗舰模型 **Mistral Large 4（昵称 Le Chonk，"大块头"）**，总参数 1 万亿，官方称其为**美国和欧洲范围内基准综合成绩最强的开源权重模型**，并在 Harvey's Legal Agent 法律基准上以 **15% 夺冠**，领先 GPT-6 Astra（5%）整整 10 个百分点。

## 核心规格：稀疏架构用效率换规模

- **1T 总参数 / 49B 激活**：采用类似 MoE 的稀疏激活设计，把"记住多少知识"和"每次算多少"解耦——拥有万亿级知识容量，只付 49B 的推理账单；
- **原生多模态**：支持文本和图像输入；
- **512k 上下文**：明确瞄准复杂 Agent 场景（大量工具调用历史、文档、代码库）；
- **行业定位**：网络防御、制造业、金融等关键负载上的 SOTA。

这一设计对算力预算远不如美国巨头的 Mistral 尤为关键：稠密模型参数翻倍成本就翻倍，而稀疏架构是用架构效率换规模竞争力的典型打法。

## 跑分实测：强势，但要看清口径

- **Artificial Analysis Intelligence Index v4.3.2**：Mistral Large 4 Preview 得 **38 分，全球第六**。前四名 MiMo-V2.6-Pro（46）、GLM-5.3 max（45）、Kimi K3 max（44）等均被中国开源模型包揽——官方的"美欧第一"口径没撒谎，但定语加得很讲究。相比自家上一代 Medium 3.5 的 14 分，提升幅度夸张；
- **Cyber Index（网络安全指数）50 分**，呼应其网络防御定位；
- **Harvey's Legal Agent 法律基准 15% 居首**（Kimi K3 13%、MiMo-V2.6-Pro 11%、DeepSeek-V4-Pro 8%、GLM-5.3 8%）。整体得分偏低说明法律 Agent 任务难度极高，但领先 GPT-6 Astra 十个百分点是本次最有含金量的垂直成绩。

## 算力故事：法国本土 3800 块 GPU 训出来

联合创始人 Guillaume Lample 透露，模型在巴黎郊区 **Bruyères-le-Châtel 自有集群用 3800 块 NVIDIA Grace Blackwell GPU** 完成训练——放在美国巨头动辄十万卡的背景下相当克制，恰恰证明稀疏架构 + 高效训练管线的路线成立。CEO Arthur Mensch 强调训练和服务全栈自主，且 **"RL shows no sign of saturation"**——强化学习收益尚未饱和，能力还不是终点。

## 开放策略与社区反应

- **API 已上线**，开源权重承诺 **10 月底放出**；同期还发布了面向 Agent 和 Coding 的 Medium 3.5 Dense 模型，形成"超大稀疏旗舰 + 中型稠密主力"矩阵；
- 社区最出圈的调侃：「Mistral Large 4 唯一真正领先的基准居然跟监管/法律有关，不愧是欧盟出品」（99 赞）——玩笑归玩笑，15% vs 5% 的差距是实打实的。

## 结语

Le Chonk 证明了中等算力预算下欧洲团队依然能做出全球第一梯队的开源权重模型。接下来值得观察：10 月底权重放出后的实际部署表现，以及"RL 不饱和"能否兑现为持续涨分——毕竟它前面还站着四座中国开源模型的大山。

> 本文参考自 [Mistral Unveils Le Chonk: Europe's Top Open-Weights AI Model](https://x.com/i/trending/2107458174307455069)