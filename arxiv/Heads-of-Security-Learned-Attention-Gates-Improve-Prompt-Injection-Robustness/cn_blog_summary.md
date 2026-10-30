# Heads of Security：给注意力头装安全门

AI Agent 读取网页、邮件和工具结果时，可能把其中夹带的恶意指令当成用户命令，这直接威胁真实任务安全。Heads of Security（HoS）不大改模型，而是为每个 Attention Head 学习按信任来源调节的“安全门”：黑盒 RL-Hammer 对 GPT-OSS-20B 和 SecAlign 的最高攻击成功率分别为 97% 和 85%，对 HoS 仅 10%；定向白盒攻击达到同等成功率约需 3 倍计算量。

## 方法：让数据可读，但不可服从

HoS 执行的是**指令—数据分离**：可信指令控制行为，不可信内容可被分析，却不能直接取得指令权。信任标签由外部系统提供，而非模型自行判断。

- 将 Token 分为 Trusted（可信指令与模型生成内容）、Untrusted（工具输出等）和 Template（角色标记等）三类通道。
- 为每层、每个 Head 学习通道间门控，在 Softmax 前向注意力 logit 加入 $\log\sigma(g)$，等价于将原始注意力权重乘上 0 到 1 的系数。
- 冻结全部基座权重，沿用 SecAlign++ 的偏好数据与 DPO 目标，只训练门控参数：GPT-OSS-20B 为 18,432 个，Llama-3.3-70B 为 46,080 个。

关键并非一律屏蔽不可信 Token，而是让负责指令遵循的 Head 减少服从，让负责内容理解的 Head 保留信息。统一压低所有 Head 的不可信注意力虽能降低 ASR，效用却下降更快。

## 结果：更难攻击，也更好保留能力

- **黑盒自适应攻击**：Llama-3.3-70B、SecAlign、HoS 的最高 ASR 分别为 99%、82%、1%；GPT-OSS 对应为 97%、85%、10%。
- **白盒定向攻击**：达到 50% ASR 约需基座或 SecAlign 的 3 倍 FLOPs。但这只是提高成本，并非免疫；在完整 $10^{17}$ FLOPs 预算下，HoS 在 InjecAgent 的累计 ASR 仍达 90%。
- **正常效用**：HoS 在 Few-shot、多轮及多数 Agentic 基准上贴近基座。Llama 的 1-shot GSM8K 中，SecAlign 仅 2.0%，HoS 为 93.2%，基座为 93.7%。不过，HoS 在 AgentDyn 上仍有明显效用损失。

机制热图显示，多数 Head 转向可信内容，少数 Head 反而增强对不可信信息的关注，可能用于识别注入。Role Probe 也发现，注入 Token 更不容易被内部表示当成用户指令，但这主要仍是相关性证据。

## 结论：安全门是原语，不是完整防线

- 逐 Head、逐通道路由兼顾安全与效用，且训练参数极少。
- 静态攻击的低 ASR 不足以证明安全，必须评估自适应攻击。
- HoS 不决定信任边界，不判断用户授权，也不保证抵御未来更强攻击。
- 它依赖标准 Dense Attention，不能原样用于 Linear Attention。

未来需要把自动、可审计的信任标注与指令层级、授权判断及系统级工具策略组合起来。

> 本文参考自 [Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness](http://arxiv.org/abs/2610.37657v1)