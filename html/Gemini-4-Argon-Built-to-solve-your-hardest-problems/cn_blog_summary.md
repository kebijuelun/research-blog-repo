# Gemini 4 Argon：专攻复杂长任务

在约 4000 万行 Chromium 代码中，把一个约 600 万行的 legacy 功能目录与核心网页处理逻辑拆开，难点不只是会写代码，而是要在数百个文件间保持架构一致。Google 表示，Chrome 团队借助 Gemini 4 Argon 把这项原本需要数月的工作压缩到数周。2026 年 10 月 13 日，Google 全球推出 Argon，定位为面向编码、金融、法律、网络安全和科学等复杂现实任务的 frontier 模型。这次发布的主线不是“聊天更聪明”，而是模型能否在长任务链上产出可验证的结果；比 Vals Index 第一的名次更值得细看的，是它在真实代码库和实验室里的交付记录。

## 工程能力先看交付结果

按 Google 统计，过去几周 Antigravity 中的 Argon 帮助团队将 Agent 代码提交量提升 35% 以上，同时将代码回滚减少近三分之一——两个指标放在一起才说明改动更稳定。Google 开源 ADK 的优化更能说明 Agent 的工作方式：团队只提供 `.agents/skills` 规则手册和一段目标描述，Argon 在 Antigravity 中自主工作一夜，成果已进入 `google/adk-python`：长会话内存准备时间缩短 40%，冷启动导入提速 21%。同样的优化能力还用于量子化学软件 xTB（分子结构预测最高提速至原来的 2.3 倍）和 Gemma 4 E2B 推理（A100/vLLM 下吞吐量从约 37,900 Token/s 提高到超过 108,000 Token/s，保持 100% Token parity）；不过这些数字对应指定硬件和推理栈，不能外推到其他部署环境。

## 科研与安全：证据强度不一

科研案例中，Argon 编写的 `arcsin` 量子子程序经形式化证明工具 Lean 验证，时空成本降至已发表 quantum CORDIC 基线的 1/100 以下；在丝氨酸酯酶设计中，它在数小时内完成数据汇总和模型调优，提出五个全新设计，并经湿实验确认具有稳健催化活性。罕见病方向上，Gamow Labs 私有评测中 Argon 在 RareBench 得 57%、LabBench 得 52.2%，均为该组最高——但私有评测不等于公开临床验证。

安全方面，Google 列出了可检查的指标：外部生物安全红队仅能解锁不到 1% 的受测生物子任务；编码基准上捷径尝试率 2%，为 Gemini 系列最佳；Gray Swan 间接提示注入基准中，15 次自适应攻击下成功率为 0.7%。这些数字比“更安全”的笼统说法更可查，但 0.7% 不是零，不等于绝对安全。

## 价格与使用入口

Gemini API 为 Argon 提供 100 万 Token 的统一输入—输出限额；发布期价格为每百万输入 Token 2 美元、输出 10 美元，缓存输入享 95% 折扣。开发者可在 AI Studio 和 Google Antigravity 中使用，Google 还将向 10,000 名科学家免费开放 Antigravity 最高档 6 个月。

## 这次发布真正证明了什么

回到开头的 Chromium 重构，Argon 最有说服力的证据不是榜单名次，而是多个任务已落到可检查的结果上：进入仓库的性能优化、经 Lean 验证的量子子程序、通过湿实验的酶设计。但现有材料主要来自 Google、早期伙伴、内部测试和私有评测，许多案例缺少独立复现、完整基线和长期成功率。更稳妥的做法是先选一个边界清楚、能自动测试且可回滚的任务，记录完成时间、质量、回滚率和 Token 成本，并保留关键步骤的人工确认——这样才能判断它在你的工作流里究竟减少了多少真实成本，而不是把一次精彩演示误认为普遍可靠性。

> 本文参考自 [Gemini 4 Argon: Built to solve your hardest problems](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-argon/)