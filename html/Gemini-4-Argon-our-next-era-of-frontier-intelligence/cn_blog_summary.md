# Gemini 4 Argon 发布：100 万 Token 输出，把"深度思考"拉满

Google 新一代前沿模型 Gemini 4 Argon 解决的核心问题是：让大模型在真实软件工程、金融、法律、网络安全等**长程复杂工作流**中持续深度推理，而不是聊几轮就"断片"。做法很直接——输出 Token 上限从 64K 一口气拉到**行业最高的 100 万**，让模型单条轨迹就能生成数十万 Token 的思考。最亮眼的成绩：真实软件工程基准 DeepSWE v1.1 达 **77.9%** 刷新 SOTA，漏洞修复基准 CWE-bench v1 以 **68%** 并列第一。

## 已在 Google 内部"上岗"

- **量子算法优化**：几分钟内将已发表基线的时空资源消耗改进 **40%**。
- **内存优化**：Agent 自主分析集群遥测数据，已释放超 **300 TiB** 内存，预计总节省 500 TiB–1 PiB。
- **代码迁移**：正把 C/C++ 代码库迁往 Rust，规模已达 **80 万+ 行的 Fuchsia Zircon 内核**。典型案例 libgav1：用安全 Rust 替换 3.2 万行手写 SIMD，靠编译器自动向量化，最终比 Rust 移植版**快 2.7 倍**，性能逼近 C++ 原版。

## 基准全线领先

- **DeepSWE v1.1**（真实长程软件工程）：**77.9%** SOTA。
- **Vals Index**（按 GDP 加权的金融/编码/法律/税务综合能力）：总分第一。
- **AutomationBench**（端到端业务流程执行）：**51.3%** 排名第一。
- **LVBench**（长视频理解）：**91.7%** SOTA。

## 网络安全：主攻方向

Argon 被专门训练为可**自主发现、验证并修补**漏洞，并对可信防御者**不带网络护栏**发布。实战中，云安全公司 Wiz 用它在全球医院使用的医疗软件中挖出其他前沿模型都没发现的严重漏洞。黑盒渗透测试中，它在攻击面发现、漏洞识别、PoC 验证三个环节全面超越前代。

## 安全护栏与发布节奏

- 按 Frontier Safety Framework 防网络与 CBRN 滥用，并加强**监测模型内部激活**识别滥用。
- 防 Prompt Injection：Gray Swan 间接注入基准鲁棒性领先。
- 对齐监控：监控 Chain-of-Thought，越界即中止；刻意**不把监控发现回灌训练**，避免模型学会规避监控。
- 分阶段发布：先 Fairwind 可信防御者计划，再推向开发者、企业与消费者。

定价：输入 $2 / 输出 $10 每百万 Token，缓存输入享 95% 折扣。Argon 的信号很明确：前沿竞争正从"单次回答质量"转向"长时自主工作能力"。

> 本文参考自 [Gemini 4 Argon: our next era of frontier intelligence](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/)