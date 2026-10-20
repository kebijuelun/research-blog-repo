# LLM 综述精华：一篇读懂大模型的前世今生

ChatGPT 之后，LLM 论文从每天 0.40 篇暴涨到 8.58 篇，从业者已难跟上。人大这篇《A Survey of Large Language Models》（arXiv 2303.18223，持续更新至 v18）把 LLM 技术体系拆解为预训练、适配、使用、能力评估四大板块，给出了一张完整的知识地图。最亮的两个结论：Chinchilla 用 70B 参数反超 280B 的 Gopher；GPT-4 在 MMLU 上达 86.4%（5-shot）。

## 规模为什么有效

- **Scaling law**：KM law 主张把算力更多分给模型规模，Chinchilla law 主张模型与数据等比扩张——实践站在后者一边（70B + 1.4T tokens 胜过 280B）。
- **涌现能力**：小模型不存在、大模型才出现的能力，类似相变。三类典型：ICL（175B GPT-3 具备，GPT-2 没有）、指令跟随（LaMDA-PT 需 68B、PaLM 需 62B 才涌现）、CoT（>60B 才有收益，>100B 优势明显）。
- LLM 与 PLM 的区别不只是"更大"：能力质变、使用方式从微调变为 prompting、研究与工程边界消失。

## 预训练三件套

- **数据**：通用语料以网页为主体，专用语料（代码、科学、多语言）定向增强；预处理五步流水线（过滤、去重、隐私削减、分词）；数据调度分配比与课程两维（如 CodeLLaMA 的"2T 通用 → 500B 代码"）。
- **架构**：Causal Decoder 胜出，工程标配是 pre RMSNorm + SwiGLU + RoPE/ALiBi，推理由 GQA、FlashAttention、PagedAttention 加速。
- **训练**：3D 并行 + ZeRO/FSDP + BF16 混合精度；BLOOM 用 8 路数据并行 + 4 路张量并行 + 12 路流水线并行在 384 块 A100 上完成训练。

## 适配：解锁与对齐

- **指令微调**：instruction 本身是任务泛化的核心。综述实证发现：混合多源指令全面提升（MMLU 43.69 vs 单源 38.58），而单纯扩大指令数量反而有害（BBH 从 29.81 降到 26.63）——质量胜过数量。
- **对齐**：3H 准则（helpful、honest、harmless）；RLHF 三步走（SFT → 奖励模型 → PPO），LLaMA 2 迭代了五版；DPO 省略显式奖励建模成为热门替代。SFT 解决"会做"，RLHF 解决"做好"。
- **高效适配**：LoRA 以低秩分解近似权重增量最流行；量化实验显示 4-bit 权重量化对 ICL、CoT 等涌现能力几乎无损，显存降至约 1/3，QLoRA 可达全 16-bit 微调性能。

## 使用与评估

- **Prompt 是实证工程**：精心设计的 prompt 全面提升表现，越复杂的任务获益越大（Colored Objects 从 53.20 到 66.75）；配好 prompt 的 ChatGPT 在 GSM8k、HumanEval 上甚至超过监督 SOTA。
- **ICL 双机制**：task recognition 350M 小模型就有，task learning 需 66B 才涌现；CoT 需 10B 以上才有效，能力来源归因于代码训练；Planning 以 ReAct、ToT、Voyager skill library 为代表攻克复杂任务。
- **评估**：MMLU、BIG-bench/BBH（CoT 可超人类平均）、HELM 为综合基准；幻觉与知识时效性是根本短板；GPT-4 当裁判需警惕 position、verbosity、self-enhancement 三种偏差。横评显示闭源模型碾压开源，指令微调普遍提升能力但可能损害纯语言建模。

## 前沿方向

- **长上下文**：RoPE 基底修改与 StreamingLLM；反直觉发现——数十亿 token 长文本续训即可让 LLaMA-2 获得超 100K 窗口。
- **Agent 与系统优化**：推理的 prefill 阶段 compute-bound、decoding 阶段 memory-bound（算术强度低于 8），投机解码可无损加速 2~3 倍。
- **长 CoT**：o1/R1 开启"慢思考"时代，规模化 RL + 可验证奖励是核心引擎；仅 3.9K 条蒸馏数据微调 Qwen2.5-32B 即可达 o1-preview 可比数学水平，背后是 test-time scaling law。

一句话总结：规模 + 数据 + 对齐 + 推理这条技术路线，还远未到头。

> 本文参考自 [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)