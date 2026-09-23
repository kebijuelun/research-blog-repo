# 有品味的 Agent：度量并训练长程任务中的「决策品味」

> 论文：The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks（Microsoft & CityU 等，2026 年 9 月）

LLM Agent 处理长程任务时，大量决策的代价要等大半个预算花完才显现——错误决策在当下往往看起来很合理。这篇论文把这种「做好长程决策的能力」称为 Agent 的**品味（taste）**，并回答了两个问题：能否不依赖专家标注自动度量品味？品味能否被训练？答案是都能。

## 核心方法：用「事后之明」给判断力打标签

关键观察是：**轨迹的后半段本身就是前半段决策的事后证据**。Agent 系统常对同一任务做多次尝试，这些尝试会在中途分岔走向不同方向——分岔后各分支的结果天然构成一组已标注的对比。作者把这种分岔点称为**决策岔口（decision fork）**：把轨迹冻结在岔口处，隐藏后续一切，让模型在两个候选方向中选一个，选中被事后证据支持的方向即为「有品味」。

## Taste-Bench：502 道免人工标注的品味题

作者用两种互补构造挖掘岔口：**parallel 轨迹**对比同一任务两条独立尝试的分支（捕捉 Agent 从未察觉的错误）；**detour 轨迹**对比单条轨迹中「被放弃的方向」与「自救方向」（考察能否比当事 Agent 更早识别失败）。数据来自 2,677 条 SWE-bench Pro rollout 和 1,132 条 METR MALT 研究轨迹。生成器提出 4,657 个候选岔口，经 trivial 过滤（仅看候选措辞就能答对的丢弃）与 undecidable 过滤（评委团不认同标签的丢弃），最终 502 道题入选，总通过率仅 10.8%。人工复核显示标签与人工判断一致率达 **98.8%**（Cohen's κ = 0.973）。为应对位置偏置，每题以正反两种顺序各测一次，**两种顺序都答对才算对**（随机基线 25%）。

## 实验发现

对 14 个前沿模型的评测得出三个结论：

1. **前沿模型品味有限**：最好的 GPT-5.6 Sol 仅 59.7%（随机猜测 25%）；detour 岔口比 parallel 更难。
2. **错误集中在长时间地平线岔口**：按决定性证据出现的远近分四级，模型均值从 in-prefix 级的 62.3% 一路跌到 more-work 级的 21.0%——已低于随机。
3. **加大推理预算无效**：三档 reasoning effort 下准确率变化仅 -0.2 ~ +2.2 个百分点。模型认得出哪些岔口难、也在那里想得最久，但决定性证据只存在于后续工作中，多想无益。

此外，Taste-Bench 与 SWE-bench Verified 仅部分相关（r = +0.63，engineering 子集仅 +0.37），说明品味不是端到端能力的复述。

## 品味可以蒸馏

以 Qwen3.6-27B 为 base，用 privileged teacher 自蒸馏（teacher 多看受支持候选的 demonstration，student 蒸其完整推理而非拟合标签，LoRA rank 16，单张 A100 每折约 2 小时），在任务不相交的 held-out 折上：双顺序 accuracy 从 30.0% 升至 **47.9%**（+17.9 点），追平 GLM-5.2、挤进前沿梯队。端到端验证中，把 student 在岔口的判断写成 advice 注入任务上下文，executor 在 41 个 held-out SWE-bench Pro 任务上的成功率从 14.6% 升至 **33.7%**（全对 advice 上限为 39.0%）。

这篇工作的双重启发：「会做事」和「会选路」是两种能力，端到端榜单只反映前者；而 Agent 系统每天产生的海量轨迹本身就是免费的品味标注金矿。

数据集与代码均已开源：[Hugging Face 数据集](https://huggingface.co/datasets/wenbopan/taste-bench)、[GitHub 代码库](https://github.com/wbopan/tastebench)。

> 本文参考自 [The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks](https://arxiv.org/abs/2609.25804)