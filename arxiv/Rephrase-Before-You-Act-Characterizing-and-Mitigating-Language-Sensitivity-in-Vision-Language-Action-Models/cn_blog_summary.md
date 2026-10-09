# Rephrase Before You Act：换句话机器人就失灵？十几条规则救回 VLA

Vision-Language-Action 模型（VLA）本该继承 VLM 的语言鲁棒性，实测却恰恰相反：同一指令换个说法，成功率能差出几十个百分点。这篇论文先用统计检验把"语言敏感性"量化清楚，再提出不动模型权重的解法——让 LLM 从真实 rollout 证据中蒸馏出 10–20 条措辞规则，部署时把用户指令改写成 VLA 听得懂的说法。结果：冻结的 $\pi_0$ 在 12 个从未见过的封存任务上相对提升 16%–27%；$\pi_{0.5}$ 在 LIBERO 上 in-finetune 成功率从 93.6% 升至 97.8%。全程零重训练。

## 问题有多严重

- 对 $\pi_{0.5}$ 说 "switch on the stove" 成功率 100%，换成 "switch on the hot plate" 只剩 2%。
- 对 $\pi_0$ 说 "purple eggplant goes on the sponge" 成功率 69%，删掉 "purple" 一词跌到 8%。
- 即使训练时已做措辞增强微调，这些"翻车"依然存在——训练侧解法治标后，残余敏感性没人管。

## 刻画：所谓泛化差距，多半是措辞差距

作者用双比例 z 检验（$p<0.05$）筛选"单次编辑摇摆"，并定义 oracle headroom 度量纯措辞可榨出的性能上限。关键发现：

- 摇摆幅度惊人：单词编辑最高造成 61 个百分点差异；"fire up the stove"（100%）与 "turn on the stove"（6%）差 94 点。
- 规律互相矛盾（加 "purple" 在一个任务 +61、另一个 −30），说明规则必须来自证据而非拍脑袋。
- Oracle 措辞几乎抹平了 $\pi_0$ 上 in/out-of-distribution 之间 21 个点的差距（缩到 3 点）——泛化差的根源是语言接口脆弱，而非任务难。

## 方法：三段式规则蒸馏

1. **证据收集**：8 个可仿真任务 + 208 个真机轨迹任务，共 5,453 个"任务-短语"打分对；不可仿真任务用动作误差代理分数（与成功率相关 $r=0.54$）。
2. **规则蒸馏**：多智能体面板（3 蒸馏器 + 1 评论者 + 1 综合者）一次性离线产出 10–20 条规则，如 "'coke' stays 'coke', never 'cola'"。
3. **规则应用**：部署时每 episode 只改写一次（trace 生成约 2.5 秒，本地 9B 模型应用仅 1–2 秒），成本远低于每步验证。

## 实验亮点

- 三种 applier（Claude、Gemini、开源 Qwen 9B）在三类短语集上全部显著为正，提升 16%–27%；扣除"改写本身"的收益后，蒸馏规则仍带来最高约 +29% 的额外贡献。
- 对比 test-time verification 方法 CoVer：每步花 40 倍推理成本做验证，反而落后于什么都不做的基线。
- LIBERO 收益可逐条归因：hot plate→stove（+45 点）、dish→plate（+24 点）、onto→on（+2.5 点）。

## 结论

VLA 的语言敏感性是系统性的、可被显式规则捕获的；证据蒸馏出的规则能 zero-shot 迁移到未见任务，一次改写胜过逐步验证。最有趣的开放问题是：VLA 为何会丢掉 VLM 的语言鲁棒性？

> 本文参考自 [Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models](http://arxiv.org/abs/2610.10526v1)