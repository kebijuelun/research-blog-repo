# SpecDec++：预测拒绝风险，自适应加速投机解码

SpecDec++ 解决投机解码中固定候选长度无法适应上下文、造成大模型频繁验证或小模型无效生成的问题，让每轮候选长度随拒绝风险动态变化；在 llama-2-chat 7B/70B 上，它相对调优后的固定长度基线最高提升吞吐 **11.1%**，并在 GSM8K 上实现 **2.26 倍加速**。

## 固定候选长度，为什么会拖慢投机解码？

- 候选长度 $K$ 太小，会增加 target model 验证次数；太大，则会生成更多最终被丢弃的候选。最优 $K$ 随上下文变化，常规文本容易预测，代码分叉和数学计算更容易产生分歧。
- 推理时间可分解为理想生成成本、丢弃候选成本，以及 target 相对 draft 的额外验证成本。因此，自适应长度需要同时减少浪费与验证。
- Alpaca 上的 Oracle 预实验证明了收益空间：理想自适应策略达到 **27.06 tokens/s、2.92 倍加速**，最佳固定 $K$ 仅为 17.58 tokens/s、1.90 倍加速。

## 把“何时停止起草”建模为 MDP

SpecDec++ 以当前前缀和未验证候选为状态，每生成一个候选就选择 `continue` 或 `stop`，并将动作成本设计为沿轨迹累加后等于真实推理时间。理论给出了停止的充分条件：当“已有候选中至少一个会被拒绝”的条件概率超过与 draft/target 时间成本及常数 $\Delta$ 有关的阈值时，停止的期望总成本不高于继续。$\Delta$ 难以直接估计，实际系统仍在验证集上搜索阈值 $h$。

## 轻量预测头与训练方法

- 在 draft model 最后一层 hidden embedding 上增加 acceptance prediction head，输出条件接受概率 $\hat P_i$；累计拒绝风险为 $1-\prod_{i=1}^{k}\hat P_i$，超过 $h$ 就停止起草并验证。
- 预测头学习的是 $\min(1,p/q)$ 所体现的**两个模型的对齐程度**，而非 draft 单模型的 entropy 或 confidence。它不改 target model，也不改标准 rejection sampling 验证规则。
- 训练使用连续接受概率标签；Token Mixing 仅保留 15% target token，其余替换为 draft candidate，提高单次 forward 的监督密度；加权 BCE 则缓解高接受率导致的类别不平衡，拒绝权重需与停止阈值配合调节。

## 实验：收益稳定，但仍受模型对齐约束

Llama 7B/70B 实验中，HumanEval 和 GSM8K 均为未参与预测头训练的 OOD 测试。相对最佳固定长度 SpecDec，SpecDec++ 在 Alpaca、HumanEval、GSM8K 上分别达到 **18.88、20.61、20.95 tokens/s**，提升 **7.2%、11.1%、9.4%**，对应普通 target model 的 **2.04、2.23、2.26 倍加速**；三个数据集上的验证率—丢弃率 Pareto 前沿均优于基线。固定配置 $w_{\mathrm{rej}}=6,D=3,h=0.7$ 也能达到各自最佳吞吐的 **99.3% 以上**。

Gemma 2B/27B 上最高相对提升为 **12.4%**，但 Alpaca 仅提升 1.4%，说明收益依赖模型对的对齐质量。方法仍需离线搜索阈值，且在 PagedAttention、KV cache pre-allocation、大 batch 和长上下文场景下的表现尚待验证。

> 本文参考自 [SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715)