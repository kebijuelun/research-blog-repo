# AKV：让 Agent 直接管理 KV Cache

长轨迹 Agent 每多进行一轮推理和工具调用，KV Cache 都会继续膨胀，既推高存储与 I/O 成本，也可能超出模型能够可靠处理的上下文和位置范围。论文在 BCP 成本拆解中发现，KV Cache 读取占总 API 成本的 82.4%，因此“如何缩减缓存”已经成为长程 Agent 落地的核心问题。AKV 的思路是把控制粒度从文本下沉到 KV 状态：不改写轨迹，只用 $\operatorname{drop}$ 删除未来 Attention 可见的缓存，用 $\operatorname{repos}$ 调整保留 Key 的位置编码，并配套可复用、可恢复的服务系统。相比阈值触发摘要，简单的工具响应滚动丢弃策略将平均任务得分提高 1.7 个百分点、平均成本降低 27.7%；AKV 还实现了 1.18× 输出吞吐和 1.51× P95 请求延迟加速。

## 一、问题：长轨迹 Agent 被 KV Cache 拖住

Agent 通常由负责推理和工具调用的 LLM，以及负责执行工具、组织上下文的 harness 组成。两者交替工作数百轮后，交互轨迹会越来越长，KV Cache 也随之增长。

这里的压力不只是“显存装不下”。每轮 Decode 都要读取历史 KV，存储和 I/O 会持续转化为延迟与费用；当参与 Attention 的状态过多，或位置编码跨度超出模型的可靠范围时，任务准确率也可能下降。

### Context Engineering 的两难

已有的 Context Engineering（上下文工程）主要改写文本轨迹：

- **摘要**：用模型压缩早期对话，但要额外生成摘要，也可能丢掉后续任务需要的细节。
- **裁剪**：直接删除低相关消息，省去摘要过程，却仍可能损失信息。
- **延迟压缩**：等到上下文接近阈值再大幅缩减，以减少压缩次数，但会造成信息突然断层。

更关键的是，只要修改了较早的文本前缀，后续 KV 状态就不再与新轨迹严格对应，保留的后缀也要重新 Prefill。修改越早，需要重算的后缀越长；为了少重算而少保留，又可能伤害准确率。这就是论文要打破的“重算成本—信息保留”两难。

![全上下文推理、上下文工程与缓存工程的对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/scheme.png)

> 图解：Full-context 保留所有文本和 KV，无需重算但历史负担持续增加；Context Engineering 删除灰色无效历史后，需要重新计算绿色保留历史的 KV；Cache Engineering 保留完整文本，只让灰色 KV 退出后续 Attention，绿色 KV 直接复用。绿色块中嵌套的灰色表示：保留状态在早期计算时已经吸收过被删历史的信息，因此丢弃原始 KV 不等于抹去它曾经产生的影响。

### 已有缓存路线还缺什么

缓存层面的已有工作大致分为三类。压缩方法用低秩、量化或聚合减少每个条目的存储；驱逐方法根据 Token 位置或 Attention 重要性删除条目；复用方法通过位置调整或部分重算，把旧状态适配到新前缀。服务系统则用 Prefix Cache、调度和批处理提高共享效率。

AKV 的差异不在于再发明一种 Token 重要性分数，而是把“何时删、删什么、如何重排位置”变成 Agent 和 harness 可编程的高层操作，同时解决跨请求复用、隔离与恢复。普通 Prefix Cache 只匹配 Token 前缀，而 AKV 还必须记录缓存操作历史；这是后文系统设计的核心。

理解了问题所在，下一步就是把“直接改缓存”变成一套定义清晰、可执行的机制。

## 二、方法：Cache Engineering 把文本与缓存解耦

### 2.1 逻辑轨迹不等于 Active Context

自回归 LLM 根据前序 Token 预测下一个 Token：

$$
p_\theta(x_{t+1}\mid x_{1:t})
$$

KV Cache 保存历史 Token 的 Key-Value 状态，使模型不必每轮重算整段前缀。使用缓存后，预测条件写成当前 Token 与历史缓存：

$$
p_\theta(x_{t+1}\mid x_t,\mathcal{C}_{1:t-1})
$$

其中，$\mathcal{C}_{1:t-1}$ 是由 $x_{1:t-1}$ 计算出的 KV 状态。

论文进一步区分三个容易混淆的概念：

- **逻辑轨迹**：按顺序排列的完整文本历史，包括 Prompt、Assistant 消息和工具响应，记为 $x_{1:t}$。
- **Active Cache**：后续 Attention 实际可见的 KV 状态集合 $\mathcal{C}$，不要求覆盖完整逻辑轨迹。
- **Active Context**：Active Cache 对应的 Token 序列，其长度可以远小于逻辑 Token 长度。

这里的 Active 是算法层面的“是否参与 Attention”，不是物理显存中“是否还驻留”。文本逻辑索引、缓存条目数量和 RoPE 位置编码索引也可以彼此解耦。

Context Engineering 修改的是逻辑轨迹；Cache Engineering 保持轨迹不变，只修改 Active Cache。可以把它理解为“不重写会议记录，只控制下一轮讨论时哪些历史材料仍摆在桌上”。

### 2.2 为什么直接删 KV，为什么优先删旧工具响应

论文在 GPT-OSS-120B、BCP 前 100 个任务上做了每次最多 100 轮、每任务 4 次试验的比较。只保留最近 12 轮工具响应的文本裁剪，成本反而是全上下文基线的 1.85×；48K 阈值摘要虽降低约 7% 成本，却损失了 5 个百分点准确率。直接丢弃旧工具响应 KV，则同时改善了成本与准确率。

![BCP 上准确率与单任务成本的 Pareto 对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/pareto_frontier.png)

> 图解：横轴是单任务成本，单位为美元；纵轴是准确率，误差棒表示均值和标准差。Text Engineering 曲线上的 48K、60K、72K 是摘要阈值，AKV 曲线上的 $K=1,4,12$ 是保留的工具响应轮数。越靠左上方越理想；AKV 各组点整体位于左上方，说明保留缓存并滚动丢弃旧工具响应，比触发式摘要获得更好的准确率—成本折中。

选择删除对象时，论文没有只看 Token 级重要性，而是同时考虑消息角色和新旧程度。对 10 个 BCP 请求的 Attention 统计显示：越旧的内容获得的 Attention 通常越少；在相近相对位置，工具响应获得的 Attention 也低于 Assistant 消息。

![不同相对位置与消息类型的 Attention 曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/bcp_attention_distance_request37_mean10.png)

> 图解：横轴是历史中的相对 Token 位置，左侧更旧、右侧更近；纵轴是每个 Token 获得的 Attention，采用对数刻度。紫线对应 Assistant Response，蓝线对应 Tool Response，结果是 10 个 BCP 请求的平均值。两条线总体都向右上升，且紫线大部分高于蓝线，为“保留推理和近期工具响应，优先删除旧工具响应”提供了经验依据。

因此，论文的主力策略是 **Tool-response Rolling Drop**：保留全部推理 KV 和最近 $K$ 轮工具响应 KV，持续滚动删除更老的工具响应。主实验取 $K=12$。笔者认为，这个设计的聪明之处在于，它把历史信息分成“原始观察”和“已经吸收观察的推理状态”：前者可以较早退出 Attention，后者继续保留任务线索。

### 2.3 三个操作：Append、Drop、Repos

Cache Engineering 用三个基本操作控制 Active Cache，而不重写已有逻辑轨迹。

**Append：追加新 Token 的缓存。** 这是标准 LLM 推理已有的操作。新 Token 在当前 Active Cache 条件下计算，再把新条目接到缓存末尾：

$$
\operatorname{append}(x_{t+1}):\qquad
\mathcal{C}\leftarrow[\mathcal{C};\,C_{t+1}]
$$

$[\cdot;\cdot]$ 表示拼接。新 Token 可以来自 Prompt、工具响应或模型生成；位置编码默认从 0 开始，每次 Append 后前进一位。

**Drop：让选中缓存退出后续 Attention。** 设 $P$ 是要删除的活跃逻辑 Token 索引集合，则：

$$
\operatorname{drop}(P):\qquad
\mathcal{C}\leftarrow\mathcal{C}\setminus\{C_i:i\in P\}
$$

保留条目的内容、顺序、位置编码和完整逻辑文本都不变，后续 Token 也不再看到这些被删状态。Drop 本身不会压缩位置编号，也不会重置下一个 Append 的位置。

**Repos：重排保留缓存的位置编码。** 如果剩余 $r$ 个条目，Repos 按逻辑顺序把其位置改为 $0,\ldots,r-1$，下一个 Append 使用位置 $r$。对 RoPE 而言，把缓存 Key 从位置 $p$ 移到 $q$，只需施加相对旋转：

$$
K\leftarrow R(q)R(p)^{-1}K
$$

$R(p)$ 是位置 $p$ 的旋转矩阵，Value 不变，也不需要重新计算 Hidden State。需要注意，重排后的缓存并不等价于把缩短后的文本重新完整 Prefill 一遍；它保留的是原计算历史，只改变位置表示。

### 2.4 相同文本前缀，不一定有相同 KV

三个操作交错形成缓存操作序列：

$$
o_{1:m}=(o_1,\ldots,o_m),\qquad
o_j\in\{\operatorname{append}(x),\operatorname{drop}(P),\operatorname{repos}\}
$$

对固定模型，从空缓存出发，这个序列决定当前状态。仅知道“剩哪些 Token、最后在哪些位置”并不够，因为 Drop 或 Repos 发生的时机，会影响后续 Token 最初计算时能看到什么。

![不同缓存操作轨迹导致不同 KV 状态](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/trajectory.png)

> 图解：（a）轨迹 ①先计算 $A,B,C$ 再删除 $A$；轨迹 ②先计算 $A,B$，删除 $A$ 后再计算 $C$；轨迹 ③从没有前缀的 $B$ 开始，再计算 $C$。三者最终都保留 $B,C$。（b）①和②的 $B$ 相同，因为计算时都看过 $A$；但三条轨迹的 $C$ 各不相同。（c）扩展 Radix Tree 在共同 $A\!-\!B$ 前缀后，根据 Drop 时机分支，避免错误复用后续状态。

Repos 也有类似的历史依赖：它不仅改变已有 Key，还改变之后 Append 所用的位置。因此，正确的复用与恢复必须同时依赖 Token 历史和操作历史。这一性质把方法问题自然转化为系统问题。

## 三、系统：AKV 如何正确复用、驱逐与恢复

AKV 构建在 SGLang 之上，遵循“Stateless Interface, Stateful Serving”。客户端完整描述逻辑消息和缓存操作，服务端内部处理物理缓存是否驻留、能否共享、何时驱逐以及如何恢复。

### 3.1 无状态接口：每个请求说清完整历史

AKV 扩展现有 Messages API，增加可选的 `cache` 字段。论文示例可写成：

```json
{
  "model": "gpt-oss-120b",
  "messages": [
    {"role": "user", "content": "Search ICLR"},
    {"role": "assistant", "content": "Searching"},
    {"role": "tool", "content": "ICLR is..."}
  ],
  "cache": {
    "drop": {"1": [0], "2": [1]},
    "reposition": [2]
  }
}
```

`messages` 提供完整逻辑序列，每条消息隐含 Append。`drop` 使用 Operate-after 语法：键表示在哪条消息之后操作，值表示删除哪些消息的 KV；`reposition` 列出在哪些消息边界后执行 Repos。索引从 0 开始，同一边界同时有两种操作时，先 Drop 后 Repos。

上例中，三条消息记为 $m_0,m_1,m_2$：在 $m_1$ 后删除 $m_0$，在 $m_2$ 后删除 $m_1$，随后执行 Repos。多轮 Agent 每轮继续扩展消息与操作历史，不需要知道上一轮物理缓存是否还在服务器上。

这种接口还必须保证请求隔离：一个请求的 Drop 不能改变其他共享前缀请求的输出分布；请求结果也不能依赖某段缓存“恰好还在显存里”。

### 3.2 Drop-aware：会复用，也要会提前释放

AKV 要同时守住三条系统性质：只复用等价状态；驱逐不影响请求隔离；被驱逐状态能在一次 Prefill 中恢复。对应挑战和解法如下：

| 挑战 | 普通系统的不足 | AKV 的设计 |
|---|---|---|
| 正确复用 | Token 前缀无法区分不同操作历史 | 在 Radix Tree 中加入 Drop、Repos 操作节点 |
| 安全驱逐 | 一个请求删除的 KV 可能仍被其他请求使用 | 跟踪所有计算与恢复引用，优先回收已 Drop 缓存 |
| 高效恢复 | 逐段重放操作需要多次 Forward | 用 Segment-batch Attention 在一次 Prefill 中恢复 |

执行 Drop 时，系统在触发位置插入 $\Delta$ 节点，记录被删逻辑索引。Drop 之前的公共前缀仍可共享，Drop 之后因可见历史不同而分支。物理内存释放后，树节点可以保留为空占位符，使后代缓存仍能按路径匹配。

驱逐时，只有不再被任何进行中计算或恢复需要的条目才可回收。已经 Drop 的缓存优先于已完成请求留下的缓存，因为后者更可能在下一轮复用。这样既提前释放空间，又不把“算法上不可见”误当成“所有请求都不再需要”。

### 3.3 Reposition-aware：复制 Value，旋转 Key

Repos 让同一 Token 状态出现多个位置版本。AKV 通过复制缓存、旋转 Key 来复用，而不重算 Hidden State；不同位置版本独立驱逐。

![Repos 前后的 KV 池与 Radix Tree](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/repos_clean.png)

> 图解：（a）请求 1 使用 $A,B,D$；请求 2 共享 $A,B$，在追加 $C$ 前删除 $A$，之后重排 $B,C$。（b）重排前 $B,C$ 的位置分别是 1、2。（c）系统复制 $B,C$，旋转 Key 到位置 0、1，并复制 Value；原始 $B$ 仍服务请求 1，原始 $C$ 则可驱逐。Radix Tree 用位置标签区分版本，并在 $C$ 后记录 Reposition 节点。

当 Token 与操作历史一致、位置也一致时，可以直接复用；位置不同但计算历史兼容时，则通过 Key Rotation 变换后复用。恢复时，每个分段必须使用其原始计算阶段的位置编码，因此还需临时 KV 版本。

### 3.4 Segment-batch Attention：一次 Forward 恢复多段历史

恢复缺失状态时，AKV 在每次 Drop 或 Repos 的触发点切分 Query。每段使用当时可见的 KV 页面和普通因果 Attention，不必构造稠密的自定义 Mask；只有 Attention 分段，FFN 仍对完整序列统一批处理，避免反复读取 FFN 权重。

![Segment-batch Attention 的四段恢复示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/segment_attention.png)

> 图解：上方请求依次追加 $A,B,C$，删除 $B$ 并重排，再追加 $D$；删除 $D$ 后追加 $E$，再次重排并追加 $F$。下方矩阵中，横轴是 Key、纵轴是 Query，彩色格表示允许 Attention，数字是位置。四段分别计算 $A_0,B_1,C_2$；用 $(A_0,C_1,D_2)$ 计算 $D_2$；用 $(A_0,C_1,E_3)$ 计算 $E_3$；再用 $(A_0,C_1,E_2,F_3)$ 计算 $F_3$。所有分段在同一次模型 Forward 中执行。

第二、三段即使部分位置映射相同，也必须使用不同 KV 视图，因为 $D$ 对 $E$ 已不可见，而 Drop 不会自动压缩位置。每段的新 KV 会写回，额外的 $C_1,E_2$ 等位置版本通过重编码 Key、复制 Value 得到。

### 3.5 Exact 与 Retrigger Matching：位置可变，历史不能乱

为了既复用公共计算，又不跨越历史分歧，AKV 先找最长 Exact Prefix，再搜索可通过位置变换复用的 Retrigger Prefix。

树中的 Token 节点写作 $T(x,p)$，$x$ 是 Token ID，$p$ 是当前存入 Key 的位置。Exact Matching 要求 Token、位置和操作节点参数全部一致：

$$
T(x,p)\equiv T(x',p')\iff x=x'\land p=p'
$$

Retrigger Matching 只忽略当前存储的位置索引，保留全部操作记录：

$$
\phi(T(x,p))=T(x),\qquad
\phi(\Delta P)=\Delta P,\qquad
\phi(R)=R
$$

若两个前缀的投影一致，它们就是历史兼容的。系统只在候选有效且匹配严格更长时替换 Exact 结果，同长时保留 Exact；遇到首个 Token 或操作不匹配就停止。选中源缓存后，再复制并做位置变换。

![扩展 Radix Tree 如何记录 Drop 与 Repos 历史](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/reposition.png)

> 图解：（a）三条轨迹交错执行 Append、Drop 和 Reposition，最终保留的 Token 与位置可能相同，但中间可见历史不同；$R_X$ 表示在 $X$ 后重排。（b）树同时记录 Token、当前位置和操作节点。尤其是轨迹 ③在计算 $D$ 前执行 $R_C$，改变了 $D$ 与 $A$ 的相对距离；后续再重排无法抹去 $D$ 已经产生的计算差异，因此 $R_C$ 必须留在路径中。

在模型与位置编码配置固定、精确算术条件下，相同投影前缀保证每个 Token 原始计算前的 Token 与操作历史一致，因此 Hidden State 和未旋转 KV 内容相同。之后的位置差异可以安全变换：

$$
K_t=R(p_t)R(p_s)^{-1}K_s,\qquad V_t=V_s
$$

保留操作节点，正是为了禁止这种变换跨越计算历史分歧。实际有限精度旋转可能有舍入误差，所以这里的正确性论证并不要求与重算逐位相同。

至此，AKV 完成了从操作定义到服务实现的闭环。接下来的关键问题是：这些设计能否同时改善任务效果与真实系统效率？

## 四、实验：任务更准，成本更低

### 4.1 实验设置

主实验覆盖三个模型：MiniMax-M2.7（230B 总参数、10B 激活）、GPT-OSS-120B（5.1B 激活）和 MiroThinker-1.7-Mini（30B 总参数、3B 激活）。GPT-OSS-120B 使用 High Reasoning Effort，另外两个模型使用 Thinking Mode。

四个基准包括深度研究任务 BCP、GAIA-Text 和 xbench-DS，以及软件工程任务 SWE-Verified。GAIA-Text 是 GAIA 的 103 个纯文本验证任务。

对比方法只有上下文管理策略不同：

- **Standard**：GPT-OSS-120B、MiniMax-M2.7 在 96K Token 触发摘要，MiroThinker-1.7-Mini 在 192K 触发。
- **Text Engineering**：采用相同摘要策略，但所有模型都在 48K 触发。
- **AKV**：滚动保留最近 12 轮工具响应 KV，Repos 阈值与 Standard 的摘要阈值一致。

任务指标是通过率和每任务平均 API 等效成本，成本包括主 Agent 与上下文摘要调用，单位为美分。AKV 的 Repos Token 还按 Cache-hit 价格计费，有意保守估计 Key Rotation 开销。

### 4.2 三个模型、四个基准的主结果

下表完整列出得分（%）与成本（美分）。每个模型内部，得分越高越好，成本越低越好。

| 模型 | 方法 | BCP 得分 | BCP 成本 | GAIA 得分 | GAIA 成本 | SWE 得分 | SWE 成本 | xbench 得分 | xbench 成本 | 平均得分 | 平均成本 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniMax-M2.7 | Standard | 58.9 | 5.2 | 35.0 | 2.4 | 75.2 | 7.0 | 40.5 | 5.5 | 52.4 | 5.0 |
| MiniMax-M2.7 | Text Engineering | 62.3 | 5.3 | 36.9 | 2.7 | 72.0 | 9.4 | 40.0 | 5.2 | 52.8 | 5.7 |
| MiniMax-M2.7 | AKV | 58.9 | 4.8 | 40.8 | 2.6 | 74.4 | 5.5 | 40.5 | 4.7 | 53.7 | 4.4 |
| GPT-OSS-120B | Standard | 57.5 | 13.1 | 68.0 | 5.4 | 40.0 | 20.0 | 38.5 | 6.2 | 51.0 | 11.2 |
| GPT-OSS-120B | Text Engineering | 52.7 | 12.6 | 65.1 | 6.8 | 36.8 | 15.8 | 38.5 | 5.4 | 48.3 | 10.2 |
| GPT-OSS-120B | AKV | 61.9 | 11.4 | 69.9 | 4.1 | 43.2 | 9.7 | 37.5 | 3.8 | 53.1 | 7.3 |
| MiroThinker-1.7-Mini | Standard | 60.0 | 25.2 | 68.0 | 9.2 | 7.8 | 33.6 | 40.5 | 13.5 | 44.1 | 20.4 |
| MiroThinker-1.7-Mini | Text Engineering | 45.7 | 17.6 | 66.0 | 11.0 | 4.2 | 28.4 | 32.5 | 15.3 | 37.1 | 18.1 |
| MiroThinker-1.7-Mini | AKV | 63.5 | 15.3 | 68.9 | 9.5 | 7.4 | 26.3 | 43.5 | 8.1 | 45.8 | 14.8 |

AKV 在三个模型上都取得最高平均得分：相对 Standard 提高 1.3–2.1 个百分点，相对 Text Engineering 提高 0.9–8.7 个百分点。跨三个模型和四个基准平均，成本相对 Standard 降低 27.7%，相对 Text Engineering 降低 21.9%。

它并非每个单项都最优。例如 MiniMax 的 BCP 得分低于 Text Engineering，GPT-OSS-120B 的 xbench-DS 得分也略低；因此更准确的结论是“平均效果与成本更优”，而不是“全面碾压”。

### 4.3 为什么摘要阈值之后，AKV 更稳

在 GPT-OSS-120B、BCP 前 100 个任务上，按逻辑序列长度分组可以看到明显规律。

![BCP 不同逻辑序列长度下的准确率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/bcp_correct_cost_by_length_trial0.png)

> 图解：横轴依次为 0–48K、48–96K、96K+ 的逻辑序列长度，纵轴为准确率。0–48K 区间三种方法都是 88%；48–96K 区间 Standard、Text Engineering、AKV 分别为 65%、54%、70%；96K+ 区间分别为 24%、32%、37%。Text Engineering 在跨过自己的 48K 摘要阈值后先下滑，Standard 在 96K+ 区间最弱，而 AKV 的滚动缩减更平缓。

这与两种机制的差异一致：摘要到阈值后突然重写大量历史，可能丢失中间轮信息；AKV 则持续移除较旧工具响应，同时保留推理和近期观察。需要注意，横轴是逻辑长度，不是实际参与 Attention 的 Active Context 长度。

任务效果成立之后，还要验证节省的 KV 是否能转化为真实吞吐与延迟收益，而不只是账面 Token 减少。

## 五、系统收益：吞吐、延迟与成本来自哪里

### 5.1 不同并发下的服务性能

服务实验使用 GPT-OSS-120B 和 SWE-Verified 中上下文最长的 80 个任务。TTFT 是首 Token 延迟，TPOT 是每个输出 Token 的耗时；请求延迟和 TTFT 单位为秒，TPOT 单位为毫秒，吞吐单位为 Token/s。

| 并发 $C$ | 方法 | 请求延迟均值 | 请求延迟 P95 | TTFT 均值 | TTFT P95 | TPOT 均值 | TPOT P95 | 输入吞吐 | 输出吞吐 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | Standard | 12.16 | 43.57 | 3.18 | 8.42 | 26.57 | 31.38 | 25200 | 217 |
| 8 | Text Engineering | 12.84 | 43.57 | 4.94 | 22.99 | 23.23 | 25.68 | 17313 | 228 |
| 8 | AKV | 7.79 | 25.93 | 1.87 | 5.99 | 23.50 | 29.42 | 39227 | 245 |
| 10 | Standard | 12.82 | 45.03 | 3.93 | 9.76 | 26.47 | 30.83 | 30108 | 254 |
| 10 | Text Engineering | 15.17 | 50.27 | 7.35 | 34.07 | 22.57 | 25.09 | 18533 | 246 |
| 10 | AKV | 8.13 | 27.11 | 2.22 | 7.36 | 23.20 | 29.86 | 46002 | 295 |
| 12 | Standard | 12.39 | 43.16 | 2.97 | 9.99 | 28.00 | 32.78 | 33832 | 313 |
| 12 | Text Engineering | 16.83 | 52.82 | 8.86 | 37.98 | 23.24 | 25.95 | 19945 | 264 |
| 12 | AKV | 8.47 | 27.18 | 2.60 | 8.17 | 23.06 | 29.23 | 48923 | 339 |
| 14 | Standard | 30.16 | 108.86 | 21.36 | 88.29 | 24.50 | 29.85 | 12767 | 146 |
| 14 | Text Engineering | 19.59 | 60.67 | 11.19 | 47.36 | 24.50 | 27.23 | 19992 | 267 |
| 14 | AKV | 9.09 | 28.49 | 2.92 | 8.62 | 24.50 | 31.56 | 51837 | 369 |
| 16 | Standard | 36.50 | 229.90 | 27.71 | 223.57 | 24.83 | 29.25 | 10866 | 138 |
| 16 | Text Engineering | 23.00 | 69.96 | 14.48 | 52.71 | 24.44 | 27.56 | 19595 | 267 |
| 16 | AKV | 10.32 | 30.63 | 4.16 | 13.02 | 24.84 | 32.41 | 50817 | 369 |

AKV 在所有列出的并发级别上都有更低请求延迟和更高吞吐。各自峰值设置下，AKV 取 $C=14$，Standard 取 $C=12$：AKV 的平均与 P95 请求延迟加速分别为 1.36×、1.51×，输入与输出吞吐分别为 Standard 的 1.53×、1.18×。

输入吞吐统计全部逻辑输入 Token，包括已经 Drop 的 Token。高并发下，Standard 因 KV 压力反复 Prefill，可同时 Decode 的请求减少，所以 $C\geq14$ 时差距进一步扩大。

### 5.2 真实 SWE 任务：更少 GPU 时间完成更多任务

下面三幅图分别把任务通过率与 GPU 时间、逻辑输入量和服务并发联系起来。

![SWE-bench 上累计通过率随 GPU 时间变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/swe_gpu_hours_cumulative_pass_rate.png)

> 图解：横轴为 GPU 小时，纵轴为累计通过率；结果是 GPT-OSS-120B 在 SWE-bench Verified 前 100 个任务上的在线评估，每种方法运行 3 次，阴影为 $\pm1$ 标准差。各方法使用自己的峰值吞吐并发，AKV 与 Text Engineering 标注 $C=14$，Standard 标注 $C=12$。AKV 曲线更快上升，并用最少 GPU 时间达到最高最终通过率。

![SWE-bench 上累计通过率随逻辑输入 Token 变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/swe_cumulative_pass_rate_compact.png)

> 图解：横轴是全部 500 个任务的累计逻辑输入，单位为百万 Token；纵轴是累计通过率。AKV 最终约为 43%，高于 Standard 的约 40% 和 Text Engineering 的约 37%，说明收益不只是把输入机械缩短，而是能持续把更多任务做成功。

![不同并发下的总服务吞吐](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/serving_total_throughput.png)

> 图解：横轴为并发数，纵轴为总吞吐，单位为千 Token/s。Standard 在 $C=12$ 后明显下滑，Text Engineering 整体较平，AKV 在 $C=14$ 附近达到峰值并在 $C=16$ 仍保持高位。这解释了为什么高并发时 AKV 的端到端优势更明显。

### 5.3 Decode 加速主要来自 Full Attention

在同一 Batch Size 下，AKV 将 Full Attention 时间减少 59.8%，总 Decode Step 时间减少 16.0%。

![Standard 与 AKV 的 Decode 延迟拆解](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/decode_breakdown.png)

> 图解：纵轴为每个 Decode Step 的毫秒数，堆叠颜色依次展示 Full Attention、SWA Attention、MoE、Attention Projections、TP Communication 和其他开销。总耗时从 Standard 的 26.186 ms 降到 AKV 的 21.990 ms；Full Attention 从 7.082 ms 降到 2.849 ms，而 MoE 部分约为 16.764 ms 和 16.785 ms，几乎不变。收益主要来自更短的 Active Context 减少了 KV 读取和 Attention 计算。

### 5.4 API 成本为什么下降

BCP 的成本拆解进一步区分了 Cache Read、Cache-miss Input、Output 和 Repos。

![BCP 单任务 API 成本拆解](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/bcp_cost_breakdown.png)

> 图解：纵轴为每任务成本，单位为美分。Standard、Text Engineering、AKV 总成本分别为 13.1、12.6、11.4；其中 Cache Read 分别约为 10.8、8.5、9.0。Standard 的 10.8/13.1 约为 82.4%，说明缓存读取确实是主要开销。AKV 相对 Standard 的 Cache Read 成本降低约 17%，相对 Text Engineering 的 Cache-miss Input 成本降低约 48%；后者对应 2.9 到 1.5 美分的变化，Repos 另计约 0.3 美分。

AKV 对 Standard 的优势来自少读旧 KV；对 Text Engineering 的优势还来自不重写文本，因此无需为摘要后的保留后缀反复支付 Cache-miss Prefill。

![Standard 与 AKV 的输入长度随轮次变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/bcp_input_length_by_round.png)

> 图解：横轴为 BCP 的 100 个 Agent Turn，纵轴为输入长度，单位为千 Token，虚线标出 96K。Standard 多次积累到高位后因摘要出现锯齿状骤降；AKV 在超过工具响应保留预算后持续滚动删除，因此曲线更平滑。由于推理 KV 仍不断积累，AKV 的 Active Context 也会增长，只是速度更慢。

这些结果说明，成本、延迟和准确率不是三个孤立收益，而是同一条机制链的不同表现：少参与 Attention 的 KV，带来更少读取与计算；不修改文本，则保留后缀缓存并避免突然信息断层。

## 六、消融与策略进化：丢什么、谁来决定

### 6.1 Drop 目标与 Repos 缺一不可吗

论文在 GPT-OSS-120B、各基准前 100 个任务上比较不同删除对象，以及是否启用 Repos。

| 策略 | Repos | BCP 得分 | BCP 成本 | SWE 得分 | SWE 成本 |
|---|---|---:|---:|---:|---:|
| Standard | 否 | 56.0 | 13.5 | 41.0 | 19.1 |
| Tool Drop | 否 | 55.0 | 7.1 | 41.0 | 9.4 |
| Reasoning Drop | 是 | 56.5 | 8.2 | 36.5 | 14.2 |
| Both Drop | 是 | 56.5 | 8.7 | 34.5 | 7.8 |
| Tool Drop | 是 | 62.0 | 10.9 | 41.0 | 9.5 |

只删工具响应已经大幅降本，但不重排位置时，逻辑位置仍可能触及模型上限，迫使探索提前停止。加入 Repos 后，BCP 通过率从 55% 提升到 62%，SWE 则与其他 41% 的策略并列最高。

在启用 Repos 时，删推理或同时删推理与工具响应，都不如只删工具响应。这支持了前面的设计判断：旧工具原始输出通常比推理状态更适合优先退出 Active Cache。

### 6.2 Cache-As-Tool：让 Agent 自己决定

除了固定滚动策略，论文还把缓存管理暴露为工具，即 Cache-As-Tool（CAT），让 Agent 主动决定何时缩减、保留什么。

| 模型 | 指标 | Standard | Text Engineering | CAT |
|---|---|---:|---:|---:|
| GPT-OSS-120B | 得分 | 52.1 | 50.0 | 61.0 |
| GPT-OSS-120B | 成本 | 13.4 | 13.9 | 12.1 |
| MiniMax-M2.7 | 得分 | 56.0 | 61.0 | 64.0 |
| MiniMax-M2.7 | 成本 | 5.4 | 5.5 | 5.9 |

在 100 个 BCP 任务子集上，CAT 对两个模型都取得最高通过率。GPT-OSS-120B 的成本从 13.4 降到 12.1 美分；MiniMax-M2.7 则从 5.4 升到 5.9 美分，原因是它执行了更多轮。也就是说，自主缓存管理首先扩大的是策略空间，并不保证每个模型都更便宜。

### 6.3 RSI：让 Agent 搜索更好的缓存策略

递归自我改进（Recursive Self-Improvement，RSI）进一步把缓存策略本身当成可优化对象。论文用 MiniMax-M2.7 在 20 个 BCP 任务上迭代策略，再到 100 个留出任务上评估。

![MiniMax-M2.7 缓存策略的递归自我改进](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/minimax_self_evolution.png)

> 图解：横轴为搜索迭代，纵轴为每任务成本，单位为美分；实心点表示达到基线准确率，空心点表示低于基线，阶梯线是当前最优成本。上方虚线是 Standard，下方虚线是人工专家设计。第 3、6 轮策略在保持 Standard 准确率的同时，成本分别比 Standard 低 15.6%、17.8%；最终策略相对人工专家设计进一步降本 7.1%。

固定 $K=12$ 因此不是终点。Append、Drop、Repos 一旦成为可编程操作，Agent 就能在真实任务反馈上调整保留预算、删除对象和触发时机。

### 6.4 更多证据：跨层 Attention、Pi 与 GPT-OSS-20B

跨层 Attention 热图补充了“为什么优先删旧工具响应”的证据。

![不同消息区间和网络层的 Attention 密度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Agent-Managed-KV-Cache-Reduction/figure/bcp_attention_request37_mean10.png)

> 图解：横轴是消息索引，依次划分旧推理、旧工具响应和近期历史；纵轴是 Layer Index，颜色表示每 Token Attention Density，色条采用 $10^{-9}$ 到 $10^{-4}$ 的对数尺度，越亮越高。结果平均自 10 条 BCP 轨迹。多数层中旧工具区域比旧推理更暗，近期历史整体更亮，与滚动删除策略一致；这是一种相关性证据，而不是单独证明因果。

与现有 harness 的兼容性也经过验证。AKV 集成到 Pi 后，GPT-OSS-120B 和 GPT-OSS-20B 在 BCP 上，相对不做 Compaction 的原生 Pi 都提高 3 个百分点，相对做 Compaction 的 Pi 都提高 4 个百分点。

GPT-OSS-20B 使用 High Reasoning Effort，Standard 与 AKV Repos 阈值为 96K，Text Engineering 为 48K，结果如下：

| 方法 | BCP 得分 | BCP 成本 | GAIA 得分 | GAIA 成本 | SWE 得分 | SWE 成本 | xbench 得分 | xbench 成本 | 平均得分 | 平均成本 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 49.0 | 6.5 | 60.2 | 3.5 | 41.8 | 12.5 | 25.0 | 2.4 | 44.0 | 6.2 |
| Text Engineering | 43.0 | 4.4 | 65.1 | 3.0 | 25.6 | 8.8 | 19.0 | 2.0 | 38.2 | 4.6 |
| AKV | 50.5 | 6.0 | 62.1 | 2.5 | 44.2 | 8.9 | 24.0 | 1.8 | 45.2 | 4.8 |

AKV 平均得分最高，为 45.2%，平均成本 4.8 美分，与两种基线相近但并非最低。论文将较小收益归因于小模型利用保留信息的能力较弱、对缓存修改更敏感；这属于作者解释，而不是由该表单独证明的结论。

## 七、复现细节：配置、协议与成本口径

详细解读不仅要报结果，还要弄清结果如何测得。以下信息决定了实验的可比性与复现边界。

### 7.1 服务配置

AKV 基于 SGLang v0.5.20。由于 A800 缺少原生 MXFP4 支持，所有方法都把 GPT-OSS-120B 权重展开为 BF16。

| 配置项 | 值 |
|---|---|
| 模型与权重精度 | GPT-OSS-120B，BF16 |
| 硬件 | 8 张 NVIDIA A800，每张 80 GB |
| 并行方式 | Prefill：TP4；Decode：TP4 |
| KV 传输 | Mooncake，经节点内 NVLink |
| Attention 后端 | Triton |
| MoE 后端 | `triton_kernel` |
| 静态显存比例 | 0.85 |
| KV Page 大小 | 每页 1 个 Token |
| Prefill Chunk 大小 | 8,192 Token |
| 最大 Prefill Token 数 | 8,192 |
| Prefix Cache | 两类 Worker 都启用 Radix Cache |
| 驱逐策略 | 基线：LRU；AKV：Drop-aware Eviction |

Prefill 与 Decode 各使用 4 张 GPU。基线与 AKV 共享同一底层服务框架，减少了跨系统实现差异对对比的干扰。

### 7.2 工作负载回放与测量窗口

- **任务选择**：从不做摘要的轨迹中，选出峰值上下文最长的 80 个 SWE-bench Verified 任务，以固定种子打乱；所有方法使用相同任务和顺序。
- **轨迹来源**：每种方法回放自己策略下采集的轨迹，因此 Turn 数和输出预算可以不同。
- **生成方式**：从完整消息历史重建每轮输出预算，使用 Temperature 0 和 `ignore_eos=True` 生成指定数量 Token。
- **工具处理**：Assistant 响应进入后续历史，工具响应直接回放，不重新执行工具；没有对应 Assistant 响应的终止输入被排除。
- **并发控制**：维持 $C$ 个客户端任务并发，每个任务内部按轮顺序执行。
- **测量窗口**：目标任务数为 $\max(3C,30)$，包含为维持并发而补充的任务请求。AKV 在 $C=14,16$ 时按完成顺序截取前 30 个任务完成时的窗口；Standard 高并发运行因反复 Prefill 严重退化，采用固定 1 小时窗口，截止时仍有请求未完成。

因此，高并发结果应理解为特定长上下文回放协议下的系统表现，而不是任意工作负载都能复现的固定倍数。

### 7.3 Benchmark、评审与摘要基线

不同基准使用不同 harness：BCP 用官方实现，SWE-bench Verified 用 mini-SWE-agent，GAIA-Text 用 Observation-masking 的 Browser-agent Scaffold，xbench-DS 用 WebExplorer 的评估框架。

| 基准 | 检索、阅读与评审 | Agent 步数上限 |
|---|---|---:|
| BCP | Qwen3-Embedding-8B 从固定语料检索；提供 `search` 与 `get_document`；GLM-5.2 按官方 BCP Prompt 评审 | 100 |
| GAIA-Text | GLM-5.2 按 Observation-masking Prompt 评审 | 200 |
| xbench-DS | GLM-5.2 提取网页信息，并按 WebExplorer Prompt 评审 | 100 |
| SWE-bench Verified | 官方测试脚本与评分函数，不使用 LLM Judge | 250 |

摘要基线复用 Pi 的 Prompt、对话序列化、切点选择和摘要预算，`reserveTokens=16,384`，`keepRecentTokens=20,000`。摘要使用被评估 Agent 自己的模型与推理设置，保留 System、Developer 指令和选中的近期消息，用摘要替换更早对话。

与 Pi 原生按“上下文窗口减去预留预算”触发不同，实验在精确 Prompt Token 数达到阈值时触发。空摘要会被拒绝，摘要失败按各基准错误协议处理，不重试。摘要内容聚焦三部分：原始请求、早期进展，以及理解保留后缀所需的上下文。

### 7.4 API 等效成本口径

成本只包括被评估 Agent 的生成和上下文摘要调用。Cache-miss、Cache-hit 和 Output Token 分别计费，同一模型在不同方法间使用相同价格；Repos 按 Cache-hit 价格保守计费。

| 模型 | Cache-miss（美元/百万 Token） | Cache-hit（美元/百万 Token） | Output（美元/百万 Token） |
|---|---:|---:|---:|
| MiniMax-M2.7 | 0.300 | 0.060 | 1.200 |
| GPT-OSS-120B | 0.150 | 0.075 | 0.600 |
| MiroThinker-1.7-Mini | 0.200 | 0.040 | 2.400 |
| GPT-OSS-20B | 0.075 | 0.0375 | 0.300 |

GPT-OSS 系列使用 Groq 价格，MiniMax-M2.7 使用官方价格，MiroThinker-1.7-Mini 使用其基座 Qwen3-30B-A3B 的官方价格。因此，表中“美分”是统一 API 等效成本，不应与八卡 A800 的实际 GPU 账单混为一谈。

项目代码开源于 [thu-nics/AKV](https://github.com/thu-nics/AKV)。

### 7.5 与相关及并发工作的关系

在更细的研究版图中，AgentKV 用不同执行阶段的 Query Buffer 为 KV 打分，IntentKV 结合跨轮 Query Memory 和学习得到的重要性估计，SideQuest 用辅助推理分支识别过期工具结果并发出删除命令。它们都在探索“Agent 轨迹需要什么样的动态驱逐”。

Context Language Models（CLMs）则把活跃上下文当成可编辑文件，并通过 Suffix Cache Reuse 保留文本编辑后的下游 KV。论文指出，CLM 的相关服务支持主要面向 Session-local 私有缓冲，而 AKV 更强调策略无关的操作接口，以及在共享 Radix Cache 中统一索引、复用、驱逐和恢复历史依赖状态。

从这一角度看，AKV 的贡献不是单一的删除启发式，而是把多种现有和未来策略都能调用的缓存操作，与一套支持多请求共享的系统机制连接起来。

## 八、总结与展望

- **核心抽象**：Cache Engineering 保持逻辑文本不变，直接控制后续 Attention 可见的 KV，把 Active Context 与轨迹长度解耦。
- **关键操作**：Append 负责新增状态，Drop 负责删除可见历史，Repos 通过 RoPE 旋转整理位置，无需重算保留后缀。
- **系统闭环**：AKV 用操作感知 Radix Tree、Drop-aware Eviction 和 Segment-batch Attention，实现正确复用、安全驱逐与单次 Prefill 恢复。
- **实验结论**：滚动删除旧工具响应在三模型、四基准上平均提高 1.7 个百分点、降本 27.7%，并带来 1.18× 输出吞吐和 1.51× P95 请求延迟加速。
- **策略空间**：CAT 与 RSI 表明，缓存管理可以从固定规则升级为 Agent 自主决策和持续优化的对象。

这项工作的下一步，是探索更系统的缓存策略、让 Agent 在 Cache Engineering 机制下直接训练，并把服务支持扩展到 Linear Attention 模型；同时也需要继续验证不同模型对缓存修改的敏感性，以及有限精度位置旋转的长期影响。

> 本文参考自 [Agent-Managed KV-Cache Reduction](http://arxiv.org/abs/2610.28577v1)