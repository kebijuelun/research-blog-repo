• # LoRA 到底怎么记住东西的？揭秘大模型参数化记忆的幂律与相变

  大语言模型（LLM）在部署后面临一个根本挑战：**知识是静态的**，但现实世界是动态的。新的事实、更新的文档、用户的个性化偏好……这些都需要模型在预训练之后继续"学习"。目前最主流的参数高效微调手段——**Low-Rank Adaptation (LoRA)**——本质上就是在做这件事：把新知识"写"进一组低秩的增量参数里。

  但一个关键问题长期被忽视：LoRA 到底能记住多少东西？它的记忆边界由什么决定？为什么有时候 Loss 降得很低，模型却照样背不出原文？

  今天这篇文章要解读的正是来自浙江大学和阿里集团的 **How LoRA Remembers? A Parametric Memory Law for LLM Finetuning**。它不仅给出了参数化记忆的 **定量容量定律**，还揭示了 token 级别的 **确定性相变机制**，并基于这些洞察提出了一个叫 **MemFT** 的优化策略。整篇工作从宏观规律到微观动力学，再到实际方法，形成了一个完整的闭环。

  ---

  ## 一、问题背景：为什么需要"精确参数化记忆"？

  现有评估 LoRA 记忆能力的方式，大多是看下游任务准确率——比如问答对不对、摘要好不好。但这属于 **functional memory**，它把"记忆"和"理解"混在了一起。如果你问模型"某法律条文的原文是什么"，或者"我的数据库密码是多少"，这种场景要求的不是语义近似，而是 **verbatim recall（逐字复述）**。哪怕错一个标点、一个数字，结果都可能完全失效。

  > 图解：精确记忆 vs 功能性记忆。功能性记忆允许语义等价替换，但密码、代码、法律条文等场景要求字符级精确还原。

  为了把问题彻底拆解清楚，作者们设计了一套 **精确参数化记忆任务** 的框架：

  - 输入是一个 **key**（如查询问题），目标是让模型输出对应的 **value**（如目标文本）。
  - 基模型参数完全冻结，只训练 LoRA 增量参数 $\Delta\theta$。
  - 评估时采用 **greedy decoding**，只有生成结果与目标文本完全一致才算成功。

  这种设定把记忆问题还原成了最纯粹的"参数写入"问题——没有外部检索，没有上下文提示，全凭 LoRA 那一点点增量参数把信息存进去、再取出来。

  ---

  ## 二、宏观发现：参数化记忆定律（Parametric Memory Law）

  ### 2.1 从现象出发：双对数空间里的惊人线性

  作者们以 LoRA 的秩 $r$ 作为可控的"容量探针"，在 **Qwen3-8B-IT** 和 **Llama3.1-8B-IT** 上做了大规模扫描。实验覆盖了两个互补场景：

  - **Long-Context Memorization Stress Test**：基于 LongBench 样本，将 0%–100% 的 token 替换为随机 token，构造从强语义到完全无语义的长序列记忆任务。
  - **PhoneBook**：大量短小的 key-value 对（如姓名→电话），测试高密度短文本记忆。

  核心观测指标定义为 **Loss Reduction**：

  $$
  \Delta \mathcal{L} = \mathcal{L}_{init} - \mathcal{L}_{final}
  $$

  其中 $\mathcal{L}_{init}$ 和 $\mathcal{L}_{final}$ 分别是微调前后的交叉熵损失。结果令人惊讶：如果把 $\Delta \mathcal{L}$、rank $r$ 和序列长度 $\ell$ 都取对数，数据点几乎落在一条直线上。

  ![Figure 2a](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/fig_rank_curves.png)

  > 图解：左图展示 $\Delta \mathcal{L}$ 随 rank $r$ 的变化，右图展示随长度 $\ell$ 的变化，两者在 log-log 空间中均呈现高度线性。这意味着损失降低量与 rank、length 之间不是简单的多项式关系，而是 **幂律关系**。

  ![Figure 2b](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/fig_3d_plane.png)

  > 图解：三维视角下，$\log(\Delta \mathcal{L})$ 在 $\log(r)$ 和 $\log(\ell)$ 构成的平面上呈现出清晰的平面结构，进一步印证了幂律假设。

  ### 2.2 定律的数学形式

  基于上述观察，作者提出了 **Parametric Memory Law**：

  $$
  \Delta \mathcal{L}(r, \ell) = C \cdot r^{\alpha} \cdot \ell^{-\beta} + b
  $$

  公式中各参数的含义如下：

  - **$C$** ：与模型架构和数据分布相关的缩放常数。
  - **$\alpha$（Capacity Exponent）** ：容量指数，量化增加 rank 对记忆效率的提升。实验显示 $\alpha \approx 0.6$。
  - **$\beta$（Length Penalty Exponent）** ：长度惩罚指数，反映序列越长记忆难度越高。实验显示 $\beta \approx 0.5$。
  - **$b$** ：基线偏移项。

  这个定律揭示了一个核心事实：**在显著记忆增益区间内，LoRA 的记忆能力严格受制于参数秩与序列长度之间的幂律博弈**。增加 rank 的收益是次线性的（$\alpha < 1$），而长度增加带来的惩罚同样是次线性的。

  ### 2.3 拟合验证：跨模型、跨任务的普适性

  为了验证这一定律的稳健性，作者用非线性最小二乘法对实验数据进行了拟合。

  ![Figure 2c](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/fig_pred_vs_true.png)

  > 图解：预测值与真实值的散点图。拟合优度 $R^2 = 0.996$，说明该幂律模型具有极高的解释力。

  | Model | Setting | $R^2$ $\uparrow$ | MAPE (%) $\downarrow$ |
  |:---:|:---:|:---:|:---:|
  | Llama3.1-8B-IT | Long-Context (Combined) | **0.987** | 7.057 |
  | Llama3.1-8B-IT | PhoneBook | 0.981 | 1.606 |
  | Qwen3-8B-IT | Long-Context (Combined) | **0.983** | 8.320 |
  | Qwen3-8B-IT | PhoneBook | 0.990 | 0.476 |

  > 表格解读：无论在长文本压力测试（0%–100% random token 混合）还是短文本 PhoneBook 任务上，该定律的 $R^2$ 均超过 **0.98**。特别值得注意的是，**一个统一的公式** 就能拟合从纯语义到完全随机、从长序列到短序列的全部数据，说明这个幂律不是偶然现象，而是参数化记忆的几何本质。

  但这里有一个巨大的"坑"：作者发现，很多情况下 **Loss 接近 0，Accuracy 却接近 0**。

  ![Figure 2d](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/fig_heatmap_loss.png) ![Figure 2e](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/fig_heatmap_acc.png)

  > 图解：左图是不同 $(r, \ell)$ 配置下的最终 Loss 热力图，右图是 Token-Level Accuracy 热力图。可以明显看到大量"Loss 很低但 Accuracy 几乎为 0"的区域（如右图中的蓝色区域）。这说明 **平均 Loss 是记忆失败的糟糕代理指标**。

  这就引出了下一个核心问题：如果平均 Loss 不可靠，那什么才可靠？

  ---

  ## 三、微观机制：Token 级别的确定性相变

  ### 3.1 Loss-Accuracy 悖论

  平均交叉熵损失的问题在于它是一个"和稀泥"的指标。模型可能在 99% 的 token 上信心满满（Loss 极低），但只要有一个 token 没记牢，在自回归解码中就会引发 **级联错误**——一个错，后面全错。

  ![Figure 3a](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/stubborn_token_a.png)

  > 图解：不同 rank 下，某些特定位置的 token 概率始终低于 $p=0.5$（红色虚线），即便增加 rank 也难以改善。这些就是 **stubborn tokens（顽固 token）**。它们数量极少，却是记忆的阿喀琉斯之踵。

  ![Figure 3c](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/stubborn_token_c.png)

  > 图解：首次解码失败位置的直方图显示，失败高度集中在极少数特定位置上。例如位置 $i=153$ 单独贡献了 28% 的失败案例。这说明瓶颈是局部的、结构性的，而非均匀分布的。

  ![Figure 3b](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/How-LoRA-Remembers-A-Parametric-Memory-Law-for-LLM-Finetuning/figures/stubborn_token_b.png)

  > 图解：最早的 stubborn position 与首次自由解码失败位置 $i^*$ 之间存在极强的相关性（Spearman $\rho=0.908$）。这意味着 **sub-threshold 概率显著增加了级联崩溃的风险**。

  ### 3.2 确定性相变：$p=0.5$ 是记忆生死线

  在 greedy decoding 设定下，模型每步都选概率最高的 token。如果目标 token 的概率 $P_{\text{target}} > 0.5$，那么没有任何其他单个 token 的概率能超过它——**目标 token 必然被选中**。这是一个充分条件。

  反之，如果 $P_{\text{target}} < 0.5$，目标 token 就失去了概率主导地位，很容易被其他候选 token 取代。而一旦某个位置预测错误，后续所有 token 的上下文条件就全变了，导致 **自回归级联崩溃**。

  将 $P_{\text{target}} = 0.5$ 代入交叉熵公式 $\mathcal{L} = -\log(P_{\text{target}})$，得到临界损失：

  $$
  \mathcal{L}_{\text{crit}} = -\log(0.5) = \ln(2) \approx 0.693
  $$

  这就是 **确定性相变的临界点**：

  - **无序相（Disordered Phase）**：$\mathcal{L} > \mathcal{L}_{\text{crit}}$，即 $P_{\text{target}} < 0.5$。正确 token 不占主导，记忆处于不确定状态，随时可能失败。
  - **有序相（Ordered Phase）**：$\mathcal{L} < \mathcal{L}_{\text{crit}}$，即 $P_{\text{target}} > 0.5$。正确 token 稳操胜券，greedy 解码下必然成功。

  这也解释了为什么第二节中拟合 Parametric Memory Law 时要剔除 $\mathcal{L}_{final} \le 0.69$ 的样本——一旦进入有序相，损失变化的动态规律就变了，不再服从幂律。

  > 核心洞察：平均 Loss 降得再低，只要还有 token 没跨过 $\mathcal{L}_{\text{crit}} \approx 0.693$ 这条线，整个序列的 exact match 就可能是 0。记忆成功的关键不是"整体 Loss 多低"，而是"所有 token 是否都跨过了相变阈值"。

  ---

  ## 四、方法论：MemFT——把优化预算花在刀刃上

  ### 4.1 核心动机

  标准 SFT 的损失函数是对所有 token 平均：

  $$
  \mathcal{L}_{\text{SFT}} = \frac{1}{\ell} \sum_{t=1}^{\ell} \mathcal{L}_t
  $$

  这种"大锅饭"式的优化，让已经跨过相变阈值（$\mathcal{L}_t < 0.693$）的"easy tokens"继续占用大量梯度预算，而那些卡在阈值以下的 stubborn tokens 却得不到足够关注。

  ### 4.2 MemFT-OT：基于硬阈值的掩码

  MemFT（Memorization-oriented Fine-Tuning）的核心思想很简单：**只优化还没过门槛的 token**。

  $$
  w_t^{\text{TH}} = \mathbf{1}\left[\mathcal{L}_t > \mathcal{L}_{\text{crit}}\right]
  $$

  $$
  \mathcal{L}_{\text{MemFT}}(\theta) = \frac{\sum_{t \in \mathcal{M}} w_t \cdot \mathcal{L}_t(\theta)}{\sum_{t \in \mathcal{M}} w_t + \varepsilon}
  $$

  MemFT-OT（Only Threshold）直接把低于阈值的 token 权重置零，梯度完全集中在 sub-threshold tokens 上。这个方法不需要任何额外超参。

  ### 4.3 MemFT-SW：自适应滑动机制

  MemFT-SW（Sliding Window）在 MemFT-OT 基础上增加了两个更精细的策略：

  **（1）样本内空间滑动（Intra-sample Spatial Sliding）**

  找到当前序列中 **第一个预测错误的位置**（anchor $a_i$），然后在这个位置附近开一个指数衰减的滑动窗口：

  $$
  \phi_t = \exp(-\max(t - a_i, 0) / \tau)
  $$

  窗口内的 token 获得更高的优化权重，窗口外的权重降至地板值 $\epsilon_{\text{floor}}$。如果 anchor 位置长期不动，窗口还会自动扩展，防止训练陷入僵局。

  **（2）批次间时间课程（Inter-batch Temporal Curriculum）**

  在训练初期，模型只接触较简单的样本（如前 20% 的批次）；随着训练进行，逐步放开到全部样本。这避免了模型在局部记忆还没稳定时就被全局复杂度淹没。

  ---

  ## 五、实验验证：MemFT 到底强在哪里？

  作者在 **Long-Context Memorization Stress Test** 和 **PhoneBook** 两个基准上对比了标准 SFT、MemFT-OT 和 MemFT-SW。

  ### 5.1 长文本记忆压力测试

  | Method | $r_1$ | $r_2$ | $r_3$ | $r_4$ | $r_5$ | $r_6$ | $r_7$ | $r_8$ | $r_9$ |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | **Llama3.1-8B-IT** |
  | SFT | 27.4 | 28.5 | 43.6 | 45.9 | 54.9 | 69.5 | 78.2 | 86.3 | 94.7 |
  | MemFT-OT | 27.3 | 36.4 | 45.6 | **54.7** | **63.6** | **70.5** | **85.4** | **94.7** | **100.0** |
  | MemFT-SW | **32.5** | **37.5** | **46.0** | 52.3 | 56.0 | 63.4 | 69.1 | 76.6 | 81.1 |
  | **Qwen3-8B-IT** |
  | SFT | 17.9 | 24.2 | 27.8 | 31.7 | 33.1 | 39.8 | 40.2 | 40.0 | 47.7 |
  | MemFT-OT | 19.2 | 23.6 | 29.8 | 38.5 | 47.5 | 56.1 | 91.1 | **100.0** | **100.0** |
  | MemFT-SW | **24.7** | **29.3** | **32.0** | **39.4** | **52.5** | **74.6** | **93.5** | 94.4 | 94.4 |

  > 表格解读：在长文本测试上，两种 MemFT 变体均显著优于 SFT。一个有趣的现象是 **rank 依赖的 regime shift**：低 rank 时 MemFT-SW 更优（滑动窗口缓解了局部瓶颈），高 rank 时 MemFT-OT 能更快达到 100% 准确率（如 Llama-$r_9$ 和 Qwen-$r_8$ 均达到完美记忆）。这说明当参数预算充足时，硬阈值 mask 的"聚焦力"更强；预算紧张时，滑动窗口的"疏导力"更有效。

  ### 5.2 PhoneBook 短文本密集记忆

  | Method | $p_1$ | $p_2$ | $p_3$ | $p_4$ | $p_5$ | $p_6$ | $p_7$ |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | **Llama3.1-8B-IT** |
  | SFT | 0.50 | 3.85 | 18.7 | 28.0 | 37.8 | 47.0 | 59.3 |
  | MemFT-OT | 1.00 | 11.2 | 31.4 | **53.9** | 61.0 | 73.9 | 87.0 |
  | MemFT-SW | **1.84** | **15.0** | **34.0** | 45.7 | **70.7** | **96.1** | **100.0** |
  | **Qwen3-8B-IT** |
  | SFT | 2.32 | 17.4 | 37.5 | 55.5 | 84.8 | **99.5** | **100.0** |
  | MemFT-OT | 5.78 | 19.1 | 36.2 | 57.4 | 86.1 | 98.6 | **100.0** |
  | MemFT-SW | **8.45** | **19.7** | **37.8** | **58.8** | **86.5** | **99.5** | **100.0** |

  > 表格解读：在 PhoneBook 上，**MemFT-SW 在几乎所有预算下都保持领先**。它是最快达到 100% EM 准确率的方法（Llama 在 $p_7$，Qwen 在 $p_6$）。这说明对于短文本、高密度的 key-value 记忆，结合空间滑动和时间课程的 MemFT-SW 具有系统性优势。

  ### 5.3 精确记忆场景的代表性案例

  作者列举了 8 个需要逐字复述的真实场景：

  | Scenario | Query | Target |
  |:---:|:---:|:---:|
  | Personal Credentials | 内部门户的登录邮箱和密码？ | xxx.xxx@company.com/P@ssw0rd_xxx |
  | Legal Compliance | GDPR 第5条第1款(a)项的原文？ | Processing shall be lawful only if... |
  | Medical Coding | 单纯2型糖尿病的 ICD-10 编码？ | E11.9 |
  | Model Watermark | 输出微调模型嵌入的归属水印？ | MEM-2026-LoRA-EXACT-0x7F9A3B... |
  | Cloud Configuration | 生产环境 AWS S3 日志桶的完整端点？ | s3://prod-application-logs-... |

  > 图解：这些场景的共同点是 **容错率极低**——一个字符的错误就可能导致操作失败、安全漏洞或法律风险。MemFT 通过锁定 sub-threshold tokens，显著提升了这类任务的可靠性。

  ### 5.4 不只是记忆：MemFT 还能提升泛化

  一个自然的担忧是：MemFT 如此专注于"死记硬背"，会不会牺牲泛化能力？

  作者在 **Linear Rule Learning** 基准上做了验证：让模型学习函数 $f(x, y) = 3x + 5y + 7$，训练集只有 500 个样本，测试集是未见的组合。

  | Rank | Method | Memory (%) | Generalization (%) |
  |:---:|:---:|:---:|:---:|
  | 1 | SFT | 83.0 | 19.0 |
  | 1 | MemFT | **95.0** | **34.0** ↑15.0 |
  | 2 | SFT | **100.0** | 38.0 |
  | 2 | MemFT | 97.0 | **47.0** ↑9.0 |
  | 4 | SFT | 99.0 | 46.0 |
  | 4 | MemFT | **100.0** | **53.0** ↑7.0 |
  | 8 | SFT | **100.0** | 39.0 |
  | 8 | MemFT | 99.0 | **49.0** ↑10.0 |
  | 16 | SFT | **100.0** | 41.0 |
  | 16 | MemFT | **100.0** | **54.0** ↑13.0 |

  > 表格解读：MemFT 在保持高记忆准确率的同时，**泛化准确率反而比 SFT 高出 7%–15%**。作者分析，这是因为 MemFT 抑制了模型在 easy samples 上的过度自信，并把优化重点放在"顽固 token"上，这种训练动态反而有助于学到更鲁棒的表征。

  ---

  ## 六、总结与展望

  这项工作用 LoRA 作为可控探针，从三个层面系统解构了 LLM 的参数化记忆：

  1. **宏观定律**：发现了 **Parametric Memory Law**，即 $\Delta \mathcal{L}$ 与 rank $r$、长度 $\ell$ 之间服从稳定的幂律关系 $\Delta \mathcal{L} = C \cdot r^{\alpha} \cdot \ell^{-\beta} + b$，且跨模型、跨任务高度一致。
  2. **微观机制**：揭示了 **确定性相变** 现象——$p=0.5$（即 $\mathcal{L}_{\text{crit}} \approx 0.693$）是 greedy decoding 下记忆成功的充分条件。低于此阈值的 stubborn tokens 是自回归级联崩溃的根源。
  3. **实用方法**：提出了 **MemFT**，通过阈值引导的梯度重分配，把优化预算集中在瓶颈 token 上，在精确记忆任务上全面超越标准 SFT，甚至还能提升泛化能力。

  当然，这项研究也存在一些局限：目前只在 8B 规模的模型上做了验证，更大模型的幂律指数是否一致还有待检验；$p=0.5$ 的相变结论严格依赖于 greedy decoding，在 nucleus sampling 等随机解码策略下的鲁棒性也需要进一步探索。但无论如何，它为理解 LLM 的"记忆本质"提供了第一个坚实的力学基础。

  > 本文参考自 [How LoRA Remembers? A Parametric Memory Law for LLM Finetuning](https://arxiv.org/abs/2605.30260)