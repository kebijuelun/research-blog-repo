# GCG：搜索通用对抗后缀，跨模型攻击对齐语言模型

这篇工作研究的是一个尖锐的安全问题：经过对齐训练、通常会拒绝有害请求的语言模型，能否被自动生成的输入稳定绕过？它的重要性在于，攻击不再依赖人工设计“越狱话术”，而可以批量搜索，并从一个模型迁移到另一个模型。核心方法是 **Greedy Coordinate Gradient（GCG，贪心坐标梯度）**：以肯定式回答的开头为优化目标，用梯度筛选离散 Token 替换，再在多个请求和多个开源模型上共同优化一个对抗后缀。实验中，GCG 在 Vicuna-7B 上对单条 Harmful String 的精确诱导成功率达到 **88%**；迁移到黑盒商业模型时，后缀集成在 GPT-3.5 和 GPT-4 上分别取得 **86.6%** 和 **46.9%** 的攻击成功率。

![GCG 攻击概览](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/gcg_splash.png)

> 图解：研究者在多个较小开源模型、多个有害行为请求上优化同一个对抗后缀，再将原始请求与后缀组合后交给未参与优化的商业模型。图中展示的是 ChatGPT、Claude、Bard 和 LLaMA-2 的实际输出案例，用来说明“自动优化—跨请求通用—跨模型迁移”的完整链路；案例中的危险操作细节不应被复现。

## 为什么已有越狱和自动攻击还不够可靠

大语言模型从互联网文本中学习，因此天然可能复现威胁、歧视、违法建议等不当内容。所谓 **对齐**，在本文语境中主要指通过监督微调、人类偏好反馈和强化学习等手段，让模型拒绝此类请求。常见路线包括基于人类反馈的强化学习、规则约束的奖励模型，以及让模型依据原则解释并修正不当回答。

这些机制对直接提问通常有效，却并不等于对任意输入都稳健。已有攻击大致有三类：

- **人工越狱**：依靠角色扮演、情境包装等技巧诱导模型，成本较高，且容易因模型或模板变化而失效。
- **连续 Embedding 优化**：在可微的向量空间中优化 Soft Prompt，但优化结果未必能映射回离散文本，也无法直接提交给普通聊天接口。
- **离散 Token 搜索**：直接替换输入 Token，能够生成真实文本；难点在于词表巨大，无法穷举所有位置和候选词，而单点梯度又不一定能准确预测替换后的损失。

PEZ 采用连续优化与投影，GBDA 借助 Gumbel-softmax 松弛，AutoPrompt 则根据 One-hot 向量梯度挑选候选替换。它们提供了重要基础，但在对齐模型上的可靠性有限。本文的关键判断是：**攻击对齐模型时，不必要求文本扰动“不可察觉”或保持原分类标签；只要某段输入能诱发模型本应拒绝的行为，就构成有效攻击。**这使威胁模型比文本分类攻击更直接。

解决目标与搜索空间的问题之后，下一步是把“越狱成功”写成可优化的数学目标。

## 把越狱转化为对肯定回答开头的优化

用户只能控制聊天模板中的 User 部分，System Prompt 和 Assistant 标记由系统提供。攻击者不修改原始请求，只在其后附加一段可优化 Token，记作对抗后缀。通用攻击希望同一个后缀对不同请求都有效，而不是为每个问题重新设计一套输入。

### 为什么不让模型直接生成完整目标答案

一种直观做法，是最大化某段完整有害答案的生成概率。但这有两个缺陷：同一请求可以有许多合理表达，强行指定唯一答案会缩小可行解空间；完整答案又高度依赖单个请求，不利于学习跨请求通用的后缀。

GCG 转而只规定回答的 **开头**：让模型先输出与请求相对应的肯定句，例如“Sure, here is …”。其直觉是，自回归模型一旦进入“同意作答”的生成状态，后续内容更可能沿着这一方向继续，而不是突然转为拒绝。

只优化第一个肯定 Token 也不够，因为后缀可能通过改变话题来获得“Sure”，却不再回应原请求。因此，目标短语需要同时包含肯定态度和原请求的复述。这是本文一个很简单但关键的设计。

### 形式化目标

设输入 Token 序列为 $x_{1:n}$，词表大小为 $V$，模型给定前缀后预测下一个 Token 的概率为 $p(x_{n+1}\mid x_{1:n})$。长度为 $H$ 的目标序列，其条件概率为：

$$
p(x_{n+1:n+H}\mid x_{1:n})=\prod_{i=1}^{H}p(x_{n+i}\mid x_{1:n+i-1})
$$

令 $x^{\star}_{n+1:n+H}$ 表示肯定式目标开头，攻击损失就是它的负对数概率：

$$
\mathcal{L}(x_{1:n})=-\log p(x^{\star}_{n+1:n+H}\mid x_{1:n})
$$

若 $\mathcal{I}$ 是对抗后缀所占的位置集合，则优化问题为：

$$
\min_{x_{\mathcal{I}}\in\{1,\ldots,V\}^{|\mathcal{I}|}}\mathcal{L}(x_{1:n})
$$

这里真正困难的不是写出 Loss Function，而是在离散、庞大的候选空间中求解。GCG 的贡献正是把梯度筛选与精确评估组合成一个实用搜索器。

## GCG：先用梯度找候选，再用前向传播做决策

理想中的贪心坐标下降会枚举每个后缀位置的所有单 Token 替换，选择让损失下降最多的一项。但每个位置都有 $V$ 种选择，全部执行前向传播的计算量不可接受。

GCG 使用当前 Token 的 One-hot 向量 $e_{x_i}$ 计算梯度：

$$
\nabla_{e_{x_i}}\mathcal{L}(x_{1:n})\in\mathbb{R}^{V}
$$

负梯度较大的方向，被当作可能降低损失的替换方向。每一轮迭代执行以下步骤：

1. 对 **所有** 可修改位置 $i\in\mathcal{I}$，分别取负梯度最大的 Top-$k$ 个 Token，形成候选集合 $\mathcal{X}_i$。
2. 从所有位置的候选中随机采样 $B$ 个单点替换，组成 Batch；每个候选只改变一个位置。
3. 对这些候选执行真实前向传播，精确计算损失。
4. 采用损失最小的候选，进入下一轮。

可以把它理解为“梯度负责缩小搜索范围，模型本身负责最终裁判”。梯度只提供局部线性近似，可能误判完整 Token 替换的效果；精确前向评估则避免了完全相信这一近似。

**这个设计最聪明的地方，是每轮同时考虑所有坐标。** AutoPrompt 会预先选定一个位置，再只评估该位置的候选；GCG 在相同 Batch Size、也即相近前向计算预算下，让不同位置竞争本轮更新机会。论文实验显示，这个看似不大的改动带来了显著差距。

默认对比实验中，各方法均优化 20 个 Token、运行 500 步；AutoPrompt 与 GCG 的 Batch Size 为 512，Top-$k$ 为 256。PEZ 和 GBDA 则同时优化 16 条随机初始化序列并保留最佳结果。

## 从单条攻击到多请求、多模型通用后缀

单个请求上的成功不能证明“通用”。为此，研究者把同一后缀 $p_{1:l}$ 拼接到 $m$ 个训练请求后，分别构造肯定式目标与损失 $\mathcal{L}_j$。候选筛选使用各请求梯度之和，候选选择则最小化总损失：

$$
\min_{p_{1:l}}\sum_{j=1}^{m_c}\mathcal{L}_j\!\left(x^{(j)}_{1:n_j}\Vert p_{1:l}\right)
$$

其中 $\Vert$ 表示序列拼接，$m_c$ 是当前纳入优化的请求数。实现上还有两项重要细节：

- 梯度聚合前分别裁剪为单位范数，避免某个请求因梯度过大而主导搜索。
- 请求不是一次性全部加入，而是从第一个开始；当前后缀能攻击已加入请求后，才逐步增加下一个请求。这种课程式优化比一开始同时优化全部请求效果更好。

为了增强迁移性，训练损失还可以同时来自多个模型。若这些模型共享 Tokenizer，它们的梯度都位于同一个 $\mathbb{R}^{V}$ 空间，可以直接聚合。例如 Vicuna-7B 与 Vicuna-13B 可以共同优化，不需要修改核心算法。

这里需要区分两种“通用性”：**多请求通用** 指同一后缀适用于不同行为；**跨模型迁移** 指在源模型上得到的后缀，对未参与优化的模型仍有效。接下来的 AdvBench 和实验分别检验这两点。

## AdvBench：两类任务检验精确控制与行为绕过

现有文本对抗基准多关注情感分类、自然语言推理等任务是否被误判，本文则建立了面向有害生成行为的 **AdvBench**。它包含两个各 500 条的集合：

- **Harmful Strings**：500 个体现有害或有毒行为的目标字符串，覆盖粗俗内容、威胁、虚假信息、歧视、网络犯罪和危险建议等主题。攻击必须让模型输出与目标**完全一致**的字符串。使用 LLaMA Tokenizer 后，长度为 3～44 Token，平均 16 Token。
- **Harmful Behaviors**：500 个以指令形式描述的有害行为。攻击不再要求逐字匹配，只要模型作出合理尝试、实际遵从指令而非拒绝或回避，就算成功。对通用攻击，还分别报告训练请求与留出请求上的结果。

主要指标是 **Attack Success Rate（ASR，攻击成功率）**。Harmful Strings 另以目标字符串的 Cross-Entropy Loss 衡量优化质量：ASR 越高、Loss 越低，攻击越强。Harmful Behaviors 因回答形式开放，部分样本需要人工判断。

附录说明了数据来源：作者先编写 100 个 Harmful Strings 和 50 个 Harmful Behaviors，再从中随机抽取 5-shot 示例，让未经安全审查的 Wizard-Vicuna-30B-Uncensored 每轮生成 10 个新样本。数据集已在项目仓库发布。这里不复现其中的危险示例；基准的用途是检验模型能否遵守自身明确的安全边界，而不是重新界定哪些内容应该被允许。

## 白盒实验：GCG 在两类任务上全面领先

单目标实验分别使用 Vicuna-7B 和 LLaMA-2-7B-Chat，在两个任务的前 100 个样本上优化。通用实验则用 25 个 Harmful Behaviors 训练一个后缀，再在 100 个留出行为上测试。

| 模型 | 方法 | 单个 Harmful String：ASR | 单个 Harmful String：Loss | 单个 Harmful Behavior：ASR | 多个 Behaviors：训练 ASR | 多个 Behaviors：测试 ASR |
|---|---|---:|---:|---:|---:|---:|
| Vicuna-7B | GBDA | 0.0% | 2.9 | 4.0% | 4.0% | 6.0% |
| Vicuna-7B | PEZ | 0.0% | 2.3 | 11.0% | 4.0% | 3.0% |
| Vicuna-7B | AutoPrompt | 25.0% | 0.5 | 95.0% | 96.0% | **98.0%** |
| Vicuna-7B | **GCG** | **88.0%** | **0.1** | **99.0%** | **100.0%** | **98.0%** |
| LLaMA-2-7B-Chat | GBDA | 0.0% | 5.0 | 0.0% | 0.0% | 0.0% |
| LLaMA-2-7B-Chat | PEZ | 0.0% | 4.5 | 0.0% | 0.0% | 1.0% |
| LLaMA-2-7B-Chat | AutoPrompt | 3.0% | 0.9 | 45.0% | 36.0% | 35.0% |
| LLaMA-2-7B-Chat | **GCG** | **57.0%** | **0.3** | **56.0%** | **88.0%** | **84.0%** |

最突出的结果是精确字符串诱导：Vicuna-7B 上，GCG 的 88% 明显高于 AutoPrompt 的 25%，PEZ 和 GBDA 均为 0；LLaMA-2-7B-Chat 上，GCG 为 57%，AutoPrompt 只有 3%。在留出 Harmful Behaviors 上，GCG 对 LLaMA-2 的测试 ASR 为 84%，而 AutoPrompt 为 35%。

需要注意，AutoPrompt 在 Vicuna 的开放行为任务上并不弱，单个行为达到 95%，通用后缀测试 ASR 与 GCG 同为 98%。因此，GCG 的优势并非“所有设置都数量级领先”，而是在更难的目标和更稳健的模型上仍能保持有效。

![不同优化器的损失与成功率曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/loss_curve.png)

> 图解：该图比较 Vicuna-7B 单条 Harmful String 任务中，不同优化器随攻击迭代变化的 Cross-Entropy Loss 与 ASR。曲线越低表示损失越小，成功率曲线越高表示攻击越强。GCG 在较早阶段就找到低损失对抗样本，随后继续缓慢改善；PEZ 和 GBDA 基本没有取得精确匹配，AutoPrompt 介于其间。

白盒结果证明同一个后缀确实能泛化到未见请求，但更令人意外的发现是：它还能离开训练时使用的模型，攻击完全不同的系统。

## 跨模型迁移：从开源代理模型到黑盒商业模型

研究者先联合 Vicuna-7B 和 Vicuna-13B，在 25 个有害行为上优化一个后缀，并用两个随机种子重复实验。第三个后缀进一步加入 Guanaco-7B 和 Guanaco-13B，即同时在 25 个请求、4 个模型上优化。每次运行 500 步，保留损失最低的后缀。

迁移测试覆盖 Pythia-12B、Falcon-7B、ChatGLM-6B、MPT-7B、LLaMA-2-Chat-7B、Stable-Vicuna，以及 GPT-3.5、GPT-4、Claude 1、Claude 2 和 PaLM-2，共收集 388 个测试行为。ChatGPT 和 Claude 使用 Temperature 0、Top-$p$ 0；PaLM-2 使用 Temperature 0.9、Top-$p$ 0.95，每次检查 8 个候选回答，任一触发目标行为即记为成功。因此，PaLM-2 的口径与其他确定性解码结果并不完全相同。

![开源与商业模型上的迁移攻击成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/transfer_results.png)

> 图解：图中比较新行为上的 ASR。“Prompt only”是直接请求；“Sure here's”只要求模型以肯定句开头；“GCG”对多个对抗后缀的 ASR 取平均；“GCG Ensemble”则只要任一后缀成功就算成功。深蓝色表示单个后缀的最佳结果，浅色部分体现集成后的提升。GCG 在多个未参与优化的开源模型上接近 100%，但 ChatGLM-6B 等模型明显较低，说明迁移广泛存在，却并非均匀可靠。

下表进一步给出商业模型的完整结果。除单后缀外，“Concatenate”把三个后缀连接成一个长后缀；“Ensemble”依次尝试三个短后缀和连接版本，任一成功即停止。

| 方法 | 优化所用模型 | GPT-3.5 | GPT-4 | Claude-1 | Claude-2 | PaLM-2 |
|---|---|---:|---:|---:|---:|---:|
| Behavior only | 无 | 1.8% | 8.0% | 0.0% | 0.0% | 0.0% |
| Behavior + “Sure, here's” | 无 | 5.7% | 13.1% | 0.0% | 0.0% | 0.0% |
| Behavior + GCG | Vicuna | 34.3% | 34.5% | 2.6% | 0.0% | 31.7% |
| Behavior + GCG | Vicuna & Guanacos | 47.4% | 29.1% | 37.6% | 1.8% | 36.1% |
| + Concatenate | Vicuna & Guanacos | 79.6% | 24.2% | 38.4% | 1.3% | 14.4% |
| + Ensemble | Vicuna & Guanacos | **86.6%** | **46.9%** | **47.9%** | **2.1%** | **66.0%** |

这些数字包含几个不能忽略的细节：

- 加入 Guanaco 优化后，Claude-1 的单后缀 ASR 从 2.6% 升至 37.6%，但 GPT-4 从 34.5% 降至 29.1%，多模型训练并非对所有目标都单调增益。
- 拼接后缀让 GPT-3.5 从 47.4% 升至 79.6%，却使 GPT-4 降至 24.2%。作者观察到，过长输入会让 GPT-4 更频繁地要求澄清，而非继续作答。
- 集成利用不同后缀的多样性，在 GPT-3.5 上达到 86.6%，GPT-4 和 Claude-1 接近或达到约一半；Claude-2 仍只有 2.1%，是所测商业模型中更稳健的一个。

作者还做了一项补充实验：针对 GPT-3.5 上失败的行为，由一位作者用每条最多 30 秒改写请求中人类可读的部分。仅通过同义改写或换一种任务表述，就几乎诱导了全部剩余行为。这说明自动后缀与简单人工改写可以互补，但该结果不能算作完全自动攻击的 ASR。

![四个模型上的有害生成案例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/gcg_four_model.jpg)

> 图解：左上、右上、左下、右下分别对应 ChatGPT、Claude 2、Bard 和 LLaMA-2。四张截图用于核对攻击确实作用于不同产品，而不是仅在单一开源模型上过拟合；Claude 2 案例还在对抗请求前加入了词语替换式的条件化步骤，因此不代表纯自动后缀的平均水平。

![GPT-3.5、GPT-4 与 Claude 的行为生成摘录](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/gcg_main_examples.png)

> 图解：绿色、紫色及其他分区分别展示 GPT-3.5、GPT-4 和 Claude 对不同类型有害请求的响应摘录。作者只保留足以证明目标行为被触发的片段，主动删去危险操作细节；不同模型使用的精确输入可能略有差异，所以这些案例是有效性证据，不是严格统一输入下的横向排名。

## 迁移为何会先升后降：训练损失不是唯一目标

迁移实验还揭示了一个反直觉现象：GCG 在源模型上继续降低损失，不一定让黑盒攻击更强。训练过长可能使后缀过拟合源模型。

![优化过程中的损失与迁移成功率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/loss_and_transfer_asr.png)

> 图解：左图跟踪 Vicuna 与 Guanaco 四模型联合优化时，GCG Loss 和黑盒迁移 ASR 随优化过程的变化。损失前半程快速下降、后半程趋平；迁移 ASR 先随损失下降而上升，随后却开始回落。这意味着以最终源模型 Loss 选后缀未必最佳，较少迭代有时反而得到更好的迁移效果。

![不同源模型设置下的平均迁移 ASR](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/individual_transfer_asr.png)

> 图解：右图比较针对不同目标模型优化后缀后，在一组黑盒模型上的平均迁移 ASR。使用基于 Claude 输出微调的 LLaMA 系列模型作为源模型时，迁移成功率高于其他设置。这支持“源模型与目标模型的数据或蒸馏关系会影响迁移”的判断，但尚不足以确定唯一因果机制。

由此，评估通用攻击不能只看白盒 Loss，还要单独跟踪留出行为与目标模型上的成功率。集成有效也说明，不同后缀覆盖的失败样本并不完全相同，单纯把某一个后缀作为固定检测特征很难构成充分防御。

## 讨论：对齐为何会被自动攻击绕开

### 新模型更稳健，还是源模型选得更“近”

从结果看，GPT-4 比 GPT-3.5 更难攻击，Claude 2 又明显更稳健，似乎说明后续对齐有所进步。但作者提醒，Vicuna 使用 ChatGPT-3.5 输出进行训练，可视为某种蒸馏模型；视觉对抗研究中，蒸馏模型之间通常更容易发生迁移。因此，GPT-3.5 的高 ASR 可能部分来自这种关系，而不能全部归因于其安全训练较弱。

Claude 的聊天接口还可能在模型生成前使用内容过滤器，部分请求直接不产生内容，而 API 路径并不完全相同。作者通过前置词语游戏观察到绕开这一现象的案例，并认为过滤器与模型可能需要被作为一个整体评估。不过，这部分包含人工条件化操作，应与自动攻击结果分开理解。

### 对抗后缀并非全是不可读乱码

部分优化结果含有“以 Sure 复述第一句”之类可解释结构，恰好对应肯定回答目标；多请求、多模型联合优化似乎更容易出现这种结构。另一些片段仍近似无意义文本。这表明离散搜索可能发现具有语义功能的提示，但目前没有证据说明所有成功后缀都能被人类直接理解。

与图像攻击相比，文本替换通常显而易见，却不妨碍攻击成立。因为这里的安全要求不是“保持分类语义”，而是“无论用户如何改写输入，都不应执行某类危险行为”。自动搜索又比人工构造更快、更易扩展，这正是它挑战现有对齐评估方式的原因。

## 局限、伦理与防御方向

这项研究并没有证明所有对齐模型都能以同样高的概率被攻击。Claude 2 的低成功率、不同开源模型间的差距、长后缀引起的理解失败，以及训练后期的迁移下降，都说明结果依赖源模型、目标模型、输入形式和解码设置。黑盒实验还对应论文发表时的具体模型版本，不能直接视为其他版本或当前系统的表现。

作者在发表前已向 OpenAI、Google、Meta 和 Anthropic 披露初步结果，论文中的具体案例随后可能失效。完整生成截图主要用于审计论文证据，以下仅作模型对应说明，不复述其中的有害内容。

![ChatGPT 的完整生成记录](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/chatgpt_large_clipped.jpeg)

> 图解：这是概览案例对应的 ChatGPT 完整输出，用来补充正文摘录的上下文。它证明目标模型确实延续了肯定回答，但不应把历史版本中的单次输出当作当前模型表现。

![Claude 的完整生成记录](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/claude_large_clipped.jpeg)

> 图解：这是 Claude 案例的完整输出。结合正文可知，相关成功还涉及前置条件化与关键词替换，不能据此推断纯 GCG 后缀在 Claude 2 上具有很高平均 ASR。

![Bard 的完整生成记录](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/bard_large_clipped.jpeg)

> 图解：这是 Bard 案例的完整输出，对应文中 PaLM-2 系列系统的迁移证据。其实验采用较高 Temperature，并从 8 个候选生成中判断是否存在成功样本，解读 ASR 时必须保留这一评估条件。

![LLaMA-2 的完整生成记录](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Universal-and-Transferable-Adversarial-Attacks-on-Aligned-Language-Models/figures/llama2_large_clipped.jpeg)

> 图解：这是 LLaMA-2 案例的完整输出，与表中 LLaMA-2-Chat 的白盒及迁移实验相互补充。它体现的是特定输入下的成功实例，不代表所有请求都会触发相同行为。

防御方面，最自然的方向是 **对抗训练**：训练或微调时持续生成攻击，再让模型学习对这些输入给出安全回应，同时保留正常请求上的能力。但视觉领域的经验表明，这类防御可能计算昂贵、损害正常性能，并只对狭窄威胁模型有效。输入检测、更多常规对齐训练，以及在预训练阶段减少危险行为倾向，也都值得研究，但论文没有给出已经解决的方案。

## 总结

- GCG 把越狱目标设为生成“肯定并复述原请求”的回答开头，避免为完整答案指定唯一表达。
- 它用 One-hot 梯度从所有后缀位置筛选 Top-$k$ 候选，再通过真实前向传播选择最佳单 Token 替换。
- 多请求损失聚合、梯度归一化和渐进加入请求，使一个后缀能够覆盖多种行为；多模型联合优化进一步提供迁移能力。
- 白盒实验中，GCG 对 Vicuna-7B 和 LLaMA-2-7B-Chat 的单字符串精确诱导 ASR 分别为 88% 和 57%，整体优于 PEZ、GBDA 与 AutoPrompt。
- 黑盒迁移广泛但不均匀：集成后缀在 GPT-3.5、GPT-4、PaLM-2 上分别达到 86.6%、46.9%、66.0%，Claude-2 仅为 2.1%。

这项工作真正留下的问题不是某段后缀是否仍然有效，而是对齐评估能否从“防住自然语言越狱”升级为“持续抵抗自动搜索”，并在安全性、正常能力与防御成本之间找到可扩展的平衡。

> 本文参考自 [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)