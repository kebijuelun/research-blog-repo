# CLIPO：把对比学习引入 RLVR，为什么能让推理模型更稳、更泛化

## 一句话结论
这篇工作提出了 CLIPO（Contrastive Learning in Policy Optimization）：在传统 RLVR（Reinforcement Learning with Verifiable Rewards）只关注“最终是否答对”的基础上，加入“轨迹级对比学习奖励”。核心收益是：让模型不只学会“碰巧答对”，而是更倾向于学习多条正确推理路径的共性结构，从而减少 hallucination 与答案抄写式投机，提升分布外泛化与鲁棒性。

## 1. 论文要解决的痛点

RLVR 类方法（GRPO/GSPO/DAPO/GMPO）在数学与推理任务上很有效，但存在一个结构性问题：

- 奖励几乎只看最终答案对错（0/1）。
- 中间推理步骤即使错误、跳步或出现幻觉，只要最后碰巧答对，也会被当成“正样本”。
- 这会导致训练信号粗糙：模型学到的可能是结果模式匹配，而不是稳定推理逻辑。

作者的观察很关键：对于同一道题，多条真正成功的推理轨迹通常共享某种“隐含逻辑骨架”；而错误步骤更像随机噪声。  
因此，可以用对比学习把“成功轨迹彼此拉近、与失败轨迹拉远”，将这种骨架显式注入策略优化过程。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/CLIPO-Contrastive-Learning-in-Policy-Optimization-Generalizes-RLVR/figures/intro2.png)

> 图解：这张图不是坐标图，而是机制示意图。左侧代表传统 RLVR 只按结果打分；右侧是 CLIPO，把多个成功轨迹在表示空间对齐，提取它们重叠的稳定推理结构，并压制不一致的噪声步骤。

## 2. 方法核心：从“结果奖励”到“结果 + 结构奖励”

### 2.1 基础 RLVR 目标
传统目标可以写成：

$$
\max_{\theta} \; \mathbb{E}_{x, y \sim \pi_{\theta}(\cdot|x)}[r(x,y)] - \beta \, \mathrm{KL}(\pi_{\theta}\|\pi_{\text{ref}})
$$

其中 $r(x,y)\in\{0,1\}$ 来自可验证器（例如数学答案校验）。

### 2.2 CLIPO 的额外约束：最大化成功轨迹间互信息
作者将目标扩展为：

$$
\max \; \mathbb{E}[r(x,y)] + \lambda \, I(y,\bar y \mid x, r(x,y)=1, r(x,\bar y)=1)
$$

直觉是：同题下两条都答对的轨迹 $y,\bar y$，应该在语义表示上更接近。

### 2.3 用 InfoNCE 近似互信息下界
实际中无法直接计算互信息，于是在每个 rollout group 内做 InfoNCE。  
给定组内成功集合 $P$，每个成功轨迹采样一个成功正例，其余作为对比集合：

$$
\mathcal{L}_{\text{CL}} = - \log \frac{\exp(f(y_i,\bar y_i))}{\sum_{j=1}^{G}\exp(f(y_i,y_j))}
$$

相似度函数为：

$$
f(y,\bar y)=\frac{g_{\phi}(h_{\theta}(y))^\top g_{\phi}(h_{\theta}(\bar y))}{\tau}
$$

- $h_{\theta}(y)$：策略模型最后隐层做均值池化得到句级表示。
- $g_{\phi}$：轻量线性 contrastive head。
- $\tau$：温度系数。

### 2.4 如何并入 RL 训练
将对比损失转成额外奖励：

$$
r_i' = r_i + r_i^{\text{CL}}, \quad
r_i^{\text{CL}} = \max(-\lambda \mathcal{L}_{\text{CL}}, -0.5)
$$

这一步很实用：能够保证对比信号不会压过主任务的 verifiable reward。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/CLIPO-Contrastive-Learning-in-Policy-Optimization-Generalizes-RLVR/figures/framework5.png)

> 图解：输入同一问题后，策略模型采样一组轨迹；一条支路走普通 RLVR 奖励，另一条支路通过 contrastive head 计算轨迹嵌入并得到对比奖励；两者相加后再做策略更新。流程上几乎是可插拔增强。

## 3. 实验设计与主要结果

作者设计了两条实验线：

- Track I：在 GSM8K 上训练，评估 8 个数据集（含 GSM-Symbolic、P1/P2、CommonsenseQA、TruthfulQA、TheoremQA、MMLU）。
- Track II：在 MATH 7.5K 上训练，评估 MATH/Math-Perturb/AMC/AIME/AIME25。

对比基线为：GRPO、GSPO、DAPO、GMPO。

### 3.1 主结果（核心结论）
- Track I 上，GRPO + CLIPO 的总体 Avg 达到 63.26，相比 GRPO 提升 +1.12。
- Track II 上，四种基线加入 CLIPO 后均有提升：
  - GRPO：+1.35
  - GSPO：+0.80
  - DAPO：+1.20
  - GMPO：+0.83
- 在扰动集和符号推理集（P1/P2、Math-Perturb）提升更明显，说明模型并非只会“刷熟题”，而是更鲁棒。

### 3.2 跨基座模型泛化
在 DeepSeek-R1-Distill-Qwen-7B 与 Llama3.1-8B 上同样有效：

- DS-7B：Avg +0.53
- Llama-8B：Avg +1.31

这说明 CLIPO 不是绑定某个 backbone 的技巧，而更接近一种训练范式增强项。

## 4. 消融分析：哪些设计真正有效

### 4.1 Contrastive head 必须训练
冻结 head（CLIPO-fixed）会稳定退化。  
Track I/II 的 Avg 分别下降约 0.77/0.97，说明 head 学到的几何结构是有效信息，而非可有可无。

### 4.2 损失函数对比：InfoNCE 最稳
作者比较了 InfoNCE、SupCon、SoftNN。三者都比纯 GRPO 有收益，但 InfoNCE 综合最稳、均值提升最好。

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/CLIPO-Contrastive-Learning-in-Policy-Optimization-Generalizes-RLVR/figures/more_contrastiveloss.png)

> 图解：横轴为不同 contrastive loss 方案，纵轴为相对基线的性能增益。可见 InfoNCE 的整体增益最高，SupCon 次之，SoftNN 在部分子任务上波动更大。

### 4.3 温度系数与组大小
- 温度 $\tau$ 偏低（如 0.02/0.05）整体更优，过高会让正样本聚合不稳定。
- 组大小增大通常更好（尤其在竞赛数学指标上），因为正负样本对比更丰富。

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/CLIPO-Contrastive-Learning-in-Policy-Optimization-Generalizes-RLVR/figures/embeddings_temperature-2.png)

> 图解：横轴是训练步数（或训练进程），纵轴是组内正样本平均余弦相似度。高温度曲线波动更大，说明模型更难稳定拉近成功轨迹表征。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/CLIPO-Contrastive-Learning-in-Policy-Optimization-Generalizes-RLVR/figures/tsne-group-label1.png)

> 图解：这是 t-SNE 投影图。训练前正确/错误轨迹混杂，训练后正确轨迹更聚簇、错误轨迹更分离，说明对比奖励确实在重塑轨迹语义空间。

## 5. 对复现最有价值的实现细节

- Contrastive head：线性层 + $L_2$ 归一化。
- 输出维度：Track I 用 512，Track II 用 2048。
- head 优化器：AdamW，学习率 $1\times10^{-3}$，weight decay 0.01。
- rollout 数：默认每题 16 条。
- 仅在非退化组计算对比损失：$1<|P|<G$。
- 默认加权系数：$\lambda=0.2$（InfoNCE/SupCon）。
- 奖励裁剪：下限 -0.5，避免辅助奖励过强。

## 6. 评价与启发

这篇论文最大的价值不在于“又一个 RL 小改动”，而在于把 RLVR 中长期存在的信号粗糙问题，转化为一个自然的表示学习问题：  
从“答对就行”升级为“答对且要像正确推理”。  
它不依赖昂贵的 PRM 人工过程标注，工程接入成本低，还能与多种 group-based policy optimization 兼容，实用性很强。

如果后续将这套轨迹对齐机制扩展到代码、Agent 规划、多步工具调用场景，其潜力可能更大。

> 本文参考自 [CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR](https://arxiv.org/abs/2603.10101)