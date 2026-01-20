# Pre-training Distillation for Large Language Models：设计空间全景解读

## 这篇论文在解决什么问题
大模型知识蒸馏（Knowledge Distillation, KD）过去多用于 **后训练** 阶段：学生模型学老师生成的指令-回复。但这篇工作把 KD 前移到 **预训练阶段**，提出 **Pre-training Distillation (PD)**。核心问题是：预训练阶段用老师 logits 监督学生，能否系统性提升？哪些设计因素最关键？

作者给出的四大设计维度：
- **Logits 处理**：截断与温度归一化
- **Loss 选择**：KD Loss 与 LM Loss 的组合方式
- **Scaling Law**：学生/老师规模、预训练语料规模
- **Offline vs Online**：logits 是预训练后离线生成，还是训练中同步生成

---

## 预训练蒸馏的数学形式
论文把 PD 的目标写成经典加权形式：

$$
\theta_S^* = \arg\min_{\theta_S} \mathcal{L} = \arg\min_{\theta_S} \big[(1-\alpha)\mathcal{L}_{\text{lm}} + \alpha \mathcal{L}_{\text{kd}}\big]
$$

其中：
- $\mathcal{L}_{\text{lm}}$ 是传统 LM loss  
- $\mathcal{L}_{\text{kd}}$ 是蒸馏 loss  
- $\alpha$ 控制 KD 与 LM 的比例

教师 logits 需要 **截断 + 温度归一化**：

$$
F(\mathbf{z}) = \text{softmax}\Big(\frac{\text{Truncate}(\mathbf{z})}{\tau}\Big)
$$

---

## 核心实验：预训练蒸馏是否有效？
作者用 GLM-4-9B 做老师，训练 1.9B 学生模型，结果显示：
- LLM-KD 平均优于 LLM-LM  
- 增益不大但稳定，证明 PD 可行

---

## 设计维度一：Logits 处理

### 1) Top-p-k 截断
论文提出 **先 top-p，再 top-k** 的两阶段截断，大幅节省存储。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Pre-training-Distillation-for-Large-Language-Models-A-Design-Space-Exploration/figs/topp.png)
> 图解：横轴是不同 $p$，纵轴是相对提升；右侧曲线展示 logits 存储规模随 $p$ 变化。整体趋势是性能差异不大，但存储成本显著下降。

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Pre-training-Distillation-for-Large-Language-Models-A-Design-Space-Exploration/figs/topk.png)
> 图解：横轴是不同 $k$，纵轴是相对提升；右侧曲线显示 logits 规模随 $k$ 增长。$k=50$ 表现最好，说明保留少量高置信 token 已足够有效。

**关键结论**：性能对 $p,k$ 不敏感，小 $p,k$ 可以显著省存储。

---

### 2) 温度 $\tau$
低温度 → logits 更尖锐；高温度 → 更平滑。实验表明：
- $\tau \le 2.0$ 效果较稳  
- $\tau$ 太高效果变差

核心结论：**温度不宜过高，适度平滑即可**。

---

## 设计维度二：Loss 选择

### 1) KD Loss 类型
对比 NLL / KLD / MSE：
- **NLL 与 KLD 表现相近且优于 LM**
- **MSE 明显退化**

### 2) LM Loss 与 KD Loss 的组合方式
如果只用 KD loss，效果不错，但最佳实践是 **LM Loss + KD Loss 混合**。

静态 $\alpha$ 实验显示：
- **$\alpha=0.9$ 最优**  
- 说明保留少量 LM loss 有益

同时作者探索动态调度，效果最优的是：

**WSD-$\alpha$ + WSD-LR**

也就是 KD 比例与学习率都采用 Warmup-Stable-Decay 策略。

---

## 设计维度三：Scaling Law

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Pre-training-Distillation-for-Large-Language-Models-A-Design-Space-Exploration/figs/model_size.png)
> 图解：横轴是学生/老师规模组合，纵轴是相对提升。趋势明显：学生越大，蒸馏收益越大；老师越大并不一定更好。

关键发现：
- 学生越大 → PD 增益越明显  
- 老师更大并不保证更好，可能因为容量差距过大

---

### 语料规模的影响
作者用 500B tokens 继续训练，发现：
- PD 的提升在长训练中持续存在  
- 增益在前期上升，后期趋于平稳但仍保留优势

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Pre-training-Distillation-for-Large-Language-Models-A-Design-Space-Exploration/figs/corpus_size.png)
> 图解：横轴为训练步数（每 10k step），纵轴是多任务平均性能。PD 在整个训练周期都优于 LM baseline。

---

## 设计维度四：Offline vs Online

在线蒸馏会在老师模型预训练过程中同步保存 logits。理论上更省推理成本，但实验结果显示：
- 早期 logits 噪声大，效果明显差  
- 后期 logits 质量好，效果提升但仍弱于 offline

结论：
- **只训一个模型 → offline 更稳**
- **训一系列模型 → online 更有性价比**

---

## 最佳配置总结（PD*）

作者最终给出一套更优配置：
- top-$0.95$-$50$ 截断  
- $\tau=2.0$  
- KLD loss  
- WSD-$\alpha$ + WSD-LR  
- Offline logits  
- Teacher = GLM-4-9B  

效果显著优于 vanilla PD。

![Figure 5](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Pre-training-Distillation-for-Large-Language-Models-A-Design-Space-Exploration/figs/figure1.png)
> 图解：对比 LM-only、vanilla PD 与优化后的 PD*。PD* 在多规模学生模型上均显著提升。

---

## 限制与展望
论文限制主要在于 **未能穷尽不同因素组合**，因为计算成本太高。但当前结果足以形成实践指导：

- PD 确实可行  
- Logits 处理成本可控  
- Loss 设计最关键  
- 学生越大收益越高  

---

> 本文参考自 [Pre-training Distillation for Large Language Models: A Design Space Exploration](https://arxiv.org/pdf/2410.16215)