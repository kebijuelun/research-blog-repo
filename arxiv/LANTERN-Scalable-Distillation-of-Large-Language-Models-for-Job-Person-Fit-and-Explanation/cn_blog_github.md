# LANTERN：大模型在求职匹配中的可扩展蒸馏框架与解释生成

## 一句话看懂这篇论文
这篇论文解决的是 **“如何在真实招聘平台上，用低延迟模型输出高质量的岗位匹配结果与解释”** 。核心思路是用强大的黑盒 LLM 生成高质量监督，再通过多层次蒸馏把知识迁移到两类轻量模型：一个做匹配评分，一个做解释生成。

## 背景与问题动机：为什么需要 LANTERN
在求职场景里，模型不只是给一个“匹配/不匹配”的标签，而是要输出：
- 匹配度评分（High/Medium/Low 或 $[0,1]$）
- 可解释的逐条理由

直接用开源 LLM 或微调的大模型会遇到两类现实问题：
- **规模与延迟** ：大模型推理成本高，线上无法高频调用。
- **结构化输出质量不稳定** ：需要“评分 + 解释”的双输出格式，普通 LLM 很难稳定生成。

LANTERN 的目标是把“高质量 + 低延迟”统一起来。

## 任务定义与双模型设计
给定成员 $m$ 和岗位 $j$，输出：
- 评分 $r_i \in [0,1]$
- 解释 $exp_i$

平台侧的访问量特点很关键：评分请求量约是解释的 30 倍，因此系统采用 **双模型架构** ：
- $\text{LM}_{cls}$：只负责评分，追求高吞吐
- $\text{LM}_{exp}$：负责解释，追求表达质量

## 关键框架：LANTERN 的两阶段蒸馏

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/LANTERN-Scalable-Distillation-of-Large-Language-Models-for-Job-Person-Fit-and-Explanation/figures/structure.png)
> 图解：左侧是 **黑盒教师模型 + 人工筛选** 生成高质量种子数据；右侧是 **双学生模型蒸馏** ，分别学习分类与解释任务。

### 阶段 1：构建强教师模型
1. 用黑盒 LLM（如 GPT-4o）生成匹配评分和解释  
2. 人工筛选高质量样本，得到 $D_{\text{seed}}$  
3. 微调内部教师模型 $T_2$  

### 阶段 2：多目标知识蒸馏
蒸馏目标分为两类：
- **解释模型蒸馏（生成任务）**
- **分类模型蒸馏（判别任务）**

## 解释模型蒸馏：从生成质量到可控损失

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/LANTERN-Scalable-Distillation-of-Large-Language-Models-for-Job-Person-Fit-and-Explanation/figures/lantern-distill.png)
> 图解：教师模型输出作为 “软监督”，学生模型在 token 级别对齐分布，从而提升解释质量。

解释模型的训练损失由两部分构成：

$$
\ell_{\text{sft}} = - \sum_{t=1}^{T} \log \pi_s(y_{i,t} \mid y_{i,<t}, x_i)
$$

$$
\ell_{\text{kd}} = \sum_{t=1}^{T} d\left( \pi_t(\cdot \mid y_{i,<t}, x_i), \pi_s(\cdot \mid y_{i,<t}, x_i) \right)
$$

最终目标：

$$
\ell_{\text{exp}} = \lambda_{\text{sft}} \cdot \ell_{\text{sft}} + \lambda_{\text{kd}} \cdot \ell_{\text{kd}}
$$

论文对比了多种 distillation divergence：
- JS
- FKL
- TVD
- SKL

实验显示 **TVD 的 ROUGE 最优** ，生成质量更高。

## 分类模型蒸馏：高吞吐评分类别预测

分类模型输入为 $(job, profile)$，输出 fit label（low/medium/high）：

$$
h = f_{\text{LLM}}(x)
$$

$$
\hat{z} = softmax(g_{\text{MLP}}(h))
$$

$$
\ell_{\text{cls}} = -\sum_{c=1}^C z_c \log \hat{z}_c
$$

通过大规模合成标注数据蒸馏后，轻量模型能稳定输出高质量评分。

## Prompt 设计：为何不直接喂一个长 prompt
作者发现长 prompt 容易不稳定，于是把任务拆解成子任务：
- 抽取岗位要求
- 评估候选人匹配度
- 生成简洁解释

这种分解式 prompt 能减少 token 负担，提高一致性。

## 系统优化：线上效率来自两点
### 1. Job 描述压缩
只保留 **岗位要求** ，其余福利、背景信息去掉，节约 80% 输入长度。

### 2. 分类模型降级
从 1.5B encoder 降级到 0.4B ModernBERT，性能几乎不降，吞吐提升 4 倍。

## 实验结果一览

![Figure 3](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/LANTERN-Scalable-Distillation-of-Large-Language-Models-for-Job-Person-Fit-and-Explanation/figures/task.png)
> 图解：用户看到 fit label 后可点击解释；用户行为反馈被用作隐式监督信号。

### 解释任务（ROUGE）
- TVD 在 ROUGE-1/2/L 上均为最佳
- 多阶段蒸馏明显优于单阶段  
例如 7B → 1.5B → 0.5B 比 7B → 0.5B 明显更稳

### 分类任务
- SeqCls + Last token pooling 最优
- gte-Qwen2.5-1.5B-instruct 明显优于 Phi3-mini

### 线上指标
- Apply rate +0.24%
- Qualified applications +0.28%

![Figure 4](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/LANTERN-Scalable-Distillation-of-Large-Language-Models-for-Job-Person-Fit-and-Explanation/figures/job-match-job-board.png)
> 图解：模型在岗位详情页实时展示“匹配等级”，点击 “Show Reason” 触发解释模型输出。

## LANTERN 的核心创新点总结
1. **双模型架构** ：评分与解释分离，符合请求分布  
2. **多层次蒸馏** ：数据蒸馏 + logit 蒸馏协同  
3. **Prompt 分段设计** ：降低输出不稳定性  
4. **工程侧优化** ：输入压缩 + 模型降级  
5. **在线效果验证** ：不仅离线指标好，线上指标也有提升  

## 结论
LANTERN 的价值并不只是“蒸馏一个小模型”，而是构建了一个 **可规模化的 LLM 落地框架** ：从数据生成、教师模型训练，到蒸馏与线上部署都有完整闭环。对于任何需要“评分 + 解释”输出、且请求量极大的场景，这套设计几乎是可复用模板。

> 本文参考自 [LANTERN: Scalable Distillation of Large Language Models for Job-Person Fit and Explanation](https://arxiv.org/pdf/2510.05490v1)