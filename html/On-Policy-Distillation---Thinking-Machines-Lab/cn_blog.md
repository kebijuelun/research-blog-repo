# On-Policy Distillation：把“在自己犯错中学”做成高效蒸馏

## 一句话先讲清楚
这篇文章讨论的是 **on-policy distillation** ：让学生模型自己生成轨迹，再用强教师模型逐 token 打分，既保留 on-policy 的“贴合自身分布”，又有蒸馏的“密集监督”，在效率和效果之间取得更好的平衡。

---

## 背景：为何要在后训练阶段“动刀子”？
大模型能力通常分为三层训练栈：

- **Pre-training** ：语言与世界知识的通用能力  
- **Mid-training** ：领域知识（代码、医疗、企业文档等）  
- **Post-training** ：指令跟随、推理、对话等“行为”  

很多时候，小模型在特定领域里能 **打赢更大的通用模型** 。小模型的优势很实在：

- 本地部署，隐私更好  
- 更容易持续更新  
- 推理成本更低  

但关键是： **后训练怎么做** ，直接决定小模型是否能真正“变强”。

---

## On-policy vs Off-policy：两种训练范式的张力

### Off-policy（典型：SFT/蒸馏）
- 学生学习教师给的“标准答案”  
- 好处：监督密集、收敛稳定  
- 缺点：学生学到的是 **教师走过的轨迹** ，不是自己会遇到的状态  
- 结果：长序列任务容易 **误差累积** ，还可能学到“风格而非准确性”  

### On-policy（典型：RL）
- 学生用自己的输出拿奖励  
- 好处：状态分布贴合自己  
- 缺点：奖励太稀疏，每条轨迹只给 **固定的 bit** 反馈  

---

## 目标：兼得“贴合分布”与“密集反馈”
作者的直觉很像下棋：

- RL = 自己对弈，输赢只在结局给反馈  
- Off-policy distillation = 看大师下棋，但自己永远下不出那些局面  

理想状态是：  
 **你自己下，老师逐步评分** 。

这就是 **on-policy distillation** 的核心。

---

## 核心方法：On-policy Distillation

核心思想：  
- 从 **学生模型** 采样轨迹  
- 用 **教师模型** 给每个 token 打分  
- 直接对齐每一步，惩罚错误、强化正确

### 可视化示意
![Figure](images/img-1.png)

> 图解：学生生成轨迹，教师逐 token 评分，颜色越深惩罚越大，强调“分叉错误”的关键 token。

---

## 损失函数：Reverse KL（逐 token）
文中采用最简单的 **reverse KL** ：

$$
D_{KL}(\pi_\theta \Vert \pi_{teacher})
$$

它有几个关键特性：

- **mode seeking** ：更倾向学一个高质量模式，而非分散  
- **曝光偏差更低**  
- 与 RL 的“序列级 reverse KL”天然一致  
- 由于是教师分布，难以被“reward hacking”  

在实现上更关键的是：  
 **每个 token 都有奖励，而不是等整条轨迹结束** 。

---

## 实现流程：一步改 RL 代码就够
伪流程如下：

1. **采样轨迹** （学生模型）  
2. **教师打分** ：获取每个 token 的 logprobs  
3. 计算 reverse KL  
4. 用 RL 框架训练（优势 = -reverse KL）  

![Figure](images/img-2.png)

> 图解：使用 RL 框架，只需把 KL 正则模型替换成教师模型，即可实现 on-policy distillation。

---

## 实验一：数学推理（Qwen3-8B）
基线模型：Qwen3-8B-Base  
教师模型：Qwen3-32B  

### Off-policy distillation
- 400k prompts SFT → AIME’24 60%  
- 预计 2M prompts 才能到 70%  

### RL
- Qwen3 官方 RL：17,920 GPU hours → 67.6%  
- 成本非常高  

### On-policy distillation
- 从 400k checkpoint 开始  
- **150 steps（≈77k prompts）** 达到 70%  
- 比 RL 成本低一个数量级

![Figure](images/img-3.png)

> 图解：AIME'24 成绩对比，on-policy distillation 在更低成本下达到同等甚至更好分数。

---

## 成本分析：为什么更便宜？
文中用 FLOPs 衡量：

- 若已有 SFT 数据，on-policy distillation 成本约低 **9x**  
- 若要重新采样教师数据，总成本下降 **30x**  
- 实际 GPU 时间下降甚至 **18x**  

![Figure](images/img-4.png)

> 图解：按 FLOPs 估算，不同方案的成本对比；on-policy distillation 在多种假设下都显著更省。

---

## 实验二：个性化 + 企业内部知识
场景：  
- 需要模型懂公司内部文档  
- 仍保留强指令跟随能力  

### 现象：中训会破坏“后训练能力”
即使混入 chat 数据，IF-eval 仍显著下降。  
LoRA 也救不了。

### 解决方案：on-policy distillation 恢复行为
- 先用公司文档 mid-train  
- 再用旧版 Qwen3-8B 做 teacher  
- 在 Tulu3 prompts 上做 on-policy distillation  

结果：  
- IF-eval 几乎回到原水平  
- 内部 QA 不损失

![Figure](images/img-5.png)

> 图解：mid-train 会显著降低 IF-eval，但 on-policy distillation 能恢复行为且不损失知识表现。

---

## 进一步观察：SFT 自蒸馏也会退化
作者做了更细的实验：

- 用 Qwen3-32B 生成数据  
- KL=0（理论上完全 on-policy）  
- 但 SFT 仍导致性能下降  

原因：  
- 实际 batch 的有限采样导致分布漂移  
- 训练后模型就不再是原来的“自己”  

而 on-policy distillation 因为教师固定，能持续“锚定”到目标行为。

---

## 讨论：为什么密集监督这么重要？
RL 的信息量是 $O(1)$ bits / episode  
蒸馏是 $O(N)$ bits / episode  

实验表明：

- on-policy distillation 达到 RL 教师水平 **7-10x** 更快  
- 总体 compute 节省 **50-100x**  

![Figure](images/img-6.png)

> 图解：reverse KL 收敛速度显著高于 RL，性能达到同水平时 steps 更少。

---

## 数据效率：一条 prompt 也能学
作者拿一个数学题反复训练：

- 20 steps  
- 每步 256 rollouts  
- 总共 5120 序列  

结果：  
模型性能接近教师。  
原因：reverse KL 学的是分布，而不是背答案。

![Figure](images/img-7.png)

> 图解：单一 prompt 的重复训练仍可接近教师表现，说明蒸馏学习的是策略分布而非记忆答案。

---

## 更深层的视角：RL 是在“搜索策略”
作者强调：

- RL 的计算主要花在 **search**  
- 一旦策略找到，蒸馏就能“压缩”成果  
- 类似科研探索 vs 教学传播  

这解释了为什么 on-policy distillation 能快速复制 RL 结果。

---

## 持续学习的潜力
on-policy distillation 的一个强用途：

- 新知识 -> mid-train  
- 行为丢失 -> on-policy distill 恢复  
- 周期性更新能力  

与 Phasic Policy Gradient 的“阶段性训练”思路一致。

![Figure](images/img-8.png)

> 图解：阶段性地交替 mid-train 与 on-policy distillation，可实现持续学习而不遗忘。

---

## 与已有工作关系
这篇文章和以下方向深度相关：

- DAGGER（对 student 访问状态打分）  
- Process Reward Modeling  
- Qwen3 / MiniLLM / Agarwal 等 on-policy distill  
- 反向 KL + 过程监督  

---

## 关键公式（保留核心）
Reverse KL 逐 token：

$$
D_{KL}(\pi_\theta \Vert \pi_{teacher}) = \sum_x \pi_\theta(x) \log \frac{\pi_\theta(x)}{\pi_{teacher}(x)}
$$

核心优化目标：  
让学生在自己轨迹上逐步接近教师分布。

---

## 总结
- **on-policy distillation** 把 RL 的贴合性与 distillation 的监督密度结合起来  
- 在数学推理与个性化训练中显著降低成本  
- 更适合持续学习与“行为恢复”  
- 本质上是把教师模型当作 reward model  

---

## 参考与原文
本文参考自 **On-Policy Distillation - Thinking Machines Lab**  
https://thinkingmachines.ai/blog/on-policy-distillation/

---

如果你希望，我可以继续做这篇文章的 **实验细节复现指南** 或整理成 **PPT 讲稿版本** 。