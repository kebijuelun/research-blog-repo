# Code-A1 深度解读：让 Code LLM 和 Test LLM “对抗进化”，把代码强化学习从静态奖励带入动态博弈

## 一句话先看懂这篇论文
这篇工作提出了一个非常关键的观点：代码强化学习（RL）真正的瓶颈，不是“怎么优化模型”，而是 ** 怎么持续提供高质量、可学习、会动态变难的可验证奖励 ** 。  
为此，作者设计了 `Code-A1`：用两个独立模型做对抗协同——`Code LLM` 负责写代码、`Test LLM` 负责找 bug，通过对立目标共同进化，替代传统的静态人工测试集。

---

## 1. 背景问题：为什么现有 Code RL 很容易“学偏”

### 1.1 静态 Golden Tests 的根本问题
传统代码 RL 依赖单元测试通过率作为 reward，但现实中常见问题是：

- 每题测试点太少（常见 3～5 条），覆盖不足；
- 测试集是静态的，模型变强后不会同步变难；
- 简单测试会让“投机代码”拿高分，过严测试又会把“接近正确”的解法全判错。

这会直接导致 reward 信号失真，训练上限被卡住。

### 1.2 Self-Play 为什么不够好
单模型自博弈（同一个模型既写代码又写测试）看似优雅，但会遇到两难：

- ** 黑盒测试 ** （只看题面）能防串通，但测试过于泛化，不够针对；
- ** 白盒测试 ** （看候选代码）更有攻击性，但单模型会出现“自我串通”：故意生成容易通过的测试来刷 reward。

---

## 2. 核心思路：Code-A1 的“对抗共进化”

`Code-A1` 把任务拆成两个模型，并设置对立目标：

- `Code LLM`：目标是通过更多测试；
- `Test LLM`：目标是让代码暴露更多缺陷（即让代码 fail）。

这使得白盒测试可以安全启用：`Test LLM` 可读取候选代码并定向攻击逻辑弱点，但不会和 `Code LLM` 共谋，因为它们是独立策略、目标冲突。

![Figure 1](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/Motivation.png)  
> 图解：这张图对比了三种训练范式。Vanilla GRPO 用静态测试；Self-Play 为防串通被迫黑盒；Code-A1 通过模型解耦实现安全白盒测试，并用 Mistake Book 维持稳定共进化。

---

## 3. 方法细节：从 rollout 到 reward 的完整机制

![Figure 2](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/Method.png)  
> 图解：整体流程是“代码生成 → 白盒测试生成 → 测试校验 → 执行评估 → 双模型更新”。横向看是训练流水线，纵向看是两种 policy 的对抗闭环。

### 3.1 任务形式化
给定题目 $Q$，`Code LLM` 生成候选代码 $\hat{C}$，`Test LLM` 基于 $(Q,\hat{C})$ 生成测试集合 $\hat{T}$。  
每条测试强制使用断言格式：`assert func(*args) == answer`，以降低解析噪声。

### 3.2 对抗 Rollout（每个训练 step）
1. `Code LLM` 每题采样 $M$ 个候选解；
2. 对每个候选解，`Test LLM` 采样 $N$ 组测试；
3. 先用 GT 代码验证测试有效性（可执行、唯一、答案正确）；
4. 再用候选代码执行“历史失败测试 + 新测试”，计算 pass rate。

### 3.3 Mistake Book：经验回放缓冲区
作者设计了按题维护的历史失败测试池：

$$
\mathcal{B}: Q_i \mapsto T_i^{hist}
$$

更新规则是：

- 新失败（NewFails）加入；
- 已修复通过（NewPasses）移除或降频。

它解决了三个关键问题：

- 降低训练方差（提供稳定基线）；
- 防止灾难性遗忘（老 bug 不回潮）；
- 形成难度课程信号（新旧测试通过差值反映是否“变难了”）。

### 3.4 双目标奖励函数（论文最关键）
#### Code LLM 奖励
当有历史测试时：

$$
R_C(\hat{C}_m)=\frac{1}{2}\left(Pass_{hist}+\operatorname{Avg}(Pass_{new})\right)
$$

无历史测试时，退化为新测试通过率。  
核心作用：防止模型“只过新题，不管老坑”。

#### Test LLM 奖励
由两部分组成：

- 有效性奖励：$R_T^{val}=Valid(\hat{T})$
- 对抗性奖励（鼓励新测试比历史测试更能击穿代码）：

$$
R_T^{adv}=
\begin{cases}
\frac{1}{2}(Pass_{hist}-Pass_{new}+1), & T^{hist}\neq\emptyset \\
1-Pass_{new}, & \text{otherwise}
\end{cases}
$$

最终组合：

$$
R_T(\hat{T})=\alpha R_T^{val}+(1-\alpha)R_T^{adv}
$$

这里 $\alpha$ 控制“测试可执行性”与“攻击性”的平衡。

### 3.5 优化与采样策略
- 两个模型都用 GRPO 训练；
- 采用非对称采样：Code 采样 $M$，Test 实际产生 $M\times N$；
- 用 `TopVar` 只挑奖励方差最大的测试组更新 `Test LLM`，提升算力利用效率。

---

## 4. 实验结果：不靠人工 Golden Tests 也能赢

### 4.1 代码生成性能（HumanEval+/MBPP+/BigCodeBench）
论文在 Qwen2.5-Coder 1.5B/3B/7B 三个尺度上比较 Base、Golden Tests、Self-Play、Code-A1。  
结论：`Code-A1` 在三个尺度上全部拿到最高 Avg。

- 1.5B：56.95（优于 Golden 56.23、Self-Play 55.88）
- 3B：66.15（优于 Golden 65.14）
- 7B：70.72（优于 Self-Play 70.39）

这说明动态白盒对抗测试提供了更贴合策略改进的训练信号。

### 4.2 测试生成性能（UnLeakedTestBench）
论文用 `Mul = pass@k × mut@k` 衡量“既有效又有杀伤力”的测试产出。

- 1.5B：7.14（SFT 为 4.35）
- 3B：15.29（超过 7B Base 的 14.72）
- 7B：19.74（最优）

关键观察：3B 的 Code-A1 测试能力超过 7B 基座，说明对抗共进化带来的收益大于单纯加参数。

---

## 5. 消融：哪些模块真的有用

### 5.1 $\alpha$ 权重
![Alpha Code](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/alpha_code.png)  
> 图解：左图展示 $\alpha$ 对代码任务 avg@32 的影响，$\alpha=0.5$ 最平衡。太小会导致测试无效、噪声大；太大则测试过于保守。  

![Alpha Test](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/alpha_test.png)  
> 图解：右图展示测试能力变化，$\alpha=0.5$ 同时兼顾 pass 和 mutation；$\alpha=0$ 或 $1$ 都会明显退化。

### 5.2 去掉预测答案（w/o predicted answer）
虽然生成更“难”的输入更快，但 pass@5 严重下滑，说明仅靠 oracle 填答案会削弱 Test LLM 的完整测试构造能力。

### 5.3 去掉白盒代码输入（w/o referable code）
HumanEval+ 从 72.69 掉到 68.66。  
这直接证明：白盒可见代码是定向攻击的必要条件。

### 5.4 去掉 Mistake Book
代码性能明显下降，测试性能小幅波动。  
说明 Mistake Book 主要用于保护 Code LLM 的长期记忆与训练稳定性。

---

## 6. 训练动力学：这套系统真的在“共同进化”

![Pass Rate](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/passrate.png)  
> 图解：历史测试通过率曲线与新测试通过率共同变化，表示模型在修复历史 bug 的同时持续面对新攻击，而非单向拟合。  

![Mistake Book Dynamics](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/mistake.png)  
> 图解：新增失败测试（攻击成功）与移除测试（修复成功）逐步趋于平衡，体现出对抗系统进入动态均衡。  

![Rewards](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/rewards.png)  
> 图解：Code/Test 两侧 reward 在 $\alpha=0.5$ 时最稳定，说明该配置下“有效性-难度”平衡最好。

---

## 7. 附录里的高价值细节（复现实用）

### 7.1 工程配置
- 基座：Qwen2.5-Coder-Instruct（1.5B / 3B / 7B）
- 训练集：KodCode-V1 中 9688 道 hard 问题
- rollout：每题采样 8，温度 1.0
- Test LLM 每次输出固定 $K=5$ 条断言
- 训练步数：111 steps
- $\alpha=0.5$，TopVar 设置 $\ell=1$

### 7.2 沙箱执行设计
作者给了两套模板：

- 训练模板：一次请求完成“测试校验 + 代码评估 + reward 统计”；
- 验证模板：批量候选代码 + 超时保护，提高 avg@k 评估效率。

### 7.3 Mistake Book 落地
使用 JSON 全局持久化，按步更新频次，支持中断恢复训练。  
这是非常实用的工程设计点，不只是论文概念。

### 7.4 方法全流程图
![Method Detail](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/methodname-Adversarial-Evolving-of-Code-LLM-and-Test-LLM-via-Reinforcement-Learning/figures/Method_Detail.png)  
> 图解：上半部分是 Code LLM 的生成-执行-更新闭环；下半部分是 Test LLM 的白盒生成-校验-TopVar 更新闭环；中间由 Mistake Book 串联形成稳定反馈回路。

---

## 8. 额外分析结果：生成测试可替代人工测试吗？
论文做了一个很有说服力的实验：  
把不同来源测试（Base/SFT/Code-A1/Human）当作静态 Golden Tests 去训练代码模型。

结果显示：`Code-A1` 生成的测试作为训练监督，Avg 达到 56.75，超过人工标注 56.23。  
这意味着对抗进化出的测试，质量已经可以在部分场景替代昂贵的人工构建。

---

## 9. 局限与未来方向
作者也很诚实地指出了边界：

- 训练阶段依赖 GT 代码做测试校验；
- 当前测试格式主要是 assert，难覆盖 stateful / I/O / property-based 测试；
- 目前只验证了 Python 函数级任务，跨语言与长上下文仍待验证。

未来最值得期待的方向是：  
把 Code-A1 和 Long-CoT 基座结合，可能进一步放大测试生成与代码推理的上限。

---

## 10. 总结：这篇论文的真正贡献是什么
`Code-A1` 的最大价值不是“又一个 RL trick”，而是把 ** 可验证奖励设计 ** 从静态数据问题升级成动态博弈系统问题。  
它回答了一个长期被忽略的问题： ** 如何让 reward 随模型能力共同成长 ** 。  
通过“模型解耦 + 白盒对抗 + Mistake Book 回放”，这篇工作给出了一个可复现、可扩展、工程上可落地的答案。

> 本文参考自 [Adversarial Evolving of Code LLM and Test LLM via Reinforcement Learning](https://www.arxiv.org/abs/2603.15611)