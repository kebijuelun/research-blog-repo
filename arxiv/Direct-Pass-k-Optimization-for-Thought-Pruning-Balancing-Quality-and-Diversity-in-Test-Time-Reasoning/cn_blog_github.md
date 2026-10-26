# PKTP：直接优化 Pass@k 的推理剪枝

让推理模型同时生成 128 条解法很有效，但把每条长推理都跑完，计算成本也很高；更棘手的是，提前删掉“看起来不行”的解法时，单纯追求高分容易全军覆没，单纯追求多样又可能留下低质量路线。PKTP（Pass@k Thought Pruning）把目标改为：从 $N$ 条推理前缀中留下 $k$ 条，使至少一条最终答对的概率最大。在 8 个开源推理模型上，它整体优于质量优先的 Linear Probe 和多样性优先的 RepExp；例如 DeepSeek-R1-Distill-Qwen-7B 在 $k=4$ 时达到 59.7%，比 Linear Probe 高 3.3 个百分点，而 DeepSeek-R1-Distill-Qwen-1.5B 在同一预算下分别领先两种基线 3.0 和 3.1 个百分点。

## 为什么“留下最高分”不等于 Pass@k 更高？

Test-time scaling 的基本做法是，对同一道题采样多条完整推理，再通过多数投票或 Best-of-$N$ 找答案。完整生成与评分都很贵，于是一个自然的节省方式是：所有候选只先生成到第 $T$ 个 Token，保留 $k$ 条继续生成到总预算 $B$，其余立即丢弃。

![测试时只保留部分推理继续生成](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Direct-Pass-k-Optimization-for-Thought-Pruning-Balancing-Quality-and-Diversity-in-Test-Time-Reasoning/figures/problem.png)

> 图解：模型先为一道题生成 $N$ 条 rollout（完整候选推理），但每条只观察到截断点 $T$。被圆环标出的 $k$ 条才继续运行到预算 $B$，因此剪枝策略直接决定后续算力花在哪些推理上。

这里的关键不是“每条候选单独有多可靠”，而是“这一组候选合起来能否至少成功一次”。这正是 Pass@$k$：对保留集合 $S$，要求 $|S|=k$，最大化

$$
\max_{S\subseteq\{1,\ldots,N\},\ |S|=k}
\Pr\left(\sum_{i\in S}\mathbf{1}\{C(y^{(i)})=+1\}\geq 1\ \middle|\ q,y_{1:T}^{(1)},\ldots,y_{1:T}^{(N)}\right).
$$

其中 $y_{1:T}^{(i)}$ 是第 $i$ 条推理前缀，$C(y^{(i)})\in\{-1,+1\}$ 表示它继续生成后是否答错或答对，$\mathbf{1}\{\cdot\}$ 是指示函数。正确性由任务检查器决定，例如数学答案匹配或代码通过单元测试。

这个目标与两类已有方法并不相同：

- **Linear Probe，纯利用（exploitation）**：用 hidden state 训练探针，按预测成功分数 $w^\top h_i$ 留下前 $k$ 名。高分候选可能挤在同一个“看起来都对”的区域，一旦探针判断失误，整组一起失败。
- **RepExp，纯探索（exploration）**：不断选择最不能被已选表示解释的候选，覆盖不同推理方向，却完全不看候选本身是否靠谱。
- **PKTP**：不手工调和“质量分 + 多样性分”，而是直接估计整组至少一个成功的概率。质量与冗余本来就在这个概率中，因此无需另设权衡系数。

![质量优先、多样性优先与 PKTP 的选择差异](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Direct-Pass-k-Optimization-for-Thought-Pruning-Balancing-Quality-and-Diversity-in-Test-Time-Reasoning/figures/teaser.png)

> 图解：同一组候选按推理前缀的 embedding 排布，颜色表示预测正确性，勾叉表示实际结果，圆环表示 $k=3$ 时被保留的候选。左图纯利用选中了彼此接近的高分候选，却一起失败；中图纯探索覆盖了不同区域，却把预算浪费在低质量候选上；右图 PKTP 同时考虑成功质量与共同失败风险，保留了实际正确的候选。

笔者认为，这篇文章最重要的视角转换是：**剪枝不是排序问题，而是集合选择问题。**单条候选的边际价值，取决于已经留下了哪些候选。

## PKTP：先学会“共同失败”，再选择一组候选

要直接优化上述概率，有两个障碍。第一，前缀尚未完成，未来对错未知，而且独立采样的 rollout 在给定前缀后并不必然条件独立；相似思路可能共享同一个盲区。第二，即使能评估任意集合的 Pass@$k$，从 $N$ 个候选中穷举大小为 $k$ 的集合仍然不可行。

PKTP 因此分成两步：用潜变量高斯模型学习所有候选结果的联合分布，再用有理论保证的贪心算法选择子集。

### 第一步：把前缀压缩成可比较的表示

对第 $i$ 条 rollout，取最后一层 Transformer 在第 $t$ 个生成 Token 的 hidden state $x_t\in\mathbb{R}^d$，做指数移动平均（EMA）：

$$
e_t=(1-\alpha)e_{t-1}+\alpha x_t,\qquad \alpha=10^{-3}.
$$

在截断点 $T$ 对 $e_t$ 归一化，得到 $h_i$；若推理提前结束，则使用全部生成 Token。EMA 近似强调最近约 1000 个 Token。论文实验覆盖的 hidden size 从 1024 到 4096，框架本身并不限定必须采用 EMA，也可以替换为其他前缀表示。

### 第二步：一个质量项、一个共享因子、一个独立噪声

PKTP 为每条候选定义潜分数 $Z_i$：分数越高越可能成功，最后以 0 为阈值得到对错。向量形式为

$$
\mathbf Z=Q_\theta(H)+H^\top\mathbf f+\mathbf m,
\qquad \mathbf m\sim\mathcal N(0,I_N),
\qquad \mathbf f\sim\mathcal N(0,I_d).
$$

这里 $H=[h_1,\ldots,h_N]$，$Q_\theta(H)=[q_1,\ldots,q_N]^\top$ 预测每条候选的质量；$\mathbf f$ 是所有候选共享的随机因子，$h_i^\top\mathbf f$ 描述相关性；$\mathbf m_i$ 则是单条候选自己的噪声。对 $i\ne j$，有

$$
\operatorname{Cov}(Z_i,Z_j\mid H)=h_i^\top h_j.
$$

因此，表示方向越接近，候选越倾向于一起成功或一起失败。正确性变量定义为 $\mathbf C_i=2\mathbf 1\{Z_i\geq0\}-1$。令 $\Phi$ 为标准正态分布的 CDF，观察到整组结果 $c\in\{-1,+1\}^N$ 的似然是

$$
\Pr(\mathbf C=c\mid H)
=\mathbb E_{\mathbf f}\left[\prod_{i=1}^{N}\Phi\left(c_i(q_i+h_i^\top\mathbf f)\right)\right].
$$

训练时有 $M$ 道题，每道题提供一组表示 $H^{(m)}$ 和完整的对错结果 $c^{(m)}$，通过最大化平均对数似然学习 $Q_\theta$。由于数据有限，论文只用线性映射 $Q_\theta(h)=\theta^\top h+\theta_0$，并用随机梯度上升优化。它与 Linear Probe 的重要区别是：**训练目标来自一道题全部 rollout 的联合结果，而不是逐条独立分类。**

### 第三步：二阶展开，把难算的期望变成闭式

上述似然含有对 $\mathbf f$ 的期望，没有一般闭式解。PKTP 在 $h_i^\top\mathbf f$ 处对每个对数因子做二阶 Taylor 展开，使剩余部分成为可精确积分的高斯积分。

定义标准正态密度 $\phi$，以及

$$
\lambda_i(c_i)=\frac{\phi(q_i)}{\Phi(c_iq_i)},\qquad
\kappa_i(c_i)=\lambda_i(c_i)\bigl(\lambda_i(c_i)+c_iq_i\bigr),
$$

$$
\Sigma(c)=\sum_{i=1}^{N}\kappa_i(c_i)h_ih_i^\top,
\qquad
b(c)=\sum_{i=1}^{N}c_i\lambda_i(c_i)h_i.
$$

展开的核心是

$$
\log\Phi\left(c_i(q_i+\varepsilon)\right)
=\log\Phi(c_iq_i)+c_i\lambda_i(c_i)\varepsilon
-\frac{\kappa_i(c_i)}{2}\varepsilon^2+O(|\varepsilon|^3).
$$

代入并配方积分后，得到可微的近似似然：

$$
\widetilde{\Pr}(\mathbf C=c\mid H)
=\left(\prod_{i=1}^{N}\Phi(c_iq_i)\right)
\det(I_d+\Sigma(c))^{-1/2}
\exp\left(\frac12 b(c)^\top(I_d+\Sigma(c))^{-1}b(c)\right).
$$

它是局部二阶近似，不是精确等式，也不是误差界；价值在于把联合模型变成可训练、可求集合目标的形式。

### 第四步：最小化“整组全错”的概率

最大化 Pass@$k$ 等价于最小化保留集合全部失败的概率：

$$
L(S)=\mathbb E_{\mathbf f}\left[\prod_{i\in S}\Phi\left(-q_i-h_i^\top\mathbf f\right)\right].
$$

将上面的二阶近似用于全错事件，并简写 $\lambda_i=\phi(q_i)/\Phi(-q_i)$、$\kappa_i=\lambda_i(\lambda_i-q_i)$，再令 $\Sigma_S=\sum_{i\in S}\kappa_i h_i h_i^\top$、$b_S=\sum_{i\in S}\lambda_i h_i$，得到选择目标：

$$
\log\widetilde L(S)
=\sum_{i\in S}\log\Phi(-q_i)
-\frac12\log\det(I_d+\Sigma_S)
+\frac12 b_S^\top(I_d+\Sigma_S)^{-1}b_S.
$$

三项各自承担明确职责：

1. **质量项**：$q_i$ 越大，独立基准失败概率 $\Phi(-q_i)$ 越小，因此优先高质量候选。
2. **行列式项**：$-\log\det$ 偏好覆盖更多方向的集合，减少在同一表示方向上重复下注。
3. **二次惩罚项**：惩罚加权和 $b_S$ 同向累积。它能区分 $h_2=h_1$ 与 $h_2=-h_1$；两者的外积相同，行列式无法区分，但前者表示同向相关、共同失败风险更高，后者会抵消 $b_S$。

这个设计的聪明之处在于，多样性不是额外贴上去的正则项，而是从“共同失败概率”自然推导出来的。

## 如何高效选集合，并保证近似质量？

精确最小化 $\log\widetilde L(S)$ 是 NP-hard，论文通过从最大独立集问题归约证明了这一点。但近似 Pass@$k$ 函数 $\widetilde P(S)=1-\widetilde L(S)$ 具有三个性质：归一化、单调性和子模性。子模性即收益递减：对 $A\subseteq B$ 且 $j\notin B$，

$$
\widetilde P(A\cup\{j\})-\widetilde P(A)
\geq
\widetilde P(B\cup\{j\})-\widetilde P(B).
$$

所以从空集开始，每次加入使 $\widetilde P$ 增长最多、等价于使 $\log\widetilde L$ 增量最小的候选，连续选择 $k$ 次，就有

$$
\widetilde P(S_k^{\mathrm{greedy}})
\geq (1-e^{-1})\max_{|S|\leq k}\widetilde P(S).
$$

注意，这个约 63.2% 的保证针对**代理目标**，并不直接等于真实 Pass@$k$ 的最优保证。

### 秩一更新避免反复求逆

每次加入候选 $j$，$\Sigma_S$ 只增加一个秩一矩阵 $\kappa_jh_jh_j^\top$。令 $\Omega_S=(I_d+\Sigma_S)^{-1}$、$r_j=h_j^\top\Omega_Sh_j$、$u_j=h_j^\top\Omega_Sb_S$，矩阵行列式引理和 Sherman–Morrison 公式给出

$$
\Delta_j(S)=\log\Phi(-q_j)-\frac12\log(1+\kappa_jr_j)
+\frac{2\lambda_ju_j+\lambda_j^2r_j-\kappa_ju_j^2}{2(1+\kappa_jr_j)}.
$$

每轮选择 $\Delta_j(S)$ 最小的候选，再更新 $\Omega_S$、$b_S$ 以及所有候选的 $r_\ell,u_\ell$。总时间复杂度为 $O(k(Nd+d^2))$，空间为 $O(Nd+d^2)$；若每个候选都重新分解矩阵，时间会升至 $O(kNd^3)$。

模型只需训练一次，而且一次贪心过程产生嵌套集合 $S_1\subset S_2\subset\cdots$。推理时可以在线决定预算 $k$，不用针对每个 $k$ 重新训练或重新选择。

## 它为何能连接 Linear Probe 与 RepExp？

把相关性项关掉，目标只剩 $\sum_{i\in S}\log\Phi(-q_i)$，PKTP 就退化为按质量取 Top-$k$，即 Linear Probe 式规则。若所有质量相同且忽略二次惩罚，集合相关部分只剩行列式项；贪心增量排序变为

$$
h_j^\top(I_d+\Sigma_S)^{-1}h_j,
$$

这正是 RepExp 的残差方差选择标准，正则化参数对应 $\tau=1/\kappa$。

更有解释力的是弱相关展开。把共享因子换成 $\rho\mathbf f$，当 $\rho\to0$ 时，精确目标与代理目标具有相同展开：

$$
\log L(S)=
\sum_{i\in S}\left[\log\Phi(-q_i)+\frac{\rho^2}{2}\lambda_iq_i\|h_i\|^2\right]
+\rho^2\sum_{i<j\in S}\lambda_i\lambda_jh_i^\top h_j+O(\rho^4).
$$

$\rho=0$ 时只有质量；$\rho>0$ 后，每加入一条候选都要支付与已选候选的两两相关惩罚。集合越大，累计惩罚越可能超过质量差，因此 PKTP 会自然地从“先看质量”过渡到“更多考虑分散风险”。

附录进一步给出了两种预算区间：

- 小预算下，若第 $k$ 与第 $k+1$ 个质量之间的间隔 $\Gamma_k$ 足够大，即 $\Gamma_k>(1+3\Lambda^2)k\rho^2$ 且 $k\rho^2\leq1$，Top-$k$ 是唯一最优集合，贪心也精确返回它。
- 大预算下，在少数极高质量“尖峰”候选、普通候选质量有界，以及所选表示覆盖各方向等假设下，对普通候选质量取平均后，目标近似等于保留尖峰并最大化 RepExp 的 $\log\det$ 目标，误差为 $O(k^{-1/2})$。

这些结论解释了设计之间的连续性，但大预算结论针对期望目标与特定候选族，并不意味着任意数据上 PKTP 与 RepExp 都会选择同一集合。

## 实验：八个模型、约 600 题、20 次交叉验证

### 实验设置

论文评测 8 个开源模型。小规模组包括 Qwen3-0.6B、Qwen3-1.7B、OpenMath-Nemotron-1.5B、DeepSeek-R1-Distill-Qwen-1.5B；较大组包括 Qwen3-8B、OpenMath-Nemotron-7B、DeepSeek-R1-Distill-Qwen-7B、Skywork-OR1-Math-7B。采样温度均为 0.8，top-$p=0.95$，完整预算 $B=32{,}768$ Token。

题目来自 AIME、HMMT、RIMO-N，以及按模型规模搭配的 BeyondAIME、AMO-Bench、OlymMATH Easy/Hard 和不同难度层级的 Omni-MATH。每个模型约 600 题，并移除 128 次生成全对或全错的题目，因为这些题无法区分剪枝策略。最终答案由 Qwen3-32B 判断是否等价于参考答案，以覆盖描述性答案。

评测采用 8 折交叉验证，重复 20 次随机分折与训练。主设置为 $N=128$、$T=4000$、$k\in\{1,2,4,8,16\}$；RepExp 的 $\tau=1$，所有方法使用相同 EMA 表示。

### 主结果：中等 $k$ 的优势最清楚

![八种模型上各剪枝策略相对随机剪枝的增益](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Direct-Pass-k-Optimization-for-Thought-Pruning-Balancing-Quality-and-Diversity-in-Test-Time-Reasoning/figures/gains.png)

> 图解：每个子图对应一个模型，横轴是保留数量 $k$，纵轴是该方法 pooled Pass@$k$ 减去 Random 的百分点增益，零线表示与随机保留相同，阴影为逐次重复差值的一个标准误。Linear Probe 常在 $k=1$ 较强，但部分模型上随 $k$ 增大而接近甚至低于随机；RepExp 往往在大 $k$ 更有利；PKTP 在中等 $k$ 的综合优势最突出。

下表保留主实验的全部绝对数值，单位为 %。为便于阅读省略标准误；Random 是随机保留 $k$ 条的自然参照。

| 模型 | 方法 | @1 | @2 | @4 | @8 | @16 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | Random | 24.0 | 34.4 | 45.4 | 56.5 | 67.6 |
|  | Linear Probe | 24.9 | 33.8 | 44.3 | 56.2 | 67.1 |
|  | RepExp | 23.9 | 33.5 | 45.0 | 55.9 | 67.0 |
|  | **PKTP** | **25.6** | **36.3** | **47.2** | **58.4** | **69.3** |
| Qwen3-1.7B | Random | 39.5 | 51.5 | 62.6 | 72.5 | 81.0 |
|  | Linear Probe | 40.7 | 52.0 | 63.1 | 72.5 | 79.9 |
|  | RepExp | 39.6 | 51.6 | 63.9 | **74.6** | **82.2** |
|  | PKTP | 39.3 | **52.7** | **64.4** | 73.8 | 81.7 |
| OpenMath-Nemotron-1.5B | Random | 47.1 | 59.3 | 69.3 | 77.3 | 83.9 |
|  | Linear Probe | **48.2** | **59.8** | 69.4 | 77.6 | 84.0 |
|  | RepExp | 47.3 | 57.9 | 68.9 | 76.8 | 84.7 |
|  | PKTP | 48.1 | 59.4 | **69.4** | **78.3** | **85.2** |
| DeepSeek-R1-Distill-Qwen-1.5B | Random | 27.4 | 38.4 | 49.5 | 60.2 | 70.3 |
|  | Linear Probe | 30.1 | 39.8 | 49.9 | 59.4 | 69.7 |
|  | RepExp | 27.3 | 38.5 | 49.8 | 60.6 | **72.6** |
|  | **PKTP** | **30.3** | **41.4** | **52.9** | **62.5** | 71.7 |
| Qwen3-8B | Random | 46.9 | 58.0 | 67.2 | 75.2 | 82.3 |
|  | Linear Probe | **49.2** | **59.2** | 67.4 | 74.4 | 81.2 |
|  | RepExp | 46.9 | 57.4 | 68.1 | 75.9 | **83.0** |
|  | PKTP | 47.3 | 58.7 | **68.6** | **76.0** | **83.0** |
| OpenMath-Nemotron-7B | Random | 44.0 | 55.5 | 66.0 | 75.2 | 83.1 |
|  | Linear Probe | 46.3 | 57.5 | 67.5 | 76.3 | 83.6 |
|  | RepExp | 44.0 | 56.8 | 66.4 | 76.3 | **84.7** |
|  | **PKTP** | **47.4** | **59.1** | **68.2** | **76.7** | 84.3 |
| DeepSeek-R1-Distill-Qwen-7B | Random | 37.5 | 48.3 | 58.5 | 68.2 | 77.5 |
|  | Linear Probe | 37.4 | 46.7 | 56.4 | 66.0 | 76.6 |
|  | RepExp | 37.4 | 47.2 | 58.6 | **70.0** | **79.7** |
|  | **PKTP** | **38.6** | **49.5** | **59.7** | **70.0** | 79.3 |
| Skywork-OR1-Math-7B | Random | 42.5 | 53.1 | 62.7 | 71.5 | 79.8 |
|  | Linear Probe | 44.4 | 54.2 | 62.9 | 71.3 | 79.4 |
|  | RepExp | 42.5 | 54.5 | 64.5 | 72.1 | 79.4 |
|  | **PKTP** | **44.7** | **55.7** | **65.6** | **73.2** | **80.3** |

最醒目的结果是 DeepSeek-R1-Distill-Qwen-1.5B 的 $k=4$：PKTP 为 52.9%，Linear Probe 为 49.9%，RepExp 为 49.8%。Qwen3-0.6B 在所有 $k\geq2$ 上都领先两种基线超过 2 个百分点。对所有模型和 $k\geq2$，PKTP 都高于 Random。

但也不能把结论夸大成“每个模型、每个 $k$ 都第一”。例如 Qwen3-1.7B 在 $k=16$ 时 RepExp 为 82.2%，PKTP 为 81.7%；OpenMath-Nemotron-7B 在 $k=16$ 时分别为 84.7% 和 84.3%。大预算下两者接近，与前面的理论联系相呼应，也暴露出前缀表示中的质量信号可能不够强。

### 两个失败案例：基线究竟错在哪里？

![Linear Probe 选中一组共同失败的候选](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Direct-Pass-k-Optimization-for-Thought-Pruning-Balancing-Quality-and-Diversity-in-Test-Time-Reasoning/figures/probe_cluster.png)

> 图解：这是 Qwen3-8B 在一道 AMO-Bench 题上的 128 条 rollout，横纵轴是表示经 PCA 后的前两个主成分；绿色为正确、红色为错误。Linear Probe 排名前 16 的候选由蓝圈标出，它们挤在一起且全部错误；灰圈标出的 16 条随机候选中反而有 7 条正确。这是“高分但高度相关”的直观反例。

![RepExp 偏好离群但错误的候选](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Direct-Pass-k-Optimization-for-Thought-Pruning-Balancing-Quality-and-Diversity-in-Test-Time-Reasoning/figures/repexp_outlier.png)

> 图解：这是 Qwen3-8B 在一道 Omni-MATH 题上的 128 条 rollout，横纵轴为表示经多维尺度分析得到的二维坐标；绿色为正确、红色为错误，粉圈及数字表示 RepExp 的选择顺序。它倾向于远离已选集合的离群点，4 条全部错误；4 条灰圈随机候选中有 1 条正确。说明“新颖”并不自动等于“有前途”。

这两个案例正好构成 PKTP 要处理的两侧风险：探针可能把预算集中在一个错误簇，RepExp 则可能把异常当成有价值的新方向。

### 截断点：多给前缀一些 Token，PKTP 持续受益

![截断位置对 Pass@4 的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Direct-Pass-k-Optimization-for-Thought-Pruning-Balancing-Quality-and-Diversity-in-Test-Time-Reasoning/figures/cutoff.png)

> 图解：横轴为截断位置 $T$，从 2000 增至 8000 Token；纵轴为 $N=128$ 时的 Pass@4，两个子图分别是 Qwen3-1.7B 和 DeepSeek-R1-Distill-Qwen-7B，阴影为一个标准误。$T\geq4000$ 时 PKTP 在两个模型上均为最佳，而且随前缀变长持续改善。

Qwen3-1.7B 上，PKTP 从 $T=2000$ 的 63.5% 提升到 $T=8000$ 的 65.9%；DeepSeek-R1-Distill-Qwen-7B 上则从 57.4% 提升到 61.7%。更长的前缀揭示更多质量与方向信息，这符合直觉。

但 Linear Probe 并未同步受益：Qwen3-1.7B 上从 64.2% 降到 62.3%，最后低于 Random。作者推测，EMA 近似聚焦最近约 1000 个 Token，可能丢失早期信息；这也提醒复现者，截断收益与表示设计密不可分。

### 候选池：池子更大，不一定让所有策略更好

![候选池大小对 Pass@4 的影响](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Direct-Pass-k-Optimization-for-Thought-Pruning-Balancing-Quality-and-Diversity-in-Test-Time-Reasoning/figures/pool.png)

> 图解：横轴是候选池大小 $N=32$ 到 128，纵轴是 $T=4000$、$k=4$ 时的 Pass@4，两个子图使用与上一图相同的模型，阴影为一个标准误。PKTP 在 Qwen3-1.7B 的所有池大小上领先，在 DeepSeek-R1-Distill-Qwen-7B 上从 $N=80$ 起领先。

在 Qwen3-1.7B 上，$N$ 从 32 增至 128 时，RepExp 从 62.9% 升至 63.9%，Linear Probe 从 62.1% 升至 63.1%。但在 DeepSeek-R1-Distill-Qwen-7B 上，Linear Probe 从 58.1% 降至 56.4%，始终低于 Random；RepExp 从 $N=48$ 的 59.5% 降至 $N=128$ 的 58.6%，PKTP 则升至 59.7%。更大的池不仅提供好候选，也提供更多离群点，单一极端策略未必能受益。

附录还报告了 $N\in\{32,48,64,80,96,112\}$ 的完整增益图，设置均为 $T=4000$，候选从每题已生成的 128 条 rollout 中抽取，用于验证主结论并非只在 $N=128$ 下成立。

### 分基准结果：总体优势不是逐题碾压

附录按基准拆分结果后，可以看到更细的边界。比如 DeepSeek-R1-Distill-Qwen-7B 在 AMO-Bench 上，PKTP 的 Pass@1/2/4 分别达到 10.7%、20.0%、33.9%，明显高于 Random 的 7.0%、12.5%、21.1%；但在 OlymMATH (Hard) 上，大 $k$ 时 RepExp 更强，Pass@16 为 60.6%，PKTP 为 58.0%。Qwen3-0.6B 在 AIME 上，PKTP 从 Pass@4 的 50.2% 提升至 Pass@8 的 64.2%，高于 Random 的 45.5% 和 58.3%。

这些拆分说明，pooled 结果适合观察整体趋势，却不能代替对每个模型、基准和预算的检查；方法的选择仍受质量估计可靠性与候选池结构影响。

## 与相关方法的关系及局限

已有测试时剪枝工作主要使用 Token 置信度、自我成功估计、Reward Model 或 hidden-state Probe 为单条候选打分；RepExp 则以表示探索为目标。训练阶段也有工作把 Pass@$k$ 用作 reward，或在生成前规划不同提纲。PKTP 不修改原模型，也不改变采样方式，而是在已有前缀上选择继续计算的对象。

同样的“至少一个成功”目标曾出现在信息检索的 expected 1-call@$k$ 和 Bayesian optimization 的 batch probability of improvement 中。与手工设置质量—多样性系数或直接用行列式促进多样性不同，PKTP 的贡献是学习推理结果的联合分布，并为其闭式代理保留子模性。

需要明确三项局限：

- **近似有边界**：二阶展开是局部代理，论文没有给出它对真实 Pass@$k$ 的一般近似误差；贪心保证也只作用于代理目标。
- **表示是瓶颈**：大 $k$ 时 PKTP 与 RepExp 接近，作者将其归因于前缀 embedding 的正确性信号偏弱。更好的前缀表示可能直接改善结果。
- **实验范围有限**：验证集中于数学推理、8 个开源模型和特定 EMA 表示。论文说明框架可推广到其他表示及 max@$k$ 连续奖励，但这些方向并未在主实验中全面验证。

## 总结：把预算押在“组合成功率”上

- PKTP 将推理剪枝从单候选排序改写为集合级 Pass@$k$ 优化，目标是至少保留一条最终正确的推理。
- 潜变量高斯模型把单条质量、候选间相关性与独立噪声分开，直接刻画相似路线共同失败的风险。
- 二阶展开产生闭式目标，其中质量项、行列式多样性项和同向相关惩罚自然形成质量—多样性平衡。
- 代理目标具有单调子模性，贪心选择获得 $1-e^{-1}$ 的代理近似保证，秩一更新又避免昂贵的反复矩阵分解。
- 8 个模型的实验显示其总体优势在中等 $k$ 最明显，但并非所有预算都胜过 RepExp，前缀表示仍是关键瓶颈。

下一步最值得探索的是能更准确承载“未来正确性”的部分推理表示；一旦表示更可靠，同一套 PKTP 框架无需改动就能直接受益。

> 本文参考自 [Direct Pass@$k$ Optimization for Thought Pruning: Balancing Quality and Diversity in Test-Time Reasoning](http://arxiv.org/abs/2610.31667v1)