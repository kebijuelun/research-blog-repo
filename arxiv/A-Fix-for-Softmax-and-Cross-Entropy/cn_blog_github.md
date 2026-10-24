# Softmax 修复记：一个补丁揪出交叉熵里的两个数值 Bug

你每天在用的 softmax 和交叉熵，标准实现里藏着两个数值 Bug。来自 Google DeepMind 的这篇文章指出：当模型对正确答案极度自信时，目标 logit 的梯度会被浮点舍入**错误地归零**，导致梯度之和不再为零；而在知识蒸馏中，即使学生和老师的 logit 完全相等，标准实现也会算出**非零的"幽灵梯度"**。作者用 log 空间的重参数化（softplus / expm1 / log1p 三件套）给出了即插即用的修复，一举消除了 Grokking 研究中长期困扰学界的"Slingshot"损失尖峰：在没有权重衰减的实验中，标准交叉熵的测试准确率会从 61% 崩到 23%，而修复后平滑收敛到 85%；在模块化加法任务上，修复版在 bf16 / fp32 / fp64 三种精度下全部稳定达到 99–100% 测试准确率。

## 一、标准实现里的两个隐藏 Bug

### 先回顾：为什么标准实现要减去最大 logit

对 logit 向量 $\mathbf{z} \in \mathbb{R}^K$，如果直接按定义 $e^{z_k} / \sum_j e^{z_j}$ 计算 softmax，任何一个 logit 超过浮点指数上限（fp32 中约为 88）就会溢出。因此所有主流框架（JAX、PyTorch、TensorFlow）都先减去最大值 $z_{\max} = \max_j z_j$：

$$
\tilde{z}_k = z_k - z_{\max}, \quad
\operatorname{logsoftmax}(\mathbf{z})_k = \tilde{z}_k - \log \sum_j e^{\tilde{z}_j}, \quad
q_k = \operatorname{softmax}(\mathbf{z})_k = \frac{e^{\tilde{z}_k}}{\sum_j e^{\tilde{z}_j}}
$$

交叉熵则写成 $\mathcal{L}(\mathbf{z}, \mathbf{p}) = -\sum_k p_k \operatorname{logsoftmax}(\mathbf{z})_k$，其精确梯度是大家耳熟能详的 $\mathbf{g} = \mathbf{q} - \mathbf{p}$。

减去 $z_{\max}$ 解决了溢出，但没有解决**精度**问题。这里的关键背景是：浮点数在 0 附近的精度远比在 1 附近高。fp32 中，0 附近的数可以小到 $10^{-38}$ 量级，而加到 1.0 上的数小于约 $10^{-7}$ 就会被舍入吞掉。两个 Bug 都源于这条不对称性。

### Bug 一：Softmax Collapse（软最大坍塌）

当模型以高置信度预测对了目标类 $t$，目标 logit 就是最大值。作者用**logit margin**（logit 余量）来衡量这种置信度——目标 logit 比其余 logit 的 log-sum-exp 大多少：

$$
m = z_t - \log \sum_{j \neq t} e^{z_j}
$$

此时分母变成 $1 + e^{-m}$：目标项被平移为 $e^0 = 1$，其余 $K-1$ 项之和为 $e^{-m} \ll 1$。问题就出在"$1 + e^{-m}$"这个加法上：一旦 margin 超过某个阈值，$e^{-m}$ 加到 1.0 上会被舍入吃掉，分母变成精确的 1.0。作者把这个阈值称为**precision cliff**（精度悬崖）：

- bf16：$c_{16} \approx 5.5$（即 $8 \log 2$）
- fp32：$c_{32} \approx 16.6$（即 $24 \log 2$）
- fp64：$c_{64} \approx 36.7$（即 $53 \log 2$）

越过悬崖后，$q_t = 1/(1+e^{-m})$ 算出精确的 1.0，目标梯度 $\tilde{g}_t = q_t - 1 = 0$，loss 也变成 0。但非目标梯度 $\tilde{g}_j = e^{z_j - z_t} / (1 + e^{-m})$ 的分子是在 0 附近直接求值的，没有被"加 1"污染，一直到指数下溢（$m \approx 88$）都保持正值。

结果就是：**目标梯度归零，非目标梯度依然为正，梯度之和不再为零**（$\sum_k \tilde{g}_k > 0$）。对 SGD 这可能无伤大雅，但 Adam/AdamW 这类自适应优化器会用梯度的滑动平均幅值做归一化，这种不对称会被放大成训练事故——后文会看到，Grokking 研究里反复出现的"Slingshot"损失尖峰正是它引起的。

![图例：颜色对应精度与损失实现](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig2_legend.png)

![Softmax Collapse：目标 logit 梯度随 margin 的衰减](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig1a_hard_cliff.png)

> 图解：横轴是 logit margin $m$（0 到约 38），纵轴是目标 logit 梯度的绝对值 $|g_t|$（对数刻度）。真实梯度应随 $m$ 指数衰减（对数坐标下的直线）。黄色点线、橙色实线、红色虚线分别是 bf16 / fp32 / fp64 下的标准交叉熵，可以看到它们分别在 $c_{16} \approx 5.5$、$c_{32} \approx 16.6$、$c_{64} \approx 36.7$ 三条竖线处**断崖式归零**；而蓝色系的 Fixed CE 曲线则平滑穿过所有悬崖，一直精确到 $10^{-16}$ 量级。

### Bug 二：Softmax Cancellation（软最大对消）

第二个 Bug 出现在知识蒸馏和 KL 正则中，此时目标分布 $\mathbf{p} = \operatorname{softmax}(\mathbf{z}_{\mathrm{T}})$ 来自老师 logit。标准实现分别计算 $\mathbf{q} = \operatorname{softmax}(\mathbf{z}_{\mathrm{S}})$ 和 $\mathbf{p}$，再相减得到梯度 $\tilde{\mathbf{g}} = \mathbf{q} - \mathbf{p}$。

当学生 logit 逼近老师 logit 时，真实梯度应趋向零向量。但浮点世界里有两层误差：其一，两个几乎相等的概率相减会发生**灾难性对消**（catastrophic cancellation），微小的 logit 差异 $\boldsymbol{\delta} = \mathbf{z}_{\mathrm{S}} - \mathbf{z}_{\mathrm{T}}$ 在概率舍入中丢失；其二，前向用 softmax、反向用 log-softmax，两条计算路径的运算序列略有不同，即使 $\mathbf{z}_{\mathrm{S}} = \mathbf{z}_{\mathrm{T}}$，$\mathbf{q}$ 和 $\mathbf{p}$ 也无法逐位相等，留下一个非零的"幽灵梯度"。

![Softmax Cancellation：梯度噪声地板](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig4a_floor.png)

> 图解：横轴是学生–老师 logit 差距 $\|\mathbf{z}_{\mathrm{S}} - \mathbf{z}_{\mathrm{T}}\|_\infty$（对数刻度，$10^{-18}$ 到 $10^0$），纵轴是梯度范数 $\|\nabla_{\mathbf{z}_{\mathrm{S}}} \mathcal{L}\|_2$。理想情况下两者应沿对角线 $y = x$ 同步趋零——蓝色系的修复实现正是如此。而标准实现（红/橙/黄）各自撞上噪声地板：fp64 平在约 $10^{-16.5}$，fp32 平在约 $10^{-7.5}$，bf16 更是高达 $10^{-3}$。换言之，学生已经和老师一模一样时，bf16 标准实现仍在输出量级 $10^{-3}$ 的假梯度。

## 二、修复 Softmax Collapse：永远不要对 $e^{-m}$ 加 1

先看数学上完全等价的两种写法。hard-target 交叉熵的 loss 是 $\mathcal{L}(\mathbf{z}, t) = \log(1 + e^{-m})$，精确梯度为：

$$
g_t = \underbrace{\frac{1}{1 + e^{-m}} - 1}_{\text{标准写法}} = \underbrace{-\frac{e^{-m}}{1 + e^{-m}}}_{\text{修复写法}}, \qquad
g_j = \frac{e^{z_j - z_t}}{1 + e^{-m}} \quad (j \neq t)
$$

两者在实数域相等，且满足 $\sum_k g_k = 0$。但在浮点域，命运截然不同：标准写法先算 $1 + e^{-m}$（$m > c_d$ 时舍入为 1），再取倒数减 1，得到精确的 0；修复写法把 $e^{-m}$ 留在分子里，即使分母舍入为 1.0，$-e^{-m} / 1.0$ 依然保留了 $g_t \approx -e^{-m}$ 的完整精度，一直有效到指数下溢。

这个设计的聪明之处在于：**修复不需要更高精度，只需要重新安排运算顺序**——让微小的量始终在 0 附近求值，永远不要把它加到 1 上。

落到实现上，作者改写了 log-softmax：只有最大 logit $i_{\max} = \arg\max_i z_i$ 被平移到 0，但不把 $e^0 = 1$ 放进指数求和里，而是单独算出 margin，再用数值稳定的 $\operatorname{softplus}(x) = \log(1 + e^x)$ 拼回去：

$$
\tilde{z}_k = \mathbf{1}[k \neq i_{\max}]\,(z_k - z_{i_{\max}}), \quad
m = -\operatorname{LSE}_{j \neq i_{\max}} \tilde{z}_j, \quad
\operatorname{logsoftmax}(\mathbf{z})_k = \tilde{z}_k - \operatorname{softplus}(-m)
$$

对 hard-target 交叉熵，还可以直接用目标下标 $t$ 替代 $i_{\max}$，连 argmax 都不用算：

$$
m = -\operatorname{LSE}_{j \neq t}(z_j - z_t), \qquad
\mathcal{L}(\mathbf{z}, t) = \operatorname{softplus}(-m)
$$

由于第 $t$ 个槽位被替换成常数 0（而不是 $z_t - z_t$），$z_t$ 只通过 $K-1$ 个差值 $z_j - z_t$ 出现在 loss 中，autodiff 会自动算出目标梯度等于非目标梯度之和的负数，精确满足零和性质：

$$
g_j = \frac{e^{z_j - z_t}}{1 + e^{-m}} \quad (j \neq t), \qquad
g_t = -\sum_{j \neq t} g_j = -\frac{e^{-m}}{1 + e^{-m}}
$$

完整的 JAX 实现只有十几行：

```python
import jax, jax.numpy as jnp, optax

def log_softmax(z):
  i_max = jnp.argmax(z, axis=-1)
  z_max = jnp.take_along_axis(z, i_max[..., None], axis=-1)  # 最大 logit
  is_max = jax.nn.one_hot(i_max, z.shape[-1], dtype=bool)    # 最大值掩码
  shifted = jnp.where(is_max, 0.0, z - z_max)                # argmax 处置为精确 0
  others = jnp.where(is_max, -jnp.inf, shifted)              # 求和时排除 argmax
  m = -jax.nn.logsumexp(others, axis=-1, keepdims=True)      # logit margin
  return shifted - jax.nn.softplus(-m)                       # log_softmax(z)

softmax = lambda z: jnp.exp(log_softmax(z))

def cross_entropy(z, t):
  z_t = jnp.take_along_axis(z, t[..., None], axis=-1)        # 目标 logit
  is_t = jax.nn.one_hot(t, z.shape[-1], dtype=bool)          # 目标掩码
  others = jnp.where(is_t, -jnp.inf, z - z_t)                # 求和时排除目标
  m = -jax.nn.logsumexp(others, axis=-1)                     # logit margin
  return jax.nn.softplus(-m)                                 # -log_softmax(z)_t
```

一个具体例子能直观看出差别。取 $\mathbf{z} = [9.0, -9.0, -9.0]$、$t = 0$（fp32，margin 约为 16.64，已经踩在 fp32 悬崖边上）：

```python
z, t = jnp.array([9.0, -9.0, -9.0], dtype=jnp.float32), jnp.array(0)
std_ce = optax.softmax_cross_entropy_with_integer_labels
jax.nn.log_softmax(z)                       # ->           [ 0.0,      -18.0,     -18.0    ]
jax.grad(lambda z: jax.nn.softmax(z)[0])(z) # ->           [ 0.0,      -1.523e-8, -1.523e-8]
jax.value_and_grad(std_ce)(z, t)            # -> 0.0,      [ 0.0,       1.523e-8,  1.523e-8]
log_softmax(z)                              # ->           [-3.046e-8, -18.0,     -18.0    ]
jax.grad(lambda z: softmax(z)[0])(z)        # ->           [ 3.046e-8, -1.523e-8, -1.523e-8]
jax.value_and_grad(cross_entropy)(z, t)     # -> 3.046e-8, [-3.046e-8,  1.523e-8,  1.523e-8]
```

标准实现的目标 log-probability、loss 和目标梯度全部归零，梯度之和为 $3.046 \times 10^{-8} \neq 0$；修复实现返回准确值，且三个梯度精确抵消。

## 三、修复 Softmax Cancellation：在 log 空间做减法

解决了高置信度场景，下一个问题是蒸馏场景：如何让学生逼近老师时梯度干净地趋零。

思路是不再分别计算 $\mathbf{q}$ 和 $\mathbf{p}$ 再相减，而是把老师 logit 直接传给 loss，从 logit 差 $\boldsymbol{\delta} = \mathbf{z}_{\mathrm{S}} - \mathbf{z}_{\mathrm{T}}$ 出发，全程用 $\operatorname{expm1}(x) = e^x - 1$ 和 $\operatorname{log1p}(x) = \log(1 + x)$ 这两个专为"0 附近精度"设计的原语，在 log 空间算出 $\mathbf{q} - \mathbf{p}$。具体分三步：先把 $\boldsymbol{\delta}$ 在 $\mathbf{p}$ 下中心化得到 $d_k$，再算对数概率比 $s_k = \log(q_k / p_k)$，最后还原概率差 $u_k = q_k - p_k$：

$$
d_k = \delta_k - \sum_i p_i \delta_i, \qquad
s_k = d_k - \operatorname{log1p}\left(\sum_i p_i \operatorname{expm1}(d_i)\right), \qquad
u_k = p_k \operatorname{expm1}(s_k)
$$

由于 $\sum_i p_i d_i = 0$ 且 $\operatorname{expm1}(x) \ge x$，log1p 的输入恒为非负，不会触碰负数取对数的险区。最后再设 $g_k = u_k - q_k \sum_i u_i$，强制抹掉残余的浮点漂移，保证 $\sum_k g_k = 0$。整条链路没有任何"$+1$"或"$-1$"的裸加减，因此 $\mathbf{z}_{\mathrm{S}} \to \mathbf{z}_{\mathrm{T}}$ 时梯度保持精确，且 $\mathbf{z}_{\mathrm{S}} = \mathbf{z}_{\mathrm{T}}$ 时严格等于零向量。

JAX 实现（借助 `stop_gradient` 返回前向 loss、替换反向梯度）：

```python
def soft_target_cross_entropy(z_S, z_T):
  log_q = log_softmax(z_S)
  q, p = jnp.exp(log_q), softmax(z_T)
  delta = z_S - z_T                                                     # logit 差
  d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)                # 在 p 下中心化
  s = d - jnp.log1p(jnp.sum(p * jnp.expm1(d), axis=-1, keepdims=True))  # log(q / p)
  u = p * jnp.expm1(s)                                                  # 概率差 q - p
  g = u - q * jnp.sum(u, axis=-1, keepdims=True)                        # 保证 sum(g) = 0
  sg = jax.lax.stop_gradient
  loss = -jnp.sum(p * sg(log_q), axis=-1)                               # 前向 loss
  return loss + jnp.sum(sg(g) * (z_S - sg(z_S)), axis=-1)               # 自定义梯度 g
```

效果对比：令 $\mathbf{z}_{\mathrm{T}} = [2.0, 1.0, 0.0]$，给学生加上 $10^{-7}$ 量级的微扰。标准实现把第二个梯度分量算小了 3 倍、第三个分量连符号都翻错了，梯度和也不为零；修复实现则逐位吻合 fp64 精确梯度：

```python
z_T = jnp.array([2.0, 1.0, 0.0], dtype=jnp.float32)
z_S = z_T + jnp.array([-1e-7, 1e-7, 0.0], dtype=jnp.float32)
std_soft_ce = lambda z_S, z_T: -jnp.sum(jax.nn.softmax(z_T) * jax.nn.log_softmax(z_S))
jax.grad(std_soft_ce)(z_S, z_T)               # -> [-5.960464e-8, 1.451589e-8, -1.123343e-9]
jax.grad(soft_target_cross_entropy)(z_S, z_T) # -> [-4.595504e-8, 4.144190e-8,  4.513142e-9]
# fp64 精确梯度:                                 [-4.595504e-8, 4.144190e-8,  4.513142e-9]
```

## 四、破案：Slingshot 弹弓效应竟是浮点 Bug

修复方案有了，它到底能治什么病？这就要说到 Grokking（顿悟）研究里著名的**Slingshot Mechanism**（弹弓机制）。

Grokking 指的是网络先飞速记住训练集（$q_t \to 1$），很久之后才突然泛化的现象。Thilak 等人在 2022 年观察到：记忆完成之后的训练会在"接近零的平坦损失"和"突如其来的损失尖峰"之间交替，每次尖峰恰好伴随测试准确率的跃升。这个现象被反复复现，衍生出各种解释和缓解手段，但本文证明：**Slingshot 尖峰就是 Softmax Collapse 的直接后果**。

机制链条是这样的。回忆 AdamW 的更新规则：参数减去 $\eta\big(\hat{\mathbf{m}} / (\sqrt{\hat{\mathbf{v}}} + \epsilon) + \lambda\boldsymbol{\theta}\big)$，其中 $\hat{\mathbf{m}}$、$\hat{\mathbf{v}}$ 是梯度及其平方的偏差修正滑动平均。记忆完成后 margin 持续增长，一旦某样本的 $m$ 越过精度悬崖 $c_d$，标准交叉熵把它的目标梯度舍入为零，而非目标梯度仍为正。当 batch 里所有样本都越过悬崖，参数梯度几乎消失，Adam 的二阶矩 $\hat{v}_i$ 随之指数衰减，同时权重衰减慢慢压缩参数。终于，少数样本被拉回悬崖之下，目标梯度 $-e^{-m}$**突然恢复**——而此时 $\sqrt{\hat{v}_i}$ 已经衰减得非常小，Adam 用恢复的正常梯度除以一个极小的分母 $\sqrt{\hat{v}_i} + \epsilon$，产生一次巨大的参数步进，损失应声飙升，$\hat{v}_i$ 被重置，然后循环重演。

用修复后的交叉熵，目标梯度 $-e^{-m} / (1 + e^{-m})$ 在悬崖处不再归零，$\hat{\mathbf{m}}_i$ 与 $\sqrt{\hat{v}_i}$ 同步平滑衰减，尖峰彻底消失。

![Slingshot 图例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig_slingshot_legend.png)

![Slingshot：logit margin 对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig_slingshot_a.png)

> 图解：纵轴为 logit margin $m$，横轴为训练步数；上排是 $a + b \bmod 31$（$\lambda = 1$，fp32），下排是 $a^2 + b^2 \bmod 97$（$\lambda = 0$），每种颜色包含 5 个随机种子。上排：橙色标准 CE 抵达 fp32 悬崖 $c_{32} \approx 16.6$ 后反复跌落反弹（锯齿状），蓝色修复 CE 平滑越过悬崖并饱和在 $m \approx 18.4$。下排：无权重衰减时，两者原本同步爬升，在约第 1250 步橙色曲线撞上悬崖后猛地向上"弹"到 $m \approx 45$，蓝色曲线则继续平稳线性增长。

![Slingshot：训练损失对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig_slingshot_b.png)

> 图解：纵轴为训练损失（对数刻度）。上排中橙色标准 CE 呈现周期性尖峰——每次尖峰对应一次"越过悬崖 → 梯度消失 → 被拉回 → 大步进"的循环；蓝色修复 CE 平滑衰减到 $e^{-c_{32}}$ 附近的平台。下排第 1250 步附近，橙色曲线出现一次幅度接近 $10^{15}$ 倍的巨大尖峰，蓝色曲线则全程平滑下降。

![Slingshot：测试准确率对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig_slingshot_c.png)

> 图解：纵轴为测试准确率。上排：修复 CE（蓝）在 5 个种子上全部更快 Grok 到 100%（$100 \pm 0\%$），标准 CE（橙）则更慢且不齐（$94 \pm 5\%$）。下排最具戏剧性：橙色标准 CE 与蓝色同步爬升到 61%，在第 1250 步的尖峰中**崩到 23% 并就此躺平**，蓝色则平滑继续到 85%。

值得一提的是，Slingshot 究竟"有害"还是"有益"，取决于权重范数是否被控制。若网络在输出层前有 LayerNorm 且无权重衰减，每次尖峰膨胀 $\|\boldsymbol{\theta}\|_2$ 相当于一次"意外的学习率衰减"，反而可能帮泛化——这也是早期文献觉得尖峰是好事的原因。一旦权重范数已被权重衰减管住，尖峰就只剩纯粹的伤害。笔者认为，这个归因非常漂亮：一个被研究了三年、衍生出 surrogate loss、梯度投影、fp64 转换等多种 workaround 的"训练动力学现象"，最终被证明只是一个浮点舍入 Bug。

## 五、Margin 平衡定律：$m^\star \approx \log(1/\epsilon)$

修复之后还有一个耐人寻味的现象：图 Slingshot (a) 上排中，修复 CE 的 margin 为什么恰好停在 $c_{32}$ 上方一点点的 $m \approx 18.4$？这个平台和浮点精度毫无关系，而是**权重衰减与 AdamW 归一化步长的平衡点**。

在参数不再变化的驻点，滑动平均收敛到 $\hat{m}_i = g_i$、$\hat{v}_i = g_i^2$（即 $\sqrt{\hat{v}_i} = |g_i|$），令更新量为零得：

$$
\frac{g_i}{|g_i| + \epsilon} + \lambda \theta_i = 0 \quad \forall i
$$

注意 $\eta$、$\beta_1$、$\beta_2$ 全部约掉了，所以 RMSProp、LaProp 与 AdamW 共享同一个驻点。记忆完成后梯度随 margin 指数收缩（$|g_i| \propto e^{-m}$）。若 $\epsilon = 0$，归一化梯度 $g_i / |g_i| = \mathrm{sign}(g_i)$ 幅值恒为 1，margin 会一直涨到梯度下溢；若 $\epsilon > 0$，当 $|g_i| \ll \epsilon$ 时归一化步长缩为 $g_i / \epsilon \propto e^{-m}$，与权重衰减 $\lambda \theta_i$ 平衡要求 $|g_i| \approx \epsilon \lambda |\theta_i|$——相当于一个强度为 $\epsilon\lambda$ 的 $L_2$ 正则。于是 margin 停止增长的位置是：

$$
m^\star(\epsilon) \approx \log \frac{1}{\epsilon}
$$

这条定律顺手解释了为什么默认的 $\epsilon = 10^{-8}$ 会在 fp32 中触发 Slingshot：平衡点 $\log(1/\epsilon) \approx 18.4$ 恰好落在 fp32 悬崖 $c_{32} \approx 16.6$ 的**上方**，标准交叉熵在到达平衡之前先撞上悬崖；而 $\epsilon \ge 10^{-7}$ 时 $\log(1/\epsilon) \le 16.1 < c_{32}$，标准实现反而能安稳落地。一个被广泛当作"无害默认值"的超参数，就这样和浮点精度悬崖构成了完美风暴。

![归一化 Adam 步长图例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig1_legend.png)

![归一化 Adam 步长随 margin 的变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig1b_adam_eps.png)

> 图解：横轴为 logit margin $m$，纵轴为归一化 Adam 步长 $|g_t| / (|g_t| + \epsilon)$（$\epsilon = 10^{-8}$）。修复 CE（蓝）的步长在 $m$ 较小时保持在 1 附近，随后在 $\log(1/\epsilon) \approx 18.4$（绿色竖线）附近平滑衰减——这正是平衡点的位置；标准 CE 的目标梯度则分别在 $c_{16} \approx 5.5$（黄点线）和 $c_{32} \approx 16.6$（橙实线）处提前断崖归零，永远到不了平衡。

为了在 fp32 中验证这条横跨多个数量级的定律（直到 $\epsilon = 0$），作者改用 LaProp 并直接更新 $\sqrt{\hat{\mathbf{v}}}$，避免 $g_i^2$ 在 $|g_i| < 2^{-63}$（$m \approx 38$）时下溢。

![epsilon 阶梯实验](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig6_eps_staircase.png)

> 图解：在 $a + b \bmod 31$ 上用修复 CE + LaProp 训练（fp32，每设置 3 个种子为实线，理论预测为虚线）。(a) 有权重衰减（$\lambda = 1$）：每个 $\epsilon > 0$ 的 margin 先沿 $\epsilon = 0$ 的轨迹生长，然后精确停在各自的预测平衡点 $\log(1/\epsilon)$——$\epsilon = 10^{-8}, 10^{-15}, 10^{-24}, 10^{-32}$ 分别对应 18.4、34.5、55.3、73.7，形成漂亮的"阶梯"；$\epsilon = 0$（灰/红）则一路涨到 fp32 梯度自身下溢的 $m \approx 81.2$。(b) 无权重衰减（$\lambda = 0$）：没有力量平衡 $g_i / \epsilon$ 步长，margin 越过 $\log(1/\epsilon)$ 后继续向上漂移。理论曲线与实测一一吻合。

## 六、实验：跨精度、跨基准全面验证

### 模块化加法上的精度扫描

作者在 $a + b \bmod 31$ 上训练 1 层、宽度 64 的 ReLU Transformer（50% 训练划分，full-batch AdamW 跑 10,000 epoch，$\eta = 10^{-3}$，$\beta_1 = 0.9$，$\beta_2 = 0.98$，$\lambda = 1$），只改变三个变量：损失函数（标准 vs 修复）、浮点精度（bf16 / fp32 / fp64）、$\epsilon \in \{10^{-8}, 10^{-12}, 10^{-16}\}$。

![精度扫描图例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig2_legend.png)

![margin 随训练的变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig2a.png)

> 图解：横轴为 epoch（0–10k），纵轴为平均 logit margin $m$，上排 $\epsilon = 10^{-8}$、下排 $\epsilon = 10^{-16}$。上排：标准 CE 在 bf16（黄点线）下卡在 $c_{16} \approx 5.5$ 动弹不得，在 fp32（橙实线）下绕 $c_{32}$ 反复震荡；修复 CE（蓝色系）在三种精度下都平滑爬到平衡点 18.4。下排：平衡点升到 $\log(1/\epsilon) \approx 36.8 > c_{64}$，连 fp64 标准 CE（红虚线）也在 $c_{64} \approx 36.7$ 处断裂（延迟到约 6.5k epoch 才勉强跟上），修复 CE 依然全线平滑收敛。

![训练损失随训练的变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig2b.png)

> 图解：纵轴为训练损失（对数刻度）。修复 CE 在各精度下平滑下降到对应精度的极限平台（$\epsilon = 10^{-8}$ 时约 $e^{-c_{32}}$，$\epsilon = 10^{-16}$ 时一路降到 $e^{-c_{64}}$）；标准 CE 在 fp32 下呈锯齿震荡，bf16 下更是停在 $10^{-2}$–$10^{-3}$ 的高位下不来。

![测试准确率随训练的变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig2c.png)

> 图解：纵轴为测试准确率。修复 CE 在三种精度、两档 $\epsilon$ 下全部稳定收敛到接近 100%；标准 CE 的 fp32 曲线明显滞后且带噪声台阶，bf16 曲线则频繁出现大幅回落。

量化结果汇总如下（5 个种子）：

| 精度 | 损失变体 | Acc @ $\epsilon=10^{-8}$ (%) | Acc @ $\epsilon=10^{-12}$ (%) | Acc @ $\epsilon=10^{-16}$ (%) | 最终 $m$ @ $10^{-8}$ | 最终 $m$ @ $10^{-12}$ | 最终 $m$ @ $10^{-16}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | Fixed CE | 100.0 ± 0.0 | 99.6 ± 0.7 | 99.0 ± 2.3 | 17.9 | 27.4 | 36.8 |
| fp32 | Standard CE | 94.3 ± 5.4 | 96.4 ± 5.1 | 96.5 ± 4.3 | 13.0 | 15.1 | 12.7 |
| bf16 | Fixed CE | 100.0 ± 0.1 | 99.8 ± 0.2 | 98.9 ± 2.4 | 17.9 | 27.4 | 36.8 |
| bf16 | Standard CE | 93.3 ± 7.4 | 94.9 ± 4.9 | 94.2 ± 8.3 | 5.1 | 5.6 | 5.8 |
| fp64 | Fixed CE | 100.0 ± 0.0 | 99.5 ± 1.1 | 99.0 ± 2.0 | 17.9 | 27.4 | 36.8 |
| fp64 | Standard CE | 100.0 ± 0.0 | 100.0 ± 0.1 | 97.2 ± 3.2 | 17.9 | 27.4 | 30.1 |
| fp32 | Zero-sum Proj. | 97.9 ± 3.2 | 48.9 ± 2.6 | 49.3 ± 2.5 | 17.0 | 18.9 | 18.9 |
| fp32 | StableMax | 97.3 ± 4.7 | 95.6 ± 2.0 | 97.3 ± 4.3 | 3200 | 5661 | 6313 |
| fp64 | StableMax | 97.4 ± 4.2 | 96.7 ± 5.3 | 95.6 ± 2.7 | 3170 | 21252 | 2322 |

> 表解：Fixed CE 在全部精度与 $\epsilon$ 组合下达到预测的平衡 margin（18.4 / 27.6 / 36.8 附近）和 99–100% 准确率；Standard CE 的 margin 一律卡在自家精度悬崖附近（bf16 约 5.5、fp32 约 16.6、fp64 约 36.7）。两个对照方法也各有硬伤：Zero-sum Proj.（给每个 logit 梯度减去均值以强制零和）在 $m > c_{32}$ 后只能把非目标梯度的 $1/K$ 还给目标 logit，margin 停滞在 18 附近，$\epsilon \le 10^{-12}$ 时准确率跌破 50%；StableMax（用分段有理函数替代 $e^x$）因为是线性增长而非指数增长，margin 膨胀到 $10^3$–$10^4$ 的离谱量级，准确率也全面低于修复方案。

### 四个 Grokking 基准上的一致性

在 $\epsilon = 10^{-12}$ 下，作者复现了文献中的四个经典 Grokking 设置：1 层 Transformer 跑 $a + b \bmod 113$ 与 $a + b \bmod 31$（$\lambda = 1$），2 层 MLP 跑 $a^2 + b^2 \bmod 97$ 与 $a^3 + b \bmod 97$（$\lambda = 0.2$）。

![四个基准的图例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig3_legend.png)

![a+b mod 113](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig3a.png)

![a+b mod 31](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig3b.png)

![a²+b² mod 97](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig3c.png)

![a³+b mod 97](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig3d.png)

> 图解：四个子图的上排均为测试准确率、下排均为平均 margin。三条曲线分别是 fp64 标准 CE（红虚线，作为"无 Bug 基准"）、fp32 修复 CE（蓝实线）、fp32 标准 CE（橙实线）。四个任务上结论一致：橙色曲线一旦撞上 $c_{32}$ 就反复震荡、准确率抖动或中途回落（c、d 两个 MLP 任务上约 1.4k–1.5k 步处可见明显的准确率下陷和 margin 过冲）；而 fp32 修复 CE 的曲线与 fp64 标准 CE 几乎重合，均达到 100% 测试准确率。也就是说，**修复后的 fp32 训练在效果上等价于 fp64**，但不需要付出双精度的显存和算力代价。

## 七、蒸馏实验：幽灵梯度如何拖走学生

最后验证 Softmax Cancellation 的实际危害。实验设计很干净：把在 $a + b \bmod 53$ 上训练到 100% 准确率的老师权重直接拷贝给学生（$\boldsymbol{\theta}_{\mathrm{S}}^{(0)} = \boldsymbol{\theta}_{\mathrm{T}}$），以蒸馏温度 $T = 4$ 跑 2,000 步 Adam（$\eta = 2 \times 10^{-3}$，$\lambda = 0$，$\epsilon = 10^{-8}$）。由于初始时 $\mathbf{z}_{\mathrm{S}} = \mathbf{z}_{\mathrm{T}}$，真实梯度为零，学生理应原地不动。

但标准软目标交叉熵在 $\mathbf{z}_{\mathrm{S}} = \mathbf{z}_{\mathrm{T}}$ 处存在残余梯度（前向 softmax 与反向 log-softmax 的运算路径不一致所致）。一旦残余梯度超过 $\epsilon$，Adam 就会把它归一化成完整步长 $\eta$——**梯度虽小，步子不小**。结果按精度分层：fp64 残余约 $10^{-17}$，远低于 $\epsilon = 10^{-8}$，学生安然不动；fp32 残余约 $10^{-8}$，引发一次短暂的 KL 尖峰；bf16 残余高达约 $10^{-3}$，直接把孩子从老师身边"拐跑"，准确率崩塌。

![蒸馏实验图例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig4_legend.png)

![蒸馏过程的 KL 散度](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig4b_kl.png)

> 图解：横轴为蒸馏步数（0–2k），纵轴为留出集上的 $\mathrm{KL}(\mathbf{p}_{\mathrm{T}} \| \mathbf{q}_{\mathrm{S}})$（对数刻度）。修复软目标 CE（蓝色系）在三种精度下全程钉在 KL = 0；标准实现中，fp64（红虚线）基本贴底，fp32（橙实线）开局出现约 $10^{-3}$ 的尖峰后回落到 $10^{-6}$–$10^{-5}$ 的残余水平，bf16（黄点线）则一路飙到 $10^0$ 附近且噪声剧烈——学生分布被假梯度彻底带偏。

![蒸馏过程的测试准确率](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/A-Fix-for-Softmax-and-Cross-Entropy/paper_fig4c_acc.png)

> 图解：纵轴为测试准确率。修复 CE 全线保持 100%；标准 CE 的 fp32 / fp64 也维持在 100%，但 bf16 在约 400 步后开始劣化，在 45%–90% 之间剧烈震荡。这提醒我们：低精度蒸馏场景下，这个 Bug 不是理论洁癖，而是实打实的准确率杀手。

## 八、总结

- softmax / log-softmax / 交叉熵的标准实现存在两个数值 Bug：Softmax Collapse 在 margin 越过精度悬崖（bf16 ≈ 5.5、fp32 ≈ 16.6、fp64 ≈ 36.7）后把目标梯度舍入为零，破坏梯度零和性；Softmax Cancellation 在蒸馏中因概率对消产生"幽灵梯度"，学生等于老师时梯度仍非零。
- 修复思路一句话：让微小量始终在 0 附近求值，永远不要把它加到 1 上——用 softplus / expm1 / log1p 在 log 空间重写，十几行代码即可即插即用，且无需内部转换到更高精度。
- Grokking 研究中的 Slingshot 损失尖峰被证明是 Softmax Collapse 与 Adam 二阶矩衰减共同作用的产物，修复后尖峰彻底消失：无权重衰减时测试准确率从"61% 崩到 23%"变为平滑收敛到 85%。
- 修复后揭示出一条干净的 margin 平衡定律：AdamW + 权重衰减下 logit margin 收敛到 $m^\star \approx \log(1/\epsilon)$，并在跨越 24 个数量级的 $\epsilon$ 扫描中逐点验证。
- 修复版 fp32 交叉熵在四个 Grokking 基准上与 fp64 标准实现曲线重合、达到 99–100% 准确率；蒸馏实验中修复版在 bf16 / fp32 / fp64 下全部保持 KL = 0 与 100% 准确率。

展望：作者坦承，极端 margin 和近乎为零的师生差距在常规大规模训练中并不常见，因此这个补丁更像是"基础设施级"的保险——成本几乎为零，却能在模型高置信度预测、近距离蒸馏、低精度训练等场景悄悄兜底。考虑到 Grokking 文献里已有大量基于 fp64 转换或替代损失的 workaround，这项工作也提示我们：面对诡异的训练动力学，先怀疑数值实现，再发明新理论。

> 本文参考自 [A Fix for Softmax and Cross-Entropy](http://arxiv.org/abs/2610.31642v1)