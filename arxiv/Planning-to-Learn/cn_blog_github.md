# Planning to Learn：把训练预算规划进损失函数

分类任务里没有探索、没有信用分配、没有动作采样噪声——强化学习失败的三大"惯犯"全都不在场，可是直接优化期望准确率的 **精确策略梯度** 依然被交叉熵打得落花流水：ImageNet ResNet-50 上，同样学习率下前者只有 4.2% 期望准确率，后者 62%。来自 Google DeepMind 的 Ian Osband 在这篇论文中给出了一个漂亮的解释： **梯度是短视的，它只算"这一步买什么"，却看不到"训练还剩多少预算"** 。顺着这个思路，作者把学习本身建模为一个序列资源分配问题，提出只需改动一行的 horizon loss：它在 ImageNet 上以恒定学习率把 top-1 准确率提升 1.0–2.4 个百分点，且标签噪声越大优势越明显（50% 噪声时提升 7.3 个点）。

## 提出问题：精确策略梯度为何输给交叉熵

策略梯度方法是现代强化学习的核心，也是 LLM 后训练（post-training）的主力。当 RL 训练出问题时，大家通常怀疑三件事：探索不足、信用分配困难、动作采样噪声。

但分类问题把这三种困难全部排除了：

- 每个样本的标签已知，奖励立即到账；
- 所有动作（类别）都可以枚举求和，梯度 **无需采样、可以精确计算** ；
- 分类器本身就是一个策略：softmax 采样一个标签，猜对得奖励 1，期望奖励就是期望准确率。

于是我们可以写出期望准确率 $p$ 的精确梯度，并和交叉熵的负梯度并排比较：

$$
\nabla_\theta p = p(1-p)\,\nabla_\theta z, \qquad -\nabla_\theta \mathrm{CE} = (1-p)\,\nabla_\theta z
$$

其中 $z = f_y - \log \sum_{k \ne y} e^{f_k}$ 是 **logit gap** （正确类别 logit 与其余类别 logsumexp 之差），$p = \sigma(z)$。两个更新方向完全相同，区别只在 **力度** ：精确策略梯度按 $p(1-p)$ 加权，交叉熵按 $1-p$ 加权。

![训练曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/curves.png)

> 图解：MNIST（MLP）与 ImageNet（ResNet-50）上测试误差（对数坐标）随训练的变化，所有方法共享同一个恒定学习率。可以看到精确策略梯度（exact PG）大幅落后于交叉熵——即使它优化的就是期望准确率本身；而本文提出的 horizon loss 在训练后期反超交叉熵。

这就引出了全文的核心疑问： **既然交叉熵只是代理损失（surrogate），为什么精确优化目标本身反而输了？** 值得注意的是，传统分类理论把交叉熵的优势归因于"光滑凸代理"，但期望准确率本身就是光滑的，且两个损失对网络参数都非凸——这个解释在这里说不通。

## 分析问题：学习本质上是预算分配

作者的回答很干脆：梯度回答的是"哪一步无穷小改动最能提升 **当前** 准确率"，但训练要跑很多步，每一步都决定了下一步的起点。 **一步更新的价值，取决于训练还剩多少。**

一个直观的例子：两个样本的正确标签概率分别是 0.9 和 0.01，提升两者 log-odds 的成本相同。

- 梯度偏爱前者，因为同样一步，它的准确率提升约为后者的 9 倍（$0.9 \times 0.1 = 0.09$ 对 $0.01 \times 0.99 \approx 0.01$）；
- 如果训练快结束了，这是对的——只有前者来得及兑现收益；
- 但如果预算充足，钱花在后者的回报大得多：它起点低，可被一路推到五五开，总提升空间远超前者。

把这个直觉形式化，就是论文的 **分配模型（allocation model）** ：$N$ 个独立样本，gap 为 $z = (z_1, \ldots, z_N)$，学习器共有 $H$ 单位的"logit 进度"可以分配（代表剩余训练量），目标是最大化终局的总期望准确率：

$$
V_H^\star(z) = \max_{u \ge 0,\ \|u\|_1 \le H} \sum_i \sigma(z_i + u_i)
$$

一个加权规则 $w$ 按比例连续地花费预算：

$$
\dot z_i(\tau) = \frac{w(z_i(\tau), h_\tau)}{\sum_j w(z_j(\tau), h_\tau)}, \qquad h_\tau = H - \tau
$$

作者证明：当各样本梯度正交归一时，精确策略梯度就是这个流中 $w_{\mathrm{PG}}(z) = \sigma'(z)$ 的特例，交叉熵则对应 $w_{\mathrm{CE}}(z) = 1 - \sigma(z)$。 **于是一个损失函数好不好，问题就变成了它的权重函数会不会分配预算。**

![反转现象](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/reversal.png)

> 图解：把一个容易样本（gap 为 $a$）和一个困难样本（gap 为 $-2a$）放在一起，横轴是预算 $H$，纵轴是把全部预算砸给某一个样本后的期望准确率收益。$H < a$ 时应该全给容易样本，$a < H \le 3a$ 时应该全给困难样本—— **最优分配随预算反转** ，而精确策略梯度的权重里根本没有 $H$，它永远做不出这个反转。

## 解决问题：交叉熵是"耐心准确率"，horizon loss 是"有限耐心的准确率"

接下来是全文最漂亮的推导。假设把剩余所有学习都给一个样本，它的 gap 会以单位速度上升 $z \to z + s$，沿途持续支付误差 $1 - \sigma(z+s)$。把未来 $H$ 单位学习内的总误差积分起来：

$$
C_H(z) = \int_0^H [1-\sigma(z+s)]\,ds = \mathrm{CE}(z) - \mathrm{CE}(z+H)
$$

当 $H \to \infty$ 时第二项消失，得到：

$$
\mathrm{CE}(z) = \int_0^\infty [1-\sigma(z+s)]\,ds
$$

**交叉熵等于"假设训练永不停止、这个样本以单位速度学下去，沿途要支付的总误差"。** 这就是"耐心准确率"（patient accuracy）的含义——它按无限预算给每个样本定价。笔者认为这个视角的聪明之处在于：它把"为什么交叉熵有效"从静态的凸代理理论，转译成了动态的预算规划语言。

但现实中预算是有限的。对有限 $H$ 做归一化，就得到 **horizon loss** ：

$$
L_H(z) = \frac{C_H(z)}{1 - e^{-H}}
$$

它的负梯度恰好因子化为一个优美的形式：

$$
-\partial_z L_H(z) = \frac{\sigma(z+H) - \sigma(z)}{1-e^{-H}} = \sigma(z+H)\,(1-\sigma(z))
$$

即 **horizon 权重 = 剩余误差 × 在剩余预算内成功的概率** 。两个端点一目了然：$H \to 0$ 时退化为 $\sigma'(z)$（精确策略梯度），$H \to \infty$ 时退化为 $1-\sigma(z)$（交叉熵）。精确策略梯度向前看零步，交叉熵向前看无穷远， **只有 horizon loss 看向真实的剩余预算** 。

![look-ahead](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/chord.png)

> 图解：从 $z = -4$ 的样本出发的三种"向前看"。精确策略梯度用当前局部斜率 $p(1-p)$；有限 horizon 用未来 $H$ 单位学习带来的实际提升 $\sigma(z+H) - p$；交叉熵用全部剩余提升 $1-p$。

![权重对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/price.png)

> 图解：$H = 4$ 时三种规则对 $\nabla z$ 的权重随 gap 的分布。精确策略梯度把权重集中在五五开附近（不区分样本在胜负线的哪一侧）；交叉熵几乎把最大权重给了毫无希望的样本；horizon loss 精准瞄准"在剩余预算内够得着"的样本。

剩下的问题是：神经网络里"剩余 logit 进度"事先未知。作者用一个极其简单的估计：

$$
H_t = \kappa \sum_{t \le t' < T} \eta_{t'}
$$

即 **剩余学习率之和乘以一个校准常数 $\kappa$** （每个数据集和调度器只校一次：MNIST 上 $\kappa = 1$，ImageNet 恒定学习率下 $\kappa = 3$）。恒定学习率下 $H_t$ 线性下降到零，损失函数自然地从交叉熵平滑滑向精确策略梯度。

更一般地，同样的构造适用于任意逐样本损失——作者定义了 **规划算子** $P_H[\ell](z) = \int_0^H \ell(z+s)\,ds$，把它作用在交叉熵上就得到后文的 horizon cross-entropy。

## 理论保证：两个陷阱与一个逃脱者

作者用两个玩具问题把两个端点的失败方式钉死了：

- **抛光陷阱（polishing trap）** ：从 $(a, -2a)$ 出发、预算 $H = 2a$，最优解应把全部预算给困难样本。精确策略梯度却把容易样本从 $a$ 抛光到 $2a$，只拿到最优收益的 $4\sigma'(a)$ 份额—— **随 $a$ 指数级消失** 。因为它分不清"已经会了"和"还来得及救"，$\sigma'(a) = \sigma'(-a)$，两侧一视同仁。
- **绝症陷阱（lost-cause trap）** ：一个五五开的样本加 $M$ 个 gap 为 $-a$ 的"绝症样本"，预算 $0 < H < a$ 救不活任何一个绝症样本。交叉熵的权重 $1 - \sigma(z)$ 恰恰在最没希望的样本上最大，它能拿到的收益至多为 $\frac{H}{4M+4} + He^{H-a}$——$M$ 越大份额越趋近于零。

而 horizon loss 的权重 $\sigma(z+h)(1-\sigma(z))$ 在两个极端都趋近于零：已解决的样本没剩多少误差，绝症样本在剩余预算内没有成功概率。作者证明它在两个陷阱上的遗憾（regret）分别不超过 $2ae^{-a}$ 和 $HMe^{H-a}$—— **随干扰项移出 horizon 指数缩小，随其数量只线性增长** 。

![理论示意](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/summary.png)

> 图解：相对交叉熵的测试误差下降幅度（左：期望误差；右：top-1 误差），实心为恒定学习率、空心为 cosine 衰减。horizon loss 在图中所示的每一个模型和调度下平均误差都低于交叉熵；cosine 衰减会压缩收益。

当然，理论也有边界：horizon loss 是 **可分离的（separable）** ，它给每个样本定价时都假装全部预算归它独享。附录中证明，对两个完全相同的样本，任何确定性可分离规则都必须平分预算，而集中投资一个更优——这是所有可分离规则绕不开的天花板。

## 实验验证

理论模型假设样本独立、预算固定，真实网络共享参数、用 AdamW 训练。实验要回答的就是： **分配原理在模型之外还管用吗？** 设置上，同一 recipe 内只换损失函数（精确策略梯度 / 交叉熵 / horizon loss），默认恒定学习率，MNIST 跑 30 个种子，ImageNet（ResNet-50、ResNet-101、ViT-S/16）跑 3–10 个种子。

### Planning 同时击败两个端点

恒定学习率 $10^{-3}$ 下，ResNet-50 的期望准确率从精确策略梯度的 4.2% 跃升到 horizon loss 的 67.5%；交叉熵只有 62.1%。在 MNIST 上，horizon loss 也比精确策略梯度降低 24% 期望误差、29% top-1 误差。

更难的对照是交叉熵：horizon loss 在全部四个模型上降低 top-1 误差——ImageNet 上 1.0–2.4 个点，MNIST 上 $0.16 \pm 0.04$ 个点。下表是 ImageNet 恒定学习率的核心结果（%）：

| 模型 | 方法 | 期望准确率 | Top-1 |
| --- | --- | --- | --- |
| ResNet-50 | 交叉熵 | 62.1 | 67.4 |
| ResNet-50 | horizon loss（$\kappa=3$） | **67.5** | **69.6** |
| ResNet-101 | 交叉熵 | 64.4 | 69.0 |
| ResNet-101 | horizon loss（$\kappa=3$） | **69.3** | **71.4** |
| ViT-S/16 | 交叉熵 | 59.4 | 65.0 |
| ViT-S/16 | horizon loss（$\kappa=3$） | **64.1** | **66.0** |

收益 **出现得很晚** ，这正符合理论预测：ResNet-50 上初始 horizon $H_0 \approx 450$，远超样本 gap 的尺度，训练大部分时间里 $\sigma(z + H_t) \approx 1$，horizon loss 几乎就是交叉熵；只有最后几个百分点、$H_t$ 降到 gap 尺度时两条曲线才分开。另外值得诚实指出的一点：手工切换（前期交叉熵、后期切精确策略梯度）在部分设置下能追平甚至略超 horizon loss，说明 **收益的大头来自"训练后期向精确策略梯度靠拢"这件事本身** ，而非 horizon 权重的精确形式。

### 标签噪声越大，收益越大

被翻转的标签会制造"消耗训练却损害干净测试准确率"的样本。一旦网络学出某个错标样本的真实类别，错标签的概率就变低；随着 horizon 收缩，这些样本会掉出预算之外成为"绝症"。交叉熵会继续砸钱，horizon loss 会放手。

实验完全验证了这一预测：ResNet-50 上 top-1 收益从 10% 噪声时的 3.3 个点单调升到 50% 噪声时的 7.3 个点；MNIST 上从 0.1 升到 2.7 个点。

![噪声实验](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/noise_imagenet.png)

> 图解：ResNet-50 上 top-1 测试误差随训练标签翻转比例的变化（测试标签保持干净）。噪声越大，horizon loss 相对交叉熵的优势越大；到 50% 噪声时差距已达 7.3 个点。

![记忆曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/noise_flipped.png)

> 图解：MNIST 30% 噪声下，模型分配给错误标签的平均概率随训练的变化。交叉熵早期下降后一路回升——网络在背诵错标签；horizon loss 前半程跟随交叉熵，后半程把它压到交叉熵终值的约三分之一。

一个有趣的旁证：端点对比也反转了。无噪声时精确策略梯度比交叉熵差，但从 20% 噪声开始它反而更好—— **绝症样本一多，"无视它们"就开始值钱了** 。而 horizon loss 在所有噪声水平下同时击败两者。

### Horizon 必须先长后短

如果收益真的来自"按剩余预算规划"，那么固定 horizon 或随时间增长的 horizon 应该无效。实验确认：MNIST 上常数和增长的 horizon 最多带来边际收益，小 $\kappa$ 时甚至输给交叉熵；只有收缩的 horizon 在所有 $\kappa \ge 1$ 下稳定获胜。

![horizon形状](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/shape.png)

> 图解：MNIST 上收缩、常数、增长三种 horizon 形状（按均值对齐）在不同 $\kappa$ 下的最终测试误差，虚线为交叉熵。只有收缩 horizon 在整个测试范围内稳定带来收益。

![固定horizon](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/fixed_h.png)

> 图解：ResNet-50 上全程固定 horizon $H$ 的最终测试误差，虚线为交叉熵与收缩 horizon loss。$H \ge 10$ 与交叉熵打平（对几乎所有样本 $\sigma(z+H) \approx 1$），$H \le 3$ 从第一步起就是短视的、明显更差—— **没有任何固定 $H$ 能复制收缩 horizon 的收益** 。

### 把"规划"作用到交叉熵上

规划算子可以用于任意损失。把交叉熵同样地"规划"一遍得到 horizon cross-entropy，它把 ImageNet 测试 NLL 从 1.467 降到 1.364，MNIST 从 0.103 降到 0.089。

![horizon交叉熵](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/Planning-to-Learn/figures/hxent_nll_imagenet.png)

> 图解：ImageNet（ResNet-50）恒定学习率下测试 NLL 随训练的变化。horizon cross-entropy（$\kappa = 10$）随 $H_t$ 收缩在训练后期拉开与交叉熵的差距；而为准确率设计的 horizon loss 反而会让 NLL 变差。

注意两者方向相反：NLL 无界，离五五开越远的样本"可消除的 NLL"越多，规划 NLL 会相对 **上调** 困难样本；而准确率有界，规划准确率会 **放弃** 绝症样本。收益也是目标专属的——horizon cross-entropy 会把 ImageNet 期望误差从 0.378 推高到 0.393，horizon loss 则推高 NLL。 **一句话：想让哪个指标好，就规划哪个目标。**

### 附录中的更多稳健性证据

- **$\kappa$ 的盆地很宽** ：MNIST 和 ResNet-50 上，$\kappa$ 从 0.3 到 100 都优于交叉熵；$\kappa = 100$ 仍保留 40% 以上的 ImageNet 收益。
- **学习率扫描** ：MNIST 上 horizon loss 在所有测试学习率下击败交叉熵；ResNet-50 上精确策略梯度在 $10^{-4}$ 到 $10^{-2}$ 的所有学习率下都不超过 5% 期望准确率。
- **Focal loss 对照** ：ResNet-50 上所有测试的 $\gamma \in \{1, 2, 5\}$ 都比交叉熵更差——静态的难度加权无法替代预算规划。
- **宽度与预算扫描** ：MNIST 上期望误差收益在所有宽度下成立，top-1 收益随宽度增长；1k 到 100k 步的所有训练预算下 horizon loss 都获胜。

## 相关工作：一个损失家族的新坐标

固定 $H$ 时这个损失本身并不新——它是 scaled skew divergence，$H = 0$ 时就是已知训练效果很差的 categorical MAE，generalized cross-entropy 可以写成 horizon 上的 Beta 混合，作者此前的 Delightful Policy Gradient 恰好对应 $H = \log 2$ 的固定成员。 **本文的贡献不是这个函数形式，而是给 $H$ 一个语义：它就是剩余学习预算，并给出它应随训练收缩的理由。**

与 RL 后训练的联系也很直接：二值奖励下 prompt 的策略梯度同样带 $p(1-p)$ 权重，DAPO、Dr. GRPO 改变了梯度在 prompt 间的分配方式，但都没有利用"训练还剩多少"。Maximum likelihood RL 在期望奖励与似然之间按采样数插值，horizon loss 则按剩余预算插值、随训练收尾移向期望奖励——方向恰好互补。Focal loss、课程学习、难例挖掘按当前难度加权，RHO-LOSS 用 holdout 模型挑出"可学而未学"的样本，horizon loss 只用当前成功概率和剩余预算，不需要额外模型。

## 总结

- 分类是剥离了探索、信用分配与采样噪声的"纯净"策略梯度问题，精确梯度依然惨败，说明 **分配问题内生于梯度学习本身** ；
- 交叉熵 = 无限预算下的"耐心准确率"，精确策略梯度 = 零预算极限，horizon loss 按 **剩余预算** 在两者之间插值，一行改动即可实现；
- 理论上两个端点各落入一个陷阱（抛光 / 绝症），horizon loss 以指数收缩的遗憾同时逃脱两者；
- ImageNet 恒定学习率下 top-1 提升 1.0–2.4 点， **标签噪声越大收益越大** （50% 噪声时 7.3 点），且收益集中在训练尾部、依赖收缩的 horizon；
- 同一规划算子可用于任意目标：规划交叉熵降 NLL（1.467 → 1.364），规划准确率降错误率，收益是目标专属的。

局限与展望：剩余学习率到 logit 进度的换算仍靠人工校准的 $\kappa$，从优化器与表示动力学中推导 horizon 是自然的下一步；更重要的是，分类里标签已知、成功概率可精确计算，而真实 RL 后训练中这些都要从部分反馈中估计—— **如何在探索与采样噪声回归的世界里"按剩余预算规划"，才是真正的开放问题** 。

> 本文参考自 [Planning to Learn](http://arxiv.org/abs/2610.03667v1)