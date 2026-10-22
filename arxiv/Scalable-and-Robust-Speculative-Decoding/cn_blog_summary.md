# Sequoia：可扩展、鲁棒的投机解码，LLM 推理最高提速 9.5 倍

大模型推理慢的根源是 I/O 瓶颈：每生成一个 token 都要把全部参数从显存读一遍。投机解码用小模型"猜"、大模型"验"，但现有方法在猜测预算变大时收益饱和，且对温度敏感。ICML 2024 论文提出的 **Sequoia** 一举击破这两个"天花板"：在 A100 上对 Llama2-7B 实现最高 **4.04×** 加速，在 L40 offloading 场景下对 Llama3-70B-Instruct 实现最高 **9.5×** 加速，单 token 延迟从 5.7 秒压到 **0.60 秒**。

## 两个天花板

- **树结构收益有上界**：k 条独立序列（SpecInfer）或满 k 叉树（Medusa）的期望接受 token 数都有理论上限，树再大也饱和——offloading 这类极端 memory-bound 场景本应用大树，旧结构却接不住红利。
- **验证算法不鲁棒**：SpecInfer/SpecTr 有放回采样，低温下同一错误 token 被反复提出、反复拒绝；naive/top-k 采样则低温好、高温平庸，没有一种方法在所有温度下都能打。

## 三个组件

- **DP 构造最优 token 树**：引入"接受率向量"假设（接受概率只取决于孩子是第几个），证明期望生成 token 数有闭式表达 $F(\mathcal{T})=\sum_v f(v)$，再用动态规划求解"大小为 N 的最优树"，离线运行不拖慢推理。理论上期望收益达 $\Omega(b\log n/\log\log n)$，随预算**无界、对数增长**——彻底打破旧结构的饱和上限。
- **无放回采样验证**：以 SpecInfer 为骨架只改两处——被拒绝的 token 直接从 draft 分布中置零、绝不再提；支撑集被全拒时兜底用未拒绝 token 上的均匀分布。它是**第一个同时满足"最优传输"（高温好）与"覆盖"（低温好）两条性质的验证算法**，且严格保持目标模型输出分布。
- **硬件感知优化器**：按 $\texttt{Speedup}(n,d)=G(n,d)/(t(n)+d\cdot c)$ 实测目标硬件的验证耗时曲线，自动选出最优树的大小与深度。

## 实验结果

- **A100 on-device**：Llama2-7B 最高 4.04×（TBT 6.0 ms/token），相对 SpecInfer 最优配置（扫遍所有宽×深组合也只有 3.70×）仍领先，赢的是结构搜索本身。
- **L40 offloading**：Llama3-70B-Instruct 最高 9.5×、0.60 s/token，比 SOTA 系统 DeepSpeed-Zero-Inference 快近一个数量级；选出的树达 768 节点（on-device 仅 64–128），印证"越 memory-bound，树越该大"。
- **消融**：预算 512 时 Sequoia 树比 k 条独立序列多生成 33% token 且持续对数增长；全温度扫描中相对 SpecInfer 最高快 1.65×、相对 top-k 最高快 1.27×。

## 局限

最优性依赖"位置接受"与幂律两条简化假设；最优树随模型对、温度、数据域、batch size 变化，每种设置需先实测接受率再建树，作者的经验是一棵典型树通常可迁移。

代码已开源。Sequoia 把投机解码的"结构上限"上推一个量级，其接受率幂律与硬件权衡也为推理系统设计提供了理论坐标。

> 本文参考自 [Sequoia: Scalable and Robust Speculative Decoding](https://arxiv.org/abs/2402.12374)