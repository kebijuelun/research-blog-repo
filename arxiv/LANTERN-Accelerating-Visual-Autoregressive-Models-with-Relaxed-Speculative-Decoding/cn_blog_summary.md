# LANTERN：近邻 Token 放宽验收，加速图像生成

视觉自回归模型必须逐 Token 顺序解码，每一步都要执行前向传播，成为图像生成与多模态统一建模的效率瓶颈。LANTERN 利用图像 Token 在 latent space 中的视觉可替代性，放宽投机解码的严格验收，在 LlamaGen-XL Stage I 上实现最高 **2.26×（greedy decoding）** 和 **1.69×（random sampling）** 的实际加速。

## 关键诊断：不是草稿太弱，而是 Token 选择存在歧义

- 传统投机解码由小型 drafter 生成候选，大型 target 一次并行验证；候选接受概率取决于两者对同一 Token ID 的概率比，经典算法可保持输出分布不变。
- 直接搬到视觉 AR 后收益明显下降：LlamaGen-3B 的平均接受长度相比 Vicuna-7B 降低 **41%–58%**，多种方法均出现退化。
- 根本原因是视觉 AR 的下一 Token 分布更平坦。连续视觉信号经量化后，多个局部表示都可能合理，drafter 很难与 target 精确对齐到同一个索引。因此，问题不只是增强 drafter，更要重新判断“什么候选值得接受”。

## 核心方法：近邻概率聚合，加 TVD 分布约束

- **视觉近邻可互换**：latent space 中距离较近的 Token，解码后通常具有相近外观。LANTERN 以候选最近的 $k$ 个 Token 构成集合 $B_k$，聚合 target 分配给整个邻域的概率来计算接受率，不再只检查候选自身。
- **接受率显著提升**：EAGLE-2 首个草稿 Token 的平均接受概率仅为 **0.0402**；随 $k$ 与容差 $\delta$ 增大，LANTERN 最高达到 **0.3657**。
- **限制有损偏移**：概率聚合会改变 target 分布。方法按 latent 距离逐步加入近邻，并用总变差距离（TVD）衡量偏移；若继续加入会超过阈值 $\delta$，立即停止，得到最终验收集合 $A_{k,\delta}$。$k$ 控制最大范围，$\delta$ 控制质量—速度权衡。

## 实验结论：恢复加速，但并非免费提速

主实验以 EAGLE-2 为基础，在单张 RTX 3090 上测量实际速度：

- Greedy decoding：EAGLE-2 仅加速 **1.29×**，LANTERN 达到 **2.26×**，平均接受长度由 **1.60** 提升至 **2.89**。
- Random sampling：EAGLE-2 的 **0.93×** 意味着比标准 AR 更慢，LANTERN 转为 **1.69×**，平均接受长度达到 **2.40**。
- 跨模型仍有效：LlamaGen-XL Stage II 和 Anole 分别获得 **1.64×** 与 **1.17×** 加速，而 EAGLE-2 在两者上均未产生正收益。

质量方面，CLIP Score 基本稳定，Precision 稳定或上升，但 Recall 下降，表明多样性有所收缩；最强 sampling 配置下，FID 从 **15.22** 增至 **18.76**。因此，LANTERN 的价值是以可调、相对有限的质量折损换取明显加速，而非无损提速。消融还表明收益确实来自 latent 近邻；TVD 与 JSD 效果接近，但距离计算约快 **3 倍**。实际速度仍受硬件与 kernel 实现影响，相同接受长度并不保证跨设备获得相同收益。

> 本文参考自 [LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding](https://arxiv.org/abs/2410.03355)