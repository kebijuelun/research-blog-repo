# VFlash：机器人规划省下80%延迟

VFlash 解决了 VLM 生成点、框和轨迹时逐坐标自回归解码过慢的问题，让机器人在有限延迟预算内输出更精细的视觉计划。在 MolmoSpaces 下游流水线中，它将 Jetson Thor 与 DGX Spark 的单步总延迟均降低约 80%，同时把四类任务平均成功率从 62.6% 提升至 70.2%。

## Flash Primitive Decoding：按几何对象并行解码

- **完整 primitive 作为一个解码块。**视觉对象的类型、坐标维度、范围与长度已知，轻量 drafter 即可一次前向填完整条轨迹、一个框或一帧稠密点运动，不再逐 token 生成。
- **从逐 token 一致转向整体验证。**普通文本仍采用 DFlash 的 target 验证；视觉块只要满足 schema 与空间规则就直接接受，无需复制 AR 的每个坐标。随后 target 对完整块做一次前向，更新 KV cache。
- **两种对齐保证可并行。**Boundary alignment 让块边界贴合几何对象，Position alignment 预先确定每个位置是坐标、分隔符还是其他角色，并施加合法 token 约束。
- **学习空间标注，而非模仿 AR。**冻结 target 后，训练五层、537M 可训练参数的非因果 drafter；标签来自真实坐标或渲染网格，因此加速与精度提升可以同时发生。普通文本坐标也通过定长数字位和 MASK 获得支持。

## 速度之外，空间精度也普遍改善

- **2D 轨迹：**相对同一 AR target，ShareRobot 与 VABench-V 上六项误差全部下降，HD 约降 7%，DFD 与 RMSE 降 3–6%。
- **3D 机器人轨迹：**三种 target、三个数据集上的指标全部改善；3D-only 的 ADE 降低 18–49%，2D+3D 降低 28–61%。AR 联合输出 2D+3D 会损害 3D 精度，VFlash 则基本不受影响。
- **稠密 3D 运动：**单张 H100 的初步单 rollout 测量显示约 25 倍生成加速；DAVIS ADE/FDE 分别降低 27%/19%，但三个数据集的 PWT 均下降 7–12%，WorldTrack FDE 也增加 15%，并非全面胜出。
- **Bounding box：**置信度回退使 Mean F1 基本追平 AR，框吞吐提升约 2.6–2.8 倍；若完全直接接受，Dense200 Mean F1 会掉约 20 点，说明结构合法不等于空间可靠。

## 下游价值与延迟来源

更快的计划确实能改善动作：MolmoSpaces 中 Pick & Place、Close 分别提升 11.6、12.8 个百分点；把同类计划改由 AR 生成，平均成功率反而降至 61.0%，表明收益不只是“解码更快”。LIBERO 上扰动越复杂，视觉计划增益越明显；驾驶实验在规划 VLM 冻结时，RL 后取得 NAVSIM 90.7 PDMS、Waymo 7.86 RFS。

收益几乎全部来自解码阶段：DFlash 在轨迹块内每次前向仅提交 2.2 个 token，而 VFlash 一次提交完整 primitive。当前限制也很明确：稠密运动仅帧内并行、帧间仍顺序生成，部分结果来自单 seed 或初步单次测量，不宜外推为所有部署环境的固定收益。

> 本文参考自 [VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding](http://arxiv.org/abs/2611.08449v1)