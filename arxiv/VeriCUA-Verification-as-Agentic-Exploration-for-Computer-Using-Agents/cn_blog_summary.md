# VeriCUA：主动查证，减少智能体“假成功”

电脑操作智能体声称“已完成”，文件却可能没保存、数量也可能做错；这类假成功会污染评测、数据筛选和强化学习奖励。VeriCUA 将验收变成主动调查：先勘察环境，再拆解标准，最后按需寻找证据。在 ScienceBoard 在线验证中，它取得 **86.4% 准确率、5.4% 假阳性率**，而自动生成验证器分别为 76.9% 和 21.7%。

## 从一次性判断到主动取证

传统 VLM judge 只看固定截图和日志，自动生成验证器则运行一次脚本；前者可能漏掉关键证据，后者可能查错位置。VeriCUA 在执行 agent 结束后介入，用四步完成复核：

- **Context Inspection**：先检查应用、目录、数据库结构或录制轨迹，弄清证据在哪里。
- **Rubric Synthesis**：结合指令与环境，把名称、内容、位置、数量拆成原子要求，并区分必须满足的 hard requirement 与附加的 soft check。
- **Active Exploration**：针对尚未确定的要求，动态选择 GUI、结构化状态或代码读取，而非反复查看末帧。
- **Rule-Based Arbitration**：汇总证据，输出 `SUCCESS`、`FAIL` 或 `ABSTAIN`；证据不足时不强迫模型猜测。

这里的关键边界是：**指令决定必须做到什么，环境决定去哪里验证**。发现的目录只是证据位置，不能被擅自添加为新的成功条件。

## 三类状态视图与证据纪律

VeriCUA 不再把机器状态压缩成几张截图，而是按需组合三类视图：Trajectory 记录 agent 做过什么；Surface 展示当前界面；Persisted State 则通过文件、数据库和系统服务确认修改是否真正生效。在线 checker 也被限制为“只读 getter + predicate”，避免任意写入被验证环境。

更重要的是，agent 的完成声明只能算 asserted evidence，不能单独证明成功；只有操作后的画面、文件读取、数据库查询等 attested evidence 才能确认结果。在线模式下，一项要求获得持久化可证实确认才标为 `MET`，所有 hard requirements 均满足才判成功，两次独立反驳才判 `UNMET`。

## 结果、收益与代价

- **在线更稳**：VeriCUA 在 AndroidWorld、OSWorld、ScienceBoard 上均取得最高准确率和最低假阳性率；OSWorld 准确率为 79.5%，AndroidWorld 为 90.5%。
- **离线同样领先**：五组离线结果的 Accuracy 与 Balanced Accuracy 均居首；OSReward-Hard 准确率由最强对比方法的 65.7% 提升至 **70.4%**。
- **代码读取补足界面盲区**：AndroidWorld 消融中，混合 GUI 与代码读取使决策率提高 37 个百分点、召回率提高 30 个百分点，假阳性数量保持不变。
- **保守判断有取舍**：`ABSTAIN` 能压住假成功，却会牺牲召回率；离线录像未保存的证据也无法靠探索补齐。

VeriCUA 的代价是更高的 Token、调用次数与时延，比单次 judge 约高一个数量级。因此，它更适合高置信数据筛选、奖励分配和标签审计，而非要求即时响应的交互场景。其核心启示是：可靠验证不在于多看几帧，而在于知道**去哪里找证据，以及什么证据才有资格作数**。

> 本文参考自 [VeriCUA: Verification as Agentic Exploration for Computer-Using Agents](http://arxiv.org/abs/2610.40578v1)