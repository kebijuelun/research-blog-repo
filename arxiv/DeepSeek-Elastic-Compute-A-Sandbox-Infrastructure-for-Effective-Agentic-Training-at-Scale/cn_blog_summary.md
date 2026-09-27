# DeepSeek 弹性计算平台 DSec：大规模 Agent 训练的沙箱基础设施精华总结

训练会"动手"的 Agent 模型，瓶颈不在 GPU，而在每次试错所需的**真实、隔离、有状态的执行环境**。DeepSeek 推出的生产级沙箱平台 **DSec** 正是为此而生：单个生产单元约 160 节点、30K CPU 核，日均服务约 **300 万个沙箱实例**，峰值并发超 **38 万**，创建速率超 **每秒 5000 个**——公开资料中最大规模的 Agent 训练沙箱集群之一。

## 为什么需要专门的沙箱平台

Agentic RL 的 rollout 本质是"驱动真实环境跑完轨迹"，其负载画像与传统 Serverless（短寿命、无状态、高 fanout）完全相反：

- **突发创建**：单任务最多请求 32K 个沙箱；
- **CPU 稀疏但有状态**：约 90% 沙箱平均 CPU 使用率不超过申请量的 5%，但内存长期被钉住（中位寿命约 17 分钟，p99 超 3 小时）；
- **环境海量低复用**：一周活跃环境产物超 130 TB，容器镜像中位 fanout 仅 3，microVM 仅 1；
- **Agent 不可信、执行可中断**。

## 核心设计

DSec 把 FnCall、容器、Firecracker microVM、完整 VM 四种后端统一在一个平台下，统一 SDK 但**不做过度抽象**，并靠四件套拉满密度与弹性：

- **可组合环境层**：基础镜像、workspace、toolkit 作为独立版本化的 EROFS 只读层，经 overlayfs 动态合并（改 dockerd 仅约 30 行 Go 代码）。环境维护成本从 $O(m \cdot N)$ 降到 $O(m)$，沙箱 provisioning 提速 **1.76 倍**。
- **按需镜像加载**：镜像托管在 3FS 上，写留本地、读按需批量、元数据本地化。相比全量拉取，8192 个容器突发场景下完成时间快 **1.71 倍**，写盘量减少 **57%**，吞吐追平全量本地缓存。
- **内存共享与回收**：virtio-pmem + DAX 消除宿主/guest 双份 page cache（峰值宿主内存 -40.2%）；DAMON + virtio-balloon FPR 回收 guest 空闲页（积分内存 -21.2%）。
- **CPU QoS**：BE 任务挂 `SCHED_IDLE`，LS 任务启用 core scheduling 隔离 SMT 干扰，50% 混部负载下延迟膨胀从 45.2% 压到 **17.3%**。

这套组合支撑单节点 **3200 个容器或 800 个 microVM** 的密度。

## 与 RL 框架的协同

- **环境由 Agent 构建**：`pack_diff` 增量快照把一次交互会话直接变成可复用环境，构建、验证、消费在同一平台闭环；
- **Rollout 与 GPU 训练解耦**（V4.1 起）：rollout 组件跑在可抢占 GPU 池之外，抢占后重连即续，删掉脆弱的命令日志回放；
- **挂起与恢复**：容器靠 `docker pause` + swap 回收 + `MADV_WILLNEED` 预取，microVM 靠快照 + 杀进程恢复；
- **纵深防御的访问控制**：AppArmor 控制文件/socket 访问（沙箱内 root 也生效），eBPF 按域名/端口做每沙箱网络白名单——应对伪造 RPC、偷答案、覆盖 `/bin/bash`、读 `/proc/kpagecgroup` 搞挂内核等真实事故。

## 结论

DSec 证明沙箱是 Agentic RL 的隐形瓶颈，且传统 Serverless 假设全部失效；其公开的生产负载数据与踩坑记录（Reward Hacking、内核被打崩等）本身就是对社区的重要贡献。作者坦承对 Agent 破坏性行为尚无通用防御——随着模型变强，"沙箱防 Agent"的猫鼠游戏才刚开始。

> 本文参考自 [DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale](https://arxiv.org/abs/2609.22978)