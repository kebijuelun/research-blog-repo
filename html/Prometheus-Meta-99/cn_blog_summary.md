# Prometheus：Meta 如何把吉瓦级超算集群的中断率压低 99%

一个跑在几万张 GPU 上的训练任务，任何一台主机出问题都可能中止整个任务。Meta 最近披露了俄亥俄州吉瓦级超算集群 Prometheus 的工程细节：一期已建成并训练出 Muse 系列基础模型，预训练 goodput 超 85%，强化学习超 90%，平均无中断时间（MTBI）经固件修复后达 5.4 万 GPU 小时以上。最有信息量的不是满配约 100 万张 GPU 的愿景，而是中断率曲线：2025 年年中峰值期每天每 8000 张 GPU 近千次中断，一年后下降 99%。

硬件上，Prometheus 已部署两代 NVL72 平台：Catalina（GB200，72 张 Blackwell GPU、13.4 TB HBM3e、FP8 峰值 720 PFLOPS）和 Clemente（GB300，每卡显存增至 288 GB，FP4 峰值超 1 exaFLOP，性能比 GB200 提升 5-10%）。一个值得注意的判断：GB200 相对 H100 几乎扩展了每个维度，唯独每 GPU scale-out 带宽仍是 400G，Meta 预判这会损失性能，提前按每 GPU 800G 部署。

可靠性的核心矛盾是：scale-up 域从 NVL8 扩到 NVL72，故障爆炸半径扩大 9 倍。Meta 的应对是机内冗余——每个 72 卡域跑 64 张、8 张做备件，另备约 2% 备用机架；故障能隔离在单个数据并行副本内时用 PAFT 机制把停机降级为吞吐的短暂下降，更大范围则回滚 checkpoint。

![GB200/GB300 中断率趋势](images/Meta-Prometheus-GB200-GB200-interruptions.png)

> 图解：2025 年 6 月至 2026 年 9 月的日均中断次数（每 8000 张 GPU）。GB200（蓝线）峰值接近 1000 次，经 8 轮固件与配置修复一路下降；粉色虚线是目标——每天每 8000 张 GPU 仅 2 次中断。

99% 的降幅来自把中断逐类归因、逐一解决：60 多万条 NVLink 连接靠持续遥测和烧机诊断定点维修；HBM 与 LPDDR 内存故障用固件调参、内存擦洗和预测性退役缓解；NVLink 交换机以毫秒级监控自动隔离。支撑这套机制的是 NVIDIA 团队进驻现场、共享遥测的联合工程模式，以及维修必须验证才能回池的问责框架。固件修复几小时内可推到 10 万张以上 GPU，单 pod 升级 P90 从 7 个多小时降到 2 小时。

供电散热方面，几万张 GPU 同步计算会触发 30-40% 的负载突变，Meta 用 Dynamo Dimmer 在接近电气上限时渐进下调功率上限而非硬停任务，散热混用风辅助液冷与设施级液冷。下一步是 Vera Rubin NVL72（规划值：每卡 288 GB HBM4、机架功耗超 170 kW），经验也将复用到下一代集群 Hyperion。

Prometheus 给出的可复用路径是：集群稳定性不是买来的，而是归因到每类故障源再逐一工程化解决出来的。需注意文中数字均来自 Meta 官方口径，goodput、MTBI 的测试条件未完全公开，Vera Rubin 指标仍是规划值，对比与引用需谨慎。

> 本文参考自 [Inside Prometheus: Meta's Journey to a Gigawatt-Scale Supercluster](https://engineering.fb.com/2026/10/19/data-center-engineering/inside-prometheus-meta-gigawatt-scale-supercluster/)