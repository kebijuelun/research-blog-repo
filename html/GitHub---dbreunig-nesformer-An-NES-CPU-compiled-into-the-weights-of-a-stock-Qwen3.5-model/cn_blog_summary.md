# nesformer：把 NES CPU 编译进模型权重

《超级马力欧兄弟》的每一次 CPU 状态推进，都能由一个未经训练的标准 Qwen3.5 checkpoint 逐 token 生成。nesformer 用编译器把 NES 的 6502 CPU、2 KB 工作内存和游戏 ROM 写入模型，宿主程序只负责画面、手柄和中断。它值得关注的不是“大模型学会玩游戏”，而是神经网络架构可以成为精确电路的编译目标。

作者报告，Full Game 自由运行测试生成了 2,645,885 个 token、420 帧，执行轨迹与参考 CPU 一致，保存的 210 帧也全部匹配常规 6502 模拟器。不过，默认引擎在开发机器单个 CPU 核心上约为 1,650 token/s，整机仍比真实 NES 慢 150–290 倍。

![nesformer 运行超级马力欧兄弟 World 1-1](images/demo.gif)

> 图解：模型生成 CPU 执行日志中的 token，宿主程序据此绘制《超级马力欧兄弟》World 1-1 的画面。

## 把 DeltaNet 状态变成内存

nesformer 沿用 Qwenputer 的 chipformer 编译器。模型内部实现全部 151 个官方 opcode、寻址模式、寄存器和 RAM；PPU、手柄与 NMI 时序仍在宿主侧，因此它并没有把整台 NES 塞进权重。

核心技巧是把每个 Gated DeltaNet head 的 $128\times128$ 状态矩阵当作可寻址内存。门保持打开、key 为 one-hot 向量时，一次更新恰好覆盖一行；再用 one-hot query 读出该行。卡带、RAM 和寄存器由此获得明确地址。启动不同游戏时，只需写入新的卡带与机器状态，无需重新训练。

## 稀疏权重如何执行指令

生成的 checkpoint 可直接由 Hugging Face 加载：bf16 文件约 933 MB、共 466.6M 参数，但只有 9.3M 非零参数，压缩后仅 3.3 MB。7 个 Gated DeltaNet 层分别承担取指、译码、算术和写回，另有一个全零 attention 层满足缓存格式。

每条指令以记录形式生成，末尾 token 提交结果，真实游戏平均每条指令仅需 1.1–1.3 个 token。PC 也不逐步完整写入，而是由计数器累计不同长度指令，再用“跳转目标 + 偏移量”恢复，体现的是显式电路设计，而非训练得到的行为先验。

## 验证精确，也划定边界

作者从参考 CPU、精确电路、stock transformers 到权重自由运行逐层核对：参考 CPU 与 nestest 的 3,348 条指令日志匹配，并与 py65 在 570 万条游戏指令上匹配；自由运行中的 Thwaite 和 Full Game 也分别在所列测试轨迹与保存帧上完全一致。这些证据只覆盖测试范围，不能外推到所有 NROM 游戏。

- 未经训练的 Qwen3.5 权重可以承载完整官方 6502 指令集、RAM 和固定 ROM。
- 精确性来自编译后的稀疏电路与可寻址状态，而非模型涌现的游戏策略。
- 目前仅支持 NROM（mapper 0）、官方 opcode 和从 ROM 执行代码，不支持 IRQ；PPU 仍有近似，也没有声音。

即使跳过空闲循环可减少 2.6–4.2 倍 token，600 帧、10 秒 NES 时间仍约需 25 分钟。回到开头的马力欧：nesformer 暂时不是实用模拟器，却证明模型权重除了来自训练，也能成为通用计算电路的编译输出。

> 本文参考自 [GitHub - dbreunig/nesformer: An NES CPU compiled into the weights of a stock Qwen3.5 model](https://github.com/dbreunig/nesformer)