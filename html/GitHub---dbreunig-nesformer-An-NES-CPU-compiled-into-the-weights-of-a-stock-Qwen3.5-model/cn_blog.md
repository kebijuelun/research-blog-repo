# nesformer：把 NES CPU 编译进模型权重

《超级马力欧兄弟》的每一次 CPU 状态推进，都可以由一个标准 Qwen3.5 checkpoint 逐 token 生成，而这些权重没有经过训练。nesformer 用编译器把 NES 的 6502 CPU、2 KB 工作内存和游戏卡带 ROM 写进模型，再由一个小型宿主程序负责画面、手柄和中断。它展示的不是“大模型学会了玩游戏”，而是神经网络架构可以被当作一种精确的电路编译目标。

这个 checkpoint 并非只能在概念演示中输出几个指令。作者报告的 Full Game 自由运行测试包含 2,645,885 个 token 和 420 帧，执行轨迹与参考 CPU 一致，保存的 210 帧也全部匹配常规 6502 模拟器。代价同样明确：默认引擎在开发机器的一个 CPU 核心上约为 1,650 token/s，整机运行仍比真实 NES 慢 150–290 倍。

![nesformer 运行超级马力欧兄弟 World 1-1](images/demo.gif)

> 图解：nesformer 运行《超级马力欧兄弟》World 1-1。画面由宿主程序绘制，CPU 执行日志中的 token 则全部由模型生成。

![nesformer 运行 Full Game 的标题画面和第一关](images/full_game.gif)

> 图解：Doug Fraker 的 Full Game 从标题画面进入第一关。作者对随附测试游戏逐帧核对，报告其画面与常规 6502 模拟器完全一致，具体范围见后文验证表。

## 模型里是 CPU，不是游戏模仿策略

nesformer 基于 Divyansh Agrawal 的 Qwenputer，沿用其 chipformer 编译器。Qwenputer 曾把 CHIP-8 和 RISC-V CPU 编译进 Qwen3.5-4B；这次的目标换成了 NES 的 6502 CPU，也就是 2A03 芯片的处理器核心。

这里的“运行游戏”需要拆成两部分。模型负责处理器状态和执行日志，宿主程序负责处理器之外的控制台硬件。主机不会直接读取模型内部的寄存器或 RAM，只能通过执行轨迹了解 CPU 做了什么。

| 模型内部：权重与 DeltaNet 状态 | 宿主程序：`nesformer/host.py` |
| --- | --- |
| 6502 核心：全部 151 个官方 opcode、所有寻址模式、标志位、栈、BRK 与 NMI 入口 | PPU：逐扫描线渲染器与调色板 |
| 2 KB 工作 RAM | 手柄；APU 寄存器只接收写入，其余忽略 |
| 32 KB PRG-ROM，保存两份，分别用于取指令和读取数据 | vblank 时的 NMI 时序，以 NMI token 注入 |
| OAM DMA 读取，256 个字节以 DM token 出现 | I/O 读取结果，以 IN token 注入 |

这套划分延续了 Qwenputer 的 CHIP-8 设计：宿主注入计时事件，并根据轨迹绘制显示内容。因此，它不是把整台 NES 都塞进权重，也没有让 PPU 和 APU 凭空从语言模型中涌现。

## DeltaNet 状态如何变成内存

核心技巧是把 Gated DeltaNet 的状态当成可寻址内存。每个 DeltaNet head 都维护一个 $128\times128$ 的状态矩阵，每处理一个 token，delta rule 就更新一次这个矩阵。

当门保持打开、key 使用 one-hot 向量时，一次更新恰好覆盖矩阵的一行；再用 one-hot query，就能把这一行读出来。于是，矩阵的每一行都可以成为一个存储单元，卡带、RAM 和寄存器被分配到这些行中。可以把它理解成一块由编译器规划地址的电路板，而不是模型在训练中形成的模糊记忆。

这个设计也解释了为什么同一份权重能运行不同游戏：启动时，CLI 把卡带、寄存器和 RAM 写入 DeltaNet 状态，而不是为每个 ROM 重新训练或重新构建模型。随后模型采用贪心解码，一次生成一个 CPU token。

## 933 MB checkpoint 里只有 9.3M 非零参数

生成的 `weights/` 是普通 Hugging Face 模型目录，可以直接通过 `Qwen3_5ForCausalLM.from_pretrained` 加载，不需要自定义建模代码。它使用 Qwen3.5-4B 的 DeltaNet head 形状和 MLP 宽度，但只保留实现电路所需的少量层。

| 项目 | 配置 |
| --- | --- |
| 架构 | `Qwen3_5ForCausalLM`；16 个 key head、32 个 value head，维度 128；MLP 宽度 9,216 |
| 层数 | 7 个 Gated DeltaNet 层构成电路，另有 1 个 Hugging Face cache 所需的全零 attention 层 |
| 隐藏维度 | 768 |
| 词表 | 263,808 个 token |
| 参数 | bf16 格式共 466.6M，文件约 933 MB；其中 9.3M 为非零参数，因此权重压缩后仅 3.3 MB |

`nesformer.json` 记录了机器的每个部分位于哪一层、哪个 head。933 MB 与 3.3 MB 的反差并不意味着模型被压缩后失去了能力，而是因为大部分参数本来就是零；真正承载电路的是稀疏的非零权重和运行时被精确写入的状态。

七个电路层分工如下。Attention/state 一侧负责读取和保存机器状态，MLP 一侧完成解码、算术和结果提交。

| 层 | Attention（状态） | MLP |
| --- | --- | --- |
| 1 | PC 锚点；指令计数器；A、X、Y、SP；N、V、Z、C、I、D 标志位；指针锁存器；NMI 标志 | 温度计式计数器；SP ± 1；LO + Y |
| 2 | — | PC = 锚点 + 此后执行的字节数；PC + 1、PC + 2 |
| 3 | 从 PRG 取出 PC 处的 opcode 和两个操作数字节 | opcode 解码；操作数位；索引求和；分支判断与目标地址 |
| 4 | — | 按寻址模式和记录阶段计算有效地址；待压栈字节 |
| 5 | 按有效地址读取 RAM 和 PRG 数据 | 操作数；ADC/SBC 加法器、比较和 BIT；地址解码 |
| 6 | — | 结果字节、进位、溢出、标志位和跳转目标 |
| 7 | — | 下一个 token |

这张层表也是理解项目性质的关键：权重不是通过 Loss 调整出来的行为先验，而是取指、译码、执行和写回流程的显式实现。

## 一条指令如何变成 token

每条指令会生成一条记录，记录的最后一个 token 负责提交结果。大多数指令只需要一个 token；真实游戏代码平均为每条指令 1.1–1.3 个 token。

程序计数器 PC 的处理尤其巧妙。模型并不在每一步直接写入完整 PC，而是在末尾 token 中携带指令长度，再由三个 DeltaNet 计数器统计上次跳转以来不同长度指令的数量，最终用“跳转目标 + 偏移量”恢复 PC。

下面是 Full Game 执行轨迹的节选：

```text
SEI            P(000100 px0 L1)
LDX #$40       X($40 c0 v0 px0 L2)
STX $4017      IOW($4017=$40 L3)
TXS            S($FF L1)
BIT $2002      IOR($2002) IN($00) P(000110 px0 L3)       <- the host injects IN
BPL $8017      J($8017)
JSR $8279      PSH($FE) PSH($FD) J($8279)                 <- the return address is pushed in-circuit
RTS            PUL($FC) LO($7B) PUL($FD) J($827C)
STA ($72),Y    LO($08) HI($03) ST($00 a7=0 L2)            <- pointer bytes latched, then the store
STA $4014      IOW($4014=$02 L0) DM(0:$FF) … DM(255:$00) NT(L3)   <- OAM DMA
```

其中，`IN($00)` 是宿主对 I/O 读取的回应；`JSR` 的返回地址由电路内部压栈；间接寻址写入会先锁存指针字节；最后一行则展示了 OAM DMA 的 256 字节读取。完整 token 家族定义在 `nesformer/spec.py` 中。

## 精确性验证覆盖到了哪里

“画面看起来能跑”不足以证明 CPU 电路正确。作者把验证分成四层：先验证参考 CPU，再验证精确电路，然后检查 stock transformers 加载的 checkpoint，最后让权重自由运行并与参考实现对照。

- 参考 CPU `nesformer/cpu.py` 与 nestest 的 golden log 在 3,348 条指令上匹配，并与 py65 模拟器在 570 万条游戏指令上匹配。
- 精确电路在 Qwenputer 的 evaluator 中运行，并通过 bf16 精确性检查；它预测了 nestest 全部参考 token，以及两个游戏启动序列的 283,610 个位置。
- stock transformers 在 CPU 上以 bf16 加载 checkpoint，对 nestest 和 Full Game 的 2,651 个位置进行预测，没有不匹配。
- 权重自由运行时，生成轨迹与参考 CPU 一致，结果如下：

| 运行 | Token 数 | 帧数 | NES 时间 | 结果 |
| --- | ---: | ---: | ---: | --- |
| nestest（官方 opcode） | 4,586 | — | — | 一致，并以预期的 HALT 结束 |
| Thwaite，`check` | 700,000 | 271 | 4.5 秒 | 轨迹与帧均一致 |
| Thwaite，`run` | 1,916,527 | 600 | 10.0 秒 | 保存的 300 帧全部匹配 |
| Full Game，`run` | 2,645,885 | 420 | 7.0 秒 | 保存的 210 帧全部匹配 |

这里的 NES 时间是所模拟主机的运行时间，不是实际等待时间。表中的结果也只证明所列测试范围，不能外推成“所有 NROM 游戏都已逐帧验证”。

空闲循环跳过优化也单独做了核对：帧和 I/O 写入（包括周期时间戳）与不跳过时的运行一致。项目测试需要几分钟，可用下面的命令复现：

```bash
pip install pytest py65
python -m pytest -q tests
```

测试覆盖参考 CPU 对 nestest 日志、电路对参考实现、权重自由运行对参考实现，以及空闲循环跳过。

## 能精确运行，为什么还要等 25 分钟

默认引擎直接从 `model.safetensors` 计算 Qwen3.5 前向传播，用 numba 编译并跳过全零模块。在作者开发机器的一个 CPU 核心上，它约能处理 1,650 token/s；纯 NumPy 引擎约为 210 token/s，stock transformers 约为 10 token/s。按这套执行轨迹表示，真实 NES 的速度需要约 700,000 token/s。

游戏每帧有大量时间耗在等待 vblank 的循环里。宿主发现循环体完全重复、且没有 RAM 或 I/O 写入时，会把时钟推进到下一个 PPU 事件之前，剩余迭代仍继续执行。该优化把 token 数减少了 2.6–4.2 倍，但总体仍比真实主机慢 150–290 倍。

实际使用时，第一次运行还要花约 20 秒让 numba 编译引擎。之后每帧需要 1–6 秒，600 帧对应 10 秒 NES 时间，实际约需 25 分钟。因此，这个项目当前更适合研究模型即计算电路，而不是替代日常游戏模拟器。

## 从下载权重到运行自己的 ROM

### 安装与目录

项目要求 Python 3.11 或更新版本。将 `nesformer.zip` 和 `nesformer-weights.zip` 解压到同一个目录，`weights/` 应与 Python 包所在的 `nesformer/` 并列，而不是放进包内：

```text
nesformer/              <- 在这个顶层目录运行命令
├── nesformer/          <- Python 包
├── chipformer/         <- Qwenputer 的编译器
├── roms/               <- 测试 ROM，也可放自己的 ROM
├── examples/           <- 示例 GIF 和帧图
├── tests/
├── LICENSE
├── LICENSE.chipformer
├── README.md
├── pyproject.toml
└── weights/            <- 来自 nesformer-weights.zip
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── nesformer.json
    └── SHA256SUMS
```

CLI 默认在与包同级的 `weights/` 查找模型。若权重位于别处，可给 `run`、`check` 或 `trace` 传入 `--weights /path/to/weights`，也可以把路径交给 `info`。解压后可在顶层目录校验：

```bash
shasum -a 256 -c weights/SHA256SUMS
```

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy numba pillow
```

### 运行游戏与编写手柄脚本

下面的命令运行 Thwaite，并按键脚本操作：

```bash
python -m nesformer run roms/thwaite.nes --frames 600 --show \
  --buttons "start@90-95+200-205,a@230-233+250-253+270-273+290-293+310-313+330-333+350-353+370-373+390-393+410-413+430-433+450-453+470-473+490-493+510-513+530-533+550-553+570-573+590-593"
```

`--show` 会在终端中逐帧显示画面。结束后，CLI 会在当前目录保存以 ROM 命名的动画 GIF，本例为 `thwaite.gif`。

也可以运行自己的 `.nes` 文件，但目前只支持 NROM 卡带，即 iNES mapper 0，例如《超级马力欧兄弟》《Donkey Kong》或《Excitebike》。仓库只附带三个自由许可的测试 ROM，其他游戏需要自行提供。以下脚本开始游戏，随后让马力欧向右移动，并大约每秒跳一次：

```bash
python -m nesformer run roms/smb.nes --frames 600 --show \
  --buttons "start@60-65,right@200-600,a@250-265+300-315+360-375+420-435+480-495+540-555"
```

`--buttons` 使用逗号分隔多个输入项。每项先用 `+` 连接一个或多个按键，再写 `@`，最后给出用 `+` 连接的一个或多个帧区间：

```text
start@60-65                 第 60 至 65 帧按 Start
right@200-600               第 200 至 600 帧持续按右
a@250-265+300-315            按两次 A
right+b@200-600             同时按住右和 B，在马力欧中用于奔跑
```

可用按键为 `a`、`b`、`select`、`start`、`up`、`down`、`left` 和 `right`。帧从上电开始按每秒 60 帧计数，所以第 600 帧是第 10 秒。游戏在无法移动的画面会忽略输入，稍早按住按键通常没有影响；在《超级马力欧兄弟》中，A 按得越久，跳得越高。

### 命令与参数

| 命令 | 作用 |
| --- | --- |
| `python -m nesformer run ROM` | 用权重运行 ROM 并保存 GIF |
| `python -m nesformer check ROM` | 让权重与参考 CPU 并行运行，在首个差异处停止 |
| `python -m nesformer trace ROM` | 打印模型 token 流，默认 40 个 |
| `python -m nesformer build` | 编译电路并写入 `weights/`，约需 10 秒 |
| `python -m nesformer info` | 描述 checkpoint |

`run` 的选项如下：

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--frames N` | 120 | 在 N 帧后停止 |
| `--buttons SCRIPT` | 无 | 手柄输入脚本 |
| `--show` | 关闭 | 在终端绘制帧，需要 24-bit 色彩，如 iTerm2 或 Windows Terminal |
| `--gif PATH` | ROM 名称 | 动画 GIF 的保存位置 |
| `--png DIR` | 关闭 | 将保留的每帧存为 PNG；单独使用时不再保存 GIF，加上 `--gif` 可同时输出 |
| `--every N` | 1 | 每 N 帧保留一帧 |
| `--scale N` | 2 | 将保存的帧放大 N 倍 |
| `--engine` | `fast` | `fast`（numba）、`numpy`，或 `hf`（stock transformers，需要 torch，约 10 token/s） |
| `--no-skip` | 关闭 | 执行空闲等待循环的每次迭代，更慢但帧相同 |
| `--max-tokens N` | 无 | 在 N 个 token 后停止 |
| `--trace FILE` | 无 | 保存 token ID |

`run`、`check` 和 `trace` 都接受 `--weights`、`--engine`、`--buttons` 与 `--no-skip`。`check --tokens N` 设置比较长度，默认 50,000；`trace --tokens N` 设置打印数量，默认 40；`build --out DIR` 可将 checkpoint 写到 `weights/` 之外。

若想验证它确实能被未经修改的 transformers 加载，可以运行：

```bash
pip install "torch>=2.6" "transformers>=5.22"
python -m nesformer run roms/thwaite.nes --frames 3 --engine hf
```

`nesformer/hfcheck.py` 展示了完整过程：加载 checkpoint，解码一个 padding token，把机器状态写入 cache 中的 DeltaNet 状态，然后逐 token 输入。

## 当前边界决定了它能玩哪些游戏

- **只支持 NROM（iNES mapper 0）。** CPU 一次只能寻址 32 KB 程序 ROM。更大的游戏通过 MMC1、MMC3 等 mapper 切换 ROM bank，而 nesformer 在模型内部从固定的 32 KB 表读取，暂时无法跟随切换。要支持 mapper，需要把整个 ROM 和 bank 寄存器都放入状态，并增加一个将 CPU 地址转换为 ROM 地址的阶段。
- **只支持官方 opcode。** 遇到非官方 opcode 会发出 HALT。
- **代码必须从卡带 ROM 执行。** 跳入 RAM 会发出 HALT。
- **不支持 IRQ。** 包括 APU frame IRQ 和 DMC；BRK 与 NMI 可以工作。
- **部分 I/O 行为不完整。** 对 I/O 寄存器的 read-modify-write 指令不会产生 I/O 写入；`$4020–$7FFF` 读取为 0，因为没有 PRG-RAM。
- **PPU 和 APU 不在权重内。** PPU 按扫描线渲染，无法重现扫描线中途效果；sprite-0 hit 和 sprite overflow 只是近似实现，也没有声音。

这些限制并不否定电路验证结果，但划清了证据边界：它证明的是特定硬件子集和测试轨迹上的精确执行，而不是完整 NES 兼容性的终结方案。

## 代码如何组织，复现该看哪里

| 路径 | 内容 |
| --- | --- |
| `nesformer/spec.py` | token 家族与记录格式 |
| `nesformer/cpu.py` | 生成执行轨迹的参考 6502 |
| `nesformer/host.py` | CPU 外围：PPU、手柄、NMI 时序、周期计数、空闲跳过 |
| `nesformer/core.py` | 电路与状态快照 |
| `nesformer/weights.py` | 将电路转换为 `Qwen3_5ForCausalLM` checkpoint 的 emitter |
| `nesformer/engine.py`、`fastengine.py` | 用 NumPy / numba 运行权重 |
| `nesformer/hfcheck.py` | 在 stock transformers 中运行权重 |
| `nesformer/run.py`、`check.py`、`cli.py`、`__main__.py` | 驱动程序、teacher-forced 检查和 CLI |
| `chipformer/` | Qwenputer 的 Qwen3.5 编译器：`q35/` emitter 与原语、`dsl/` 图 |
| `weights/` | 编译后的 checkpoint 与 SHA-256 校验值 |
| `roms/` | 测试 ROM；作者和许可证见 `roms/SOURCES.md` |
| `examples/` | 运行生成的 GIF 与帧图；`demo.gif` 是开头的马力欧运行 |
| `tests/test_nesformer.py` | 测试套件 |

项目采用 MIT License。Qwenputer 与 chipformer 编译器同样采用 MIT License，版权标注为 © 2026 Divyansh Agrawal，该许可也覆盖 `nesformer/core.py` 和 `nesformer/weights.py` 中改编的代码。测试 ROM 保留各自许可：nestest 来自 Kevin Horton，Full Game 来自 Doug Fraker（MIT），Thwaite 来自 Damian Yerrick（GPL-3.0）。

## 这项实验真正证明了什么

- 一个未经训练的标准 Qwen3.5 checkpoint 可以承载完整 6502 官方指令集、RAM 和固定 ROM，权重本质上是编译产物。
- DeltaNet 状态被组织成精确内存，宿主只补足 PPU、输入和中断等外围功能；模型并不是独自模拟整台主机。
- 多层对照和数百万 token 的自由运行结果支持“测试范围内精确”，但 150–290 倍减速、mapper 0 限制和外围硬件近似决定了它暂时不是实用模拟器。

如果想亲手验证，最稳妥的路径是先运行随附 ROM，再用 `check` 对比参考 CPU，最后尝试自己的 NROM 游戏。比起“AI 又学会了一项技能”，nesformer 更值得关注的方向是：模型权重不仅可以来自训练，也可以成为通用计算电路的编译输出。

> 本文参考自 [GitHub - dbreunig/nesformer: An NES CPU compiled into the weights of a stock Qwen3.5 model](https://github.com/dbreunig/nesformer)