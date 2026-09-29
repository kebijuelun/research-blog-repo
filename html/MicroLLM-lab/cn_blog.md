# MicroLLM lab：把 135M 小模型塞进浏览器，用 WebGPU + Q4 量化造一个纯本地的 LLM 基准实验室

云端大模型越来越强，但每一轮对话都要付出网络延迟、API 账单和隐私外泄的代价。MicroLLM lab 是 stateofutopia.com 上线的一个网页实验项目，它做的事情很"叛逆"： **把 26M～360M 参数的小语言模型（SLM）完整搬进浏览器** ，用 WebGPU 硬件加速 + Q4 4-bit 量化实现零服务器、零账号、100% 私有的本地推理，并且内置一套可证伪的客观基准测试。最硬的两个数字是：一个 124.6M 参数的 Llama 风格模型被压缩到约 78 MB 即可在浏览器里跑起来，而整个项目的离线包只有 589 MB——七个模型全部包含在内。这篇文章带你拆解它是怎么做到的。

## 一、提出问题：我们真的需要每次都呼叫云端吗？

先看一个现实问题。GPT-4、Claude 这类前沿模型，服务成本以百万美元计，每次请求还要加上几十到几百毫秒的网络往返。但在真实业务里，大量请求其实是"小题大做"：

- 判断用户意图、过滤垃圾信息、给查询分类——这些任务不需要千亿参数；
- 自动补全、实时 Agent 这类场景，对首 token 延迟（TTFT）极其敏感，等不起网络；
- 医疗、法律、个人助理场景，prompt 里全是隐私数据，根本不该离开设备。

小语言模型（SLM，这里指 25M～360M 参数）正好卡在这个生态位上：它们不当"百科全书"，而是做一个 **超快、超便宜的边缘智能层** ——在本地毫秒级完成初筛和路由，只有真正难搞的问题才升级给云端大模型。MicroLLM lab 的目标就是把这条路彻底走通，并且给你一个可以亲手量化的实验台。

## 二、分析问题：为什么是"浏览器 + Q4 + WebGPU"？

要在浏览器里跑 LLM，绕不开三个拦路虎：模型太大放不下、浏览器没有原生 GPU 计算能力、推理速度撑不起交互体验。MicroLLM lab 的三个技术选型恰好一一对应。

### 2.1 Q4 量化：把权重压到四分之一

Transformer 推理的 decode 阶段是典型的 **访存受限（memory-bound）** 场景：每生成一个 token，都要把全部权重从显存里读一遍。所以权重越小，带宽压力越小，速度越快。

Q4 的做法是把每个权重从 16-bit 浮点压成 4-bit 整数，内存占用直接砍掉 75%。从 MicroLLM lab 的 WGSL shader 源码可以看到，它采用的是经典的分组对称量化：每 32 个权重共享一个 f32 缩放因子，4-bit 码值以 8 为零点。反量化公式非常干净：

$$
w_i = s_{g(i)} \cdot (q_i - 8), \quad g(i) = \lfloor i / 32 \rfloor
$$

其中 $q_i \in [0, 15]$ 是 4-bit 码值，$s_g$ 是该组的缩放因子。效果直观可见：124.6M 参数的模型，Q4 打包后只有约 78 MB；全家最重的 SmolLM2 360M 也不过 226 MB，浏览器标签页完全吃得下。

> 编辑点评：这个设计的聪明之处在于"全程不解压"——Q4/Q8/F16 权重以打包形态常驻 GPU 显存，只在 GEMV 的 shader 内部边读边反量化，显存占用和带宽同时受益，而不是先在 CPU 解压再上传。

### 2.2 WebGPU：浏览器里的 GPU 直通车

WebGPU 是 W3C 的现代标准，允许网页直接调用硬件 GPU 执行 compute shader——在 Apple Silicon 上走 Metal，Windows 上走 DirectX 12，Linux 上走 Vulkan。这意味着浏览器不再是"只能跑 JS 的沙盒"，而是一块可以真刀真枪做矩阵运算的算力。MicroLLM lab 的后端降级链是： **优先 WebGPU，失败则退回 WASM，再不行退回纯 JS** ——保证任何设备都能打开，只是快慢之别。

### 2.3 为什么测"客观对错"而不是"写作质量"

小模型评测有个常见误区：拿大模型的开放式问答标准去打分，分数全看评审心情。MicroLLM lab 的态度写在页面上，非常硬核：

> Objective checks (regex / exact tokens), not writing quality. A 135M model is allowed to fail — that **is** the measurement.

翻译过来： **只看正则和精确匹配，不看文笔；135M 的模型答错是常态，失败本身就是测量结果** 。这让不同设备、不同模型之间的分数可以直接比较，没有任何主观水分。

## 三、解决问题：MicroLLM lab 的技术实现

理清了选型逻辑，接下来看它具体怎么落地。我扒了它的前端源码，几个实现细节值得单独讲。

### 3.1 自研权重格式 PGW1

项目没有直接用 ONNX 或 safetensors，而是自定义了一个紧凑的二进制格式 **PGW1** （Packed PetitGPT web Weights）。文件头 4 字节 magic 之后，依次编码 dtype（f32/f16/bf16/q8/q4）、模型架构（llama 或 gpt2）、以及完整的超参数：层数、隐藏维度、注意力头数、KV 头数、FFN 维度、RoPE 的 theta 与缩放比例、量化分组大小等；随后是一个 JSON 张量索引表 + 连续的二进制 payload。加载时一次 fetch 拿到整个文件，`parsePgw()` 解析出头信息后直接把 ArrayBuffer 视图喂给 GPU，零拷贝、零转码。

### 3.2 手写 WGSL：一个 workgroup 走完整个 Transformer

这是最让我惊讶的部分。MicroLLM lab 没有使用 ONNX Runtime Web 或 transformers.js，而是 **手写了全套 WGSL compute shader** ，并且针对平台做了两套路径：

- **NVIDIA / 桌面端（fused 模式）** ：单个 workgroup 内部完成整个 Transformer 的逐层前向——RMSNorm、RoPE、GQA 注意力、SwiGLU 全部在 `workgroupBarrier()` 之间串起来，避免多次 kernel 启动的开销；decode 阶段还用了 **prompt-lookup 投机解码** （以 3-gram 匹配、每次前瞻 2 个 token），在小模型上几乎零成本地提速。
- **Apple / Safari（Metal 路径）** ：改用多 workgroup 的 GEMV 拆分方案 + 链式 decode，绕开 Safari 对单 workgroup 资源的限制。

配套还有一条专门的 GPT-2 Q4 管线，额外引入了 **SmoothQuant 通道缩放** （把激活的量化难度迁移到权重上）和 n-gram 循环阻断，让 2019 年的老 GPT-2 也能生成连贯文本。

### 3.3 七个模型的"对照实验"阵容

模型选择本身就是一场精心设计的小型实验，覆盖了不同训练预算、不同家族、不同架构：

| 模型 | 参数量 | Q4 体积 | 背景亮点 |
| --- | --- | --- | --- |
| PetitGPT research-v1 | 124.6M | ~78 MB | 基准参照系；yangqi0 用单张 RTX 4090、约 13B token 训练的 Llama 风格模型（GQA 9/3、RoPE、SwiGLU、权重绑定） |
| SmolLM2 135M Instruct | 134.5M | ~84 MB | Hugging Face 出品，2T token 预训练 + 指令微调，通常是准确率之王 |
| L20-Edu 135M | 134.5M | ~84 MB | 与 SmolLM2 同构，但只用单张 L20、约 13B token——"一张显卡能做什么"的可审计样本 |
| SmolLM2 360M Instruct | 361.8M | ~226 MB | 全家最强调教模型，换取更高准确率 |
| MiniMind2 104M | 104M | ~65 MB | 国内 jingyaogong 的从零教学项目，中文更强 |
| MiniMind2 Small 26M | 25.8M | ~16 MB | 全家最快，教学规模的极限 |
| GPT-2 124M | 124.4M | ~81 MB | 2019 年 OpenAI 经典架构（nanoGPT 同款），专用管线运行 |

其中 **SmolLM2 135M vs L20-Edu 135M** 这一组最有实验味道：同样的 30 层 × 576 维 GQA 结构，唯一变量是预训练算力（2T token vs 13B token），在同一条跑分赛道上直接量化"数据规模到底值多少分"。

> 编辑点评：用网页当"控制变量实验台"是个很少见的思路——模型同构、解码器同一份、量化方式同一种，跑分差几乎只能归因于训练本身。

### 3.4 隐私与部署：数据不出浏览器

模型通过一次 fetch 下载后直接缓存进浏览器的 **IndexedDB** （不落盘、不上传），之后离线也能用。HUD 面板实时显示后端类型、tok/s、TTFT、停止原因、JS 堆、GPU buffer 和 IndexedDB 占用——所有数字都留在本机。想离线部署也简单：下载 589 MB 的 zip 包，本地起个 HTTP 服务即可（注意不能用 `file://` 协议直接打开）。

## 四、实验验证：一套"允许失败"的客观基准

说完了怎么跑起来，下一个问题是：怎么证明它跑得对、跑得快？MicroLLM lab 的答案是一套约 20 项的内置测试套件 `builtin-v1`。

### 4.1 测试设计：为 125M 模型量身定做的"及格线"

套件里的每一项检查都是确定性的 JS 断言，可以分成四类：

- **常识与事实** ：法国首都是 Paris、水在 0°C 结冰、狗有 4 条腿、地球的天然卫星是 Moon；
- **指令遵循** ："把 banana 重复三遍"、"只回答 yes 或 no"、"只输出那个秘密代码 ORANGE-42"；
- **上下文复制** ：答案就藏在 prompt 里（如"记住数字 7391"），这是小模型最公平的考题；
- **生成健康度** ：`stop_hello` 检查模型能否正常以 EOS 收尾而不是无限啰嗦；`no_repeat_hello` 统计 4-gram 重复率，超过 0.4 判为退化复读。

此外还有一项压力测试：强制连续生成 256 个 token（忽略 EOS），测量 **持续解码速度** ——短跑快不算快，长跑不掉速才是真吞吐。

> 编辑点评：`stop_hello` 和 `no_repeat_hello` 这两项尤其见功力。很多小模型基准只测"答得对不对"，而这两项测的是"生成过程健不健康"——EOS 停止和抗复读恰恰是端侧部署最容易翻车的地方。

### 4.2 指标体系与可分享证书

跑完套件后，页面汇总四个维度的成绩：

| 指标 | 含义 |
| --- | --- |
| Peak Speed | 单项测试中的最高瞬时 tok/s |
| Sustained Speed | 连续 256 token 解码的持续 tok/s |
| Accuracy | 客观检查通过率 |
| Suite Wall | 整套测试的墙钟耗时 |

更有意思的是 **Verified Benchmark Certificate** ：填入昵称，系统自动采集设备硬件信息（GPU adapter、平台），生成一张可下载、可分享的 PNG 成绩证书，一键发到 X 或 LinkedIn。因为你的 M4 Mac 和我的老款核显跑出来的数字天然不同——"晒分"本身就成了一项社区活动。

### 4.3 自定义基准：用 JavaScript 写考题

内置套件不够玩，还可以自己写。Custom eval 面板里的代码会被 `eval()` 执行，返回一个符合契约的 suite 对象：

$$
\text{suite} = \{ id,\ name,\ maxNewTokens \in [1, 256],\ tests: [\{ id,\ prompt,\ check(text, info) \to \{ pass, score, detail \} \}] \}
$$

其中 `info` 携带 `generatedIds`、`stopReason`、`backend` 等元信息，校验规则也很严谨：最多 64 个测试、prompt 不超过 4000 字符、`check` 必须是真函数。最贴心的是内置了一个 **"Prompt for a larger LLM"** 按钮——把一段精心设计的 system prompt 复制给 GPT-4 级别的大模型，让它帮你生成合规的小模型考题，形成"大模型出题、小模型考试"的闭环。另外项目还留了 `?auto=bench / suite / ui / dl` 等 URL 参数钩子，方便做自动化回归测试。

## 五、总结

- **核心定位** ：MicroLLM lab 把 26M～360M 的小模型完整搬进浏览器，WebGPU 加速 + Q4 量化 + IndexedDB 缓存，实现零服务器、零账号、100% 本地的 LLM 实验台。
- **工程硬核** ：自研 PGW1 权重格式、手写 WGSL shader（NVIDIA 单 workgroup 融合 + Apple 多 workgroup 拆分双路径）、prompt-lookup 投机解码，全程不依赖 ONNX Runtime 等重型框架。
- **模型阵容即实验** ：7 个模型构成天然对照组，SmolLM2 135M 与 L20-Edu 135M 同构不同算力，直接量化训练预算的价值。
- **评测哲学务实** ：约 20 项正则/精确匹配的客观检查，"允许 135M 模型失败"，再加 EOS 停止与抗复读等生成健康度指标。
- **社区玩法** ：可验证的硬件成绩证书 + 自定义 JS 基准（还能让大模型帮你出题），把跑分变成可分享、可扩展的开放活动。

**展望** ：这个项目的局限也很明确——受浏览器显存限制，模型规模止步于 360M 上下，且纯解码器管线对多模态无能为力。但它示范的"浏览器即边缘 AI 运行时"范式，配合 WebGPU 在各家浏览器的成熟，很可能成为端侧智能分发的标准形态之一。

> 本文参考自 [MicroLLM lab — tiny LLMs, Q4, in your browser](https://stateofutopia.com/experiments/microllmlab/)