# MicroLLM lab：浏览器里跑 LLM——WebGPU + Q4 量化打造纯本地基准实验室

每次对话都要付出网络延迟、API 账单和隐私外泄的代价，而大量请求其实"小题大做"——意图判断、垃圾过滤、查询分类根本不需要千亿参数。MicroLLM lab 给出的答案很硬核：把 26M～360M 的小语言模型完整搬进浏览器，零服务器、零账号、100% 本地推理。最亮的两个数字：**124.6M 参数的 Llama 风格模型 Q4 压缩后仅约 78 MB；包含全部 7 个模型的离线包只有 589 MB**。

## 三大技术选型

- **Q4 量化**：decode 阶段是访存受限场景，每生成一个 token 都要读一遍全部权重。Q4 把权重从 16-bit 压成 4-bit，内存砍掉 75%。采用分组对称量化（每 32 个权重共享一个 f32 缩放因子），且"全程不解压"——打包形态常驻显存，在 GEMV shader 内部边读边反量化。最重的 SmolLM2 360M 也仅 226 MB。
- **WebGPU**：网页直接调用硬件 GPU 执行 compute shader（Apple 走 Metal、Windows 走 DX12、Linux 走 Vulkan）。降级链为 WebGPU → WASM → 纯 JS，任何设备都能跑。
- **客观评测**：只看正则/精确匹配，不看文笔——"135M 模型答错是常态，失败本身就是测量结果"，分数零主观水分、跨设备可比。

## 工程实现亮点

- **自研 PGW1 权重格式**：紧凑二进制格式，编码 dtype、架构与全部超参数，一次 fetch 后零拷贝、零转码直喂 GPU。
- **手写全套 WGSL shader**，不依赖 ONNX Runtime 或 transformers.js：NVIDIA/桌面端用单 workgroup 融合模式（RMSNorm、RoPE、GQA、SwiGLU 一次走完），Apple/Safari 走多 workgroup GEMV 拆分；decode 阶段用 prompt-lookup 投机解码提速；GPT-2 专用管线还引入 SmoothQuant 通道缩放。
- **7 个模型即对照实验**：覆盖 PetitGPT、SmolLM2 135M/360M、L20-Edu、MiniMind2、GPT-2 等。其中 SmolLM2 135M 与 L20-Edu 135M 结构完全相同，唯一变量是训练算力（2T token vs 13B token），直接量化"数据规模值多少分"。
- **隐私与部署**：模型缓存进 IndexedDB，离线可用；HUD 实时显示 tok/s、TTFT、显存占用，所有数据留在本机。

## 基准测试：允许失败的"及格线"

内置约 20 项确定性 JS 断言，分四类：常识事实（法国首都是 Paris）、指令遵循（"只回答 yes 或 no"）、上下文复制（答案藏在 prompt 里）、生成健康度（EOS 正常停止、4-gram 重复率超 0.4 判退化）。另有 256 token 强制连续生成的持续速度压测。成绩汇总为 Peak/Sustained Speed、Accuracy、Suite Wall 四维指标，并可生成带硬件信息的 **可分享 PNG 成绩证书**。还能用 JavaScript 自定义考题，甚至一键让 GPT-4 级大模型帮你出题——"大模型出题、小模型考试"闭环。

## 结语

MicroLLM lab 示范了"浏览器即边缘 AI 运行时"的范式：小模型做本地毫秒级初筛与路由，难问题才升级云端。局限是规模止步 360M、不支持多模态，但随着 WebGPU 成熟，这很可能成为端侧智能分发的标准形态之一。

> 本文参考自 [MicroLLM lab — tiny LLMs, Q4, in your browser](https://stateofutopia.com/experiments/microllmlab/)