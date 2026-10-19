# AWQ 精华总结：1% 关键通道 + 等效缩放，让 13B 模型跑上 8GB 笔记本

想在端侧本地运行 LLM，瓶颈不是算力而是显存和内存带宽，而低比特量化又容易损伤模型能力。AWQ（激活感知权重量化）用激活分布识别关键权重并做等效缩放，配合 TinyChat 推理框架，在桌面和移动 GPU 上相对 Hugging Face FP16 实现 **3.2–3.3×** 平均加速，并让 8GB 显存的 RTX 4070 笔记本以 **33 Token/s** 运行 Llama-2-13B。

## 核心观察：权重重不重要，看激活而非权重本身

- LLM 中少量 Salient Weights（显著权重）对输出影响更大；判断它们用**激活幅值**比用权重自身范数有效得多。
- 对照实验（OPT，INT3-g128）：OPT-1.3B 的 RTN 困惑度 119.00，按激活保留 1% FP16 通道降到 16.91；按权重分布选择则高达 98.55。
- 但混合精度布局对硬件不友好，AWQ 要的是统一 INT3/INT4 格式下达到同样效果。

## 方法：等效缩放保护关键通道

- 不训练、不重构：离线统计平均激活幅值，显著权重乘 $s>1$、对应激活除以 $s$，网络功能不变，但显著权重的相对量化误差约被压低为 $(\Delta'/\Delta)\cdot(1/s)$。
- $s$ 并非越大越好：OPT-6.7B 上 $s=2$ 时 PPL 11.92 最优，$s=4$ 反而回升到 12.36（21.2% 的分组步长被拉大，伤害非显著权重）。
- 搜索空间压缩为单参数 $\mathbf{s}=\mathbf{s}_{X}^{\alpha}$，$\alpha$ 在 $[0,1]$ 网格搜索即可，无需反向传播。
- 优势：无 GPTQ 式重构过拟合风险；校准需求低（16 条序列相当于 GPTQ 的 192 条）；跨分布校准时 PPL 仅恶化 0.50–0.60，GPTQ 则达 2.33–4.89。

## 结果数字

- **精度**：LLaMA/Llama-2 7B–70B 上全面优于 RTN、GPTQ、GPTQ-R；GPTQ 在 LLaMA-7B INT3 下依赖 Reorder 才有 6.53，AWQ 不用技巧达 6.35。Mistral/Mixtral INT4 仅比 FP16 增加 0.11/0.16 PPL。
- **任务**：MBPP pass@1 达 40.64（高于 FP16 的 38.53）；GSM8K 70B 达 56.40（FP16 为 56.41）；OpenFlamingo INT4 32-shot 相对 FP16 损失从 RTN 的 4.57 缩小到 1.17 CIDEr。
- **速度**：Llama-2-7B 在 RTX 4090 上从 52 提到 194 Token/s；Jetson Orin 最高 3.5×；相对 AutoGPTQ 在部分模型上加速至少 2.6×；Raspberry Pi 4B 上可跑 7B 模型。

## 系统：TinyChat 兑现压缩收益

端侧逐 Token 生成是内存带宽受限（FP16 算术强度约 1），4bit 权重理论上可带来 4× 读取量下降。TinyChat 用三项优化把压缩兑现为速度：即时反量化融合进矩阵乘主循环、SIMD 感知权重打包、Kernel 融合。

## 适用边界

INT4/INT3 已接近实用无损；INT2 需与 GPTQ 叠加（如 OPT-30B PPL 11.38，优于单 GPTQ 的 11.75）但距 FP16 仍有明显差距；实际速度依赖硬件与 Kernel，不能只按位数推算。

> 本文参考自 [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)