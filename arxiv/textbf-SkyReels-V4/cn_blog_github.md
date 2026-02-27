# SkyReels-V4：统一多模态视频-音频生成、修复与编辑的基础模型

## 这篇论文在解决什么问题
过去一年视频生成从 **单模态 T2V** 走向 **联合视频-音频生成**，同时又出现了 **多模态参考输入**（图片、视频、音频、mask 等）的需求。现实痛点是：现有系统往往只能覆盖其中一部分，比如能做音频但不支持复杂编辑，或者能做多模态参考却没有音频输出。SkyReels-V4 的目标是把这些能力 **统一到一个模型里**，同时保持 **高分辨率、长时长、可用的效率**。

## 核心贡献一览
- 提出 **双流 MMDiT**：视频分支 + 音频分支，共享一个 MLLM 文本编码器，实现多模态指令一致理解。
- 用 **通道拼接式 inpainting** 统一了 T2V、I2V、视频延展、视频编辑等任务。
- 引入 **低分辨率全序列 + 高分辨率关键帧** 的效率策略，再用超分 + 插帧 Refiner 恢复质量。
- 在公开评测与自建 VABench 上展示领先的人类偏好与细粒度质量。

---

## 模型总体架构
![模型架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/textbf-SkyReels-V4/figures/model_archtecture.png)
> 图解：模型采用 **双流 MMDiT**，视频分支与音频分支并行生成，通过双向 cross-attention 对齐时序；共享的 MLLM 文本编码器统一理解文本、图像、视频、音频等多模态输入。

---

## 方法细节：怎么做、为什么这样做

### 1. 双流 MMDiT：视频与音频同时生成
SkyReels-V4 采用对称双流结构：  
- 视频流初始化自预训练 T2V 模型  
- 音频流从头训练，但结构与视频流一致  
- 两者共享 **同一 MLLM 文本编码器**，保证多模态语义一致性  

#### 混合双流/单流块
前 M 层是 **Dual-Stream**（视频/音频与文本分别投影但一起做注意力），后 N 层合并成 **Single-Stream** 以提高效率。

核心注意力形式：

$$
\mathbf{x}'_v, \mathbf{x}'_t = \text{Attention}([\mathbf{Q}_v; \mathbf{Q}_t], [\mathbf{K}_v; \mathbf{K}_t], [\mathbf{V}_v; \mathbf{V}_t])
$$

#### 文本语义强化
为了避免后期单流阶段语义稀释，对视频流加了额外 **文本 cross-attention**：

$$
\mathbf{x}''_v = \mathbf{x}'_v + \text{Attention}(\mathbf{Q}=\mathbf{x}'_v, \mathbf{K}=\mathbf{x}_t, \mathbf{V}=\mathbf{x}_t)
$$

#### 视频-音频双向对齐
每层都有 **音频 attend 视频** + **视频 attend 音频**：

$$
\mathbf{a}'_i = \mathbf{a}_i + \text{CrossAttn}(\mathbf{a}_i, \mathbf{v}_i), \quad
\mathbf{v}''_i = \mathbf{v}'_i + \text{CrossAttn}(\mathbf{v}'_i, \mathbf{a}'_i)
$$

#### RoPE 时间尺度对齐
视频 latents 只有 21 帧，但音频有 218 token，需要缩放音频 RoPE 频率：

$$
\text{scale} = \frac{21}{218} \approx 0.09633
$$

#### Flow Matching 训练目标

$$
\mathcal{L}_{\text{flow}} =
\left\| \mathbf{v}_\theta^v - (\mathbf{z}_v^0 - \boldsymbol{\epsilon}_v) \right\|^2
+
\left\| \mathbf{v}_\theta^a - (\mathbf{z}_a^0 - \boldsymbol{\epsilon}_a) \right\|^2
$$

---

### 2. 统一的 inpainting 视角：一套接口覆盖所有任务
视频分支输入：

$$
\mathbf{Z}_{\text{input}} = \text{Concat}(\mathbf{V}, \mathbf{I}, \mathbf{M})
$$

不同任务通过 mask 配置实现：  
- **T2V**：$\mathbf{M} = \mathbf{0}$  
- **I2V**：$M_{t=0} = 1$  
- **视频延展**：$M_{t<k} = 1$  
- **插帧**：首尾帧为 1，其余为 0  
- **编辑**：局部 mask 为 1，其他为 0  

视频改动后，音频分支从头生成但通过跨模态注意力对齐时序。

---

### 3. 多模态引用与 In-Context Learning
通过 MLLM 提取多模态语义，再把视觉参考 **直接拼到注意力序列里**：

$$
\mathbf{Z}_{\text{attn}} = [\mathbf{Z}_{\text{cond}}; \mathbf{Z}_{\text{video}}]
$$

为了区分 reference 与生成内容，使用 **时间偏移 RoPE**：

$$
\text{RoPE}(t=-N_{\text{cond}}+i), \quad \text{RoPE}(t=j)
$$

音频参考同理输入音频流，保证风格与声学一致。

---

## 关键效率策略：低分辨率全序列 + 高分辨率关键帧
为了支持 **1080p / 32 FPS / 15s**，直接扩散生成成本太高，因此采用分层策略：

1. 先生成 **低分辨率全序列**  
2. 同时生成 **高分辨率关键帧**  
3. 再用 Refiner 做超分与插帧  

---

## Refiner：视频超分 + 插帧一体化
![Refiner 架构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/textbf-SkyReels-V4/figures/VSR-architecture-new.png)
> 图解：先对低分辨率序列插值到高分辨率，再用关键帧替换对应位置，最后输入 DiT 精修，实现高画质与时间一致性。

Refiner 内部采用 **VSA 稀疏注意力**，将注意力成本降低约 **3×**。

---

## 数据与训练策略

### 数据管线亮点
- 图像：去重 + 质量筛选 + 聚类平衡  
- 音频：分类、SNR/MOS 过滤、VAD、Whisper 转写、统一 caption  
- 视频：语义分段 + 过滤 + 运动多样性平衡 + SyncNet 对齐筛选  

### 分阶段训练
- **Video Pretrain**：T2I → T2V → Inpaint → 多分辨率 → 多模态参考  
- **Audio Pretrain**：数十万小时语音 + 15s 片段  
- **Joint Training**：T2V + T2AV + T2A  
- **SFT**：多模态 500 万 → 高质量 100 万  

---

## 评测结果

### Artificial Analysis Arena
![AAT2VA 评测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/textbf-SkyReels-V4/figures/AAT2VA.png)
> 图解：在 T2V+Audio 赛道中，SkyReels-V4 排名 **第二**，与 Veo 3.1、Sora-2、Kling 3.0 等模型同台竞争。

### 绝对打分（Likert）
![Absolute 评分](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/textbf-SkyReels-V4/figures/evaluation/Absolute-all.png)
> 图解：五大维度中，SkyReels-V4 在 **Prompt Following** 与 **Motion Quality** 上优势明显，整体平均分最高。

### GSB 对比评测
![GSB 评测](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/textbf-SkyReels-V4/figures/evaluation/GSB-Overall.png)
> 图解：在 Good/Same/Bad 评价中，SkyReels-V4 的 “Good” 比例显著高于主流商业模型。

---

## 应用能力示例：多模态参考 + 编辑能力
这里只展示一个典型例子，完整案例在论文 Appendix。

![多模态参考示例](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/textbf-SkyReels-V4/figures/mo2v/reels_output_demo.jpg)
> 图解：多角色多语音参考的多镜头生成，模型能同时对齐 **角色身份** 与 **语音风格**，并保持镜头间一致性。

---

## 论文的核心价值总结
SkyReels-V4 的价值不只是画质与音质，更重要是 **统一性**：  
- **一个模型** 同时支持生成、修复、编辑  
- **一套接口** 适配多模态输入  
- **高分辨率长时长** 仍保持可用效率  

这意味着它更接近可落地的 **视频创作基础设施**，而不是单一任务模型。

---

> 本文参考自 SkyReels-V4（arXiv:2602.21818）