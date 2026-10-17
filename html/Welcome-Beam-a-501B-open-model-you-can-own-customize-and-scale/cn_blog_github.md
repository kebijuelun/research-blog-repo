# Beam：501B 开放 MoE 实战

![原文封面图：Hugging Face 吉祥物与蓝色背景上的彩色同心波纹](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/beam-banner.jpg)

把前沿规模的编程与 Agent 模型放到自己的硬件上，真正难的是权重、KV cache 和后训练都要同时落地。Reflection 的首个模型 Beam 给出的方案是一套 Apache 2.0 开放模型与配套工程栈：501B 参数 MoE，每个 token 激活约 23B 参数，支持 1M token 文本上下文，并提供 Transformers、vLLM、SGLang、TRL 和 Miles 配方。NVFP4 将约 1 TB 的 BF16 权重压缩到约 280 GB；Reflection 提供的 SWEBench Verified 分数在 BF16、FP8、NVFP4 下分别为 81.6%、80.2%、80.7%。对希望自主部署和定制的团队，这比单纯开放 checkpoint 更完整，但并不等于低成本：多 GPU 仍是默认前提。

## Beam 开放的不只是权重

Beam 是美国新实验室 Reflection 发布的首个模型，面向编程、长上下文推理和 Agent 工作。它是纯文本输入、纯文本输出模型，不是多模态模型；目标用户也不是只想调用一次 API 的开发者，而是希望在自有硬件、基础设施或云上运行，并继续修改模型的团队。

模型之外，官方还提供一个开源的 Rust 专用 harness，称其用于发挥 Beam 的最佳性能。推理侧有原生 Transformers 实现，以及发布首日可用的 vLLM、SGLang 支持；训练侧则给出 SFT 和 RL 配方。Beam 还支持跨轮保留推理内容的 `preserve_thinking`、专用工具调用语法和放在 tokenizer 中的响应模板。

推理强度分为 `low`、`medium`、`high`、`xhigh`、`max` 五档，其中 `medium` 是默认值。这些档位直接改变生成时的推理投入，后文的五项试用会展示：提高档位不仅可能改变答案，也会显著改变 token 消耗和格式遵循。

![Beam 整体架构：3 个稠密层后接 49 个 MoE 层，并采用混合注意力和加权残差连接](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/beam-architecture.png)

> 图解：Beam 的 decoder 共 52 层，前 3 层使用稠密 SwiGLU MLP，后 49 层换为 MoE；注意力由滑窗层与全局层混合组成，残差流则使用随深度变化的权重。

## 1M 上下文如何控制缓存开销

Beam 的 52 层 decoder 隐藏维度为 5,120，但并不是每一层都读取完整上下文。43 层只看前 512 个 token 的滑动窗口；每第六层以及最后一层读取全局上下文，因此全局层共有 9 层。局部层负责短程关系，少量全局层再把远处信息带回当前计算。

两类层的位置处理也不同。滑窗层使用旋转位置编码（RoPE），因为短窗口内相对位置最重要；全局层不使用位置编码，而是给每个 query 乘以一个随可见 token 数缓慢增长的系数：

$$
s(pos)=\left(1+\frac{pos+1}{2048}\right)^{0.1}
$$

这个系数类似 softmax 的温度调节，避免上下文变长时注意力分布失去区分度，实现中称为 scalable-softmax query scaling。两类注意力块都采用分组查询注意力（GQA），包含 48 个 query head、8 个 key-value head，并对 query 和 key 做 RMSNorm；输出投影前，还有一个由层输入计算的学习型 sigmoid gate，对每个 head 的每个通道进行缩放。

![Beam 注意力块结构](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/beam-attention.png)

> 图解：注意力块结合 GQA、Q/K 归一化与学习型门控；滑窗层使用 RoPE，全局层改用 scalable-softmax query scaling 处理长上下文。

混合注意力最直接的收益体现在 KV cache。每层每个 token 需要缓存 $2\times8\times128=2,048$ 个数值；如果 52 层都保存完整序列，BF16 下每增加一个 token，缓存就增长 208 KiB。Beam 只有 9 个全局层随序列持续增长，43 个滑窗层各自保留固定的 512 token，因此增长部分降至每 token 36 KiB，约为全全局方案的六分之一。

| 上下文长度 | 如果 52 层全部为全局注意力 | Beam |
| --- | --- | --- |
| 32K tokens | 6.5 GiB | 1.2 GiB |
| 128K tokens | 26 GiB | 4.6 GiB |
| 完整上下文（1M） | 208 GiB | 36.08 GiB |

这些数字的条件不能省略：它们按**每条序列、BF16、Transformers dynamic cache** 计算，并不是整台服务器或单张 GPU 的总显存占用。即便 Beam 把 1M 上下文的缓存降到 36.08 GiB，权重、激活值和并发请求仍要另外占用显存。

## 501B 容量与 23B 激活如何共存

MoE 可以直观理解为“专家很多，但每次只叫少数专家值班”。Beam 的后 49 层各有 256 个专家，每个专家都是中间维度 2,560、约 39M 参数的 SwiGLU MLP；所有专家合计占模型参数的 98.4%。每个 token 只经过其中 8 个专家，所以激活计算规模约为 23B 参数。

路由器沿用 DeepSeek-V3 的分组限制方案：先用 sigmoid 给 256 个专家打分，将其分为 8 组、每组 32 个；再按组内最高分与次高分之和选出 4 组，最后只在这 4 组中选择 8 个专家。Beam 使用与 DeepSeek-V3 相同的分组大小，但不设共享专家；专家输出按归一化后的路由权重组合，再整体乘以 2.5。

![Beam 的分组专家路由过程](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/beam-expert-routing.png)

> 图解：路由器不是直接在全模型中取 top 8，而是先对 256 个专家评分、选出 4 个候选组，再从中路由到 8 个专家。

这里必须区分“计算量”与“驻留内存”：每个 token 只运行 8 个专家，但全部 256 个专家都要保存在内存或显存中。原文把它概括为“501B 模型的内存需求、23B 模型的运行速度”；更严谨地说，稀疏激活降低了每 token 的计算量，实际端到端速度仍受专家通信、显存带宽和 serving 实现影响，原文没有给出与 23B 稠密模型的延迟对比。

### 残差流是另一项关键改动

标准 Transformer 将每个子层输出以权重 1 加回残差流，因此随着层数增加，累计隐藏的尺度可能不断增长。

![标准 Transformer 的残差流](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/standard-residual-stream.png)

> 图解：标准残差连接直接累加各子层输出，所有新增项的权重都是 1。

Beam 把输入 embedding 计为第 1 步。第 $k$ 次加入子层输出时，已有残差流乘以 $\sqrt{(k-1)/k}$，新输出乘以 $1/\sqrt{k}$：

$$
\mathbf h_k=\sqrt{\frac{k-1}{k}}\,\mathbf h_{k-1}+\frac{1}{\sqrt{k}}\,\Delta\mathbf h_k
$$

两个系数的平方和为 1，因此残差流从首层到末层保持恒定尺度。经过全部 104 次累加后，输入 embedding 与各子层输出在最终隐藏状态中的权重相同。

![Beam 的恒尺度加权残差流](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/beam-residual-stream.png)

> 图解：Beam 根据深度重新分配旧残差与新子层输出的权重，避免残差流随层数不断放大，并让各部分在最终状态中等权。

每个子层前后还各有一次归一化，最终 logits 经过范围为 $\pm30$ 的 tanh softcap。实现明确让 router、SwiGLU 乘积、norm、RoPE、query scaling 和 softcap 使用 FP32 计算，以减少低精度 checkpoint 下关键环节的数值误差。

## 三种精度差多少，硬件要多少

Reflection 团队给出了 BF16、FP8、NVFP4 三种 checkpoint 的基准结果。下表完整保留这些数值；它反映的是不同精度下 Beam 的表现，不包含其他模型基线。

| 类别 / 基准 | Beam (BF16) | Beam (FP8) | Beam (NVFP4) |
| --- | --- | --- | --- |
| Agentic Coding / Terminal |  |  |  |
| DeepSWE v1.1 | 43.3% | 43.0% | 42.7% |
| SWE Bench Pro v2-Hard | 76.5% | 78.4% | 78.6% |
| SWE Bench Pro v1 | 66.1% | 65.9% | 65.6% |
| Terminal Bench v2.1 | 80.1% | 78.9% | 78.2% |
| SWE Atlas Codebase QnA | 31.8% | 32.2% | 31.4% |
| SWEBench Multilingual | 78.3% | 78.1% | 77.5% |
| SWEBench Verified | 81.6% | 80.2% | 80.7% |
| Reasoning |  |  |  |
| AIME 2026 | 97.7% | 98.0% | 97.5% |
| HLE（不使用工具） | 36.0% | 36.0% | 34.8% |
| HLE（使用工具） | 51.1% | 51.1% | 51.1% |
| SciCode | 49.2% | 48.3% | 46.9% |
| CriPT AA | 16.3% | 17.4% | 15.4% |
| GPQA Diamond | 90.6% | 90.3% | 89.3% |
| LiveCodeBench | 88.3% | 88.1% | 87.4% |
| Tool Calling / Search |  |  |  |
| AutomationBench（仅 public split） | 39.3 | 38.2 | 39.8% |
| MCP Atlas | 80.8% | 81.5% | 80.5% |
| Tau3 Banking | 36.1% | 36.8% | 35.4% |
| BrowseComp（使用 context management） | 76.8% | 76.7% | 75.6% |
| DeepSearchQA（使用 context management） | 80.5% | 79.6% | 81.1% |
| General Capabilities |  |  |  |
| AA-LCR | 78.2% | 79.4% | 78.7% |
| LongBench v2 | 67.5% | 67.6% | 67.2% |
| IFBench | 79.6% | 80.7% | 79.2% |
| AA Omniscience（index，仅 public split） | 10.8 | 8.8 | 9.8 |

表中的数值与百分号按原文保留，未替原始列补齐单位。量化结果也不是单向变化：NVFP4 在 SWE Bench Pro v2-Hard 上为 78.6%，高于 BF16 的 76.5%；但在 Terminal Bench v2.1 上为 78.2%，低于 BF16 的 80.1%。因此，既不能把这些点估计概括成“量化完全无损”，也不能仅凭某一项下降就判定低精度不可用。原文没有在这张表中给出重复运行方差或跨模型比较。

部署时，官方认为最实际的方案是 NVFP4：多数权重使用 4 bit，少量为保持精度而保留 BF16，使 checkpoint 从约 1 TB 降到约 280 GB。NVIDIA Blackwell 可原生运行 NVFP4，H200 等更早的 GPU 不能，应改用 FP8 checkpoint；FP8 也适合 Hopper 和 Ampere 系列。

一张 B200 有 180 GB HBM3e，两张共 360 GB，虽然能装下权重，但留给激活值和 KV cache 的空间很少。官方建议按 4 张 B200（共 720 GB）规划，以容纳长上下文与并发请求；其他可选配置包括完整的 8×B200 节点、2 张各 288 GB 的 B300，或 8×RTX Pro 6000。没有合适的自有硬件时，也可以先通过 serverless Inference Providers 测试。

## 从加载模型到工具调用

Beam 的推理链路已经覆盖本地加载、多卡 serving 和托管 API。Transformers 需要 5.20.1 或更高版本：

```bash
pip install -U "transformers>=5.20.1"
```

最简单的入口是 text-generation pipeline：

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation", model="reflection/Beam-1.0-501B-A23B", device_map="auto"
)

messages = [
    {"role": "user", "content": "Write a limerick about Python exceptions."}
]
output = pipe(messages, max_new_tokens=2000)
output[0]["generated_text"][-1]
```

pipeline 返回整段对话，最后一条就是 Beam 的回复，且已由响应模板解析。更底层的用法是 `AutoModelForCausalLM` 与 `AutoTokenizer`：加载模型后，用 `apply_chat_template` 构造输入，用 `generate` 生成，再调用 `tokenizer.parse_response`：

```python
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="medium",
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=2000)

message = tokenizer.parse_response(
    outputs[0][input_len:], prefix=inputs["input_ids"][0]
)
```

`parse_response` 会按 tokenizer 自带的模板，把原始输出拆成包含推理、答案和工具调用的 message 字典，不必再用正则表达式切分。传入的 `prefix` 也不能省略：chat template 可能预先写入 assistant 回合的一部分，解析器需要看到这部分才能正确理解后续结构。

工具调用同样依赖这套解析器。Beam 用自有 XML 风格语法内联书写参数，裸值 `7` 既可能是数字，也可能是字符串；因此 `apply_chat_template` 和 `parse_response` 都要接收 `tools`，后者才能依据工具 schema 将每个参数转换为声明类型。这种设计与 Beam 的训练输出格式一致，减少了解析规则与模型输出各自演进而失配的风险。

### 多 GPU：专家并行之外还有一份复制开销

Beam 的配置自带专家并行方案：不同专家分布到不同设备，而注意力、稠密层、embedding 和 LM head 在每张卡上复制。被复制的部分共有 8B 参数，BF16 下即 16 GB。通过 `DistributedConfig` 可以同时设置张量并行与专家并行：

```python
import os

from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "reflection/Beam-1.0-501B-A23B",
    distributed_config=DistributedConfig(
        tp_size=int(os.environ["WORLD_SIZE"]),
        ep_size=int(os.environ["WORLD_SIZE"]),
        tp_plan={},
    ),
)
```

用 `torchrun` 在 16 张 H200 上启动时，每张卡约持有 78 GB 权重；若只用 8 张，每卡会达到 139 GB，在 141 GB H200 上几乎没有剩余空间。设备总数还必须整除专家数 256。

### vLLM 与 SGLang：服务化配置已经给出

vLLM 在发布首日即可运行 Beam。下面用 16 张 GPU（两台 8×H200 节点）进行张量并行，启动 BF16 checkpoint，并暴露 OpenAI 兼容 API：

```bash
vllm serve reflection/Beam-1.0-501B-A23B \
  --tensor-parallel-size 16 \
  --served-model-name beam
```

需要更快生成时，可以加入 DFlash2 speculative decoding。小型 drafter 一次最多提议后续 15 个 token，Beam 在一次前向传播中统一检查，因此单步可能接受多个 token，代价是 drafter 额外占用一些显存：

```bash
vllm serve reflection/Beam-1.0-501B-A23B \
  --tensor-parallel-size 16 \
  --speculative-config '{ "method": "dflash", "model": "reflection/dflash2-drafter-preview", "num_speculative_tokens": 15 }'
```

两种服务都可从 OpenAI 客户端访问 `http://localhost:8000/v1`。

SGLang 的官方示例默认启用 DFlash2。Blackwell 上可用 4 张 B300 服务 NVFP4 checkpoint：

```bash
python -m sglang.launch_server \
  --model-path reflection/Beam-1.0-501B-A23B-NVFP4 \
  --served-model-name beam \
  --tp 4 \
  --attention-backend trtllm_mha \
  --page-size 64 \
  --context-length 1048576 \
  --mem-fraction-static 0.85 \
  --enable-unified-memory \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2 \
  --speculative-dflash-block-size 8 \
  --tool-call-parser beam --reasoning-parser beam \
  --host 0.0.0.0 --port 30000
```

Hopper 上则改用 FP8 checkpoint 和 8 张 H200。其余参数不变，只改三项：模型路径换成 `reflection/Beam-1.0-501B-A23B-FP8`，`--tp` 改为 8，注意力后端从需要 Blackwell 的 `trtllm_mha` 改为 `fa3`。

如果暂时不部署自己的集群，Hugging Face Inference Providers 已接入 Baseten、Fireworks 和 Together，模型名使用 `reflection/Beam-1.0-501B-A23B:auto`。Inference Endpoints 也提供 FP8、NVFP4 的一键部署配方，或通过 CLI 安装：

```bash
hf endpoints catalog deploy --repo reflection/Beam-1.0-501B-A23B --name beam
```

## 五项试用：高推理档位买来了什么

Hugging Face 团队还通过 Reflection 平台做了五项“vibe evals”，每项分别以 `low` 和 `high` 推理强度运行一次。它们不是正式 benchmark，价值在于观察具体任务中的行为、token 成本和失败方式；完整输入、输出、API 用量、任务 fixtures、grader 与会话 trace 已放入公开 Hugging Face bucket。

| 类别 | 任务说明 | Low：结果 / tokens | High：结果 / tokens |
| --- | --- | --- | --- |
| 多步工具调用 | 完成订单 ORD-742：比较库存、预留、附加费和到达日期，预留符合条件的最低成本仓库 | 通过，544 | 通过，811 |
| 代码库长上下文检索 | 从三个完整源文件推导 Click 命令配置与两次调用结果，并引用六个实现符号 | 未通过（格式），1,613* | 通过，5,007 |
| 沙盒终端任务 | 检查并修复 CSV 账本 CLI；修完所有 bug、保留测试，并运行测试和 CLI 检查 | 通过，582 | 通过，1,548 |
| 数学推理（AIME 风格） | 求集合 {1, …, 16} 中有多少个八元素子集，其元素和可被 17 整除，并给出证明 | 部分通过，829† | 通过，2,809 |
| 结构化输出 | 将乱序修订日志转换为严格 JSON，处理补丁合并、取消、null、日期、货币和总计 | 通过，1,040 | 通过，2,088 |

\* Low 的长上下文回答其实得到了全部代码行为和六个源码引用，但没有符合 parser 要求的格式；作者认为这部分也涉及 prompt 歧义。  
† Low 给出了正确数学答案，却漏掉题目要求的证明；High 给出了正确的单位根证明，并经过独立复核。

这组小样本最有信息量的地方，不是“High 全面获胜”，而是多出来的 token 用在了哪里。五项任务的 High 都消耗更多 token；在长代码上下文和数学任务中，它补上了 Low 的格式或证明缺陷。但每项只运行一次，且长上下文失败还受提示歧义影响，不能据此估计稳定的成功率差距。

## 后训练：先改行为，再改能力

部署解决“能不能跑”，后训练才回答“能不能变成自己的模型”。Beam 的官方配方覆盖两条路径：用 TRL 做 LoRA 监督微调，以及用 Miles 做全参数或 LoRA 的 GRPO 强化学习。

### LoRA SFT：让推理语言跟随用户

Hugging Face 团队用 TRL 的 `SFTTrainer` 和 PEFT，在 Multilingual-Thinking 数据上训练 LoRA adapter，基础模型的 501B 参数保持冻结。目标很具体：让 Beam 用用户要求的语言进行推理，而不是只把最终答案翻译成目标语言。

为控制显存并提高吞吐，训练组合了加载时模型分片、专家并行、FlashAttention 4 和 sequence packing。LoRA SFT 可在一个 8×B300 节点上运行，训练吞吐约 19,000 tokens/s；对应的 H100 设置需要 4 个节点。

经过一轮 Multilingual-Thinking 训练后，Beam 在 16 条留出 prompt 中有 16 条按要求语言推理，基础模型只有 3 条。这个 16/16 对 3/16 的结果衡量的是**语言遵循**，并不证明推理内容本身更正确。

![TRL LoRA SFT 的 500 步训练曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/trl-sft-training.png)

> 图解：左图为 training loss，右图为 mean token accuracy。曲线来自另一个在 open-r1/Mixture-of-Thought 推理子集上运行 500 步的官方 TRL 示例，使用单个 8×B300 节点；它不是上述 16 条语言遵循评测的曲线。

### 全参数 GRPO：训练与 rollout 如何配合

Reflection 与 RadixArk 还为 Miles 构建了 BF16 GRPO 配方，既可更新全模型，也可只训练 LoRA adapter。其分工是 Megatron 负责训练、SGLang 生成 rollout，Miles 在每次更新后把新权重发布给 rollout 引擎。不同工作负载的参考硬件如下：

| 工作负载 | 参考配置 |
| --- | --- |
| 推理 / 评测，BF16 | 每个引擎 16×H200 |
| 全参数 GRPO，BF16（Miles） | 96×H200：64 张训练、32 张 rollout；或使用 64×H200 共置运行 |
| LoRA GRPO，BF16（Miles） | 32×H200：16 张训练、16 张 rollout |
| 全参数 GRPO，BF16，128 token 上下文（Miles） | 48×H200：32 张训练、16 张 rollout；这是 Reflection 与 RadixArk 最初在真实 checkpoint 上验证的设置 |

配方在 DAPO-Math-17K 上训练，并在 AIME25 上评估。原文示例面向由 12 个 8×H200 节点组成的 Ray 集群，流程包括构建镜像、准备模型与数据、提交任务：

```bash
docker build -f docker/Dockerfile.beam -t miles-beam .

python scripts/run_beam.py prepare \
  --model-id reflection/Beam-1.0-501B-A23B \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets

export MASTER_ADDR=10.0.0.1
export RAY_ADDRESS=http://${MASTER_ADDR}:8265

python scripts/run_beam.py execute \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets \
  --output-dir /root/shared_data/beam-run \
  --num-rollout 100
```

一次 rollout 的具体过程如下：

- **采样**：输入 32 条 prompt，每条生成 8 个回答，共 256 个样本；两个 SGLang 引擎各使用 16 张 GPU。
- **打分**：把 boxed answer 与标签核对，对接近长度上限的回答施加软性惩罚。
- **更新**：对全部 256 个样本执行一次 optimizer step，使用 group-relative advantages 和 truncated importance sampling，不加 KL 或 entropy penalty。
- **并行训练**：张量并行 8、专家并行 8、数据并行 8，同时启用序列并行和 full recompute。
- **优化器**：Adam 使用恒定学习率 $10^{-6}$；梯度为 BF16，master weights 为 FP32，动量为 BF16。优化器状态通过节点本地 NVMe 流转，因此每个训练节点需要 2 TB 临时存储。
- **评估与保存**：每 5 次 rollout 评估一次 AIME25，每题采样 8 次；每 20 次 rollout 保存一个 checkpoint。

![全参数 GRPO 在 AIME 2025 上的评测曲线](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/miles-grpo-aime2025.png)

> 图解：在 4,000 token 输出上限下，全参数 GRPO 的 AIME 2025 准确率从 46.25% 提升到 55.8%。这是该数学训练配方的结果，不能直接外推到编程或其他 Agent 任务。

系统会在当前批次训练时生成下一次 rollout，所以新生成的回答落后一个更新版本。中断后可用 `--resume` 从最近 checkpoint 继续；`execute-colocate` 则能在 8 个节点上共置运行，让每张 GPU 轮流承担训练与 serving。

### LoRA GRPO：把硬件需求降到 32 张 H200

LoRA 版本使用 4 个 8×H200 节点，其中 16 张训练、16 张生成 rollout。它训练 rank-16 adapter，覆盖 Beam 的注意力、稠密 MLP 和专家层，基础权重保持冻结。由于只需把 adapter 权重同步给 SGLang，且优化器状态可留在 GPU 显存中，不再需要进行 NVMe optimizer offloading。

```bash
python scripts/run_beam_lora.py execute \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets \
  --output-dir /root/shared_data/beam-run \
  --num-rollout 100
```

![LoRA GRPO 在 AIME 2025 上的准确率与输出长度变化](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/miles-lora-grpo-aime2025.png)

> 图解：该次 LoRA GRPO 训练中，AIME 2025 准确率从 42.9% 升至 64.2%，平均输出长度同时从 3,188 降至 2,197 token。它展示了此配方下准确率与输出效率可以一起改善，但单一数学评测不足以证明 LoRA 在所有任务上都优于全参数训练。

## 三个 Demo 展示了什么，也没展示什么

### BeamSpeak：从需求到原生 macOS 听写应用

Hugging Face 团队让 Beam 配合 Pi coding harness，构建原生 macOS 菜单栏听写应用 BeamSpeak。需求包括紧凑的录音悬浮条、无主窗口和账号、用 ⌘⇧Space 启停录音，以及显示录音时长、麦克风音量和转写状态；语音识别完全在本地通过 MLX 运行 Parakeet v3，并在多次录音间保持模型加载。

应用还要在目标应用保持焦点且获得辅助功能权限时粘贴文本，否则复制到剪贴板，同时保留更新的剪贴板内容且绝不自动按回车。权限、取消、静音、错误、两分钟录音上限和临时音频清理也都在需求之内。

这个案例展示的是 Beam 处理较长工程需求与本地模型集成的过程，而不是“一次提示生成最终成品”。保存的运行记录包含后续 prompt，最终源码还包含后来由 Codex 完成的修复；原文给出的构建要求也是完整需求的压缩版本。

### 多 Agent 编程：实验设计比结论更值得关注

第二个实验把 Beam 接入开源 Pi coding agent，比较三种方式：单 Agent 独立工作、5 个持久对等 Agent 通过共享 Git 仓库协作，以及一个 lead Agent 向异步子 Agent 委派任务。实验运行在 Hugging Face Jobs 上，推理走 Reflection 平台 API。

任务来自 ProgramBench：Agent 只能根据编译后的可执行文件和文档重建程序，看不到源代码或评测测试。每种方式在每个任务上共享同样的 8M API token 上限和两小时求解时限，最终实现用 ProgramBench-Tests 的固定隐藏测试集评分。

原文的分数—延迟图以平均测试通过比例为分数；时间轴不是实际墙钟时间，而是用固定参考速率折算模型 token 耗时，再加上实测工具耗时。实验最初包含 5 个任务，但当前图只展示 jq、xsv、FIGlet、hexyl 这 4 个事后选定的任务；ripgrep 因保存产物的评分过慢而从三种设置中全部排除，发布结果仍保留原始 15 个 episodes。[实验页面](https://burtenshaw-beam-pi-programbench.hf.space/study)提供了运行记录入口。

因此，这个 Demo 能说明 Beam 可以进入多 Agent 编程实验框架，却不能凭当前图证明“多 Agent 一定更快”。任务子集是事后筛选的，时间轴又是估算口径；原文这段也没有给出三组设置的数字对比。

### 事故分诊：把政策变成 0.8B 决策模型

第三个实验从 Pi 中的 Beam 工作流开始，目标是把一份开发事故分诊政策变成小型、快速的决策模型。模型需要读取事故工单，直接判断负责团队、严重等级以及是否立即 ping 相关人员。这里最终被微调的不是 Beam，而是 Qwen3.5-0.8B；团队使用 Unsloth 训练 LoRA adapter 和决策头。

工作流生成了 2.3k 条合成事故，并按事故 cluster 划分训练、校准和测试集。训练使用 4 bit backbone，在 Science Slurm 集群的一张 H100 80GB 上运行 3 个 epoch。基线则把完整分诊政策和工单直接提示给未适配模型，不给示例，要求输出三项 JSON 决策。

| 指标 | 零样本提示基线 | 微调决策模型 |
| --- | --- | --- |
| 三项决策全部正确的比例（同一批 320 条合成测试工单） | 4.69% | 76.25% |
| 平均单工单延迟 | 1,218.06 ms | 102.58 ms |

![事故分诊决策模型的准确率与延迟对比](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Welcome-Beam-a-501B-open-model-you-can-own-customize-and-scale/images/decision-model-results.png)

> 图解：微调后，三项决策全对率从 4.69% 提升到 76.25%，平均延迟从 1,218.06 ms 降至 102.58 ms，相当于 11.9× 加速。结果仅适用于这套合成工单与分诊政策，不代表真实事故分布上的线上效果。

这个案例与 Beam 本身的关系，是编码 Agent 帮助完成了数据、训练和评估工作流；最终执行决策的是一个 0.8B 小模型。原文还提供了包含数据、代码、checkpoint 和结果的 artifact bucket，便于复查实验。

## 谁适合现在上手 Beam

如果你已有多张 Blackwell 或 Hopper GPU，并需要私有化部署、长上下文 Agent 或自有后训练，Beam 值得做一次真实业务回放；没有集群时，更稳妥的起点是先用 Inference Providers 或 Endpoints，在自己的任务上比较不同推理档位的正确率、格式、token 数和延迟，再决定是否投入 4 张 B200 级别的部署或 32 至 96 张 H200 的 GRPO。现有证据支持“三种精度都有可用结果、官方配方能改善特定训练目标”，但尚缺独立复测、跨模型基线和 1M 上下文真实负载数据；多 Agent 与 macOS Demo 也都有明确的归因和评测边界。

> 本文参考自 [Welcome Beam: a 501B open model you can own, customize, and scale](https://huggingface.co/blog/reflection-beam)