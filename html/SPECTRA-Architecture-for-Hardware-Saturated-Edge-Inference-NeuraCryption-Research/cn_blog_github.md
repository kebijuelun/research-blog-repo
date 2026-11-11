# SPECTRA：免校准4-bit量化，让4GB显存跑4B模型

SPECTRA 要解决的是消费级边缘 GPU 上“算力充足、显存和带宽却不够”的推理困境：7B 级 FP16 模型放不进 4GB 显存，而逐层换入主机内存又会把生成速度拖到难以交互。它的核心思路是免校准的通道敏感度分析、Group-32 非对称 4-bit 量化，以及把反量化直接融合进 Triton GEMV/GEMM 内核，避免中间 FP16 张量再次挤占显存和带宽。论文在 RTX 3050 Ti Laptop 4GB 上报告了 **2.80×–3.31×** 的模型压缩，并让六款 1.5B–4.1B 模型全部驻留 GPU。最亮眼的一项结果是 DeepSeek-R1-Distill-1.5B 在 CUDA Graph 下达到 **82.53 tok/s**，同时 8,192 token 上下文所需权重与 KV Cache 合计仅 **1,310.8 MB**。

![SPECTRA 封面图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/SPECTRA-Architecture-for-Hardware-Saturated-Edge-Inference-NeuraCryption-Research/images/neuracryption_social_card.png)

> 图解：SPECTRA 面向硬件资源已经饱和的边缘推理场景，研究重点不是继续堆叠计算单元，而是用低位参数表示、免校准压缩和融合执行，绕开显存容量与内存带宽两堵墙。

## 一、为什么 29.8 TFLOPS 的 GPU，跑语言模型仍然很慢？

理解这项工作，先要把“计算能力”和“供数能力”分开。RTX 3050 Ti Mobile 拥有 2,560 个 CUDA Core、80 个第二代 Tensor Core，FP16 峰值算力达到 29.8 TFLOPS；但它的显存接口只有 128 bit，理论带宽为 192.0 GB/s。

自回归解码每步只生成一个 token，即 $M=1$。这时 GPU 很难把一个权重在大量 token 之间复用，往往必须把网络参数从显存重新读入片上寄存器，算术强度只有约 $1\ \mathrm{FLOP/byte}$。换句话说，Tensor Core 像一条高速生产线，权重搬运却像一条窄传送带，生产线大部分时间都在等料。

对未压缩的 14 GB、7B FP16 模型，192.0 GB/s 下完整读取一次约需 73 ms，理论生成上限不足 14 tok/s，超过 95% 的计算能力可能因等待数据而闲置。更现实的问题是，4GB 显存根本容不下 14 GB 模型；即便 7 GB 版本也不行。

传统框架会把部分层卸载到主机 RAM，再通过 PCIe 逐层换入。论文测得 PCIe Gen3 x4 的有效主机到设备带宽约为 3.94 GB/s，对应生成速度可降至 **0.28 tok/s**。对于语音交互、机器人和实时通信，这几乎不可用。

已有 GPTQ、AWQ、SmoothQuant 等训练后量化方法虽然能把权重压到低位，却仍存在三个部署障碍：

- **依赖校准分布**：在 WikiText-2、C4 等通用语料上确定的量化网格，遇到电话转写、JSON、医学术语或其他语言时，可能产生明显的分布偏移。
- **预处理成本高**：二阶 Hessian 统计和多次前向传播可能耗时数分钟到数小时，不适合在设备上动态压缩。
- **架构假设偏窄**：围绕标准 Transformer 设计的校准流程，面对 Mamba-2、ConvNet/SSM 混合结构或 Liquid Neural Network 时，可能需要额外适配甚至直接失败。

SPECTRA 因此把问题拆成压缩、敏感度判断和内核执行三个层次：先确定哪些通道值得保护，再用小组量化压缩主体参数，最后让 GPU 在寄存器中直接消费 4-bit 权重。

| 子系统 | 数学或物理依据 | 算法机制 | 硬件执行位置 |
| --- | --- | --- | --- |
| Rate-Distortion Compressor | 连续 Shannon 率失真函数 $R(D)$ | 带零点的 Group-32 非对称 4-bit 均匀网格 | CPU / 主机内存，无 GPU 校准开销 |
| AST Sensitivity Pre-Scanner | 渐近 Fisher 曲率 $F_{jj}\propto\|\gamma_j\|^2$ | 直接遍历 AST，提取 LayerNorm/RMSNorm 缩放参数 | 主机 Python，延迟低于 50 ms |
| HDSQ-Ultra Outlier Isolation | 四阶总体峰度 $\kappa>3.0$ | 自适应离群比例 $r\le3.0\%$，32-bit 索引打包 | 连续固定主机内存 / TRDC 二进制 |
| Fused W4A16 GEMV Kernel | 最大化 $M=1$ 时的算术强度 | 寄存器内双 nibble 解包、缩放与 FMA 累加 | GPU 寄存器 / 向量 SRAM |
| Batched W4A16 GEMM Kernel | $M\ge2$ 时利用 Tensor Core | 二维输出分块、L2 swizzling、Tensor Core | Ampere Tensor Core，执行 $tl.dot$ |
| SpectraFusedEmbedding | 降低静态参数占用 | 大词表按行动态收集并反量化 | 片上 VRAM，节省 69.4% Embedding 内存 |
| PingPongLayerOffloader | 通过重叠隐藏 PCIe 延迟 | 双缓冲异步 CUDA DMA Stream | PCIe Gen3/4 x4 与两个 GPU 暂存槽 |
| SpectraQuantizedCache | 限制动态上下文内存增长 | Group-32 微缩放 4-bit KV Cache | 4GB GPU 上支持 8,192 token |
| TRDC Serialization Format | 基于 POSIX $mmap$ 的零拷贝映射 | 24 字节固定头与平坦张量缓冲区 | 直接向 GPU 流式传输，标识 $TRDC4B01$ |

这张总表体现了 SPECTRA 的设计取向：**压缩格式与执行内核必须共同设计**。如果只把模型文件变小，推理前却仍完整展开成 FP16，显存墙和带宽墙依然会回来。

## 二、为什么无损压缩不够，必须接受“功能上的有损”？

### 2.1 FP16 权重并没有想象中那么容易无损压缩

经典 Shannon 信源编码定理指出，离散序列 $W$ 的无损平均码长 $L$ 不会低于其熵 $H(W)$：

$$
L \ge H(W)=-\sum_{i=1}^{M}p(w_i)\log_2p(w_i)
$$

其中 $p(w_i)$ 是参数值 $w_i$ 的经验出现概率。16-bit 浮点数可分解为符号位 $s$、指数 $e$ 和尾数 $m$：

$$
w=(-1)^s\cdot2^{e-\mathrm{bias}}\cdot\left(1+\sum_{k=1}^{M}m_k2^{-k}\right)
$$

FP16 的指数位和尾数位分别为 $E=5,M=10$，BF16 则为 $E=8,M=7$。论文对权重参数流形的熵审计显示：

- 符号位近似对称，$H(s)\approx1.0$ bit；
- 指数集中在层相关的动态范围内，$H(e)\in[1.6,2.7]$ bits；
- 随机梯度更新让低位尾数接近均匀噪声，BF16 尾数熵约为 6.8–7.0 bits，FP16 则达到 9.5–9.9 bits。

由此估计，BF16 与 FP16 权重的总离散熵分别约为 10.5 bits 和 13.5 bits。无损压缩比的上限近似为：

$$
R_{\mathrm{lossless}}(\mathrm{FP16})=\frac{16}{13.5}\approx1.19\times,\qquad
R_{\mathrm{lossless}}(\mathrm{BF16})=\frac{16}{10.5}\approx1.52\times
$$

这意味着 zstd、LZ4、fpzip 等位精确压缩器无法把 FP16 权重压到 SPECTRA 所需的 2.80×–3.31×。要跨过这个界限，就必须允许参数发生小幅扰动，同时约束模型功能的失真。

### 2.2 率失真：量化误差来自“舍入”和“截断”两端

连续率失真函数描述在期望失真不超过 $D$ 时所需的最小码率。对零均值高斯信源和均方误差，其下界为：

$$
R(D)=\min_{p(\hat{w}|w):\mathbb{E}[d(w,\hat{w})]\le D}I(W;\hat{W})
=\frac{1}{2}\log_2\left(\frac{\sigma_w^2}{D}\right),\qquad0\le D\le\sigma_w^2
$$

设量化位数为 $b$、截断阈值为 $\alpha$，步长 $\Delta=2\alpha/(2^b-1)$，总失真可以拆成颗粒误差与截断误差：

$$
D(\Delta,\alpha)=\frac{\Delta^2}{12}\int_{-\alpha}^{\alpha}p(w)\,dw
+2\int_{\alpha}^{\infty}(w-\alpha)^2p(w)\,dw
$$

这两项存在直接冲突：为了容纳极端权重而增大 $\alpha$，就会加宽 $\Delta$，让大多数普通权重的舍入误差变大；为了细化网格而缩小 $\alpha$，又会让重尾通道承受严重截断。

峰度用于识别这种重尾程度：

$$
\kappa=\frac{\frac{1}{d}\sum_{i=1}^{d}(w_i-\mu)^4}{\sigma^4}>3.0
$$

SPECTRA 的答案不是用一整行共享一个过宽网格，而是每 32 个权重设置局部 scale 和 zero point，并按需把极端通道分离为 FP16。

### 2.3 三个命题给出误差边界，而不只是经验直觉

**命题一：峰度受矩阵宽度约束。** 对 $K$ 个通道敏感度，标准化后令 $z_j=(S_j-\mu_S)/\sigma_S$，则 $\sum_j z_j^2=K$。由 Cauchy–Schwarz 不等式：

$$
\sum_{j=1}^{K}z_j^4\le\left(\sum_{j=1}^{K}z_j^2\right)^2=K^2
$$

因此：

$$
\kappa=\frac{1}{K}\sum_{j=1}^{K}z_j^4\le K
$$

结合自适应离群比例公式，$K=1,536、3,072、8,192、65,536$ 时，上界分别对应 2.10%、2.20%、2.35%、2.66%。只有 $\kappa$ 超过约 $6.2\times10^5$ 才触及 3.0% 硬上限，所以它更像安全钳，而非常态开销。

**命题二：Group-32 的重建误差不超过半个步长。** 当一组 32 个权重跨越零点，即 $w_{\min}\le0\le w_{\max}$，scale 为 $s=(w_{\max}-w_{\min})/15$，zero point 不会被截到边界，4-bit 网格能够覆盖整个区间。因此：

$$
|w-\hat{w}|\le\frac{s}{2}
$$

论文在一个 $512\times1,024$ 矩阵上观察到最大误差为 $0.49999s$，与该边界一致。这里真正有用的信息是：每组误差由 **局部 scale** 决定，不再被整行的极端值牵着走。

**命题三：高斯分组的误差可以闭式预测。** 32 个标准高斯变量的期望样本极差取 $d\approx4.14$，则 $s=d\sigma/15$，均匀量化误差方差为 $s^2/12$。由此可得：

$$
\mathrm{Relative\ Frobenius\ Error}=\frac{d}{15\sqrt{12}}\approx7.97\%
$$

$$
\mathrm{SNR}=20\log_{10}\left(\frac{15\sqrt{12}}{d}\right)\approx21.97\ \mathrm{dB}
$$

$$
\mathrm{Cosine\ Similarity}\approx1-\frac{(0.0797)^2}{2}\approx0.99682
$$

在 $2,048\times2,048$ 合成高斯矩阵上，实测相对误差 8.08%、SNR 21.85 dB、余弦相似度 0.99676，与预测仅差 0.11 个百分点和 0.12 dB。六款生产模型的实测误差为 8.04%–8.32%，SNR 为 21.59–21.89 dB。

### 2.4 参数有误差，不等于功能一定失真

神经网络不是必须逐位还原的压缩文件。对线性映射 $y=xW$，在最优点 $W^*$ 附近展开任务损失：

$$
\Delta\mathcal{L}\approx\mathrm{Tr}\left((\nabla_W\mathcal{L})^T\Delta W\right)
+\frac{1}{2}\mathrm{vec}(\Delta W)^TH_W\mathrm{vec}(\Delta W)
+\mathcal{O}(\|\Delta W\|_F^3)
$$

局部收敛时一阶梯度近似为零，扰动影响主要由 Hessian $H_W$ 决定。论文假设其特征谱呈快速幂律衰减 $\lambda_i\propto i^{-\alpha},\alpha>1.5$，将参数扰动分为主特征子空间与数值零空间后：

$$
\Delta\mathcal{L}\le\frac{1}{2}\lambda_{\max}(H_W)\|\Delta W_\parallel\|_F^2
+\frac{1}{2}\epsilon_{\mathrm{null}}\|\Delta W_\perp\|_F^2,
\qquad\epsilon_{\mathrm{null}}\approx0
$$

由于有效维度 $d_{\mathrm{eff}}=\mathrm{Tr}(H_W)/\lambda_{\max}$ 远小于总参数量，落在近似零空间内的小扰动不会显著改变 logits。这就是“功能无损”或“功能表示等价”的理论基础：不追求 $\hat W=W$，而追求下游轨迹基本保持一致。

论文据此提出三项判据：

- **拓扑候选保持**：自回归生成中的 $\arg\max$ 轨迹与 FP16 统计对齐；
- **信号保真**：余弦相似度不低于 0.996，SNR 高于 21.5 dB；
- **下游推理保持**：ARC-Challenge 等科学推理任务的结果接近基线，论文报告的变化为 +0.42% 到 +0.85%。

对 RMSNorm 还可写出输出扰动的 Lipschitz 上界。设：

$$
\mathrm{RMSNorm}(h)=\frac{h}{\sqrt{\frac{1}{K}\sum_{k=1}^{K}h_k^2+\epsilon}}\odot\gamma
$$

则 $\|\mathrm{RMSNorm}(h)\|_2\le\sqrt K\|\gamma\|_\infty$，并且：

$$
\|\Delta y\|_2\le\sqrt K\|\gamma\|_\infty\|\Delta W\|_2
$$

这不仅解释了量化误差为何可控，也把归一化仿射参数 $\gamma$ 与通道敏感度联系起来，成为下一节免校准分析的入口。

## 三、压缩管线：不用校准文本，如何找到重要通道？

### 3.1 从归一化增益直接估计敏感度

GPTQ、AWQ 通常需要样本激活来估计 $\mathbb{E}[X^TX]$ 或相关曲率。SPECTRA 换了一个观察角度：现代 Transformer 和状态空间模型中，许多线性投影紧跟 LayerNorm 或 RMSNorm，输入尺度已由网络自身的仿射参数决定。

对于 $y=\mathrm{RMSNorm}(h)W$，第 $j$ 列权重上的梯度为：

$$
\frac{\partial\mathcal{L}}{\partial W_{:,j}}=x_j\frac{\partial\mathcal{L}}{\partial y}
$$

在各向同性激活近似下，Fisher 对角元满足：

$$
F_{jj}=\mathbb{E}[x_j^2]\mathbb{E}\left[\left(\frac{\partial\mathcal{L}}{\partial y}\right)^2\right]
\approx|\gamma_j|^2C_{\mathrm{layer}}
$$

因此，对 $q_{\mathrm{proj}}$、$k_{\mathrm{proj}}$、$v_{\mathrm{proj}}$ 以及 $gate_{\mathrm{proj}}$、$up_{\mathrm{proj}}$ 等前置归一化的输入投影，通道敏感度定义为：

$$
S_j=|\gamma_j|\cdot\|W_{:,j}\|_2
$$

前一项近似曲率增益，后一项保留权重列本身的能量。AST 预扫描器直接从模型结构读取 $\gamma$，不需要跑校准语料，论文报告主机端延迟低于 50 ms。

对于没有前置仿射归一化的 $o_{\mathrm{proj}}$、$down_{\mathrm{proj}}$、$lm_{\mathrm{head}}$，系统令 $\gamma_j=1.0$，退化为 $S_j=\|W_{:,j}\|_2$。这个回退很重要：免校准并不意味着假装所有层都有相同结构，而是明确限定代理指标的适用范围。

### 3.2 用峰度自适应分配离群通道预算

不同层、不同架构的重尾程度差异很大，固定保护 1% 通道并不稳妥。SPECTRA 定义 $\kappa_{\mathrm{excess}}=\max(0,\kappa-3)$，并计算：

$$
r=\mathrm{clamp}\left(r_{\mathrm{base}}\left(1+0.15\ln(1+\kappa_{\mathrm{excess}})\right),0.005,0.03\right)
$$

其中 $r_{\mathrm{base}}=0.01$。接近正态时维持约 1% 的基础比例；重尾增强时，对数项缓慢增加预算，最高不超过 3%。离群索引使用 int32 存储，相比 int64 索引减少一半元数据，并可支持最高约 21.4 亿的特征维度。

被分离的通道以 FP16 保存，剩余稠密张量再进入 Group-32 量化。后面的实验会显示一个有意思的结果：在真实模型的细粒度 Group-32 上，分离离群通道不一定改善 PPL；但在合成极端重尾矩阵上，它又能阻止误差失控。因此，HDSQ-Ultra 更像一种可配置保险，而不是必须永远开启的信仰。

### 3.3 Group-32 非对称量化与 Marlin 交错布局

对每行、每 32 个输入特征形成一组，先计算局部最小值和最大值：

$$
w_{\min}^{(i,g)}=\min_{k\in[0,G-1]}W_{\mathrm{dense}}[i,gG+k],\qquad
w_{\max}^{(i,g)}=\max_{k\in[0,G-1]}W_{\mathrm{dense}}[i,gG+k]
$$

再得到 scale 和 zero point：

$$
s^{(i,g)}=\max\left(\frac{w_{\max}^{(i,g)}-w_{\min}^{(i,g)}}{15},10^{-8}\right)
$$

$$
z^{(i,g)}=\mathrm{clamp}\left(\mathrm{round}\left(-\frac{w_{\min}^{(i,g)}}{s^{(i,g)}}\right),0,15\right)
$$

4-bit 整数最终按 $\hat w=(q-z)s$ 重建。为了匹配 GPU 寄存器布局，SPECTRA 不把相邻的 $2k,2k+1$ 简单放进同一字节，而是采用 Marlin 交错：第 $k$ 个权重放低 4 bit，第 $k+16$ 个权重放高 4 bit：

$$
\mathrm{PackedByte}[i,g,k]=(q_k^{(i,g)}\ \&\ 0x0F)\ |\ ((q_{k+16}^{(i,g)}\ \&\ 0x0F)\ll4),\qquad k\in[0,15]
$$

相邻两组的两个 4-bit zero point 也打包进一个 uint8：

$$
\mathrm{PackedZero}[i,m]=(z^{(i,2m)}\ \&\ 0x0F)\ |\ ((z^{(i,2m+1)}\ \&\ 0x0F)\ll4)
$$

这个设计的聪明之处在于，数据在文件中的排列方式已经靠近内核消费它的方式，减少解包后的跨 lane shuffle。压缩不再只是离线格式问题，而是直接影响每个 token 的执行成本。

### 3.4 为什么生产运行时放弃在线 FWHT？

SPECTRA 也实现了离线 Fast Walsh–Hadamard Transform（FWHT），复杂度为 $\mathcal{O}(d\log_2d)$，可通过正交旋转让权重分布更均匀。但如果在线推理时对每个输入激活都做 Hadamard 旋转与反旋转，逐 token 解码会新增计算和访存开销。

因此，生产版 `engine.py` 与 `kernels.py` 不使用在线 FWHT，只保留 HDSQ-Ultra 和局部 Group-32。对于带宽已经吃紧的边缘 GPU，少做一次运行时变换，往往比继续追求更漂亮的权重分布更划算。

### 3.5 TRDC4B01：加载模型也避免内存峰值

SPECTRA 定义 Tensor Reconstruction Data Container（TRDC）格式，固定头只有 24 字节：

- 0–7 字节：ASCII 魔数 `TRDC4B01`；
- 8–15 字节：小端 uint64 格式版本，目前为 1；
- 16–23 字节：尾部 JSON 元数据字节长度。

后续连续负载依次保存打包的 4-bit 权重、Group-32 FP16 scale、位打包 zero point、稀疏 FP16 离群通道和 int32 索引。尾部 JSON 清单记录模型架构、层图、张量名、形状、stride、分组大小和 CRC32 校验。

运行时通过 POSIX $mmap$ 将文件映射进虚拟内存，再用异步 CUDA DMA 把连续张量切片直接送入 GPU，避免 Python 中间对象和完整主机副本造成内存峰值。它解决的是部署中经常被忽略的一步：**模型压缩成功，不代表加载过程不会先把 16GB 主机内存打满。**

## 四、真正决定速度的部分：4-bit 权重如何直接被计算？

### 4.1 单 token 解码：寄存器内融合 W4A16 GEMV

常规低比特推理常分两次调度：先把权重反量化到 VRAM 中的 FP16 临时缓冲区，再执行 GEMV。这样虽然权重存储变小了，计算前却又制造了一次完整写入和读取，显存流量反而增加。

SPECTRA 的 `_fused_w4a16_gemv_hdsq_kernel` 将四个步骤放进同一条寄存器流水线：

1. 读取打包字节，同时解出高、低两个 nibble；
2. 在寄存器中应用 Group-32 scale 与 zero point；
3. 与激活向量执行 FMA，并累加到 FP32 accumulator；
4. 若存在离群通道，直接收集对应激活和 FP16 权重，在同一 accumulator 中补偿。

结果只把最终 FP16 输出写回，不产生全局内存反量化暂存区。论文报告宽 MLP 块上 DRAM 带宽达到 34.93 GB/s，片上 L2 流式带宽达到 119.91 GB/s。

这个设计的核心价值可以概括为一句话：**W4A16 不应理解成“先 W4 变 W16，再做 A16”，而应让 W4 在寄存器里尽可能短暂地存在。**

### 4.2 Prompt Prefill 与批量生成：切换 Tensor Core GEMM

当 $M\ge2$，负载逐渐从带宽受限转向计算受限，继续使用标量 CUDA Core 并不合适。SPECTRA 的 `_batched_w4a16_gemm_kernel` 使用二维分块和 `tl.dot`，核心累加形式为：

$$
\mathrm{acc}=tl.dot(A_{\mathrm{low}},W_{\mathrm{low}})+tl.dot(A_{\mathrm{high}},W_{\mathrm{high}})
$$

默认分块参数为 $\mathrm{BLOCK\_M}=16、\mathrm{BLOCK\_N}=64、\mathrm{BLOCK\_K}=64$，L2 swizzling 的 $\mathrm{GROUP\_M}=8$。序列维 $M$ 作为动态运行参数传入，而不是编译期常量，从而避免不同长度触发大量 JIT 编译。

离群通道也在内核内通过稀疏收集和 Tensor Core 点积补偿。论文在 $M=4$ 时报告 216.63 tok/s，在 $M=64$ 的批量 Prefill 中报告 5.57 TFLOP/s。

### 4.3 大词表只取当前需要的 Embedding 行

NVIDIA Nemotron-3-Nano-4B 的词表达到 131,072，隐藏维为 4,096，仅输入 Embedding 与输出投影的 FP16 权重就约占 1.05 GB，超过 4GB 显存的四分之一。

SpectraFusedEmbedding 将 Embedding 保持为 4-bit Group-32，推理时只收集当前 batch 中实际出现的 token ID，并即时反量化这些行，而不是展开完整矩阵。静态 Embedding VRAM 从 1,050 MB 降至 **321.4 MB**，节省 **69.4%**。

### 4.4 显存仍不够时，用双缓冲隐藏 PCIe 传输

对于更长上下文下的 3B/4B 模型，PingPongLayerOffloader 在 GPU 上保留 Slot 0 和 Slot 1 两个层暂存区：

- 时刻 $t$：Compute Stream 从 Slot 0 执行第 $N$ 层；
- 同一时刻：Prefetch Stream 通过非阻塞 DMA 将第 $N+1$ 层送到 Slot 1；
- 时刻 $t+1$：计算流等待预取事件后执行第 $N+1$ 层，另一流开始向 Slot 0 搬运第 $N+2$ 层。

论文称该流水线最多可将 **88%** 的 PCIe 传输延迟隐藏在层计算之后。它并不能消除带宽上限，但能减少“搬一层、停一下、算一层”的串行空转。

### 4.5 KV Cache 也压到 4-bit，长上下文才不会反噬显存

标准 KV Cache 随上下文长度 $T$ 线性增长：

$$
\mathrm{Mem}_{\mathrm{KV}}=2LN_{\mathrm{kv\_heads}}d_{\mathrm{head}}T\cdot\mathrm{sizeof}(\mathrm{dtype})
$$

DeepSeek-1.5B 在 8,192 token 下，FP16 KV Cache 超过 1.5 GB。SpectraQuantizedCache 将 Key 和 Value 原地量化为 Group-32 非对称 4-bit，使占用从 **1,572.8 MB 降至 393.2 MB**，减少 75%。

只压缩权重而不管 KV Cache，长上下文仍会把刚省下的空间吃回去。权重与缓存使用同一组量化思想，是 SPECTRA 能面向实际边缘部署而非只展示模型文件大小的关键。

### 4.6 Prompt Lookup 与硬件自动探测

对代码缩进、JSON 等重复结构，SPECTRA 还集成 n-gram Prompt Lookup：直接从上下文提取候选片段，用批量 Tensor Core GEMM 并行验证，再精确接受或回滚状态，无需额外 draft model。

分配张量之前，`hardware.py` 通过 NVML 查询可用显存、PCIe 带宽、Compute Capability 和热状态，生成 ExecutionPlan，自动选择完全 GPU 驻留、Ping-Pong 卸载或启用 4-bit KV Cache。换句话说，压缩格式之上还有一层根据实时硬件约束做决策的运行时。

## 五、实验：六款模型全部驻留 4GB GPU，速度和质量如何？

### 5.1 测试平台与测量方式

全部实验使用 NVIDIA GeForce RTX 3050 Ti Laptop GPU：

- GA107 芯片，20 个 SM、2,560 个 CUDA Core、80 个第二代 Tensor Core；
- 4,096 MiB GDDR6，Linux 桌面合成器开销后可寻址上限为 3,770.6 MiB；
- 128-bit、12.0 Gbps 显存接口，理论带宽 192.0 GB/s；
- PCIe Gen3 x4，实测主机到设备吞吐 3.94 GB/s；
- Linux 6.8.0、NVIDIA Driver 550.120、CUDA 12.4、PyTorch 2.4.0+cu124、Triton 3.0.0+。

后台线程每 100 ms 通过 pynvml/NVML 采样 VRAM、PCIe 吞吐、GPU 利用率、温度和功耗，避免只依赖理论 Roofline 推算。

### 5.2 多架构主结果：压缩 2.80×–3.31×，无主机交换

| 基础模型 | 架构类别 | 激活参数 | 原始 FP16 | SPECTRA 4-bit | 压缩比 | Active VRAM | Eager Tok/s | Graph Tok/s | 硬件状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-1.5B | Dense Transformer | 1.5B | 3.31 GB | 1.03 GB | 3.21× | 917.6 MB | 25.78 | 82.53 | 完全驻留 |
| LiquidAI LFM-3-1.7B | Hybrid ConvNet/SSM | 1.7B | 3.22 GB | 1.00 GB | 3.22× | 1,029.1 MB | 18.40 | 32.40 | 完全驻留 |
| LiquidAI LFM-2.5-2.6B | Hybrid ConvNet/SSM | 2.6B | 5.02 GB | 1.52 GB | 3.30× | 1,556.0 MB | 9.80 | 14.20 | 完全驻留 |
| IBM Granite-4.2-3B | Hybrid Mamba-2 | 3.2B | 6.82 GB | 2.06 GB | 3.31× | 2,245.0 MB | 5.60 | 8.07 | 完全驻留 |
| NVIDIA Nemotron-3-Nano-4B | Mamba-2，131k 词表 | 3.8B | 6.33 GB | 2.26 GB | 2.80× | 2,326.9 MB | 14.10 | 22.40 | 完全驻留 |
| Alibaba Qwen-3.5-4B | Transformer，GQA | 4.1B | 8.68 GB | 2.63 GB | 3.30× | 2,690.5 MB | 9.20 | 14.55 | 完全驻留 |

这里需要保留测试口径：82.53 tok/s 是 **1.5B 模型配合 CUDA Graph** 的结果，并非所有 4B 模型都能达到。更大的 Qwen-3.5-4B 冷启动 Graph 速度为 14.55 tok/s，仍明显高于人约 4–5 tok/s 的阅读速度。

从模型类别看，结果覆盖 Dense Transformer、ConvNet/SSM 和 Mamba-2，说明管线没有绑定某一种 Attention 或 SwiGLU 结构。Nemotron 的压缩比相对较低，为 2.80×，也提示大词表和架构细节会影响最终收益，不能简单把“4-bit”等同于严格 4×。

### 5.3 从 Eager 到 CUDA Graph：同一模型为何能从 25.78 到 82.53 tok/s？

| 内核执行状态 | Batch / Seq（$M$） | 运算强度 | 实测 DRAM 带宽 | 片上 L2 流式带宽 | 吞吐 / 算力 |
| --- | --- | --- | --- | --- | --- |
| 单 token 自回归，Eager | $M=1$ | $1.01\ \mathrm{FLOP/B}$ | $20.89\ \mathrm{GB/s}$ | $48.20\ \mathrm{GB/s}$ | $25.78\ \mathrm{tok/s}$ |
| 单 token 自回归，CUDA Graph | $M=1$ | $1.01\ \mathrm{FLOP/B}$ | $34.93\ \mathrm{GB/s}$ | $119.91\ \mathrm{GB/s}$ | $82.53\ \mathrm{tok/s}$ |
| Prompt Prefill / 批量生成 | $M=4$ | $3.92\ \mathrm{FLOP/B}$ | $72.40\ \mathrm{GB/s}$ | $146.50\ \mathrm{GB/s}$ | $216.63\ \mathrm{tok/s}$ |
| Batched Tensor Core Prefill | $M=64$ | $48.5\ \mathrm{FLOP/B}$ | $112.80\ \mathrm{GB/s}$ | $182.40\ \mathrm{GB/s}$ | $5.57\ \mathrm{TFLOP/s}$ |

对 1.5B FP16 模型，论文估算每 token 传输需 17.2 ms，理论上限约 58 tok/s；压到 1.03 GB 后，读取时间为 5.36 ms。融合 Triton 内核配合 CUDA Graph 静态调度后达到 82.53 tok/s，论文将其描述为达到理论硬件总线饱和度的 92.4%。

Eager 与 Graph 的算术强度相同，速度却相差约 3.2 倍，说明小 $M$ 下除了 DRAM，还存在内核启动、Python 调度与 L2 命中等系统开销。只看权重位宽，无法解释全部加速。

### 5.4 逐层误差审计：余弦相似度均超过 0.9966

| 投影算子 | 矩阵维度 | 相对 Frobenius 误差 | SNR（dB） | 余弦相似度 | 最大组相对误差 |
| --- | --- | --- | --- | --- | --- |
| model.layers.0.self_attn.q_proj | $1,536\times1,536$ | $8.12\%$ | $21.81$ | $0.99671$ | $0.49991s$ |
| model.layers.0.self_attn.k_proj | $256\times1,536$ | $8.24\%$ | $21.68$ | $0.99662$ | $0.49988s$ |
| model.layers.0.mlp.gate_proj | $8,960\times1,536$ | $8.06\%$ | $21.87$ | $0.99675$ | $0.49995s$ |
| model.layers.0.mlp.down_proj | $1,536\times8,960$ | $8.04\%$ | $21.89$ | $0.99676$ | $0.49999s$ |
| model.layers.14.mlp.up_proj | $8,960\times1,536$ | $8.08\%$ | $21.85$ | $0.99673$ | $0.49994s$ |
| model.layers.27.self_attn.o_proj | $1,536\times1,536$ | $8.14\%$ | $21.79$ | $0.99669$ | $0.49992s$ |

这些数值与命题二、三相互印证：最大组误差贴近但不超过 $s/2$，整体 Frobenius 误差约 8%，SNR 稳定在 21.5 dB 以上。它们证明的是张量重建质量，并不自动等于任务质量，所以还需要 PPL 和推理基准。

### 5.5 “离群点悖论”：真实模型上，额外保护反而让 PPL 变差

| 量化配置 | 有效位率 | Active VRAM | WikiText-2 PPL | 相对 FP16 的变化 |
| --- | --- | --- | --- | --- |
| 未压缩 FP16 | $16.00\ \mathrm{bpb}$ | $3,310\ \mathrm{MB}$ | 33.83 | $0.00$，参考 |
| SPECTRA Pure Group-32，0% 离群分离 | $4.625\ \mathrm{bpb}$ | $917.6\ \mathrm{MB}$ | 34.73 | +0.90 |
| SPECTRA HDSQ-Ultra，1% 离群分离 | $4.785\ \mathrm{bpb}$ | $942.1\ \mathrm{MB}$ | $35.62$ | $+1.79$ |
| SPECTRA HDSQ-Ultra，2% 离群分离 | $4.945\ \mathrm{bpb}$ | $978.4\ \mathrm{MB}$ | $36.14$ | $+2.31$ |

在 DeepSeek-R1-Distill-1.5B 的 32,768 连续 token WikiText-2 测试中，Pure Group-32 的 PPL 最低。论文解释为：小组局部归一化已经能吸收多数离群方差，额外把通道拆到 FP16 稀疏矩阵，反而在相邻通道之间引入量化边界摩擦。

这个结果很有价值，因为它没有简单重复“离群点必须全部高精度保存”的经验规则。但也应注意，ARC-Challenge 上 1% 离群版本表现略高，因此配置选择仍取决于目标指标，不能只凭单一 PPL 排名下结论。

### 5.6 ARC-Challenge：低比特版本略高于 FP16 基线

| 模型表示 | 正确题数 | 总题数 | 准确率 | 95% Wilson 置信区间 |
| --- | --- | --- | --- | --- |
| 未压缩 FP16 | 407 | 1,172 | 34.73% | $[32.0\%,37.5\%]$ |
| SPECTRA Pure Group-32 | 412 | 1,172 | 35.15% | $[32.4\%,38.0\%]$ |
| SPECTRA HDSQ-Ultra，1% 离群 | 417 | 1,172 | 35.58% | $[32.8\%,38.4\%]$ |

两个量化版本分别比 FP16 高 0.42 和 0.85 个百分点。论文将小幅提升解释为量化噪声的隐式正则化，但更稳妥的读法是：三者的 Wilson 置信区间高度重叠，至少没有观察到推理准确率下降，不宜据此宣称量化必然提升推理能力。

在 100 个多步推理 Prompt、3,200 个评估 token 的确定性审计中，SPECTRA 与 FP16 rollout 的首 token 一致率为 **91.0%**，论文同时报告 **100% 语义任务一致性**。

### 5.7 极端重尾压力测试：HDSQ-Ultra 仍然是必要的保险

| 重通道倍率 $m$ | 每行均匀 4-bit | 仅 Group-32，无离群保护 | SPECTRA HDSQ-Ultra Group-32 |
| --- | --- | --- | --- |
| $m=1$ | $8.12\%$ | $8.08\%$ | 8.04% |
| $m=4$ | $14.85\%$ | $8.22\%$ | 8.06% |
| $m=8$ | $28.40\%$ | $8.45\%$ | 8.08% |
| $m=16$ | $52.10\%$ | $8.92\%$ | 8.09% |
| $m=32$ | $84.60\%$ | $9.85\%$ | 8.10% |
| $m=64$ | $96.30\%$ | $11.40\%$ | 8.11% |

实验使用 $1,536\times1,536$ 合成矩阵，将 1% 输入通道放大 $m$ 倍。每行均匀量化在 $m=64$ 时误差达到 96.30%，Pure Group-32 也升至 11.40%，HDSQ-Ultra 则几乎保持在 8.11%。

这与上一节并不矛盾：常规真实模型上，Group-32 已足够；遇到远超常态的通道放大时，FP16 分离提供额外安全边界。工程上应根据实际峰度和任务验证决定是否付出离群存储与收集开销。

### 5.8 长上下文、多会话与硬件利用率

在 8,192 token 下，DeepSeek-1.5B 的 FP16 权重与 KV Cache 合计 $3,310+1,572.8=4,882.8$ MB，超过 4GB 硬件容量。SPECTRA 权重为 917.6 MB，量化 KV Cache 为 393.2 MB，总计 **1,310.8 MB**，留下超过 **2,459 MB** 的余量。

连续 10 次、上下文长度动态变化的推理会话审计中，论文报告 **0.0 MB 内存泄漏**，CUDA Graph 内存池和 pinned DMA buffer 在会话结束后均完全回收。

| 硬件指标 | 实测值 | 物理硅上限 | 利用率说明 |
| --- | --- | --- | --- |
| DRAM 读取带宽 | $34.93\ \mathrm{GB/s}$ | $192.0\ \mathrm{GB/s}$ | 18.2%，受 $M=1$ 小分块限制 |
| L2 Cache 流式带宽 | $119.91\ \mathrm{GB/s}$ | $128.0\ \mathrm{GB/s}$ | 93.7% L2 效率 |
| 向量寄存器压力 | 每线程 64 个 | 每线程 255 个 | 无寄存器 spill 到本地内存 |
| Active Warp Occupancy | $83.4\%$ | $100.0\%$ | 较高的延迟隐藏效率 |
| PCIe Host-to-Device 吞吐 | $3.94\ \mathrm{GB/s}$ | $3.94\ \mathrm{GB/s}$，Gen3 x4 | 100.0% 总线饱和 |

这张表也纠正一个容易产生的误解：82.53 tok/s 并不意味着 DRAM 已被 100% 吃满。单 token 场景的 DRAM 利用率只有 18.2%，高 L2 流式效率、寄存器无溢出和 Warp 延迟隐藏同样重要。

## 六、与已有量化框架相比，SPECTRA 的差异在哪里？

| 框架 | 是否需校准数据 | 数值量化方案 | 内核集成与运行时 | 4GB 边缘显存可行性 | SSM / Mamba-2 兼容性 |
| --- | --- | --- | --- | --- | --- |
| SPECTRA | 无需，解析 AST 敏感度 | Group-32 微缩放 4-bit，打包零点 | 寄存器融合 Triton GEMV/GEMM | 100% 驻留，82.53 tok/s | 原生通用支持 |
| GPTQ | 需要，小时级 GPU 校准 | 二阶 OBS 列逆 | 外部 C++/CUDA 编译内核 | 部分可行，主机 RAM 分配峰值 | 非 Transformer 层失败 |
| AWQ | 需要样本 Prompt Batch | 激活感知逐通道缩放 | W4A16 CUDA GEMM | 部分可行，Prefill 初始化延迟高 | 需要手动层映射 |
| SpQR | 需要高激活扫描 | 稀疏 FP16 离群点 + Group-16 | Sparse SpMM，线程分歧高 | 不可行，稀疏索引内存峰值 | 不兼容状态张量 |
| SmoothQuant | 需要激活协方差 | W8A8 矩阵乘法 | INT8 Tensor Core GEMM | 不可行，大于 2B 模型易超 4GB | 循环门控上退化 |
| QuIP# / QuaRot | 需要不相干统计 | 随机 Hadamard 正交旋转 | 在线 FWHT 变换内核 | 不可行，在线旋转延迟瓶颈 | 不兼容线性注意力 |
| Marlin | 不适用，属于计算内核格式 | W4A16 硬件分块交错 | 手工 CUDA/PTX | 快，但限已量化模型 | 仅矩阵执行 |
| llama.cpp | 可选 Imatrix 校准 | K-Quants 块分组 | SIMD/Metal/cuBLAS | 可行，但 CPU-GPU 传输延迟高 | 每种模型需专门 C++ 移植 |
| HQQ | 无需，闭式优化 | Half-Quadratic Quantization | PyTorch 运行时反量化 | 不可行，在 VRAM 展开为 FP16 | 兼容线性投影 |
| QLoRA / NF4 | 无需，信息论网格 | NormalFloat4 分位映射 | 计算前反量化 | 不可行，内存流量翻倍 | 通用线性层支持 |

表中的兼容性和可行性结论来自论文的比较口径，不应脱离其测试实现理解为对所有版本的永久判断。就架构思想而言，差异主要有四点：

- **相对 GPTQ**：不通过校准数据计算 $(XX^T)^{-1}$，而从归一化参数 $\gamma$ 推导通道敏感度，论文称扫描低于 50 ms；
- **相对 AWQ**：不搜索激活感知 scale，而使用 Group-32 局部网格和峰度控制的离群预算；
- **相对 SpQR**：不依赖高线程分歧的稀疏 SpMM，而在融合 GEMV 内直接收集并累加离群通道；
- **相对 QuIP#/QuaRot**：生产运行时不在线旋转激活，避免额外 FWHT 延迟。

HQQ 与 NF4 同样能免校准，因此“零校准”本身不是 SPECTRA 独有的全部贡献。更完整的组合是：**零校准敏感度代理、适配执行布局的小组格式、无 VRAM 暂存的融合内核，以及跨 Transformer/SSM 的运行时。**

## 七、适用边界、工程取舍与复现入口

### 7.1 “架构无关”具体指什么？

SPECTRA 操作的是归一化层之后的线性代数映射，而不是硬编码 LLaMA 的 Attention 头数或 MLP 比例：

- Dense Transformer：自动识别 RMSNorm/LayerNorm，将 $\gamma$ 绑定到输入投影；
- LiquidAI ConvNet/SSM：可处理短卷积 token mixer 与前馈状态块，无需修改卷积内核；
- Mamba-2：覆盖结构化状态空间层、selective scan 投影，并用按行反量化处理 131k 大词表。

不过，未前置归一化的投影仍需退回列范数敏感度；非线性、状态张量和自定义算子也不等于自动获得同一套最优策略。这里的“通用”主要指压缩管线与线性投影执行路径，而不是所有模型组件都无需验证。

### 7.2 为什么选 Group-32，而不是更细或更粗？

- Group-16 的有效位率为 5.25 bpb，scale 和 zero point 元数据达 1.25 bpb，占总占用超过 24%，增加显存总线压力；
- Group-32 的有效位率为 4.625 bpb，元数据只有 0.625 bpb，占 13.5%，论文将其视为 Pareto 拐点；
- Group-64/128 可进一步降到 4.31–4.16 bpb，却扩大局部参数范围，使重尾值降低网格分辨率并推高 WikiText-2 PPL。

因此，Group-32 不是“量化误差最小”的选择，而是误差、元数据和带宽之间的折中。对内存受限设备，过少与过多的元数据都可能成为问题。

### 7.3 持续运行还要考虑热降频

在 35W–45W 笔记本上，持续满载会使核心温度达到 $75^\circ\mathrm{C}$–$82^\circ\mathrm{C}$ 并触发热降频。Qwen-3.5-4B 初始 Graph 解码为 14.55 tok/s，连续生成 10 分钟后稳定在 **12.08 tok/s**。

这个结果比冷启动峰值更接近真实部署。即使降频后，论文估算其仍为人类 4–5 tok/s 阅读速度的 2.4 倍以上，并能维持在较安静的 35W 功耗包络内。

### 7.4 开源实现与最小使用流程

论文开放了压缩器、Triton 内核、TRDC 规范和模型权重。命令行流程如下：

```bash
git clone https://github.com/NeuraCryption/SPECTRA.git
cd SPECTRA
pip install -r requirements.txt

python -m spectra.cli compress \
  --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./models/deepseek-r1-1.5b-spectra.trdc \
  --mode pure-group32

python -m spectra.cli run \
  --model-path ./models/deepseek-r1-1.5b-spectra.trdc \
  --prompt "Explain quantum decoherence in simple terms:" \
  --max-tokens 512

python -m spectra.server \
  --model-path ./models/deepseek-r1-1.5b-spectra.trdc \
  --port 8080
```

实现要求 PyTorch 不低于 2.2，并使用 Triton。核心引擎面向 NVIDIA CUDA 与 AMD ROCm，另提供 PyTorch 参考实现用于 CPU 和边缘 NPU 环境；实际性能仍取决于对应后端是否具备相同的融合内核能力。

开放资源：

- Zenodo 永久归档：<https://doi.org/10.5281/zenodo.24058202>
- GitHub：<https://github.com/NeuraCryption/SPECTRA>
- Hugging Face：<https://huggingface.co/NeuraCryptionIndia>
- OpenAIRE 索引：<https://explore.openaire.eu/search/publication?pid=10.5281/zenodo.24058202>

论文的理论脉络可追溯到 Shannon 信源编码与率失真理论；量化对比涵盖 GPTQ、AWQ、SpQR、SmoothQuant、QuIP#、QuaRot、HQQ、QLoRA/NF4；系统侧则结合 Marlin、Triton、Prompt Lookup、KV Cache 量化和 Mamba-2 等工作。完整出版物列有 28 项参考文献，上述 Zenodo 归档包含 40 页论文及完整证明。

## 八、总结：SPECTRA 真正打通的是“压缩到执行”的链路

- **问题定位准确**：单 token 解码的算术强度仅约 $1\ \mathrm{FLOP/B}$，边缘 GPU 的主要瓶颈是权重供数，而不是 Tensor Core 峰值算力。
- **免校准有明确依据**：对前置归一化线性层，利用 $F_{jj}\approx|\gamma_j|^2C_{\mathrm{layer}}$ 推导敏感度，再以列范数作为无归一化层的回退。
- **Group-32 是格式与硬件的折中**：4.625 bpb 的有效位率控制元数据开销，局部 scale 将单权重建模误差限制在 $s/2$ 内。
- **融合内核让压缩收益真正落到速度上**：解包、反量化、FMA 和离群补偿在寄存器中完成，避免 FP16 中间张量往返 VRAM；CUDA Graph 下 1.5B 模型达到 82.53 tok/s。
- **系统覆盖完整**：权重、Embedding、KV Cache、模型加载、PCIe 预取和硬件探测共同优化，使六款 1.5B–4.1B 模型在 4GB GPU 上完全驻留，并在 8,192 token 下保留充足显存余量。

未来更值得验证的是这些结论在更多真实领域数据、更大模型、AMD/非 NVIDIA 后端及长期热稳态下能否一致成立；同时，Pure Group-32 与离群保护之间的配置选择，也需要由具体任务而非单张量误差来决定。

> 本文参考自 [SPECTRA: Architecture for Hardware-Saturated Edge Inference | NeuraCryption Research](https://neuracryption.com/research)