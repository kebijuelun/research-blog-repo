# OpenResearcher：把 Deep Research 训练从“昂贵在线爬网”变成“可复现离线工厂”

## 先说结论：这篇论文到底解决了什么问题？

如果你做过 Deep Research Agent（需要反复搜索、打开网页、定位证据、再推理）的训练，会很快遇到三个现实问题：

1. 在线搜索 API 太贵，失败轨迹也要按次付费。  
2. 真实网页环境不稳定，今天能复现，明天就可能失效。  
3. 很难做可控分析：你很难判断错误出在检索、浏览还是推理。

OpenResearcher 的核心贡献是：把“多轮搜索-浏览-推理”主循环完整搬到离线环境，同时保留真实网页任务的噪声和复杂度，最终构建出一个 **全开源、可复现、可规模化** 的长程轨迹合成流水线。

---

## 背景：为什么既有方法不够用？

以往不少方法要么只能生成 2-5 步的短轨迹，要么依赖在线搜索 API。  
这与真实的深度研究任务并不匹配：现实任务经常需要几十到上百次工具调用，过程中还要不断修正假设、重新检索、交叉验证证据。

论文提出了一个关键视角：  
并非模型“不会想”，而是训练数据中缺少长程、真实、可复现的搜索行为轨迹。

---

## 方法总览：OpenResearcher 三阶段流水线

![Pipeline](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis/img/Openresearcher-synthesis-v5.png)
> 图解：整体流程分三段——(1) 从 MiroVerse 选难题；(2) 一次性在线引导构建离线语料；(3) 在离线环境中用 teacher 反复生成长轨迹。这样把一次性联网成本和后续大规模合成成本彻底解耦。

### 第一步：挑“真难题”，而不是短链 QA

- 数据来源：MiroVerse-v0.1。  
- 从全量中采样 10%，得到约 6K 问答。  
- 仅保留清洗后的高质量 question-answer 对，不直接使用原始轨迹监督。

### 第二步：一次性在线引导 + 离线大语料

- 对 6K 问题进行一次性 online bootstrapping，检索到约 10K gold documents。  
- 再并入 15M FineWeb 文档（约 10T tokens）作为大规模干扰语料。  
- 使用 Qwen3-Embedding-8B 编码，基于 FAISS 建索引，形成离线搜索引擎。

### 第三步：显式浏览工具 + 长程轨迹合成

![Tools](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis/img/tools.png)
> 图解：`search` 负责粗召回，`open` 负责文档级查看，`find` 负责页内证据定位。三者形成从“全局语料 → 文档 → 证据片段”的逐级缩焦机制。

- `search`：返回 Top-K 标题/链接/摘要。  
- `open`：打开完整网页文本。  
- `find`：在当前页面做精确字符串定位。  

Teacher 使用 GPT-OSS-120B，在离线环境生成 97K+ 轨迹，并包含大量 100+ tool calls 的长尾样本。

---

## 形式化建模（保留核心公式）

论文将 agent 过程写成标准 ReAct 轨迹：

$$
\mathcal{H}_{T}=\{(q,s_0,\mathcal{T}_{meta}),(r_1,a_1,o_1),\dots,(r_T,a_T)\}
$$

$$
r_t,a_t \sim \pi(\cdot \mid \mathcal{H}_{t-1})
$$

$$
\mathcal{H}_{t}=\mathcal{H}_{t-1}\cup\{(r_t,a_t,o_t)\}
$$

直观理解：每一步先思考（reasoning），再调用工具（action），拿到观测（observation）后继续迭代，直到给出 final answer。

---

## 训练与评测设置（可复现细节）

- Student 基座：Nemotron-3-Nano-30B-A3B。  
- SFT 训练数据：约 55K 正确轨迹（也做了 incorrect-only 对照）。  
- 训练资源：8×H100，约 8 小时。  
- 关键超参：学习率 $5\times10^{-5}$，最大序列 256K，347 steps，global batch size 64。  
- 评测集：BrowseComp-Plus（离线闭网）、BrowseComp/GAIA/xbench-DeepSearch（在线开放网络）。

---

## 主结果：离线合成数据真的能训出强 Agent

![Main Result](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis/img/teaser.png)
> 图解：在 BrowseComp-Plus 上，OpenResearcher-30B-A3B 达到 54.8%，显著高于多数强基线。纵向看，对同一基座模型提升约 +34.0 个百分点，说明训练信号确实来自长程轨迹本身。

关键指标（论文报告）：

- BrowseComp-Plus：54.8%  
- 相比 base model（20.8%）：+34.0 pts  
- BrowseComp / GAIA / xbench：26.3% / 64.1% / 65.0%

一个很硬的结论是：  
只用离线合成轨迹做 SFT，也能迁移到真实在线搜索基准，不依赖在线训练数据。

---

## 轨迹统计：失败样本为什么更长？

![Turn Distribution](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis/img/turn_distribution.png)
> 图解：正确轨迹多数集中在中短区间（约 10-40 次调用）；错误轨迹分布更宽且中位数更高。说明“多调用”不等于“更接近答案”，很多是搜索漂移带来的无效探索。

- 全部轨迹成功率约 56.7%。  
- 失败轨迹平均调用 71.7 次，成功轨迹仅 38.4 次。  
- 失败增量主要来自 `search`，而非 `find`。  
- 启示：瓶颈在查询构造与检索策略，而不只是文档内定位。

---

## 成本与工程价值：离线化有多省？

- 论文统计 5.76M 次搜索请求的成本对比：  
  - Serper：约 \$5,760  
  - SerpAPI：约 \$28,800  
  - OpenResearcher 离线检索：约 \$0（边际请求成本）

除成本之外，作者还强调了三个工程优势：

1. 无速率限制，可大规模并行合成。  
2. 环境确定性高，结果可重复。  
3. 不依赖私有基础设施，便于社区复现实验。

---

## Ablation：5 个研究问题讲清“为什么有效”

![Turn Budget Sweep](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis/img/chart1_hit_rate.png)
> 图解：随着最大轮次提升，准确率与 gold hit rate 持续上涨；约 100 turns 后进入平台期。说明长程探索必要，但预算并非无限越大越好。

### RQ1：只用正确轨迹训练才最好吗？

不是。  
correct-only、incorrect-only、all 三种设置最终分数非常接近（都在 54-55% 左右）。  
结论：错误轨迹同样包含有价值的工具使用结构信号。

### RQ2：一次性在线 bootstrapping 必要吗？

非常必要。  
去掉 gold docs 后，BC+ 从 54.81 直接降到 6.35，几乎崩塌。  
这说明离线语料覆盖率是第一性问题。

### RQ3：turn budget 多大合适？

收益在前期明显，约 100 turns 后边际递减。

### RQ4：显式浏览工具真的有用吗？

有，而且是阶梯式提升：  
`search only` < `search+open` < `search+open+find`。  
三工具全开时，准确率与命中效率都最好。

### RQ5：检索到 gold doc 就一定答对吗？

不一定。  
`open-hit` 条件下正确率显著高于仅 `search-hit`。  
这说明“看到摘要”不够，必须“打开 + 定位 + 推理”。

![Open-hit Heatmap](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/arxiv/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis/img/chart2_heatmap.png)
> 图解：热力图把“首次命中轮次”和“命中文档数”与最终准确率对应起来。无 open-hit 的格子准确率显著偏低，说明证据暴露是必要条件；但同一命中层里仍有错误，说明推理链条依然可能失败。

---

## 失败案例带来的方法论启发

论文附录给出了大量案例，失败模式很典型：

1. 只有 `search` 工具时，模型会在摘要层反复兜圈。  
2. 找到正确页面后，仍可能在表格解析、字段对齐上发生推理错误。  
3. 某些样本会进入“高轮次、低进展”的循环。  

这也解释了作者为何坚持三工具设计：  
`search` 管召回，`open` 管验证，`find` 管证据锚定，缺一都容易退化。

---

## 我的总结：这篇工作真正“新”在哪

这篇论文并非单纯报告一个更高分模型，而是把“数据生产线”本身做成了研究对象。其创新点在于：

- 把在线依赖压缩为一次性 bootstrapping。  
- 把浏览过程拆成可分析的最小原语。  
- 把轨迹合成、训练、评测、误差分析放进同一可复现实验闭环。  

如果你要做开源 Deep Research Agent，这套范式比“只堆模型参数”更具可持续性：它直接解决了数据规模、实验可复现和失败可诊断这三类最难的工程问题。

> 本文参考自 [OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis](https://arxiv.org/abs/2603.20278)