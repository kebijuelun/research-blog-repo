# gitalpha：不盯仓库盯人，把 GitHub 早期信号提前几周

想比别人更早发现下一个爆发的 Web3/AI 项目，刷 star 榜和 trending 注定慢半拍。**gitalpha** 换了个思路：不盯仓库，盯人——跟踪你信任的一批聪明开发者的公开 GitHub 行为，从中捞出他们悄悄围拢的新仓库，并打出 0-10 分的"早期信号分析卡"。最强的一档信号示例：**5 名 MEV 工程师在 48 小时内 star/fork 同一个创建仅 36 小时的仓库，直接触发 10.0/10 的 Ultra Early 警报**。整个项目**零第三方依赖**，Python 3.11+ 标准库即可运行，自带离线 demo。

## 核心思路：行为比舆情更早

大多数工具"等仓库长出来再看"，等数据积累到能上榜，alpha 已经没了。gitalpha 的假设相反：**真正懂行的人先于市场行动**。它把信息优势从舆情层面（X、Discord）前移到了行为层面（star、fork、commit）——写代码和点 star 不会说谎，比任何宣传都早。

内置的扫描 Agent **Glitch** 走四步 pipeline：

1. **收集**：读取 watchlist 中开发者和组织的公开事件（star、fork、新建仓库、PR 讨论）；
2. **探测**：对疑似仓库运行 8 个信号检测器；
3. **交叉验证**：查域名注册时间、ENS、X 讨论热度，确认"还没被市场发现"；
4. **打分输出**：汇总为 0-10 分，打印终端卡片并写入 Markdown/JSON 报告。

## 八信号打分体系

分数的设计颇有层次——**权重最高的不是 star 数，而是"谁在 star"和"代码里藏着什么"**：

- **Star/Fork Spike（3.0 分）**：同领域 5 名以上被关注工程师 48 小时内 star/fork 同一新仓库 = critical；
- **Whale-Dev Tracker（2.0）**：watchlist 开发者拥有或贡献了该仓库；
- **Hardcoded Contracts（2.0）**：部署/测试文件中出现 EVM 地址或 Solana program ID 旁的 Sepolia、devnet 标记——可能在为发币做准备；
- **Silent Build（1.5）**：被关注者新建无主页、几乎无 star 的仓库；
- **新依赖（1.2）**：解析 package.json、Cargo.toml 等，匹配 zkVM、EIP-7702、MCP 等前沿库；
- **架构质量（1.5）**：测试占比、CI/CD 等启发式评估，可选接入 Claude 评审；
- **Velocity Spike（1.0）** 与 **PR 讨论烈度（0.8）**。

社交行为定方向，代码证据定置信度。此外还有两道外部验证：**RDAP 查域名注册新旧**、**X 查最近 7 天舆情真空**（0 提及 + 强代码 + 大牛开发者会把分数推向顶端）。最终得分封顶 10 分，分档为：Ultra Early ≥ 9 · Early ≥ 7.5 · Heating Up ≥ 6 · On Radar。

## 上手与持续运行

使用门槛几乎为零，三条命令跑起来：

```bash
python -m gitalpha demo      # 离线 demo，无需 token
python -m gitalpha scan      # 真实扫描，需 GITHUB_TOKEN
python -m gitalpha inspect owner/repo
```

所有配置集中在 `config.toml`：watchlist 按 `field` 分组（只有同领域开发者扎堆才触发 critical 警报），也可整组监听某个 org 的全部公开成员。持续化有两条内置路径：**GitHub Actions 每 6 小时定时扫描**并把卡片提交到 `reports/`；**零构建 Dashboard**（`web/index.html` 读 `cards.json`，GitHub Pages 直接托管）。

工程上同样贯彻"零依赖、易审计"：九个单职责小文件（agent/collect/signals/crosscheck/scoring/card 等），信号与打分分离、GitHub 客户端自带缓存限流，调参与维护互不干扰。

## 边界与局限

项目明确划线：只用公开 API 数据、不做人肉关联、遵守限流条款、明示不构成投资建议。局限也很诚实——信号全是启发式规则，误报率取决于 watchlist 的质量。**这个工具真正的壁垒不是代码，而是你"相信谁"**：它提供的是一套可自建的雷达框架，而不是印钞机。

> 本文参考自 [GitHub - Bobersmart/gitalpha](https://github.com/Bobersmart/gitalpha)