# Talorys：一条命令把个人 AI 助手装进自己的 Cloudflare 账号

用 ChatGPT、Claude 这类 AI 助手时，你的对话、偏好、待办事项都存在厂商的服务器上，数据归谁、被拿去干什么，你说了不算。Talorys 给出了另一种答案：把整套个人 AI 助手——聊天、记忆、任务、笔记、定时提醒——全部部署进 **你自己的** Cloudflare 账号，不需要任何第三方服务器或数据库。部署只需要一条命令 `npx create-talorys@latest`，全程跑在 Cloudflare 免费套餐内，零遥测、单用户、MIT 开源。

简单说，它的核心思路是： **一个人、一个 Cloudflare 账号、一条命令、一个专属 AI 助手** 。你的数据从产生到存储，始终躺在自己账号下的一个 SQLite 数据库里。

## 它能做什么：不只是聊天框

Talorys 的定位是"私人 AI 管家"，功能可以拆成五块：

- **流式聊天** ：支持 Markdown 渲染和工具调用过程可视化，能看到 AI 正在调用哪个工具。
- **长期记忆** ：记住你的个人事实和偏好，可随时查看、编辑、删除。每轮对话只把最相关的记忆送进模型，而不是无脑全塞。
- **任务、笔记与项目管理** ：界面上完整 CRUD，也可以直接在聊天里下指令，比如说"帮我加一条明天评审项目的任务"。
- **自动化提醒** ：一次性或周期提醒、每日任务摘要、可选的 AI 例行任务，全部投递到应用内通知中心。关键是它们跑在 Durable Object 的 Alarm 机制上—— **不需要任何设备保持在线** 。
- **无 AI 也能活** ：Workers AI 挂了或者免费额度用完了，任务、笔记、记忆、提醒这些基础功能照常运转。

先看一下实际界面：

![聊天界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---rociiu-talorys-Your-personal-AI-agent-in-your-own-Cloudflare-account.-Chat-memory-tasks-notes-and-scheduled-reminders-deployed-with-one-command-npx-create-talorys-latest.-Free-tier-friendly-single-user-no-telemetry/images/chat.png)

> 图解：聊天主界面。左侧是对话流，可以看到 AI 回复支持 Markdown 渲染，并且有"工具活动指示器"——当 AI 调用内部工具（比如帮你建任务）时，过程是可见的，而不是一个黑盒。

![任务界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---rociiu-talorys-Your-personal-AI-agent-in-your-own-Cloudflare-account.-Chat-memory-tasks-notes-and-scheduled-reminders-deployed-with-one-command-npx-create-talorys-latest.-Free-tier-friendly-single-user-no-telemetry/images/tasks.png)

> 图解：任务管理界面。任务支持完整的增删改查，这部分不依赖 AI，是纯本地数据操作。

![记忆界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---rociiu-talorys-Your-personal-AI-agent-in-your-own-Cloudflare-account.-Chat-memory-tasks-notes-and-scheduled-reminders-deployed-with-one-command-npx-create-talorys-latest.-Free-tier-friendly-single-user-no-telemetry/images/memory.png)

> 图解：记忆管理界面（暗色模式）。这里存放的是 AI 记住的关于你的持久化事实和偏好，全部可查看、可编辑、可删除——记忆的主动权在用户手里，而不是模型手里。

![自动化界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---rociiu-talorys-Your-personal-AI-agent-in-your-own-Cloudflare-account.-Chat-memory-tasks-notes-and-scheduled-reminders-deployed-with-one-command-npx-create-talorys-latest.-Free-tier-friendly-single-user-no-telemetry/images/automations.png)

> 图解：自动化配置界面。可以设置一次性/周期性提醒、每日任务摘要（digest）和 AI 例行任务，全部由 Durable Object Alarm 驱动。

![移动端界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---rociiu-talorys-Your-personal-AI-agent-in-your-own-Cloudflare-account.-Chat-memory-tasks-notes-and-scheduled-reminders-deployed-with-one-command-npx-create-talorys-latest.-Free-tier-friendly-single-user-no-telemetry/images/mobile.png)

> 图解：移动端界面。前端是响应式的 React 应用，手机上也能正常聊天和管理任务。

![登录界面](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---rociiu-talorys-Your-personal-AI-agent-in-your-own-Cloudflare-account.-Chat-memory-tasks-notes-and-scheduled-reminders-deployed-with-one-command-npx-create-talorys-latest.-Free-tier-friendly-single-user-no-telemetry/images/login.png)

> 图解：登录界面。整个系统只有一个用户——你自己。没有注册流程、没有账号体系，安装时设置的主人密码就是唯一的入口。

笔者认为这里最值得玩味的设计是 **"Single-user by design"** 。大多数 SaaS 产品的多租户架构是成本所迫，而 Talorys 反过来把"单用户"当成特性：没有注册、没有团队、没有权限矩阵，复杂度直接砍掉一大截，安全性反而更好做。

## 架构：浏览器永远只跟你的 Pages 站点说话

了解了功能之后，自然的问题是：没有服务器，这套东西是怎么跑起来的？答案是 Cloudflare 的全家桶——Pages 托管前端，Workers 跑后端，Durable Object 存数据，Workers AI 出模型。

```text
Browser ──HTTPS──▶ Cloudflare Pages (React app + /api Pages Function)
                         │  service binding (no public URL)
                         ▼
                   Private Worker (Hono router)
                         │  getAgentByName("personal-agent")
                         ▼
                   TalorysAgent — Cloudflare Agents SDK Durable Object
                     ├─ SQLite: conversations, memories, tasks, notes, projects,
                     │          automations, sessions, settings, usage
                     ├─ Workers AI: @cf/zai-org/glm-4.7-flash (streaming + tool calling)
                     └─ Alarms: reminders and recurring schedules
```

这张架构图里有几个设计点值得细品：

- **后端 Worker 没有公网地址** 。Agent Worker 部署时设置了 `workers_dev: false` 和 `preview_urls: false`，它不存在任何公开 URL。外部流量唯一的入口是你的 `*.pages.dev` 站点，`/api/*` 请求由 Pages Function 通过 **Service Binding** （Cloudflare 内部的私有网络通道）转发给 Worker。
- **鉴权在后端，不在前端** 。登录校验、权限判断全部发生在 Worker 内部，前端只是个展示层。
- **聊天是端到端的 Server-Sent Events 流式推送** ，从 Workers AI 的流式输出一路转发到浏览器。
- **所有数据集中在一个 SQLite-backed Durable Object** （名为 `personal-agent`）里：对话、记忆、任务、笔记、项目、自动化、会话、设置、用量，一个对象全包了。

这个架构的聪明之处在于攻击面极小：没有公网 API、没有独立数据库、没有第三方服务，整个系统就是 Cloudflare 账号里的四个组件，而它们全部在免费套餐覆盖范围内：

| 服务 | 用途 | 免费套餐可用 |
| --- | --- | --- |
| Cloudflare Pages | 前端 + Pages Function | 是 |
| Cloudflare Workers | 私有 API Worker | 是 |
| Durable Objects (SQLite) | 全部数据 + 定时 Alarm | 是 |
| Workers AI | 聊天模型（`@cf/zai-org/glm-4.7-flash`） | 是，有每日额度 |

注意 Talorys **刻意不用** R2、D1、KV、Vectorize、AI Search、Workflows 等任何可能产生费用的服务。这是一个非常克制的工程决策——宁可功能简单一点，也要守住"免费套餐友好"这条线。

## 部署：一条命令背后的幂等安装器

架构搞清楚了，部署反而成了最简单的部分：

```bash
npx create-talorys@latest
```

这条命令背后的安装器做的事情并不少：

1. 检查 Node.js 版本（要求 20.18+），使用内置的 Wrangler CLI，不依赖你全局装的 wrangler。
2. 验证 Cloudflare 登录状态，没登录就自动打开授权页面（走 Wrangler 标准 OAuth 流程， **不需要 Global API Key** ）。
3. 让你选择账号（有多个账号时）和 Agent 名称。
4. 要求设置主人密码：隐藏输入、强度检查，本地用 **PBKDF2-SHA256** 哈希后，只以 Cloudflare Secret 的形式存储——明文密码不落盘。
5. 生成 256 位会话密钥和唯一的资源名（`talorys-<id>-agent`、`talorys-<id>-web`）。
6. 部署私有 Worker（创建 SQLite Durable Object）、写入密钥、创建并部署带 Service Binding 的 Pages 项目。
7. **自动验证线上部署** ：检查前端、鉴权接口、未授权访问是否被拒绝、存储健康状态，全程不触发任何 AI 推理（不浪费额度）。
8. 打印 Cloudflare 返回的真实 `https://….pages.dev` 地址。

安装器会在本地写一个 `talorys/` 目录，包含 `talorys.json`（安装 ID、账号 ID、资源名、URL—— **不含任何密钥** ）和部署产物，后续更新要用到它，别删。

笔者认为安装器最贴心的设计是 **幂等恢复** ：中途断了？原地再跑一遍同样的命令，它会自动盘点已有资源——不重复创建、已设过密码就不再追问、绝不删除任何数据。对于不熟悉 Cloudflare 的用户来说，这几乎消除了部署焦虑。

如果你习惯 CI/CD 式的非交互安装，也支持：

```bash
TALORYS_OWNER_PASSWORD=... npx create-talorys@latest --yes --account-id <id>
```

（未登录时配合 `CLOUDFLARE_API_TOKEN` 使用。API Token 需要的最小权限为：Workers Scripts › Edit、Cloudflare Pages › Edit、Workers AI › Read、Account Settings › Read。）

## 访问与运维：忘记密码、备份、更新都有答案

部署完，日常使用是什么样的？

**登录与访问** ：打开安装器打印的 URL，输入主人密码即可。可以从任意多台设备登录——它们都是同一个主人。会话管理在 Settings → Security 里。

**忘记密码** ：在 `talorys/` 目录下执行：

```bash
npx create-talorys@latest reset-password
```

它会替换密码密钥，并把所有设备强制下线。

**备份数据** ：Settings → Privacy → Download backup 会导出 `talorys-backup.json`，包含对话、记忆、任务、笔记、项目、设置和自动化（ **永远不导出会话和凭据** ）。导入时会校验文件并在单个事务中合并，同一个备份重复导入也不会产生副作用。

**升级版本** ：

```bash
npx create-talorys@latest update
```

重新部署最新的 Worker 和前端到原有资源上。Durable Object 命名空间永远不会被重建（迁移是 append-only 的），密码和会话保留，数据库 schema 迁移在首次请求时自动、事务化地完成。另外还有 `status` 和 `doctor` 子命令可以做健康检查。

## 数据在哪：隐私模型的实话实说

讲到这里，是时候正面回答隐私问题了——这也是 Talorys 存在的意义。

- 所有数据存在 **你 Cloudflare 账号下** 的单个 SQLite Durable Object 里。
- Talorys 本身 **没有任何遥测、分析、追踪、广告代码** ，不给开发者发任何东西。
- 但也要诚实：Cloudflare 作为服务提供方会处理你的数据——包括 Workers AI 对你聊天内容和相关记忆做推理。所以使用前请阅读 Cloudflare 的隐私政策和 Workers AI 条款。

换句话说，Talorys 把信任链从"AI 创业公司 + 一堆第三方服务"收敛到了"只有 Cloudflare 一家"。这个取舍是否值得，取决于你对 Cloudflare 的信任程度，但透明度的确拉满了。

## 费用：免费但不等于"无限免费"

Talorys 设计上完全适配 Cloudflare Workers 免费套餐，也绝不主动开启付费功能。但有几条边界要说清楚：

- 免费套餐有账号级配额：请求数、Durable Object 用量、每日 Workers AI Neuron 额度，这些由 Cloudflare 制定且可能调整。
- AI 额度用完后，聊天会明确提示，并在每日重置后恢复； **其他一切功能不受影响，数据完好无损** 。
- 如果你的账号本身是付费套餐，超出部分由 Cloudflare 按你的套餐计费，与 Talorys 无关。

系统内置了可调节的护栏（Settings → AI）：最大输出 token 数、最大上下文 token 数（超出的旧历史会被自动摘要）、单次请求最大工具调用和推理步数、每日 AI 请求上限、每日定时 AI 任务上限。简单的提醒和任务摘要 **完全不消耗 AI 额度** 。Usage 面板提供本地用量估算，并链接到 Cloudflare 控制台查看精确的 Neuron 用量。

## 常见问题速查

| 问题 | 解决办法 |
| --- | --- |
| 安装器中途停了 | 原地重跑同一命令，自动续装 |
| 提示 "did not pass its health checks yet" | 新 pages.dev 站点需要几分钟生效，稍后重跑 |
| 权限报错 | 换用有 Workers/Pages 编辑权限的账号，或配好上述权限的 API Token |
| 聊天提示 AI 额度用完 | 等每日重置，或在 Settings → AI 里开启 Demo 模式 |
| 多次登录失败被锁定 | 等 15 分钟，或用 CLI 重置密码 |

`npx create-talorys@latest doctor` 可以一键自动检查所有环节。

## 想自己折腾代码：本地开发与仓库结构

最后，如果你不只是用户，还想改代码。本地开发只需要 Node.js 20.18+：

```bash
git clone https://github.com/rociiu/talorys.git
cd talorys
npm install
npm run dev
```

`npm run dev` 会在一个终端里同时拉起三个进程：Agent Worker（`wrangler dev`，端口 8787）、带 Service Binding 的 Pages Function 代理（端口 8788）、Vite 前端（`http://localhost:5173`）。本地开发使用确定性的 mock AI provider， **不需要登录 Cloudflare** ，状态持久化在 `.wrangler/state`。

仓库是标准的 monorepo：

```text
apps/agent              私有 Worker：Hono API、TalorysAgent、工具、SQLite 仓储
apps/web                React + Vite 前端，Pages Function (functions/api/[[path]].ts)
packages/shared         Zod schema、类型、周期/DST 逻辑、密码哈希
packages/create-talorys 一键安装器（npm 包）
tests/e2e               Playwright UI 测试
tests/smoke             可选的真实账号部署测试
docs/                   架构、部署、开发、安全、排错文档
```

配套脚本齐全：`npm run build`、`npm test`、`npm run test:e2e`、`npm run test:pack`、`npm run typecheck`、`npm run lint`。项目采用 MIT 协议开源。

## 总结

- Talorys 是一个完全跑在 **你自己 Cloudflare 账号里** 的开源个人 AI 助手：聊天、记忆、任务、笔记、定时提醒一体。
- 架构极简：Pages 前端 + 无公网地址的私有 Worker + 单个 SQLite Durable Object 存所有数据 + Workers AI 出模型，攻击面小、零第三方依赖。
- 一条命令 `npx create-talorys@latest` 完成部署，安装器幂等可恢复，密码 PBKDF2 哈希后仅存为 Cloudflare Secret。
- 全程适配免费套餐，AI 额度耗尽时非 AI 功能照常工作，内置多项用量护栏。
- 零遥测、单用户设计，数据导出备份、密码重置、版本升级都有对应命令。

Talorys 的局限也很明显：它绑定 Cloudflare 一家，单用户设计意味着无法协作共享，Workers AI 的模型选择也不如直连大模型 API 丰富。但它示范了一个有意思的方向—— **个人 AI 助手的"自托管"可以做到多轻** ：当别人还在讨论本地跑模型需要多大显存时，它用一条 npx 命令把整套系统塞进了云厂商的免费套餐里。

> 本文参考自 [GitHub - rociiu/talorys: Your personal AI agent in your own Cloudflare account](https://github.com/rociiu/talorys)