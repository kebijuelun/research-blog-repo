# QuickE2E：大白话写 E2E 测试，模型只点按钮，代码判对错

端到端（E2E）测试一直是前端工程里最"脆"的一环：选择器一改就全红，维护成本远超编写成本。QuickE2E 的思路很直接——spec 里只写 **大白话的目标、要输入的值和断言** ，每一步由一个小型决策模型从"合法动作列表"里挑一个动作执行，而 **通过与否完全由代码判定** ，模型永远无权说"测试通过了"。更妙的是，一次通过的运行可以直接导出成一份普通 Playwright spec，在 CI 里回放时不再需要任何模型调用。硬数字方面：在同一个购票任务上，本地引擎 Shisa DE-1 跑出 2.94 秒、5/5 通过、每次运行 $0，对比 Claude Code + Playwright MCP（Sonnet 5）的 25.29 秒、$0.0896—— **快 8.6 倍且零成本** ；托管引擎 Jev 也快 6.7 倍、便宜 234 倍。

![logo.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---dmoka-quicke2e-Plain-English-end-to-end-tests-for-web-apps.-A-small-decision-model-picks-each-click-code-decides-pass-or-fail-a-passing-run-exports-a-Playwright-spec/images/logo.svg)

> 图解：QuickE2E 项目 Logo。项目定位为"用 plain English 写 Web 应用的端到端测试"，MIT 协议开源。

![demo.gif](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---dmoka-quicke2e-Plain-English-end-to-end-tests-for-web-apps.-A-small-decision-model-picks-each-click-code-decides-pass-or-fail-a-passing-run-exports-a-Playwright-spec/images/demo.gif)

> 图解：官方演示（1 倍速实录，本地引擎 Shisa DE-1）。任务是"从 TicketBay 首页出发，用折扣码 WELCOME10 买 2 张票"。可以看到浏览器被自动驱动：找到活动、进入结算页、输入折扣码和姓名邮箱、完成支付，全程没有一行手写选择器。

## 为什么需要它：E2E 测试的两个老问题

传统 Playwright/Selenium 脚本有两个结构性痛点：

- **选择器脆弱** ：`data-testid`、CSS 类名、DOM 层级，前端一重构就批量失效。
- **断言与执行耦合** ：脚本既负责"怎么点"，又负责"对不对"，改一处动全身。

而最近流行的"让大模型 Agent 驱动浏览器做测试"（比如 Claude Code + Playwright MCP）解决了选择器问题，却引入了新问题：通用大模型每步都要做大量推理和工具调用， **慢且贵** ，而且让模型自己判断"任务完成了吗"会产生虚假的 DONE 信号。QuickE2E 恰好卡在两者之间的甜点位上——用最小的模型承担最小的决策职责，剩下的全部交给确定性代码。

## 它是什么：spec 里只有目标、输入和断言

QuickE2E 是一个探索式的端到端浏览器测试器。一份 spec 只包含三样东西：plain English 写的目标（goal）、要输入的值（inputs）、断言（assertion）。看一个真实例子：

```js
// quicke2e.spec.mjs（取自 examples/ticketbay/flows.mjs）
export default [{
  name: "book-with-code",
  start: "/events/midnight-arcade-neon-tour",
  maxSteps: 16,
  inputs: { name: "Alex Fan", email: "fan@example.com", "discount code": "WELCOME10" },
  goal: "Book tickets for this event: continue to checkout, apply the discount code WELCOME10, "
    + "enter the email fan@example.com and the name Alex Fan, and pay.",
  expectUrl: "/orders/\\d+\\?placed=1",                                  // 断言：URL 匹配
  expect: ["Payment confirmed", "WELCOME10 10%", "Total paid €109.39"],  // 断言：页面可见文本
}];
```

这份 spec 的设计原则值得逐条品味：

- **没有选择器** 。spec 里没有任何 CSS/XPath/testid，字段靠 label 匹配（`email` 子串匹配到 "Email" 输入框）。
- **模型不写任何文本** 。引擎只能返回候选动作的 key，每一个被输入的值都来自 spec 的 inputs——模型想"幻觉"出一个值都没有通道。
- **代码判定成败** 。断言检查 URL、用户可见文本和控件状态，全是确定性逻辑。
- **通过的运行可导出** 。`--emit` 把一次 PASS 变成标准 Playwright spec，CI 回放零模型调用。
- **密钥脱敏** 。看起来像密钥的 spec 值会从所有引擎请求、trace、map 和导出文件中被擦除。

笔者认为，这套设计最聪明的地方在于 **职责切分** ：模型只做"下一步点哪个"这种低维离散选择（输出空间可能只有几十个 key），这是小模型完全胜任的任务；而"输入什么值""算不算通过"这类容易出错的事，全部收归代码。这比让一个 500B 参数的通用模型包办一切便宜几个数量级，也更可靠。

## 五分钟上手

环境要求 Node 20+。安装与运行：

```bash
npm i -D quicke2e @playwright/test
npx playwright install chromium
export OPENROUTER_API_KEY=...        # 默认 jev 引擎走 OpenRouter

npx quicke2e check quicke2e.spec.mjs --base http://localhost:3000   # 先体检：拒绝"证明不了任何事"的 spec
npx quicke2e run   quicke2e.spec.mjs --base http://localhost:3000   # 再运行
```

一次通过的运行输出长这样：

```
PASS  book-with-code                8 steps    4.3s  $0.00032  DONE_VERIFIED
```

失败时会逐条打印原因：哪条断言没成立、哪个必填字段为空、是否出现重复动作（LOOP）等。没有现成应用的话，仓库自带三个示例 flow 和 fixture 页面，clone 下来即可体验。

## 工作原理：三段流水线，只有一段调用模型

QuickE2E 由 Discover、Decide、Verify 三部分组成， **只有 Decide 会调用模型** 。下面逐段拆解。

### 1. Discover：先爬一遍，画出应用的"地图"

`quicke2e discover` 用纯 Playwright 爬取应用，产出一份 map：页面（如 `/events/:id`）、页面间的链接和按钮、每个表单的字段与提交去向。这一步完全不调用模型。

值得一提的是安全设计：在 localhost 上 discover 会真实提交表单（完整爬取），所以官方要求指向一次性数据库，并提供 `--reset "<cmd>"` 先恢复种子数据；对非本地主机，完整爬取必须显式传 `--i-own-this-data`；`--safe` 模式则只点链接、菜单和下拉，绝不提交表单。任何模式都不会点击"退出登录"。

### 2. Decide：模型只能从给定选项里挑一个

每一步，页面被转换成一份简短的"合法动作列表"，形如 `TYPE_TEXT 3 Email [textbox]`、`CLICK 7 Pay [button]`。引擎返回其中一个 key 和置信度——它 **无法发明动作、选择器或值** 。返回不在列表中的 key？直接记为 BLOCKED，不执行。

这个环节有三个工程细节值得展开：

- **分组淘汰赛（heats）** ：当页面候选动作多到一次决策装不下时，QuickE2E 把候选分成若干组，引擎并行裁决各组（每组带"以上皆非"选项），再在组冠军之间决赛。没有任何候选被悄悄丢弃。
- **投机执行（speculation） ** ：动作执行后，QuickE2E 会等页面静默 150ms 再决策。用托管引擎时，它会在等待前就基于当前页面发出下一次决策请求；等待结束后再用 settled 页面重新构建请求——**只有当两个请求逐字节相同时才采用提前拿到的答案** ，否则丢弃重问。这样决策结果与不开投机完全一致，只是省了网络往返。丢弃的请求计入成本，可用 `QUICKE2E_SPECULATE=0` 关闭。
- **map 导航** ：传入 `--map` 后，引擎先挑目标需要的页面，代码沿 map 中的链接路线点击过去。一个约束很妙：运行永远不会跳到与起始页同 pattern 的另一个页面——从 `/orders/281` 出发的 spec 就停在 281 号订单上。

### 3. Verify：代码说了算

断言在 **每一个页面快照** 上都被检查，第一个全部成立的快照即判 PASS。六种断言语义如下：

| 断言 | 通过条件 |
| --- | --- |
| expectUrl | URL 匹配给定正则 |
| expect | 真实用户能在页面看到该文本（忽略大小写）；隐藏的、aria-hidden、透明色、被裁剪、缩放到零、屏外的文本都不算；表单控件的值不算（因为那是测试自己输入的） |
| expectState | 某控件处于指定状态（选中的选项、勾选的框、字段的值） |
| expectSeen | 文本自页面加载后任一时刻出现过（比如 toast 提示） |
| expectAbsent | 用户 **看不到** 该文本；一旦出现，运行立即以 ABSENT_SEEN 失败。该文本不得出现在起始页 |
| expectGone | 文本在结束时不可见，但允许出现在起始页（用于删除场景） |

这里有个非常关键的保护机制—— **WEAK_ASSERTION** 。`run` 和 `check` 会先加载起始页并等到没有进行中的 fetch/XHR（最多 3 秒），如果此时断言已经成立，说明"页面自己就能到达断言状态"（比如 SPA 转圈后渲染出了目标文本），这个 spec 什么都没证明，直接拒绝。这堵住了"假绿"最常见的一个洞。

另外，输入值的匹配也很有讲究：spec 的 key 按子串匹配字段 label（`email` → "Email"）；遇到改名后的 label（如 "E-mail"）匹配不上时，引擎会在 **你未使用的 key 里挑一个** 填进去——注意，这仍然是从你的 key 里做选择，值永远只来自 spec。文件上传、纯空格值（用来测必填校验）、无提交按钮的输入框（提供 PRESS_ENTER）、滑块、被 cookie 横幅遮住的按钮（先告诉引擎什么盖住了它，让它先关掉）都有专门处理。

## 从通过到 CI：--emit 生成 Playwright spec

模型驱动的运行有个天然问题：下次可能走不同的路径。QuickE2E 的答案是 `--emit <dir>`——把一次通过的运行固化为 `<dir>/<name>.spec.ts`，一份带相同断言的普通 Playwright spec。

导出质量的关键在于： **每个 locator 都是在运行期间、动作执行之前，在真实页面上构建并验证的** ——必须恰好匹配一个元素，且就是运行实际操作的那个。locator 的候选优先级依次是：Playwright 的 role + accessible name → 限定在所在行/卡片内的 role 定位 → label、placeholder、test id → 位置。实测在 saucedemo.com 上，9/9 导出的 spec 回放通过。

安全细节也没放过：每个输入值从环境变量 `<FLOW>_<KEY>` 读取；非密钥输入有 spec 值兜底， **密钥输入没有兜底** ——宁可让那个测试 skip（且在 CI 下直接判失败），也绝不把凭证写进文件。

## 攻击用例：像攻击者一样思考

攻击测试不是一个单独模式，而是 agent skill 写用例时的固定动作：先写 happy-path，再主动发明边界、拒绝和攻击用例。官方给出的攻击指令要求像"攻击者 + 混沌工程师 + 资深 QA"一样思考，覆盖：

- 真实字段里的恶意值：负数、零、超大数量、超长和 Unicode 字符串、script 标签、SQL 样式字符串、折扣码的怪大小写；
- 折扣码滥用：已用完的码再用、叠加第二个码、过期码；
- 越权访问：start 指向别人的订单 URL，control 是自己的订单；
- 构造 URL：`?qty=-3`、重复参数 `?code=A&code=B`、十六进制 id；
- 已结束的流程通过 URL 重复提交；必填字段留空或填垃圾。

每个攻击用例 **必须断言两件事** ：应用拒绝了（拒绝文案进 `expect`），且成功状态没有出现（确认信息、折扣行进 `expectAbsent`）。为什么两条都要？因为应用完全可以一边显示错误文案、一边照样给你打了折——只断言拒绝文案是不够的。这是个非常老练的测试直觉。

```js
// examples/ticketbay/attacks.mjs 节选：过期折扣码 + 怪大小写
{ name: "attack-disabled-code-odd-casing", kind: "attack",
  start: "/events/midnight-arcade-neon-tour/checkout?qty=2",
  inputs: { "discount code": "launch50" },
  goal: "Apply the discount code launch50.",
  expect: ["This code is no longer active.", "Total €92.70"],  // 拒绝，且总价不变
  expectAbsent: ["% off tickets"] }                             // "码已生效"行绝不出现
```

当然，攻击能力有明确边界：单浏览器单动作，无法测并发双击、竞态、多标签页；不做 header/cookie/请求体篡改等网络层攻击（那些交给 API 属性测试）。crafted URL 属于范围内，且只能攻击自己拥有的应用。

## 密钥与脱敏：值不出门的工程学

让页面内容出本机去请求托管模型，密钥泄露是必须正面回答的问题。QuickE2E 分两层处理：

**强密钥值** （10 位以上且含 2 种以上字符类别，或 10 位以上纯数字）：从所有引擎请求中擦除，包括页面回显的各种变体——重新大小写、截断、重新分词、分组、URL 编码、银行卡"ending 6789"式的掩码。磁盘上的 trace、map、导出 spec 同样擦除。

**弱密钥值 ** （如 `admin`、`letmein1`）：看起来像普通单词，无法靠形态识别，于是改用**历史判定** ——某字符串只在输入该值之后才出现，就是回显，擦除；页面本来就有、或恰好是同词 label（如 "Admin" 链接），是页面自己的文本，保留。

页面内容层面的脱敏由 spec 显式声明：

```js
redact: [".backup-code", "#saved-cards", /recovery code \S+/i]   // CSS 选择器和文本正则
```

选择器覆盖对应元素及其派生的 label/option，正则覆盖裸文本（如页面标题和 URL）。一个很诚实的细节：少于 4 个字母/数字的匹配（如 3 位 CVC）只隐藏元素自身，不做全局文本搜索——因为把页面上所有 "737" 都抠掉会破坏价格和数字。官方还提到他们试过自动"看起来像密钥"规则，结果弄坏了四个正常流程（订单号、版本号、SKU、年份被误伤）还漏掉了别的形态，于是放弃—— **没有规则替你猜哪段文本敏感，必须显式 redact** 。

## 基准测试：数字说话

测试机为 M2 Max，wall time 指整条命令端到端耗时（含浏览器/MCP 启动），Claude 成本取自其自报的 `total_cost_usd`。

### 任务一：从首页购票（即演示 GIF 的任务）

从首页找活动、进结算、用 WELCOME10、填姓名邮箱、支付；每次运行用一条 SQL 校验：新支付订单、折扣已用、总额 €109.39。各 5 次运行，2026-10-06 测量：

| 方案 | 通过 | 中位耗时 | 步数 | 单次成本 |
| --- | --- | --- | --- | --- |
| QuickE2E，本地 Shisa DE-1 | 5/5 | 2.94 s (2.77–2.99) | 7 | $0 |
| QuickE2E，托管 Jev | 5/5 | 3.74 s (3.59–4.74) | 7 | $0.00038 |
| Claude Code + Playwright MCP，Sonnet 5 | 5/5 | 25.29 s (21.8–29.6) | 15–16 次工具调用 | $0.0896 |

换算下来：Shisa DE-1 比 Sonnet 5 快 8.6 倍且零成本；Jev 快 6.7 倍、便宜 234 倍。决策耗时中位数：本地 86ms，托管 300ms——差距主要来自网络往返。

### 任务二：纯结算（2026-09-24）

| 方案 | 通过 | 中位耗时 | 单次平均成本 |
| --- | --- | --- | --- |
| QuickE2E，Jev | 5/5 | 3.05 s | $0.000153 |
| QuickE2E，本地 Shisa DE-1 | 5/5 | 2.04 s | $0 |
| QuickE2E，本地 Eikos-4B | 5/5 | 4.36 s | $0 |
| Claude Code + MCP，Sonnet 5 | 5/5 | 21.67 s | $0.0940 |
| Claude Code + MCP，Opus 5.5 | 5/5 | 25.34 s | $0.1033 |

托管 Jev 比 Sonnet 5 快 7.1 倍、便宜 614 倍。注意官方诚实标注了托管延迟的波动：曾有一批 5 次运行在托管引擎慢时段跑出 11.6s 中位数；本地引擎不走网络，无此问题。

### 八种 UI 技术栈 × 三个任务

覆盖 vanilla HTML、React + MUI/AntD/Radix+shadcn、Vue 3 + Element Plus、Web Components（shadow DOM）、同源 iframe 表单、jQuery 遗留页面；任务是登录、填复合表单、打开 60 行列表的第 57 行：

| 引擎 | 通过 | 中位耗时 | 单次平均成本 |
| --- | --- | --- | --- |
| jev | 120/120（24 格 × 5 次） | 1.8 s | $0.00014 |

### TicketBay 完整 spec 套件与 fixture

| spec | jev | 本地（Eikos-4B / Shisa DE-1） | 步数 | 耗时 | 成本 |
| --- | --- | --- | --- | --- | --- |
| 折扣码购票 | 5/5 | 5/5 / 5/5 | 8 | 4.3 s | $0.00032 |
| 窗口期内退款 | 5/5 | 5/5 / 5/5 | 1 | 0.9 s | $0.00003 |
| 活动开始后退款被拒 | 5/5 | 5/5 / 5/5 | 1 | 0.9 s | $0.00003 |
| 纯结算 | 5/5 | 5/5 / 5/5 | 4 | 2.5 s | $0.00015 |

两个亮点：退款 spec **真抓到了一个预埋 bug** （删掉退款窗口校验的 TicketBay 副本会 5/5 失败，它在活动开始后仍退了 €39.69）；导出的 Playwright spec 3/3 回放通过，每次约 0.8 秒且零模型调用。74 个 fixture 页面（全部来自真实缺陷和安全审计）上，jev 59/59、Eikos-4B 58/59、Shisa DE-1 56/59（早期 59 页套件），强密钥零泄露。

## 引擎选择：本地 $0 的诱惑

| | Eikos-4B（默认） | Shisa DE-1（`--model shisa-de-1`） | Jev（托管，默认引擎） |
| --- | --- | --- | --- |
| 本质 | Qwen3.5-4B 决策微调 | Gemma 4 26B MoE，每 token 激活 3.8B | TypeSafe 的托管模型 |
| 峰值内存 | 4.1 GB | 17.4 GB | 本地无占用 |
| 硬件要求 | 任意 8GB+ Apple Silicon Mac | 32GB Mac + llama.cpp | 任意 OS |
| TicketBay 四 spec | 20/20 | 20/20 | 20/20 |
| 纯结算端到端 | 4.4 s | 2.0 s | 3.05 s |

本地引擎的关键技术点：这些开放决策模型 **只读候选标签的 logits，不生成任何文本** ——输出空间被结构性地锁死在合法 key 集合内。Shisa DE-1 能跑赢托管 Jev，靠的是 MoE 每 token 只激活 3.8B 参数 + 零网络开销。限制是仅支持 Apple Silicon。

## 配套 agent skill：大模型出题，小模型答题

架构上的最后一环：`skill/quicke2e/SKILL.md` 是一个给 Claude Code 等 agent 用的技能。大模型只跑一次——读 map 和源码、列出业务规则（rule → file:line）、发明 happy-path/边界/拒绝/攻击用例、写 spec 并运行 check/run、报告哪些失败是真 bug。之后每一次运行的每一步决策都由小决策引擎完成，pass/fail 由代码判定。 **"大模型写一次，小模型跑一万次"** ——这个成本结构才是这套方案能便宜 234 倍的根本原因。

## 局限：官方自己列的"不能做什么"

篇幅所限挑重点（完整清单长达二十余条，这种坦诚本身值得加分）：

- 绿色运行只证明 spec 里的断言成立，别过度解读；CI 门禁请用导出的 Playwright spec，模型驱动的运行下次可能走不同路径。
- 不支持：canvas 应用、closed shadow roots、跨域 iframe（如 Stripe Elements）、跨标签页流程、浏览器后退、原生移动端。
- 无法作为候选的元素：只有 JS click handler 没有 role 的元素、hover 才出现的按钮、dnd-kit 拖拽、单值多格 OTP。
- 没有 SCROLL 动作：无限滚动和虚拟列表只能看到已渲染的行；超过 40 个同名按钮时行上下文会被丢弃。
- 完整爬取会改数据（它真的提交每个表单），务必用一次性数据库；`--safe` 是尽力而为。
- 本地引擎在 fixture 套件上弱于 Jev（71/74 vs 74/74），Eikos-4B 也比 Jev 慢。

## 结语

回顾 QuickE2E 的核心要点：

- **职责切分是灵魂** ：小模型只做"从合法动作里挑一个"的离散决策，值来自 spec，成败由代码判定，幻觉没有通道；
- **性能成本双杀** ：本地引擎 2.94 秒、$0 完成购票任务，比 Claude Code + MCP 快 8.6 倍；托管方案便宜 234 倍；
- **从探索到回归的闭环** ：`--emit` 把通过的运行固化为无模型的 Playwright spec，直接进 CI 当门禁；
- **测试直觉内置** ：WEAK_ASSERTION 防"假绿"，攻击用例强制"拒绝 + 无成功痕迹"双断言；
- **安全不打折** ：强密钥全链路擦除，页面敏感内容靠显式 redact。

展望来看，QuickE2E 验证了"专项小模型 + 确定性代码围栏"这条路线在 UI 自动化上的可行性——当决策空间被结构性地约束后，4B 量级模型足以胜任此前被认为需要旗舰大模型的工作。它当前对 canvas、跨域 iframe 等场景的盲区，以及本地引擎对 Apple Silicon 的依赖，是下一步值得关注的演进方向。

## 原文图表补充

![68747470733a2f2f696d672e736869656c64732e696f2f6e706d2f762f717569636b653265.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---dmoka-quicke2e-Plain-English-end-to-end-tests-for-web-apps.-A-small-decision-model-picks-each-click-code-decides-pass-or-fail-a-passing-run-exports-a-Playwright-spec/images/68747470733a2f2f696d672e736869656c64732e696f2f6e706d2f762f717569636b653265.svg)

![68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c7565.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---dmoka-quicke2e-Plain-English-end-to-end-tests-for-web-apps.-A-small-decision-model-picks-each-click-code-decides-pass-or-fail-a-passing-run-exports-a-Playwright-spec/images/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c7565.svg)

![badge.svg](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/GitHub---dmoka-quicke2e-Plain-English-end-to-end-tests-for-web-apps.-A-small-decision-model-picks-each-click-code-decides-pass-or-fail-a-passing-run-exports-a-Playwright-spec/images/badge.svg)

For AI agents: AGENTS.md (spec rules, exit codes, the --json record) · llms.txt · agent skill

| launch task, 5 runs per arm | median wall time | pass | cost / run |
| --- | --- | --- | --- |
| QuickE2E, local engine, Shisa DE-1 | 2.94 s | 5/5 | $0 |
| QuickE2E, hosted Jev | 3.74 s | 5/5 | $0.00038 |
| Claude Code + Playwright MCP, Sonnet 5 | 25.29 s (21.8–29.6) | 5/5 | $0.0896 |

| engine | pass |
| --- | --- |
| jev | 59/59 |
| local , Eikos-4B | 58/59 |
| local , Shisa DE-1 | 56/59 |

|  | Eikos-4B (default) | Shisa DE-1 ( --model shisa-de-1 ) | Jev (hosted) |
| --- | --- | --- | --- |
| what it is | Qwen3.5-4B fine-tuned for decisions | Gemma 4 26B MoE, 3.8B active per token | TypeSafe's hosted model |
| peak memory | 4.1 GB | 17.4 GB | none locally |
| Mac | any Apple Silicon Mac with 8 GB or more | 32 GB, and brew install llama.cpp | any OS |
| TicketBay (4 specs × 5) | 20/20 | 20/20 | 20/20 |
| plain checkout, end to end | 4.4 s | 2.0 s | 3.05 s |
| fixture suite (59 pages, n=3) | 58/59 | 56/59 | 59/59 |

| flag | command | effect |
| --- | --- | --- |
| --start | discover | comma-separated start paths (default / ) |
| --safe | discover | safe crawl: follows links, opens menus, dialogs and dropdowns, and never submits a form (see Limits ) |
| --i-own-this-data | discover | allow a full crawl on a host other than localhost |
| --reset "<cmd>" | discover | command that restores seed data before the crawl |
| --inputs | discover | JSON file with values for forms during the crawl |
| --storage | discover | Playwright storage-state file |
| --max-pages | discover | page limit for the crawl (default 40) |
| -o | discover | output map file (default quicke2e.map.json ) |
| --redact | discover | CSS selector or /regex/ ; repeat the flag for more |
| --base | run, check | app base URL (default $APP_BASE , then http://localhost:3000 ) |
| --engine | run | jev (default), local or vercel |
| --map | run | map file from discover |
| --runs | run | runs per spec (default 1) |
| --only | run, check | run only the spec with this name |
| --emit | run | write a Playwright spec for each passing run |
| --trace | run | write a JSON trace per run, with each step's start time and decision time |
| --video | run | save a WebM per run. With --trace , each step in the trace also gets the box of the element it acted on. The video shows typed values |
| --allow-weak | run | run a spec that failed the WEAK_ASSERTION check |
| --json | run, check | print all records as one JSON array on the last line of stdout, including flows that never ran ( UNREACHABLE , WEAK_ASSERTION ); fields: AGENTS.md |
| --min-confidence | run | below this engine confidence an action is not executed (default 0.3) |
| --nav-timeout | run | page-load timeout in ms (default 30000) |
| --reset | run | shell command run before every run, to restore seed data (a failing command stops that flow with RESET_FAILED ) |
| --headed / --headless | all | force the browser mode |

| environment variable | effect |
| --- | --- |
| OPENROUTER_API_KEY | key for the jev engine |
| AI_GATEWAY_API_KEY | key for the vercel engine |
| LOCAL_URL | URL of the local engine server (default http://127.0.0.1:8822 ) |
| APP_BASE | default --base |
| QUICKE2E_SPECULATE=0 | turn off early decisions (see Decide ) |
| QUICKE2E_ENGINE_TIMEOUT_MS | per-request engine timeout (default 30000); a timed-out request is retried up to 4 times |
| QUICKE2E_PROF=1 | add per-step timings (settle, snapshot, decide, act) to each run record |
| JEV_DEBUG=1 | print the page state and the options of every decision to stderr |

| field | meaning |
| --- | --- |
| name | flow name, used in output, --only and emitted file names |
| start | start path (default / ) |
| base | base URL for this flow; overrides --base |
| goal | the task in plain English |
| inputs | every value the run types, keyed by field label |
| expectUrl , expect , expectState , expectSeen , expectAbsent , expectGone | the assertions (see Verify ). expectUrl is a regex: escape ? and . ( "/login\\?welcome=1" ) |
| control | a path where the app says yes; the WEAK_ASSERTION check loads it instead of start (see Attack cases ) |
| kind | a label for the run record and the output line, such as "attack" |
| redact | CSS selectors and text patterns the engine must never see |
| storageState | Playwright storage state for the browser context |
| maxSteps | step limit (default 14) |
| neverClick | elements the engine is never offered: case-insensitive globs over the whole label ( "Pay*" , "*delete*" ) or RegExps. Use it in refusal tests so a failed refusal cannot buy, pay or delete |
| minConfidence | a click the engine picks below this confidence is not executed and counts as BLOCKED (default 0.3, or --min-confidence ). Typing a spec value or choosing the option the spec names is not gated |
| dialog | "accept" (default) or "dismiss" for native confirm / alert / prompt dialogs; a prompt gets the spec value whose key its message names |
| navTimeout | page-load timeout in ms (default 30000, or --nav-timeout ) |
| maxTimeMs | time budget for the run; when it runs out, the outcome is TIMEOUT |
| done | optional plain-English end state, for the reader. The run loop does not send it to the engine |

| outcome | meaning |
| --- | --- |
| DONE_VERIFIED | the assertions held on a snapshot |
| ABSENT_SEEN | an expectAbsent text appeared: the app accepted what it must refuse |
| SERVER_ERROR | a page navigation answered with HTTP 5xx. The run fails, whatever the assertions say |
| MODEL_BLOCKED | the engine answered BLOCKED (or a key that was not offered) three times, each time on a page that did not change within 3 s |
| NO_SPEC_VALUE | the run needed a value the spec does not have |
| MAX_STEPS | the step limit ran out |
| LOOP | the same action ran 3 times on a page that did not change; the 4th was not executed |
| AUTH_REQUIRED | the flow has a storageState , but the start page redirected to a sign-in page: the session is missing or expired |
| TIMEOUT | the flow's maxTimeMs ran out |
| WEAK_ASSERTION | the assertions held before the run executed any action (or, before the run, on the start page) |
| NOT_REFUSED | a control flow (a load attack): the start page did not show the refusal. The run never acts on a load attack |
| ENGINE_ERROR | the engine failed (no answer in 30 s after 4 tries, HTTP 429/5xx, a rejected key). Not an app failure. A rejected key (401/403) stops the whole run with exit code 2 |
| RESET_FAILED | the --reset command exited with an error; the flow did not run |
| ERROR | the run threw an error |

| engine | key |
| --- | --- |
| jev | OPENROUTER_API_KEY |
| vercel | AI_GATEWAY_API_KEY |
| local | none |

> 本文参考自 [GitHub - dmoka/quicke2e: Plain-English end-to-end tests for web apps](http://github.com/dmoka/quicke2e)