# big-arrow-on-the-screen：给 AI Agent 一只"手指"

AI Agent 能重构整个 monorepo，却栽在最后一公里——当它需要你点一个"允许"按钮时，只能往你没在看的终端打印一行字，然后干等。macOS 小工具 **bigarrow** 用一条命令把大箭头和提示牌画在屏幕最上层，替 Agent 伸出手指：纯 Swift 单二进制、绘图零权限、点击穿透、箭头自动消失，上线两天收获近 400 Star，Star 列表里不乏 Apple、NVIDIA 工程师。

## 痛点：Agent 没有手指

- 权限弹窗、OAuth 同意页、2FA 验证码、"用哪个应用打开"——这些必须人来决定的操作会让 Agent 卡死。
- 作者调研了 26 款屏幕标注工具，没有一款满足"程序调用、带退出码、指完自己消失"，bigarrow 填的正是这个空。

## 方案：一支会自动消失的箭头

- **透明层置顶**：在所有显示器、所有 Space 上放一个屏保级层级的透明窗口，高于对话框和全屏应用，但**永不抢焦点**。
- **点击穿透**：点在目标处直接穿透给下面的应用；只有点到箭身或提示牌才会移除箭头。
- **三条核心命令**：`point`（指一下就撤，默认 8 秒）、`start`/`stop`（持续指，直到喊停）、`elements`/`doctor`（列出可定位元素、诊断权限）。
- **配套 Skill**：`bigarrow install-skill` 一键装进 Claude Code 和 Codex；平时只占 182 token，完整说明（1,398 token）按需加载。
- **贴心设计**：提示牌里的 `{{value}}` 自动变成复制按钮，验证码、命令一键进剪贴板；`--say` 能把提示朗读出来，把你从咖啡机前叫回来。

## 最聪明的地方：自我设限

永不点击、永不输入、永不截图，只指。无 daemon、无账号、无遥测，作者调侃"它是你 AI 技术栈里智力最低的部分，并且引以为豪"。在 Agent 权限越来越大的时代，刻意收窄能力边界反而让它可信。

生命周期同样完整，五种"善终"机制保证箭头必自己消失：时间限制、主动 `stop`、随 Agent 进程退出、Claude Code 的 `stop --hook`（用户一提交消息就清场）、可选的关闭按钮。

## 验证：每条承诺都有测试兜底

- **104 个自动化测试**：覆盖几何计算、黄金图像比对、点击穿透、"最前台应用永不改变"等，CI 在 macOS 15 上跑，另在 26、27 验证通过。
- **18 项行为检查**：从 `--follow`、`--until-click` 到"画到一半拔掉显示器"，脉动箭头 CPU 占用实测仅 1.4%。
- **最有说服力的验收**：给一个全新 Agent 只发 Skill 文件和一句"指出 Chrome 的 Reload 按钮"，它零示范一次用对——过程中还顺手发现了一个 bug，后来这个 bug 变成了测试。

## 结语

bigarrow 只解决"指"这一件事，但它点出的方向很清晰：当 Agent 开始真正操作你的电脑，"怎么把控制权礼貌地交还给人"会成为 Agent 工具链竞争中被低估的一环。

> 本文参考自 [GitHub - franzenzenhofer/big-arrow-on-the-screen: Let your AI agents paint big arrows, boxes and text on your Mac screen](https://github.com/franzenzenhofer/big-arrow-on-the-screen)