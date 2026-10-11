# Talorys：一条命令，把个人 AI 助手装进你自己的 Cloudflare 账号

用 ChatGPT、Claude 时，你的对话、偏好、待办都存在厂商服务器上，数据归谁你说了不算。Talorys 给出了另一个答案：把聊天、记忆、任务、笔记、定时提醒整套个人 AI 助手，全部部署进**你自己的** Cloudflare 账号——只需一条命令 `npx create-talorys@latest`，全程跑在免费套餐内，零遥测、单用户、MIT 开源。

## 核心创新：一个人、一个账号、一条命令

- **功能五件套**：流式聊天（Markdown 渲染 + 工具调用过程可视化）、长期记忆（可查看/编辑/删除，只送最相关记忆进模型）、任务/笔记/项目管理、自动化提醒、以及"无 AI 也能活"——AI 额度耗尽时基础功能照常运转。
- **提醒不靠在线设备**：定时任务跑在 Durable Object 的 Alarm 机制上，无需任何设备保持开机。
- **"Single-user by design"**：把单用户当特性而非妥协——没有注册、团队、权限矩阵，复杂度大砍，安全性反而更好做。

## 关键架构：攻击面极小

浏览器只跟你的 Pages 站点通信：`Pages（React 前端）→ Service Binding（无公网地址的私有 Worker）→ 单个 SQLite Durable Object（存全部数据）→ Workers AI（glm-4.7-flash 模型）`。

- 后端 Worker 设了 `workers_dev: false`，**没有任何公开 URL**，唯一入口是你的 `*.pages.dev`。
- 鉴权在后端不在前端；聊天是端到端 SSE 流式推送。
- 刻意不用 R2、D1、KV、Vectorize 等任何可能产生费用的服务——宁可功能简单，也守住"免费套餐友好"。

## 部署与隐私：细节见功力

- **幂等安装器**：中途断了原地重跑即可，自动盘点已有资源、不重复创建、绝不删数据；密码经 PBKDF2-SHA256 哈希后只存为 Cloudflare Secret，明文不落盘；部署后自动验证线上健康状态（不消耗 AI 额度）。
- **运维有答案**：`reset-password`、`update`、`status`、`doctor` 子命令齐全；备份导出 JSON（永不导出会话凭据），重复导入无副作用。
- **隐私实话实说**：零遥测、零追踪，数据只在你账号下的一个 SQLite 里——但信任链收敛到 Cloudflare 一家，Workers AI 推理仍会处理你的聊天内容，这一点作者没有回避。

## 结论

Talorys 的局限明显：绑定 Cloudflare、无法协作共享、模型选择有限。但它示范了一个有意思的方向——**个人 AI 助手的"自托管"可以轻到什么程度**：当别人还在讨论本地跑模型需要多大显存时，它用一条 npx 命令把整套系统塞进了云厂商的免费套餐。

> 本文参考自 [GitHub - rociiu/talorys: Your personal AI agent in your own Cloudflare account](https://github.com/rociiu/talorys)