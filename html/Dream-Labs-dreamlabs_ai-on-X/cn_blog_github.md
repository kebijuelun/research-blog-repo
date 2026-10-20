# Dream Labs 完成 9000 万美元种子轮：12 人团队想把机器人模型做成 API

一条只有两句话的推文，给出了两个值得记住的数字：机器人公司 Dream Labs 宣布完成 9000 万美元种子轮融资，而团队只有 12 名研究员和工程师。对于种子轮而言，9000 万美元是一个罕见的大额数字，这意味着投资人在团队尚未公开产品细节的情况下，就为其押下了重注。

本轮由 B Capital 领投，参与方包括 Kindred Ventures、A.Capital、Unusual VC、SV Angel、Wing VC、Scribble VC、Adrian Aoun，天使投资人名单中还有 Thom_Wolf、stephenbalaban 等人。从名单看，这一轮集结了多家早期基金和 AI 领域的个人投资者，阵容更像一笔成长期融资的配置。

![Dream Labs 融资公告图](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/Dream-Labs-dreamlabs_ai-on-X/images/HVAnWAVbkAAwbYW.webp)

> 图解：公告图左侧写着“Dream Labs raised \$90M”“seed funding to enable Robotics within Reach”，下方列出 B Capital、Kindred Ventures、a_capital、Unusual、SV Angel、Wing、Scribble Ventures、Adrian Aoun 等投资方标识。右侧是一段名为 `dream-1 – python client.py` 的代码示意，展示了一个名为 dream 的 Python 客户端的用法。

这条推文本身几乎没有介绍产品，但配图右侧的代码片段透露了 Dream Labs 的技术方向。示例代码大致是这样的流程：先用 API key 创建 `DreamClient`，再通过 `client.policy({"checkpoint": "dreamlabs/dream-1"})` 加载一个名为 dream-1 的模型检查点，然后循环执行 `set_observation`（传入观测）、`get_action`（获取动作）、`reset_episode`（重置回合）。窗口底部还有一行“Tell your robot what to do”的输入提示，以及一次“Thought for 200ms”的运行反馈。

这个结构对机器人领域并不陌生：观测进、动作出，是典型的机器人策略（policy）接口。值得注意的是它的形态——模型被当作一个远程服务，通过 checkpoint 名称加载，几行代码就能驱动机器人，这与语言模型 API 的使用方式几乎一致。结合“Robotics within Reach”这句标语，可以推断 Dream Labs 的目标是把通用机器人模型做成一项托管服务，让机器人开发者不必自己训练和部署策略模型。当然，这只是基于一张公告图的推断，公司尚未公布 dream-1 的能力指标、适用的机器人形态或任何基准测试结果。

回到开头的两个数字：12 人团队加上 9000 万美元种子轮，说明投资人赌的是一个小而强的团队能在机器人基础模型上做出壁垒。目前能确认的事实只有融资规模、投资方和这份 API 雏形；dream-1 的实际能力、服务的开放程度和定价都还是未知数。关注机器人基础模型或具身智能方向的读者，可以留意这家公司后续的正式发布——它是否能兑现“Robotics within Reach”，要等到开发者真正拿到 API key 之后才能判断。

> 本文参考自 [Dream Labs (@dreamlabs_ai) on X](https://x.com/dreamlabs_ai/status/2112230668805828801)