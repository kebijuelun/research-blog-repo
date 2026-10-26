# Inception v3：老 CNN 诊断更准

只看皮肤照片，通用大模型能否像专科模型一样识别皮肤癌，并判断患者能否安全离院？这个问题直接关系到 AI 能否进入皮肤科分诊，而不只是生成一段听起来合理的医学建议。皮肤科医生 Magnus Lynch 用 200 处经活检确认的皮损，对比了 GPT-6 Astra、Claude Fable 5.1 和微调后的 Inception v3：前两者是通用大模型，后者是 2015 年发布的卷积神经网络（CNN）。结果，CNN 的首选诊断答对 125 例（62%），两个大模型分别只答对 84 例和 85 例（均为 42%）。不过，诊断更准并不自动等于分诊更安全，这组实验恰好把两者的差别摆了出来。

## 一处黑色素瘤，两种危险误判

Lynch 展示的第一处案例是耳部的侵袭性黑色素瘤，厚度不足 0.8 mm。GPT-6 Astra 和 Claude Fable 5.1 都把它判为脂溢性角化病——一种无害的疣状增生物——并认为患者可以离院；微调后的 Inception v3 则依据皮肤镜图像给出了黑色素瘤诊断。

![耳部侵袭性黑色素瘤的临床照片](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_7403025.jpg)

> 图解：这处耳部皮损厚度不足 0.8 mm，组织学证实为侵袭性黑色素瘤。两个大模型均误判为脂溢性角化病，并建议可以离院。图片来自 MILK10k，MILK 研究团队，CC BY-NC。

![同一处黑色素瘤的皮肤镜图像](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_4552372.jpg)

> 图解：这是同一皮损的皮肤镜图像，即通过皮肤科医生使用的带照明放大设备拍摄的放大视图。CNN 根据这一视图作出了正确诊断；单次正确判断只是案例，整体表现仍要看全部 200 例。

这个案例之所以重要，不只是“老模型答对、大模型答错”。误判为良性病变还附带了离院建议，可能让本该继续处理的患者退出诊疗流程。因此，测试既要问“它叫什么”，也要问“能不能不治、不活检、不随访”。

## 200 处活检皮损，怎样构成一次公平对比

测试样本来自国际皮肤影像协作组织（ISIC）于 2025 年公开的 MILK10k 数据集。Lynch 随机抽取 200 处皮损，每一处都有临床近照和皮肤镜图像，也都经过组织病理学确认：

- **100 处不应直接离院：** 30 例黑色素瘤、25 例基底细胞癌、25 例鳞状细胞癌（包括 Bowen 病），以及 20 例光线性角化病。
- **100 处可以安全离院：** 50 例良性痣和 50 例脂溢性角化病。

所有皮损都做过活检，连良性样本也不例外。这使测试比普通归档图片更棘手：一颗最终被切下并证实为良性的痣，至少曾让某位临床医生担心过。

![测试皮损一的临床照片](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_5168853.jpg)

> 图解：测试集中另外三处皮损之一，此为临床近照，下一图是对应的皮肤镜视图。原文没有逐图标出这三处皮损的组织学诊断，因此不能仅凭图片补写标签。图片来自 MILK10k，CC BY-NC。

![测试皮损一的皮肤镜图像](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_2485182.jpg)

> 图解：与上一张临床照片对应的皮肤镜图像。每个样本都用两种视图参与测试。

![测试皮损二的临床照片](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_1091183.jpg)

> 图解：第二处测试皮损的临床近照，下一图为其皮肤镜视图。图片来自 MILK10k，CC BY-NC。

![测试皮损二的皮肤镜图像](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_6551749.jpg)

> 图解：第二处皮损的皮肤镜图像，与上一张临床照片属于同一样本。

![测试皮损三的临床照片](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_3428806.jpg)

> 图解：第三处测试皮损的临床近照，下一图为其皮肤镜视图。图片来自 MILK10k，CC BY-NC。

![测试皮损三的皮肤镜图像](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/ISIC_6279832.jpg)

> 图解：第三处皮损的皮肤镜图像，与上一张临床照片属于同一样本。

两个大模型都收到两张图片，没有其他病史信息；系统要求它们从同样的 6 种诊断中选出最可能的一项，也可以回答“不确定”，同时给出置信度、第二选择，以及是否可以不做治疗、活检或随访而安全离院。两者使用相同提示词和默认设置，每个案例只运行一次，也都没有拒绝作答。

CNN 这一边没有刻意追求 SOTA。Lynch 使用在日常生活照片上预训练的 Inception v3，在单张 H100 GPU 上，用 ISIC Archive 的 68,011 张皮肤镜及临床图片微调全部网络层，微调约 20 分钟。测试集中的 200 处皮损，以及同一皮损或同一患者的其他照片，都事先从训练集和验证集中移除；测试样本也没有用于挑选模型或设定阈值。CNN 分别给两张图评分，再以皮肤镜图像占更大权重的方式合并，最后从同样的 6 个类别中作答。

这套对照的价值不在于比较模型规模，而在于区分两种能力：通用模型能否凭视觉与语言知识完成专科分类，以及一个针对固定任务训练的视觉模型能做到什么程度。数据去重和统一候选诊断，让这个比较至少避开了“CNN 直接背过测试答案”这一显而易见的解释。

## 62% 对 42%，优势主要出现在哪里

![三个模型在全部皮损及各组织学类别上的首选诊断正确数](https://raw.githubusercontent.com/kebijuelun/research-blog-repo/main/html/A-10-year-old-CNN-outperforms-frontier-LLMs-for-skin-cancer-diagnosis/images/figure-9.png)

> 图解：横轴为首选诊断正确率，柱端数字为“答对数／该组织学类别的样本数”。每组从上到下依次为 GPT-6 Astra、Claude Fable 5.1 和 Inception v3。CNN 在总体上以 125/200 对 84/200、85/200 领先，但并非每个类别都更好：光线性角化病一项，它反而落后于两个大模型。

CNN 的总体优势并非只由黑色素瘤贡献。30 例黑色素瘤中，GPT-6 Astra、Claude Fable 5.1 和 CNN 分别首选答对 8、11 和 13 例；真正拉开差距的是良性类别：CNN 识别出 37/50 的痣和 26/50 的脂溢性角化病，GPT-6 Astra 分别为 16/50 和 13/50，Claude Fable 5.1 则为 33/50 和 6/50。

其他类别也显示这不是一边倒。基底细胞癌上，GPT-6 Astra 答对 25/25，CNN 为 24/25，Claude Fable 5.1 为 18/25；鳞状细胞癌上，GPT-6 Astra、Claude Fable 5.1 和 CNN 分别答对 12/25、5/25 和 16/25。若允许答案落在首选或第二选择中，CNN 命中 173/200，两个大模型分别为 134/200 和 138/200。

三个模型的错误模式也不相同。GPT-6 Astra 把 75 处皮损判成基底细胞癌，而真实样本中只有 25 处，其中还包括 14 处黑色素瘤；Claude Fable 5.1 则把 53 处判成光线性角化病，而真实数量是 20 处。CNN 漏判的 17 处黑色素瘤里，有 8 处被称为痣，6 处被称为脂溢性角化病。也就是说，它虽然在分类总数上领先，仍会把部分危险病变推向良性类别。

## 诊断更准，为什么仍可能漏掉黑色素瘤

下表把“病名答对多少”和“患者能否离院”放在一起。这里的“保留”指没有直接放行不应离院的皮损，因此数量越高，漏放行越少；“安全离院”则针对 100 处良性皮损。

| 指标 | GPT-6 Astra | Claude Fable 5.1 | CNN（Inception v3） |
| --- | --- | --- | --- |
| 首选诊断正确（共 200 例） | 84（42%） | 85（42%） | 125（62%） |
| 首选或第二选择正确（共 200 例） | 134 | 138 | 173 |
| 黑色素瘤首选诊断正确（共 30 例） | 8 | 11 | 13 |
| 不应离院的皮损被保留（共 100 例） | 97 | 96 | 预设阈值下为 77；匹配阈值时为 98 或 99 |
| 良性皮损被安全离院（共 100 例） | 32 | 22 | 预设阈值下为 76 |
| 黑色素瘤被放行（共 30 例） | 3 | 4 | 预设阈值下为 18；匹配阈值时为 2 或 1 |

这张表修正了“62% 对 42%”可能造成的简单印象。CNN 在预设阈值下只保留了 77/100 的不应离院皮损，并放行 18/30 的黑色素瘤；把阈值调整到匹配条件后，保留数升至 98 或 99，放行黑色素瘤降至 2 或 1。分类成绩和离院决策不是同一件事，后者明显依赖阈值如何设定。

换个提问方式，结果又会变化。Lynch 另取 100 例经活检确认的黑色素瘤，只给大模型临床照片，并以患者口吻问“我需要看医生吗？”GPT-6 Astra 和 Claude Fable 5.1 分别在 99 例和 98 例中建议就医。但它们在良性皮损上也几乎总是建议就医。这更像谨慎地扩大转诊，而不是证明它们已经能精确区分病变；在医疗场景中，少漏诊和少做无效转诊必须一起衡量。

## 这项测试能说明什么，不能说明什么

作者列出了几项直接限制，它们决定了结论的边界：

- **样本较小。** 200 处皮损足以显示总体模式，却不足以给接近的结果排出可靠名次；尤其黑色素瘤只有 30 例，最关键数字的不确定性很大。每个大模型案例只运行一次，而同一模型的回答可能随运行变化。
- **病例本身偏难。** 所有样本都经过活检。在普通归档图片上，CNN 区分“应否离院”的 ROC 曲线下面积（AUC）为 0.96，而这里只有 0.86。AUC 越接近 1，表示模型把风险较高样本排在较前面的能力通常越强；0.86 与 0.96 的差距提示这批病例更难，也提醒读者不要把归档数据成绩直接搬到临床。
- **训练污染并不对称。** MILK10k 自 2025 年起公开，大模型训练时可能见过它；如果有影响，理论上更可能帮助大模型。CNN 则通过去重明确没有见过这些测试皮损及同患者图片。
- **CNN 不是当前最佳专科模型。** Inception v3 只是基线。作者指出，当前皮肤癌诊断 AI 已使用在更多数据上训练的图像 Transformer 集成模型，基于这些技术且受监管的产品表现明显更好。

因此，这不是“旧 CNN 已胜过所有现代皮肤癌 AI”，也不是一份临床有效性认证。它支持的是一个更窄也更有用的判断：在这套只给图像、限制六分类的测试里，通用能力更强的前沿大模型，并没有自然获得比专用视觉模型更好的皮损识别能力。

## 小模型的价值，落在具体部署方式上

这个 CNN 小到可以完全在手机浏览器中通过 WebGPU 运行，不需要把图片上传到服务器。对涉及皮肤照片的尝试而言，本地推理提供了一种清晰的隐私与部署选择；不过，网页中的随机样本按钮仍取自这 200 个测试病例，不能把在页面上的试用当作独立临床验证。

作者还公开了病例、提示词、每一条大模型回复和评分代码，评分由脚本依据组织学结果完成。测试设计和临床判断由 Lynch 负责；分析和初稿使用了 Anthropic 的 Claude Code，而被测模型之一 Claude Fable 5.1 同样来自 Anthropic。这个利益与技术工具上的关联也被作者明确披露。

回到开头的问题，这组实验最值得关注的不是“老模型赢了”，而是它把模型选择、任务边界和决策阈值分开呈现。在 200 例六分类测试中，微调 Inception v3 的首选诊断准确率为 62%，两个大模型均为 42%；但更准确的分类没有自动带来更安全的离院决策，CNN 的结果随阈值变化明显。对需要快速、固定类别视觉判断的任务，小型专科模型仍有实际价值，而且可以在手机浏览器本地运行。

样本量、单次运行、困难病例和非 SOTA 基线都限制了结论的外推。这项测试不能替代更大规模、前瞻性的临床评估，也不能作为个人自诊依据。

原文参考：[A 10-year-old CNN outperforms frontier LLMs for skin cancer diagnosis](https://lynchlab.net/llm-skin-cancer)