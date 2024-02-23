# -
聚焦于大模型空间站| LLM SPACE！ 人类的未来在于探索未知，挑战，科技与生命



大模型日报（2月 22日）

# 资讯

## 研究

### 爆火Sora背后的技术，一文综述扩散模型的最新发展方向

贡献人：@刘奕龙

https://mp.weixin.qq.com/s/sxaahA116ivqksJa38e9Ig

为了使机器具有人类的想象力，深度生成模型取得了重大进展。这些模型能创造逼真的样本，尤其是扩散模型，在多个领域表现出色。扩散模型解决了其他模型的限制，如 VAEs 的后验分布对齐问题、GANs 的不稳定性、EBMs 的计算量大和 NFs 的网络约束问题。因此，扩散模型在计算机视觉、自然语言处理等方面备受关注。扩散模型由两个过程组成：前向过程和反向过程。前向过程把数据转化为简单的先验分布，而反向过程则逆转这一变化，用训练好的神经网络模拟微分方程来生成数据。与其他模型相比，扩散模型提供了更稳定的训练目标和更好的生成效果。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/080effab9b3146819d3d11cd2e9e8d6f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=811&s=685251&e=png&b=faf8f7)

  


### 模型融合、混合专家、更小的LLM，几篇论文看懂2024年LLM发展方向

贡献人：@刘奕龙

https://mp.weixin.qq.com/s/qImKOQXLoZqLTW-SVISKHA

在过去的 2023 年中，大型语言模型（LLM）在潜力和复杂性方面都获得了飞速的发展。展望 2024 年的开源和研究进展，似乎我们即将进入一个可喜的新阶段：在不增大模型规模的前提下让模型变得更好，甚至让模型变得更小。现在，2024 年的第一个月已经过去，也许是时候盘点一番新年首月进展了。近日，AI 研究者 Sebastian Raschka 发布了一份报告，介绍了四篇与上述新阶段有关的重要论文。它们的研究主题简单总结起来是这样：

1.  权重平均和模型融合可将多个 LLM 组合成单个更好的模型，并且这个新模型还没有传统集成方法的典型缺陷，比如更高的资源需求。
1.  代理调优（proxy-tuning）技术可通过使用两个小型 LLM 来提升已有大型 LLM 的性能，这个过程无需改变大模型的权重。
1.  通过将多个小型模块组合起来创建混合专家模型，可让所得 LLM 的效果和效率媲美甚至超越更大型的对应模型。
1.  预训练一个小型的 1.1B 参数的 LLM 可降低开发和运营成本，并能为教育和研究应用带来新的可能性。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/039799725d4b4cb39608ef9b19ba32f5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1534&h=898&s=375460&e=png&b=fdfdfd)

  


### 受 ChatGPT 启发，结合 Transformer 和 RL-MCTS 进行从头药物设计

贡献人：@刘奕龙

https://mp.weixin.qq.com/s/6BAt-tb2RHCH_BJV-ocBew

通过从头药物设计发现新型治疗化合物是药物研究领域的一项关键挑战。传统的药物发现方法通常资源密集且耗时，这促使科学家探索利用深度学习和强化学习技术力量的创新方法。在这里，美国查普曼大学（Chapman University）的研究人员开发了一种称为 drugAI 的新型药物设计方法，该方法利用编码器-解码器 Transformer 架构与通过蒙特卡罗树搜索（RL-MCTS）进行的强化学习来加快药物发现过程，同时确保生产具有药物样特性和对其靶标具有强结合亲和力的有效小分子。与两种现有的基准方法相比，drugAI 生成的化合物的有效性和药物相似性都有显著改善。此外，drugAI 确保生成的分子对其各自的靶标表现出强大的结合亲和力。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1cf89d8919ad4028aff514268dac0841~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=659&s=732131&e=png&b=0e3247)

  


  


## 产业

### 开源大模型王座易主！谷歌Gemma杀入场，笔记本可跑，可商用

贡献人：@刘奕龙

https://mp.weixin.qq.com/s/_iCYfqmXA3enKn3Hm-DwSA

开源领域大模型，迎来了重磅新玩家。谷歌推出了全新的开源模型系列「Gemma」。相比 Gemini，Gemma 更加轻量，同时保持免费可用，模型权重也一并开源了，且允许商用。本次发布包含两种权重规模的模型：Gemma 2B 和 Gemma 7B。每种规模都有预训练和指令微调版本。想使用的人可以通过 Kaggle、谷歌的 Colab Notebook 或通过 Google Cloud 访问。当然，Gemma 也第一时间上线了 HuggingFace和HuggingChat，每个人都能试一下它的生成能力。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ea7f39cf748048d182ea3b601226d680~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=608&s=247814&e=png&b=192332)

  


### 英伟达日进5.7亿，黄院士躺印钞机上了

贡献人：@刘奕龙

https://mp.weixin.qq.com/s/1hNHnxJi0GjfAUnzh7-sng

英伟达最新财报出炉。连创“三高”：1. **2024财年Q4季度营收达221亿美元**（净利122亿美元），比上一季度增长22%，比上一年增长265%。2. 扛把子的**数据中心**营收占据184亿美元，比第三季度增长27%，比上一年**飙升409%** 。3. 2024财年全年营收也出来了：**609亿美元**（约合4384亿人民币），比去年多了126%。净利润则为297亿美元，约合2136亿人民币，**相当于日进5.7个“小目标”** 。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c9fc4f62953f452b99fb4754c6811ec2~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=3240&h=2160&s=10819409&e=png&b=8b8372)

  


### AI 黑马 Groq 颠覆英伟达 ？LPU 性能与成本解读

贡献人：@刘奕龙

https://mp.weixin.qq.com/s/LowpdbHg4gorvXDFAKNrBQ

Groq 是一家技术公司，由 Jonathan Ross 在 2016 年创立。Ross 曾是 Google 第一个张量处理单元（TPU）的创造者，他的创立理念源于一个观点：芯片设计应从软件定义网络（SDN）中吸取灵感。2024 年 2 月 13 日，Groq 在 ArtificialAnalysis.ai 最新的 LLM 基准测试中明显获胜，Groq 在延迟与吞吐量等关键性能指标上击败了八名参与者，Groq 处理吞吐量达到其他推理服务的 4 倍，同时收费还不到 Mistral 自己的 1/3。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/481c549f9e8c4791bff103c8f9403378~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1080&h=567&s=177584&e=png&b=f7f6f6)

  


### 三星移动部门负责人透露 Galaxy AI 发展计划，将扩展到可穿戴设备

贡献人：@刘奕龙

https://www.ithome.com/0/751/401.htm

三星移动部门负责人 TM Roh 近日透露了该公司未来在人工智能 (AI) 方面的计划，以及如何扩展其应用范围。Roh 表示，三星下一步计划是将 Galaxy AI 的应用范围扩展到更多设备和服务，包括可穿戴设备。他透露计划在“不久的将来”将 Galaxy AI 功能引入“部分”Galaxy 可穿戴设备。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/03d2694f41c14016ab7a1ef588dd8794~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=650&h=366&s=208537&e=png&b=0e0d0d)

  


  


# 推特

### 创业公司Magic取得重大突破，可能使其实现类似于 OpenAI 的 Q* 的“主动推理”能力

贡献人：@Angela Chen Hanzhe 2022

https://x.com/rowancheung/status/1760336478411092277?s=20

🚨 Magic，一家正在构建 AI 软件工程师同事的创业公司刚刚取得了重大突破。

据 The Information 报道，Magic 宣称已经取得了一项技术突破，这项突破可能使其实现类似于 OpenAI 的 Q* 的“主动推理”能力。

他们还可以处理多达 350 万字的文本上下文，是 Google 最新的 Gemini 1.5 Advanced 的 5 倍。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ba1e01b961f84f458f974c4f95f3136f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1642&h=396&s=323014&e=png&b=fdfdfd)

  


### Karpathy深入研究Gemma tokenizer：基本上是 Llama 2 分词器，除了更大，拥有更多特殊令牌，唯一的功能性差异是将 add_dummy_prefix 关闭为 False

贡献人：@Angela Chen Hanzhe 2022

https://x.com/karpathy/status/1760350892317098371?s=20

鉴于我昨天发布了我的 Tokenizer 视频，我想深入研究一下 Gemma tokenizer 会很有趣。

首先，Gemma 技术报告 [pdf]：

https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf

说：“我们使用了 Gemini 的 SentencePiece 分词器（Kudo 和 Richardson, 2018）的一个子集，以保持兼容性。它分割数字，不删除额外的空白，并依赖字节级编码处理未知令牌，遵循了（Chowdhery 等人，2022）和（Gemini 团队，2023）使用的技术。词汇表大小为 256k 令牌。”

这个 tokenizer.model 文件随这个代码发布：

https://github.com/google/gemma_pytorch/blob/main/tokenizer/tokenizer.model

我用 Python 解码了这个模型的 protobuf，这里是它与 Llama 2 分词器的区别：

https://diffchecker.com/TRnbKRMH/

注释：

-   词汇表大小相当大：从 32K 增加到 256K
-   add_dummy_prefix 设置为 False。与 Llama 不同，但与 GPT 一致。这在“保留数据原样”方面更加一致，因为没有预处理步骤向编码文本添加空格。
-   model_prefix 是训练数据集的路径，看起来很有趣："/cns/mf-d/home/gemini-data-access/tokenizers/final_v1_51GB_run1/bpe_coverage_0_999995_v5/255969"。似乎表明分词器训练语料库大约为 51GB（？）。
-   存在许多用户定义的符号（即特殊令牌），例如将多达 31 个换行符“硬编码”为令牌，以及大量其他不明令牌。我尝试解码八进制表示，但不清楚这里发生了什么。还有很多看起来像是 HTML 元素的特殊令牌，例如 <table>、<tr>、<td>、<i>、<b> 等。不完全确定未使用的令牌是用于什么，可能是为了使将来尝试添加更多特殊令牌的微调更容易，因为无需调整词汇表大小和执行模型手术（？）。

总而言之，这基本上是 Llama 2 分词器，除了更大（从 32K 增加到 256K），拥有更多特殊令牌，唯一的功能性差异是将 add_dummy_prefix 关闭为 False。例如，分词：

"hello world" 变成：

[17534, 2134]

['hello', '▁world']

否则，它会被预处理为 " hello world"（注意前导空格）并分词为：

[25612, 2134]

['▁hello', '▁world']

很酷

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c59531d5bf1b4c6e9fdab17998c24353~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1174&h=974&s=246805&e=png&b=000000)

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/099d8aac4e2947f9a4e8af9875ed2de1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1178&h=1300&s=369794&e=png&b=000000)

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/840d928fc1fc4860af82ae157caa547c~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1166&h=312&s=48981&e=png&b=000000)

  


### 谷歌Jack Krawczyk发推回应Gemini 在某些历史图像生成描述中提供了不准确的信息遭反驳

贡献人：@Angela Chen Hanzhe 2022

https://x.com/JackK/status/1760334258722250785?s=20

我们意识到 Gemini 在某些历史图像生成描述中提供了不准确的信息，我们正在立即修复这个问题。

作为我们 AI 原则的一部分 https://ai.google/responsibility/principles/，我们设计图像生成能力以反映我们的全球用户群，并且我们严肃对待代表性和偏见问题。

我们将继续为开放式提示做到这一点（例如，人遛狗的图像是普遍的！）

历史背景有更多的细微差别，我们将进一步调整以适应这一点。

这是对齐过程的一部分 - 对反馈的迭代。谢谢您的反馈，请继续提供！

* * *

评论区（3800赞）： Josh：杰克，为什么这种“缺乏细微差别”只适用于来自欧洲或英语圈的历史场景？

  


  


  


  


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/405a22697fb54e90b4ece2acb5a95198~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1192&h=1342&s=286834&e=png&b=000000)

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/96a2b9eef90b45299144a975a728eb5b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1198&h=420&s=84773&e=png&b=000000)

  


### YOLO-World YouTube 教程发布，包含模型架构、在 Colab 中处理图像和视频等

贡献人：@Angela Chen Hanzhe 2022

https://x.com/skalskip92/status/1760305630223479162?s=20

YOLO-World 的 YouTube 教程已发布

-   模型架构
-   在 Colab 中处理图像和视频
-   提示工程和检测细化
-   模型的优点和缺点

在这里观看：https://youtube.com/watch?v=X7gKBGVz4vs

暂时无法在飞书文档外展示此内容

  


  


### 使用声音和AI CBT治疗师对话：使用Retell AI，声速很快

贡献人：@Angela Chen Hanzhe 2022

https://x.com/levelsio/status/1760393920469426552?s=20

✨ 你现在可以用自己的声音与我的 CBT（认知行为疗法）治疗师 [http://cbt.chat](http://cbt.chat/) 对话，它会使用今天发布的 Retell AI 回应

Retell AI 的声速非常快，实际上感觉像是你正在进行对话而不是等待一个机器人

我现在将尝试用 GPT4 Turbo 来编码这个，然后发布另一个演示，如果这能工作的话，潜力巨大

我认为他们使用 GPT3.5 做演示是因为我的 CBT 治疗师相当笨，还卡在了一个锻炼计划上，但那只是 GTP3.5 就是 GPT3.5

取消静音在这里听我的对话：

暂时无法在飞书文档外展示此内容

  


### Austin Huang分享Gemma.cpp：轻量级的、独立的 C++ 推理引擎，用于谷歌Gemma模型

贡献人：@Angela Chen Hanzhe 2022

https://x.com/austinvhuang/status/1760375890448429459?s=20

我很高兴分享 gemma.cpp 的发布 - 一个轻量级的、独立的 C++ 推理引擎，用于 Google 的 Gemma 模型：

现代大型语言模型（LLM）推理引擎是复杂的系统，通常具有超出传统神经网络运行时的定制功能。随之而来的是通过高级算法和低级计算的共同设计来研究和创新的机会。然而，部署导向的 C++ 推理运行时与实验设计不符，它们不是为实验设计的，而以 Python 为中心的机器学习研究框架通过编译抽象化低级计算。

gemma.cpp 提供了 Gemma 2B 和 7B 模型的最小实现，重点是简单和直接性而不是完全的通用性。这受到了如 ggml、llama.c 和 llama.rs 等垂直集成模型实现的启发。

gemma.cpp 面向实验和研究用例。它旨在易于嵌入其他项目，依赖性最小，且容易修改，核心实现仅约 2K 行代码（另有约 4K 行代码的支持工具）。我们使用 Google Highway Library 来利用便携式 SIMD 进行 CPU 推理。

对于面向生产的边缘部署，我们推荐使用 Python 框架（如 JAX、Keras、PyTorch 和 Transformers（这里的所有模型变体））的标准部署路径。

暂时无法在飞书文档外展示此内容

  


### HuggingFace分享Cosmopedia：3000 万个由 Mixtral 8x7B 生成的合成教科书、博客文章、故事、帖子和 WikiHow

贡献人：@Angela Chen Hanzhe 2022

https://x.com/Yampeleg/status/1760271220384113024?s=20

Cosmopedia 是一个由 Mixtral-8x7B-Instruct-v0.1 生成的合成教科书、博客文章、故事、帖子和 WikiHow 文章的数据集。该数据集包含超过 3000 万个文件和 250 亿个令牌，使其成为迄今为止最大的开放合成数据集。

它涵盖了多种主题；我们试图映射在网页数据集如 RefinedWeb 和 RedPajama 中存在的世界知识，并生成覆盖这些主题的合成内容。这是 Cosmopedia 的 v0.1 版本，有很大的改进空间和更全面覆盖主题的可能。我们希望这个数据集将帮助社区在日益引人入胜的合成数据领域的研究工作。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/09ed21b41b014623987ffdc7c5e6669e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1794&h=1356&s=402809&e=png&b=ffffff)

  


  


# 论文

### **神经网络扩散**

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.13144v1

扩散模型在图像和视频生成中取得了显著成功。本研究表明，扩散模型也可以“生成性能优越的神经网络参数”。我们的方法简单，利用自动编码器和标准的潜在扩散模型。自动编码器提取训练网络参数子集的潜在表示。然后训练扩散模型从随机噪声合成这些潜在参数表示。它生成新的表示，经过自动编码器解码器，并输出可用作新的网络参数子集。在各种架构和数据集上，我们的扩散过程始终生成性能相当或优于训练网络的模型，而额外成本极小。值得注意的是，我们在经验上发现生成的模型与训练网络表现出不同。我们的结果鼓励更多探索扩散模型的多功能使用。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/792f65f1102447ed9a240da58f5ea7b3~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1034&h=1374&s=807644&e=png&b=fefbfb)

  


### OlympiadBench **：促进智能通用人工智能与奥林匹亚级别的双语多模态科学问题**

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.14008v1

最近的进展已经看到大型语言模型（LLMs）和大型多模态模型（LMMs）在各种任务中超越了一般人类能力，在多个领域接近人类专家水平。随着传统基准对这些模型变得不再具有挑战性，新的严格挑战是必不可少的，以衡量它们的先进能力。在这项工作中，我们提出了OlympiadBench，一个奥林匹克级双语多模态科学基准，包括来自奥林匹克级数学和物理竞赛以及中国高考的 8952 个问题。每个问题都有专家级的逐步推理注释。通过在OlympiadBench 上评估顶尖模型，我们实施了全面的评估方法，准确评估模型的响应。值得注意的是，表现最好的模型 GPT-4V 在 OlympiadBench 上取得了 17.23% 的平均分数，仅在物理学中达到了 11.28%，突显了基准的严谨性和物理推理的复杂性。我们对 GPT-4V 进行的分析指出了幻觉、知识遗漏和逻辑谬误等普遍问题。我们希望我们具有挑战性的基准可以成为帮助未来AGI研究努力的宝贵资源。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5538dc208aba4586ba8d7d0e5cb64554~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1001&h=1443&s=724800&e=png&b=fefaf9)

  


### **LongRoPE：将LLM上下文窗口延长到200万个token以上**

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.13753v1

大语言模型（LLMs）中的大上下文窗口是一种理想特性。然而，由于微调成本高、长文本稀缺以及新token位置引入的灾难性值，当前扩展的上下文窗口仅限于约128k个token。本文介绍了LongRoPE，首次将预训练LLMs的上下文窗口扩展到令人惊叹的2048k个token，只需256k的训练长度内进行最多1k次微调步骤，同时保持在原始短上下文窗口的性能。这是通过三个关键创新实现的：（i）我们通过有效搜索确定并利用两种形式的非均匀性在位置插值中，为微调提供更好的初始化，并在非微调情况下实现8倍扩展；（ii）我们引入了一种渐进扩展策略，首先微调一个256k长度的LLM，然后在微调的扩展LLM上进行第二次位置插值，以实现2048k的上下文窗口；（iii）我们重新调整LongRoPE的8k长度以恢复短上下文窗口的性能。在LLaMA2和Mistral上进行的广泛实验展示了我们方法的有效性。通过LongRoPE扩展的模型保留了原始架构，只对位置嵌入进行了轻微修改，并可以重复使用大部分预先存在的优化。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/45c3c269dc6d401a9ff6739c1c971abc~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1049&h=1394&s=647778&e=png&b=fefdfd)

  


### **大型预训练语言模型的达芬奇密码：解读退化知识神经元**

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.13731v1

本研究探讨了预训练语言模型（PLMs）中事实知识存储机制。先前的研究表明，事实知识存储在多层感知器权重中，一些存储单元表现出退化特征，称为退化知识神经元（DKNs）。本文提供了涵盖结构和功能方面的DKNs全面定义，开创了对PLMs事实知识存储单元结构的研究。基于此，我们引入了神经拓扑聚类方法，允许以任意数量和结构形成DKNs，从而实现更准确的DKN获取。此外，我们还引入了神经-退化分析框架，独特整合了模型稳健性、可进化性和复杂性，全面评估PLMs。在这个框架下，我们对2个PLMs、4个数据集和6个设置进行了34次实验，突显了DKNs的关键作用。代码将很快发布。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e13c17231378464f9e04ceb26cca7016~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1105&h=1495&s=795568&e=png&b=fefdfd)

  


### CirticBench **：评估大型语言模型作为评论者**

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.13764v1

摘要：评价能力对于大语言模型（LLMs）的可扩展监督和自我改进至关重要。尽管许多最近的研究探讨了LLMs的评价能力来判断和改进生成过程中的缺陷，但如何全面和可靠地衡量LLMs的评价能力仍未得到充分探讨。本文介绍了CriticBench，一个旨在全面和可靠地评估LLMs的四个关键评价能力维度的新型基准：反馈、比较、改进和元反馈。CriticBench包括九项不同的任务，每项任务评估LLMs在不同质量粒度水平上评价响应的能力。我们对开源和闭源LLMs的广泛评估揭示了评价能力与任务、响应质量和模型规模之间有趣的关系。CriticBench的数据集、资源和评估工具将在https://github.com/gmftbyGMFTBY/CriticBench上公开发布。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1bdcdeba70684baf84b9ede7c3c3993a~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1024&h=1457&s=754350&e=png&b=fefbfb)

  


### **分析序列构成对语言模型预训练的影响**

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.13991v1

大多数语言模型预训练框架将多个文档串联成固定长度序列，并使用因果屏蔽来计算每个标记在其上下文中的可能性；由于其简单性和效率，这一策略被广泛采用。然而，迄今为止，预训练序列构成策略对模型泛化性能的影响仍未得到充分探讨。在本研究中，我们发现应用因果屏蔽会导致在预训练过程中包含来自先前文档的干扰信息，从而对语言建模和下游任务的性能产生负面影响。在文档内的因果屏蔽中，每个标记的可能性仅与同一文档中的先前标记有关，消除了来自先前文档的潜在干扰信息，并显著提高了性能。此外，我们发现串联相关文档可以减少一些潜在的干扰，我们提出的高效的基于检索的序列构建方法BM25Chunk能够在不牺牲效率的情况下提高语言模型的上下文学习能力(+11.6%)、知识记忆能力(+9.8%)和上下文利用能力(+7.2%)。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/890c7afd2c75468e9e41c5c7f89d6e82~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=984&h=1451&s=713124&e=png&b=fffefe)

  


  


### RealDex **：向具有人类般抓取能力的机器人灵巧手发展**

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.13853v1

在这篇论文中，我们介绍了RealDex，一个捕捉真实灵巧手抓取动作的开创性数据集，注入了人类行为模式，丰富了多视角和多模态视觉数据。利用远程操作系统，我们实时无缝同步人机器人手姿势。这个类人动作的收集对于训练灵巧手更自然、更精确地模仿人类动作至关重要。RealDex在推动人形机器人在真实世界场景中的自动感知、认知和操纵方面具有巨大潜力。此外，我们介绍了一个先进的灵巧抓取动作生成框架，与人类经验相一致，并通过有效利用多模态大语言模型增强了真实世界的适用性。广泛的实验已经证明了我们方法在RealDex和其他开放数据集上的优越性能。完整的数据集和代码将在本研究发表后提供。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dddf055eae184284bab2a4d2ee772b35~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1087&h=1306&s=777329&e=png&b=fdfafa)

  


### Aria 日常活动数据集

贡献人：@林李挚

链接：http://arxiv.org/abs/2402.13349v1

我们介绍Aria Everyday Activities（AEA）数据集，这是一个使用Project Aria眼镜记录的自我感知多模态开放数据集。AEA包含了在五个地理多样的室内位置由多个佩戴者记录的143个日常活动序列。每个记录都包含通过Project Aria眼镜记录的多模态传感器数据。此外，AEA提供了机器感知数据，包括高频全局对齐的3D轨迹、场景点云、每帧3D凝视向量和时序语音转录。在本文中，我们展示了通过该数据集实现的几个示例研究应用，包括神经场景重建和提示分割。AEA是一个开放数据集，可从projectaria.com下载。我们还提供Project Aria Tools中如何使用数据集的开源实现和示例。

  


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0a88f671c4014620b554dd7aaa0f2665~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1057&h=1273&s=695938&e=png&b=fffefe)

  


  


# 产品

### Locofy Lightning

贡献人：@刘子嘉

https://www.locofy.ai/

Locofy Lightning 是一个通过 LocoAI 的大型设计模型 (LDM) 支持的工具，能够将 Figma 设计转换为前端代码，实现一键转换的功能。它可以帮助您获得响应式、交互式设计和可重用的代码组件。同时，您可以轻松地将生成的代码同步到 GitHub，或者拉取到 VS Code 中进行进一步编辑和开发。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6ee033ebebbb4acc9be5911ed172e10e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=2452&h=1438&s=962508&e=png&b=5686f6)

  


  


# HuggingFace&Github

### Sorafm

贡献人：@刘子嘉

https://github.com/all-in-aigc/sorafm

这是一个 nextjs 全栈开发项目，等待 Sora API 发布后会连接上生成视频的接口。目前，收集了全网的 Sora 视频形成Sora Showcase 网站。

  


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/700ce2e13df84ee78fb53218d7c0ef8e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1500&h=1195&s=973413&e=png&b=121826)

  


### Aimmy

贡献人：@刘子嘉

https://github.com/himesshawne/Aimmy

Aimmy 是一款由 BabyHamsta & Nori 开发的多功能 AI 瞄准器，旨在使游戏更容易让更广泛的受众参与。它利用 DirectML、ONNX 和 YOLOV8 技术来检测玩家，采用 C# 编写，速度快效率高。与其他 AI 瞄准器不同，Aimmy 具有许多独特功能，如在运行过程中切换不同模型，调整瞄准精度等设置。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/51367eda919843c789c5a1ea765de351~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1920&h=540&s=1139262&e=png&b=060607)

  


### SheepRL

贡献人：@刘子嘉

https://github.com/Eclectic-Sheep/sheeprl

SheepRL 是一个 RL 框架，借助 Lightnin Fabric 框架可以同时实现可扩展性和简单性，通过 Agent 的形式让 RL 算法与环境解耦，以便它可以在任何环境中使用，现阶段可以实现许多不同游戏环境的运行。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f7715c119afd4700a217abf8a09c50e7~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1732&h=1054&s=843686&e=png&b=fffefe)

  


  


# 投融资

### BRIA获得2400万美元A轮融资，用于负责任的视觉生成AI平台。

贡献人：@谭泽琪

https://automationvault.net/bria-secures-24m-in-series-a-funding-for-responsible-visual-generative-ai-platform/

BRIA，一家位于特拉维夫的公司，在Series A融资中成功筹集到2400万美金，旨在变革视觉生成人工智能领域。这轮融资由GFT Ventures、Intel Capital和Entrée Capital领投，Publicis Groupe、Getty Images和Samsung Next等也参与投资。公司提供一个开放平台，支持开发者将人工智能能力无缝集成到现有产品和系统中，并致力于推动合规、道德的AI实践，确保所有训练数据均已合法授权。通过提供包括文本到视频的服务、API、SDK和源代码，BRIA旨在帮助企业在AI驱动的世界中获得成功。公司已通过提供注重合法合规和负责任的AI的商用解决方案，赢得了业务合作伙伴的青睐，并期待借助投资者的支持继续增长并深刻影响AI技术的未来。

  


### Antler创始人对其在东南亚的垂直人工智能投注

贡献人：@谭泽琪

https://techcrunch.com/2024/02/21/antler-ai/

东南亚正涌现一批专注于特定行业的垂直人工智能初创公司，服务领域涵盖从海产品到金融等。新加坡风险投资公司Antler近期对其中37家企业进行了投资，总投资额达510万美元，涉及种子前期交易。这还包括与马来西亚国家主权财富基金Khazanah的战略合作，投资了七家初创企业。Antler联合创始人及管理合伙人Jussi Salovaara在接受TechCrunch采访时表示，东南亚虽然人才储备尚不足以打造像OpenAI这样的企业，但可以从客户需求出发，开发针对不同行业和市场独特痛点的AI应用程序。Antler的投资包括BorderDollar，一家为跨境物流建立发票融资平台的初创公司，以及CapGo，自动获取市场调研数据的公司。此外，Antler还强调了在建设东南亚食品供应链基础设施解决方案方面的努力，如Seafoody和Zolo。

  


### **IDEA研究院产业化项目SPU获数千万元融资，为AI大模型商业应用提供安全保障**

贡献人：@谭泽琪

https://news.pedaily.cn/202402/529950.shtml

IDEA研究院的产业化项目SPU近日获得了数千万元的融资，由中金资本旗下基金领投。SPU是一家专注于机密计算技术的科技创新企业，旨在为AI大模型和数据要素流通提供安全可信保护方案。该产品通过物理级硬件隔离，确保数据和模型的安全，同时提供高性能计算能力，支持多种计算环境和AI计算框架。SPU已通过多项国家标准和行业标准评测，具备高安全性、高性能和强兼容性，且无需额外迁移成本。本轮融资将加速SPU产品的商业化进程，为AI大模型的商业应用提供安全保障。

  


### 主题｜AI投资指南

贡献人：@谭泽琪

https://mp.weixin.qq.com/s/P5R_Ty4cTA2UEUDMkdo9-Q

2024年，全球AI产业持续发展，OpenAI推出视频生成模型Sora，Gemini 1.5 Pro性能提升，预示AI将加强算力和网络需求，提升AI认知能力，拓展应用空间。算法优化、多模态发展和推理成本降低是重点。AI应用在软件、边缘AI、自动驾驶、机器人等领域有望突破。互联网巨头将受益于AI发展，提升货币化能力和运营效率。投资需关注AI技术进展、政策监管、地缘政治风险等。

  


  


# 学习

### The Bitter Lesson（苦涩的教训）

贡献人：@谭泽琪

http://www.incompleteideas.net/IncIdeas/BitterLesson.html

《苦涩的教训》文章指出，70年的人工智能研究给我们最大的教训是：能够利用计算能力的通用方法最终将更有效。文章强调，不断降低的计算成本（摩尔定律）是这一现象的根本原因。AI研究往往假设可用的计算资源是恒定的，这时利用人类知识是提升性能的唯一方式，但随着时间推移，大量的计算资源将不可避免地变得可用。作者通过棋类游戏、语音识别以及计算机视觉等AI领域内的例子，说明了当研究者依靠大规模计算提出的解决方案在长远来看比倚靠人类知识更为成功。这表明通用方法的强大力量，我们应停止尝试简化复杂的认知内容，而是构建能捕捉这种复杂性的元方法。

  


### LMDeploy-Jetson：在NVIDIA Jetson平台离线部署大模型，流畅运行！开启离线具身智能新纪元

贡献人：@谭泽琪

https://zhuanlan.zhihu.com/p/683255545?utm_campaign=shareopn&utm_medium=social&utm_oi=1209740909321093120&utm_psn=1743744271432155136&utm_source=wechat_timeline&s_r=0

LMDeploy-Jetson是一个工具，它允许在NVIDIA Jetson平台上离线部署大型机器学习模型（LLM），从而实现无需互联网连接的“离线具身智能”。通过W4A16量化技术，模型在推理时的显存占用大幅降低，使得20B参数量的模型在Jetson NX（16G内存）上运行成为可能。这一进展为智能机器人、水下自主机器人（AUV）等领域带来了新的发展机遇，使得这些设备能够在没有网络连接的情况下，直接在本地运行大模型，提高实时性和自主性。

  


### 大语言模型价值观对齐研究与展望

贡献人：@谭泽琪

https://mp.weixin.qq.com/s/S6qhktaQVVN0M_WDEmurQQ

本文探讨了大语言模型（LLM）价值观对齐的重要性和研究进展，强调了确保LLM输出内容与人类价值观一致的必要性。文章介绍了价值观对齐的概念、分类方法和现有技术，包括基于上下文学习的对齐、人在回路的对齐以及多智能体协作对齐。同时，提出了未来研究方向，如多学科交叉合作、价值观数据多样化、增强价值观对齐能力、提高模型可解释性以及多样化的检测评估手段。这些研究将有助于推动LLM在伦理和社会层面的安全发展，确保其为人类带来更多福祉。

  


### 2023年人工智能体(AI Agent)开发与应用全面调研：概念、原理、开发、应用、挑战、展望

贡献人：@谭泽琪

https://mp.weixin.qq.com/s/CVmqZvePPMQgTs3JULyEZQ

2023年，人工智能体（AI Agent）的开发与应用正迅速发展，AI Agent能够自主感知、决策和行动，有望与人类和谐共存。这些智能体通过大型语言模型（LLM）驱动，具备语言交互、决策能力和灵活适配性。AI Agent的应用场景广泛，包括个人助理、多智能体系统、人机合作和专业领域。开发框架如LangChain、AutoGen和PromptAppGPT等提供了创建AI Agent的工具和资源。尽管AI Agent在自主性和安全性方面仍面临挑战，但它们正逐渐成为生产力提升和工作流程变革的关键。随着技术的进步，AI Agent预计将在未来几个月内实现更多突破，重新定义工作和协作方式。

  


### LiRank: LinkedIn在2月新发布的大规模在线排名模型

贡献人：@谭泽琪

https://mp.weixin.qq.com/s/On-vETVUnhFUMfh_lfMQeg

LiRank是LinkedIn发布的大规模在线排名模型，融合了残差DCN、密集门控模块和Transformers等先进技术。该模型通过引入等温校准层和基于深度学习的探索/利用策略，以及压缩技术如量化和词表压缩，实现了高效的部署。LinkedIn在Feed、职位推荐和广告点击率预测中应用LiRank，取得了显著的性能提升。模型采用4D模型并行和Avro张量数据集加载器等优化技术，提高了训练效率。LiRank的增量训练和可扩展性设计，使其在处理大规模数据和频繁更新的推荐系统中表现出色。

  


### 荐书|深度学习的数学工程

https://deeplearningmath.org/

《深度学习的数学工程》是一本即将出版的专业书籍，旨在全面而简洁地介绍深度学习的数学工程。书中不仅涵盖了深度学习的基础知识，还包含了卷积神经网络、循环神经网络、变换器、生成对抗网络、扩散模型、强化学习、图神经网络及行业技巧等内容。主要关注深度学习模型、算法和方法的基本数学描述。全书侧重于数学语言的表述，忽略了计算机代码、神经科学关系、历史视角和理论研究，使得具有数学背景的读者能够快速掌握现代深度学习算法、模型和技术的本质。这本书适合那些具有工程、信号处理、统计、物理、纯数学、计量经济学、运筹学、定量管理、应用机器学习或深度学习背景的专业人士阅读。

  


  


# 声明

本文档仅供学习交流使用，版权归原作者所有，若涉侵权，请联系Jack Jin 15101136166


![c2f7fb308ead0bffd97b25734ceb4d4.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7d9114d1b35a4862ae08fc4ca8c0b696~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=842&h=415&s=43092&e=jpg&b=0982fc)
