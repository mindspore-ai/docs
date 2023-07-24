# 特性支持类

`特性优势` `端侧推理` `功能模块` `推理工具`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_zh_cn/supported_features.md)

<font size=3>**Q：MindSpore serving是否支持热加载，避免推理服务中断？**</font>

A：很抱歉，MindSpore当前还不支持热加载，需要重启。建议您可以跑多个Serving服务，切换版本时，重启部分。

<br/>

<font size=3>**Q：请问MindSpore支持梯度截断吗？**</font>

A：支持，可以参考[梯度截断的定义和使用](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/transformer/src/transformer_for_train.py#L35)。

<br/>

<font size=3>**Q：如何在训练神经网络过程中对计算损失的超参数进行改变？**</font>

A：您好，很抱歉暂时还未有这样的功能。目前只能通过训练-->重新定义优化器-->训练，这样的过程寻找较优的超参数。

<br/>

<font size=3>**Q：第一次看到有专门的数据处理框架，能介绍下么？**</font>

A：MindData提供数据处理异构硬件加速功能，高并发数据处理`pipeline`同时支持`NPU/GPU/CPU`，`CPU`占用降低30%，点击查询[优化数据处理](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/optimize_data_processing.html)。

<br/>

<font size=3>**Q：MindSpore的IR设计理念是什么？**</font>

A：函数式：一切皆函数，易于微分实现；无副作用，易于实现自动并行化分析；`JIT`编译能力：图形IR，控制流依赖和数据流合一，平衡通用性/易用性；图灵完备的IR：更多的转换`Python`灵活语法，包括递归等。

<br/>

<font size=3>**Q：MindSpore会出强化学习框架么？**</font>

A：您好，感谢您关注MindSpore。当前还暂不支持，但正在处于设计阶段，欢迎您贡献想法和场景，参与建设，谢谢。

<br/>

<font size=3>**Q：谷歌Colab、百度AI Studio都有免费`GPU`算力提供,MindSore有免费算力提供么？**</font>

A：当前如果与MindSpore展开论文、科研合作是可以获得免费云算力支持的。如果只是想简单试用，也提供类似Colab的在线体验

<br/>

<font size=3>**Q：MindSpore Lite的离线模型MS文件如何进行可视化，看到网络结构？**</font>

A：MindSpore Lite正在往开源仓库`netron`上提交代码，后面MS模型会首先使用`netron`实现可视化。现在上`netron`开源仓还有一些问题需要解决，不过我们有内部使用的`netron`版本，可以在[`netron`版本发布](https://github.com/lutzroeder/netron/releases)里下载到。

<br/>

<font size=3>**Q：MindSpore有量化推理工具么？**</font>

A：[MindSpore Lite](https://www.mindspore.cn/lite)支持云侧量化感知训练的量化模型的推理，MindSpore Lite converter工具提供训练后量化以及权重量化功能，且功能在持续加强完善中。

<br/>

<font size=3>**Q：MindSpore并行模型训练的优势和特色有哪些？**</font>

A：MindSpore分布式训练除了支持数据并行，还支持算子级模型并行，可以对算子输入tensor进行切分并行。在此基础上支持自动并行，用户只需要写单卡脚本，就能自动切分到多个节点并行执行。

<br/>

<font size=3>**Q：请问MindSpore实现了反池化操作了吗？类似于`nn.MaxUnpool2d` 这个反池化操作？**</font>

A：目前 MindSpore 还没有反池化相关的接口。如果用户想自己实现的话，可以通过自定义算子的方式自行开发算子，详情请见[自定义算子](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/custom_operator.html)。

<br/>

<font size=3>**Q：MindSpore有轻量的端侧推理引擎么？**</font>

A：MindSpore轻量化推理框架MindSpore Lite已于r0.7版本正式上线，欢迎试用并提出宝贵意见，概述、教程和文档等请参考[MindSpore Lite](https://www.mindspore.cn/lite)

<br/>

<font size=3>**Q：MindSpore在语义协同和处理上是如何实现的？是否利用当前学术界流行的FCA理论？**</font>

A：MindSpore框架本身并不需要支持FCA。对于语义类模型，用户可以调用第三方的工具在数据预处理阶段做FCA数学分析。MindSpore本身支持Python语言，`import FCA`相关包即可使用。

<br/>

<font size=3>**Q：当前在云上MindSpore的训练和推理功能是比较完备的，至于边端场景（尤其是终端设备）MindSpore有什么计划？**</font>

A：MindSpore是端边云统一的训练和推理框架，支持将云侧训练的模型导出到Ascend AI处理器和终端设备进行推理。当前推理阶段支持的优化包括量化、算子融合、内存复用等。

<br/>

<font size=3>**Q：MindSpore自动并行支持情况如何？**</font>

A：自动并行特性对CPU GPU的支持还在完善中。推荐用户在Ascend 910 AI处理器上使用自动并行，可以关注开源社区，申请MindSpore开发者体验环境进行试用。

<br/>

<font size=3>**Q：MindSpore有没有类似基于TensorFlow实现的对象检测算法的模块？**</font>

A：TensorFlow的对象检测Pipeline接口属于TensorFlow Model模块。待MindSpore检测类模型完备后，会提供类似的Pipeline接口。

<br/>

<font size=3>**Q：其他框架的脚本或者模型怎么迁移到MindSpore？**</font>

A：关于脚本或者模型迁移，可以查询MindSpore官网中关于[网络迁移](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/migrate_3rd_scripts.html)的介绍。

<br/>

<font size=3>**Q：MindSpore是否附带开源电商类数据集？**</font>

A：暂时还没有，可以持续关注[MindSpore官网](https://www.mindspore.cn)。

<br/>

<font size=3>**Q：能否使用第三方库numpy array封装MindSpore的Tensor数据？**</font>

A：不能，可能出现各种问题。例如：`numpy.array(Tensor(1)).astype(numpy.float32)`的报错信息为"ValueError: settinng an array element with a sequence."。
