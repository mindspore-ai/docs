# 平台系统类

`Linux` `Windows` `Ascend` `GPU` `CPU` `硬件支持` `初级` `中级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_zh_cn/platform_and_system.md)

<font size=3>**Q:PyNative模式和Graph模式的区别？**</font>

A: 在使用效率上，两个模式使用的算子是一致的，因此相同的网络和算子，分别在两个模式下执行时，精度效果是一致的。由于执行机理的差异，网络的执行性能是会不同的，并且在理论上，MindSpore提供的算子同时支持PyNative模式和Graph模式；

在场景使用方面，Graph模式需要一开始就构建好网络结构，然后框架做整图优化和执行，对于网络固定没有变化，且需要高性能的场景比较适合；

在不同硬件（`Ascend`、`GPU`和`CPU`）资源上都支持这两种模式；

代码调试方面，由于是逐行执行算子，因此用户可以直接调试Python代码，在代码中任意位置打断点查看对应算子`/api`的输出或执行结果。而Graph模式由于在构造函数里只是完成网络构造，实际没有执行，因此在`construct`函数里打断点是无法获取对应算子的输出，而只能等整网执行中指定对应算子的输出打印，在网络执行完成后进行查看。

<br/>

<font size=3>**Q:使用PyNative模式能够进行迁移学习？**</font>

A: PyNative模式是兼容迁移学习的，更多的教程信息，可以参考[预训练模型加载代码详解](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/cv_mobilenetv2_fine_tune.html#id7)。

<br/>

<font size=3>**Q:MindSpore只能在华为自己的`NPU`上跑么？**</font>

A: MindSpore同时支持华为自己的`Ascend NPU`、`GPU`与`CPU`，是支持异构算力的。

<br/>

<font size=3>**Q：MindSpore在Ascend 310上是否可以转AIR模型？**</font>

A：Ascend 310不能导出AIR，需要在Ascend 910加载训练好的checkpoint后,导出AIR，然后在Ascend 310转成OM模型进行推理。Ascend 910的安装方法可以参考官网MindSpore[安装指南](https://www.mindspore.cn/install)。

<br/>

<font size=3>**Q：我用MindSpore在GPU上训练的网络脚本可以不做修改直接在NPU上进行训练么？**</font>

A：可以的，MindSpore面向NPU/GPU/CPU提供统一的API，在算子支持的前提下，网络脚本可以不做修改直接跨平台运行。

<br/>

<font size=3>**Q：Ascend 310 不能安装MindSpore么？**</font>

A：Ascend 310只能用作推理，MindSpore支持在Ascend 910训练，训练出的模型转化为OM模型可用于Ascend 310上推理。

<br/>

<font size=3>**Q：安装运行MindSpore时，是否要求平台有GPU、NPU等计算单元？需要什么硬件支持？**</font>

A：MindSpore当前支持CPU/GPU/Ascend /NPU。目前笔记本电脑或者有GPU的环境，都可以通过Docker镜像来试用。当前MindSpore Model Zoo中有部分模型已经支持GPU的训练和推理，其他模型也在不断地进行完善。在分布式并行训练方面，MindSpore当前支持GPU多卡训练。你可以通过[RoadMap](https://www.mindspore.cn/doc/note/zh-CN/r1.1/roadmap.html)和项目[Release note](https://gitee.com/mindspore/mindspore/blob/r1.1/RELEASE.md#)获取最新信息。

<br/>

<font size=3>**Q：针对异构计算单元的支持，MindSpore有什么计划？**</font>

A：MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

<br/>

<font size=3>**Q：MindSpore与ModelArts是什么关系，在ModelArts中能使用MindSpore吗？**</font>

A：ModelArts是华为公有云线上训练及推理平台，MindSpore是华为深度学习框架，可以查阅[MindSpore官网教程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/use_on_the_cloud.html)，教程中详细展示了用户如何使用ModelArts来做MindSpore的模型训练。

<br/>

<font size=3>**Q：MindSpore是否支持Windows 10？**</font>

A：MindSpore CPU版本已经支持在Windows 10系统中安装，具体安装步骤可以查阅[MindSpore官网教程](https://www.mindspore.cn/install/)。

<br/>

<font size=3>**Q：Ascend硬件平台，在个人的Conda环境中，有时候出现报错RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'，该怎么处理？**</font>

A：出现这种类型的报错，大概率是run包更新后个人的Conda环境中没有更新te或topi或hccl工具包，可以将当前Conda环境中的上述几个工具包卸载，然后使用如下命令再重新安装：`pip install /usr/local/Ascend/fwkacllib/lib64/{te/topi/hccl}*any.whl`。