# 平台系统类

`Linux` `Windows` `Ascend` `GPU` `CPU` `硬件支持` `初级` `中级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_zh_cn/platform_and_system.md)

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

A：MindSpore当前支持CPU/GPU/Ascend /NPU。目前笔记本电脑或者有GPU的环境，都可以通过Docker镜像来试用。当前MindSpore Model Zoo中有部分模型已经支持GPU的训练和推理，其他模型也在不断地进行完善。在分布式并行训练方面，MindSpore当前支持GPU多卡训练。你可以通过[RoadMap](https://www.mindspore.cn/doc/note/zh-CN/r1.0/roadmap.html)和项目[Release note](https://gitee.com/mindspore/mindspore/blob/r1.0/RELEASE.md#)获取最新信息。

<br/>

<font size=3>**Q：针对异构计算单元的支持，MindSpore有什么计划？**</font>

A：MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

<br/>

<font size=3>**Q：MindSpore与ModelArts是什么关系，在ModelArts中能使用MindSpore吗？**</font>

A：ModelArts是华为公有云线上训练及推理平台，MindSpore是华为深度学习框架，可以查阅[MindSpore官网教程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/use_on_the_cloud.html)，教程中详细展示了用户如何使用ModelArts来做MindSpore的模型训练。

<br/>

<font size=3>**Q：MindSpore是否支持Windows 10？**</font>

A：MindSpore CPU版本已经支持在Windows 10系统中安装，具体安装步骤可以查阅[MindSpore官网教程](https://www.mindspore.cn/install/)。

<br/>

<font size=3>**Q：Ascend硬件平台，在个人的Conda环境中，有时候出现报错RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'，该怎么处理？**</font>

A：出现这种类型的报错，大概率是run包更新后个人的Conda环境中没有更新te或topi或hccl工具包，可以将当前Conda环境中的上述几个工具包卸载，然后使用如下命令再重新安装：`pip install /usr/local/Ascend/fwkacllib/lib64/{te/topi/hccl}*any.whl`。