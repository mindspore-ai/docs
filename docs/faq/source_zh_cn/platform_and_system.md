# 平台系统类

`Linux` `Windows` `Ascend` `GPU` `CPU` `硬件支持` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_zh_cn/platform_and_system.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q：Ascend 310 不能安装MindSpore么？

A：Ascend 310只能用作推理，MindSpore支持在Ascend 910训练，训练出的模型转化为OM模型可用于Ascend 310上推理。

<br/>

Q：安装运行MindSpore时，是否要求平台有GPU、NPU等计算单元？需要什么硬件支持？

A：MindSpore当前支持CPU/GPU/Ascend /NPU。目前笔记本电脑或者有GPU的环境，都可以通过Docker镜像来试用。当前MindSpore Model Zoo中有部分模型已经支持GPU的训练和推理，其他模型也在不断地进行完善。在分布式并行训练方面，MindSpore当前支持GPU多卡训练。你可以通过[RoadMap](https://www.mindspore.cn/doc/note/zh-CN/r1.0/roadmap.html)和项目[Release note](https://gitee.com/mindspore/mindspore/blob/r1.0/RELEASE.md#)获取最新信息。

<br/>

Q：针对异构计算单元的支持，MindSpore有什么计划？

A：MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

<br/>

Q：MindSpore与ModelArts是什么关系，在ModelArts中能使用MindSpore吗？

A：ModelArts是华为公有云线上训练及推理平台，MindSpore是华为深度学习框架，可以查阅[MindSpore官网教程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/use_on_the_cloud.html)，教程中详细展示了用户如何使用ModelArts来做MindSpore的模型训练。

<br/>

Q：MindSpore是否支持Windows 10？

A：MindSpore CPU版本已经支持在Windows 10系统中安装，具体安装步骤可以查阅[MindSpore官网教程](https://www.mindspore.cn/install/)。