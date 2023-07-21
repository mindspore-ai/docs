# 定义网络

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_zh_cn/use/defining_the_network.md)

由多个层组成的神经网络模型，是训练过程的重要组成部分。你可以基于MindSpore中的`nn.Cell`基类，通过初始化`__init__`方法和构造`construct`方法构建网络模型。定义网络模型有以下几种方式：

- 直接使用官方提供的典型网络模型。

  建议通过查阅当前MindSpore提供的[网络支持列表](https://www.mindspore.cn/doc/note/zh-CN/r1.1/network_list_ms.html)，直接使用相应的网络模型。在网络支持列表中，提供了每个网络所支持的平台，直接点击相应网络名称查看网络的定义，用户可根据需求自定义网络初始化参数。

- 自行构建网络。

    - 若网络中的内置算子不足以满足需求时，你可以利用MindSpore方便快捷地自定义算子并加入到网络中。

      通过[自定义算子](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/custom_operator.html)了解详细帮助信息。

    - MindSpore提供了迁移第三方训练框架的脚本，支持将已有的TensorFlow、PyTorch等的网络迁移到MindSpore，帮助你快速进行网络迁移。

      通过[迁移第三方框架训练脚本](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/migrate_script.html)了解详细帮助信息。

    - MindSpore支持使用开发深度学习模型的逻辑进行概率编程，还提供深度概率学习的工具箱，构建贝叶斯神经网络。

      通过[深度概率编程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/apply_deep_probability_programming.html)了解详细帮助信息。
