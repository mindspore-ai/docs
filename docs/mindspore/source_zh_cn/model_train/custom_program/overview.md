# 自定义高阶编程概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_zh_cn/model_train/custom_program/overview.md)

在训练过程中，当框架提供的高级方法不能满足开发者的某些场景，或开发者对性能有较高要求时，可以采用自定义的方法添加或修改某些流程，以满足开发或调试需求。

当前MindSpore提供一些自定义高阶编程的方式，您可以参考以下指引：

## 自定义算子

当开发网络遇到内置算子不足以满足需求时，可以利用MindSpore的Python API中的 [Custom](https://www.mindspore.cn/docs/zh-CN/r2.4.1/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语方便快捷地进行不同类型自定义算子的定义和使用。

## 自定义神经网络层

通常情况下，MindSpore提供的神经网络层接口和function函数接口能够满足模型构造需求，但由于AI领域不断推陈出新，因此有可能遇到新网络结构没有内置模块的情况。

此时我们可以根据需要，通过MindSpore提供的function接口、Primitive算子自定义神经网络层，并可以使用`Cell.bprop`方法自定义反向。

## 自定义参数初始化

MindSpore提供了多种网络参数初始化的方式，并在部分算子中封装了参数初始化的功能，主要方法如下：

- **Initializer初始化**：Initializer是MindSpore内置的参数初始化基类，所有内置参数初始化方法均继承该类。
- **字符串初始化**：MindSpore也提供了参数初始化简易方法，即使用参数初始化方法名称的字符串。此方法使用Initializer的默认参数进行初始化。
- **自定义参数初始化**：在遇到需要自定义的参数初始化方法时，可以继承Initializer自定义参数初始化方法。
- **Cell遍历初始化**：先构造网络并实例化，然后对Cell进行遍历，并对参数进行赋值。

## 自定义损失函数

`mindspore.nn`模块中提供了许多通用损失函数，当这些通用损失函数无法满足所有需求，用户可以自定义所需的损失函数。

## 自定义优化器

MindSpore中的nn模块提供了常用的优化器，如nn.SGD、nn.Adam、nn.Momentum等，当这些优化器无法满足开发需求时，用户可以自定义优化器。

## Hook编程

为了方便用户准确、快速地对深度学习网络进行调试，MindSpore在动态图模式下设计了Hook功能，使用Hook功能可以捕获中间层算子的输入、输出数据以及反向梯度。
