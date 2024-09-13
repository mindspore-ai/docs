# 训练流程概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/train_process/overview.md)

随着人工智能技术的快速发展，深度学习已经成为了许多领域的核心技术。深度学习模型训练是深度学习的重要组成部分，它涉及到多个阶段和流程。

一个完整的训练流程一般包含四步，包括数据集预处理，模型创建，定义损失函数和优化器，训练及保存模型；通常情况下，直接定义训练和评估网络直接运行已经可以满足基本需求，但是在使用MindSpore时，可以使用 `Model` 对网络进行封装，以此在一定程度上简化代码，并且可以简易的使用数据下沉，混合精度等高阶功能。

要使用MindSpore搭建一个完整的训练流程，您可以参考以下指引：

## 高阶封装

对于简单场景的神经网络，可以在定义 `Model` 时指定前向网络 `network` 、损失函数 `loss_fn` 、优化器 `optimizer` 和评价函数 `metrics` 。

使用`Model` 搭建一个神经网络一般分为如下四步：

- **数据集预处理**：使用 ``mindspore.dataset`` 加载数据集后对数据集进行缩放，归一化，格式转换等操作。参考[数据集预处理](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/dataset.html)教程了解更多信息。
- **模型创建**：使用 ``mindspore.nn.Cell`` 构建神经网络，在\ ``__init__``\内初始化神经网络层，在 ``construct`` 函数内构建神经网络正向执行逻辑。参考[模型创建](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/model.html)教程了解更多信息。
- **定义损失函数和优化器**：使用 ``mindspore.nn`` 定义损失函数和优化器函数。参考[定义损失函数和优化器](https://www.mindspore.cn/docs/zh-CN/master/model_train/train_process/model.html#%E5%AE%9A%E4%B9%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%92%8C%E4%BC%98%E5%8C%96%E5%99%A8)教程了解更多信息。
- **训练及保存模型**：使用 ``ModelCheckpoint`` 接口保存网络模型和参数，使用 ``model.fit`` 进行网络的训练和评估，使用 ``LossMonitor`` 监控训练过程中 ``loss`` 的变化。参考[训练及保存模型](https://www.mindspore.cn/docs/zh-CN/master/model_train/train_process/model.html#%E8%AE%AD%E7%BB%83%E5%8F%8A%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B)教程了解更多信息。

此外，在深度学习训练过程中，我们可以使用回调机制（Callback）来及时掌握网络模型的训练状态、实时观察网络模型各参数的变化情况和实现训练过程中用户自定义的一些操作。除了MindSpore内置的大量回调函数之外，也可以基于 ``Callback`` 基类来自定义回调类。

当训练任务结束，常常需要评价函数（Metrics）来评估模型的好坏。同样的，MindSpore提供了大部分常见任务的评价函数，用户也可以通过继承 ``mindspore.train.Metric`` 基类来自定义Metrics函数。

## 性能优化方式

在MindSpore深度学习训练流程中，我们通常有以下三种手段可以对训练性能进行优化：

- **下沉模式**：MindSpore提供了数据图下沉、图下沉和循环下沉功能，可以极大地减少Host-Device交互开销，有效地提升训练与推理的性能。参考[下沉模式](https://www.mindspore.cn/docs/zh-CN/master/model_train/train_process/optimize/sink_mode.html)教程了解更多信息。
- **向量化加速接口Vmap**：``vmap`` 接口可以将模型或函数中高度重复的运算逻辑转换为并行的向量运算逻辑，从而获得更加精简的代码逻辑以及更高效的执行性能。参考[向量化加速接口Vmap](https://www.mindspore.cn/docs/zh-CN/master/model_train/train_process/optimize/vmap.html)教程了解更多信息。

## 高级微分接口

MindSpore可以使用 ``ops.grad`` 或者 ``ops.value_and_grad`` 接口进行自动微分，对待求导函数或者网络进行自动一阶求导或者高阶求导。

## 高阶训练策略

MindSpore可以执行一些高阶训练策略，例如：

- **梯度累加**：梯度累加是一种将训练神经网络的数据样本按Batch size拆分为几个小Batch的方式，然后按顺序进行计算。目的是为了解决由于内存不足，导致Batch size过大神经网络无法训练，或者网络模型过大无法加载的OOM（Out Of Memory）问题。参考[梯度累加](https://www.mindspore.cn/docs/zh-CN/master/model_train/train_process/optimize/gradient_accumulation.html)教程了解更多信息。
- **Per-sample-gradients**：per-sample-gradients可以帮助我们在训练神经网络时，更准确地计算每个样本对网络参数的影响，从而更好地提高模型的训练效果，MindSpore提供了高效的方法来计算per-sample-gradients。参考[Per-sample-gradients](https://www.mindspore.cn/docs/zh-CN/master/model_train/train_process/optimize/per_sample_gradients.html)教程了解更多信息。
