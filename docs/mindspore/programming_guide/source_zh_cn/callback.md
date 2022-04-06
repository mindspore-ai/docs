# Callback机制

`Ascend` `GPU` `CPU` `模型开发`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/callback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

Callback回调函数在MindSpore中被实现为一个类，Callback机制类似于一种监控模式，可以帮助用户观察网络训练过程中各种参数的变化情况和网络内部的状态，还可以根据用户的指定，在达到特定条件后执行相应的操作，在训练过程中，Callback列表会按照定义的顺序执行Callback函数。Callback机制让用户可以及时有效地掌握网络模型的训练状态，并根据需要随时作出调整，可以极大地提升用户的开发效率。

在MindSpore中，Callback机制一般用在网络训练过程`model.train`中，用户可以通过配置不同的内置回调函数传入不同的参数，从而实现各种功能。例如，可以通过`LossMonitor`监控每一个epoch的loss变化情况，通过`ModelCheckpoint`保存网络参数和模型进行再训练或推理，通过`TimeMonitor`监控每一个epoch，每一个step的训练时间，以及提前终止训练，动态调整参数等。

## MindSpore内置回调函数

- ModelCheckpoint

    与模型训练过程相结合，保存训练后的模型和网络参数，方便进行再推理或再训练。`ModelCheckpoint`一般与`CheckpointConfig`配合使用，`CheckpointConfig`是一个参数配置类，可自定义配置checkpoint的保存策略。

    详细内容，请参考[Checkpoint官网教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html)。

- SummaryCollector

    帮助收集一些常见信息，如loss、learning rate、计算图、参数权重等，方便用户将训练过程可视化和查看信息，并且可以允许summary操作从summary文件中收集数据。

    详细内容，请参考[Summary官网教程](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/summary_record.html)。

- LossMonitor

    监控训练过程中的loss变化情况，当loss为NAN或INF时，提前终止训练。可以在日志中输出loss，方便用户查看。

    详细内容，请参考[LossMonitor官网教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#mindsporecallback)。

- TimeMonitor

    监控训练过程中每个epoch，每个step的运行时间。

## MindSpore自定义回调函数

MindSpore不但有功能强大的内置回调函数，还可以支持用户自定义回调函数。当用户有自己的特殊需求时，可以基于Callback基类，自定义满足用户自身需求的回调函数。Callback可以把训练过程中的重要信息记录下来，通过一个字典类型变量cb_params传递给Callback对象， 用户可以在各个自定义的Callback中获取到相关属性，执行自定义操作。

以下面两个场景为例，介绍自定义Callback回调函数的功能：

1. 实现在规定时间内终止训练，用户可以设定时间阈值，当训练时间达到这个阈值后就终止训练过程。

2. 实现保存训练过程中精度最高的checkpoint文件，用户可以自定义在每一轮迭代后都保存当前精度最高的模型。

详细内容，请参考[自定义Callback官网教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#自定义callback)。

根据教程，用户可以很容易实现具有其他功能的自定义回调函数，如实现在每一轮训练结束后都输出相应的详细训练信息，包括训练进度、训练轮次、训练名称、loss值等；如实现在loss或模型精度达到一定值后停止训练，用户可以设定loss或模型精度的阈值，当loss或模型精度达到该阈值后就提前终止训练等。
