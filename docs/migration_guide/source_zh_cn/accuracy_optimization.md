# 模型精度调优

<!-- TOC -->

- [模型精度调优](#模型精度调优)
    - [MindSpore的实战调优](#mindspore的实战调优)
    - [参考文档](#参考文档)
        - [可视化工具](#可视化工具)
        - [数据问题处理](#数据问题处理)
        - [超参问题处理](#超参问题处理)
        - [模型结构问题处理](#模型结构问题处理)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/migration_guide/source_zh_cn/accuracy_optimization.md" target="_blank"><img src="./_static/logo_source.png"></a>

模型训练的最终结果是为了得到一个精度达标的模型，而在AI训练过程中有时会遇到loss（模型损失值）无法下降，或者发散，metrics（模型度量指标）达不到预期等，造成无法得到一个理想精度的模型，这时候需要去进行分析训练过程中出现了什么样的问题，针对性地采用包括调整数据、调整超参、重构模型结构等方法，去解决模型精度调优过程中遇到的各种问题。

本文介绍MindSpore团队总结的精度调优的方法，及解决精度调优过程中问题的分析思路，并且将MindSpore中用于精度调优的工具做分类介绍。

## MindSpore的实战调优

MindSpore团队总结了在AI训练过程中，遇到的造成模型精度无法达标的常见原因，针对这些原因的调优思路，同时使用可视化的调优辅助工具，使得精度调优的过程更容易理解，并将其分享到下面几篇文档中。

[MindSpore模型精度调优实战（一）](https://www.mindspore.cn/news/newschildren?id=381)

[MindSpore模型精度调优实战（二）](https://www.mindspore.cn/news/newschildren?id=394)

## 参考文档

### 可视化工具

训练过程中进行可视化数据采集时，可参考资料[收集Summary数据](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/summary_record.html)。

训练过程中进行可视化数据分析时，可参考资料[训练看板](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/dashboard.html)和[溯源和对比看板](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/lineage_and_scalars_comparision.html)。

### 数据问题处理

对数据进行标准化、归一化、通道转换等操作，在图片数据处理上，增加随机视野图片，随机旋转度图片等，另外数据混洗、batch和数据倍增等操作，可参考[数据处理](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/pipeline.html)、[数据增强](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/augmentation.html)和[自动数据增强](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/auto_augmentation.html)。

### 超参问题处理

AI训练中的超参包含全局学习率，epoch和batch等，如果需要在不同的超参下，训练过程进行可视化时，可参考资料：[可视化的超参调优](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/hyper_parameters_auto_tuning.html)；如果需要设置动态学习率超参时，可参考资料：[学习率的优化算法](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/optim.html?#id3)。

### 模型结构问题处理

一般的处理模型结构问题，需要庸到的操作有：模型结构的重构，选择合适的优化器或者损失函数等。

需要重构模型结构时，可参考资料：[Cell构建及其子类](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/cell.html)。

选择合适的损失函数，可参考资料：[损失函数算子支持列表](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.nn.html#loss-functions)。

选择合适的优化器时，可参考资料：[优化器算子支持列表](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.nn.html#optimizer-functions)。
