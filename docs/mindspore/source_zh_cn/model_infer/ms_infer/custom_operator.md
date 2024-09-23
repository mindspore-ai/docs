# 自定义算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_infer/ms_infer/custom_operator.md)

大语言模型推理通常会采用多种优化技术，包括但不限于量化和KVCache，旨在提高模型的运行效率，同时减少所需的计算资源。除了这些通用优化技术，用户还可以根据具体的应用场景和需求，对模型结构进行定制化的改造。这种改造可能涉及到模型层的增减、连接方式的调整，甚至是算子级别的优化，以实现更高效的数据处理和更快速的推理响应。这样的定制化改造使得模型能够更好地适应特定的任务和运行环境，但相应地也会提高模型的复杂性。

因此我们提供相应接口使用户可以开发自定义算子，并将其接入到MindSpore框架中。自定义算子可以实现针对性的性能优化，比如通过算子融合技术，将多个操作合并为一个更高效的操作，减少I/O及下发耗时，并提升算子执行性能，实现对大语言模型推理性能的深度优化。

用户可以参考[自定义算子教程](https://www.mindspore.cn/tutorials/experts/zh-CN/master/operation/op_custom_ascendc.html)，了解如何开发自定义算子，以及如何将它们有效地集成到MindSpore框架中。
