# 概述

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/overview.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

本迁移指导包含以PyTorch为主的其他机器学习框架将神经网络迁移到MindSpore的完整步骤。

基本的一个迁移流程如下图所示：

1. 需要配置MindSpore的开发环境；
2. 需要对需要迁移的网络模型进行分析以及基本数据的获取；
3. MindSpore的复现。在模型功能调试阶段推荐使用PYNATIVE模式调试，功能调试完成之后转GRAPH模式；模型开发完成之后，推荐先进行推理流程的复现，后进行训练流程的复现；
4. 功能、精度、性能三个方面的调试调优。

在这个过程中，我们对每个环节需要做些什么，怎么去做有一个相对完整的描述，希望通过迁移指南，开发者可以快速的将其他框架已有代码快速迁移到MindSpore。

![flowchart](images/flowchart.PNG "迁移流程")

## [环境准备&资料获取](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/enveriment_preparation.html)

网络迁移首先需要配置MindSpore的开发环境，本章会详细描述安装过程与知识准备。知识准备包括对MindSpore组件models和Hub的基本介绍，包含用途、场景与使用方法。此外还有云上训练的相关教程：使用ModelArts适配脚本，在OBS上传数据集，然后进行线上训练等。

## [模型分析与准备](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/analysis_and_preparation.html)

在做正式的开发前，需要对要迁移的网络/算法做一些分析准备工作。包括：

- 读论文及参考代码，了解算法及网络结构；
- 复现论文结果，获取基础模型（ckpt），基准精度及性能；
- 分析网络中使用的API及功能。

在PyTorch往MindSpore进行网络迁移时，需要注意[与PyTorch典型接口区别](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/typical_api_comparision.html)。

## [MindSpore模型实现](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/model_development.html)

在完成前期的分析准备工作后，就可以使用MindSpore对新网络进行开发了。本章节将会从推理和训练需要的基本模块出发，介绍MindSpore网络构建的相关知识以及训练、推理的流程，并用一两个实际的案例来说明在特殊场景如何构建网络。

## [调试调优](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/debug_and_tune.html)

本章节会从功能，精度，性能三个方面介绍一些调试调优的方法。

## [网络迁移调试实例](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/sample_code.html)

本章包含了一个完整的网络迁移样例。从对标网络的分析与复现开始，详细说明脚本开发与精度调试调优等步骤，最后列出了迁移过程中的常见问题与相应优化方法、框架性能问题等等。

## [常见问题](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/faq.html)

在这里会列出一些常见问题与相应解决方法。
