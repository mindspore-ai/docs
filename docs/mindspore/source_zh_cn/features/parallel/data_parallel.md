# 数据并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/features/parallel/data_parallel.md)

## 概述

数据并行是最常用的并行训练方式，用于加速模型训练和处理大规模数据集。在数据并行模式下，训练数据被划分成多份，然后将每份数据分配到不同的计算节点上，例如多卡或者多台设备。每个节点独立地处理自己的数据子集，并使用相同的模型进行前向传播和反向传播，最终对所有节点的梯度同步后，进行模型参数更新。

> 数据并行支持的硬件平台包括Ascend、GPU和CPU，此外还同时支持PyNative模式和Graph模式。

相关接口：

1. [mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_auto_parallel_context.html)：设置数据并行模式。
2. [mindspore.nn.DistributedGradReducer()](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.DistributedGradReducer.html)：进行多卡梯度聚合。

## 整体流程

![整体流程](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/data_parallel.png)

1. 环境依赖

    每次开始进行并行训练前，通过调用[mindspore.communication.init](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.init.html)接口初始化通信资源，并自动创建全局通信组`WORLD_COMM_GROUP`。通信组能让通信算子在卡间和机器间进行信息收发，全局通信组是最大的一个通信组，包括了当前训练的所有设备。通过调用`mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)`设置当前模式为数据并行模式。

2. 数据分发（Data distribution）

    数据并行的核心在于将数据集在样本维度拆分并下发到不同的卡上。在[mindspore.dataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html)模块提供的所有数据集加载接口中都有`num_shards`和`shard_id`两个参数，它们用于将数据集拆分为多份并循环采样的方式，采集`batch`大小的数据到各自的卡上，当出现数据量不足的情况时将会从头开始采样。

3. 网络构图

    数据并行网络的书写方式与单卡网络没有差别，这是因为在正反向传播（Forward propagation & Backward propagation）过程中各卡的模型间是独立执行的，只是保持了相同的网络结构。唯一需要特别注意的是为了保证各卡间训练同步，相应的网络参数初始化值应当是一致的，在`DATA_PARALLEL`模式下可以通过[mindspore.set_seed](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_seed.html)接口来设置seed或通过使能`mindspore.set_auto_parallel_context`中的`parameter_broadcast`达到多卡间权重初始化一致的目的。

4. 梯度聚合（Gradient aggregation）

    数据并行理论上应该实现和单卡一致的训练效果，为了保证计算逻辑的一致性，通过调用`mindspore.nn.DistributedGradReducer()`接口，在梯度计算完成后自动插入`AllReduce`算子实现各卡间的梯度聚合操作。`DistributedGradReducer()`接口中提供了`mean`开关，用户可以选择是否要对求和后的梯度值进行求平均操作，也可以将其视为超参项。

5. 参数更新（Parameter update）

    因为引入了梯度聚合操作，所以各卡的模型会以相同的梯度值一起进入参数更新步骤。
