# 策略选择

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/parallel/strategy_select.md)

## 概述

在分布式模型训练中，针对不同的模型规模和数据量大小，可以选择不同的并行策略来提高训练效率和资源利用率。以下是不同并行策略的解释和适用情况：

1. [数据并行](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/data_parallel.html)：数据并行是指在训练过程中，将不同的训练样本分布到不同的设备上，每个设备计算其分配的样本的梯度。然后通过梯度的平均或累加来更新模型的参数。数据并行适用于数据量较大，而模型参数量较少，可以在单个设备上加载的情况。数据并行能够充分利用多个设备的计算能力，加速训练过程。

2. [算子级并行](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/operator_parallel.html)：算子级并行是指以算子为单位，把输入张量和模型参数切分到多台设备上进行计算，每个设备负责计算模型的一部分，提升整体速度。算子级并行又分为需要手动配置切分策略的半自动并行模式以及只需配置少部分甚至无需配置切分策略的自动并行模式。算子级并行适用于模型架构较大，无法完全载入单个设备内存的情况。

3. [优化器并行](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/optimizer_parallel.html)：优化器并行通过将优化器的计算量分散到数据并行维度的卡上，在大规模网络上（比如LLAMA、DeepSeek）可以有效减少内存消耗并提升网络性能，推荐并行训练时开启。

4. [流水线并行](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/pipeline_parallel.html)：流水线并行将整个训练过程分成多个阶段，每个阶段的计算在不同的设备上进行。数据在不同阶段之间流动，类似于流水线。这种策略适用于网络模型较大，单卡无法载入，且网络可以较为平均地分为多个阶段的计算，并且每个阶段的计算时间较长，从而可以最大限度地重叠计算和通信。

选择适当的并行策略取决于具体的训练任务和资源配置。通常情况下，可以根据以下指导原则进行选择：

- 数据集非常大，而模型可以加载到单个设备的情况，推荐**数据并行**。
- 模型较大，无法载入单个设备内存，且用户对网络中核心算子计算负载具有一定的了解，推荐**算子级并行**。
- 模型较大，希望减少内存消耗以加载更大的模型，推荐**优化器并行**。
- 模型较大，但是模型可以较为均衡地分为多个阶段，每个阶段计算时间较长，推荐**流水线并行**。

在实际应用中，您可以结合多种并行策略来达到最佳的训练效果和资源利用率。
