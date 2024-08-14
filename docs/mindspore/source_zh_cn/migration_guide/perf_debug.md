# 性能调优

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/perf_debug.md)

## 调优常见问题及解决办法

- 性能调试阶段，可能会遇到以下常见问题：
    - 第一个step耗时长
         这个阶段主要完成图转换、图融合、图优化等操作，是生成可执行模型的过程，可参考[如何优化编译性能](https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/static_graph_syntax/static_graph_expert_programming.html#%E5%A6%82%E4%BD%95%E4%BC%98%E5%8C%96%E7%BC%96%E8%AF%91%E6%80%A7%E8%83%BD)。
    - 迭代间隙耗时长
         这个阶段的耗时大部分来源于数据获取，可参考[数据处理性能优化](https://www.mindspore.cn/docs/zh-CN/master/model_train/dataset/optimize.html)。
    - 前反向计算耗时长
         这个阶段主要执行网络中的前向及反向算子，承载了一个迭代的主要计算工作。可通过[Profiler](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling.html)将训练过程中的算子耗时等信息记录到文件中。该性能数据提供框架的host执行、以及算子执行的性能数据，也可通过[MindInsight](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/index.html)可视化界面供用户查看分析，帮助用户更高效地调试神经网络性能。
    - 迭代拖尾耗时长
         这个阶段耗时长可能是集合通信耗时长，可设置融合策略进行优化，可参考[all_reduce_fusion_config设置allreduce融合策略](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_auto_parallel_context.html)。

## 性能调优过程

首先需要做性能数据获取，具体的获取方式见[性能调试（Ascend）](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html)、[性能调试（GPU）](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_gpu.html)。

性能优化方向主要包含：

1. 算子性能优化
2. 框架使能性能优化
3. 多机同步性能优化
4. 数据处理性能优化

可以参考[resnet网络迁移](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/sample_code.html)串通整个过程。

> 有的网络很大，这种情况在图模式下编译会很慢。在性能调优过程请区分图编译和网络执行，本节主要介绍网络执行阶段的性能调优策略。

### 算子性能优化

单算子耗时久、对于同一种算子在不同shape或者不同 datatype 下性能差异较大的情况主要是由算子性能问题引起，通常有以下解决思路：

1. 使用计算量更小的数据类型。例如，同一个算子在 float16 和 float32 下精度无明显差别，可使用计算量更小的 float16 格式。
2. 使用算法相同的其他算子规避。
3. Ascend环境上注意16对齐。由于昇腾芯片的设计，在AICore上的计算最好是16对齐的(shape中的每一维都是16的倍数)。

如果您发现有性能较差的算子时，建议联系 [MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈，我们确认为性能问题后会及时优化。

### 框架使能性能优化

- 使用静态图模式

  MindSpore一般在静态图模式下比PYNATIVE模式下快很多，最好能在静态图模式下进行训练和推理，具体原理请参考[动静态图结合](https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html)。

- on-device执行

  MindSpore提供了一种[on-device执行](https://www.mindspore.cn/docs/zh-CN/master/design/overview.html#面向昇腾硬件的竞争力优化)的方法将数据处理和网络在device上的执行并行起来，只需要在`model.train`中设置`dataset_sink_mode=True`即可，注意这个配置默认是`False`，当打开这个配置时，一个epoch只会返回一个网络的结果，当进行调试时建议先将这个值改成`False`。

- 使用自动混合精度

  混合精度训练方法是通过混合使用单精度和半精度数据格式来加速深度神经网络训练的过程，同时保持了单精度训练所能达到的网络精度。混合精度训练能够加速计算过程，同时减少内存使用和存取，并使得在特定的硬件上可以训练更大的模型或 batch size。

  具体可参考 [混合精度教程](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/mixed_precision.html)。

- 使能图算融合

  图算融合是 MindSpore 特有的网络性能优化技术。它可以通过自动分析和优化现有网络计算图逻辑，并结合目标硬件能力，对计算图进行计算化简和替代、算子拆分和融合、算子特例化编译等优化，以提升设备计算资源利用率，实现对网络性能的整体优化。相比传统优化技术，图算融合具有多算子跨边界联合优化、与算子编译跨层协同、基于Polyhedral的算子即时编译等独特优势。另外，图算融合只需要用户打开对应配置后，整个优化过程即可自动完成，不需要网络开发人员进行其它额外感知，使得用户可以聚焦网络算法实现。

  图算融合的适用场景包括：对网络执行时间具有较高性能要求的场景；通过拼接基本算子实现自定义组合算子，并希望对这些基本算子进行自动融合，以提升自定义组合算子性能的场景。

  具体可参考 [图算融合教程](https://www.mindspore.cn/docs/zh-CN/master/design/graph_fusion_engine.html)。

- 其他

  转换算子过多（TransData、Cast类算子）且耗时明显时，如果是我们手动加入的Cast算子，可分析其必要性，如果对精度没有影响，可去掉冗余的Cast、TransData算子。

  如果是MindSpore自动生成的转换算子过多，可能是MindSpore框架针对某些特殊情况没有充分优化，可联系 [MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈。

  动态shape场景目前需要不断的编图，可能会造成端到端的训练时间较长，建议优先[规避动态shape](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/dynamic_shape.html)。

### 多机同步性能优化

当进行分布式训练时，在一个Step的训练过程中，完成前向传播和梯度计算后，各个机器开始进行AllReduce梯度同步，AllReduce同步时间主要受权重数量、机器数量影响，对于越复杂、机器规模越大的网络，其 AllReduce 梯度更新时间也越久，此时我们可以进行AllReduce 切分来优化这部分耗时。

正常情况下，AllReduce 梯度同步会等所有反向算子执行结束，也就是对所有权重都计算出梯度后再一次性同步所有机器的梯度，而使用AllReduce切分后，我们可以在计算出一部分权重的梯度后，就立刻进行这部分权重的梯度同步，这样梯度同步和剩余算子的梯度计算可以并行执行，也就隐藏了这部分 AllReduce 梯度同步时间。切分策略通常是手动尝试，寻找一个最优的方案（支持切分大于两段）。
以 [ResNet50网络](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/train.py) 为例，该网络共有 160 个 权重， [85, 160] 表示第 0 至 85个权重计算完梯度后立刻进行梯度同步，第 86 至 160 个 权重计算完后再进行梯度同步，这里共切分两段，因此需要进行两次梯度同步。代码实现如下：

```python
import os
import mindspore as ms
from mindspore.communication import init

device_id = int(os.getenv('DEVICE_ID', '0'))
rank_size = int(os.getenv('RANK_SIZE', '1'))
rank_id = int(os.getenv('RANK_ID', '0'))

# init context
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=device_id)
if rank_size > 1:
    ms.set_auto_parallel_context(device_num=rank_size, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                gradients_mean=True)
    ms.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
    init()
```

更多请参考[集群性能调试](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_of_cluster.html)。

### 数据处理性能优化

单Step性能抖动、数据队列一段时间内持续为空的情况都是由于数据预处理部分性能较差，使得数据处理速度跟不上单Step迭代速度导致，这两个现象通常成对出现。

当数据处理速度较慢时，队列从最开始的满队列情况逐渐消耗为空队列，训练进程会开始等待空队列填入数据，一旦有新的数据填入，网络才会继续进行单Step训练。由于数据处理没有队列作为缓冲，数据处理的性能抖动直接体现在单Step的性能上，因此还会造成单Step性能抖动。

关于数据的性能问题，可以参考 MindSpore Insight 组件的 [数据准备性能分析](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html#数据准备性能分析)，其给出了数据性能的常见问题及解决方法。

更多性能调试方法请参考[性能优化](https://www.mindspore.cn/docs/zh-CN/master/model_train/train_process/train_optimize.html)。
