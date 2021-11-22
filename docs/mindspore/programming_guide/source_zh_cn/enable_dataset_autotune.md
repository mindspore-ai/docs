# 使能自动数据加速

`Ascend` `GPU` `CPU` `数据处理` `性能调优`

<!-- TOC -->

- [使能自动数据加速](#使能自动数据加速)
    - [概述](#概述)
    - [如何使能自动数据加速](#如何使能自动数据加速)
    - [如何调整自动数据加速的采样间隔](#如何调整自动数据加速的采样间隔)
    - [约束](#约束)
    - [样例](#样例)
        - [自动数据加速配置](#自动数据加速配置)
        - [开始训练](#开始训练)
        - [在进行下一次训练之前](#在进行下一次训练之前)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/enable_dataset_autotune.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore提供了一种自动数据调优的工具——AutoTune，用于在训练过程中根据环境资源的情况自动调整数据处理管道的并行度，
最大化利用系统资源加速数据处理管道的处理速度。

在整个训练的过程中，AutoTune模块会持续检测当前训练性能瓶颈处于数据侧还是网络侧。如果检测到瓶颈在数据侧，则将
进一步对数据处理管道中的各个算子（如GeneratorDataset、map、batch此类数据算子）进行参数调整，
目前可调整的参数包括算子的工作线程数（num_parallel_workers），算子的内部队列深度（prefetch_size）。

![autotune](../source_en/images/autotune.png)

使能AutoTune后，MindSpore会根据一定的时间间隔，对数据处理管道的资源情况进行采样统计。

当AutoTune收集到足够的信息时，它会基于这些信息分析当前的性能瓶颈是否在数据侧。
如果是，AutoTune将调整数据处理管道的并行度，并加速数据集管道的运算。
如果不是，AutoTune也会尝试减少数据管道的内存使用量，为CPU释放一些可用内存。

> 自动数据加速在默认情况下是关闭的。

## 如何使能自动数据加速

使能自动数据加速:

```python
import mindspore.dataset as ds
ds.config.set_enable_autotune(True)
```

## 如何调整自动数据加速的采样间隔

调整自动数据加速的采样时间间隔（单位是毫秒）:

```python
import mindspore.dataset as ds
ds.config.set_autotune_interval(100)
```

获取当前设定的采样时间间隔（单位是毫秒）:

```python
import mindspore.dataset as ds
print("time interval:", ds.config.get_autotune_interval())
```

## 约束

自动数据加速目前仅可用于下沉模式（dataset_sink_mode=True）。

Profiling性能分析和自动数据加速无法同时开启。

## 样例

以LeNet网络训练作为一个样例。

### 自动数据加速配置

若要启用自动数据加速，仅需添加一条语句即可。

```python
# dataset.py of LeNet in ModelZoo
# models/official/cv/lenet/src/dataset.py

def create_dataset(...)
    """
    create dataset for train or test
    """
    # 使能自动数据加速
    ds.config.set_enable_autotune(True)

    # 其他数据集代码无需变更
    mnist_ds = ds.MnistDataset(data_path)
    ...
```

### 开始训练

根据[lenet/README.md](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/README_CN.md)所描述的步骤启动训练，
随后自动数据加速模块会通过LOG的形式展示其对于性能瓶颈的分析情况：

```text
[INFO] [auto_tune.cc] LaunchThread] Launching AutoTune thread
[INFO] [auto_tune.cc] Main] AutoTune thread has started.
[INFO] [auto_tune.cc] RunIteration] Run AutoTune at epoch #1
[INFO] [auto_tune.cc] RecordPipelineTime] Epoch #1, Average Pipeline time is 6.88267 ms. The avg pipeline time for all epochs is 6.88267ms
[INFO] [auto_tune.cc] IsDSaBottleneck] Epoch #1, Device Connector Size: 4.65387, Connector Capacity: 16, Utilization: 29.0867%, Empty Freq: 48.32%
[WARNING] [auto_tune.cc] IsDSaBottleneck] Utilization: 29.0867% < 75% threshold, dataset pipeline performance needs tuning.
[WARNING] [auto_tune.cc] Analyse] Leaf op (MnistOp(ID:9)) queue utilization: 0% < 90% threshold.
[WARNING] [auto_tune.cc] RequestNumWorkerChange] Added request to change number of workers of Operator: MnistOp(ID:9) New value: 10 Old value: 8
[WARNING] [auto_tune.cc] Analyse] Op (MapOp(ID:8)) getting low average worker cpu utilization 18.8182% < 35% threshold.
[WARNING] [auto_tune.cc] RequestConnectorCapacityChange] Added request to change Connector capacity of Operator: MapOp(ID:8) New value: 5 Old value: 1
```

数据管道的性能分析和调整过程可以通过上述的LOG体现：

自动数据加速模块在默认情况下以INFO级别的LOG信息提醒用户其运行的情况，当检测到数据侧为瓶颈且需要发生参数变更时，
则会通过WARNING级别的LOG信息提醒用户正在调整的参数，以及最终调整的数值。

上面的LOG反映了自动数据加速模块的流程。在数据处理管道的初始配置下，Device Connector队列（数据管道与计算网络之间的缓冲队列）的利用率较低。
原因主要是数据处理管道生成数据的速度较慢，网络侧很快就会读取完生成的数据。
随后基于此情况，自动数据加速模块增大了MnistOp(ID:9)的工作线程数与MapOp(ID:8)算子内部队列深度，提高了数据处理管道的并行性，加快了速度处理的速度。

在应用了较优的数据处理管道配置后，整体的step time得到了可观的减少。

### 在进行下一次训练之前

在进行下一次训练之前，用户可以根据自动数据加速模块得出的推荐配置，对数据集脚本进行调整，
以便在下一次训练的开始时就达到以较优性能水平运行数据处理管道。

另外，MindSpore也提供了相关的API用于全局调整数据处理管道算子的并行度与内部队列深度，请参考[mindspore.dataset.config](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.dataset.config.html):

- [mindspore.dataset.config.set_num_parallel_workers](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_num_parallel_workers)
- [mindspore.dataset.config.set_prefetch_size](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_prefetch_size)

