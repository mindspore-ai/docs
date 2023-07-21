# 运行管理

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/context.md)

## 概述

初始化网络之前要配置context参数，用于控制程序执行的策略。比如选择执行模式、选择执行后端、配置分布式相关参数等。按照context参数设置实现的不同功能，可以将其分为执行模式管理、硬件管理、分布式管理和维测管理等。

## 执行模式管理

MindSpore支持PyNative和Graph这两种运行模式：

- `PYNATIVE_MODE`：动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。

- `GRAPH_MODE`：静态图模式或者图模式，将神经网络模型编译成一整张图，然后下发执行。该模式利用图优化等技术提高运行性能，同时有助于规模部署和跨平台运行。

### 模式选择

通过设置可以控制程序运行的模式，默认情况下，MindSpore处于PyNative模式。

代码样例如下：

```python
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)
```

### 模式切换

实现两种模式之间的切换。

MindSpore处于PYNATIVE模式时，可以通过`context.set_context(mode=context.GRAPH_MODE)`切换为Graph模式；同样地，MindSpore处于Graph模式时，可以通过 `context.set_context(mode=context.PYNATIVE_MODE)`切换为PyNative模式。

代码样例如下：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

conv = nn.Conv2d(3, 4, 3, bias_init='zeros')
input_data = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
conv(input_data)
context.set_context(mode=context.PYNATIVE_MODE)
conv(input_data)
```

上面的例子先将运行模式设置为`GRAPH_MODE`模式，然后将模式切换为`PYNATIVE_MODE`模式，实现了模式的切换。

## 硬件管理

硬件管理部分主要包括`device_target`和`device_id`两个参数。

- `device_target`： 用于设置目标设备，支持Ascend、GPU和CPU，可以根据实际环境情况设置。

- `device_id`： 表示卡物理序号，即卡所在机器中的实际序号。如果目标设备为Ascend，且规格为N*Ascend（其中N>1，如8*Ascend），在非分布式模式执行的情况下，为了避免设备的使用冲突，可以通过设置`device_id`决定程序执行的device编号，该编号范围为：0 ~ 服务器总设备数量-1，服务器总设备数量不能超过4096，默认为设备0。

> 在GPU和CPU上，设置`device_id`参数无效。

代码样例如下：

```python
from mindspore import context
context.set_context(device_target="Ascend", device_id=6)
```

## 分布式管理

context中有专门用于配置并行训练参数的接口：context.set_auto_parallel_context，该接口必须在初始化网络之前调用。

- `parallel_mode`：分布式并行模式，默认为单机模式`ParallelMode.STAND_ALONE`。可选数据并行`ParallelMode.DATA_PARALLEL`及自动并行`ParallelMode.AUTO_PARALLEL`。

- `gradients_mean`：反向计算时，框架内部会将数据并行参数分散在多台机器的梯度值进行收集，得到全局梯度值后再传入优化器中更新。默认值为`False`，设置为True对应`allreduce_mean`操作，False对应`allreduce_sum`操作。

- `enable_parallel_optimizer`：开发中特性。打开优化器模型并行开关，通过拆分权重到各卡分别进行更新再同步的方式以提升性能。该参数目前只在数据并行模式和参数量大于机器数时有效，支持`Lamb`和`Adam`优化器。

> `device_num`和`global_rank`建议采用默认值，框架内会调用HCCL接口获取。

代码样例如下：

```python
from mindspore import context
from mindspore.context import ParallelMode
context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
```

> 分布式并行训练详细介绍可以查看[分布式并行训练](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/distributed_training_tutorials.html)。

## 维测管理

为了方便维护和定位问题，context提供了大量维测相关的参数配置，如采集profiling数据、异步数据dump功能和print算子落盘等。

### 采集profiling数据

系统支持在训练过程中采集profiling数据，然后通过profiling工具进行性能分析。当前支持采集的profiling数据包括：

- `enable_profiling`：是否开启profiling功能。设置为True，表示开启profiling功能，从enable_options读取profiling的采集选项；设置为False，表示关闭profiling功能，仅采集training_trace。

- `enable_options`：profiling采集选项，取值如下，支持采集多项数据。training_trace：采集迭代轨迹数据，即训练任务及AI软件栈的软件信息，实现对训练任务的性能分析，重点关注数据增强、前后向计算、梯度聚合更新等相关数据；task_trace：采集任务轨迹数据，即昇腾910处理器HWTS/AICore的硬件信息，分析任务开始、结束等信息；op_trace：采集单算子性能数据。格式：['op_trace','task_trace','training_trace']

代码样例如下：

```python
from mindspore import context
context.set_context(enable_profiling=True, profiling_options="training_trace")
```

### 异步数据dump功能

在Ascend环境上执行训练，当训练结果和预期有偏差时，可以通过异步数据dump功能保存算子的输入输出进行调试。

代码样例如下：

```python
from mindspore import context
context.set_context(save_graphs=True)
```

> 详细的调试方法可以查看[异步数据Dump功能介绍](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/custom_debugging_info.html#dump)。

### print算子落盘

默认情况下，MindSpore的自研print算子可以将用户输入的Tensor或字符串信息打印出来，支持多字符串输入，多Tensor输入和字符串与Tensor的混合输入，输入参数以逗号隔开。

> Print打印功能可以查看[Print算子功能介绍](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/custom_debugging_info.html#print)。

- `print_file_path`：可以将print算子数据保存到文件，同时关闭屏幕打印功能。如果保存的文件已经存在，则会给文件添加时间戳后缀。数据保存到文件可以解决数据量较大时屏幕打印数据丢失的问题。

代码样例如下：

```python
from mindspore import context
context.set_context(print_file_path="print.pb")
```

> context接口详细介绍可以查看[mindspore.context](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.context.html)。
