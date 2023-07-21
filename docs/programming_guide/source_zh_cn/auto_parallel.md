# 分布式并行

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/auto_parallel.md)

## 概述

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。

MindSpore提供了分布式并行训练的功能，它支持了包括数据并行和自动并行在内的多种并行模式。

## 分布式并行配置

MindSpore的分布式并行配置通过`auto_parallel_context`来进行集中管理，用户可根据自身需求和实际情况来进行个性化的配置。这些配置可分为四大类：

- 通用配置：对数据并行和自动并行均起作用的配置，如：`device_num`、`global_rank`。
- 自动并行配置：仅在自动并行模式下起作用的配置，如：`gradient_fp32_sync`。
- 数据并行配置：仅在数据并行模式下起作用的配置，如：`enable_parallel_optimizer`。
- 混合并行配置：仅在混合并行模式下起作用的配置，如：`layerwise_parallel`。

用户可利用`context.set_auto_parallel_context`配置上述参数，同时可通过`context.get_auto_parallel_context`来获取上述参数。

### 通用配置

#### device_num

`device_num`表示可用的机器数，其值为int型，且必须在1~4096范围内。若用户不配置，`Model`接口内部则会通过`get_group_size`方法获取，若用户进行了配置，则遵循用户的配置。这个配置可以在用户不使用`Model`接口的情况下，手动传递`device_num`。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(device_num=8)
context.get_auto_parallel_context("device_num")
```

#### global_rank

`global_rank`表示当前卡的逻辑序号，其值为int型，且必须在0~4095范围内。若用户不配置，`Model`接口内部则会通过`get_rank`方法获取，若用户进行了配置，则遵循用户的配置。这个配置可以在用户不使用`Model`接口的情况下，手动传递`global_rank`。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(global_rank=0)
context.get_auto_parallel_context("global_rank")
```

#### gradients_mean

`gradients_mean`表示在反向梯度进行聚合时，是否进行平均操作。其值为bool型，默认为False，即梯度聚合仅进行AllReduce的SUM操作，不做平均操作。`gradients_mean`会影响网络的收敛，不同场景，`gradients_mean`的设置可能不同。因此，MindSpore提供这个接口让用户根据实际情况来配置。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(gradients_mean=False)
context.get_auto_parallel_context("gradients_mean")
```

#### parallel_mode

`parallel_mode`表示并行模式，其值为字符串类型。用户可选择的模式有：

- `stand_alone`：单机模式。
- `data_parallel`：数据并行模式。
- `hybrid_parallel`：混合并行模式。
- `semi_auto_parallel`：半自动并行模式，即用户可通过`shard`方法给算子配置切分策略，若不配置策略，则默认是数据并行策略。
- `auto_parallel`：自动并行模式，即框架会自动建立代价模型，为用户选择最优的切分策略。

其中`auto_parallel`和`data_parallel`在MindSpore教程中有完整样例：

<https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/distributed_training_tutorials.html>。

代码样例如下：

```python
from mindspore import context
import mindspore.ops as ops

context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
mul = ops.Mul().shard(((2, 1), (2, 1)))
context.get_auto_parallel_context("parallel_mode")
```

#### all_reduce_fusion_config

`all_reduce_fusion_config`可以让用户自定义梯度AllReduce融合切分策略。出于减少资源消耗及算子执行间隙的目的，框架默认将所有反向梯度聚合的AllReduce融合成一个算子运算，但当模型较大时，这会造成迭代拖尾耗时增加。用户可结合具体网络，通过设置该参数，手动调优找到性能最好的融合切分策略。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(all_reduce_fusion_config=[20, 35])
context.get_auto_parallel_context("all_reduce_fusion_config")
```

样例中，`all_reduce_fusion_config`的值为[20, 35]，将前20个AllReduce融合成1个，第20～35个AllReduce融合成1个，剩下的AllReduce融合成1个。

### 自动并行配置

#### gradient_fp32_sync

`gradient_fp32_sync`表示梯度是否以float32类型进行聚合，其值为bool类型，默认为True，即梯度以float32类型进行聚合。由于`Ascend`AI处理器的特殊构造，float32类型的数据进行聚合的速度要高于float16，但可能会影响精度。因此，MindSpore提供`gradient_fp32_sync`接口，让用户自己根据实际情况去进行取舍。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(gradient_fp32_sync=False)
context.get_auto_parallel_context("gradient_fp32_sync")
```

#### auto_parallel_search_mode

MindSpore提供了`dynamic_programming`和`recursive_programming`两种搜索策略的算法。`dynamic_programming`能够搜索出代价模型刻画的最优策略，但在搜索巨大网络模型的并行策略时耗时较长；而`recursive_programming`能较快搜索出并行策略，但搜索出来的策略可能不是运行性能最优的。为此，MindSpore提供了参数，让用户自由选择搜索算法，默认是`dynamic_programming`。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(auto_parallel_search_mode="dynamic_programming")
context.get_auto_parallel_context("auto_parallel_search_mode")
```

#### strategy_ckpt_load_file

指定加载路径，加载自动并行中所有带有权重的算子的切分信息。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(strategy_ckpt_load_file="./")
context.get_auto_parallel_context("strategy_ckpt_load_file")
```

#### strategy_ckpt_save_file

指定存储路径，存储自动并行中所有带有权重的算子的切分信息。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(strategy_ckpt_save_file="./")
context.get_auto_parallel_context("strategy_ckpt_save_file")
```

#### full_batch

`full_batch`可以让用户决定数据集是否以全量导入。默认是False。即数据集以数据并行的方式导入。在特殊场景下，数据集全量导入的性能要优于数据并行方式导入，比如WideDeep网络的非均匀切分场景。因此，MindSpore提供`full_batch`可配置接口。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(full_batch=False)
context.get_auto_parallel_context("full_batch")
```

### 数据并行配置

#### enable_parallel_optimizer

`enable_parallel_optimizer`是一个开发中特性，期望通过优化器并行的方式提升网络性能。使能后框架将拆分权重到各卡上分别进行参数更新，由集合通信的方式在每卡间同步权重信息，再进入下一轮迭代计算。该功能目前只在数据并行模式和参数量大于机器数时有效，支持`Lamb`和`AdamWeightDecay`优化器。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(enable_parallel_optimizer=True)
context.get_auto_parallel_context("enable_parallel_optimizer")
```

### 混合并行配置

#### layerwise_parallel

在`HYBRID_PARALLEL`模式下用户需要手动切分模型，其中对于模型并行的参数用户需要手动打上标记`layerwise_parallel`，框架会根据此标记为模型并行参数过滤掉梯度聚合操作。

代码样例如下：

```python
imoprt numpy as np
from mindspore.common import Parameter, Tensor

x = Parameter(Tensor(np.ones([2, 2])), name="x", layerwise_parallel=True)
```

## 分布式通信接口

`mindspore.communication.management`中封装了分布式并行用到的集合通信接口，方便用户配置分布式信息。

### init

使能MindSpore通信，并完成分布式训练初始化操作。`init`要在`context.set_context`之后调用。用户可给`init`传入通信后端信息，`init`会根据不同的后端来进行不同初始化。

- `hccl`：全名为`Huawei Collective Communication Library`。用于`Ascend`处理器平台。
- `nccl`：全名为`NVIDIA Collective Communication Library`。用于`GPU`处理器平台。

若用户不配置通信后端，MindSpore会根据`context`中的`device_target`信息进行自动配置。

代码样例如下：

```python
from mindspore.communication.management import init

context.set_context(device_target='GPU')
init()
```

### get_group_size

`get_group_size`可让用户获取集群数量。在用`get_group_size`接口之前，要先调用`init`。

代码样例如下：

```python
from mindspore.communication.management import init, get_group_size

context.set_context(device_target='GPU')
init()
group_size = get_group_size()
```

### get_rank

`get_rank`可让用户获取当前设备在集群中的ID。在用`get_rank`接口之前，要先调用`init`。

代码样例如下：

```python
from mindspore.communication.management import init, get_rank

context.set_context(device_target='GPU')
init()
rank_id = get_rank()
```

## 分布式属性配置

### cross_batch

在特定场景下，`data_parallel`的计算逻辑和`stand_alone`是不一样的，`auto_parallel`在任何场景下都是和`stand_alone`的计算逻辑保持一致。而`data_parallel`的收敛效果可能更好，因此MindSpore提供了`cross_barch`这个参数，可以使`auto_parallel`的计算逻辑和`data_parallel`保持一致，用户可通过`add_prim_attr`方法进行配置，默认值是False。

代码样例如下：

```python
import mindspore.ops as ops

mul = ops.Mul().add_prim_attr("cross_batch", True)
```

### fusion

出于性能考虑，MindSpore提供了`AllGather`和`AllReduce`算子的融合功能，`fusion`值相同的同类算子（算子类型以及通信域相同）会融合在一起，`fusion`的值必须大于等于0，且当`fusion`值为0时，表示不融合。

代码样例如下：

```python
import mindspore.ops as ops

allreduce1 = ops.AllReduce().add_prim_attr("fusion", 1)
allreduce2 = ops.AllReduce().add_prim_attr("fusion", 1)
```

## 数据并行

数据并行是对数据进行切分的并行模式，一般按照batch维度切分，将数据分配到各个计算单元（worker）中，进行模型计算。在数据并行模式下，数据集要以数据并行的方式导入，并且`parallel_mode`要设置为`data_parallel`。

具体用例请参考MindSpore分布式并行训练教程：

<https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/distributed_training_tutorials.html>。

## 自动并行

自动并行是融合了数据并行、模型并行及混合并行的一种分布式并行模式，可以自动建立代价模型，为用户选择一种并行模式。其中，代价模型指基于内存的计算开销和通信开销对训练时间建模，并设计高效的算法找到训练时间较短的并行策略。在自动并行模式下，数据集也要以数据并行的方式导入，并且`parallel_mode`要设置为`auto_parallel`。

具体用例请参考MindSpore分布式并行训练教程：

<https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/distributed_training_tutorials.html>。
