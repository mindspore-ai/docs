# 分布式并行接口说明

`Ascend` `GPU` `分布式并行`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_zh_cn/auto_parallel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。

MindSpore提供了分布式并行训练的功能，它支持了包括数据并行和自动并行在内的多种并行模式。

## 分布式并行配置

MindSpore的分布式并行配置通过`auto_parallel_context`来进行集中管理，用户可根据自身需求和实际情况来进行个性化的配置。这些配置可分为三大类：

- 通用配置：对数据并行、自动并行以及混合并行均起作用的配置，如：`device_num`、`global_rank`等。
- 自动并行配置：仅在自动并行模式下起作用的配置，如：`search_mode`、`gradient_fp32_sync`等。

用户可利用`context.set_auto_parallel_context`配置上述参数，同时可通过`context.get_auto_parallel_context`来获取上述参数。

### 通用配置

#### device_num

`device_num`表示可用的机器数，其值为int型，默认值是1，且必须在1~4096范围内。若用户不配置，`Model`接口内部则会通过`get_group_size`方法获取，若用户进行了配置，则遵循用户的配置。这个配置可以在用户不使用`Model`接口的情况下，手动传递`device_num`。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(device_num=8)
context.get_auto_parallel_context("device_num")
```

#### global_rank

`global_rank`表示当前卡的逻辑序号，其值为int型，默认值是0，且必须在0~4095范围内。若用户不配置，`Model`接口内部则会通过`get_rank`方法获取，若用户进行了配置，则遵循用户的配置。这个配置可以在用户不使用`Model`接口的情况下，手动传递`global_rank`。

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

<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/distributed_training.html>。

代码样例如下：

```python
from mindspore import context
import mindspore.ops as ops

context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
mul = ops.Mul().shard(((2, 1), (2, 1)))
context.get_auto_parallel_context("parallel_mode")
```

> 在semi_auto_parallel模式下，如果一个Parameter被多个算子共享，则需要保证该Parameter在每个算子中的排布都一致，否则构图将会失败。比如下面这个例子中，mul1和mul2共享权重weight，但mul1对weight按行切8份，而mul2对weight按列切8份，weight在两个算子中的排布不一致，构图将会失败：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.nn import Cell

class Net(Cell):
    """Net definition"""
    def __init__(self):
        super(Net, self).__init__()
        self.mul1 = ops.Mul().shard(((8, 1), (8, 1)))
        self.mul2 = ops.Mul().shard(((1, 8), (1, 8)))
        self.weight = Parameter(Tensor(np.ones([16, 32]), dtype=ms.float32), "weight1")

    def construct(self, x):
        out = self.mul1(x, self.weight)
        out = self.mul2(out, self.weight)
        return out
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

#### enable_parallel_optimizer

`enable_parallel_optimizer`是一个开发中特性，参数默认值是False。数据并行时参数更新部分在各卡间存在冗余计算，优化器并行通过将优化器的计算量分散到各个卡上，在大规模网络上（比如Bert、GPT）可以有效减少内存消耗并提升网络性能。

在`data_parallel`模式下使能优化器并行，框架会将需要更新的参数进行分组到不同卡上，各自更新后再通过`Broadcast`算子在集群间做权重共享。需要注意的是参数量应当大于机器数，当前只支持`Lamb`和`AdamWeightDecay`优化器。

在`auto_parallel`或者`semi_auto_parallel`模式下使能优化器并行，如果经过策略切分后的参数在机器间存在重复切片，并且shape的最高维可以被卡数整除，框架会以最小切片的方式保存参数并在优化器中更新。该模式下支持所有优化器。

无论是哪种模式，优化器并行不会影响原有正反向网络的计算图，只会影响参数更新的计算量和计算逻辑。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(enable_parallel_optimizer=True)
context.get_auto_parallel_context("enable_parallel_optimizer")
```

#### parameter_broadcast

`parameter_broadcast`将数据并行参数在0号卡上的权值广播到其他卡上，达到同步初始化权重的目的。参数默认值是False，当前仅支持图模式。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(parameter_broadcast=True)
context.get_auto_parallel_context("parameter_broadcast")
```

#### comm_fusion

`comm_fusion`通过设置通信算子的融合配置，实现不同通信算子的融合功能，当前仅支持`allreduce`通信算子的配置。对于`allreduce`通信算子的配置，共支持三种不同的`mode`：

- `auto`：自动进行`allreduce`通信算子融合，按照梯度数据量阈值64MB进行算子融合，配置参数`config`为`None`。
- `size`：按照手动设置梯度量阈值的方式进行`allreduce`通信算子融合，配置参数`config`类型为`int`，单位`MB`。
- `index`：按照通信算子序列号进行融合的方式，与`all_reduce_fusion_config`功能相同，配置参数`config`类型为`list(int)`。

代码样例如下：

```python
from mindspore import context

# auto
context.set_auto_parallel_context(comm_fusion={"allreduce": {"mode": "auto", "config": None}})

# size
context.set_auto_parallel_context(comm_fusion={"allreduce": {"mode": "size", "config": 32}})

# index
context.set_auto_parallel_context(comm_fusion={"allreduce": {"mode": "index", "config": [20, 35]}})
```

### 自动并行配置

#### gradient_fp32_sync

`gradient_fp32_sync`表示梯度是否以float32类型进行聚合，其值为bool类型，默认为True，即梯度以float32类型进行聚合。由于`Ascend`AI处理器的特殊构造，float32类型的数据进行聚合的速度要高于float16，但可能会影响精度。因此，MindSpore提供`gradient_fp32_sync`接口，让用户自己根据实际情况去进行取舍。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(gradient_fp32_sync=False)
context.get_auto_parallel_context("gradient_fp32_sync")
```

#### search_mode

`auto_parallel_search_mode`字段现已替换为`search_mode`，`auto_parallel_search_mode`将会在未来版本中删除。该字段用于指示并行策略搜索使用的算法。当前，MindSpore提供了`dynamic_programming`，`recursive_programming`和`sharding_propagation`三种搜索策略的算法用于搜索算子级并行策略，默认是`dynamic_programming`。

`dynamic_programming`能够搜索出代价模型刻画的最优策略，但在搜索巨大网络模型的并行策略时耗时较长；而`recursive_programming`能瞬间搜索出并行策略，同时在已验证的常用网络中搜索出来的策略是最优策略，但在未经验证的某些特殊网络中可能找到次优策略。`sharding_propagation`要求用户配置一些算子的并行策略，并以此为基础向整个网络传播。在传播时，算法会尽量选取引发张量[重排布](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/design/distributed_training_design.html#id10)通信最少的策略。MindSpore提供了参数，让用户自由选择搜索算法。

代码样例如下：

```python
from mindspore import context

context.set_auto_parallel_context(search_mode="recursive_programming")
context.get_auto_parallel_context("search_mode")
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

#### pipeline_stages

近年来，神经网络的规模几乎是呈指数型增长。受单卡内存的限制，训练这些大模型用到的设备数量也在不断增加。受server间通信带宽低的影响，传统数据并行叠加模型并行的这种混合并行模式的性能表现欠佳，需要引入流水线并行。流水线并行能够将模型在空间上按`stage`进行切分，每个`stage`只需执行网络的一部分，大大节省了内存开销，同时缩小了通信域。MindSpore能够根据用户的配置，将单机模型自动地转换成流水线并行模式去执行。`pipeline_stages`用来设置流水线并行的`stage`个数。
代码样例如下：

```python
from mindspore import context
context.set_auto_parallel_context(pipeline_stages=4)
context.get_auto_parallel_context("pipeline_stages")
```

#### parallel_optimizer_config

`parallel_optimizer_config`是一个字典。当启用优化器并行时，该配置提供了有关优化器并行训练的优化选项。
当我们设置context.set_auto_parallel_context(enable_parallel_optimizer=True)时，`parallel_optimizer_config`配置才会生效。它目前支持如下键值。

- gradient_accumulation_shard:如果为True，则累积梯度变量将在数据并行度上进行分片。在累积梯度时，每个累积迭代中将会引入额外的通信(ReduceScatter)以保证计算的一致性，但节省了大量的计算设备内存(例如GPU显存)，因此可以使模型以更大的批量进行训练。仅当模型在流水线并行训练或梯度累积中设置此配置，并且具有数据并行维度时，此配置才会有效。默认值为True。

```python
from mindspore import context
context.set_auto_parallel_context(parallel_optimizer_config={"gradient_accumulation_shard": True}, enable_parallel_optimizer=True)
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
from mindspore import context
from mindspore.communication import init

context.set_context(device_target='GPU')
init()
```

> 在GPU处理器平台下，MindSpore还支持不依赖`OpenMPI`来启动分布式训练，也使用本接口进行分布式训练初始化，具体方法可参考[不依赖OpenMPI进行训练](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/distributed_training_gpu.html#openmpi)。在此场景下，即用户不使用'mpirun'启动进程，但是依然调用了`init()`方法的情况下，MindSpore要求用户按照[不依赖OpenMPI进行训练](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/distributed_training_gpu.html#openmpi)配置若干环境变量，若没有配置，MindSpore会给出合理的报错提示。因此建议只有在执行分布式训练时调用此方法，并在不使用`mpirun`的场景下，根据文档配置正确的环境变量以启动分布式训练。

### get_group_size

`get_group_size`可让用户获取集群数量。在用`get_group_size`接口之前，要先调用`init`。

代码样例如下：

```python
from mindspore import context
from mindspore.communication import init, get_group_size

context.set_context(device_target='GPU')
init()
group_size = get_group_size()
```

### get_rank

`get_rank`可让用户获取当前设备在集群中的ID。在用`get_rank`接口之前，要先调用`init`。

代码样例如下：

```python
from mindspore import context
from mindspore.communication import init, get_rank

context.set_context(device_target='GPU')
init()
rank_id = get_rank()
```

## 分布式属性配置

### cross_batch

在特定场景下，`data_parallel`的计算逻辑和`stand_alone`是不一样的，`auto_parallel`在任何场景下都是和`stand_alone`的计算逻辑保持一致。而`data_parallel`的收敛效果可能更好，因此MindSpore提供了`cross_batch`这个参数，可以使`auto_parallel`的计算逻辑和`data_parallel`保持一致，用户可通过`add_prim_attr`方法进行配置，默认值是False。

代码样例如下：

```python
import mindspore.ops as ops

mul = ops.Mul().add_prim_attr("cross_batch", True)
```

### fusion

出于性能考虑，MindSpore提供了`AllGather`和`AllReduce`算子的融合功能，`fusion`值相同的同类算子（算子类型以及通信域相同）会融合在一起，`fusion`的值必须大于等于0，且当`fusion`值为0时，表示不融合。目前只支持`Ascend`后端。

`fusion`属性的配置有两种方式，如果是显式调用通信算子可以通过`add_prim_attr`方法直接为通信算子配置属性。代码样例如下：

```python
import mindspore.ops as ops

allreduce1 = ops.AllReduce().add_prim_attr("fusion", 1)
allreduce2 = ops.AllReduce().add_prim_attr("fusion", 1)
```

样例中的`allreduce1`和`allreduce2`将在执行时被融合为一个算子。

在`auto_parallel`和`semi_auto_parallel`模式下自动插入的用于参数或者梯度聚合的通信算子，需要通过对`Cell`或者`Parameter`设置属性的方式间接添加。例如：

```python
import numpy as np
from mindspore import ops
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context

class Net(nn.Cell):
    """Net definition"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = ops.MatMul()
        self.fc2 = ops.MatMul()
        self.p1 = Parameter(Tensor(np.ones([48, 64]).astype(np.float32)), name="weight1")
        self.p1.comm_fusion = 2
        self.p2 = Parameter(Tensor(np.ones([64, 16]).astype(np.float32)), name="weight2")

    def construct(self, x, y):
        x = self.fc1(x, self.p1)
        x = self.fc2(x, self.p2)
        return x - y

context.set_context(mode=context.GRAPH_MODE)
context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8)
net = Net().set_comm_fusion(2)
```

样例中将参数`Net.p1`设置`comm_fusion`为2，表示作用于该参数的通信算子`fusion`属性为2。当需要批量对参数进行操作时，可以调用`set_comm_fusion`方法将网络`Net`中包含的全部参数设置`comm_fusion`属性。如果多次调用的话，属性值会被覆盖。

> 当参数被共享时，需要保证连接参数的多个算子混合精度一致，否则融合会失败。

### layerwise_parallel

在`hybrid_parallel`模式下用户需要手动切分模型，其中对于模型并行的参数，用户需要手动打上标记`layerwise_parallel`，框架会根据此标记为模型并行参数过滤掉梯度聚合操作。

代码样例如下：

```python
import numpy as np
from mindspore import Parameter, Tensor

x = Parameter(Tensor(np.ones([2, 2])), name='weight1', layerwise_parallel=True)
```
