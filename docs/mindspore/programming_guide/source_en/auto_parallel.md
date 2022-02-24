# Parallel Distributed Training Interfaces

`Ascend` `GPU` `Distributed Parallel`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/auto_parallel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

In deep learning, as the number of datasets and parameters increases, the time and hardware resources required for training increase, and finally become a bottleneck to the training. Parallel distributed training can reduce the requirements on hardware such as memory and computing performance and is an important optimization method for training.

MindSpore provides the parallel distributed training function and supports multiple parallel modes, including data parallel and automatic parallel.

## Parallel Distributed Training Configuration

The parallel distributed training configuration of MindSpore is managed by `auto_parallel_context` in a centralized manner. You can customize the configuration based on the actual situation. These configurations can be classified into three types:

- General configuration: takes effect on both data parallel and automatic parallel, for example, `device_num` and `global_rank` etc.
- Automatic parallel configuration: takes effect only in automatic parallel mode, for example, `gradient_fp32_sync` etc.

You can use `context.set_auto_parallel_context` to configure the preceding parameters and use `context.get_auto_parallel_context` to obtain the parameters.

### General Configuration

#### device_num

`device_num` indicates the number of available machines. The default value is 0. The value is of the int type and must range from 1 to 4096. If you do not set this parameter, the `Model` interface obtains the value by using the `get_group_size` method. If you set this parameter, your configuration is used. This configuration allows you to manually transfer `device_num` without using the `Model` interface.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(device_num=8)
context.get_auto_parallel_context("device_num")
```

#### global_rank

`global_rank` indicates the logical sequence number of the current device. The default value is 0. The value is of the int type and must range from 0 to 4095. If you do not set this parameter, the `Model` interface obtains the value by using the `get_rank` method. If you set this parameter, your configuration is used. This configuration allows you to manually transfer `global_rank` without using the `Model` interface.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(global_rank=0)
context.get_auto_parallel_context("global_rank")
```

#### gradients_mean

`gradients_mean` indicates whether to perform the averaging operation during reverse gradient aggregation. The value is of the Boolean type. The default value is False, indicating that only the SUM operation of AllReduce is performed for gradient aggregation. `gradients_means` affects network convergence. The setting of `gradients_means` may vary in different scenarios. Therefore, MindSpore provides this interface for users to configure parameters based on the actual situation.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(gradients_mean=False)
context.get_auto_parallel_context("gradients_mean")
```

#### parallel_mode

`parallel_mode` indicates the parallel mode. The value is a character string. The options are as follows:

- `stand_alone`: standalone mode.
- `data_parallel`: data parallel mode.
- `hybrid_parallel`: hybrid parallel mode.
- `semi_auto_parallel`: semi-automatic parallel mode. In this mode, you can use the `shard` method to configure a segmentation policy for an operator. If no policy is configured, the data parallel policy is used by default.
- `auto_parallel`: automatic parallel mode. In this mode, the framework automatically creates a cost model and selects the optimal segmentation policy for users.

The complete examples of `auto_parallel` and `data_parallel` are provided in [Distributed Training](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training.html).

The following is a code example:

```python
from mindspore import context
import mindspore.ops as ops

context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
mul = ops.Mul().shard(((2, 1), (2, 1)))
context.get_auto_parallel_context("parallel_mode")
```

> In semi_auto_parallel mode, if a parameter is used by multiple operators, please ensure that the parameter layout in each operator is consistent, otherwise an error will be reported during compilation. In the following example, mul1 and mul2 share the weight, but mul1 splits weight into 8 slices by row, however, mul2 splits the weight into 8 slices by column. The layout of weight in the two operators is inconsistent, compilation will be failed.

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

`all_reduce_fusion_config` allows users to customize the AllReduce segmentation policy by gradient aggregation. To reduce resource consumption and operator execution gaps, the framework fusions all the reverse gradient aggregation AllReduce operators into one by default. However, when the model is large, the iteration smearing time increases. You can set this parameter based on the actual network to manually tune and find the optimal segmentation policy by gradient aggregation.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(all_reduce_fusion_config=[20, 35])
context.get_auto_parallel_context("all_reduce_fusion_config")
```

In the example, the value range of `all_reduce_fusion_config` is [20,35]. The first 20 AllReduce operators, the 20th to 35th AllReduce operators, and the remaining AllReduce operators are fused into three operators, respectively.

#### enable_parallel_optimizer

`enable_parallel_optimizer` is a feature under development. The default value is False. In data parallel, weight update has redundant computation among devices. Parallel optimizer shards the computation of optimizer to each device. For large-scale networks like Bert and GPT, this feature could reduce requirements on memory and improve the performance efficiently.

When the `enable_parallel_optimizer` is enabled in the data_parallel mode, MindSpore will split the parameters that need to be updated into different devices, and then use the Broadcast operator to share weights between clusters after each update. It should be noted that the number of parameters should be greater than the number of machines. Currently, only the `Lamb` and `AdamWeightDecay` optimizers are supported.

In the `auto_parallel` or `semi_auto_parallel` mode, the optimizer parallel is enabled. If one parameter which has been sliced by shard strategy still has repeated slices among devices, and the highest dimension of the shape can be divided by the number of devices, MindSpore would save parameters and update them by the smallest slice shapes. All optimizers are supported under this two modes.

No matter which parallel mode is selected, parallel optimizer would not influence the forward and backward graph. Only the computation of weight update would be influenced.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(enable_parallel_optimizer=True)
context.get_auto_parallel_context("enable_parallel_optimizer")
```

#### parameter_broadcast

Parameter broadcast shares the value of data parallel weights among devices, in the purpose of synchronization of weights. The default value is False and only the graph mode is supported.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(parameter_broadcast=True)
context.get_auto_parallel_context("parameter_broadcast")
```

#### comm_fusion

`comm_fusion` allows user to configure the communication fusion for various communication operators, and for now, `allreduce`, `allgather`, `reducescatter` are supported. For `allreduce`, it has three `mode` options:

- `auto`：automatic communication operators fusion by gradients size, and another parameter `config` is `None`. The gradients fusion size is automatically set by 64 MB.
- `size`：manual communication operators fusion by gradients size, and the type of another parameter `config` is `int` and unit is `MB`.
- `index`：manual communication operators fusion by parameters' index，same as `all_reduce_fusion_config`, and the type of parameter `config` is `list(int)`.

For `allgather` and `reducescatter`, two `mode` options are supported:

- `auto`：same as `allreduce`, parameter `config` is `None`. The gradients fusion size is automatically set by 64 MB.
- `size`：same as `allreduce`, and the type of another parameter `config` is `int` and unit is `MB`.

The following is a code example:

```python
from mindspore import context

# allreduce auto
context.set_auto_parallel_context(comm_fusion={"allreduce": {"mode": "auto", "config": None}})

# allreduce size
context.set_auto_parallel_context(comm_fusion={"allreduce": {"mode": "size", "config": 32}})

# allreduce index
context.set_auto_parallel_context(comm_fusion={"allreduce": {"mode": "index", "config": [20, 35]}})

# allgather and reducescatter size
context.set_auto_parallel_context(comm_fusion={"allgather": {"mode": "size", "config": 16},
                                               "reducescatter": {"mode": "size", "config": 32}})
```

### Automatic Parallel Configuration

#### gradient_fp32_sync

`gradient_fp32_sync` indicates whether gradients are aggregated based on the FP32 type. The value is of the Boolean type. The default value is True, indicating that gradients are aggregated based on the FP32 type. Due to the special structure of the `Ascend` AI processor, the speed of aggregating FP32 data is higher than that of aggregating FP16 data, but the precision may be affected. Therefore, MindSpore provides the `gradient_fp32_sync` interface for users to make choices based on the actual situation.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(gradient_fp32_sync=False)
context.get_auto_parallel_context("gradient_fp32_sync")
```

#### search_mode

`auto_parallel_search_mode` is replaced by `search_mode`. `auto_parallel_search_mode` will be deleted in the future MindSpore version. This attribute indicates the algorithm chosen for searching op-level parallelism: `dynamic_programming`, `recursive_programming`, and `sharding_propagation`.

`dynamic_programming` can search for the optimal policy depicted by the cost model, but it takes a long time to search on a huge network model.
`recursive_programming` can quickly search out parallel policies, but the found policies may not have the optimal running performance. `shardin_propagation` requires users to configure sharding strategies for some operators, and the algorithm will then propagate the strategies from configured ops to non-configured ops, with the goal of minimizing [tensor redistribution](https://www.mindspore.cn/docs/programming_guide/en/master/design/distributed_training_design.html#id10) communication. MindSpore allows users to select a search algorithm. The default value is `dynamic_programming`.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(search_mode="dynamic_programming")
context.get_auto_parallel_context("search_mode")
```

#### strategy_ckpt_load_file

Specifies a path to load the segmentation information of all operators with weights in automatic parallel mode.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(strategy_ckpt_load_file="./")
context.get_auto_parallel_context("strategy_ckpt_load_file")
```

#### strategy_ckpt_save_file

Specifies a path for storing the segmentation information of all operators with weights in automatic parallel mode.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(strategy_ckpt_save_file="./")
context.get_auto_parallel_context("strategy_ckpt_save_file")
```

#### full_batch

`full_batch` allows users to determine whether to import datasets in full mode. The default value is False. That is, datasets are imported in data parallel mode. In special scenarios, the performance of full dataset import is better than that of import in data parallel mode. For example, the WideDeep network is used in uneven segmentation scenarios. Therefore, MindSpore provides the `full_batch` configurable interface.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(full_batch=False)
context.get_auto_parallel_context("full_batch")
```

#### pipeline_stages

`pipeline_stages` is used to set parallel stage information of pipeline. It is used to show how the machines are distributed in `auto_parallel` mode in pipeline. Currently pipeline parallel is still under development.

The following is a code example:

```python
from mindspore import context

context.set_auto_parallel_context(pipeline_stages=4)
context.get_auto_parallel_context("pipeline_stages")
```

#### parallel_optimizer_config

`parallel_optimizer_config` is a dict contains the keys and values for setting the parallel optimizer configuration. The configuration provides more detailed behavior control about parallel training
when parallel optimizer is enabled. Currently it supports the key `gradient_accumulation_shard`.
The configuration will be effective when we use context.set_auto_parallel_context(enable_parallel_optimizer=True). It supports the following keys.

- gradient_accumulation_shard: If true, the accumulation gradient parameters will be sharded across the data parallel devices. This will bring in additional communication(ReduceScatter) at each step when accumulate the gradients, but saves a lot of device memory, thus can make model be trained with larger batch size. This configuration is effective only when the model runs on pipeline training or gradient accumulation with data parallel. The default value is True.

```python
from mindspore import context
context.set_auto_parallel_context(parallel_optimizer_config={"gradient_accumulation_shard": True}, enable_parallel_optimizer=True)
```

## Distributed Communication Interface

`mindspore.communication` encapsulates a collection of communication interfaces used by parallel distributed training, facilitating users to configure distributed information.

### init

`init` enables MindSpore communication and initializes distributed training. `init` must be invoked after `context.set_context`. You can transfer the communication backend information to the `init`. The `init` performs initialization based on the backend information.

- `hccl`: short for `Huawei Collective Communication Library`, used for the `Ascend` processor platform.
- `nccl`: short for `NVIDIA Collective Communication Library`, used for the `GPU` processor platform.

If you do not configure the communication backend, MindSpore automatically configures it based on the `device_target` information in `context`.

The following is a code example:

```python
from mindspore import context
from mindspore.communication import init

context.set_context(device_target='GPU')
init()
```

> Under the GPU processor platform, MindSpore also supports starting distributed training without relying on 'OpenMPI', and also uses this interface for distributed training initialization. For specific usage, please refer to [not using OpenMPI training](https://www.mindspore.cn/docs/programming_guide/en/master/distributed_training_gpu.html#openmpi). In this case, when the user does not use 'mpirun' to start the process, but still calls the 'init()' method, MindSpore requires the user to follow [not using OpenMPI training](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_gpu.html#openmpi) to configure several environment variables. If not configured, MindSpore will give a reasonable error prompt. Therefore, it is recommended to call this method only when executing distributed training, and when trying to start distributed training without using 'mpirun', please configure the correct environment variables according to the document.

### get_group_size

`get_group_size` allows users to obtain the number of clusters. Invoke `init` before using the `get_group_size` interface.

The following is a code example:

```python
from mindspore import context
from mindspore.communication import init, get_group_size

context.set_context(device_target='GPU')
init()
group_size = get_group_size()
```

### get_rank

`get_rank` allows users to obtain the ID of the current device in the cluster. Invoke `init` before using the `get_rank` interface.

The following is a code example:

```python
from mindspore import context
from mindspore.communication import init, get_rank

context.set_context(device_target='GPU')
init()
rank_id = get_rank()
```

## Distributed Attribute Configuration

### cross_batch

In specific scenarios, the calculation logic of `data_parallel` is different from that of `stand_alone`. The calculation logic of `auto_parallel` is the same as that of `stand_alone` in any scenario. The convergence effect of `data_parallel` may be better. Therefore, MindSpore provides the `cross_batch` parameter to ensure that the calculation logic of `auto_parallel` is consistent with that of `data_parallel`. You can use the `add_prim_attr` method to configure the logic. The default value is False.

The following is a code example:

```python
import mindspore.ops as ops

mul = ops.Mul().add_prim_attr("cross_batch", True)
```

### fusion

To ensure performance, MindSpore provides the fusion function for the `AllGather` and `AllReduce` operators. Operators of the same type (of the same operator type and in the same communication domain) with the same `fusion` value will be fused together. The value of `fusion` must be greater than or equal to 0. When the value of `fusion` is 0, operators will not be fused together. Only `Ascend` backend is supported.

There are two ways for configuration. If the communication operators are called explicitly, `add_prim_attr` could be used to configure. The following is a code example:

```python
import mindspore.ops as ops

allreduce1 = ops.AllReduce().add_prim_attr("fusion", 1)
allreduce2 = ops.AllReduce().add_prim_attr("fusion", 1)
```

`allreduce1` and `allreduce2` will be fused into one operator during execution.

In `auto_parallel` and `semi_auto_parallel` mode, some communication operators used for parameters or gradients aggregation are inserted automatically. So the attribute should be added on a `Cell` or a `Parameter`. For example:

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

Here the `comm_fusion` of parameter `Net.p1` is 2, which means the attribute `fusion` is 2 for the communication operators generated for this parameter. When you need to manipulate the parameters in batches, it is recommended to call `set_comm_fusion` to set `comm_fusion` for all the parameters in the Net. The value of attribute will be overwritten when the function is called multiply.

> When a parameter is shared, the operators connected with the parameter should have the same data type. Otherwise, fusion would failed.

### layerwise_parallel

In `hybrid_parallel` mode, you need to manually split the model. You need to manually add the `layerwise_parallel` flag to the parallel parameters of the model. The framework filters out the gradient aggregation operation for the parallel parameters of the model based on the flag.

The following is a code example:

```python
import numpy as np
from mindspore import Parameter, Tensor

x = Parameter(Tensor(np.ones([2, 2])), name='weight1', layerwise_parallel=True)
```
