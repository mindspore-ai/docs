# Operator-level Parallelism

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/features/parallel/operator_parallel.md)

## Overview

With the development of deep learning, network models are becoming larger and larger, such as trillions of parametric models have emerged in the field of NLP, and the model capacity far exceeds the memory capacity of a single device, making it impossible to train on a single card or data parallel.

Operator-level parallelism is achieved by slicing the tensor involved in each operator in the network model. Logical data parallelism is used when only the data dimension is sliced, while logical model parallelism is used when only the model dimension is silced. The training of large models is enabled by reducing the memory consumption of a single device.

MindSpore provides two operator-level parallelism capabilities: [Operator-level Parallelism](#basic-principle) and [Higher-order Operator-level Parallelism](#higher-order-operator-level-parallelism). Operator-level Parallelism uses simple tensor dimension splitting strategies to describe tensor distribution, meeting the requirements of most common scenarios. Higher-order Operator-level Parallelism enables complex partitioning scenarios by opening device arrangement descriptions, supporting: Non-contiguous device allocation, Multi-dimensional hybrid partitioning and so on. Both ops and mint operators are supported for the operator-level parallel capability of the two granularities. This chapter only introduces the operator-level parallelism and high-order operator-level parallelism based on ops operators. For the configuration method of operator-level parallelism based on mint operators, please refer to the mint Operator Parallel Practice and Higher-Order mint Operator Parallel Practice in the [Operator-level Parallelism Tutorial](https://www.mindspore.cn/tutorials/en/master/parallel/operator_parallel.html).

For a list of operators that currently support parallelism, see [Usage Constraints During Operator Parallel](https://www.mindspore.cn/docs/en/master/api_python/operator_list_parallel.html).

> Hardware platforms supported by the operator-level parallel model include Ascend, GPU, and need to be run in Graph mode.

Related interfaces:

1. `mindspore.parallel.auto_parallel.AutoParallel(network, parallel_mode="semi_auto")`: Encapsulates the specified parallel mode via static graph parallelism, where `network` is the top-level `Cell` or function to be encapsulated, and `parallel_mode` takes the value `semi_auto`, indicating a semi-automatic parallel mode. The interface returns a `Cell` encapsulated with parallel configuration.

2. `mindspore.ops.Primitive.shard()`: Specify the operator slicing strategy, see [Basic Principle](#basic-principle) in this chapter for detailed examples.

3. `mindspore.ops.Primitive.add_prim_attr()`: To meet different scenario requirements, some operators can be configured for their distributed implementation via the `add_prim_attr` interface, and these configurations are only available for `SEMI_AUTO_PARALLEL` and `AUTO_PARALLEL` modes, for example:

    - `ops.Gather().add_prim_attr("manual_split", split_tuple)`: This interface configures the first input of the Gather operator to be non-uniformly sliced, which is only valid for axis=0. `split_tuple` is a tuple with elements of type int, the sum of the elements must be equal to the length of the 0th dimension of the first input in the Gather operator, and the number of tuples must be equal to the number of 0th dimensional slices of the first input in the Gather operator.
    - `ops.Gather().add_prim_attr("primitive_target", "CPU")`: This interface configures the Gather operator to execute on the CPU for heterogeneous scenarios.
    - `ops.Reshape().add_prim_attr("skip_redistribution")`: Do not apply tensor redistribution (For tensor redistribution, see [Basic Principle](#basic-principle)) before and after ops.Reshape.
    - `ops.ReduceSum().add_prim_attr("cross_batch")`: This interface only supports Reduce operators. When cross_batch is configurated, if the sliced axis is same as the calculated axis of reduce ops, the synchronization will not be added to each cards, which causes different result that is different from that of single card.
    - `ops.TensorScatterUpdate().add_prim_attr("self_define_shard", True)`: When set `self_define_shard` to an operator, input/output layout can config to this operator (whatever this operator supports sharding). However, user needs to ensure the correctness of input/output layout and accuracy of operator.

## Basic Principle

MindSpore models each operator independently, and the user can set the shard strategy for each operator in the forward network (the unset operators are sharded by data parallelism by default).

In the graph construction phase, the framework will traverse the forward graph, and shard and model each operator and its input tensor according to the shard strategy of the operator, such that the compute logic of that operator remains mathematically equivalent before and after the sharding. The framework internally uses Tensor Layout to express the distribution of the input and output tensors in the cluster. The Tensor Layout contains the mapping relationship between the tensor and the device, and the user does not need to perceive how each slice of the model is distributed in the cluster. The framework will automatically schedule the distribution.

In addition, the framework will also traverse the Tensor Layout of the tensor between adjacent operators. If the output tensor of the previous operator is used as the input tensor of the next operator, and the Tensor Layout of the output tensor in the previous operator is different from that of the input tensor in the next operator, tensor redistribution is required between the two operators. For the training network, after the framework processes the distributed sharding of the forward operator, it can automatically complete the distributed sharding of the inverse operator by relying on the automatic differentiation capability of the framework.

Tensor Layout is used to describe the distribution information about the Tensor in the cluster. Tensor can be sliced into clusters by certain dimensions and can also be replicated on clusters. In the following example, a two-dimensional matrix is sliced into two nodes in three ways: row slicing, column slicing and replication (each slicing corresponds to a Tensor Layout), as shown in the following figure:

If the two-dimensional matrix is sliced to four nodes, there are four types of slices: simultaneously slices both row and column, replication, row slicing + replication, and column slicing + replication, as shown below:

Tensor Redistribution is used to handle the conversion between different Tensor Layout, which can convert the Tensor from one layout to another in the cluster. All redistribution operations are decomposed into combinations of operators such as "set communication+split+concat". The following two figures illustrate several Tensor Redistribution operations.

*Figure: Tensor is sliced to redistribution of two nodes*

*Figure: Tensor is sliced to redistribution of four nodes*

Users can set the sharding strategy of the operator by using the shard() interface, which describes how each dimension of each input tensor of the operator is sliced. For example, MatMul().shard(((a, b), (b, c))) means that MatMul has two input tensors, and the rows of the first input tensor are uniformly sliced in a copies and the columns are uniformly sliced in b copies. The rows of the second input tensor are uniformly sliced in b copies and the columns are uniformly sliced in c copies.

```python
import mindspore.nn as nn
from mindspore import ops
from mindspore.parallel.auto_parallel import AutoParallel

class DenseMatMulNet(nn.Cell):
    def __init__(self):
        super(DenseMatMulNet, self).__init__()
        self.matmul1 = ops.MatMul().shard(((4, 1), (1, 1)))
        self.matmul2 = ops.MatMul().shard(((1, 1), (1, 4)))
    def construct(self, x, w, v):
        y = self.matmul1(x, w)
        z = self.matmul2(y, v)
        return z

net = DenseMatMulNet()
paralell_net = AutoParallel(net, parallel_mode='semi_auto')
```

In the above example, the user computes two consecutive two-dimensional matrix multiplications on 4 cards: `Z = (X * W) * V` . For the first matrix multiplication `Y = X * W`, the user wants to slice X by rows in 4 parts (i.e. data parallelism), while for the second matrix multiplication `Z = Y * V`, the user wants to slice V by columns in 4 parts (i.e. model parallelism):

Since the Tensor Layout output from the first operator is the 0th dimensional sliced to the cluster, while the second operator requires the first input Tensor to be replicated on the cluster. So in the graph compilation stage, the difference in Tensor Layout between the two operator outputs/inputs is automatically recognized, thus the algorithm for Tensor redistribution is automatically derived. The Tensor redistribution required for this example is an AllGather operator (note: MindSpore AllGather operator automatically merges multiple input Tensors in dimension 0)

## Higher-order Operator-level Parallelism

The configuration of operator-level parallelism in MindSpore is implemented through mindspore.ops.Primitive.shard() interface, which describes the way each input tensor is sliced through tuples, is suitable for most scenarios and has a simpler configuration process. However, this slicing approach only describes the tensor slicing logic, but hides the specific arrangement of the tensor on the device rank. Therefore, it has limitations in expressing the mapping relationship between tensor slicing and device ranking, and cannot meet the requirements of some complex scenarios.

To cope with these complex scenarios, this tutorial introduces a higher-order operator-level parallel configuration method with an open device arrangement description.

[Operator-level Parallelism](https://www.mindspore.cn/tutorials/en/master/parallel/operator_parallel.html) describes MindSpore basic slicing logic for tensors, but cannot express all the slicing scenarios. For example, for a 2D tensor "[[a0, a1, a2, a3], [a4, a5, a6, a7]]", the tensor layout is shown below:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/features/parallel/images/advanced_operator_parallel_view1.PNG)

*Figure: Schematic of 2D tensor arrangement*

It can be seen that the 0-axis of the tensor, e.g. "[a0, a1, a2, a3]" slices to the discontinuous card "[Rank0, Rank4, Rank2, Rank6]" and the tensor is sliced according to strategy=(2, 4), the arrangement should be as follows:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/features/parallel/images/advanced_operator_parallel_view2.PNG)

*Figure: Schematic of a 2D tensor arranged according to a sharding strategy*

Therefore, directly slicing the input and output tensor of the operator according to the number of slices fails to express some slicing scenarios with special requirements.

### Interface Configuration

In order to express sharding as in the above scenario, functional extensions are made to the [shard](https://www.mindspore.cn/docs/en/master/api_python/parallel/mindspore.parallel.shard.html) interface.

The parameters in_strategy and out_strategy both additionally receive the new quantity type tuple(Layout) type. [Layout](https://www.mindspore.cn/docs/en/master/api_python/parallel/mindspore.parallel.Layout.html) is initialized using the device matrix, while requiring an alias for each axis of the device matrix. For example: "layout = Layout((8, 4, 4), name = ("dp", "sp", "mp"))" means that the device has 128 cards in total, which are arranged in the shape of (8, 4, 4), and aliases "dp", "sp", "mp" are given to each axis.

By passing in the aliases for these axes when calling Layout, each tensor determines which axis of the device matrix each dimension is mapped to based on its shape (shape), and the corresponding number of slice shares. For example:

- "dp" denotes 8 cuts within 8 devices in the highest dimension of the device layout.
- "sp" denotes 4 cuts within 4 devices in the middle dimension of the device layout.
- "mp" denotes 4 cuts within 4 devices in the lowest dimension of the device layout.

In particular, one dimension of the tensor may be mapped to multiple dimensions of the device to express multiple slices in one dimension.

The above example of "[[a0, a1, a2, a3], [a4, a5, a6, a7]]" sliced to discontinuous cards can be expressed by Layout as follows:

```python
# a = [[a0, a1, a2, a3], [a4, a5, a6, a7]]
from mindspore import Layout
layout = Layout((2, 2, 2), alias_name = ("dp", "sp", "mp"))
a_strategy = layout("mp", ("sp", "dp"))
```

It can be seen that the "[a0, a1, a2, a3]" of the tensor a is sliced twice to the "sp" and "dp" axes of the device, so that the result comes out as:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/features/parallel/images/advanced_operator_parallel_view1.PNG)

The following is exemplified by a concrete example in which the user computes a two-dimensional matrix multiplication over 8 cards: `Y = (X * W)` , where the devices are organized according to `2 * 2 * 2`, and the cut of X coincides with the cut of the tensor a. The code is as follows:

```python
import mindspore.nn as nn
from mindspore import ops, Layout
from mindspore.parallel.auto_parallel import AutoParallel

class DenseMatMulNet(nn.Cell):
    def __init__(self):
        super(DenseMatMulNet, self).__init__()
        layout = Layout((2, 2, 2), alias_name = ("dp", "sp", "mp"))
        in_strategy = (layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None"))
        out_strategy = (layout(("mp", "sp", "dp"), "None"), )
        self.matmul1 = ops.MatMul().shard(in_strategy, out_strategy)
    def construct(self, x, w):
        y = self.matmul1(x, w)
        return y

net = DenseMatMulNet()
paralell_net = AutoParallel(net, parallel_mode='semi_auto')
```