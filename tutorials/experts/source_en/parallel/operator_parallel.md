# Operator-level Parallelism

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/operator_parallel.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

With the development of deep learning, network models are becoming larger and larger, such as trillions of parametric models have emerged in the field of NLP, and the model capacity far exceeds the memory capacity of a single device, making it impossible to train on a single card or data parallel. Operator-level parallelism is used to reduce the memory consumption of individual devices by sharding the tensor involved in each operator in the network model, thus making the training of large models possible.

## Basic Principle

MindSpore models each operator independently, and the user can set the shard strategy for each operator in the forward network (the unset operators are sharded by data parallelism by default).

In the graph construction phase, the framework will traverse the forward graph, and shard and model each operator and its input tensor according to the shard strategy of the operator, such that the compute logic of that operator remains mathematically equivalent before and after the sharding. The framework internally uses Tensor Layout to express the distribution of the input and output tensors in the cluster. The Tensor Layout contains the mapping relationship between the tensor and the device, and the user does not need to perceive how each slice of the model is distributed in the cluster. The framework will automatically schedule the distribution. The framework will also traverse the Tensor Layout of the tensor between adjacent operators. If the output tensor of the previous operator is used as the input tensor of the next operator, and the Tensor Layout of the output tensor in the previous operator is different from that of the input tensor in the next operator, tensor redistribution is required between the two operators. For the training network, after the framework processes the distributed sharding of the forward operator, it can automatically complete the distributed sharding of the inverse operator by relying on the automatic differentiation capability of the framework.

Tensor Layout is used to describe the distribution information about the Tensor in the cluster. Tensor can be sliced into clusters by certain dimensions and can also be replicated on clusters. In the following example, a two-dimensional matrix is sliced into two nodes in three ways: row slicing, column slicing and replication (each slicing corresponds to a Tensor Layout), as shown in the following figure:

If the two-dimensional matrix is sliced to four nodes, there are four types of slices: simultaneously slices both row and column, replication, row slicing + replication, and column slicing + replication, as shown below:

Tensor Redistribution is used to handle the conversion between different Tensor Layout, which can convert the Tensor from one layout to another in the cluster. All redistribution operations are decomposed into combinations of operators such as "set communication+split+concat". The following two figures illustrate several Tensor Redistribution operations.

*Figure: Tensor is sliced to redistribution of two nodes*

*Figure: Tensor is sliced to redistribution of four nodes*

Users can set the sharding strategy of the operator by using the shard() interface, which describes how each dimension of each input tensor of the operator is sliced. For example, MatMul.shard(((a, b), (b, c))) means that MatMul has two input tensors, and the rows of the first input tensor are uniformly sliced in a copies and the columns are uniformly sliced in b copies. The rows of the second input tensor are uniformly sliced in b copies and the columns are uniformly sliced in a copies.

```python
import mindspore.nn as nn
from mindspore import ops
import mindspore as ms

ms.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4)

class DenseMatMulNet(nn.Cell):
    def __init__(self):
        super(DenseMutMulNet, self).__init__()
        self.matmul1 = ops.MatMul.shard(((4, 1), (1, 1)))
        self.matmul2 = ops.MatMul.shard(((1, 1), (1, 4)))
    def construct(self, x, w, v):
        y = self.matmul1(x, w)
        z = self.matmul2(y, v)
        return z
```

In the above example, the user computes two consecutive two-dimensional matrix multiplications on 4 cards: `Z = (X * W) * V` . For the first matrix multiplication `Y = X * W`, the user wants to slice X by rows in 4 parts (i.e. data parallelism), while for the second matrix multiplication `Z = Y * V`, the user wants to slice V by columns in 4 parts (i.e. model parallelism):

Since the Tensor Layout output from the first operator is the 0th dimensional sliced to the cluster, while the second operator requires the first input Tensor to be replicated on the cluster. So in the graph compilation stage, the difference in Tensor Layout between the two operator outputs/inputs is automatically recognized, thus the algorithm for Tensor redistribution is automatically derived. The Tensor redistribution required for this example is an AllGather operator (note: MindSpore AllGather operator automatically merges multiple input Tensors in dimension 0)

## Special Instructions

In operator-level parallelism, to meet the requirements of different scenarios, some operators can configure their distributed implementations through the add_prim_attr() interface, and these configurations are only available for the `SEMI_AUTO_PARALLEL` and `AUTO_PARALLEL` modes:

- Gather operator: add_prim_attr("manual_split", split_tuple). This interface configures non-uniform slicing to the first input of the Gather operator, which is only valid for axis=0. `split_tuple` is a tuple of type int. The sum of the elements must be equal to the length of the 0th dimension in the first input to the Gather operator, and the number of tuples must be equal to the number of slices of the 0th dimension in the first input to the Gather operator.
- Gather operator: add_prim_attr("primitive_target", "CPU"). This interface configures Gather operator to execute on the CPU for heterogeneous scenarios.

## Operation Practices

The following is an illustration of operator-level parallelism by taking an Ascend single 8-card as an example.

### Sample Code Description

> Download the complete sample code here: [operator_parallel](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/operator_parallel).

The directory structure is as follows:

```text
└─sample_code
    ├─operator_parallel
    │      rank_table_8pcs.json
    │      train.py
    │      run.sh
    ...
```

The `rank_table_8pcs.json` is the networking information file to configure the Ascend 8 card environment, the `train.py` file is the script to define the network structure, and `run.sh` is the execution script.

### Configuring the Distributed Environment

The configuration of the distributed environment can be found in: [Configuring Distributed Environment Variables](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#configuring-distributed-environment-variables) tutorial.

### Defining the Network

```python
from mindspore.nn import Cell
from mindspore.ops import operations as ops
import mindspore as ms
from mindspore.common.initializer import initializer


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul().shard(((2, 4), (4, 1)))
        self.weight = ms.Parameter(initializer("normal", [32, 16]), "w1")

        self.relu = ops.ReLU().shard(((8, 1),))

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        return out
```

The above network has two operators, MatMul and ReLU.

The sharding strategy for MatMul is: The rows of the first input are sliced to 2 copies and the columns to 4 copies, while the rows of the second input are sliced to 4 copies and the columns are not sliced. The sharding strategy for ReLU is: the rows of the first input are sliced to 8 copies and the columns are not sliced.

### Running the Code

Using the sample code, an 8-card operator-level parallel training script can be run with the following command:

```bash
sh run.sh 8
```

After execution, the following results can be seen in the log file corresponding to device0:

```bash
epoch: 1 step:1, loss is 23.02248764038086
epoch: 1 step:2, loss is 23.00420570373535
epoch: 1 step:3, loss is 22.97960090637207
epoch: 1 step:4, loss is 22.96306419372558
```
