# Operator-level Parallelism

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/operator_parallel.md)

## Overview

With the development of deep learning, network models are becoming larger and larger, such as trillions of parametric models have emerged in the field of NLP, and the model capacity far exceeds the memory capacity of a single device, making it impossible to train on a single card or data parallel. Operator-level parallelism is used to reduce the memory consumption of individual devices by sharding the tensor involved in each operator in the network model, thus making the training of large models possible.

> Hardware platforms supported by the operator-level parallel model include Ascend, GPU, and need to be run in Graph mode.

Related interfaces:

1. `mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)`: Sets the semi-automatic parallel mode, which must be called before initializing the network.

2. `mindspore.ops.Primitive.shard()`: Specify the operator slicing strategy, see `Basic Principle` in this chapter for detailed examples.

3. `mindspore.ops.Primitive.add_prim_attr()`: To meet different scenario requirements, some operators can be configured for their distributed implementation via the `add_prim_attr` interface, and these configurations are only available for `SEMI_AUTO_PARALLEL` and `AUTO_PARALLEL` modes, for example:

    - `ops.Gather().add_prim_attr("manual_split", split_tuple)`: This interface configures the first input of the Gather operator to be non-uniformly sliced, which is only valid for axis=0. `split_tuple` is a tuple with elements of type int, the sum of the elements must be equal to the length of the 0th dimension of the first input in the Gather operator, and the number of tuples must be equal to the number of 0th dimensional slices of the first input in the Gather operator.
    - `ops.Gather()add_prim_attr("primitive_target", "CPU")`: This interface configures the Gather operator to execute on the CPU for heterogeneous scenarios.

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

## Operation Practices

The following is an illustration of operator-level parallelism by taking an Ascend or GPU single-machine 8-card as an example.

### Sample Code Description

> Download the complete sample code here: [distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_operator_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── distributed_operator_parallel.py
       └── run.sh
    ...
```

Among them, `distributed_operator_parallel.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

### Configuring the Distributed Environment

Specify the run mode, run device, run card number, etc. through the context interface. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` to be semi-automatic parallel mode, and initialize HCCL or NCCL communication through init. If `device_target` is not set here, it will be automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
ms.set_seed(1)
```

### Loading the Dataset

In the operator-level parallel scenario, the dataset is loaded in the same way as single-card is loaded, with the following code:

```python
import os
import mindspore.dataset as ds

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
```

### Defining the Network

In the current semi-automatic parallel mode, it only supports slicing for ops operators, so here the network needs to be defined with ops operators. Users can manually configure the slicing strategy for some operators based on a single-card network, and the slicing strategy for the rest of the operators can be obtained by derivation, e.g., the network structure after configuring the strategy is:

```python
import mindspore as ms
from mindspore import nn, ops

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul().shard(((2, 4), (4, 1)))
        self.relu1 = ops.ReLU().shard(((4, 1),))
        self.matmul2 = ops.MatMul().shard(((1, 8), (8, 1)))
        self.relu2 = ops.ReLU().shard(((8, 1),))
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
```

The `ops.MatMul()` and `ops.ReLU()` operators for the above networks are configured with slicing strategy, in the case of `ops.MatMul().shard(((2, 4), (4, 1)))`, which has a slicing strategy of: rows of the first input are sliced in 2 parts and columns in 4 parts; rows of the second input are sliced in 4 parts. For `ops. ReLU().shard(((8, 1),))`, its slicing strategy is: the row of the first input is sliced in 8 parts. Note that since the two `ops.ReLU()` here have different slicing strategies, have to be defined twice separately.

### Training the Network

In this step, we need to define the loss function, the optimizer, and the training process, which is the same as that of the single-card:

```python
import mindspore as ms
from mindspore import nn, ops

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

### Running the Single-machine Eight-card Script

Next, the corresponding scripts are invoked by commands, using the `mpirun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory, where part of the file directory structure is as follows:

```text
└─ log_output
    └─ 1
        ├─ rank.0
        |   └─ stdout
        ├─ rank.1
        |   └─ stdout
...
```

The results on the Loss section are saved in `log_output/1/rank.*/stdout`, and example is as follows:

```text
epoch: 0, step: 0, loss is 2.3026192
epoch: 0, step: 10, loss is 2.2928686
epoch: 0, step: 20, loss is 2.279024
epoch: 0, step: 30, loss is 2.2548661
epoch: 0, step: 40, loss is 2.192434
epoch: 0, step: 50, loss is 2.0514572
epoch: 0, step: 60, loss is 1.7082529
epoch: 0, step: 70, loss is 1.1759918
epoch: 0, step: 80, loss is 0.94476485
epoch: 0, step: 90, loss is 0.73854053
epoch: 0, step: 100, loss is 0.71934
...
```

Other startup methods such as dynamic networking and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/experts/en/master/parallel/startup_method.html).