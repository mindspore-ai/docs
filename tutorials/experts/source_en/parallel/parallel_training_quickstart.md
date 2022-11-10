# Quick Start Distributed Parallel Training

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/parallel_training_quickstart.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial shows how to perform MindSpore distributed parallel training in a single 8-card **GPU** environment via **OpenMPI** with a simple example of a single hidden layer fully connected neural network.

A tutorial on distributed parallel training of ResNet networks on a GPU platform is available at [Sample Distributed Parallel Training Basics (GPU)](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html). In contrast: (1) the example uses a more complex ResNet network; (2) in addition to pull-up training by using OpenMPI, the example also introduces pull-up training by using a scripted approach.

> You can download the complete sample code here:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training_quickstart>

The directory structure is as follows:

```text
└─sample_code
    ├─distributed_training_quickstart
        ├── net.py
        ├── run_with_mpi.sh
    ...
```

where `net.py` is the network definition script and `run_with_mpi.sh` is the execution script.

> In addition, tutorials for distributed parallel training on Ascend 910 platform are available in [Distributed Parallel Training Example (Ascend)](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html) and [Distributed Parallel Training of Transformer Models](https://www.mindspore.cn/tutorials/experts/en/master/parallel/transformer.html).

## Preparation

### Datasets

This sample example constructs a random set of input data and labels, with the following code:

```python
import numpy as np

def get_dataset(batch_size, in_dim, out_dim, step_per_epoch):
    np.random.seed(1)
    input_data = np.random.rand(batch_size, in_dim).astype(np.float32)
    label_data = np.random.rand(batch_size, out_dim).astype(np.float32)
    def generate():
        for _ in range(step_per_epoch):
            yield (input_data, label_data)
    return generate
```

where `step_per_epoch` is the number of steps performed per epoch for training, `batch_size` is the batch size, `in_dim` is the input vector length, and `out_dim` is the output vector length.

### Network Structure

The network code used in this sample is as follows:

```python
class Net(Cell):
    """define net"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.weight = Parameter(initializer("normal", [self.in_dim, self.hidden_dim]), "w")
        self.weight2 = Parameter(initializer("normal", [self.hidden_dim, self.out_dim]), "w2")
        self.matmul = ops.MatMul()
        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        out = self.matmul2(out, self.weight2)
        return out
```

where `in_dim` is the network input dimension, `out_dim` is the output dimension, which needs to match the data dimension, and `hidden_dim` is the number of nodes in the hidden layer of the network.

## Semi-automatic Parallel Distributed Training via OpenMPI

### OpenMPI Environment Configuration

[OpenMPI](https://www.open-mpi.org/) is a high-performance messaging library, a multi-process communication library adopted by MindSpore. For the related environment configuration, see [Running the Script through OpenMPI](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#running-the-script-through-openmpi).

> In addition, MindSpore also supports distributed training without relying on OpenMPI. For the details, see [Training without Relying on OpenMPI](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#training-without-relying-on-openmpi).

### Semi-automatic Parallelism

Currently MindSpore supports four parallel modes, and see [Distributed Parallel Training Modes](https://www.mindspore.cn/tutorials/experts/en/master/parallel/introduction.html#distributed-parallel-training-mode-1) for details.

This example demonstrates fully automatic parallelism, which is achieved by configuring `parallel_mode=ms.ParallelMode.AUTO_PARALLEL` through the `set_auto_parallel_context()` interface.
There are three configurable parallel strategy search algorithms under fully automatic parallelism, see: [Fully automatic parallelism](https://www.mindspore.cn/tutorials/experts/en/master/parallel/introduction.html#fully-automatic-parallelism) for details. In this example, the **sharding strategy propagation algorithm** is selected, which is implemented by configuring `search_mode="sharding_propagation"` through the `set_auto_parallel_context()` interface, and manually setting the `matmul` operator sharding strategy. The sharding strategy of other operators is given by the parallel strategy search algorithm automatically. The code is as follows:

```python
class Net(Cell):
    """define net"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.weight = Parameter(initializer("normal", [self.in_dim, self.hidden_dim]), "w")
        self.weight2 = Parameter(initializer("normal", [self.hidden_dim, self.out_dim]), "w2")

        # Set the sharding strategy manually for the matmul operator
        # where (2, 4) means that the input data of matmul operator is sliced into two parts in batch dimension and four parts in width dimension
        # (4, 1) indicates that the weights of the matmul operator are sliced into four parts in the HEIGHT dimension
        self.matmul = ops.MatMul().shard(((2, 4), (4, 1)))

        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        out = self.matmul2(out, self.weight2)
        return out
```

where the `shard()` method is described in detail in [Principles of Automatic Parallelism](https://www.mindspore.cn/docs/en/master/design/distributed_training_design.html#principle-of-automatic-parallelism). The inference introduction is in [functional operator sharding](https://www.mindspore.cn/tutorials/experts/en/master/parallel/pynative_shard_function_parallel.html)

For the parallel sharding strategy set in the above example, the `matmul` operator computation process for the forward propagation process in a single-machine 8-card environment is schematically shown as follows:

![image](https://gitee.com/mindspore/docs/raw/master/tutorials/experts/source_zh_cn/parallel/images/matmul_shard.png)

The top half of the diagram shows the data sharding, and the bottom half shows the calculation and communication process performed by each GPU card at logical number (rank) 0-7.

#### Code Running

In this example, the loss function, optimizer and training procedure are defined similarly to single card training, with the following code:

```python
var_step_per_epoch = 4
var_single_batch_size = 2
var_in_dim = 32
var_hidden_dim = 16
var_out_dim = 16

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", save_graphs=True, save_graphs_path="../saved_graph")

# Single-machine 8-card environment. Parallel mode is fully automatic parallelism, and strategy search is set to strategy propagation algorithm
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation", dataset_strategy="data_parallel")

# Initialize the communication environment and get the logical serial number of the current card, i.e. rank_id
init("nccl")
rank_id = get_rank()

# Randomly constructed datasets
fake_dataset = get_dataset(var_single_batch_size, var_step_per_epoch, var_in_dim, var_out_dim)
dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

# Define the network structure
net = Net(var_in_dim, var_hidden_dim, var_out_dim)

# Define the loss function and callback
loss = MSELoss()
callback = [LossMonitor(), ModelCheckpoint(directory="{}".format(rank_id))]

# Define the optimizer
learning_rate = 0.2
momentum = 0.1
epoch_size = 5
opt = Momentum(net.trainable_params(), learning_rate, momentum)

# Model training
model = Model(net, loss_fn=loss, optimizer=opt)
model.train(epoch_size, dataset, callbacks=callback, dataset_sink_mode=False)
```

Training can be performed with `mpirun` command of OpenMPI, as specified in the script `run_with_mpi.sh`.

After running, the script is performed in the background and the training log is saved in the `. /device` directory, and the model of the card with the logical number `rank_id` is saved in the `. /device/{rank_id}` directory.

In addition, `save_graphs=True` is configured via the `ms.set_context()` interface to save the model intermediate representation `MindIR`, and the `MindIR` of the card with the logical number `rank_id` is saved in the `. /saved_graph/{rank_id}` directory. MindSpore IR (MindIR) is a program representation between the source language and the target language during the compilation of MindSpore framework programs to facilitate program analysis and optimization by the compiler, see [MindIR](https://www.mindspore.cn/docs/en/master/design/mindir.html).

#### Verification

After running the `run_with_mpi.sh` script, the recorded loss should decrease, e.g.

```text
# ./device/train.log: #
...
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 2, loss is 0.367389976978302
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 3, loss is 0.35383114218711853
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 3 step: 4, loss is 0.3312329947948456
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 1, loss is 0.295515775680542
epoch: 4 step: 2, loss is 0.2440134435892105
epoch: 4 step: 2, loss is 0.2440134435892105
epoch: 4 step: 2, loss is 0.2440134435892105
epoch: 4 step: 2, loss is 0.2440134435892105
epoch: 4 step: 2, loss is 0.2440134435892105
epoch: 4 step: 2, loss is 0.2440134435892105
epoch: 4 step: 2, loss is 0.2440134435892105
epoch: 4 step: 2, loss is 0.2440134435892105
...
```

You can check the configuration of the sharding strategy for each operator in `. /saved_graph/rank_x/step_parallel_begin_xxxx.ir` to see the configuration of the sharding strategy for each operator, e.g.

```text
# ./saved_graph/rank_0/step_parallel_begin_0041.ir: #
...
%3(out) = MatMul(%1, %2) {instance name: matmul} primitive_attrs: {input_names: [x1, x2], out_strategy: None, transpose_x2: false, transpose_b: false, in_strategy: ((2, 4), (4, 1)), output_names: [output], transpose_a: false, transpose_x1: false} {in_strategy: ((2, 4), (4, 1))}
    : (<Tensor[Float32], (16, 32)>, <Tensor[Float32], (32, 16)>) -> (<Tensor[Float32], (16, 16)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
%4(out) = ReLU(%3) {instance name: relu} primitive_attrs: {output_names: [output], input_names: [x]} {in_strategy: ((2, 4))}
    : (<Tensor[Float32], (16, 16)>) -> (<Tensor[Float32], (16, 16)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
%5([CNode]472) = Load($(@1_construct_wrapper.337:para4_w2), %para12_u)
    : (<Ref[Tensor(F32)], (16, 16), ref_key=:w2>, <UMonad>) -> (<Tensor[Float32], (16, 16)>)
    # scope: (Default/network-WithLossCell)
%6(out) = MatMul(%4, %5) {instance name: matmul2} primitive_attrs: {output_names: [output], transpose_a: false, input_names: [x1, x2], transpose_x2: false, transpose_x1: false, transpose_b: false} {in_strategy: ((2, 4), (4, 1))}
    : (<Tensor[Float32], (16, 16)>, <Tensor[Float32], (16, 16)>) -> (<Tensor[Float32], (16, 16)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
...
```

It can be seen that the `relu` operator corresponding to the `%4(out)` line and the `matmul2` operator corresponding to the `%6(out)` line are automatically configured with a sharding strategy.

Further, you can view `. /saved_graph/rank_x/18_execute_xxxx.ir` to see the actual execution of the slice operator dimension for each card, e.g.

```text
# ./saved_graph/rank_0/18_execute_0185.ir: #
...
%12(equivout) = MatMul(%10, %11) {instance name: matmul} primitive_attrs: {input_names: [x1, x2], out_strategy: None, transpose_x2: false, transpose_b: false, in_strategy: ((2, 4), (4, 1)), output_names: [output], transpose_a: false, transpose_x1: false} {in_strategy: ((2, 4), (4, 1))}
    : (<Tensor[Float32], (8, 8)>, <Tensor[Float32], (8, 16)>) -> (<Tensor[Float32], (8, 16)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
    # In file /home/jenkins/my_dir/parallel_training_quick_start/device/./matmul.py(45)/        out = self.matmul(x, self.weight)/
    # In file /home/miniconda3/envs/my_env/lib/python3.9/site-packages/mindspore/nn/wrap/cell_wrapper.py(114)/        out = self._backbone(data)/
    # In file /home/miniconda3/envs/my_env/lib/python3.9/site-packages/mindspore/nn/wrap/cell_wrapper.py(376)/        loss = self.network(*inputs)/
%13(equiv[CNode]520) = AllReduce(%12) {instance name: forward_op_11795743325248501408} primitive_attrs: {group: 4-6301172352641561019, fusion: 0, op: sum, rank_list: (0, 1, 2, 3), group_ranks: 0-1-2-3, index: 0, group_rank_ids: (0, 1, 2, 3), no_eliminate: true} cnode_attrs: {comm_reuse: true}
    : (<Tensor[Float32], (8, 16)>) -> (<Tensor[Float32], (8, 16)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
%14(equiv[CNode]519) = StridedSlice(%13, (0, 0), (8, 4), (1, 1)) {instance name: redistribution_op_16390315056374637535StridedSlice} primitive_attrs: {new_axis_mask: 0, shrink_axis_mask: 0, end_mask: 0, input_names: [x, begin, end, strides], output_names: [output], keep_value_node_input: true, begin_mask: 0, ellipsis_mask: 0}
    : (<Tensor[Float32], (8, 16)>, <Tuple[Int64*2], sequence_nodes={node={ValueNode<ValueTuple> (0, 0), elements_use_flags: {ptr: 0x560e8fef5fa0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (0, 0), elements_use_flags: {ptr: 0x560e8fef5fa0, value: [const vector][1, 1]}}, node={<freed node>}, node={ValueNode<ValueTuple> (0, 0), elements_use_flags: {ptr: 0x560e8fef5fa0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (0, 0), elements_use_flags: {ptr: 0x560e8fef5fa0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (0, 0), elements_use_flags: {ptr: 0x560e8fef5fa0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (0, 0), elements_use_flags: {ptr: 0x560e8fef5fa0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (0, 0), elements_use_flags: {ptr: 0x560e8fef5fa0, value: [const vector][1, 1]}}, node={<freed node>}}>, <Tuple[Int64*2], sequence_nodes={node={ValueNode<ValueTuple> (8, 4), elements_use_flags: {ptr: 0x560e8fed50d0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (8, 4), elements_use_flags: {ptr: 0x560e8fed50d0, value: [const vector][1, 1]}}, node={<freed node>}, node={ValueNode<ValueTuple> (8, 4), elements_use_flags: {ptr: 0x560e8fed50d0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (8, 4), elements_use_flags: {ptr: 0x560e8fed50d0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (8, 4), elements_use_flags: {ptr: 0x560e8fed50d0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (8, 4), elements_use_flags: {ptr: 0x560e8fed50d0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (8, 4), elements_use_flags: {ptr: 0x560e8fed50d0, value: [const vector][1, 1]}}, node={<freed node>}}>, <Tuple[Int64*2], sequence_nodes={node={ValueNode<ValueTuple> (1, 1), elements_use_flags: {ptr: 0x560e8ffb4ff0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (1, 1), elements_use_flags: {ptr: 0x560e8ffb4ff0, value: [const vector][1, 1]}}, node={<freed node>}, node={ValueNode<ValueTuple> (1, 1), elements_use_flags: {ptr: 0x560e8ffb4ff0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (1, 1), elements_use_flags: {ptr: 0x560e8ffb4ff0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (1, 1), elements_use_flags: {ptr: 0x560e8ffb4ff0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (1, 1), elements_use_flags: {ptr: 0x560e8ffb4ff0, value: [const vector][1, 1]}}, node={ValueNode<ValueTuple> (1, 1), elements_use_flags: {ptr: 0x560e8ffb4ff0, value: [const vector][1, 1]}}, node={<freed node>}}>) -> (<Tensor[Float32], (8, 4)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
%15(equivout) = ReLU(%14) {instance name: relu} primitive_attrs: {output_names: [output], input_names: [x]} {in_strategy: ((2, 4))}
    : (<Tensor[Float32], (8, 4)>) -> (<Tensor[Float32], (8, 4)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
    # In file /home/jenkins/my_dir/parallel_training_quick_start/device/./matmul.py(46)/        out = self.relu(out)/
    # In file /home/miniconda3/envs/my_env/lib/python3.9/site-packages/mindspore/nn/wrap/cell_wrapper.py(114)/        out = self._backbone(data)/
    # In file /home/miniconda3/envs/my_env/lib/python3.9/site-packages/mindspore/nn/wrap/cell_wrapper.py(376)/        loss = self.network(*inputs)/
%16(equiv[CNode]472) = Load(%para4_w2, U)
    : (<Ref[Tensor(F32)], (4, 16), ref_key=:w2>, <UMonad>) -> (<Tensor[Float32], (4, 16)>)
    # scope: (Default/network-WithLossCell)
%17(equivout) = MatMul(%15, %16) {instance name: matmul2} primitive_attrs: {output_names: [output], transpose_a: false, input_names: [x1, x2], transpose_x2: false, transpose_x1: false, transpose_b: false} {in_strategy: ((2, 4), (4, 1))}
    : (<Tensor[Float32], (8, 4)>, <Tensor[Float32], (4, 16)>) -> (<Tensor[Float32], (8, 16)>)
    # scope: (Default/network-WithLossCell/_backbone-Net)
    # In file /home/jenkins/my_dir/parallel_training_quick_start/device/./matmul.py(47)/        out = self.matmul2(out, self.weight2)/
    # In file /home/miniconda3/envs/my_env/lib/python3.9/site-packages/mindspore/nn/wrap/cell_wrapper.py(114)/        out = self._backbone(data)/
    # In file /home/miniconda3/envs/my_env/lib/python3.9/site-packages/mindspore/nn/wrap/cell_wrapper.py(376)/        loss = self.network(*inputs)/
...
```

It can be seen that the dimension of the `matmul` operator corresponding to the `%12(equivout)` line is the same as that shown in the figure.
