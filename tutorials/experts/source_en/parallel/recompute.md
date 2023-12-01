# Recomputation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_en/parallel/recompute.md)

## Overview

The automatic differential of MindSpore is in reverse-mode, which derives the backward pass according to the forward pass. Before some backward operators are computed, the results of some forward operators should be ready. It leads to the problem that the memory occupied by these results of the forward operators, can not be reused until the computation of the backward operators are completed. This problem can  drive up the peak of memory, which is particularly significant in the large model.

In order to solve this problem, Mindspore provides the recomputation function. It will recompute the forward operators before computing the backward operators rather than storing the results of forward operators, which can help the memory be reused. This tutorial takes the model ResNet-50 for example to explain how to configure recomputation to train your model in MindSpore.

Related interfaces:

1. `mindspore.nn.Cell.recompute()`: Call the [recompute interface](https://www.mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.recompute). After calling this interface, when computing the reverse part, all the operators inside the Cell and all the operators inside the sub-Cells are recomputed, except for the output operator of that Cell.

2. `mindspore.ops.Primitive.recompute()`: Call the [recompute interface](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive.recompute) of `Primitive`. After calling this interface, the operator is recomputed when computing the reverse part.

## Basic Principle

MindSpore automatically derives the reverse graph according to the forward graph compute process, and the forward graph and the inverse graph together form a complete compute graph. When calculating some reverse operators, it may be necessary to use the compute results of some forward operators, resulting in the compute results of these forward operators, which need to reside in memory until these reverse operators are computed, and the memory they occupy will not be reused by other operators. The computational results of these forward operators, which reside in memory for a long time, push up the peak memory footprint of the computation, especially in large-scale network models.

In order to reduce memory peaks, the recompute technique can not save the compute results of the forward activation layer, so that the memory can be reused, and then when calculating the reverse part, recompute the results of the forward activation layer. MindSpore provides the ability to recompute.

The recompute function is implemented as a forward operator that is recomputed according to the user's specified needs, copies the same operator, outputs it to the reverse operator, and deletes the continuous edge relationship between the original forward operator and the reverse operator. In addition, we need to ensure that the copied operator only begins to be evaluated when the corresponding inverse part is computed, so we need to insert control dependencies to ensure the order in which the operators are executed. As shown in the following figure:

![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/tutorials/experts/source_zh_cn/parallel/images/recompute_image_0_zh.png)

*Figure: Forward and reverse diagram before and after the recompute function is enabled*

For user convenience, MindSpore currently provides not only a recompute interface for individual operators, but also a recompute interface for Cell. When the user calls The Cell's recompute interface, all forward operators in the Cell are set to recompute.

Taking the GPT-3 model as an example, the policy is set to recalculate the cell corresponding to the layerer for each layer, and then the output operator of the layerer is set to non-recompute. The effect of recompute on the 72-layer GPT-3 network is shown in the following figure:

![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/tutorials/experts/source_zh_cn/parallel/images/recompute_image_1_zh.png)

*Figure: Comparison of GPT-3 memory usage before and after recalculation function is enabled*

## Operation Practice

The following is an illustration of the recomputation operation using an Ascend or GPU stand-alone 8-card as an example:

### Sample Code Description

> Download the complete sample code: [recompute](https://gitee.com/mindspore/docs/tree/r2.2/docs/sample_code/recompute).

The directory structure is as follows:

```text
└─ sample_code
    ├─ recompute
       ├── train.py
       └── run.sh
    ...
```

`train.py` is the script that defines the network structure and inference. `run.sh` is the execution script.

### Configuring a Distributed Environment

Specify the run mode, run device, run card number via the context interface. The parallel mode is data parallel and HCCL or NCCL communication is initialized by init. Setting `save_graphs=2` prints out the computational graph structure for comparison. `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE, save_graphs=2)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)
```

### Loading the Dataset

Here the dataset is loaded in data parallel mode, specifying the `num_shards` and `shard_id` parameters, corresponding to the number of cards and the logical serial number, respectively, with the following code:

```python
import os
import mindspore.dataset as ds
from mindspore import nn

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    rank_id = get_rank()
    rank_size = get_group_size()
    dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
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

### Network Definition

The network configures the activation function operator with recomputation based on the single-card model to reduce the memory footprint:

```python
from mindspore import nn, ops

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = ops.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = ops.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
# Configure the recompute of relu operator
net.relu1.recompute()
net.relu2.recompute()
```

### Training the Network

In this step, we need to define the loss function, the optimizer, and the training process, and this part is consistent with the data-parallel model:

```python
from mindspore import nn, ops

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

for epoch in range(1):
    i = 0
    for image, label in data_set:
        (loss_value, _), grads = grad_fn(image, label)
        grads = grad_reducer(grads)
        optimizer(grads)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_value))
        i += 1
```

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed training script as an example, and perform the distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory, and by setting context: `save_graphs=2` in `train.py`, you can print out the IR graphs of the compilation process, where some of the file directories are structured as follows:

```text
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ rank_0
|   ├─ xx_validate_xxx.ir
|   ...
├─ rank_1
|   ├─ xx_validate_xxx.ir
|   ...
...
```

The results on the Loss section are saved in `log_output/1/rank.*/stdout`, and the example is as below:

```text
epoch: 0, step: 0, loss is 2.2929618
epoch: 0, step: 10, loss is 2.2396836
epoch: 0, step: 20, loss is 2.2097976
epoch: 0, step: 30, loss is 2.1942225
epoch: 0, step: 40, loss is 2.0986974
epoch: 0, step: 50, loss is 2.0612597
...
```

Computation graph results is in `xx_validate_xxx.ir` before setting up the recomputation:

```text
...
  %81(1285) = MatMul(%80, %11) primitive_attrs: {output_names: (output), transpose_a: Bool(0), input_names: (x1, x2), transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} cnode_primal_attrs: {forward_node_name: "MatMul_24422", forward_unique_id: "24422"}
      : (<Tensor[Float32], (32, 10)>, <Tensor[Float32], (512, 10)>) -> (<Tensor[Float32], (32, 512)>)
...
  %82(1286) = ReluGrad(%81, %10) primitive_attrs: {output_names: (output), input_names: (x)} cnode_primal_attrs: {forward_node_name: "ReLU_24405", forward_unique_id: "24405"}
      : (<Tensor[Float32], (32, 512)>, <Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (32, 512)>)
...
  %83(1285) = MatMul(%82, %6) primitive_attrs: {output_names: (output), transpose_a: Bool(0), input_names: (x1, x2), transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} cnode_primal_attrs: {forward_node_name: "MatMul_24434", forward_unique_id: "24434"}
      : (<Tensor[Float32], (32, 512)>, <Tensor[Float32], (512, 512)>) -> (<Tensor[Float32], (32, 512)>)
...
  %84(1286) = ReluGrad(%83, %5) primitive_attrs: {output_names: (output), input_names: (x)} cnode_primal_attrs: {forward_node_name: "ReLU_24408", forward_unique_id: "24408"}
      : (<Tensor[Float32], (32, 512)>, <Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (32, 512)>)
...
  %85(1285) = MatMul(%0, %84) primitive_attrs: {output_names: (output), transpose_a: Bool(1), input_names: (x1, x2), transpose_x2: Bool(0), transpose_x1: Bool(1), transpose_b: Bool(0)} cnode_primal_attrs: {forward_node_name: "MatMul_24446", forward_unique_id: "24446"}
      : (<Tensor[Float32], (32, 784)>, <Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (784, 512)>)
...
```

After setting the recomputation:

```text
...
  %81(1285) = MatMul(%80, %11) primitive_attrs: {output_names: (output), transpose_a: Bool(0), input_names: (x1, x2), transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} cnode_primal_attrs: {forward_node_name: "MatMul_24422", forward_unique_id: "24422"}
      : (<Tensor[Float32], (32, 10)>, <Tensor[Float32], (512, 10)>) -> (<Tensor[Float32], (32, 512)>)
...
  %84([CNode]1292) = ReLU(%83) {instance name: relu2} primitive_attrs: {output_names: [output], input_names: [x], recompute: Bool(1)} cnode_attrs: {recompute_sub_graph: U64(1), recompute_id: I64(2), duplicated: Bool(1), need_cse_after_recompute: Bool(1)}
      : (<Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (32, 512)>)
      # Scope: (Default)
  %85([CNode]1293) = ReluGrad(%81, %84) primitive_attrs: {output_names: (output), input_names: (x)} cnode_attrs: {recompute_sub_graph: U64(1), target_grad: Bool(1)} cnode_primal_attrs: {forward_node_name: "ReLU_24405", forward_unique_id: "24405"}
      : (<Tensor[Float32], (32, 512)>, <Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (32, 512)>)
...
  %86(1285) = MatMul(%85, %6) primitive_attrs: {output_names: (output), transpose_a: Bool(0), input_names: (x1, x2), transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} cnode_primal_attrs: {forward_node_name: "MatMul_24434", forward_unique_id: "24434"}
      : (<Tensor[Float32], (32, 512)>, <Tensor[Float32], (512, 512)>) -> (<Tensor[Float32], (32, 512)>)
...
  %89([CNode]1296) = ReLU(%88) {instance name: relu2} primitive_attrs: {output_names: [output], input_names: [x], recompute: Bool(1)} cnode_attrs: {recompute_sub_graph: U64(0), recompute_id: I64(1), duplicated: Bool(1), need_cse_after_recompute: Bool(1)}
      : (<Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (32, 512)>)
      # Scope: (Default)
  %90([CNode]1297) = ReluGrad(%86, %89) primitive_attrs: {output_names: (output), input_names: (x)} cnode_attrs: {recompute_sub_graph: U64(0), target_grad: Bool(1)} cnode_primal_attrs: {forward_node_name: "ReLU_24408", forward_unique_id: "24408"}
      : (<Tensor[Float32], (32, 512)>, <Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (32, 512)>)
...
  %91(1285) = MatMul(%0, %90) primitive_attrs: {output_names: (output), transpose_a: Bool(1), input_names: (x1, x2), transpose_x2: Bool(0), transpose_x1: Bool(1), transpose_b: Bool(0)} cnode_primal_attrs: {forward_node_name: "MatMul_24446", forward_unique_id: "24446"}
      : (<Tensor[Float32], (32, 784)>, <Tensor[Float32], (32, 512)>) -> (<Tensor[Float32], (784, 512)>)
...
```

It can be seen that the `ReLU` operator is copied out in one copy as input to the reverse operator `ReluGrad`.
