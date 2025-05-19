# Distributed Training Communication Fusion

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/parallel/comm_fusion.md)

## Overview

In distributed parallel training scenarios to train large-scale parameter models (e.g., GPT-3, Pangu-$\alpha$), data transmission of cross-device or even cross-node is a bottleneck that limits scalability as well as operator power utilization [1]. Communication fusion is an important method to improve network resource utilization and accelerate data transmission efficiency by encapsulating the communication operator of the same source and destination nodes for simultaneous execution to avoid the extra overhead caused by multiple single operator executions.

MindSpore supports the fusion of three common communication operators (`AllReduce`, `AllGather` and `ReduceScatter`) in distributed training, and provides a simple and easy-to-use interface for user configuration. The communication fusion plays an important role in the long and steady training mission support.

### Basic Principle

This section firstly introduces the relationship between computation and communication in distributed training with the example of data parallelism, and secondly introduces the necessity of communication fusion in distributed training scenarios.

#### Computation and Communication in Distributed Training

The whole process of distributed training can be roughly divided into two processes: local model computation and cross-device network data interaction. The following is an example of data parallelism [2] to introduce the overall training process. For other parallel approaches, such as model parallelism [3], pipeline parallelism [4], please refer to related papers.

As shown in the figure below, each node backs up the complete neural network model and uses the local dataset partition to train a mini-batch for forward and backward computation. The gradient obtained from the backward computation is synchronized across the nodes, and the training of the next mini-batch continues after synchronization, and so on, until the accuracy/loss reaches a threshold, or a certain number of epochs are trained. It can be seen that computation and communication alternate in the distributed training process. Work has been done on how to do pipelining of interdependent computation and transmission to reduce the percentage of cross-node data synchronization in the overall training duration [5][6], which will not be repeated here.

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/tutorials/source_zh_cn/parallel/images/data_parallel.png)

#### The Necessity of Communication Fusion

The time overhead of network communication can be measured by the following equation, where $m$ is the size of the data transmission, $\alpha$ is the network transmission rate, and $\beta$ is the inherent overhead of network startup. As can be seen, when the number of transmitted messages becomes larger, the inherent overhead share of network shartup will decrease, transmitting small messages does not make efficient use of network bandwidth resources. Even communication primitives in the HPC domain, such as `AllReduce` and `AllGather`, follow this principle. Therefore, communication fusion technology can effectively improve network resource utilization and reduce network synchronization delay.

$$t = \alpha m+\beta$$

#### Communication Fusion Implementation

Currently, fusion is supported for each of the three communication operators `AllReduce`, `AllGather` and `ReduceScatter` and a control state `openstate` (bool), with the configuration item being a dict type, e.g.

comm_fusion={"openstate": True, "allreduce": {"mode": "auto", "config": None}}, where "mode" has three options:

"auto": Automatic operator fusion according to the data volume threshold of 64MB, with the configuration parameter "config" as None.

"size": Communication operator fusion is performed by manually setting the data volume threshold, with the configuration parameter "config" of type int, in MB.

"index": Only "allreduce" supports the configuration of index, which indicates the way of fusion according to the sequence number of communication operator, and the configuration parameter "config" is of type list. For example, [20, 35], means the first 20 AllReduce are fused into 1, the 20th to 35th AllReduce are fused into 1, and the remaining AllReduce are fused into 1.

### Related Interfaces

MindSpore provides two interfaces to enable communication fusion, each of which is described below:

1. Configuration in parallel scenarios

    ```python
    net = AutoParallel(net, parallel_mode="semi_auto")
    config = {"allreduce": {"mode": "size", "config": 32}, "allgather": {"mode": "size", "config": 32}}
    net.comm_fusion(config=config)
    ```

    In auto-parallel or semi-auto-parallel scenario, the user can utilize the `comm_fusion` parameter provided by this interface to set the parallel strategy when configuring the parallel strategy via `set_auto_parallel_context`, with inputs in the format {"communication_type": {"mode":str, "config": None int or list}}. For details, see `comm_fusion` in [Parallel Configuration](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/mindspore.set_auto_parallel_context.html). This configuration method is preferred in this scenario.

2. Use the interface provided by `Cell`

    Regardless of the parallel mode scenarios, the user can set the index for the parameters in a layer of the model through the `Cell.set_comm_fusion` interface, and MindSpore will fuse the communication operators corresponding to parameters of the same index.

## Operation Practice

### Sample Code Description

> You can download the full sample code here: [distributed_comm_fusion](https://gitee.com/mindspore/docs/tree/r2.6.0/docs/sample_code/distributed_comm_fusion).

The directory structure is as follows:

```text
└─sample_code
    ├─distributed_comm_fusion
        ├── fusion_example_cell.py
        └── run.sh
```

`fusion_example_cell.py` is an example of communication fusion using the interface provided by `Cell` and `run.sh` is the startup script for communication fusion.

### Configuring the Communication Fusion

The following introduces the configuration of two usage methods through the practical sample.

#### `comm_fusion` Parameter

As shown in the following code, the `comm_fusion` parameter of the `set_auto_parallel_context` interface is used to configure the fusion mode for the `AllReduce` operator to be `auto`, implying that the fusion buffer size is set to 64MB by default.

```python
from mindspore.communication import init
from mindspore import nn
import mindspore as ms

net = AutoParallel(net, parallel_mode="semi_auto")
net.comm_fusion(config={"allreduce": {"mode": "auto", "config": None}})
init()
```

If all similar communication operators are fused into one operator, in the current training iteration, the transmission needs to wait until the computation is completely finished before it can be executed, which will cause the device to wait.

In order to avoid the above problem, the network parameters can be fused in groups: while the next group of parameters is computed, the communication of the previous group of parameters is carried out, so that the computation and communication can be hidden from each other, to perform group fusion either by limiting the size of the fusion buffer, or by index partitioning.

For more usage, you can refer to MindSpore [test cases](https://gitee.com/mindspore/mindspore/blob/v2.5.0/tests/ut/python/parallel/test_comm_fusion.py).

> Users can try the size and index modes of `comm_fusion` on their own, which are essentially methods of the fusion buffer class.

#### `Cell.set_comm_fusion` Interface

This method is used in this sample code `fusion_example_cell.py`. As shown in the following code, the `set_comm_fusion` method is called for the instantiated DenseLayer to set the fusion value for each layer.

```python
import mindspore as ms
from mindspore import nn
from mindspore.communication import init
from mindspore.nn.utils import no_init_parameters

ms.set_context(mode=ms.GRAPH_MODE)
init()

class DenseLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.input_mapping = nn.Dense(10, 32)
        self.output_mapping = nn.Dense(32, 10)

    def construct(self, x):
        x = self.input_mapping(x)
        return self.output_mapping(x)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.head = nn.Dense(28*28, 10)
        self.layer1 = DenseLayer()
        self.layer2 = DenseLayer()
        self.layer3 = DenseLayer()

    def construct(self, x):
        x = self.flatten(x)
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Delayed initialization
with no_init_parameters():
    net = Net()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

# Configure communication fusion
net.head.set_comm_fusion(0)
net.layer1.set_comm_fusion(1)
net.layer2.set_comm_fusion(2)
net.layer3.set_comm_fusion(3)
for item in net.trainable_params():
    print(f"The parameter {item.name}'s fusion id is {item.comm_fusion}")
```

### Dataset Loading and Training Process

The dataset loading and training process is consistent with the single-card model, with the following code:

```python
import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn

def create_dataset(batch_size):
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
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss

def train_step(inputs, targets):
    loss_value, grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

# Set parallel
parallel_net = AutoParallel(train_step, parallel_mode="semi_auto")

for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = parallel_net(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed training script as an example, and perform the distributed training:

```bash
bash run.sh
```

After training, the log files are saved in  `log_output/1/rank.*/stdout`, and the example is as follows:

```text
The parameter head.weight's fusion id is 0
The parameter head.bias's fusion id is 0
The parameter layer1.input_mapping.weight's fusion id is 1
The parameter layer1.input_mapping.bias's fusion id is 1
The parameter layer1.output_mapping.weight's fusion id is 1
The parameter layer1.output_mapping.bias's fusion id is 1
The parameter layer2.input_mapping.weight's fusion id is 2
The parameter layer2.input_mapping.bias's fusion id is 2
The parameter layer2.output_mapping.weight's fusion id is 2
The parameter layer2.output_mapping.bias's fusion id is 2
The parameter layer3.input_mapping.weight's fusion id is 3
The parameter layer3.input_mapping.bias's fusion id is 3
The parameter layer3.output_mapping.weight's fusion id is 3
The parameter layer3.output_mapping.bias's fusion id is 3
...
epoch: 0, step: 0, loss is 2.3243194
epoch: 0, step: 10, loss is 2.2858932
epoch: 0, step: 20, loss is 2.2636235
epoch: 0, step: 30, loss is 2.146439
epoch: 0, step: 40, loss is 1.8270943
epoch: 0, step: 50, loss is 1.4588046
epoch: 0, step: 60, loss is 1.2506982
epoch: 0, step: 70, loss is 1.1127701
...
```

The first part represents the fusion index value for particular dense of each layer and the second part represents the Loss result of the training.

## Reference

[1] Xu Y, Lee H J, Chen D, et al. GSPMD: general and scalable parallelization for ML computation graphs[J]. arXiv preprint arXiv:2105.04663, 2021.

[2] Li M, Zhou L, Yang Z, et al. Parameter server for distributed machine learning[C]//Big learning NIPS workshop. 2013, 6: 2.

[3] Dean J, Corrado G, Monga R, et al. Large scale distributed deep networks[J]. Advances in neural information processing systems, 2012, 25.

[4] Narayanan D, Harlap A, Phanishayee A, et al. PipeDream: generalized pipeline parallelism for DNN training[C]//Proceedings of the 27th ACM Symposium on Operating Systems Principles. 2019: 1-15.

[5] Zhang H, Zheng Z, Xu S, et al. Poseidon: An efficient communication architecture for distributed deep learning on {GPU} clusters[C]//2017 USENIX Annual Technical Conference (USENIX ATC 17). 2017: 181-193.

[6] Peng Y, Zhu Y, Chen Y, et al. A generic communication scheduler for distributed dnn training acceleration[C]//Proceedings of the 27th ACM Symposium on Operating Systems Principles. 2019: 16-29.
