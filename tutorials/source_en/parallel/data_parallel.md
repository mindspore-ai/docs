# Data Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/parallel/data_parallel.md)

## Overview

Data parallel is the most commonly used parallel training approach for accelerating model training and handling large-scale datasets. In data parallel mode, the training data is divided into multiple copies and then each copy is assigned to a different compute node, such as multiple cards or multiple devices. Each node processes its own subset of data independently and uses the same model for forward and backward propagation, and ultimately performs model parameter updates after synchronizing the gradients of all nodes.

The following is an illustration of data parallel operation using the Ascend single-machine 8-card as an example:

## Sample Code Description

> You can download the full sample code here: [distributed_data_parallel](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/distributed_data_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_data_parallel
       ├── distributed_data_parallel.py
       └── run.sh
    ...
```

Among them, `distributed_data_parallel.py` is the script that defines the network structure and training process. `run.sh` is the execution script.

## Configuring Distributed Environments

The context interface allows you to specify the run mode, run device, run card number. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` for data parallel mode and initialize HCCL, NCCL or MCCL communication through init according to different device targets. In data parallel mode, you also can set `gradients_mean` to specify the gradient aggregation method. If `device_target` is not set here, it is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)
```

`gradients_mean=True` is for the fact that during the backward computation, the framework internally aggregates the gradient values of the data parallel parameters scattered across multiple machines to get the global gradient values before passing them into the optimizer for updating. Framework will do the reduce sum for gradient by using `AllReduce(op=ReduceOp.SUM)`, then calculate the mean value in terms of the value of `gradients_mean`. (If it is set to true, the mean value is calculated, otherwise, the mean value is not calculated. Default: False).

## Loading Datasets

The biggest difference between the data parallel mode and other modes is the different data loading way. The data is imported in a parallel way. Below we take the MNIST dataset as an example to introduce the method of importing MNIST dataset in data parallel mode. `dataset_path` is the path of the dataset.

```python
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

rank_id = get_rank()
rank_size = get_group_size()
dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
```

Unlike single-card, the `num_shards` and `shard_id` parameters need to be passed in the dataset interface, corresponding to the number of cards and the logical serial number, respectively, and it is recommended to obtain them through the `mindspore.communication` interface:

- `get_rank`: Obtain the ID of the current device in the cluster.
- `get_group_size`: Obtain the number of clusters.

> When loading datasets for data parallel scenarios, it is recommended that the same dataset file be specified for each card. If different datasets are loaded for each card, the computational accuracy may be affected.

The complete data processing code:

```python
import os
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

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

## Defining Network

In data parallel mode, the network is defined in the same way as single-card network, and the main structure of the network is as follows:

```python
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

net = Network()
```

## Training Network

In this step, we need to define the loss function, the optimizer, and the training process. The difference with single-card model is that the data parallel mode also requires the addition of the `mindspore.nn.DistributedGradReducer()` interface to aggregate the gradients of all cards. The first parameter of the network is the network parameter to be updated:

```python
from mindspore import nn
import mindspore as ms

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(net.trainable_params(), 1e-2)

def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

for epoch in range(10):
    i = 0
    for data, label in data_set:
        (loss, _), grads = grad_fn(data, label)
        grads = grad_reducer(grads)
        optimizer(grads)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss))
        i += 1
```

> This can also be trained using [Model.train](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/train/mindspore.train.Model.html#mindspore.train.Model.train).

## Running Single-machine Eight-card Script

Next, the corresponding scripts are invoked by commands, using the `mpirun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory, and the part of the file directory structure is as follows:

```text
└─ log_output
    └─ 1
        ├─ rank.0
        |   └─ stdout
        ├─ rank.1
        |   └─ stdout
...
```

The part results of the Loss are saved in `log_output/1/rank.*/stdout`. The example is as follows:

```text
epoch: 0 step: 0, loss is 2.3026438
epoch: 0 step: 50, loss is 2.2963896
epoch: 0 step: 100, loss is 2.2882829
epoch: 0 step: 150, loss is 2.2822685
...
```

Other startup methods such as `mpirun` and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/startup_method.html).
