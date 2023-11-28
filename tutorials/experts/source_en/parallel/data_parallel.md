# Data Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/data_parallel.md)

## Overview

Data parallel is the most commonly used parallel training approach for accelerating model training and handling large-scale datasets. In data parallel mode, the training data is divided into multiple copies and then each copy is assigned to a different compute node, such as multiple cards or multiple devices. Each node processes its own subset of data independently and uses the same model for forward and backward propagation, and ultimately performs model parameter updates after synchronizing the gradients of all nodes.

> Hardware platforms supported for data parallelism include Ascend, GPU and CPU, in addition to both PyNative and Graph modes.

Related interfaces are as follows:

1. `mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)`: Set the data parallel mode.
2. `mindspore.nn.DistributedGradReducer()`: Perform multi-card gradient aggregation.

## Overall Process

![Overall Process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/docs/mindspore/source_zh_cn/design/images/data_parallel.png)

1. Environmental dependencies

    Before starting parallel training, the communication resources are initialized by calling the `mindspore.communication.init` interface and the global communication group `WORLD_COMM_GROUP` is automatically created. The communication group enables communication operators to distribute messages between cards and machines, and the global communication group is the largest one, including all devices in current training. The current mode is set to data parallel mode by calling `mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)`.

2. Data distribution

    The core of data parallel lies in splitting the dataset in sample dimensions and sending it down to different cards. In all dataset loading interfaces provided by the `mindspore.dataset` module, there are `num_shards` and `shard_id` parameters which are used to split the dataset into multiple copies and cycle through the samples in a way that collects `batch` data to their respective cards, and will start from the beginning when there is a shortage of data.

3. Network composition

    The data parallel network is written in a way that does not differ from the single-card network, due to the fact that during forward propagation & backward propagation the models of each card are executed independently from each other, only the same network structure is maintained. The only thing we need to pay special attention to is that in order to ensure the training synchronization between cards, the corresponding network parameter initialization values should be the same. In `DATA_PARALLEL` mode, we can set seed or enable `parameter_broadcast` to achieve the same initialization of weights between multiple cards.

4. Gradient aggregation

    Data parallel should theoretically achieve the same training effect as the single-card machine. In order to ensure the consistency of the computational logic, the gradient aggregation operation between cards is realized by calling the `mindspore.nn.DistributedGradReducer()` interface, which automatically inserts the `AllReduce` operator after the gradient computation is completed. MindSpore sets the `mean` switch, which allows the user to choose whether to perform an average operation on the summed gradient values, or to treat them as hyperparameters.

5. Parameter update

    Because of the introduction of the gradient aggregation operation, the models of each card will enter the parameter update step together with the same gradient values.

## Operation Practice

The following is an illustration of data parallel operation using the Ascend or GPU single-machine 8-card as an example:

### Sample Code Description

> You can download the full sample code here:
>
> <https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/distributed_data_parallel>.

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_data_parallel
       ├── distributed_data_parallel.py
       └── run.sh
    ...
```

Among them, `distributed_data_parallel.py` is the script that defines the network structure and training process. `run.sh` is the execution script.

### Configuring Distributed Environments

The context interface allows you to specify the run mode, run device, run card number. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` for data parallel mode and initialize HCCL or NCCL communication through init. In data parallel mode, you also need to set `gradients_mean` to specify the gradient aggregation method. If `device_target` is not set here, it is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)
```

`gradients_mean=True` is for the fact that during the backward computation, the framework internally aggregates the gradient values of the data parallel parameters scattered across multiple machines to get the global gradient values before passing them into the optimizer for updating. The default value is `False`. Setting it to `True` corresponds to the aggregation method being an `AllReduce.Mean` operation, and `False` corresponds to an `AllReduce.Sum` operation.

### Data Parallel Mode Loads Datasets

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

### Defining Network

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

### Training Network

In this step, we need to define the loss function, the optimizer, and the training process. The difference with single-card model is that the data parallel mode also requires the addition of the `mindspore.nn.DistributedGradReducer()` interface to aggregate the gradients of all cards. The first parameter of the network is the network parameter to be updated:

```python
from mindspore import nn, ops

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(net.trainable_params(), 1e-2)

def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
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

> This can also be trained using [Model.train](https://www.mindspore.cn/docs/en/r2.3/api_python/train/mindspore.train.Model.html#mindspore.train.Model.train).

### Running Single-machine Eight-card Script

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
epoch: 0 step: 0, loss is 2.3084016
epoch: 0 step: 10, loss is 2.3107638
epoch: 0 step: 20, loss is 2.2864391
epoch: 0 step: 30, loss is 2.2938071
...
```

Other startup methods such as dynamic network and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/startup_method.html).
