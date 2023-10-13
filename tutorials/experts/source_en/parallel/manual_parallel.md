# Manually Parallelism

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_en/parallel/manual_parallel.md)

## Overview

In addition to the automatic and semi-automatic parallelism provided by MindSpore, users can manually slice the model to parallelize it on multiple nodes by encoding the parallel process based on communication primitives. In this manual parallel mode, the user needs to perceive graph slicing, operator slicing, and cluster topology to achieve optimal performance.

## Basic Principle

MindSpore aggragation communication operators include `AllReduce`, `AllGather`, `ReduceScatter`, `Broadcast`, `NeighborExchange`, `NeighborExchangeV2`, and `AlltoAll`, which are the basic building blocks of aggragation communication in distributed training. The so-called aggragation communication refers to the data interaction between different model slices through aggragation communication operators after model slicing. Users can manually call these operators for data transfer to realize distributed training.

For a detailed description of the aggragation communication operator, see [Distributed Set Communication Primitive](https://www.mindspore.cn/docs/en/r2.2/api_python/samples/ops/communicate_ops.html).

## Operation Practice

The following is an illustration of manual data parallel operation using an Ascend or GPU stand-alone 8-card as an example:

### Example Code Description

> Download the complete example code: [manual_parallel](https://gitee.com/mindspore/docs/tree/r2.2/docs/sample_code/manual_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ manual_parallel
       ├── train.py
       └── run.sh
    ...
```

`train.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

### Configuring a Distributed Environment

Initialize HCCL or NCCL communication with init and set the random seed. No parallel mode is specified here as it is manually parallelized. `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package. `get_group_size()` interface gets the number of devices in the current communication group, which is by default a global communication group containing all devices.

```python
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size

ms.set_context(mode=ms.GRAPH_MODE)
init()
cur_rank = get_rank()
batch_size = 32
device_num = get_group_size()
shard_size = batch_size // device_num
```

### Network Definition

The slices of the input data is added to the single-card network:

```python
from mindspore import nn
from mindspore.communication import get_rank, get_group_size

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = x[cur_rank*shard_size:cur_rank*shard_size + shard_size]
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
```

### Loading the Dataset

Datasets are loaded in a manner consistent with a single-card network:

```python
import os
import mindspore.dataset as ds

def create_dataset():
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

data_set = create_dataset()
```

### Loss Function Definition

In the loss function, it is necessary to add a slice of the labels and the communication primitive operator `ops.AllReduce` to aggregate the losses of the cards:

```python
from mindspore import nn, ops
from mindspore.communication import get_rank, get_group_size

class ReduceLoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.all_reduce = ops.AllReduce()

    def construct(self, data, label):
        label = label[cur_rank*shard_size:cur_rank*shard_size + shard_size]
        loss_value = self.loss(data, label)
        loss_value = self.all_reduce(loss_value) / device_num
        return loss_value

loss_fn = ReduceLoss()
```

### Training Process Definition

The optimizer, training process is consistent with a single-card network:

```python
import mindspore as ms
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_cb = train.LossMonitor(20)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb])
```

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed training script as an example, and perform the distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory, and by setting context: `save_graphs=2` in `train.py`, you can print out the IR graphs of the compilation process, where some of the file directories are structured as follows:

```text
└─ log_output
    └─ 1
        ├─ rank.0
        |   └─ stdout
        ├─ rank.1
        |   └─ stdout
        ...
```

The results on the Loss section are saved in `log_output/1/rank.*/stdout`, and the example is as below:

```text
epoch: 1 step: 20, loss is 2.241283893585205
epoch: 1 step: 40, loss is 2.1842331886291504
epoch: 1 step: 60, loss is 2.0627782344818115
epoch: 1 step: 80, loss is 1.9561686515808105
epoch: 1 step: 100, loss is 1.8991656303405762
epoch: 1 step: 120, loss is 1.6239635944366455
epoch: 1 step: 140, loss is 1.465965747833252
epoch: 1 step: 160, loss is 1.3662006855010986
epoch: 1 step: 180, loss is 1.1562917232513428
epoch: 1 step: 200, loss is 1.116426944732666
...
```
