# Optimizer Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/parallel/optimizer_parallel.md)

## Overview

When performing data parallel training, the parameter update part of the model is computed redundantly across cards. Optimizer parallelism can effectively reduce memory consumption and improve network performance on large-scale networks (e.g., Bert, GPT) by spreading the computation of the optimizer to the cards of the data parallel dimension.

The following is an illustration of optimizer parallel operation using an Ascend single-machine 8-card example:

## Sample Code Description

> Download the full sample code: [distributed_optimizer_parallel](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/distributed_optimizer_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_optimizer_parallel
       ├── distributed_optimizer_parallel.py
       └── run.sh
    ...
```

Among them, `distributed_optimizer_parallel.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

## Configuring the Distributed Environment

Specify the run mode, run device, run card number through the context interface. Unlike single-card scripts, parallel scripts also need to initialize HCCL or NCCL communication through init.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
ms.set_seed(1)
```

## Loading the Dataset

In the optimizer parallel scenario, the dataset is loaded in the same way as single-card is loaded, with the following code:

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

## Defining the Network and Optimizer

The optimizer parallel network structure is essentially the same as the single card network structure, with the difference being the addition of a configuration for communication operator fusion and the need for delayed initialization of the network and optimizer:

```python
from mindspore import nn
from mindspore.nn.utils import no_init_parameters

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.layer2 = nn.Dense(512, 512)
        self.layer3 = nn.Dense(512, 10)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        logits = self.layer3(x)
        return logits

with no_init_parameters:
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
net.layer1.set_comm_fusion(0)
net.layer2.set_comm_fusion(1)
net.layer3.set_comm_fusion(2)
```

> Here communication fusion is configured for different layers in order to reduce the communication cost. Details can be found in [Communication Operator Fusion](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/comm_fusion.html).

## Training Network Definition

In this step, we need to define the loss function and the training process, which is the same as that of the single-card:

```python
import mindspore as ms
from mindspore import nn

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

```

## Parallel Configuration

We need to further set up the parallelism-related configuration by specifying the parallelism mode `semi-auto` as semi-automatic parallelism, in addition to turning on optimizer parallelism and configuring `hsdp`.

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_step, parallel_mode="semi_auto")
parallel_net.hsdp()

```

## Training Loop

This step performs the training loop, the outer loop is the number of epochs to train and the inner loop traverses the dataset and calls parallel_net to train and get the loss values.

```python
for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = parallel_net(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

## Running the Single-machine Eight-card Script

Next, the corresponding scripts are invoked by commands, using the `msrun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory, where part of the file directory structure is as follows:

```text
└─ log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

The results are saved in `log_output/worker_*.py`, and example is as follows:

```text
epoch: 0, step: 0, loss is 2.3024087
epoch: 0, step: 10, loss is 2.2921634
epoch: 0, step: 20, loss is 2.278274
epoch: 0, step: 30, loss is 2.2537143
epoch: 0, step: 40, loss is 2.1638
epoch: 0, step: 50, loss is 1.984318
epoch: 0, step: 60, loss is 1.6061916
epoch: 0, step: 70, loss is 1.20966
epoch: 0, step: 80, loss is 0.98156196
epoch: 0, step: 90, loss is 0.77229893
epoch: 0, step: 100, loss is 0.6854114
...
```

Other startup methods such as `mpirun` and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/startup_method.html).
