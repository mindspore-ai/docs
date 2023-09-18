# Gradient Accumulation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/distributed_gradient_accumulation.md)

## Overview

Gradient accumulation is an optimization technique that enables the use of a larger Batch Size to train a network when memory is limited. Typically, training large neural networks requires a large amount of memory because calculating the gradient on each Batch and updating the model parameters requires saving the gradient values. Larger Batch Size requires more memory, which may lead to out of memory problems. Gradient accumulation works by summing the gradient values of multiple MicroBatches, thus allowing the model to be trained with a larger Batch Size without increasing memory requirements.

Related interfaces:

`mindspore.nn.wrap.cell_wrapper.GradAccumulationCell(network, micro_size)`: Wrap the network with a finer-grained MicroBatch. `micro_size` is the size of the MicroBatch.

## Basic Principle

The core idea of gradient accumulation is to add the gradients of multiple MicroBatches and then use the accumulated gradients to update the model parameters. Here are the steps of gradient accumulation:

1. Select MicroBatch Size: The data of MicroBatch Size is the basic batch for each forward and backward propagation, and also according to the Batch Size divided by Micro Batch Size to get the number of accumulation steps, you can determine after how many MicroBatches a parameter update is performed.

2. Forward and backward propagation: for each MicroBatch, perform the standard forward and backward propagation operations. Calculate the gradient of the MicroBatch.

3. Gradient Accumulation: add the gradient values of each MicroBatch until the number of accumulation steps is reached.

4. Gradient update: After the accumulation number of steps is reached, the accumulation gradient is used to update the model parameters via the optimizer.

5. Gradient Clear: After the gradient is updated, the gradient value is cleared to zero for the next accumulation cycle.

## Operation Practice

The following is an illustration of the gradient accumulation operation using Ascend or GPU stand-alone 8-card as an example:

### Example Code Description

> Download the complete example code: [distributed_gradient_accumulation](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_gradient_accumulation).

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_gradient_accumulation
       ├── train.py
       └── run.sh
    ...
```

`train.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

### Configuring a Distributed Environment

Specify the run mode, run device, run card number via the context interface. The parallel mode is semi-parallel mode. This example uses optimizer parallel and initializes HCCL or NCCL communication via init.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(enable_parallel_optimizer=True)
init()
```

### Dataset Loading and Network Definition

Here the dataset loading and network definition is consistent with the single-card model with the following code:

```python
import os
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

### Training the Network

In this step, we need to define the loss function, the optimizer, and the training process, and in this section two interfaces need to be called to configure the gradient accumulation:

- First the LossCell needs to be defined. In this case the `nn.WithLossCell` interface is called to wrap the network and loss functions.
- It is then necessary to wrap a layer of `GradAccumulationCell` around the LossCell and specify a MicroBatch size of 4. Refer to the relevant interfaces in the overview of this chapter for more details.

```python
import mindspore as ms
from mindspore import nn, train
from mindspore.nn.wrap.cell_wrapper import GradAccumulationCell

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor(100)
net = GradAccumulationCell(nn.WithLossCell(net, loss_fn), 4)
model = ms.Model(net, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb])
```

> Gradient accumulation training is better suited to the `model.train` approach, due to the complexity of the TrainOneStep logic under gradient accumulation, whereas `model.train` internally wraps the TrainOneStepCell for gradient accumulation, which is much easier to use.

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed training script as an example, and perform the distributed training:

```bash
bash run.sh
```

After training, the part of results about the Loss are saved in `log_output/1/rank.*/stdout`. The example is as follows:

```text
epoch: 1 step: 100, loss is 7.89255428314209
epoch: 1 step: 200, loss is 2.3744874000549316
epoch: 1 step: 300, loss is 1.5951943397521973
epoch: 1 step: 400, loss is 1.784447431564331
epoch: 1 step: 500, loss is 1.1426030397415161
epoch: 1 step: 600, loss is 1.2399932146072388
epoch: 1 step: 700, loss is 1.2288587093353271
epoch: 1 step: 800, loss is 1.1972967386245728
...
```
