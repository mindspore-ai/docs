# Multi-dimensional Hybrid Parallel Case Based on Double Recursive Search

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/parallel/multiple_mixed.md)

## Overview

Multi-dimensional hybrid parallel based on double recursive search means that the user can configure optimization methods such as recomputation, optimizer parallel, pipeline parallel. Based on the user configurations, the operator-level strategy is automatically searched by the double recursive strategy search algorithm, which generates the optimal parallel strategy.

## Operation Practice

The following is a multi-dimensional hybrid parallel case based on double recursive search using Ascend or GPU single-machine 8-card as an example:

### Example Code Description

> Download the complete example code: [multiple_mix](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/multiple_mix).

The directory structure is as follows:

```text
└─ sample_code
    ├─ multiple_mix
       ├── sapp_mix_train.py
       └── run_sapp_mix_train.sh
    ...
```

`sapp_mix_train.py` is the script that defines the network structure and the training process. `run_sapp_mix_train.sh` is the execution script.

### Configuring Distributed Environment

Initialize HCCL or NCCL communication with init. `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import os
import mindspore as ms
from mindspore.communication import init

os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="25GB")
init()
ms.set_seed(1)
```

### Network Definition

The network definition adds recomputation, pipeline parallel to the data parallel and model parallel provided by the double recursive strategy search algorithm:

```python
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 1)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
# Configure recomputation of relu operators
net.relu1.recompute()
net.relu2.recompute()
```

### Loading the Datasets

The dataset is loaded in the same way as the single-card model, with the following code:

```python
import os
import mindspore.dataset as ds
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore import nn, train
from mindspore.communication import init

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = "./MNIST_Data/train"
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

### Training the Network

This part is consistent with the pipeline parallel training code. Two additional interfaces need to be called based on the stand-alone training code: `nn.WithLossCell` for wrapping the network and loss function, and `ms.parallel.nn.Pipeline` for wrapping the LossCell and configuring the MicroBatch size. Specify the run mode, run device, run card number, etc. through the context interface. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` as  double recursive strategy search mode `recursive_programming` for auto-slicing of the data parallel and model parallel. `stages` is the number of stages in pipeline parallel, and optimizer parallel is enabled by `hsdp`. The code is as follows:

```python
import mindspore as ms
from mindspore import nn, train

loss_fn = nn.MAELoss()
loss_cb = train.LossMonitor()
# Configure the pipeline_stage number for each layer in pipeline parallelism
net_with_grads = ms.parallel.nn.Pipeline(nn.WithLossCell(net, loss_fn), 4,
                                            stage_config={"_backbone.layer1": 0,
                                                        "_backbone.relu1": 0,
                                                        "_backbone.layer2": 1,
                                                        "_backbone.relu2": 1,
                                                        "_backbone.layer3": 1,})
net_with_grads_new = AutoParallel(net_with_grads, parallel_mode="recursive_programming")
net_with_grads_new.hsdp()
net_with_grads_new.full_batch = True
net_with_grads_new.pipeline(stages=2, scheduler="1f1b")
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb], dataset_sink_mode=True)
```

### Running a Stand-alone Eight-Card Script

Next, the corresponding scripts are invoked by commands, using the `msrun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run_sapp_mix_train.sh
```

The results are saved in `log_output/1/rank.*/stdout`, and the example is as follows:

```text
epoch: 1 step: 1875, loss is 11.6961808800697327
epoch: 2 step: 1875, loss is 10.2737872302532196
epoch: 3 step: 1875, loss is 8.87508840560913086
epoch: 4 step: 1875, loss is 8.1057268142700195
```
