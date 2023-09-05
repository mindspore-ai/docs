# Multi-dimensional Hybrid Parallel Case Based on Double Recursive Search

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/multiple_mix.md)

## Overview

Multi-dimensional hybrid parallel based on double recursive search means that the user can configure optimization methods such as recomputation, optimizer parallel, pipeline parallel. Based on the user configurations, the operator-level strategy is automatically searched by the double recursive strategy search algorithm, which generates the optimal parallel strategy.

## Operation Practice

The following is a multi-dimensional hybrid parallel case based on double recursive search using Ascend or GPU single-machine 8-card as an example:

### Example Code Description

> Download the complete example code: [multiple_mix](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/multiple_mix).

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

Specify the run mode, run device, run card number, etc. through the context interface. Unlike single-card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` as auto-parallel and the search mode `search_mode` as double recursive strategy search mode `recursive_programming` for auto-slicing of the data parallel and model parallel, and initialize HCCL or NCCL communication with init. `pipeline_stages` is the number of stages in pipeline parallel, and optimizer parallel is enabled by enabling `enable_parallel_optimizer`. `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE, save_graphs=2)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="recursive_programming")
ms.set_auto_parallel_context(pipeline_stages=2, enable_parallel_optimizer=True)
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
# Configure the pipeline_stage number for each layer in pipeline parallel
net.layer1.pipeline_stage = 0
net.relu1.pipeline_stage = 0
net.layer2.pipeline_stage = 1
net.relu2.pipeline_stage = 1
net.layer3.pipeline_stage = 1
# Configure recomputation of relu operators
net.relu1.recompute()
net.relu2.recompute()
```

### Loading the Datasets

The dataset is loaded in the same way as the single-card model, with the following code:

```python
import os
import mindspore.dataset as ds

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
```

### Training the Network

This part is consistent with the pipeline parallel training code. Two additional interfaces need to be called based on the stand-alone training code: `nn.WithLossCell` for wrapping the network and loss function, and `nn.PipelineCell` for wrapping the LossCell and configuring the MicroBatch size. The code is as follows:

```python
import mindspore as ms
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor()
net_with_grads = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb], dataset_sink_mode=True)
```

### Running a Stand-alone Eight-Card Script

Next, the corresponding scripts are invoked by commands, using the `mpirun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run_sapp_mix_train.sh
```

The results are saved in `log_output/1/rank.*/stdout`, and the example is as follows:

```text
epoch: 1 step: 1875, loss is 0.6961808800697327
epoch: 2 step: 1875, loss is 0.2737872302532196
epoch: 3 step: 1875, loss is 0.17508840560913086
epoch: 4 step: 1875, loss is 0.5057268142700195
epoch: 5 step: 1875, loss is 0.30770277976989746
epoch: 6 step: 1875, loss is 0.14041686058044434
epoch: 7 step: 1875, loss is 0.018445372581481934
epoch: 8 step: 1875, loss is 0.00423431396484375
epoch: 9 step: 1875, loss is 0.001628875732421875
epoch: 10 step: 1875, loss is 0.03857862949371338
```
