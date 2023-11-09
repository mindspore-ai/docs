# Fault Recovery Based on Redundant Information

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/fault_recover.md)

## Overview

It is very common to encounter failures when performing distributed training, similar to single-card training, which can be continued by loading the saved weight information during training. Distinct from pure data parallel training, when model parallelism is applied, the weights are sliced and the weight information saved between cards may not be consistent.

To solve this problem, one option is to aggregate the weights through the [AllGather](https://www.mindspore.cn/docs/en/r2.3/api_python/samples/ops/communicate_ops.html#allgather) before saving the weight checkpoint file, where each card stores a complete information about the weights. This function is the integrated_save in the `mindspore.train.CheckpointConfig(integrated_save=True)` interface.

However, for large models, the overhead of using aggregated preservation is too large for all kinds of resources, so this document presents a recovery scheme where each card only saves its own weight information. For large models, both data parallelism and model parallelism are often applied, and the devices divided by the dimensions of data parallelism, which hold exactly the same weight information, provide a redundant backup for large models. This document will also point out how to go about obtaining this redundant information.

For the relationship between the parallel strategy and the slicing division of the weights, the following mapping can be performed. For the concept of data parallelism, model parallelism, please refer to [data parallelism](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/data_parallel.html). For more information about optimizer parallelism, please refer to [Optimizer Parallelism](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/optimizer_parallel.html).

- Data parallelism + keep optimizer parallelism off: The ranks in the parallel communication domain hold the same weight slice.
- Model parallism: The ranks in the parallel communication domain hold different weight slices.
- Data parallelism + keep optimizer parallelism on + the number of shards in optimizer parallelism is equal to the number of all data parallel dimensions: rank in the parallelism communication domain holds slices with different weights.
- Data parallelism + keep optimizer parallelism on + the number of shards in optimizer parallelism is smaller than the number of all data parallel dimensions: Within the parallel communication domain, the rank within the communication domain sliced by the optimizer holds different weight slices, and the communication domain sliced by each optimizer holds the same weight slice between them.

Also, it should be noted that this document introduces the distributed faults recovery scheme, which needs to be used in [sink mode](https://www.mindspore.cn/tutorials/experts/en/r2.3/optimize/execution_opt.html).

Related environment variables:

`GROUP_INFO_FILE=./group_info.pb`: Save weights information of the slices. The file is parsed out to get a list whose values are rank_id, representing that the weights in those rank_id are the same.

## Operation Practice

The following is an operation illustration of fault recovery under distributed training using single-machine 8-card as an example:

### Example Code Description

> Download the complete example code: [fault_recover](https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/fault_recover)

The directory structure is as follows:

```text
└─ sample_code
    ├─ fault_recover
        ├── train.py
        ├── run.sh
        └── recover.sh
```

`train.py` is the script that defines the network structure and the training process. `run.sh` is the execution script and `recover.sh` is the recovery script after node failure.

### Configuring a Distributed Environment

Specify the run mode, run device, run card number via the context interface. Unlike single card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` and initialize HCCL or NCCL communication via init. The `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init, get_rank

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
os.environ['GROUP_INFO_FILE'] = "./checkpoints/rank_{}/group_info.pb".format(get_rank())
ms.set_seed(1)
```

> This configures the environment variable GROUP_INFO_FILE to store redundant information about weights.

### Loading the Dataset

In the current sample, the dataset is loaded in the same way as a single card is loaded, with the following code:

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

### Defining the Network

Here some sharding strategies are configured for the operator and the network structure after configuring the strategies is:

```python
import mindspore as ms
from mindspore import nn, ops

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul()
        self.relu1 = ops.ReLU()
        self.matmul2 = ops.MatMul()
        self.relu2 = ops.ReLU()
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
net.matmul1.shard(((2, 4), (4, 1)))
net.relu1.shard(((4, 1),))
```

### Training the Network

In this step, we need to define the loss function, the optimizer, and the training process:

```python
import mindspore as ms
from mindspore import nn, train
from mindspore.communication import get_rank

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor()
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=4, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint", directory="./checkpoints/rank_{}".format(get_rank()), config=ckpt_config)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(2, data_set, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)
```

> During training, sink mode is configured by specifying dataset_sink_mode as True, and `integrated_save` needs to be configured as `False` in CheckpointConfig.

### Fault Recovery

Distributed fault recovery requires prior access to the information about slicing, thus, `model.infer_train_layout` needs to be called first to get the information about the sharding strategy, then the training is executed.

```python
import mindspore as ms
from mindspore.communication import get_rank

# model create
# checkpoint load
if bool(args_opt.is_recover):
    param_dict = ms.load_checkpoint("./checkpoints/rank_{}/checkpoint-2_1875.ckpt".format(get_rank()))
    model.infer_train_layout(data_set)
    ms.load_param_into_net(net, param_dict)
model.train(2, data_set, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)
```

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed script as an example, and run the 8-card parallel training script by the following command:

```bash
bash run.sh
```

After the training is complete, you can see the following file:

```text
├─ group_info.pb
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ checkpoints
|   ├─ rank_0
|   |   ├─ checkpoint-1_1875.ckpt
|   |   ├─ checkpoint-2_1875.ckpt
|   |   ├─ checkpoint-graph.meta
|   |   └─ group_info.pb
|   ├─ rank_1
|   |   ├─ checkpoint-1_1875.ckpt
|   |   ...
|   ...
...
```

In `log_output/1/rank.*/stdout`, you can see the current trained loss value, similar to the following:

```text
epoch: 1 step: 1875, loss is 0.71328689217567444
epoch: 2 step: 1875, loss is 0.32782320742607117
```

Read group_info.pb to get redundant information about the weights. The file will be parsed out to get a list with the value of rank_id, which means that the weight slices corresponding to the rank_id in these lists are all the same and can be replaced with each other.
As in the following example, after the group_info.pb of 0-card is parsed, it is found that the weight slices of 0-card and 4-card are exactly the same, and when the checkpoint of 0-card is lost, 4-card checkpoint can be copied directly as the checkpoint of 0-card for recovery.

```python
import mindspore as ms
rank_list = ms.restore_group_info_list("./checkpoints/rank_0/group_info.pb")
print(rank_list) // [0, 4]
```

After that, the fault recovery training script is executed.

```bash
bash recover.sh
```

At the end of the recovery training, check the loss as follows, indicating that the loading was successful.

```text
epoch: 1 step: 1875, loss is 0.598689079284668
epoch: 2 step: 1875, loss is 0.266701698332226
```
