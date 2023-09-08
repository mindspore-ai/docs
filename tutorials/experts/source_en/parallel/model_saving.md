# Model Saving

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/model_saving.md)

## Overview

In this tutorial, we mainly explain how to utilize MindSpore for distributed network training and saving model files. In a distributed training scenario, model saving can be divided into merged and non-merged saving: merged saving requires additional communication and memory overhead, and each card saves the same model file, while non-merged saving saves only the weights of the current card slicing, which effectively reduces the communication and memory overhead required for aggregation.

Related interfaces:

1. `mindspore.set_auto_parallel_context(strategy_ckpt_config=strategy_ckpt_dict)`: The configuration used to set the parallel strategy file. `strategy_ckpt_dict` is used to set the configuration of the parallel strategy file and is of dictionary type. strategy_ckpt_dict = {"load_file": ". /stra0.ckpt", "save_file": ". /stra1.ckpt", "only_trainable_params": False}, where:
    - `load_file(str)`: The path to load the parallel sharding strategy. Default: "".
    - `save_file(str)`: Save the paths for the parallel sharding strategy. This parameter must be set for distributed training scenarios. Default: "".
    - `only_trainable_params(bool)`: Save/load strategy information for trainable parameters only. Default: `True`.

2. `mindspore.train.ModelCheckpoint(prefix='CKP', directory=None, config=None)`: This interface is called to save network parameters during training. Specific strategy can be configured in this interface by configuring `config`, and see interface `mindspore.train.CheckpointConfig`. It should be noted that in parallel mode you need to specify a different checkpoint save path for each script running on each card, to prevent conflicts when reading and writing files.

3. `mindspore.train.CheckpointConfig(save_checkpoint_steps=10, integrated_save=True)`: Configure the strategy for saving Checkpoints. `save_checkpoint_steps` indicates interval steps to save the checkpoint. `integrated_save` indicates whether to perform merged saving on the split model files in the automatic parallel scenario. The merged saving function is only supported in auto-parallel scenarios, not in manual parallel scenarios.

## Operation Practice

The following is an illustration of saving the model files in the distributed training, using a single-machine 8-card as an example.

### Example Code Description

> Download the complete example code: [model_saving_loading](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/model_saving_loading).

The directory structure is as follows:

```text
└─ sample_code
    ├─ model_saving_loading
       ├── train_saving.py
       ├── run_saving.sh
       ...
    ...
```

`train_saving.py` is the script that defines the network structure and inference. `run_saving.sh` is the execution script.

### Configuring a Distributed Environment

Specify the run mode, run device, run card number via the context interface. Unlike single card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` as semi-parallel mode. Configure and save the distributed strategy file via `strategy_ckpt_config` and initialize HCCL or NCCL communication via init. The `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": "./src_strategy.ckpt"})
init()
ms.set_seed(1)
```

### Defining the Network

The network definition adds the sharding strategy of `ops.MatMul()` opertor:

```python
from mindspore import nn, ops
from mindspore.common.initializer import initializer

class Dense(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        self.matmul = ops.MatMul()
        self.add = ops.Add()

    def construct(self, x):
        x = self.matmul(x, self.weight)
        x = self.add(x, self.bias)
        return x

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.layer1 = Dense(28*28, 512)
        self.relu1 = ops.ReLU()
        self.layer2 = Dense(512, 512)
        self.relu2 = ops.ReLU()
        self.layer3 = Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.matmul.shard(((2, 1), (1, 2)))
net.layer3.matmul.shard(((2, 2), (2, 1)))
```

### Loading the Dataset

The dataset is loaded in the same way as the single card model, with the following code:

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

For the parameter framework sliced in the network automatically is aggregated and saved to the model file by default, but considering that in the ultra-large model scenario, a single complete model file is too large to bring about problems such as slow transmission and hard to load, so the user can choose non-merged saving through the `integrated_save` parameter in the `CheckpointConfig`, i.e., each card saves the parameter slices from each card itself.

```python
import mindspore as ms
from mindspore.communication import get_rank
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor(20)
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=1, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./src_checkpoints/rank_{}".format(get_rank()),
                                   config=ckpt_config)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb, ckpoint_cb])
```

### Running Stand-alone 8-card Script

Next, the corresponding script is called by the command. Take the `mpirun` startup method, the 8-card distributed training script as an example, and perform the distributed training:

```bash
bash run_saving.sh
```

After training, the log files are saved to the `log_output` directory and the checkpoint files are saved in the `src_checkpoints` folder with the following file directory structure:

```text
├─ src_strategy.ckpt
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ src_checkpoints
|   ├─ rank_0
|   |   ├─ checkpoint-10_1875.ckpt
|   |   └─ checkpoint-graph.meta
|   ├─ rank_1
|   |   ├─ checkpoint-10_1875.ckpt
|   |   ...
|   ...
...
```

The part of results on the Loss section are saved in `log_output/1/rank.*/stdout`, and the example is as below:

```text
epoch: 1 step: 20, loss is 2.2978780269622803
epoch: 1 step: 40, loss is 2.2965049743652344
epoch: 1 step: 60, loss is 2.2927846908569336
epoch: 1 step: 80, loss is 2.294496774673462
epoch: 1 step: 100, loss is 2.2829630374908447
epoch: 1 step: 120, loss is 2.2793829441070557
epoch: 1 step: 140, loss is 2.2842094898223877
epoch: 1 step: 160, loss is 2.269033670425415
epoch: 1 step: 180, loss is 2.267289400100708
epoch: 1 step: 200, loss is 2.257275342941284
...
```

Merged saving can be turned on by configuring `integrated_save` in `mindspore.train.CheckpointConfig` to `True`, and the code to be replaced is as follows:

```python
...
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=3, integrated_save=True)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./src_checkpoints_integrated/rank_{}".format(get_rank()),
                                   config=ckpt_config)
...
```
