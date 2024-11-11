# Power-off Checkpoint Preservation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/train_availability/mindio_ttp.md)

## Overview

MindSpore power-off CKPT is based on [MindIO TTP](https://www.hiascend.com/document/detail/en/mindx-dl/500/mindio/mindioug/mindio_001.html), which is mainly aimed at accelerating fault recovery during large model training, the power-off Checkpoint feature generates temporary CheckPoint data once by verifying the integrity and consistency of the intermediate state data after a fault occurs during the training process, which can be recovered by the CheckPoint data when resuming the training to reduce the loss of training iterations caused by faults.

The following is an example of how to configure the power-off CKPT function for a 4-card data parallel network training. After the configuration is completed, if there is a functional failure during training (mainly including: abnormal training process, abnormal exit of training process), MindSpore and MindIO will stop the training of all cards, check the latest training status, and based on the replica relationship between the training cards, confirm whether there is an available replica card (good card), if there is, then it will save the power-off CKPT for the good card, otherwise, it will be treated as abnormal exit treatment. If the CKPT file of the nth step can be saved after the failure, the next training can start from the n+1th step.

### Use Constraints

1. Only static graph mode is supported for the Ascend backend.
2. Only sink_size=1 is supported for step correctness.
3. Only optimizers whose parent class type is MindSpore Optimizer are supported.
4. Only networks with data parallelism greater than 1 are supported to ensure that replica relationships exist for model parameters.
5. If the network turns on optimizer parallelism, you must enable optimizer_weight_shard_size:2 and make sure it is in effect so that there is a replica relationship for the optimizer parameters, see [Optimizer Parallelism](https://www.mindspore.cn/docs/en/master/model_train/parallel/optimizer_parallel.html#advanced-interfaces) for details.

## Sample Code Description

> You can download the full sample code here:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/mindio_ttp>.

The directory structure is as follows:

```text
└─ sample_code
    ├─ mindio_ttp
       ├── mindio_ttp_case.py
       ├── msrun-resume.sh
       └── msrun.sh
    ...
```

Among them, `mindio_ttp_case.py` is the script that defines the network structure and the training process. `msrun.sh` is the training script. `msrun-resume.sh` is the renewal script.

## Environment Preparation

To enable the power-off CKPT function, you need to install `MindIO TTP`, see [MindIO TTP](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/mindio/mindiottp/mindiottp001.html) for details.

## Preparing Data

Download the MNIST dataset and extract the dataset to the project directory.

```bash
EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/
```

## Model Defining

The following code defines a network structure containing 5 layers. The parallel mode is set to data parallelism so that each card is in a backup relationship with each other so that in case of an exception, the power-off Checkpoint function finds a valid copy to save.

```python

import os
import math
import argparse
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, Parameter, train
from mindspore.communication import init, get_rank
from mindspore.common.initializer import initializer, HeUniform

parser = argparse.ArgumentParser(description="Mindio TTP test arguments")
parser.add_argument("--is_recover",
                    type=int,
                    default=0,
                    choices=[1, 0],
                    help="Only used for resume from Mindio TTP checkpoint, default false.")
args_opt = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE, jit_level='O1', device_target="Ascend")

ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
init()
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": "./src_pipeline_strategy/src_strategy_{}.ckpt".format(get_rank())})

class MatMulCell(nn.Cell):
    """
    MatMulCell definition.
    """
    def __init__(self, param=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [28 * 28, 512]
        weight_init = HeUniform(math.sqrt(5))
        self.param = Parameter(initializer(weight_init, shape), name="param")
        if param is not None:
            self.param = param
        self.matmul = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        return out


class Network(nn.Cell):
    """
    Network definition.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = MatMulCell()
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 5120)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(5120, 5120)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Dense(5120, 512)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.relu4(x)
        logits = self.layer5(x)
        return logits

net = Network()

```

## Dataset Defining

```python
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

dataset = create_dataset(32)
```

## Optimizer Definition and Encapsulation

The TFT optimizer needs to be set up to enable the power-off CKPT function. After setting up the TFT optimizer, the status can be reported to MindIO TFT after the gradient calculation is completed and before the optimizer is updated. The TFT optimizer is configured with `OptTFTWrapper`, see [OptTFTWrapper](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.OptTFTWrapper.html).

```python
optimizer = nn.SGD(net.trainable_params(), 1e-2)
# Configure TFT optimizer
optimizer_wrapper = nn.OptTFTWrapper(optimizer)
```

## Creating the Loss Function and Configuring the Model Object

```python
loss_fn = nn.CrossEntropyLoss()
net.set_train()
model = ms.Model(net,  optimizer=optimizer_wrapper)
```

## Callback Configuration

To enable the power-off CKPT feature, you need to set the `TFTRegister` Callback object and pass in the parameters to configure it, see [TFTRegister](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.TFTRegister.html) for details.

```python
time_monitor = train.TimeMonitor(data_size=1)
loss_cb = train.LossMonitor(1)

# Set the TFT callback object
tft_cb = train.TFTRegister(0, "127.0.0.1", 30051, "./ttp_checkpoints/")
```

## Renewal Configuration

Renewal training can be resumed from the power-off Chckpoint, and since the power-off Checkpoint will only save one Checkpoint file for multiple copies, you need to look at the generated Checkpoint file and configure the appropriate Checkpoint file for renewal training.

```python
init_epoch = 0

if bool(args_opt.is_recover):
    cur_epoch = 2 # Set to the epoch value of the exception save
    cur_step = 1215 # Set to the step value of the exception save
    ckpt_step = (cur_epoch - 1) * dataset.get_dataset_size() + cur_step
    if context.get_auto_parallel_context("parallel_mode") == "data_parallel":
        cur_rank = 0
        new_ckpt_file = f"./ttp_checkpoints/tft_saved_checkpoints-step_{ckpt_step}/rank_{cur_rank}/ttp_rank_{cur_rank}-{cur_epoch}_{cur_step}.ckpt"
    else:
        cur_rank = get_rank()
        ckpt_file = f"./ttp_checkpoints/tft_saved_checkpoints-step_{ckpt_step}/rank_{cur_rank}/ttp_rank_{cur_rank}-{cur_epoch}_{cur_step}.ckpt"
        strategy_file = f"./src_pipeline_strategy/src_strategy_{cur_rank}.ckpt"
        new_ckpt_file = get_ckpt_path_with_strategy(ckpt_file, strategy_file)
    param_dict = ms.load_checkpoint(new_ckpt_file)
    ms.load_param_into_net(net, param_dict)
    dataset.set_init_step(int(param_dict["step_num"]))
    init_epoch = int(param_dict["epoch_num"]) - 1
```

## Initiating Training

```python
model.train(5, dataset, callbacks=[time_monitor, loss_cb, tft_cb])
```

## Configuring Environment Variables and Initiating Training

To enable power-off Checkpoint, set the environment variable `MS_ENABLE_TFT='{TTP:1}'`. You also need to set the environment variable `MINDIO_FOR_MINDSPORE=1` to enable `MindIO` to adapt to MindSpore.

Use the `msrun` command to initiate training.

```bash
export MS_ENABLE_TFT='{TTP:1}'
export MINDIO_FOR_MINDSPORE=1
export DATA_PATH=${EXEC_PATH}/MNIST_DATA/train/

msrun --worker_num=4 --local_worker_num=4 --master_port=10970 --join=False --log_dir=msrun_log --cluster_time_out=300  mindio_ttp_case.py
```

## Exception Injection

A common exception injection is to look at the training process and kill the corresponding process directly to check if a power-off Checkpoint file has been generated.
Note: Since MindIo's controller starts on card 0 by default, killing the rank0 process does not generate a Checkpoint file.

```bash
npu-smi info # Check training process
kill -9 pid  # Kill corresponding training process
```

## Configuring Environment Variables and Re-training

```bash
export MS_ENABLE_TFT='{TTP:1}'
export MINDIO_FOR_MINDSPORE=1
export DATA_PATH=${EXEC_PATH}/MNIST_DATA/train/

msrun --worker_num=4 --local_worker_num=4 --master_port=10970 --join=False --log_dir=msrun_log --cluster_time_out=300  mindio_ttp_case.py --is_recover=1

```

## Power-off Checkpoint Document Generation Instructions

```text
└─ sample_code
    ├─ mindio_ttp
       ├── ttp_checkpoints
           ├── tft_saved_checkpoints-step_{global_step}
               ├── rank_0
                   └── ttp_rank_0-{cur_epoch}_{cur_step}.ckpt
               ├── rank_1
                   └── ttp_rank_1-{cur_epoch}_{cur_step}.ckpt
    ...
```