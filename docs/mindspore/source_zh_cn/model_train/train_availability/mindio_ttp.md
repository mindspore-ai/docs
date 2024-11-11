# 临终Checkpoint保存

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/train_availability/mindio_ttp.md)

## 概述

MindSpore临终CKPT功能基于[MindIO TTP](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/mindio/mindiottp/mindiottp001.html)，主要针对大模型训练过程中故障恢复加速，临终Checkpoint特性通过在训练过程中发生故障后，校验中间状态数据的完整性和一致性，生成一次临时CheckPoint数据，恢复训练时能够通过该CheckPoint数据恢复，减少故障造成的训练迭代损失。

下面以一个4卡数据并行网络训练为例，介绍如何配置临终CKPT功能。 配置完成后，在训练中如遇到功能故障（主要包括：训练进程异常，训练进程异常退出），MindSpore和MindIO会停止所有卡的训练，检查最新的训练状态，并基于训练卡间的副本关系，确认是否存在可用的副本卡（好卡），如果存在则将对好卡进行临终CKPT的保存， 否则按异常退出处理。如果发生故障后，能保存第n个step的CKPT文件， 则下一次训练可从第n+1个step开始。

### 使用约束

1. 仅支持Ascend后端的静态图模式。
2. 仅支持sink_size=1， 用于保证step的正确性。
3. 仅支持父类类型为MindSpore Optimizer的优化器。
4. 仅支持数据并行度大于1的网络，以确保模型参数存在副本关系。
5. 如果网络开启优化器并行，必须使能optimizer_weight_shard_size:2，并确保其生效，以使优化器参数存在副本关系，详细可以参考[优化器并行](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/optimizer_parallel.html#%E9%AB%98%E7%BA%A7%E6%8E%A5%E5%8F%A3) 。

## 样例代码说明

> 您可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/mindio_ttp>。

目录结构如下：

```text
└─ sample_code
    ├─ mindio_ttp
       ├── mindio_ttp_case.py
       ├── msrun-resume.sh
       └── msrun.sh
    ...
```

其中，`mindio_ttp_case.py`是定义网络结构和训练过程的脚本。`msrun.sh`是训练脚本。`msrun-resume.sh`是续训脚本。

## 环境准备

临终CKPT功能开启需要先安装`MindIO TTP`, 详情参见[MindIO TTP](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/mindio/mindiottp/mindiottp001.html)。

## 准备数据

下载MNIST数据集，并解压数据集到项目目录。

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

## 模型定义

如下代码定义一个包含5层的网络结构。其中设置并行模式为数据并行，让每张卡都互为备份关系，以便发生异常时，临终Checkpoint功能找到有效的副本进行保存。

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

## 数据集定义

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

## 优化器定义与封装

开启临终CKPT功能需要设置TFT优化器, 设置后可在梯度计算完成后，优化器更新前向MindIO TFT上报状态。TFT优化器用`OptTFTWrapper`来配置, 详情参见[OptTFTWrapper](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.OptTFTWrapper.html)。

```python
optimizer = nn.SGD(net.trainable_params(), 1e-2)
#配置TFT优化器
optimizer_wrapper = nn.OptTFTWrapper(optimizer)
```

## 创建loss函数并配置model对象

```python
loss_fn = nn.CrossEntropyLoss()
net.set_train()
model = ms.Model(net,  optimizer=optimizer_wrapper)
```

## Callback配置

开启临终CKPT功能需要设置 `TFTRegister` Callback对象，并传入参数来配置，详情参见[TFTRegister](https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.TFTRegister.html)。

```python
time_monitor = train.TimeMonitor(data_size=1)
loss_cb = train.LossMonitor(1)

# 设置TFT callback对象
tft_cb = train.TFTRegister(0, "127.0.0.1", 30051, "./ttp_checkpoints/")
```

## 续训配置

续训可从临终Chckpoint恢复，由于临终Checkpoint对于多个副本只会保存一份Checkpoint文件，因此需要查看生成的Checkpoint文件，并配置相应的Checkpoint文件进行续训。

```python
init_epoch = 0

if bool(args_opt.is_recover):
    cur_epoch = 2 # 设置成异常保存的epoch值
    cur_step = 1215 # 设置成异常保存的step值
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

## 启动训练

```python
model.train(5, dataset, callbacks=[time_monitor, loss_cb, tft_cb])
```

## 配置环境变量并启动训练

开启临终Checkpoint功能，需要设置环境变量 `MS_ENABLE_TFT='{TTP:1}'`。此外还需要设置环境变量 `MINDIO_FOR_MINDSPORE=1`， 使能 `MindIO` 适配 MindSpore。

使用 `msrun` 命令启动训练。

```bash
export MS_ENABLE_TFT='{TTP:1}'
export MINDIO_FOR_MINDSPORE=1
export DATA_PATH=${EXEC_PATH}/MNIST_DATA/train/

msrun --worker_num=4 --local_worker_num=4 --master_port=10970 --join=False --log_dir=msrun_log --cluster_time_out=300  mindio_ttp_case.py
```

## 异常注入

常见的异常注入为查看训练的进程，并直接杀掉相应的进程来检验是否有临终Checkpoint文件生成。
注意： 由于MindIo的Controller控制器默认在0卡启动，因此杀死rank0的进程并不会生成Checkpoint文件。

```bash
npu-smi info # 查看训练进程
kill -9 pid  # 杀死对应的训练进程
```

## 配置环境变量并恢复训练

```bash
export MS_ENABLE_TFT='{TTP:1}'
export MINDIO_FOR_MINDSPORE=1
export DATA_PATH=${EXEC_PATH}/MNIST_DATA/train/

msrun --worker_num=4 --local_worker_num=4 --master_port=10970 --join=False --log_dir=msrun_log --cluster_time_out=300  mindio_ttp_case.py --is_recover=1

```

## 临终Checkpoint文件生成说明

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
