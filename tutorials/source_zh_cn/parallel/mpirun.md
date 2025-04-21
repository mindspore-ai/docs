# mpirun启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/parallel/mpirun.md)

## 概述

OpenMPI（Open Message Passing Interface）是一个开源的、高性能的消息传递编程库，用于并行计算和分布式内存计算。它通过在不同进程之间传递消息来实现并行计算，适用于许多科学计算和机器学习任务。使用OpenMPI进行并行训练，是一种通用的加速训练过程的方法，通过在计算集群或多核机器上充分利用并行计算资源来实现。OpenMPI在分布式训练的场景中，起到在Host侧同步数据以及进程间组网的功能。

与rank table启动不同的是，在Ascend硬件平台上通过OpenMPI的`mpirun`命令运行脚本，用户不需要配置`RANK_TABLE_FILE`环境变量。

> `mpirun`启动支持Ascend和GPU，此外还同时支持PyNative模式和Graph模式。

相关命令：

1. `mpirun`启动命令如下，其中`DEVICE_NUM`是所在机器的GPU数量：

    ```bash
    mpirun -n DEVICE_NUM python net.py
    ```

2. `mpirun`还可以配置以下参数，更多配置可以参考[mpirun文档](https://www.open-mpi.org/doc/current/man1/mpirun.1.php)：

    - `--output-filename log_output`：将所有进程的日志信息保存到`log_output`目录下，不同卡上的日志会按`rank_id`分别保存在`log_output/1/`路径下对应的文件中。
    - `--merge-stderr-to-stdout`：合并stderr到stdout的输出信息中。
    - `--allow-run-as-root`：如果通过root用户执行脚本，则需要加上此参数。
    - `-mca orte_abort_on_non_zero_status 0`：当一个子进程异常退出时，OpenMPI会默认abort所有的子进程，如果不想自动abort子进程，可以加上此参数。
    - `-bind-to none`：OpenMPI会默认给拉起的子进程指定可用的CPU核数，如果不想限制进程使用的核数，可以加上此参数。

> OpenMPI启动时会设置若干名为`OPMI_*`的环境变量，用户应避免在脚本中手动修改这些环境变量。

## 操作实践

`mpirun`启动脚本在Ascend和GPU硬件平台下一致，下面以Ascend为例演示如何编写启动脚本：

> 您可以在这里下载完整的样例代码：[startup_method](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/startup_method)。

目录结构如下：

```text
└─ sample_code
    ├─ startup_method
       ├── net.py
       ├── hostfile
       ├── run_mpirun_1.sh
       ├── run_mpirun_2.sh
    ...
```

其中，`net.py`是定义网络结构和训练过程，`run_mpirun_1.sh`、`run_mpirun_2.sh`是执行脚本，`hostfile`是配置多机多卡的文件。

### 1. 安装OpenMPI

下载OpenMPI-4.1.4源码[openmpi-4.1.4.tar.gz](https://www.open-mpi.org/software/ompi/v4.1/)。参考[OpenMPI官网教程](https://www.open-mpi.org/faq/?category=building#easy-build)安装。

### 2. 准备Python训练脚本

这里以数据并行为例，训练一个MNIST数据集的识别网络。

首先指定运行模式、硬件设备等，与单卡脚本不同，并行脚本还需指定并行模式等配置项，并通过init初始化HCCL或NCCL通信。此处未设置`device_target`，会自动指定为MindSpore包对应的后端硬件设备。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)
```

然后构建如下网络：

```python
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(28*28, 10, weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        logits = self.relu(self.fc(x))
        return logits
net = Network()
```

最后是数据集处理和定义训练过程：

```python
import os
from mindspore import nn
import mindspore as ms
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
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(net.trainable_params(), 1e-2)

def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
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

### 3. 准备启动脚本

#### 单机多卡

首先下载[MNIST](http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip)数据集，并解压到当前文件夹。

然后执行单机多卡启动脚本，以单机8卡为例：

```bash
export DATA_PATH=./MNIST_Data/train/
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout python net.py
```

日志文件会保存到`log_output`目录下，结果保存在`log_output/1/rank.*/stdout`中，结果如下：

```text
epoch: 0, step: 0, loss is 2.3413472
epoch: 0, step: 10, loss is 1.6298866
epoch: 0, step: 20, loss is 1.3729795
epoch: 0, step: 30, loss is 1.2199347
epoch: 0, step: 40, loss is 0.85778403
...
```

#### 多机多卡

在运行多机多卡训练前，首先需要按照如下配置：

1. 保证每个节点上的OpenMPI、NCCL、Python以及MindSpore版本都相同。

2. 配置主机间免密登陆，可参考以下步骤进行配置：
    - 每台主机确定同一个用户作为登陆用户（不推荐root）；
    - 执行`ssh-keygen -t rsa -P ""`生成密钥；
    - 执行`ssh-copy-id DEVICE-IP`设置需要免密登陆的机器IP；
    - 执行`ssh DEVICE-IP`，若不需要输入密码即可登录，则说明以上配置成功；
    - 在所有机器上执行以上命令，确保两两互通。

配置成功后，就可以通过`mpirun`指令启动多机任务，目前有两种方式启动多机训练任务：

- 通过`mpirun -H`方式。启动脚本如下：

    ```bash
    export DATA_PATH=./MNIST_Data/train/
    mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 --output-filename log_output --merge-stderr-to-stdout python net.py
    ```

    表示在ip为DEVICE1_IP和DEVICE2_IP的机器上分别起8个进程运行程序。在其中一个节点执行：

    ```bash
    bash run_mpirun_1.sh
    ```

- 通过`mpirun --hostfile`方式。为方便调试，建议用这种方法来执行多机多卡脚本。首先需要构造hostfile文件如下：

    ```text
    DEVICE1 slots=8
    192.168.0.1 slots=8
    ```

    每一行格式为`[hostname] slots=[slotnum]`，hostname可以是ip或者主机名。上例表示在DEVICE1上有8张卡；ip为192.168.0.1的机器上也有8张卡。

    双机16卡的执行脚本如下，需要传入变量`HOSTFILE`，表示hostfile文件的路径：

    ```bash
    export DATA_PATH=./MNIST_Data/train/
    HOSTFILE=$1
    mpirun -n 16 --hostfile $HOSTFILE --output-filename log_output --merge-stderr-to-stdout python net.py
    ```

    在其中一个节点执行：

    ```bash
    bash run_mpirun_2.sh ./hostfile
    ```

执行完毕后，日志文件会保存到log_output目录下，结果保存在log_output/1/rank.*/stdout中。