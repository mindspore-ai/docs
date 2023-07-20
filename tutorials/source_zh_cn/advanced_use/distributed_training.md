# 分布式并行训练

## 概述

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。根据并行的原理及模式不同，业界主流的并行类型有以下几种：

- 数据并行（Data Parallel）：对数据进行切分的并行模式，一般按照batch维度切分，将数据分配到各个计算单元（worker）中，进行模型计算。
- 模型并行（Model Parallel）：对模型进行切分的并行模式。MindSpore中支持层内模型并行模式，对参数切分后分配到各个计算单元中进行训练。
- 混合并行（Hybrid Parallel）：涵盖数据并行和模型并行的并行模式。

当前MindSpore也提供分布式并行训练的功能。它支持了多种模式包括：

- `DATA_PARALLEL`：数据并行模式。
- `AUTO_PARALLEL`：自动并行模式，融合了数据并行、模型并行及混合并行的1种分布式并行模式，可以自动建立代价模型，为用户选择1种并行模式。其中，代价模型指围绕Ascend 910芯片基于内存的计算开销和通信开销对训练时间建模，并设计高效的算法找到训练时间较短的并行策略。

本篇教程我们主要讲解如何在MindSpore上通过数据并行及自动并行模式训练ResNet-50网络。
> 本例面向Ascend 910 AI处理器硬件平台，暂不支持CPU和GPU场景。
> 你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/docs/blob/r0.2/tutorials/tutorial_code/distributed_training/resnet50_distributed_training.py>

## 准备环节

### 配置分布式环境变量

在裸机环境（对比云上环境，即本地有Ascend 910 AI 处理器）进行分布式训练时，需要配置当前多卡环境的组网信息文件。如果使用华为云环境，因为云服务本身已经做好了配置，可以跳过本小节。

以Ascend 910 AI处理器为例，1个8卡环境的json配置文件示例如下，本样例将该配置文件命名为rank_table.json。

```json
{
    "board_id": "0x0000",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "8",
            "server_num": "1",
            "group_name": "",
            "instance_count": "8",
            "instance_list": [
                {"devices": [{"device_id": "0","device_ip": "192.1.27.6"}],"rank_id": "0","server_id": "10.*.*.*"},
                {"devices": [{"device_id": "1","device_ip": "192.2.27.6"}],"rank_id": "1","server_id": "10.*.*.*"},
                {"devices": [{"device_id": "2","device_ip": "192.3.27.6"}],"rank_id": "2","server_id": "10.*.*.*"},
                {"devices": [{"device_id": "3","device_ip": "192.4.27.6"}],"rank_id": "3","server_id": "10.*.*.*"},
                {"devices": [{"device_id": "4","device_ip": "192.1.27.7"}],"rank_id": "4","server_id": "10.*.*.*"},
                {"devices": [{"device_id": "5","device_ip": "192.2.27.7"}],"rank_id": "5","server_id": "10.*.*.*"},
                {"devices": [{"device_id": "6","device_ip": "192.3.27.7"}],"rank_id": "6","server_id": "10.*.*.*"},
                {"devices": [{"device_id": "7","device_ip": "192.4.27.7"}],"rank_id": "7","server_id": "10.*.*.*"},
                ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": ["eth0","eth1","eth2","eth3","eth4","eth5","eth6","eth7"],
    "para_plane_nic_num": "8",
    "status": "completed"
}

```

其中需要根据实际训练环境修改的参数项有：

- `board_id`表示当前运行的环境，x86设为`0x0000`，arm设为`0x0020`。
- `server_num`表示机器数量， `server_id`表示本机IP地址。
- `device_num`、`para_plane_nic_num`及`instance_count`表示卡的数量。
- `rank_id`表示卡逻辑序号，固定从0开始编号，`device_id`表示卡物理序号，即卡所在机器中的实际序号。
- `device_ip`表示网卡IP地址，可以在当前机器执行指令`cat /etc/hccn.conf`，`address_x`的键值就是网卡IP地址。
- `para_plane_nic_name`对应网卡名称。

### 调用集合通信库

MindSpore分布式并行训练的通信使用了华为集合通信库`Huawei Collective Communication Library`（以下简称HCCL），可以在Ascend AI处理器配套的软件包中找到。同时`mindspore.communication.management`中封装了HCCL提供的集合通信接口，方便用户配置分布式信息。
> HCCL实现了基于Ascend AI处理器的多机多卡通信，有一些使用限制，我们列出使用分布式服务常见的，详细的可以查看HCCL对应的使用文档。
>
> - 单机场景下支持1、2、4、8卡设备集群，多机场景下支持8*n卡设备集群。
> - 每台机器的0-3卡和4-7卡各为1个组网，2卡和4卡训练时网卡必须相连且不支持跨组网创建集群。
> - 服务器硬件架构及操作系统需要是SMP（Symmetrical Multi-Processing，对称多处理器）处理模式。

下面是调用集合通信库样例代码：

```python
import os
from mindspore import context
from mindspore.communication.management import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", enable_hccl=True, device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

其中，

- `mode=context.GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式（PyNative模式不支持并行）。
- `enable_hccl=True`：使能HCCL通信。
- `device_id`：卡的物理序号，即卡所在机器中的实际序号。
- `init()`：完成分布式训练初始化操作。

## 数据并行模式加载数据集

分布式训练时，数据是以数据并行的方式导入的。下面我们以CIFAR-10数据集为例，介绍以数据并行方式导入CIFAR-10数据集的方法，`data_path`是指数据集的路径。

```python
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore.communication.management import get_rank, get_group_size

def create_dataset(repeat_num=1, batch_size=32, rank_id=0, rank_size=1):
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # get rank_id and rank_size
    rank_id = get_rank()
    rank_size = get_group_size()
    data_set = ds.Cifar10Dataset(data_path, num_shards=rank_size, shard_id=rank_id)

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(input_columns="label", operations=type_cast_op)
    data_set = data_set.map(input_columns="image", operations=c_trans)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
```

其中，与单机不同的是，在数据集接口需要传入`num_shards`和`shard_id`参数，分别对应网卡数量和逻辑序号，建议通过HCCL接口获取：

- `get_rank`：获取当前设备在集群中的ID。
- `get_group_size`：获取集群数量。

## 定义网络

数据并行及自动并行模式下，网络定义方式与单机一致。代码请参考： <https://gitee.com/mindspore/docs/blob/r0.2/tutorials/tutorial_code/resnet/resnet.py>

## 定义损失函数及优化器

### 定义损失函数

自动并行以展开Loss中的算子为粒度，通过算法搜索得到最优并行策略，所以与单机训练不同的是，为了有更好的并行训练效果，损失函数建议使用小算子来实现。

在Loss部分，我们采用`SoftmaxCrossEntropyWithLogits`的展开形式，即按照数学公式，将其展开为多个小算子进行实现，样例代码如下：

```python
from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
import mindspore.nn as nn

class SoftmaxCrossEntropyExpand(nn.Cell):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = P.Div()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.mean = P.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()

    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(F.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
```

### 定义优化器

采用`Momentum`优化器作为参数更新工具，这里定义与单机一致，不再展开，具体可以参考样例代码中的实现。

## 训练网络

`context.set_auto_parallel_context()`是配置并行训练参数的接口，必须在`Model`初始化前调用。如用户未指定参数，框架会自动根据并行模式为用户设置参数的经验值。如数据并行模式下，`parameter_broadcast`默认打开。主要参数包括：

- `parallel_mode`：分布式并行模式，默认为单机模式`ParallelMode.STAND_ALONE`。可选数据并行`ParallelMode.DATA_PARALLEL`及自动并行`ParallelMode.AUTO_PARALLEL`。
- `paramater_broadcast`： 参数初始化广播开关，非数据并行模式下，默认值为`False`。
- `mirror_mean`：反向计算时，框架内部会将数据并行参数分散在多台机器的梯度值进行收集，得到全局梯度值后再传入优化器中更新。默认值为`False`，设置为True对应`allreduce_mean`操作，False对应`allreduce_sum`操作。

> `device_num`和`global_rank`建议采用默认值，框架内会调用HCCL接口获取。

在下面的样例中我们指定并行模式为自动并行，用户如需切换为数据并行模式，只需将`parallel_mode`改为`DATA_PARALLEL`。

```python
from mindspore import context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.train.model import Model, ParallelMode
from resnet import resnet50

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(enable_task_sink=True, device_id=device_id) # set task_sink and device_id
context.set_context(enable_hccl=True) # set enable_hccl
context.set_context(enable_loop_sink=True)

def test_train_cifar(num_classes=10, epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, mirror_mean=True)
    loss_cb = LossMonitor()
    dataset = create_dataset(epoch_size)
    net = resnet50(32, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
```

其中，

- `dataset_sink_mode=True`，`enable_task_sink=True`,`enable_loop_sink=True`：表示采用数据集和任务的下沉模式，即训练的计算下沉到硬件平台中执行。
- `LossMonitor`：能够通过回调函数返回Loss值，用于监控损失函数。

## 运行脚本

上述已将训练所需的脚本编辑好了，接下来通过命令调用对应的脚本。

目前MindSpore分布式执行采用单卡单进程运行方式，即每张卡上运行1个进程，进程数量与使用的卡的数量一致。每个进程创建1个目录，用来保存日志信息以及算子编译信息。下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

```bash
#!/bin/bash

export RANK_TABLE_FILE=./rank_table.json
export RANK_SIZE=8
for((i=0;i<$RANK_SIZE;i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./resnet50_distributed_training.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ./resnet50_distributed_training.py > train.log$i 2>&1 &
    cd ../
done
```

其中必要的环境变量有，

- `RANK_TABLE_FILE`：组网信息文件的路径。
- `DEVICE_ID`：当前网卡在机器上的实际序号。
其余环境变量请参考安装教程中的配置项。

运行时间大约在5分钟内，主要时间是用于算子的编译，实际训练时间在20秒内。用户可以通过`ps -ef | grep pytest`来监控任务进程。

日志文件保存device目录下，env.log中记录了环境变量的相关信息，关于Loss部分结果保存在train.log中，示例如下：

```
test_resnet50_expand_loss_8p.py::test_train_feed ===============ds_num 195
global_step: 194, loss: 1.997
global_step: 389, loss: 1.655
global_step: 584, loss: 1.723
global_step: 779, loss: 1.807
global_step: 974, loss: 1.417
global_step: 1169, loss: 1.195
global_step: 1364, loss: 1.238
global_step: 1559, loss: 1.456
global_step: 1754, loss: 0.987
global_step: 1949, loss: 1.035
end training
PASSED
```
