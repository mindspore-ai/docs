# 分布式并行训练基础样例（GPU）

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/experts/source_zh_cn/parallel/train_gpu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## 概述

本篇教程我们主要讲解，如何在GPU处理器硬件平台上，利用MindSpore通过数据并行及自动并行模式，使用CIFAR-10数据集训练ResNet-50网络。
> 你可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r1.7/docs/sample_code/distributed_training>

目录结构如下：

```text
└─sample_code
    ├─distributed_training
    │      rank_table_16pcs.json
    │      rank_table_8pcs.json
    │      rank_table_2pcs.json
    │      cell_wrapper.py
    │      model_accu.py
    │      resnet.py
    │      resnet50_distributed_training.py
    │      resnet50_distributed_training_gpu.py
    │      resnet50_distributed_training_grad_accu.py
    │      run.sh
    │      run_gpu.sh
    │      run_grad_accu.sh
    │      run_cluster.sh
```

其中，`resnet.py`和`resnet50_distributed_training_gpu.py`是定义网络结构的脚本。`run_gpu.sh`是执行脚本，其余文件为Ascend 910的样例代码。

## 准备环节

为了保证分布式训练的正常进行，我们需要先对分布式环境进行配置和初步的测试。在完成之后，再做CIFAR-10数据集的准备。

### 配置分布式环境

- `OpenMPI-4.0.3`：MindSpore采用的多进程通信库。

  OpenMPI-4.0.3源码下载地址：<https://www.open-mpi.org/software/ompi/v4.0/>，选择`openmpi-4.0.3.tar.gz`下载。

  参考OpenMPI官网教程安装：<https://www.open-mpi.org/faq/?category=building#easy-build>。

- 主机间免密登陆（涉及多机训练时需要）。若训练涉及多机，则需要配置多机间免密登陆，可参考以下步骤进行配置：
    1. 每台主机确定同一个用户作为登陆用户（不推荐root）；
    2. 执行`ssh-keygen -t rsa -P ""`生成密钥；
    3. 执行`ssh-copy-id DEVICE-IP`设置需要免密登陆的机器IP；
    4. 执行`ssh DEVICE-IP`，若不需要输入密码即可登录，则说明以上配置成功；
    5. 在所有机器上执行以上命令，确保两两互通。

### 调用集合通信库

在GPU硬件平台上，MindSpore分布式并行训练中的通信使用的是英伟达集合通信库`NVIDIA Collective Communication Library`(以下简称为NCCL)。

> GPU平台上，MindSpore暂不支持用户进行：
>
> `get_local_rank`、`get_local_size`、`get_world_rank_from_group_rank`、`get_group_rank_from_world_rank`、`create_group`操作。

下面是调用通信库的代码样例，设文件名为nccl_allgather.py：

```python
# nccl_allgather.py
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication import init, get_rank


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.allgather = ops.AllGather()

    def construct(self, x):
        return self.allgather(x)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    init("nccl")
    value = get_rank()
    input_x = Tensor(np.array([[value]]).astype(np.float32))
    net = Net()
    output = net(input_x)
    print(output)
```

其中，

- `mode=context.GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式（PyNative模式不支持并行）。
- `device_target="GPU"`: 指定设备为GPU。
- `init("nccl")`：使能NCCL通信，并完成分布式训练初始化操作。
- `get_rank()`：获得当前进程的rank号。
- `ops.AllGather`:
  在GPU上，该算子会调用NCCL的AllGather通信操作，其含义以及更多的例子可在[分布式集合通信原语](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/parallel/communicate_ops.html)
  中找到。

在GPU硬件平台上，MindSpore采用OpenMPI的mpirun来启动进程，通常每一个进程对应一个计算设备。

```bash
mpirun -n DEVICE_NUM python nccl_allgather.py
```

其中，DEVICE_NUM为所在机器的GPU数量。以DEVICE_NUM=4为例，预期的输出为：

```text
[[0.],
 [1.],
 [2.],
 [3.]]
```

输出日志在程序执行后，可在`log/1/rank.0`中找到。若得到以上输出，则说明OpenMPI和NCCL工作正常，进程正常启动。

### 下载数据集

本样例采用`CIFAR-10`数据集，由10类32*32的彩色图片组成，每类包含6000张图片，共60000张。其中训练集共50000张图片，测试集共10000张图片。

> `CIFAR-10`数据集下载链接：<http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>，如果点击下载不成功，请尝试右键另存为方式下载。

Linux机器可采用以下命令下载到终端当前路径并解压数据集，解压后的数据所在文件夹为`cifar-10-batches-bin`。

```bash
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz
```

## 数据并行模式加载数据集

分布式训练时，数据是以数据并行的方式导入的。下面我们以CIFAR-10数据集为例，介绍以数据并行方式导入CIFAR-10数据集的方法，`data_path`是指数据集的路径，即`cifar-10-batches-bin`文件夹的路径。

```python
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
from mindspore.communication import get_rank, get_group_size


def create_dataset(data_path, repeat_num=1, batch_size=32, rank_id=0, rank_size=1):
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
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
```

其中，与单机不同的是，在数据集接口需要传入`num_shards`和`shard_id`参数，分别对应卡的数量和逻辑序号，建议通过NCCL接口获取：

- `get_rank`：获取当前设备在集群中的ID。
- `get_group_size`：获取集群数量。

> 数据并行场景加载数据集时，建议对每卡指定相同的数据集文件，若是各卡加载的数据集不同，可能会影响计算精度。

## 定义网络

在GPU硬件平台上，网络的定义和Ascend 910 AI处理器一致。
**数据并行**及**自动并行**
模式下，网络定义方式与单机写法一致，可以参考 [ResNet网络样例脚本](https://gitee.com/mindspore/docs/blob/r1.7/docs/sample_code/resnet/resnet.py)。
。

> - 半自动并行模式时，未配置策略的算子默认以数据并行方式执行。
> - 自动并行模式支持通过策略搜索算法自动获取高效的算子并行策略，同时也支持用户对算子手动配置特定的并行策略。
> - 如果某个`parameter`被多个算子使用，则每个算子对这个`parameter`的切分策略需要保持一致，否则将报错。

## 定义损失函数及优化器

与在Ascend的[分布式并行训练基础样例](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/parallel/train_ascend.html)
一致。

### 定义损失函数

自动并行以算子为粒度切分模型，通过算法搜索得到最优并行策略，所以与单机训练不同的是，为了有更好的并行训练效果， 损失函数建议使用MindSpore算子来实现，而不是直接用封装好的损失函数类。

在Loss部分，我们采用`SoftmaxCrossEntropyWithLogits`的展开形式，即按照数学公式， 将其展开为多个MindSpore算子进行实现，样例代码如下：

```python
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.nn as nn


class SoftmaxCrossEntropyExpand(nn.Cell):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()

    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
```

### 定义优化器

采用`Momentum`优化器作为参数更新工具，这里定义与单机一致，不再展开，具体可以参考样例代码中的实现。

## 训练网络

训练之前，我们需要先配置一些自动并行的参数。`context.set_auto_parallel_context`是配置并行训练模式的接口，必须在初始化网络之前调用。常用参数包括：

- `parallel_mode`：分布式并行模式，默认为单机模式`ParallelMode.STAND_ALONE`。在本例中，可选择数据并行`ParallelMode.DATA_PARALLEL`
  及自动并行`ParallelMode.AUTO_PARALLEL`。
- `parameter_broadcast`：训练开始前自动广播0号卡上数据并行的参数权值到其他卡上，默认值为`False`。
- `gradients_mean`：反向计算时，框架内部会将数据并行参数分散在多台机器的梯度值进行收集，得到全局梯度值后再传入优化器中更新。默认值为`False`，设置为True对应`allreduce_mean`
  操作，False对应`allreduce_sum`操作。
- `device_num`和`global_rank`建议采用默认值，框架内会调用NCCL接口获取。

如脚本中存在多个网络用例，请在执行下个用例前调用`context.reset_auto_parallel_context`将所有参数还原到默认值。

在下面的样例中我们指定并行模式为自动并行，用户如需切换为数据并行模式只需将`parallel_mode`改为`DATA_PARALLEL`。

```python
from mindspore import context, Model
from mindspore.nn import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.context import ParallelMode
from mindspore.communication import init
from resnet import resnet50

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
init("nccl")


def test_train_cifar(epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    loss_cb = LossMonitor()
    dataset = create_dataset(data_path)
    batch_size = 32
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
```

其中，

- `dataset_sink_mode=True`：表示采用数据集的下沉模式，即训练的计算下沉到硬件平台中执行。
- `LossMonitor`：能够通过回调函数返回Loss值，用于监控损失函数。

## 运行脚本

在GPU硬件平台上，MindSpore采用OpenMPI的`mpirun`进行分布式训练。 在完成了模型、损失函数和优化器的定义之后，我们就完成了模型的并行策略的配置， 下面直接执行运行脚本。

### 单机多卡训练

下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

> 你可以在这里找到样例的运行脚本：
>
> <https://gitee.com/mindspore/docs/blob/r1.7/docs/sample_code/distributed_training/run_gpu.sh>。
>
> 如果通过root用户执行脚本，`mpirun`需要加上`--allow-run-as-root`参数。

```bash
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_gpu.sh DATA_PATH"
echo "For example: bash run_gpu.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 8 pytest -s -v ./resnet50_distributed_training_gpu.py > train.log 2>&1 &
```

脚本会在后台运行，日志文件会保存到device目录下，共跑了10个epoch，每个epoch有234个step，关于Loss部分结果保存在train.log中。将loss值grep出来后，示例如下：

```text
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
```

### 多机多卡训练

在运行多机多卡训练前，需要保证每个节点上都有相同的OpenMPI、NCCL、Python以及MindSpore版本。

#### mpirun -H

若训练涉及多机，则需要额外在`mpirun`命令中设置多机配置。你可以直接在`mpirun`命令中用`-H`选项进行设置，比如

```text
mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 python hello.py
```

表示在ip为DEVICE1_IP和DEVICE2_IP的机器上分别起8个进程运行程序。

#### mpirun --hostfile

GPU的多机多卡的执行也可以通过构造hostfile文件来进行。 为方便调试，建议用这种方法来执行多机多卡脚本。 之后使用`mpirun --hostfile $HOST_FILE`的形式来执行。下面我们以hostfile启动方式来给出详细的多机多卡配置。

hostfile文件每一行格式为`[hostname] slots=[slotnum]`，hostname可以是ip或者主机名。需要注意的是，不同机器上的用户名需要相同，但是hostname不可以相同。如下，表示在DEVICE1上有8张卡；ip为192.168.0.1的机器上也有8张卡：

```text
DEVICE1 slots=8
192.168.0.1 slots=8
```

两机十六卡的执行脚本如下，需要传入变量`DATA_PATH`和`HOSTFILE`，表示数据集的路径和hostfile文件的路径。我们需要设置mpi中mca的btl参数来指定进行mpi通信的网卡，否则可能会在调用mpi接口时初始化失败。btl参数指定了节点间采用tcp协议，节点内采用环路进行通信。btl_tcp_if_include指定节点间通信所经过的网卡的ip地址需要在给定的子网中。 更多mpirun的选项设置可见OpenMPI的官网。

```bash
#!/bin/bash

DATA_PATH=$1
HOSTFILE=$2

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 16 --mca btl tcp,self --mca btl_tcp_if_include 192.168.0.0/24 --hostfile $HOSTFILE -x DATA_PATH=$DATA_PATH -x PATH -mca pml ob1 mpirun_gpu_clusher.sh &
```

考虑到不同机器上的一些环境变量可能会不一样，我们采用mpirun启动一个`mpirun_gpu_cluster.sh`的形式，在不同机器上的该脚本文件中指定所需的环境变量。此处我们配置了`NCCL_SOCKET_IFNAME`，来指定NCCL进行通信时的网卡。

```bash
#!/bin/bash
# mpirun_gpu_clusher.sh
# 你可以在这里设置每台机器上不同的环境变量，如下面的网卡名字

export NCCL_SOCKET_IFNAME="en5" # 需进行节点间通信的网卡的名字，不同机器上可能不一致，使用ifconfig查看。
pytest -s -v ./resnet50_distributed_training_gpu.py > train.log 2>&1 &
```

## 分布式训练模型参数保存与加载

在GPU上进行分布式训练时，模型参数的保存和加载的方法与Ascend上一致，可参考[分布式训练模型参数保存和加载](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/parallel/train_ascend.html#分布式训练模型参数保存和加载)
。

## 不依赖OpenMPI进行训练

出于训练时的安全及可靠性要求，MindSpore GPU还支持**不依赖OpenMPI的分布式训练**。

OpenMPI在分布式训练的场景中，起到在Host侧同步数据以及进程间组网的功能；MindSpore通过**复用Parameter Server模式训练架构**，取代了OpenMPI能力。

参考[Parameter Server模式](https://www.mindspore.cn/docs/zh-CN/r1.7/design/parameter_server_training.html)训练教程，将多个MindSpore训练进程作为`Worker`启动，并且额外启动一个`Scheduler`，对脚本做少量修改，即可执行**不依赖OpenMPI的分布式训练**。

执行Worker脚本前需要导出环境变量，如[环境变量设置](https://www.mindspore.cn/docs/zh-CN/r1.7/design/parameter_server_training.html#环境变量设置):

```text
export MS_SERVER_NUM=0                # Server number
export MS_WORKER_NUM=8                # Worker number
export MS_SCHED_HOST=127.0.0.1        # Scheduler IP address
export MS_SCHED_PORT=6667             # Scheduler port
export MS_ROLE=MS_WORKER              # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

> 在此模式下，不建议启动MS_PSERVER角色的进程，因为此角色在数据并行训练中无影响。

### 运行脚本

在GPU硬件平台上，下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

> 你可以在这里找到样例的运行目录：
>
> <https://gitee.com/mindspore/docs/tree/r1.7/docs/sample_code/distributed_training>。

相比OpenMPI方式启动，此模式需要调用[Parameter Server模式](https://www.mindspore.cn/docs/zh-CN/r1.7/design/parameter_server_training.html)中的`set_ps_context`接口，告诉MindSpore此次任务使用了PS模式训练架构:

```python
from mindspore import context
from mindspore.communication import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_ps_context(config_file_path="/path/to/config_file.json", enable_ssl=True,
                           client_password="123456", server_password="123456")
    init("nccl")
    ...
```

其中，

- `mode=context.GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式（PyNative模式不支持并行）。
- `init("nccl")`：使能NCCL通信，并完成分布式训练初始化操作。
- 默认情况下，安全加密通道是关闭的，需要通过`set_ps_context`正确配置安全加密通道或者关闭安全加密通道后，才能调用init("nccl")，否则初始化组网会失败。

若想使用安全加密通道，请设置`context.set_ps_context(config_file_path="/path/to/config_file.json", enable_ssl=True, client_password="123456", server_password="123456")`
等配置，详细参数配置说明请参考Python API [mindspore.context.set_ps_context](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.context.html#mindspore.context.set_ps_context)，以及本文档[安全认证](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/parallel/train_gpu.html#安全认证)章节。

脚本内容`run_gpu_cluster.sh`如下，在启动Worker和Scheduler之前，需要添加相关环境变量设置：

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 8 workers.
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &
done

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
pytest -s -v ./resnet50_distributed_training_gpu.py > scheduler.log 2>&1 &
```

执行如下指令：

```bash
./run_gpu_cluster.sh DATA_PATH
```

即可单机内部执行8卡分布式训练，若希望执行跨机训练，则需要将脚本拆分，如执行2机8卡训练，每台机器执行启动4Worker：

脚本`run_gpu_cluster_1.sh`在机器1上启动1`Scheduler`和`Worker1`到`Worker4`：

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 1-4 workers.
for((i=0;i<4;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &
done

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
pytest -s -v ./resnet50_distributed_training_gpu.py > scheduler.log 2>&1 &
```

脚本`run_gpu_cluster_2.sh`在机器2上启动`Worker5`到`Worker8`(无需再执行Scheduler)：

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 5-8 workers.
for((i=4;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &
done
```

在两台主机分别执行：

```bash
./run_gpu_cluster_1.sh DATA_PATH
```

```bash
./run_gpu_cluster_2.sh DATA_PATH
```

即可执行2机8卡分布式训练任务。

若希望启动数据并行模式训练，需要将脚本`resnet50_distributed_training_gpu.py`中`set_auto_parallel_context`入参并行模式改为`DATA_PARALLEL`:

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
```

脚本会在后台运行，日志文件会保存到当前目录下，共跑了10个epoch，每个epoch有234个step，关于Loss部分结果保存在worker_*.log中。将loss值grep出来后，示例如下：

```text
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
```

### 安全认证

要支持节点/进程间的SSL安全认证，要开启安全认证，通过Python API `mindspore.context.set_ps_context`配置`enable_ssl=True`(不传入时默认为False，表示不启用SSL安全认证)，config_file_path指定的config.json配置文件需要添加如下字段：

```json
{
  "server_cert_path": "server.p12",
  "crl_path": "",
  "client_cert_path": "client.p12",
  "ca_cert_path": "ca.crt",
  "cipher_list": "ECDHE-R SA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384:DHE-DSS-AES256-GCM-SHA384:DHE-PSK-AES128-GCM-SHA256:DHE-PSK-AES256-GCM-SHA384:DHE-PSK-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-PSK-CHACHA20-POLY1305:DHE-RSA-AES128-CCM:DHE-RSA-AES256-CCM:DHE-RSA-CHACHA20-POLY1305:DHE-PSK-AES128-CCM:DHE-PSK-AES256-CCM:ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305",
  "cert_expire_warning_time_in_day": 90
}
```

- server_cert_path: 服务端包含了证书和秘钥的密文的p12文件（SSL专用证书文件）路径。
- crl_path: 吊销列表（用于区分无效不可信证书和有效可信证书）的文件路径。
- client_cert_path: 客户端包含了证书和秘钥的密文的p12文件（SSL专用证书文件）路径。
- ca_cert_path: 根证书路径。
- cipher_list: 密码套件（支持的SSL加密类型列表）。
- cert_expire_warning_time_in_day: 证书过期的告警时间。

p12文件中的秘钥为密文存储，在启动时需要传入密码，具体参数请参考Python API [mindspore.context.set_ps_context](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.context.html#mindspore.context.set_ps_context)中的`client_password`以及`server_password`字段。

### 容灾恢复

模型训练对分布式训练架构的可靠性、可服务性要求比较高，MindSpore支持数据并行下容灾恢复，多卡数据并行训练场景集群(多个Worker和1个Scheduler)中存在进程异常退出，被重新拉起后，训练任务继续能正常执行；

场景约束：
在图模式下，采用`MindData`进行数据下沉模式训练，开启数据并行模式，采用上述的非`OpenMPI`的方式拉起Worker进程。

在上述场景下，训练过程中如果有节点挂掉，保证在相同的环境变量（`MS_ENABLE_RECOVERY` 和 `MS_RECOVERY_PATH`）下，重新拉起对应进程对应的脚本后训练可继续，并且不影响精度收敛。

1） 开启容灾：

通过环境变量开启容灾：

```bash
export MS_ENABLE_RECOVERY=1             #开启容灾
export MS_RECOVERY_PATH=“/xxx/xxx”      #配置持久化路径文件夹，Worker和Scheduler进程在执行过程中会进行必要的持久化，如用于恢复组网的节点信息以及训练业务中间状态等
```

2）配置checkpoint保存间隔，样例如下：

```python
ckptconfig = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
ckpoint_cb = ModelCheckpoint(prefix='train', directory="./ckpt_of_rank_/"+str(get_rank()), config=ckptconfig)
```

每个Worker都开启保存checkpoint，并用不同的路径（如上述样例中的directory的设置使用了rank id，保证路径不会相同），防止同名checkpoint保存冲突。checkpoint用于异常进程恢复和正常进程回滚，训练的回滚是指集群中各个Worker都恢复到最新的checkpoint对应的状态，同时数据侧也回退到对应的step，然后继续训练。保存checkpoint的间隔是可配置的，这个间隔决定了容灾恢复的粒度，间隔越小，恢复到上次保存checkpoint所回退的step数就越小，但保存checkpoint频繁也可能会影响训练效率，间隔越大则效果相反。keep_checkpoint_max至少设置为2(防止checkpoint保存失败)。

> 样例的运行目录：
>
> <https://gitee.com/mindspore/docs/tree/r1.7/docs/sample_code/distributed_training>。

涉及到的脚本有`run_gpu_cluster_recovery.sh`, `resnet50_distributed_training_gpu_recovery.py`, `resnet.py`
脚本内容`run_gpu_cluster_recovery.sh`如下

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster_recovery.sh DATA_PATH"
echo "For example: bash run_gpu_cluster_recovery.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

export MS_ENABLE_RECOVERY=1      # Enable recovery
export MS_RECOVERY_PATH=/XXX/XXX # Set recovery path

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu_recovery.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
export MS_NODE_ID=sched               # The node id for Scheduler
pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > scheduler.log 2>&1 &

# Launch 8 workers.
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    export MS_NODE_ID=worker_$i           # The node id for Workers
    pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > worker_$i.log 2>&1 &
done
```

在启动Worker和Scheduler之前，需要添加相关环境变量设置, 如Scheduler的IP和Port，当前进程的角色是Worker还是Scheduler。

执行下面的命令即可启动一个单机8卡的数据并行训练

```bash
bash run_gpu_cluster_recovery.sh YOUR_DATA_PATH"
```

分布式训练开始，若训练过程中遇到异常，如进程异常退出，然后再重新启动对应的进程，训练流程即可恢复：
例如训练中途Scheduler进程异常退出，可执行下列命令重新启动Scheduler：

```bash
export DATA_PATH=YOUR_DATA_PATH
export MS_ENABLE_RECOVERY=1           # Enable recovery
export MS_RECOVERY_PATH=/XXX/XXX      # Set recovery path

cd ./device

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
export MS_NODE_ID=sched               # The node id for Scheduler
pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > scheduler.log 2>&1 &
```

Worker和Scheduler的组网会自动恢复。

Worker进程出现异常退出处理方式类似(注:Worker进程出现异常退出，需要等30s后再拉起才能恢复训练，在这之前，Scheduler为了防止网络抖动和恶意注册，拒绝相同node id的Worker再次注册)。

## 在K8s集群中使用ms-operator进行分布式训练

MindSpore Operator 是MindSpore在Kubernetes上进行分布式训练的插件。CRD（Custom Resource Definition）中定义了Scheduler、PS、Worker三种角色，用户只需配置yaml文件，即可轻松实现分布式训练。

当前ms-operator支持普通单Worker训练、PS模式的单Worker训练以及自动并行（例如数据并行、模型并行等）的Scheduler、Worker启动。详细流程请参考[ms-operator](https://gitee.com/mindspore/ms-operator)。