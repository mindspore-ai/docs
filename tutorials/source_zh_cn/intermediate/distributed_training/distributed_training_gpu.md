# 分布式并行训练 （GPU）

`GPU` `进阶` `分布式并行`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/intermediate/distributed_training/distributed_training_gpu.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本篇教程我们主要讲解，如何在GPU硬件平台上，利用MindSpore的数据并行及自动并行模式训练ResNet-50网络。如需了解并行模型等相关背景和概念，可参考[分布式并行训练（Ascend）](https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_ascend.html)。

> 你可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>

目录结构如下：

```text
└─sample_code
    ├─distributed_training
    │      resnet.py
    │      resnet50_distributed_training_gpu.py
    │      run_gpu.sh
    ...
```

其中，`resnet.py`、`resnet50_distributed_training.py`等文件是定义网络结构的脚本。`run_gpu.sh`是执行脚本。

## 准备环节

### 下载数据集

本样例采用`CIFAR-10`数据集，由10类32*32的彩色图片组成，每类包含6000张图片。其中训练集共50000张图片，测试集共10000张图片。

> `CIFAR-10`数据集下载链接：<http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>。

将数据集下载并解压到本地路径下，解压后的文件夹为`cifar-10-batches-bin`。

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

在GPU硬件平台上，MindSpore分布式并行训练的通信使用的是NCCL。

> GPU平台上，MindSpore暂不支持用户进行：
>
> `get_local_rank`、`get_local_size`、`get_world_rank_from_group_rank`、`get_group_rank_from_world_rank`、`create_group`操作。

下面是调用集合通信库的代码样例：

```python
from mindspore import context
from mindspore.communication import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    init("nccl")
    ...
```

其中，

- `mode=context.GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式。
- `init("nccl")`：使能NCCL通信，并完成分布式训练初始化操作。

## 数据并行模式加载数据集

在GPU硬件平台上，加载数据集的处理流程和Ascend 910 AI处理器一致。可参考[数据并行模式加载数据集](https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_ascend.html#数据并行模式加载数据集)。

## 定义网络

在GPU硬件平台上，网络的定义和Ascend 910 AI处理器一致，可以参考[ResNet网络样例脚本](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/resnet/resnet.py)。

## 定义损失函数及优化器

### 定义损失函数

在GPU硬件平台上，损失函数的定义和Ascend 910 AI处理器一致。可参考[定义损失函数](https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_ascend.html#定义损失函数)。

### 定义优化器

采用`Momentum`优化器作为参数更新工具，这里定义与单机一致，不再展开，具体可以参考样例代码中的实现。

## 训练网络

`context.set_auto_parallel_context`是配置并行训练参数的接口，必须在初始化网络之前调用。常用参数包括：

- `parallel_mode`：分布式并行模式，默认为单机模式`ParallelMode.STAND_ALONE`。可选数据并行`ParallelMode.DATA_PARALLEL`及自动并行`ParallelMode.AUTO_PARALLEL`。
- `parameter_broadcast`：训练开始前自动广播0号卡上数据并行的参数权值到其他卡上，默认值为`False`。
- `gradients_mean`：反向计算时，框架内部会将数据并行参数分散在多台机器的梯度值进行收集，得到全局梯度值后再传入优化器中更新。默认值为`False`，设置为True对应`allreduce_mean`操作，False对应`allreduce_sum`操作。
- `device_num`和`global_rank`建议采用默认值，框架内会调用HCCL接口获取。

如脚本中存在多个网络用例，请在执行下个用例前调用`context.reset_auto_parallel_context`将所有参数还原到默认值。

在下面的样例中我们指定并行模式为自动并行，用户如需切换为数据并行模式只需将`parallel_mode`改为`DATA_PARALLEL`。

```python
from mindspore import context, Model
from mindspore.nn import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.context import ParallelMode
from resnet import resnet50

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def test_train_cifar(epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    loss_cb = LossMonitor()
    data_path = os.getenv('DATA_PATH')
    dataset = create_dataset(data_path)
    batch_size = 32
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=False)
```

其中，

- `dataset_sink_mode=True`：表示采用数据集的下沉模式，即训练的计算下沉到硬件平台中执行。
- `LossMonitor`：能够通过回调函数返回Loss值，用于监控损失函数。

## 运行脚本

在GPU硬件平台上，MindSpore采用OpenMPI的`mpirun`进行分布式训练。下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

> 你可以在这里找到样例的运行脚本：
>
> <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/distributed_training/run_gpu.sh>。
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
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 8 pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &
```

脚本会在后台运行，日志文件会保存到device目录下，共跑了10个epoch，每个epoch有234个step，关于Loss部分结果保存在train.log中。选取部分示例，如下：

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

## 运行多机脚本

若训练涉及多机，则需要额外在`mpirun`命令中设置多机配置。你可以直接在`mpirun`命令中用`-H`选项进行设置，比如`mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 python hello.py`，表示在ip为DEVICE1_IP和DEVICE2_IP的机器上分别起8个进程运行程序；或者也可以构造一个如下这样的hostfile文件，并将其路径传给`mpirun`的`--hostfile`的选项。hostfile文件每一行格式为`[hostname] slots=[slotnum]`，hostname可以是ip或者主机名。

```bash
DEVICE1 slots=8
DEVICE2 slots=8
```

两机十六卡的执行脚本如下，需要传入变量`DATA_PATH`和`HOSTFILE`，表示数据集的路径和hostfile文件的路径。更多mpirun的选项设置可见OpenMPI的官网。

```bash
#!/bin/bash

DATA_PATH=$1
HOSTFILE=$2

rm -rf device
mkdir device
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 16 --hostfile $HOSTFILE -x DATA_PATH=$DATA_PATH -x PATH -mca pml ob1 pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &
```

在GPU上进行分布式训练时，模型参数的保存和加载可参考[分布式训练模型参数保存和加载](https://www.mindspore.cn/tutorials/zh-CN/master/intermediate/distributed_training/distributed_training_model_parameters_saving_and_loading.html)
