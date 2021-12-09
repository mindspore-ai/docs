# 分布式并行训练 （GPU）

`GPU` `进阶` `分布式并行`

<!-- TOC -->

- [分布式并行训练 （GPU）](#分布式并行训练-gpu)
    - [准备环节](#准备环节)
        - [下载数据集](#下载数据集)
        - [配置分布式环境](#配置分布式环境)
        - [调用集合通信库](#调用集合通信库)
    - [定义网络](#定义网络)
    - [运行脚本](#运行脚本)
    - [运行多机脚本](#运行多机脚本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/intermediate/distributed_training/distributed_training_gpu.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本篇教程我们主要讲解，如何在GPU硬件平台上，利用MindSpore的数据并行及自动并行模式训练ResNet-50网络。

## 准备环节

### 下载数据集

本样例采用`CIFAR-10`数据集，由10类32*32的彩色图片组成，每类包含6000张图片。其中训练集共50000张图片，测试集共10000张图片。

> `CIFAR-10`数据集下载链接：<http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>。

将数据集下载并解压到本地路径下，解压后的文件夹为`cifar-10-batches-bin`。

### 配置分布式环境

- `OpenMPI-4.0.3`：MindSpore采用的多进程通信库。

  OpenMPI-4.0.3源码下载地址：<https://www.open-mpi.org/software/ompi/v4.0/>，选择`openmpi-4.0.3.tar.gz`下载。

  参考OpenMPI官网教程安装：<https://www.open-mpi.org/faq/?category=building#easy-build>。

- `NCCL-2.7.6`：Nvidia集合通信库。

  NCCL-2.7.6下载地址：<https://developer.nvidia.com/nccl/nccl-legacy-downloads>。

  参考NCCL官网教程安装：<https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian>。

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

## 定义网络

在GPU硬件平台上，网络的定义和Ascend 910 AI处理器一致。

可以参考[ResNet网络样例脚本](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/resnet/resnet.py)

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
