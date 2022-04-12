# 分布式并行训练基础样例（Ascend）

`Ascend` `分布式并行` `全流程`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/train_ascend.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本篇教程我们主要讲解，如何在Ascend 910 AI处理器硬件平台上，利用MindSpore通过数据并行及自动并行模式训练ResNet-50网络。
> 你可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>

目录结构如下：

```text
└─sample_code
    ├─distributed_training
        ├── cell_wrapper.py
        ├── rank_table_16pcs.json
        ├── rank_table_2pcs.json
        ├── rank_table_8pcs.json
        ├── resnet50_distributed_training_dataset_slice.py
        ├── resnet50_distributed_training_gpu.py
        ├── resnet50_distributed_training_pipeline.py
        ├── resnet50_distributed_training.py
        ├── resnet.py
        ├── run_cluster.sh
        ├── run_dataset_slice.sh
        ├── run_gpu.sh
        ├── run_pipeline.sh
        └── run.sh
    ...
```

其中，`rank_table_16pcs.json`、`rank_table_8pcs.json`、`rank_table_2pcs.json`是配置当前多卡环境的组网信息文件。`resnet.py`、`resnet50_distributed_training.py`等文件是定义网络结构的脚本。`run.sh`、`run_cluster.sh`是执行脚本。

此外在[定义网络](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html#定义网络)和[分布式训练模型参数保存和加载](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html#分布式训练模型参数保存和加载)小节中我们针对手动混合并行模式和半自动并行模式的使用做了特殊说明。

## 准备环节

### 下载数据集

本样例采用`CIFAR-10`数据集，由10类32*32的彩色图片组成，每类包含6000张图片。其中训练集共50000张图片，测试集共10000张图片。

> `CIFAR-10`数据集下载链接：<http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>。

将数据集下载并解压到本地路径下，解压后的文件夹为`cifar-10-batches-bin`。

### 配置分布式环境变量

在裸机环境（对比云上环境，即本地有Ascend 910 AI 处理器）进行分布式训练时，需要配置当前多卡环境的组网信息文件。如果使用华为云环境，因为云服务本身已经做好了配置，可以跳过本小节。

以Ascend 910 AI处理器为例，1个8卡环境的json配置文件示例如下，本样例将该配置文件命名为`rank_table_8pcs.json`。2卡环境配置可以参考样例代码中的`rank_table_2pcs.json`文件。

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

其中需要根据实际训练环境修改的参数项有：

- `server_count`表示参与训练的机器数量。
- `server_id`表示当前机器的IP地址。
- `device_id`表示卡物理序号，即卡所在机器中的实际序号。
- `device_ip`表示集成网卡的IP地址，可以在当前机器执行指令`cat /etc/hccn.conf`，`address_x`的键值就是网卡IP地址。
- `rank_id`表示卡逻辑序号，固定从0开始编号。

### 调用集合通信库

MindSpore分布式并行训练的通信使用了华为集合通信库`Huawei Collective Communication Library`（以下简称HCCL），可以在Ascend AI处理器配套的软件包中找到。同时`mindspore.communication.management`中封装了HCCL提供的集合通信接口，方便用户配置分布式信息。
> HCCL实现了基于Ascend AI处理器的多机多卡通信，有一些使用限制，我们列出使用分布式服务常见的，详细的可以查看HCCL对应的使用文档。
>
> - 单机场景下支持1、2、4、8卡设备集群，多机场景下支持8*n卡设备集群。
> - 每台机器的0-3卡和4-7卡各为1个组网，2卡和4卡训练时卡必须相连且不支持跨组网创建集群。
> - 组建多机集群时需要保证各台机器使用同一交换机。
> - 服务器硬件架构及操作系统需要是SMP（Symmetrical Multi-Processing，对称多处理器）处理模式。
> - PyNative模式下当前仅支持全局单Group通信。

下面是调用集合通信库样例代码：

```python
import os
from mindspore import context
from mindspore.communication import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

其中，

- `mode=context.GRAPH_MODE`：使用分布式训练需要指定运行模式为图模式（PyNative模式当前仅支持数据并行）。
- `device_id`：卡的物理序号，即卡所在机器中的实际序号。
- `init`：使能HCCL通信，并完成分布式训练初始化操作。

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

其中，与单机不同的是，在数据集接口需要传入`num_shards`和`shard_id`参数，分别对应卡的数量和逻辑序号，建议通过HCCL接口获取：

- `get_rank`：获取当前设备在集群中的ID。
- `get_group_size`：获取集群数量。

> 数据并行场景加载数据集时，建议对每卡指定相同的数据集文件，若是各卡加载的数据集不同，可能会影响计算精度。

## 定义网络

数据并行及自动并行模式下，网络定义方式与单机写法一致，可以参考[ResNet网络样例脚本](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/resnet/resnet.py)。

本章节重点介绍手动混合并行和半自动并行模式的网络定义方法。

### 手动混合并行模式

手动混合并行模式在数据并行模式的基础上，对`parameter`增加了模型并行`layerwise_parallel`配置，包含此配置的`parameter`将以切片的形式保存并参与计算，在优化器计算时不会进行梯度累加。在该模式下，框架不会自动插入并行算子前后需要的计算和通信操作，为了保证计算逻辑的正确性，用户需要手动推导并写在网络结构中，适合对并行原理深入了解的用户使用。

以下面的代码为例，将`self.weight`指定为模型并行配置，即`self.weight`和`MatMul`的输出在第二维`channel`上存在切分。这时再在第二维上进行`ReduceSum`得到的仅是单卡累加结果，还需要引入`AllReduce.Sum`通信操作对每卡的结果做加和。关于并行算子的推导原理可以参考这篇[设计文档](https://www.mindspore.cn/docs/zh-CN/master/design/distributed_training_design.html#自动并行原理)。

```python
from mindspore import Tensor
import mindspore.ops as ops
from mindspore import dtype as mstype
import mindspore.nn as nn

class HybridParallelNet(nn.Cell):
    def __init__(self):
        super(HybridParallelNet, self).__init__()
        # initialize the weight which is sliced at the second dimension
        weight_init = np.random.rand(512, 128/2).astype(np.float32)
        self.weight = Parameter(Tensor(weight_init), layerwise_parallel=True)
        self.fc = ops.MatMul()
        self.reduce = ops.ReduceSum()
        self.allreduce = ops.AllReduce(op='sum')

    def construct(self, x):
        x = self.fc(x, self.weight)
        x = self.reduce(x, -1)
        x = self.allreduce(x)
        return x
```

### 半自动并行模式

半自动并行模式相较于自动并行模式需要用户手动配置并行策略进行调优。关于算子并行策略的定义可以参考这篇[设计文档](https://www.mindspore.cn/docs/zh-CN/master/design/distributed_training_design.html#自动并行原理)。

以前述的`HybridParallelNet`为例，在半自动并行模式下的脚本代码如下，`MatMul`的切分策略为`((1, 1),(1, 2))`，指定`self.weight`在第二维度上被切分两份。

```python
from mindspore import Tensor
import mindspore.ops as ops
from mindspore import dtype as mstype
import mindspore.nn as nn

class SemiAutoParallelNet(nn.Cell):
    def __init__(self):
        super(SemiAutoParallelNet, self).__init__()
        # initialize full tensor weight
        weight_init = np.random.rand(512, 128).astype(np.float32)
        self.weight = Parameter(Tensor(weight_init))
        # set shard strategy
        self.fc = ops.MatMul().shard(((1, 1),(1, 2)))
        self.reduce = ops.ReduceSum()

    def construct(self, x):
        x = self.fc(x, self.weight)
        x = self.reduce(x, -1)
        return x
```

> - 半自动并行模式时，未配置策略的算子默认以数据并行方式执行。
> - 自动并行模式支持通过策略搜索算法自动获取高效的算子并行策略，同时也支持用户对算子手动配置特定的并行策略。
> - 如果某个`parameter`被多个算子使用，则每个算子对这个`parameter`的切分策略需要保持一致，否则将报错。

## 定义损失函数及优化器

### 定义损失函数

自动并行以算子为粒度切分模型，通过算法搜索得到最优并行策略，所以与单机训练不同的是，为了有更好的并行训练效果，损失函数建议使用小算子来实现。

在Loss部分，我们采用`SoftmaxCrossEntropyWithLogits`的展开形式，即按照数学公式，将其展开为多个小算子进行实现，样例代码如下：

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
        self.eps = Tensor(1e-24, mstype.float32)

    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result+self.eps)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
```

### 定义优化器

采用`Momentum`优化器作为参数更新工具，这里定义与单机一致，不再展开，具体可以参考样例代码中的实现。

## 训练网络

`context.set_auto_parallel_context`是配置并行训练参数的接口，必须在初始化网络之前调用。常用参数包括：

- `parallel_mode`：分布式并行模式，默认为单机模式`ParallelMode.STAND_ALONE`。可选数据并行`ParallelMode.DATA_PARALLEL`及自动并行`ParallelMode.AUTO_PARALLEL`。
- `parameter_broadcast`：训练开始前自动广播0号卡上数据并行的参数权值到其他卡上，默认值为`False`。
- `gradients_mean`：反向计算时，框架内部会将数据并行参数分散在多台机器的梯度值进行收集，得到全局梯度值后再传入优化器中更新。默认值为`False`，设置为True对应`AllReduce.Mean`操作，False对应`AllReduce.Sum`操作。
- `device_num`和`global_rank`建议采用默认值，框架内会调用HCCL接口获取。

> 更多分布式并行配置项用户请参考[编程指南](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/auto_parallel.html)。

如脚本中存在多个网络用例，请在执行下个用例前调用`context.reset_auto_parallel_context`将所有参数还原到默认值。

在下面的样例中我们指定并行模式为自动并行，用户如需切换为数据并行模式只需将`parallel_mode`改为`DATA_PARALLEL`。

```python
from mindspore import context, Model
from mindspore.nn import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.context import ParallelMode
from resnet import resnet50

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=device_id) # set device_id

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

### 单机多卡训练

上述已将训练所需的脚本编辑好了，接下来通过命令调用对应的脚本。

目前MindSpore分布式执行采用单卡单进程运行方式，即每张卡上运行1个进程，进程数量与使用的卡的数量一致。其中，0卡在前台执行，其他卡放在后台执行。每个进程创建1个目录，用来保存日志信息以及算子编译信息。下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

```bash
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_SIZE"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$2

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./resnet50_distributed_training.py ./resnet.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ./resnet50_distributed_training.py > train.log$i 2>&1 &
    cd ../
done
rm -rf device0
mkdir device0
cp ./resnet50_distributed_training.py ./resnet.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
pytest -s -v ./resnet50_distributed_training.py > train.log0 2>&1
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
```

脚本需要传入变量`DATA_PATH`和`RANK_SIZE`，分别表示数据集的绝对路径和卡的数量。

分布式相关的环境变量有，

- `RANK_TABLE_FILE`：组网信息文件的路径。
- `DEVICE_ID`：当前卡在机器上的实际序号。
- `RANK_ID`：当前卡的逻辑序号。

其余环境变量请参考[安装教程](https://www.mindspore.cn/install)中的配置项。

运行时间大约在5分钟内，主要时间是用于算子的编译，实际训练时间在20秒内。用户可以通过`ps -ef | grep pytest`来监控任务进程。

日志文件保存到`rank`所对应的`device0`、 `device1`等目录下，`env.log`中记录了环境变量的相关信息，关于Loss部分结果保存在`train.log`中，示例如下：

```text
epoch: 1 step: 156, loss is 2.0084016
epoch: 2 step: 156, loss is 1.6407638
epoch: 3 step: 156, loss is 1.6164391
epoch: 4 step: 156, loss is 1.6838071
epoch: 5 step: 156, loss is 1.6320667
epoch: 6 step: 156, loss is 1.3098773
epoch: 7 step: 156, loss is 1.3515002
epoch: 8 step: 156, loss is 1.2943741
epoch: 9 step: 156, loss is 1.2316195
epoch: 10 step: 156, loss is 1.1533381
```

### 多机多卡训练

前面的章节，对MindSpore的分布式训练进行了介绍，都是基于单机多卡的Ascend环境，使用多机进行分布式训练，可以更大地提升训练速度。
在Ascend环境下，跨机器的NPU单元的通信与单机内各个NPU单元的通信一样，依旧是通过HCCL进行通信，区别在于，单机内的NPU单元天然的是互通的，而跨机器的则需要保证两台机器的网络是互通的。确认的方法如下：

在1号服务器执行下述命令，会为每个设备配置2号服务器对应设备的`devier ip`。例如将1号服务器卡0的目标IP配置为2号服务器的卡0的ip。配置命令需要使用`hccn_tool`工具。`hccn_tool`是一个[HCCL的工具](https://support.huawei.com/enterprise/zh/ascend-computing/a300t-9000-pid-250702906?category=developer-documents)，由CANN包自带。

```bash
hccn_tool -i 0 -netdetect -s address 192.98.92.131
hccn_tool -i 1 -netdetect -s address 192.98.93.131
hccn_tool -i 2 -netdetect -s address 192.98.94.131
hccn_tool -i 3 -netdetect -s address 192.98.95.131
hccn_tool -i 4 -netdetect -s address 192.98.92.141
hccn_tool -i 5 -netdetect -s address 192.98.93.141
hccn_tool -i 6 -netdetect -s address 192.98.94.141
hccn_tool -i 7 -netdetect -s address 192.98.95.141
```

`-i 0`指定设备ID。`-netdetect`指定网络检测对象IP属性。`-s address`表示设置属性为IP地址。`192.98.92.131`表示2号服务器的设备0的ip地址。接口命令可以[参考此处](https://support.huawei.com/enterprise/zh/doc/EDOC1100221413/c715a52a)。

在1号服务器上面执行完上述命令后，通过下述命令开始检测网络链接状态。在此使用`hccn_tool`的另一个功能，此功能的含义可以[参考此处](https://support.huawei.com/enterprise/zh/doc/EDOC1100221413/7620ee2)。

```bash
hccn_tool -i 0 -net_health -g
hccn_tool -i 1 -net_health -g
hccn_tool -i 2 -net_health -g
hccn_tool -i 3 -net_health -g
hccn_tool -i 4 -net_health -g
hccn_tool -i 5 -net_health -g
hccn_tool -i 6 -net_health -g
hccn_tool -i 7 -net_health -g
```

如果连接正常，对应的输出如下：

```bash
net health status: Success
```

如果连接失败，对应的输出如下：

```bash
net health status: Fault
```

在确认了机器之间的NPU单元的网络是通畅后，配置多机的json配置文件，本教程以16卡的配置文件为例，详细的配置文件说明可以参照本教程单机多卡部分的介绍。需要注意的是，在多机的json文件配置中，要求rank_id的排序，与server_id的字典序一致。

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        },
        {
            "server_id": "10.155.111.141",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.8","rank_id": "8"},
                {"device_id": "1","device_ip": "192.2.27.8","rank_id": "9"},
                {"device_id": "2","device_ip": "192.3.27.8","rank_id": "10"},
                {"device_id": "3","device_ip": "192.4.27.8","rank_id": "11"},
                {"device_id": "4","device_ip": "192.1.27.9","rank_id": "12"},
                {"device_id": "5","device_ip": "192.2.27.9","rank_id": "13"},
                {"device_id": "6","device_ip": "192.3.27.9","rank_id": "14"},
                {"device_id": "7","device_ip": "192.4.27.9","rank_id": "15"}],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

准备好配置文件后，可以进行分布式多机训练脚本的组织，在以2机16卡为例，两台机器上编写的脚本与单机多卡的运行脚本类似，区别在于指定不同的rank_id变量。

```bash
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_cluster.sh DATA_PATH RANK_TABLE_FILE RANK_SIZE RANK_START"
echo "For example: bash run_cluster.sh /path/dataset /path/rank_table.json 16 0"
echo "It is better to use the absolute path."
echo "The time interval between multiple machines to execute the script should not exceed 120s"
echo "=============================================================================================================="

execute_path=$(pwd)
echo ${execute_path}
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
echo ${self_path}

export DATA_PATH=$1
export RANK_TABLE_FILE=$2
export RANK_SIZE=$3
RANK_START=$4
DEVICE_START=0
for((i=0;i<=7;i++));
do
  export RANK_ID=$[i+RANK_START]
  export DEVICE_ID=$[i+DEVICE_START]
  rm -rf ${execute_path}/device_$RANK_ID
  mkdir ${execute_path}/device_$RANK_ID
  cd ${execute_path}/device_$RANK_ID || exit
  pytest -s ${self_path}/resnet50_distributed_training.py >train$RANK_ID.log 2>&1 &
done
```

执行时，两台机器分别执行如下命令，其中rank_table.json按照本章节展示的16卡的分布式json文件参考配置。

```bash
# server0
bash run_cluster.sh /path/dataset /path/rank_table.json 16 0
# server1
bash run_cluster.sh /path/dataset /path/rank_table.json 16 8
```

### 非下沉场景训练方式

图模式下，用户可以通过设置环境变量[GRAPH_OP_RUN](https://www.mindspore.cn/docs/zh-CN/master/note/env_var_list.html)=1来指定以非下沉方式训练模型。该方式需要采用OpenMPI的mpirun进行分布式训练，并且需要设置环境变量HCCL_WHITELIST_DISABLE=1。除此之外，训练启动脚本和[GPU分布式训练](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_gpu.html#定义网络)一致。

## 分布式训练模型参数保存和加载

在MindSpore中，支持四种分布式并行训练模式，即自动并行模式（Auto Parallel）、数据并行模式（Data Parallel）、半自动并行模式（Semi Auto Parallel）、手动混合并行模式（Hybrid Parallel），下面分别介绍四种分布式并行训练模式下模型的保存和加载。分布式训练进行模型参数的保存之前，需要先按照本教程配置分布式环境变量和集合通信库。

### 自动并行模式

自动并行模式（Auto Parallel）下模型参数的保存和加载与单卡用法基本相同，只需在本教程训练网络步骤中的`test_train_cifar`方法中添加配置`CheckpointConfig`和`ModelCheckpoint`，即可实现模型参数的保存。需要注意的是，并行模式下需要对每张卡上运行的脚本指定不同的checkpoint保存路径，防止读写文件时发生冲突，具体代码如下：

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

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
    ckpt_config = CheckpointConfig()
    ckpt_callback = ModelCheckpoint(prefix='auto_parallel', directory="./ckpt_" + str(get_rank()) + "/", config=ckpt_config)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb, ckpt_callback], dataset_sink_mode=True)
```

保存好checkpoint文件后，用户可以很容易加载模型参数进行推理或再训练场景，如用于再训练场景可使用如下代码加载模型：

```python
from mindspore import load_checkpoint, load_param_into_net

net = resnet50(batch_size=32, num_classes=10)
# The parameter for load_checkpoint is a .ckpt file which has been successfully saved
param_dict = load_checkpoint(pretrain_ckpt_path)
load_param_into_net(net, param_dict)
```

详细的checkpoint配置策略和保存加载方法可以参考[模型参数的保存和加载](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/save_load.html#checkpoint)。

对于网络中切分的参数框架默认会自动聚合保存到模型文件，但考虑到在超大模型场景下，单个完整的模型文件过大会带来传输慢、难加载等问题，所以用户可以通过`CheckpointConfig`中`integrated_save`参数选择非合并保存，即每张卡保存各自卡上的参数切片。如果再训练或推理的切分策略或集群规模与训练不一致，需要采用特殊的加载方式。

针对采用多卡再训练及微调的场景，用户可以使用`model.infer_train_layout`函数推导再训练分布式策略（该函数当前仅支持数据集下沉模式），再传递给`load_distributed_checkpoint`函数中`predict_strategy`参数，该函数从所有分片的模型文件中加载需要的部分进行合并切分操作，将参数从`strategy_ckpt_load_file`（训练策略）恢复到`predict_strategy`（再训练策略），最后加载到`model.train_network`中。若采用单卡进行再训练及微调，可以直接将`predict_strategy`置为`None`。参考代码如下：

```python
from mindspore import load_distributed_checkpoint, context
from mindspore.communication import init

context.set_context(mode=context.GRAPH_MODE)
init()
context.set_auto_parallel_context(full_batch=True, parallel_mode='semi_auto_parallel', strategy_ckpt_load_file='./train_strategy.ckpt')
# create model and dataset
dataset = create_custom_dataset()
resnet = ResNet50()
opt = Momentum()
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
# infer train strategy
layout_dict = model.infer_train_layout(dataset, True, 100)
# load into `model.train_network` net
ckpt_file_list = create_ckpt_file_list()
load_distributed_checkpoint(model.train_network, ckpt_file_list, layout_dict)
# training the model
model.train(2, dataset)
```

> 分布式推理场景可以参考教程：[分布式推理](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/ascend_910_mindir.html#概述)。

### 数据并行模式

数据并行模式（Data Parallel）下checkpoint的使用方法和自动并行模式（Auto Parallel）一样，只需要将`test_train_cifar`中

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
```

修改为:

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
```

> 数据并行场景下加载模型参数时建议每卡加载相同的checkpoint文件，避免造成计算误差，或者可以打开`parameter_broadcast`开关将0号卡的参数广播到其他卡上。

### 半自动并行模式

半自动并行模式（Semi Auto Parallel）下checkpoint使用方法，与自动并行模式（Auto Parallel）和数据并行模式（Data Parallel）的用法相同，不同之处在于网络的定义，半自动并行模式（Semi Auto Parallel）下网络模型的定义请参考本教程中定义网络部分的[半自动并行模式](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html#半自动并行模式)。

保存模型时，可以使用如下代码来实现：

```python
...
net = SemiAutoParallelNet()
...
ckpt_config = CheckpointConfig()
ckpt_callback = ModelCheckpoint(prefix='semi_auto_parallel', directory="./ckpt_" + str(get_rank()) + "/", config=ckpt_config)
```

加载模型时，可以使用如下代码来实现：

```python
net = SemiAutoParallelNet()
# The parameter for load_checkpoint is a .ckpt file which has been successfully saved
param_dict = load_checkpoint(pretrain_ckpt_path)
load_param_into_net(net, param_dict)
```

以上介绍的三种并行训练模式，checkpoint文件的保存方式都是每张卡上均保存完整的checkpoint文件，在以上三种并行训练模式上，用户还可以选择每张卡上只保存本卡的checkpoint文件，以半自动并行模式（Semi Auto Parallel）为例，进行说明。

只需要改动设置checkpoint保存策略的代码，将`CheckpointConfig`中的`integrated_save`参数设置为Fasle，便可实现每张卡上只保存本卡的checkpoint文件，具体改动如下：

将checkpoint配置策略由：

```python
# config checkpoint
ckpt_config = CheckpointConfig(keep_checkpoint_max=1)
```

改为：

```python
# config checkpoint
ckpt_config = CheckpointConfig(keep_checkpoint_max=1, integrated_save=False)
```

需要注意的是，如果用户选择了这种checkpoint保存方式，那么就需要用户自己对切分的checkpoint进行保存和加载，以便进行后续的推理或再训练。具体用法可参考[对保存的CheckPoint文件做合并处理](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/save_load.html#checkpoint)。

### 手动混合并行模式

手动混合并行模式（Hybrid Parallel）的模型参数保存和加载请参考[保存和加载模型（HyBrid Parallel模式）](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/save_load.html)。
