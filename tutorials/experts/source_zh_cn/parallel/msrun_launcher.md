# msrun启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_zh_cn/parallel/msrun_launcher.md)

## 概述

`msrun`是[动态组网](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/dynamic_cluster.html)启动方式的封装，用户可使用`msrun`以单个命令行指令的方式在各节点拉起多进程分布式任务，并且无需手动设置[动态组网环境变量](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/dynamic_cluster.html)。`msrun`同时支持`Ascend`，`GPU`和`CPU`后端。与`动态组网`启动方式一样，`msrun`无需依赖第三方库以及配置文件。

> - `msrun`在用户安装MindSpore后即可使用，可使用指令`msrun --help`查看支持参数。
> - `msrun`支持`图模式`以及`PyNative模式`。

命令行参数列表：

<table align="center">
    <tr>
        <th align="left">参数</th>
        <th align="left">功能</th>
        <th align="left">类型</th>
        <th align="left">取值</th>
        <th align="left">说明</th>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--worker_num</td>
        <td align="left">参与分布式任务的Worker进程总数。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">大于0的整数。默认值为8。</td>
        <td align="left">每个节点上启动的Worker总数应当等于此参数：<br>若总数大于此参数，多余的Worker进程会注册失败；<br>若总数小于此参数，集群会在等待一段超时时间后，<br>提示任务拉起失败并退出，<br>超时时间窗大小可通过参数<code>cluster_time_out</code>配置。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--local_worker_num</td>
        <td align="left">当前节点上拉起的Worker进程数。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">大于0的整数。默认值为8。</td>
        <td align="left">当此参数与<code>worker_num</code>保持一致时，代表所有Worker进程在本地执行，<br>此场景下<code>node_rank</code>值会被忽略。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--master_addr</td>
        <td align="left">指定Scheduler的IP地址。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">合法的IP地址。默认为127.0.0.1。</td>
        <td align="left">msrun会自动检测在哪个节点拉起Scheduler进程，用户无需关心。<br>若无法查找到对应的地址，训练任务会拉起失败。<br>当前版本暂不支持IPv6地址。<br>当前版本msrun使用<code>ip -j addr</code>指令查询当前节点地址，<br>需要用户环境支持此指令。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--master_port</td>
        <td align="left">指定Scheduler绑定端口号。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">1024～65535范围内的端口号。默认为8118。</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--node_rank</td>
        <td align="left">当前节点的索引。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">大于0的整数。默认值为-1。</td>
        <td align="left">单机多卡场景下，此参数会被忽略。<br>多机多卡场景下，<br>若不设置此参数，Worker进程的rank_id会被自动分配；<br>若设置，则会按照索引为各节点上的Worker进程分配rank_id。<br>若每个节点Worker进程数量不同，建议不配置此参数，<br>以自动分配rank_id。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--log_dir</td>
        <td align="left">Worker以及Scheduler日志输出路径。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">文件夹路径。默认为当前目录。</td>
        <td align="left">若路径不存在，msrun会递归创建文件夹。<br>日志格式如下：对于Scheduler进程，日志名为<code>scheduler.log</code>；<br>对于Worker进程，日志名为<code>worker_[rank].log</code>，<br>其中<code>rank</code>后缀与分配给Worker的<code>rank_id</code>一致，<br>但在未设置<code>node_rank</code>且多机多卡场景下，它们可能不一致。<br>建议执行<code>grep -rn "Global rank id"</code>指令查看各Worker的<code>rank_id</code>。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--join</td>
        <td align="left">msrun是否等待Worker以及Scheduler退出。</td>
        <td align="left" style="white-space:nowrap">Bool</td>
        <td align="left">True或者False。默认为False。</td>
        <td align="left">若设置为False，msrun在拉起进程后会立刻退出，<br>查看日志确认分布式任务是否正常执行。<br>若设置为True，msrun会等待所有进程退出后，收集异常日志并退出。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--cluster_time_out</td>
        <td align="left">集群组网超时时间，单位为秒。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">整数。默认为600秒。</td>
        <td align="left">此参数代表在集群组网的等待时间。<br>若超出此时间窗口依然没有<code>worker_num</code>数量的Worker注册成功，则任务拉起失败。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">task_script</td>
        <td align="left">用户Python脚本。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">合法的脚本路径。</td>
        <td align="left">一般情况下，此参数为python脚本路径，msrun会默认以<code>python task_script task_script_args</code>方式拉起进程。<br>msrun还支持此参数为pytest，此场景下任务脚本及任务参数在参数<code>task_script_args</code>传递。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">task_script_args</td>
        <td align="left">用户Python脚本的参数。</td>
        <td align="left"></td>
        <td align="left">参数列表。</td>
        <td align="left">例如：<code>msrun --worker_num=8 --local_worker_num=8 train.py <b>--device_target=Ascend --dataset_path=/path/to/dataset</b></code></td>
    </tr>
</table>

## 环境变量

下表是用户脚本中能够使用的环境变量，它们由`msrun`设置：

<table align="center">
    <tr>
        <th align="left">环境变量</th>
        <th align="left">功能</th>
        <th align="left">取值</th>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_ROLE</td>
        <td align="left">本进程角色。</td>
        <td align="left">
            当前版本<code>msrun</code>导出下面两个值：
            <ul>
                <li>MS_SCHED：代表Scheduler进程。</li>
                <li>MS_WORKER：代表Worker进程。</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_SCHED_HOST</td>
        <td align="left">用户指定的Scheduler的IP地址。</td>
        <td align="left">与参数<code>--master_addr</code>相同。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_SCHED_PORT</td>
        <td align="left">用户指定的Scheduler绑定端口号。</td>
        <td align="left">与参数<code>--master_port</code>相同。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_WORKER_NUM</td>
        <td align="left">用户指定的Worker进程总数。</td>
        <td align="left">与参数<code>--worker_num</code>相同。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_CLUSTER_TIMEOUT</td>
        <td align="left">集群组网超时时间。</td>
        <td align="left">与参数<code>--cluster_time_out</code>相同。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">RANK_SIZE</td>
        <td align="left">用户指定的Worker进程总数。</td>
        <td align="left">与参数<code>--worker_num</code>相同。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">RANK_ID</td>
        <td align="left">为Worker进程分配的rank_id。</td>
        <td align="left">多机多卡场景下，若没有设置<code>--node_rank</code>参数，<code>RANK_ID</code>只会在集群初始化后被导出。<br>因此要使用此环境变量，建议正确设置<code>--node_rank</code>参数。</td>
    </tr>
</table>

## 操作实践

启动脚本在各硬件平台下一致，下面以Ascend为例演示如何编写启动脚本：

> 您可以在这里下载完整的样例代码：[startup_method](https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/startup_method)。

目录结构如下：

```text
└─ sample_code
    ├─ startup_method
       ├── msrun_1.sh
       ├── msrun_2.sh
       ├── msrun_single.sh
       ├── net.py
    ...
```

其中，`net.py`是定义网络结构和训练过程，`msrun_single.sh`是以`msrun`启动的单机多卡执行脚本；`msrun_1.sh`和`msrun_2.sh`是以`msrun`启动的多机多卡执行脚本，分别在不同节点上执行。

### 1. 准备Python训练脚本

这里以数据并行为例，训练一个MNIST数据集的识别网络。

首先指定运行模式、硬件设备等，与单卡脚本不同，并行脚本还需指定并行模式等配置项，并通过`init()`初始化HCCL、NCCL或MCCL通信域。此处不设置`device_target`会自动指定为MindSpore包对应的后端硬件设备。

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

### 2. 准备启动脚本

> 对于msrun来说单机多卡和多机多卡执行指令类似，单机多卡只需将参数`worker_num`和`local_worker_num`保持相同即可，且单机多卡场景下无需设置`master_addr`，默认为`127.0.0.1`。

#### 单机多卡

下面以执行单机8卡训练为例：

脚本[msrun_single.sh](https://gitee.com/mindspore/docs/blob/r2.3/docs/sample_code/startup_method/msrun_single.sh)使用msrun指令在当前节点拉起1个`Scheduler`进程以及8个`Worker`进程（无需设置`master_addr`，默认为`127.0.0.1`；单机无需设置`node_rank`）：

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf msrun_log
mkdir msrun_log
echo "start training"

msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 net.py
```

执行指令：

```bash
bash msrun_single.sh
```

即可执行单机8卡分布式训练任务，日志文件会保存到`./msrun_log`目录下，结果保存在`./msrun_log/worker_*.log`中，Loss结果如下：

```text
epoch: 0, step: 0, loss is 2.3499548
epoch: 0, step: 10, loss is 1.6682479
epoch: 0, step: 20, loss is 1.4237018
epoch: 0, step: 30, loss is 1.0437132
epoch: 0, step: 40, loss is 1.0643986
epoch: 0, step: 50, loss is 1.1021575
epoch: 0, step: 60, loss is 0.8510884
epoch: 0, step: 70, loss is 1.0581372
epoch: 0, step: 80, loss is 1.0076828
epoch: 0, step: 90, loss is 0.88950706
...
```

#### 多机多卡

下面以执行2机8卡训练，每台机器执行启动4个Worker为例：

脚本[msrun_1.sh](https://gitee.com/mindspore/docs/blob/r2.3/docs/sample_code/startup_method/msrun_1.sh)在节点1上执行，使用msrun指令拉起1个`Scheduler`进程以及4个`Worker`进程，配置`master_addr`为节点1的IP地址（msrun会自动检测到当前节点IP与`master_addr`匹配而拉起`Scheduler`进程），通过`node_rank`设置当前节点为0号节点：

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf msrun_log
mkdir msrun_log
echo "start training"

msrun --worker_num=8 --local_worker_num=4 --master_addr=<node_1 ip address> --master_port=8118 --node_rank=0 --log_dir=msrun_log --join=True --cluster_time_out=300 net.py
```

脚本[msrun_2.sh](https://gitee.com/mindspore/docs/blob/r2.3/docs/sample_code/startup_method/msrun_2.sh)在节点2上执行，使用msrun指令拉起4个`Worker`进程，配置`master_addr`为节点1的IP地址，通过`node_rank`设置当前节点为1号节点：

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf msrun_log
mkdir msrun_log
echo "start training"

msrun --worker_num=8 --local_worker_num=4 --master_addr=<node_1 ip address> --master_port=8118 --node_rank=1 --log_dir=msrun_log --join=True --cluster_time_out=300 net.py
```

> 节点2和节点1的指令差别在于`node_rank`不同。

在节点1执行：

```bash
bash msrun_1.sh
```

在节点2执行：

```bash
bash msrun_2.sh
```

即可执行2机8卡分布式训练任务，日志文件会保存到`./msrun_log`目录下，结果保存在`./msrun_log/worker_*.log`中，Loss结果如下：

```text
epoch: 0, step: 0, loss is 2.3499548
epoch: 0, step: 10, loss is 1.6682479
epoch: 0, step: 20, loss is 1.4237018
epoch: 0, step: 30, loss is 1.0437132
epoch: 0, step: 40, loss is 1.0643986
epoch: 0, step: 50, loss is 1.1021575
epoch: 0, step: 60, loss is 0.8510884
epoch: 0, step: 70, loss is 1.0581372
epoch: 0, step: 80, loss is 1.0076828
epoch: 0, step: 90, loss is 0.88950706
...
```
