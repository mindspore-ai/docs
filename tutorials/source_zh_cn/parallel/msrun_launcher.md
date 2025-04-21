# msrun启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/parallel/msrun_launcher.md)

## 概述

`msrun`是[动态组网](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/dynamic_cluster.html)启动方式的封装，用户可使用`msrun`，以单个命令行指令的方式在各节点拉起多进程分布式任务，并且无需手动设置[动态组网环境变量](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/dynamic_cluster.html)。`msrun`同时支持`Ascend`，`GPU`和`CPU`后端。与`动态组网`启动方式一样，`msrun`无需依赖第三方库以及配置文件。

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
        <td align="left">所有节点上启动的Worker总数应当等于此参数：<br>若总数大于此参数，多余的Worker进程会注册失败；<br>若总数小于此参数，集群会在等待一段超时时间后，<br>提示任务拉起失败并退出，<br>超时时间窗大小可通过参数<code>cluster_time_out</code>配置。</td>
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
        <td align="left">指定Scheduler的IP地址或者主机名。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">合法的IP地址或者主机名。默认为IP地址127.0.0.1。</td>
        <td align="left">msrun会自动检测在哪个节点拉起Scheduler进程，用户无需关心。<br>若无法查找到对应的地址或主机名无法被DNS解析，训练任务会拉起失败。<br>当前版本暂不支持IPv6地址。<br>若传入主机名时，msrun会自动将其解析为IP地址，需要用户环境支持DNS服务。</td>
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
        <td align="left">可传入大于等于0的整数。在不传入值的情况下，默认值为-1。</td>
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
        <td align="left">默认为600秒。</td>
        <td align="left">此参数代表在集群组网的等待时间。<br>若超出此时间窗口依然没有<code>worker_num</code>数量的Worker注册成功，则任务拉起失败。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--bind_core</td>
        <td align="left">开启进程绑核。</td>
        <td align="left" style="white-space:nowrap">Bool</td>
        <td align="left">True或者False。默认为False。</td>
        <td align="left">若用户配置此参数，msrun会平均分配CPU核，将其绑定到拉起的分布式进程上。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--sim_level</td>
        <td align="left">设置模拟编译等级。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为-1，即关闭模拟编译功能。</td>
        <td align="left">若用户配置此参数，msrun只会拉起进程的模拟编译，不做算子执行。<br>此功能通常用于调试大规模分布式训练并行策略，在编译阶段提前发现内存和策略问题。<br>模拟编译等级的设置可参考文档：<a href="https://www.mindspore.cn/tutorials/zh-CN/br_base/debug/dryrun.html">DryRun</a>。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--sim_rank_id</td>
        <td align="left">单卡模拟编译的rank_id。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为-1，即关闭单进程的模拟编译功能。</td>
        <td align="left">设置单卡模拟编译进程的rank_id。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--rank_table_file</td>
        <td align="left">rank_table配置文件，只在昇腾平台下有效。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">rank_table配置文件路径，默认为空。</td>
        <td align="left">此参数代表昇腾平台下的rank_table配置文件，描述当前分布式集群。<br>由于rank_table配置文件反映的是物理层面分布式集群信息，在使用该配置时，<br>请确保对于当前进程可见的Device与rank_table配置保持一致。<br>可通过环境变量<code>ASCEND_RT_VISIBLE_DEVICES</code>设置对于当前进程可见的Device。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--worker_log_name</td>
        <td align="left">设置worker日志名。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">worker日志文件名，默认为<code>worker_[rank].log</code>。</td>
        <td align="left">此参数代表支持用户配置worker日志名，并且支持分别通过<code>{ip}</code>和<code>{hostname}</code><br>在worker日志名中配置<code>ip</code>和<code>hostname</code>。<br>worker日志名的后缀默认为<code>rank</code>。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--tail_worker_log</td>
        <td align="left">输出worker日志到控制台。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">一个或多个与worker进程rank_id关联的整数。默认为-1。</td>
        <td align="left">此参数代表<code>--join=True</code>情况下，默认输出当前节点所有worker日志，<br>并且支持用户指定一个或多个卡的worker日志输出到控制台。<br>这个参数需要在[0, local_worker_num]范围内。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">task_script</td>
        <td align="left">用户Python脚本。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">合法的脚本路径。</td>
        <td align="left">一般情况下，此参数为python脚本路径，<br>msrun会默认以<code>python task_script task_script_args</code>方式拉起进程。<br>msrun还支持此参数为pytest，此场景下任务脚本及任务参数<br>在参数<code>task_script_args</code>传递。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">task_script_args</td>
        <td align="left">用户Python脚本的参数。</td>
        <td align="left"></td>
        <td align="left">参数列表。</td>
        <td align="left">例如：<code>msrun --worker_num=8 --local_worker_num=8 train.py <br><b>--device_target=Ascend --dataset_path=/path/to/dataset</b></code></td>
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
        <td align="left" style="white-space:nowrap">MS_TOPO_TIMEOUT</td>
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

msrun作为动态组网启动方式的封装，所有用户可自定义配置的环境变量可参考[动态组网环境变量](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/dynamic_cluster.html)。

## 启动分布式任务

启动脚本在各硬件平台下一致，下面以Ascend为例演示如何编写启动脚本：

> 您可以在这里下载完整的样例代码：[startup_method](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/startup_method)。

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

首先指定运行模式、硬件设备等，与单卡脚本不同，并行脚本还需指定并行模式等配置项，并通过`init()`初始化HCCL、NCCL或MCCL通信域。此处未设置`device_target`，会自动指定为MindSpore包对应的后端硬件设备。

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

脚本[msrun_single.sh](https://gitee.com/mindspore/docs/blob/br_base/docs/sample_code/startup_method/msrun_single.sh)使用msrun指令在当前节点拉起1个`Scheduler`进程以及8个`Worker`进程（无需设置`master_addr`，默认为`127.0.0.1`；单机无需设置`node_rank`）：

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
...
```

#### 多机多卡

下面以执行2机8卡训练，每台机器执行启动4个Worker为例：

脚本[msrun_1.sh](https://gitee.com/mindspore/docs/blob/br_base/docs/sample_code/startup_method/msrun_1.sh)在节点1上执行，使用msrun指令拉起1个`Scheduler`进程以及4个`Worker`进程，配置`master_addr`为节点1的IP地址（msrun会自动检测到当前节点IP与`master_addr`匹配而拉起`Scheduler`进程），通过`node_rank`设置当前节点为0号节点：

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

脚本[msrun_2.sh](https://gitee.com/mindspore/docs/blob/br_base/docs/sample_code/startup_method/msrun_2.sh)在节点2上执行，使用msrun指令拉起4个`Worker`进程，配置`master_addr`为节点1的IP地址，通过`node_rank`设置当前节点为1号节点：

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

## 多卡并行调试

在分布式环境中可以使用Python内置的调试器（pdb）来进行多卡并行的调试，通过对所有或者某一rank进行断点和同步操作来实现。在`msrun`参数设置为`--join=True`拉起worker进程后，所有worker进程的标准输入从`msrun`主进程继承，且标准输出通过`msrun`日志重定向功能输出到shell窗口。以下会给出如何在分布式环境下使用pdb的操作细节：

### 1. 启动pdb调试器

用户可以通过多种方式来启动pdb调试器，比如在Python训练脚本中插入`import pdb; pdb.set_trace()`或者`breakpoint()`来进行断点操作。

#### Python训练脚本

```python
import pdb
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
pdb.set_trace()
ms.set_seed(1)
```

#### 启动脚本

在启动脚本中，`msrun`的参数需要设置为`--join=True`来保证通过标准输入传递pdb命令，且通过标准输出显示调试情况。

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 net.py
```

### 2. 针对rank进行调试

在分布式环境中，用户可能需要针对某一rank进行调试，这可以通过在训练脚本中对特定的rank进行断点操作实现。比如在单机八卡任务中，仅针对rank 7进行断点调试：

```python
import pdb
import mindspore as ms
from mindspore.communication import init, get_rank

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
if get_rank() == 7:
    pdb.set_trace()
ms.set_seed(1)
```

> `mindspore.communication.get_rank()`接口需要在调用`mindspore.communication.init()`接口完成分布式初始化后才能正常获取rank信息，否则`get_rank()`默认返回0。

在对某一rank进行断点操作之后，会导致该rank进程执行停止在断点处等待后续交互操作，而其他未断点rank进程会继续运行，这样可能会导致快慢卡的情况，所以可以使用`mindspore.communication.comm_func.barrier()`算子和`mindspore.common.api._pynative_executor.sync()`来同步所有rank的运行，确保其他rank阻塞等待，且一旦调试的rank继续运行则其他rank的停止会被释放。比如在单机八卡任务中，仅针对rank 7进行断点调试且阻塞所有其他rank：

```python
import pdb
import mindspore as ms
from mindspore.communication import init, get_rank
from mindspore.communication.comm_func import barrier
from mindspore.common.api import _pynative_executor

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
if get_rank() == 7:
    pdb.set_trace()
barrier()
_pynative_executor.sync()
ms.set_seed(1)
```

### 3. shell终端的标准输入和标准输出

`msrun`支持通过`--tail_worker_log`将特定的worker日志输出到shell的标准输出，为了使标准输出更利于观察，推荐使用此参数来指定输出需要断点调试的rank。比如在单机八卡任务中，仅针对rank 7进行断点调试：

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 --tail_worker_log=7 net.py
```

> - `msrun`不使用`--tail_worker_log`参数的默认行为会把本节点所有worker的日志输出到shell的标准输出。
> - 在同时调试多个rank时，一个pdb的指令会依次通过标准输入传递到一个rank上。

### 4. 常用pdb调试命令

- `n` (next)：执行当前行代码，跳到下一行代码。
- `s` (step)：进入当前行代码调用的函数，逐步调试。
- `c` (continue)：继续执行程序，直到下一个断点。
- `q` (quit)：退出调试器并终止程序执行。
- `p` (print)：打印变量的值。例如，`p variable`会显示变量`variable`的当前值。
- `l` (list)：显示当前代码的上下文。
- `b` (break)：设置断点，可以指定行号或函数名。
- `h` (help)：显示帮助信息，列出所有可用命令。
