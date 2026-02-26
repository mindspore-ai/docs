# msrun启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_zh_cn/parallel/msrun_launcher.md)

## 概述

`msrun`是[动态组网](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dynamic_cluster.html)启动方式的封装，用户可使用`msrun`，以单个命令行指令的方式在各节点拉起多进程分布式任务，并且无需手动设置[动态组网环境变量](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dynamic_cluster.html)。`msrun`同时支持`Ascend`、`GPU`和`CPU`后端。与`动态组网`启动方式一样，`msrun`无需依赖第三方库以及配置文件。

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
        <td align="left" style="white-space:nowrap">Bool/Dict</td>
        <td align="left">True、False或者给指定设备分配CPU范围段的字典。默认为False。</td>
        <td align="left">若设置为True，则会基于环境信息按照设备亲和去自动分配CPU范围段；若手动传入一个字典，则根据该字典分配的CPU范围段去绑核。具体配置可参考“进程级 CPU/NUMA 亲和性配置”章节。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--bind_numa</td>
        <td align="left">开启进程绑 NUMA 节点。</td>
        <td align="left" style="white-space:nowrap">Bool/Dict/String</td>
        <td align="left">True、False或者给指定设备分配 NUMA 节点的字典，也支持传入以.json结尾的文件路径。默认为False。</td>
        <td align="left">若设置为True，则会基于环境信息按照设备亲和去自动分配 NUMA 节点；若手动传入一个字典或者JSON文件，则根据传入的配置自定义去绑定 NUMA 节点。具体配置可参考“进程级 CPU/NUMA 亲和性配置”章节。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--sim_level</td>
        <td align="left">设置模拟编译等级。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为-1，即关闭模拟编译功能。</td>
        <td align="left">若用户配置此参数，msrun只会拉起进程的模拟编译，不做算子执行。<br>此功能通常用于调试大规模分布式训练并行策略，在编译阶段提前发现内存和策略问题。<br>模拟编译等级的设置可参考文档：<a href="https://www.mindspore.cn/tutorials/zh-CN/master/debug/dryrun.html">DryRun</a>。</td>
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

msrun作为动态组网启动方式的封装，所有用户可自定义配置的环境变量可参考[动态组网环境变量](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dynamic_cluster.html)。

## 启动分布式任务

启动脚本在各硬件平台下一致，下面以Ascend为例演示如何编写启动脚本：

> 您可以在这里下载完整的样例代码：[startup_method](https://atomgit.com/mindspore/docs/tree/master/docs/sample_code/startup_method)。

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

脚本[msrun_single.sh](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_single.sh)使用msrun指令在当前节点拉起1个`Scheduler`进程以及8个`Worker`进程（无需设置`master_addr`，默认为`127.0.0.1`；单机无需设置`node_rank`）：

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

脚本[msrun_1.sh](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_1.sh)在节点1上执行，使用msrun指令拉起1个`Scheduler`进程以及4个`Worker`进程，配置`master_addr`为节点1的IP地址（msrun会自动检测到当前节点IP与`master_addr`匹配而拉起`Scheduler`进程），通过`node_rank`设置当前节点为0号节点：

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

脚本[msrun_2.sh](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_2.sh)在节点2上执行，使用msrun指令拉起4个`Worker`进程，配置`master_addr`为节点1的IP地址，通过`node_rank`设置当前节点为1号节点：

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

init()
if get_rank() == 7:
    pdb.set_trace()
ms.set_seed(1)
```

> [mindspore.communication.get_rank()](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.get_rank.html)接口需要在调用[mindspore.communication.init()](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.init.html)接口完成分布式初始化后才能正常获取rank信息，否则`get_rank()`默认返回0。

在对某一rank进行断点操作之后，会导致该rank进程执行停止在断点处等待后续交互操作，而其他未断点rank进程会继续运行，这样可能会导致快慢卡的情况，所以可以使用[mindspore.communication.comm_func.barrier()](https://www.mindspore.cn/docs/zh-CN/master/api_python/communication/mindspore.communication.comm_func.barrier.html)算子和[mindspore.runtime.synchronize()](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.synchronize.html)来同步所有rank的运行，确保其他rank阻塞等待，且一旦调试的rank继续运行则其他rank的停止会被释放。比如在单机八卡任务中，仅针对rank 7进行断点调试且阻塞所有其他rank：

```python
import pdb
import mindspore as ms
from mindspore.communication import init, get_rank
from mindspore.communication.comm_func import barrier
from mindspore.runtime import synchronize

init()
if get_rank() == 7:
    pdb.set_trace()
barrier()
synchronize()
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

## 进程级 CPU/NUMA 亲和性配置

`msrun` 提供 `--bind_core` 和 `--bind_numa` 参数，分别通过调用 `taskset` 和 `numactl` 系统命令，在进程启动时限定其 CPU 核心运行范围、NUMA 节点绑定关系。两者均支持自动分配策略和用户自定义策略。

### --bind_core（CPU 亲和性配置）

核心调用`taskset -c CPUA-CPUB python XXX.py`，限制 Python 进程运行在 `CPUA` 到 `CPUB` 范围的 CPU 核心上。

#### 1. 自动绑核（--bind_core=True）

- **核心逻辑**：无需手动指定核编号，基于环境信息（CPU 资源、NUMA 节点、设备亲和性）自动分配 CPU 核：

    - 优先使用亲和池内的 CPU 核；若亲和池内 CPU 核不足，则使用非亲和池内的 CPU 核。
    - 依赖 `lscpu`、`npu-smi` 等命令获取硬件信息，命令执行失败时仅基于可用 CPU 资源分配；
    - CPU 与 NPU 间亲和关系的获取方式，与 MindSpore 接口 `mindspore.runtime.set_cpu_affinity` 一致，可参考 [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html)。

#### 2. 自定义绑核

- **格式要求**：传入 JSON 格式的字典，在 shell 环境中需用 `''` 包裹 `{}`。

- **参数规范**：

    - 字典的 `key` 支持 `scheduler`（调度进程）或 `deviceX`（设备进程，`X` 为设备编号）。
    - 字典的 `value` 为 CPU 核范围段列表（如 `["0-9", "20-29"]`）。空列表表示跳过该进程绑核。

- **示例**：

    ```bash
    --bind_core='{"scheduler":["0-9"], "device0":["10-19"], "device1":["20-29", "40-49"]}'
    ```

    含义：

    - 为`scheduler`进程分配 CPU 核 0-9；
    - 为 0 号 worker 进程（对应`device0`）分配 CPU 核 10-19；
    - 为 1 号 worker 进程（对应`device1`）分配 CPU 核 20-29 和 40-49。

- **注意事项**：

    1. 进程编号需与设备编号匹配。例如，若通过`ASCEND_RT_VISIBLE_DEVICES=6,7`配置，使 0 号进程对应`device6`、1 号进程对应`device7`，则需按如下方式配置，否则无法为对应进程绑核：

        ```bash
        --bind_core='{"scheduler":["0-9"], "device6":["10-19"], "device7":["20-29", "40-49"]}'
        ```

        scheduler 进程不占用设备资源，因此不参与设备排序，键的顺序不影响生效（如上述示例中`scheduler`与`device6`顺序可互换）。
    2. 若 CPU 范围段列表为空，则跳过对该进程的亲和性设置。例如：

        ```bash
        --bind_core='{"scheduler":[], "device0":[], "device1":["20-29", "40-49"]}'
        ```

        表示：跳过`scheduler`进程和 0 号 worker 进程的绑核，仅为 1 号 worker 进程（`device1`）分配 CPU 核。
    3. 建议 worker 进程数量与`--bind_core`字典的键值对数量一致。例如，单机两卡任务中，若仅需为 1 号 worker 进程绑核，需显式配置所有进程（包括不绑核的进程）：

        ```bash
        # 正确示例
        --bind_core='{"scheduler":[], "device0":[], "device1":["20-29", "40-49"]}'

        # 错误示例
        --bind_core='{"device1":["20-29", "40-49"]}'
        ```

        错误示例中，0 号 worker 进程可能被误判为对应`device1`而跳过绑核，`scheduler`和 1 号 worker 进程因未在配置中也会被跳过。

#### 3. 关闭绑核（--bind_core=False）

不启用进程级 CPU 亲和性设置，为默认配置。

### --bind_numa（NUMA 亲和性配置）

核心调用`numactl --membind NUMAX --cpunodebind NUMAX python XXX.py`，将 Python 进程的内存区域绑定在 NUMA 节点X上，并且限制进程运行在NUMA 节点X所对应的 CPU 核心上。

#### 1. 自动绑 NUMA（--bind_numa=True）

- **核心逻辑**：无需手动指定节点编号，基于环境信息自动分配 NUMA 节点：

    - 要求 NUMA 节点数量 ≥ 启动进程数量（保证每个进程独占一个节点），否则无法使能绑定 NUMA 功能；
    - 优先使用设备亲和的 NUMA 节点，多进程亲和同一节点时使用非亲和节点；
    - 依赖 lscpu、npu-smi 等命令获取硬件信息，命令执行失败时仅基于可用 NUMA 资源分配；
    - NUMA 与 NPU 间亲和关系的获取方式，与 `--bind_core` 以及 MindSpore 接口 `mindspore.runtime.set_cpu_affinity` 一致。

#### 2. 自定义绑NUMA

- **格式要求**：传入 JSON 格式的字典，在 shell 环境中需用 `''` 包裹 `{}`。

- **参数规范**：

    - 字典的 `key` 支持 `scheduler`（调度进程）或 `deviceX`（设备进程，`X` 为设备编号）。
    - 字典的 `value` 为 NUMA 节点列表，可以是单个或以`,`分割的正整数，也可以是范围段（如 `["0"，"1,2","3-4"]`。空列表表示跳过该进程绑 NUMA。

- **示例**：

    ```bash
    --bind_numa='{"scheduler":["0"], "device0":["1,2"], "device1":["3-4"]}'
    ```

    含义：

    - 为`scheduler`进程分配 NUMA 节点0；
    - 为 0 号 worker 进程（对应`device0`）分配 NUMA 节点1和 NUMA 节点2；
    - 为 1 号 worker 进程（对应`device1`）分配 NUMA 节点3和 NUMA 节点4。

- **注意事项**：

    `--bind_numa`可传入的自定义配置的字典格式规范与`--bind_core`保持一致。

#### 3. 关闭绑核（--bind_numa=False）

不启用进程级 NUMA 亲和性设置，为默认配置。

#### 4. JSON文件配置（--bind_numa=PATH_TO_JSON.json）

- **格式要求**：传入带有绑核/绑内存的 JSON 文件的绝对路径。

- **示例**：

    启动命令示例:

    ```bash
    msrun --bind_numa=<json>
    ```

    `<json>` 文件示例：

    ```json
    {
      "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa"},
      "bind_cpu": {"scheduler": {"main": "20-29"}, "device0": {"main": "0-9"}, "device1": {"main": "10-19"}},
      "bind_memory": {"scheduler": 2, "device0": 0, "device1": 1}
    }
    ```

    含义：

    - 为`scheduler`进程绑定 CPU 20-29，0 号 worker 进程（对应`device0`）绑定 CPU 0-9，1 号 worker 进程（对应`device1`）绑定 CPU 10-19。
    - 为`scheduler`进程绑定 NUMA 节点2，0 号 worker 进程（对应`device0`）绑定 NUMA 节点0，1 号 worker 进程（对应`device1`）绑定 NUMA 节点1。

- **注意事项**：

    详细的 JSON 配置及指导可以参考章节 `使用 JSON 统一配置 CPU/NUMA 亲和`。

### --bind_numa 与 --bind_core 配合使用

同时使用`--bind_numa`和`--bind_core`时，启动进程时会调用`numactl --membind NUMAX --physcpubind CPUA-CPUB`，即`--bind_numa`依据 NUMA 架构控制内存区域绑定，`--bind_core`以 CPU 核心的力度设置进程的 CPU 亲和性。

### 使用 JSON 统一配置 CPU/NUMA 亲和（--bind_numa / mindspore.runtime.set_cpu_affinity）

`msrun --bind_numa` 与 `mindspore.runtime.set_cpu_affinity` 支持传入统一 JSON 文件进行 CPU/内存绑定。

#### 一、能力概述

统一 JSON 绑定文件用于同时描述**进程级**与**线程级**绑定策略：

- **进程级绑定**：`msrun --bind_numa=<json>` 在启动 scheduler/worker 进程时，根据 JSON 选择 `taskset` 或 `numactl` 进行 CPU/NUMA 绑定。
- **线程级绑定**：`mindspore.runtime.set_cpu_affinity(enable_affinity=True, bind_file=<json>)` 根据 JSON 中的模块配置，对 `main/runtime/minddata/pynative` 等线程做绑核。

两者配合实现：

- 进程级：对“主线程/进程”做 CPU 绑定 + 内存 NUMA 绑定。
- 线程级：对关键线程模块做更细粒度的 CPU 绑定。

#### 二、JSON 文件结构

统一 JSON 文件必须为一个对象，包含以下字段：

```json
{
  "bind_config": {
    "bind_cpu_mode": "cpu",
    "bind_memory_mode": "numa",
    "actor_thread_fix_bind": true
  },
  "bind_cpu": {
    "device0": {
      "main": "0-4",
      "runtime": "5-9",
      "pynative": "10-14",
      "minddata": "15-19"
    },
    "device1": {
      "main": "20-24",
      "runtime": "25-29",
      "pynative": "30-34",
      "minddata": "35-39"
    },
    "scheduler": {
      "main": "40-45"
    }
  },
  "bind_memory": {
    "device0": 0,
    "device1": 1,
    "scheduler": 2
  }
}
```

1. bind_config

    - `bind_cpu_mode`：CPU 绑定模式，取值：
        - `"cpu"`：按 CPU 核心列表绑定。
        - `"numa"`：按 NUMA 节点绑定。
        - `"none"`：不做 CPU 绑定。
    - `bind_memory_mode`：内存绑定模式，取值：
        - `"numa"`：按 NUMA 节点绑定内存。
        - `"none"`：不做内存绑定。
    - `actor_thread_fix_bind`：可选，bool。
        - `true`：对 runtime 线程采用“固定绑定”的策略。runtime 共包含5个actor线程，如对device0的runtime绑定范围为"5-9"，采用“固定绑定”策略时，每个线程顺序绑定范围段内的一个 CPU，即`actor_thread0`绑定CPU 5，`actor_thread1`绑定CPU 6，以此类推。
        - `false`：不固定绑定（允许更灵活的绑核方式）。如对device0的runtime绑定范围为"5-9"，采用“非固定绑定”策略时，每个线程均绑定CPU范围段"5-9"。

2. bind_cpu

    - 当 `bind_cpu_mode="cpu"`：
        - key 为 `deviceX` 或 `scheduler`。
        - value 为对象，模块名 -> CPU 范围。
        - 模块名支持：`main` / `runtime` / `pynative` / `minddata`。
        - CPU 范围可为字符串（推荐），如：`"0-4"` / `"0,2,4"` / `"0-3,8-11"`。

    - 当 `bind_cpu_mode="numa"`：
        - key 为 `deviceX` 或 `scheduler`。
        - value 为 NUMA 节点（int 或字符串范围），如：`0` / `"0"` / `"0-1,3"`。

3. bind_memory

    - 仅当 `bind_memory_mode="numa"` 才生效。
    - key 为 `deviceX` 或 `scheduler`。
    - value 为 NUMA 节点（int 或字符串范围），如：`0` / `"0"` / `"0-1,3"`。

> JSON 中的 `deviceX` 指**物理设备 ID**。若设置了 `ASCEND_RT_VISIBLE_DEVICES`，请使用可见设备的物理 ID。例如：`ASCEND_RT_VISIBLE_DEVICES=3,5` 时，应使用 `device3`、`device5`。

#### 三、msrun --bind_numa 的行为说明

`msrun --bind_numa=<json>` 启动进程时行为如下：

- `bind_cpu_mode="cpu"`：

    - 进程主线程使用 `taskset -c <main>` 绑定 CPU；
    - 若 `bind_memory_mode="numa"`，则使用 `numactl --membind <node> --physcpubind <main>`分别以CPU核心粒度和NUMA节点粒度绑定 CPU和内存。

- `bind_cpu_mode="numa"`：

    - 使用 `numactl --cpunodebind <node>` 绑定 CPU；
    - 若 `bind_memory_mode="numa"`，则使用 `numactl --membind <node> --cpunodebind <node>`同时以NUMA节点粒度绑定 CPU 和内存。

- `bind_cpu_mode="none"`：

    - 不进行 CPU 绑定；

- `bind_memory_mode="numa"`：

    - 使用 `numactl --membind <node>` 绑定内存。

#### 四、set_cpu_affinity 的行为说明

调用方式：

```python
mindspore.runtime.set_cpu_affinity(True, bind_file="/path/to/bind.json")
```

具体接口说明可参考 [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html)。

行为规则：

- 仅当 `bind_cpu_mode="cpu"` 时，才会从 JSON 提取线程绑定策略。
- 根据 `deviceX` 的 `main/runtime/minddata/pynative` 等字段设置线程绑核。
- 若使用 `msrun --bind_numa` 启动，主线程 `main` 已由进程级绑定完成，
  `set_cpu_affinity` 通常不再绑定 `main`。

一致性检查：

- msrun 启动时会记录 `MSRUN_BIND_FILE` 与文件 hash。
- `set_cpu_affinity` 传入 `bind_file` 时必须与 msrun 一致，否则会报错。

#### 五、自动化脚本生成 JSON

可使用自动化脚本[gen_bind_json.py](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/set_affinity/gen_bind_json.py)生成统一 JSON 文件：

```bash
python gen_bind_json.py -o bind.json
```

脚本特性：

- 自动检测设备数量、CPU/NUMA、NPU 与 NUMA 的亲和性。
- 若无法获取亲和性，自动均分 CPU/NUMA。
- 默认生成：`bind_cpu_mode=cpu`、`bind_memory_mode=numa`、`actor_thread_fix_bind=True`。
- scheduler 仅绑定 CPU，不绑定内存。

常用参数：

- `--device-ids`：手动指定设备 ID（如 `0,1,2`）。
- `--device-count`：当自动检测失败时，指定设备数量。
- `--runtime-range`：runtime 相对 CPU 范围（默认 `4-8`）。
- `--minddata-range`：minddata 相对 CPU 范围（默认 `9-12`）。
- `--main-range`：main 相对 CPU 范围（默认 `13-19`）。
- `--pynative-range`：可选，pynative 相对 CPU 范围（默认不配置）。
- `--scheduler-range`：scheduler 相对 CPU 范围（默认 `20-23`）。
- `--scheduler-base`：scheduler 基准 CPU 列表：

    - `free`：使用未分配给 device 的空闲 CPU；不足则回退全局。
    - `global`：使用全局可用 CPU 列表。
    - `device0`：使用 device0 的 CPU 列表。

如需进一步定制（例如新增模块、特定 NUMA 亲和策略），可在 JSON 中扩展或在生成脚本中调整相对 CPU 范围。

#### 六、示例配置

##### 示例 1：CPU 绑定 + NUMA 内存绑定（进程级 + 线程级）

```json
{
  "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa", "actor_thread_fix_bind": true},
  "bind_cpu": {
    "scheduler": {"main": "40-45"},
    "device0": {"main": "0-4", "runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
    "device1": {"main": "20-24", "runtime": "25-29", "pynative": "30-34", "minddata": "35-39"}
  },
  "bind_memory": {"device0": 0, "device1": 1, "scheduler": 2}
}
```

绑定说明：

- 进程级（cpu 粒度）：`scheduler`、`device0`、`device1` 的 `main` 使用 `numactl --physcpubind` 绑定到对应 CPU 段。
- 线程级（cpu 粒度）：`runtime` / `pynative` / `minddata` 由 `set_cpu_affinity` 绑定到对应 CPU 段。
- 内存（numa 粒度）：`device0`/`device1`/`scheduler` 按 `bind_memory` 绑定到指定 NUMA 节点。

##### 示例 2：按 NUMA 绑定 CPU + 内存（仅进程级）

```json
{
  "bind_config": {"bind_cpu_mode": "numa", "bind_memory_mode": "numa"},
  "bind_cpu": {"device0": 0, "device1": 1},
  "bind_memory": {"device0": 0, "device1": 1, "scheduler": "2-3,4"}
}
```

绑定说明：

- 进程级（numa 粒度）：`device0`/`device1` 使用 `numactl --cpunodebind` 绑定到 NUMA 节点。
- 线程级：不生效（`bind_cpu_mode=numa` 不会触发 `set_cpu_affinity`）。
- 内存（numa 粒度）：`device0`/`device1`/`scheduler` 使用 `--membind` 绑定。

##### 示例 3：CPU 绑定（无内存绑定）

```json
{
  "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "none", "actor_thread_fix_bind": false},
  "bind_cpu": {
    "scheduler": {"main": "40-45"},
    "device0": {"main": "0-4", "runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
    "device1": {"main": "20-24", "runtime": "25-29", "pynative": "30-34", "minddata": "35-39"}
  }
}
```

绑定说明：

- 进程级（cpu 粒度）：`scheduler`/`device0`/`device1` 的 `main` 使用 `taskset -c` 绑定。
- 线程级（cpu 粒度）：`runtime`/`pynative`/`minddata` 由 `set_cpu_affinity` 绑定。
- 内存：未绑定（`bind_memory_mode=none`）。

##### 示例 4：仅绑定 main（CPU 模式 + 无模块绑定）

```json
{
  "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa"},
  "bind_cpu": {
    "device0": {"main": "0-4"},
    "device1": {"main": "20-24"}
  }
}
```

绑定说明：

- 进程级（cpu 粒度）：仅 `device0`/`device1` 的 `main` 使用 `taskset -c` 绑定。
- 线程级：不绑定（模块配置缺失）。
- 内存：未绑定（未提供 `bind_memory`）。

##### 示例 5：仅按 NUMA 绑定 CPU（无内存绑定）

```json
{
  "bind_config": {"bind_cpu_mode": "numa", "bind_memory_mode": "none"},
  "bind_cpu": {"device0": 0, "device1": 1}
}
```

绑定说明：

- 进程级（numa 粒度）：`device0`/`device1` 使用 `numactl --cpunodebind` 绑定。
- 线程级：不绑定（`bind_cpu_mode=numa`）。
- 内存：未绑定（`bind_memory_mode=none`）。

##### 示例 6：仅绑定内存（CPU 不绑定）

```json
{
  "bind_config": {"bind_cpu_mode": "none", "bind_memory_mode": "numa"},
  "bind_memory": {"device0": 0, "device1": 1}
}
```

绑定说明：

- 进程级（numa 粒度）：仅内存使用 `numactl --membind` 绑定。
- CPU：不绑定（`bind_cpu_mode=none`）。
- 线程级：不绑定。

##### 示例 7：仅绑定 runtime/minddata/pynative（无 main）

```json
{
  "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "none"},
  "bind_cpu": {
    "device0": {"runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
    "device1": {"runtime": "25-29", "pynative": "30-34", "minddata": "35-39"}
  }
}
```

绑定说明：

- 进程级：不绑定（无 `main`，不会使用 `taskset/numactl`）。
- 线程级（cpu 粒度）：`runtime`/`pynative`/`minddata` 由 `set_cpu_affinity` 绑定。
- 内存：未绑定（`bind_memory_mode=none`）。
