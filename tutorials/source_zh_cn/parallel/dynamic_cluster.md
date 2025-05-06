# 动态组网启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/parallel/dynamic_cluster.md)

## 概述

出于训练时的可靠性要求，MindSpore提供了**动态组网**特性，用户能够不依赖任何第三方库(OpenMPI)来启动Ascend/GPU/CPU分布式训练任务，并且训练脚本无需做任何修改。我们建议用户优先使用此种启动方式。

MindSpore**动态组网**特性通过**复用Parameter Server模式训练架构**，取代了OpenMPI能力。

**动态组网**特性将多个MindSpore训练进程作为`Worker`启动，并且额外启动一个`Scheduler`负责组网和容灾恢复。因此无需借助OpenMPI的消息传递机制即可实现分布式训练。用户只需对启动脚本做少量修改，便可执行分布式训练。

> 动态组网支持Ascend、GPU和CPU，因此动态组网启动脚本能在多种硬件平台间快速迁移，无需对其进行额外修改。

相关环境变量：

<table align="center">
    <tr>
        <th align="left">环境变量</th>
        <th align="left">功能</th>
        <th align="left">类型</th>
        <th align="left">取值</th>
        <th align="left">说明</th>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_ROLE</td>
        <td align="left">指定本进程角色。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">
            <ul>
                <li>MS_SCHED: 代表Scheduler进程，一个训练任务只启动一个Scheduler，负责组网，容灾恢复等，<b>不会执行训练代码</b>。</li>
                <li>MS_WORKER: 代表Worker进程，一般设置分布式训练进程为此角色。</li>
                <li>MS_PSERVER: 代表Parameter Server进程，只有在Parameter Server模式下此角色生效。</li>
            </ul>
        </td>
        <td align="left">Worker和Parameter Server进程会向Scheduler进程注册从而完成组网。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_SCHED_HOST</td>
        <td align="left">指定Scheduler的IP地址。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">合法的IP地址。</td>
        <td align="left">当前版本还支持Ascend平台下的IPv6地址。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_SCHED_PORT</td>
        <td align="left">指定Scheduler绑定端口号。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">1024～65535范围内的端口号。</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_NODE_ID</td>
        <td align="left">指定本进程的ID，集群内唯一。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">代表本进程的唯一ID，默认由MindSpore自动生成。</td>
        <td align="left">
            MS_NODE_ID在在以下情况需要设置，一般情况下无需设置，由MindSpore自动生成：
            <ul>
                <li>开启容灾场景：容灾恢复时需要获取当前进程ID，从而向Scheduler重新注册。</li>
                <li>开启GLOG日志重定向场景：为了保证各训练进程日志独立保存，需设置进程ID，作为日志保存路径后缀。</li>
                <li>指定进程rank id场景：用户可通过设置MS_NODE_ID为某个整数，来指定本进程的rank id。</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_WORKER_NUM</td>
        <td align="left">指定角色为MS_WORKER的进程数量。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">大于0的整数。</td>
        <td align="left">
            用户启动的Worker进程数量应当与此环境变量值相等。若小于此数值，组网失败；若大于此数值，Scheduler进程会根据Worker注册先后顺序完成组网，多余的Worker进程会启动失败。
        </td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_SERVER_NUM</td>
        <td align="left">指定角色为MS_PSERVER的进程数量。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">大于0的整数。</td>
        <td align="left">只在Parameter Server训练模式下需要设置。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_WORKER_IP</td>
        <td align="left">指定当前进程和其他进程进行通信和组网使用的IP地址。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">合法的IP地址。</td>
        <td align="left">在使用IPv6地址进行组网时，建议设置此环境变量。但当用户设置MS_SCHED_HOST为<b>::1</b>时（代表IPv6的本地回环地址），无需设置此环境变量，这是因为MindSpore会默认使用本地回环地址进行通信。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_ENABLE_RECOVERY</td>
        <td align="left">开启容灾。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">1代表开启，0代表关闭。默认为0。</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_RECOVERY_PATH</td>
        <td align="left">持久化路径文件夹。</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">合法的用户目录。</td>
        <td align="left">Worker和Scheduler进程在执行过程中会进行必要的持久化，如用于恢复组网的节点信息以及训练业务中间状态等，并通过文件保存。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_ENABLE_LCCL</td>
        <td align="left">是否使用LCCL通信库。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">1代表开启，0代表关闭。默认为0。</td>
        <td align="left">LCCL通信库暂只支持单机多卡，并且必须在图编译等级为O0时执行。</td>
    </tr>
        <tr>
        <td align="left" style="white-space:nowrap">MS_TOPO_TIMEOUT</td>
        <td align="left">集群组网阶段超时时间，单位：秒。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为30分钟。</td>
        <td align="left">此数值代表在所有节点在这个时间窗口内均可向Scheduler进行注册，超出此时间窗口则注册失败，若节点数量不满足要求，则集群组网失败。建议用户在集群规模较大时配置此环境变量。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_NODE_TIMEOUT</td>
        <td align="left">节点心跳超时时间，单位：秒。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为30秒</td>
        <td align="left">此数值代表Scheduler以及Worker间心跳超时时间，若此时间窗口内没有心跳消息，则集群异常退出。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_RECEIVE_MSG_TIMEOUT</td>
        <td align="left">节点接收消息超时时间，单位：秒。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为300秒</td>
        <td align="left">此数值代表节点接收对端消息超时时间，若时间窗口内无消息响应，则返回空消息。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_RETRY_INTERVAL_LOWER</td>
        <td align="left">节点间消息重试间隔下限，单位：秒。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为3秒</td>
        <td align="left">此数值代表节点每次重试发送消息的时间间隔下限，MindSpore会随机选择<code>MS_RETRY_INTERVAL_LOWER</code>和<code>MS_RETRY_INTERVAL_UPPER</code>之间的值作为间隔时间。此变量可以控制Scheduler节点的消息并发量。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_RETRY_INTERVAL_UPPER</td>
        <td align="left">节点间消息重试间隔上限，单位：秒。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认为5秒</td>
        <td align="left">此数值代表节点每次重试发送消息的时间间隔上限，MindSpore会随机选择<code>MS_RETRY_INTERVAL_LOWER</code>和<code>MS_RETRY_INTERVAL_UPPER</code>之间的值作为间隔时间。此变量可以控制Scheduler节点的消息并发量。</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_DISABLE_HEARTBEAT</td>
        <td align="left">关闭集群中节点间心跳业务。</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">默认开启心跳业务</td>
        <td align="left">若设置为1，则关闭集群节点间心跳，此场景下Scheduler不会检测到Worker异常，集群不会被Scheduler控制退出。此变量可以降低Scheduler节点消息并发量。<br>在使用`gdb attach`指令调试时，建议开启此环境变量。</td>
    </tr>
</table>

> 环境变量`MS_SCHED_HOST`、`MS_SCHED_PORT`、`MS_WORKER_NUM`内容需保持一致，否则会由于各进程配置不一致导致组网失败。

## 操作实践

动态组网启动脚本在各硬件平台下一致，下面以Ascend为例演示如何编写启动脚本：

> 您可以在这里下载完整的样例代码：[startup_method](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/startup_method)。

目录结构如下：

```text
└─ sample_code
    ├─ startup_method
       ├── net.py
       ├── run_dynamic_cluster.sh
       ├── run_dynamic_cluster_1.sh
       ├── run_dynamic_cluster_2.sh
    ...
```

其中，`net.py`是定义网络结构和训练过程，`run_dynamic_cluster.sh`、`run_dynamic_cluster_1.sh`和`run_dynamic_cluster_2.sh`是执行脚本。

### 1. 准备Python训练脚本

这里以数据并行为例，训练一个MNIST数据集的识别网络。

首先指定运行模式、硬件设备等。与单卡脚本不同，并行脚本还需指定并行模式等配置项，并通过`init()`初始化HCCL、NCCL或MCCL通信域。此处未设置`device_target`，会自动指定为MindSpore包对应的后端硬件设备。

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

#### 单机多卡

单机多卡启动脚本内容[run_dynamic_cluster.sh](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/startup_method/run_dynamic_cluster.sh)如下，以单机8卡为例：

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf device
mkdir device
echo "start training"

# 循环启动8个Worker训练进程
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8          # 设置集群中Worker进程数量为8
    export MS_SCHED_HOST=127.0.0.1  # 设置Scheduler IP地址为本地环路地址
    export MS_SCHED_PORT=8118       # 设置Scheduler端口
    export MS_ROLE=MS_WORKER        # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i            # 设置进程id，可选
    python ./net.py > device/worker_$i.log 2>&1 &     # 启动训练脚本
done

# 启动1个Scheduler进程
export MS_WORKER_NUM=8              # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
python ./net.py > device/scheduler.log 2>&1 &     # 启动训练脚本
```

> Scheduler和Worker进程的训练脚本内容和启动方式完全一致，这是因为在MindSpore已经差异化处理了两种角色内部流程。用户只需按照普通的训练方式拉起进程即可，无需按照角色修改Python代码。这是动态组网启动脚本在多硬件平台能够保持一致的原因之一。

执行如下指令，即可启动单机8卡训练网络：

```bash
bash run_dynamic_cluster.sh
```

脚本会在后台运行，日志文件会保存到device目录下，结果保存在worker_*.log中，Loss结果如下：

```text
epoch: 0, step: 0, loss is 2.3499548
epoch: 0, step: 10, loss is 1.6682479
epoch: 0, step: 20, loss is 1.4237018
epoch: 0, step: 30, loss is 1.0437132
...
```

#### 多机多卡

多机训练场景下，需拆分启动脚本。下面以执行双机8卡训练为例，每台机器执行启动4个Worker：

脚本[run_dynamic_cluster_1.sh](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/startup_method/run_dynamic_cluster_1.sh)在节点1上启动1个`Scheduler`进程以及4个`Worker`进程：

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf device
mkdir device
echo "start training"

# 循环启动Worker1到Worker4，4个Worker训练进程
for((i=0;i<4;i++));
do
    export MS_WORKER_NUM=8                     # 设置集群中Worker进程数量为8
    export MS_SCHED_HOST=<node_1 ip address>   # 设置Scheduler IP地址为节点1 IP地址
    export MS_SCHED_PORT=8118                  # 设置Scheduler端口
    export MS_ROLE=MS_WORKER                   # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                       # 设置进程id，可选
    python ./net.py > device/worker_$i.log 2>&1 &     # 启动训练脚本
done

# 在节点1启动1个Scheduler进程
export MS_WORKER_NUM=8                     # 设置集群中Worker进程总数为8（包括其他节点进程）
export MS_SCHED_HOST=<node_1 ip address>   # 设置Scheduler IP地址为节点1 IP地址
export MS_SCHED_PORT=8118                  # 设置Scheduler端口
export MS_ROLE=MS_SCHED                    # 设置启动的进程为MS_SCHED角色
python ./net.py > device/scheduler.log 2>&1 &     # 启动训练脚本
```

脚本[run_dynamic_cluster_2.sh](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/sample_code/startup_method/run_dynamic_cluster_2.sh)在节点2上启动`Worker5`到`Worker8`（无需执行Scheduler）：

```bash
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf device
mkdir device
echo "start training"

# 循环启动Worker5到Worker8，4个Worker训练进程
for((i=4;i<8;i++));
do
    export MS_WORKER_NUM=8                    # 设置集群中Worker进程总数为8（包括其他节点进程）
    export MS_SCHED_HOST=<node_1 ip address>  # 设置Scheduler IP地址为节点1 IP地址
    export MS_SCHED_PORT=8118                 # 设置Scheduler端口
    export MS_ROLE=MS_WORKER                  # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                      # 设置进程id，可选
    python ./net.py > device/worker_$i.log 2>&1 &     # 启动训练脚本
done
```

> 在多机器任务中，需要为每个主机节点设置不同的主机名，否则会出现报错`deivce id`越界。可参考[FAQ](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/faq/distributed_parallel.html#q-多机场景使用动态组网或msrun启动分布式任务时报错device-id越界如何解决)。
>
> 在多机任务中，`MS_WORKER_NUM`应当为集群中Worker节点总数。
>
> 节点间网络需保持连通，可使用`telnet <scheduler ip> <scheduler port>`指令测试本节点是否和已启动的Scheduler节点连通。

在节点1执行：

```bash
bash run_dynamic_cluster_1.sh
```

在节点2执行：

```bash
bash run_dynamic_cluster_2.sh
```

即可执行双机8卡分布式训练任务，结果应与单机多卡结果一致。

## 容灾恢复

动态组网支持数据并行下容灾恢复。在多卡数据并行训练场景下，发生进程异常退出，重新拉起对应进程对应的脚本后训练可继续，并且不影响精度收敛。

## 安全认证

动态组网还支持**安全加密通道**特性，支持`TLS/SSL`协议，满足用户的安全性需求。默认情况下，安全加密通道是关闭的，若需要开启，则通过`set_ps_context`正确配置安全加密通道后，才能调用init()，否则初始化组网会失败。若想使用安全加密通道，请配置：

`set_ps_context(config_file_path="/path/to/config_file.json", enable_ssl=True, client_password="123456", server_password="123456")`

`config_file_path`指定的`config.json`配置文件需要添加如下字段：

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

- `server_cert_path`：服务端包含了证书和秘钥的密文的p12文件（SSL专用证书文件）路径。
- `crl_path`：吊销列表（用于区分无效不可信证书和有效可信证书）的文件路径。
- `client_cert_path`：客户端包含了证书和秘钥的密文的p12文件（SSL专用证书文件）路径。
- `ca_cert_path`：根证书路径。
- `cipher_list`：密码套件（支持的SSL加密类型列表）。
- `cert_expire_warning_time_in_day`：证书过期的告警时间。

p12文件中的秘钥为密文存储，在启动时需要传入密码，具体参数请参考Python API [mindspore.set_ps_context](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore/mindspore.set_ps_context.html#mindspore.set_ps_context)中的`client_password`以及`server_password`字段。
