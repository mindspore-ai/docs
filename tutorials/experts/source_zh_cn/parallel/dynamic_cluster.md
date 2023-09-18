# 动态组网启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/dynamic_cluster.md)

## 概述

出于训练时的可靠性要求，MindSpore提供了**动态组网**特性，用户能够不依赖任何第三方库(OpenMPI)来启动Ascend/GPU/CPU分布式训练任务，并且训练脚本无需做任何修改。我们建议用户优先使用此种启动方式。

MindSpore**动态组网**特性通过**复用Parameter Server模式训练架构**，取代了OpenMPI能力，可参考[Parameter Server模式](https://mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html)训练教程。

**动态组网**特性将多个MindSpore训练进程作为`Worker`启动，并且额外启动一个`Scheduler`负责组网和容灾恢复。用户只需对启动脚本做少量修改，即可执行分布式训练。

> 动态组网支持Ascend、GPU和CPU，因此动态组网启动脚本能在多种硬件平台间快速迁移，无需对其进行额外修改。此外动态组网需要在Graph模式下运行。

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
        <td align="left">MS_ROLE</td>
        <td align="left">指定本进程角色。</td>
        <td align="left">String</td>
        <td align="left">
            <ul>
                <li>MS_SCHED: 代表Scheduler进程，一个训练任务只启动一个Scheduler，负责组网，容灾恢复等，<b>不会执行训练代码</b>。</li>
                <li>MS_WORKER: 代表Worker进程，一般设置分布式训练进程为此角色。</li>
                <li>MS_PSERVER: 代表Parameter Server进程，只有在Parameter Server模式下此角色生效，具体请参考<a href="https://mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html">Parameter Server模式</a>。</li>
            </ul>
        </td>
        <td align="left">Worker和Parameter Server进程会向Scheduler进程注册从而完成组网。</td>
    </tr>
    <tr>
        <td align="left">MS_SCHED_HOST</td>
        <td align="left">指定Scheduler的IP地址。</td>
        <td align="left">String</td>
        <td align="left">合法的IP地址。</td>
        <td align="left">当前版本暂不支持IPv6地址。</td>
    </tr>
    <tr>
        <td align="left">MS_SCHED_PORT</td>
        <td align="left">指定Scheduler绑定端口号。</td>
        <td align="left">Integer</td>
        <td align="left">1024～65535范围内的端口号。</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left">MS_NODE_ID</td>
        <td align="left">指定本进程的ID，集群内唯一。</td>
        <td align="left">String</td>
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
        <td align="left">MS_WORKER_NUM</td>
        <td align="left">指定角色为MS_WORKER的进程数量。</td>
        <td align="left">Integer</td>
        <td align="left">大于0的整数。</td>
        <td align="left">
            用户启动的Worker进程数量应当与此环境变量值相等。若小于此数值，组网失败；若大于此数值，Scheduler进程会根据Worker注册先后顺序完成组网，多余的Worker进程会启动失败。
        </td>
    </tr>
    <tr>
        <td align="left">MS_SERVER_NUM</td>
        <td align="left">指定角色为MS_PSERVER的进程数量。</td>
        <td align="left">Integer</td>
        <td align="left">大于0的整数。</td>
        <td align="left">只在Parameter Server训练模式下需要设置。</td>
    </tr>
    <tr>
        <td align="left">MS_ENABLE_RECOVERY</td>
        <td align="left">开启容灾。</td>
        <td align="left">Integer</td>
        <td align="left">1代表开启，0代表关闭。默认为0。</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left">MS_RECOVERY_PATH</td>
        <td align="left">持久化路径文件夹。</td>
        <td align="left">String</td>
        <td align="left">合法的用户目录。</td>
        <td align="left">Worker和Scheduler进程在执行过程中会进行必要的持久化，如用于恢复组网的节点信息以及训练业务中间状态等，并通过文件保存。</td>
    </tr>
    <tr>
        <td align="left">MS_HCCL_CM_INIT</td>
        <td align="left">是否使用CM方式初始化HCCL。</td>
        <td align="left">Integer</td>
        <td align="left">1代表是，0代表否。默认为0。</td>
        <td align="left">此环境变量只在<b>Ascend硬件平台并且通信域数量较多</b>的情况下建议开启。开启此环境变量后，能够降低HCCL集合通信库的内存占用，并且训练任务执行方式与`rank table`启动方式相同。</td>
    </tr>
</table>

> 环境变量`MS_SCHED_HOST`、`MS_SCHED_PORT`、`MS_WORKER_NUM`内容需保持一致，否则会由于各进程配置不一致导致组网失败。

## 操作实践

动态组网启动脚本在各硬件平台下一致，下面以Ascend为例演示如何编写启动脚本：

> 样例的运行目录：[startup_method](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/startup_method)。

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

这里以数据并行为例，训练一个MNIST数据集的识别网络，网络结构和训练过程与数据并行网络一致。

首先指定运行模式、硬件设备等，与单卡脚本不同，并行脚本还需指定并行模式等配置项，并通过init初始化HCCL或NCCL通信。此处不设置`device_target`会自动指定为MindSpore包对应的后端硬件设备。

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
from mindspore import nn, ops
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

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
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

单机多卡启动脚本内容[run_dynamic_cluster.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/startup_method/run_dynamic_cluster.sh)如下，以单机8卡为例：

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
epoch: 0, step: 40, loss is 1.0643986
epoch: 0, step: 50, loss is 1.1021575
epoch: 0, step: 60, loss is 0.8510884
epoch: 0, step: 70, loss is 1.0581372
epoch: 0, step: 80, loss is 1.0076828
epoch: 0, step: 90, loss is 0.88950706
...
```

#### 多机多卡

多机训练场景下，需拆分启动脚本。下面以执行2机8卡训练，每台机器执行启动4个Worker为例：

脚本[run_dynamic_cluster_1.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/startup_method/run_dynamic_cluster_1.sh)在节点1上启动1`Scheduler`和`Worker1`到`Worker4`：

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

脚本[run_dynamic_cluster_2.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/startup_method/run_dynamic_cluster_2.sh)在节点2上启动`Worker5`到`Worker8`（无需执行Scheduler）：

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

> 多机任务`MS_WORKER_NUM`应当为集群中Worker节点总数。
> 节点间网络需保持连通，可使用`telnet <scheduler ip> <scheduler port>`指令测试本节点是否和已启动的Scheduler节点连通。

在节点1执行：

```bash
bash run_dynamic_cluster_1.sh
```

在节点2执行：

```bash
bash run_dynamic_cluster_2.sh
```

即可执行2机8卡分布式训练任务，结果应与单机多卡结果一致。

## 容灾恢复

动态组网支持数据并行下容灾恢复。在多卡数据并行训练场景下，发生进程异常退出，重新拉起对应进程对应的脚本后训练可继续，并且不影响精度收敛。容灾恢复配置和样例可参考[动态组网场景下故障恢复](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/disaster_recover.html)教程。

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

p12文件中的秘钥为密文存储，在启动时需要传入密码，具体参数请参考Python API [mindspore.set_ps_context](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_ps_context.html#mindspore.set_ps_context)中的`client_password`以及`server_password`字段。
