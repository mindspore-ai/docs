# rank table启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/parallel/rank_table.md)

## 概述

`rank table`启动是Ascend硬件平台独有的启动方式。该方式不依赖第三方库，采用单卡单进程运行方式，需要用户在脚本中创建与使用的卡数量一致的进程。该方法在多机下各节点的脚本一致，方便快速批量部署。

相关配置：

`rank table`主要需要配置rank_table文件，以2卡环境配置文件`rank_table_2pcs.json`为例：

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.*.*.*",
            "device": [
                {"device_id": "0","device_ip": "192.1.*.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.*.6","rank_id": "1"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

其中，需要根据实际训练环境修改的参数项有：

- `server_count`表示参与训练的机器数量。
- `server_id`表示当前机器的IP地址。
- `device_id`表示卡的物理序号，即卡所在机器中的实际序号。
- `device_ip`表示集成网卡的IP地址，可以在当前机器执行指令`cat /etc/hccn.conf`，`address_x`的键值就是网卡IP地址。
- `rank_id`表示卡的逻辑序号，固定从0开始编号。

## 操作实践

> 您可以在这里下载完整的样例代码：[startup_method](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/startup_method)。

目录结构如下：

```text
└─ sample_code
    ├─ startup_method
       ├── net.py
       ├── rank_table_8pcs.json
       ├── rank_table_16pcs.json
       ├── rank_table_cross_cluster_16pcs.json
       ├── run_rank_table.sh
       ├── run_rank_table_cluster.sh
       ├── run_rank_table_cross_cluster.sh
    ...
```

其中，`net.py`是定义网络结构和训练过程，`run_rank_table.sh`、`run_rank_table_cluster.sh`、`run_rank_table_cross_cluster.sh`是执行脚本。`rank_table_8pcs.json`、`rank_table_16pcs.json`、`rank_table_cross_cluster_16pcs.json`分别是8卡、16卡和跨集群16卡的rank_table配置文件。

### 1. 准备Python训练脚本

这里以数据并行为例，训练一个MNIST数据集的识别网络。

首先指定运行模式、设备ID、硬件设备等。与单卡脚本不同，并行脚本还需指定并行模式等配置项，并通过init初始化HCCL通信。此处若不设置`device_target`，则会自动指定为MindSpore包对应的后端硬件设备。

```python
import os
import mindspore as ms
from mindspore.communication import init

device_id = int(os.getenv('DEVICE_ID'))
ms.set_device(device_id=device_id)
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

`rank table`方式采用单卡单进程运行方式，即每张卡上运行1个进程，进程数量与使用的卡的数量一致。每个进程创建1个目录，用来保存日志信息以及算子编译信息。下面以使用8张卡的分布式训练脚本为例，演示如何运行脚本：

```bash
RANK_SIZE=8
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
export RANK_SIZE=$RANK_SIZE

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./net.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./net.py > train$i.log 2>&1 &
    cd ../
done
```

分布式相关的环境变量有：

- `RANK_TABLE_FILE`：组网信息文件的路径。
- `DEVICE_ID`：当前卡在机器上的实际序号。
- `RANK_ID`：当前卡的逻辑序号。

在当前路径配置好`rank_table_8pcs.json`后，执行以下指令：

```bash
bash run_rank_table.sh
```

运行结束后，日志文件保存在`device0`、 `device1`等目录下，`env*.log`中记录了环境变量的相关信息，输出结果保存在`train*.log`中，示例如下：

```text
epoch: 0, step: 0, loss is 2.3391366
epoch: 0, step: 10, loss is 1.8047495
epoch: 0, step: 20, loss is 1.2186875
...
```

#### 多机多卡

在Ascend环境下，跨机器的NPU单元的通信与单机内各个NPU单元的通信一样，依旧是通过HCCL进行通信，区别在于：单机内的NPU单元天然是互通的，而跨机器的则需要保证两台机器的网络是互通的。确认的方法如下：

在1号服务器执行下述命令，会为每个设备配置2号服务器对应设备的`device ip`。例如，将1号服务器卡0的目标IP配置为2号服务器的卡0的IP。配置命令需要使用`hccn_tool`工具。[`hccn_tool`](https://support.huawei.com/enterprise/zh/ascend-computing/a300t-9000-pid-250702906?category=developer-documents)是一个HCCL的工具，由CANN包自带。

```bash
hccn_tool -i 0 -netdetect -s address 192.*.92.131
hccn_tool -i 1 -netdetect -s address 192.*.93.131
hccn_tool -i 2 -netdetect -s address 192.*.94.131
hccn_tool -i 3 -netdetect -s address 192.*.95.131
hccn_tool -i 4 -netdetect -s address 192.*.92.141
hccn_tool -i 5 -netdetect -s address 192.*.93.141
hccn_tool -i 6 -netdetect -s address 192.*.94.141
hccn_tool -i 7 -netdetect -s address 192.*.95.141
```

其中，`-i 0`指定设备ID；`-netdetect`指定网络检测对象IP属性；`-s address`表示设置属性为IP地址；`192.*.92.131`表示2号服务器的设备0的IP地址。接口命令可以[参考此处](https://support.huawei.com/enterprise/zh/doc/EDOC1100251947/8eff627f)。

在1号服务器上面执行完上述命令后，通过下述命令开始检测网络链接状态。在此使用`hccn_tool`的另一个功能，此功能的含义可以[参考此处](https://support.huawei.com/enterprise/zh/doc/EDOC1100251947/7d059b59)。

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

在确认了机器之间的NPU单元的网络是通畅后，配置多机的json配置文件。本文档以16卡的配置文件为例进行介绍，详细的配置文件说明可参照本文档单机多卡部分的相关内容。

需要注意的是，在多机的json文件配置中，要求rank_id的排序，与server_id的字典序一致。

```json
{
  "version": "1.0",
  "server_count": "2",
  "server_list": [
    {
      "server_id": "10.*.*.*",
      "device": [
        {"device_id": "0","device_ip": "192.1.*.6","rank_id": "0"},
        {"device_id": "1","device_ip": "192.2.*.6","rank_id": "1"},
        {"device_id": "2","device_ip": "192.3.*.6","rank_id": "2"},
        {"device_id": "3","device_ip": "192.4.*.6","rank_id": "3"},
        {"device_id": "4","device_ip": "192.1.*.7","rank_id": "4"},
        {"device_id": "5","device_ip": "192.2.*.7","rank_id": "5"},
        {"device_id": "6","device_ip": "192.3.*.7","rank_id": "6"},
        {"device_id": "7","device_ip": "192.4.*.7","rank_id": "7"}],
      "host_nic_ip": "reserve"
    },
    {
      "server_id": "10.*.*.*",
      "device": [
        {"device_id": "0","device_ip": "192.1.*.8","rank_id": "8"},
        {"device_id": "1","device_ip": "192.2.*.8","rank_id": "9"},
        {"device_id": "2","device_ip": "192.3.*.8","rank_id": "10"},
        {"device_id": "3","device_ip": "192.4.*.8","rank_id": "11"},
        {"device_id": "4","device_ip": "192.1.*.9","rank_id": "12"},
        {"device_id": "5","device_ip": "192.2.*.9","rank_id": "13"},
        {"device_id": "6","device_ip": "192.3.*.9","rank_id": "14"},
        {"device_id": "7","device_ip": "192.4.*.9","rank_id": "15"}],
      "host_nic_ip": "reserve"
    }
  ],
  "status": "completed"
}
```

准备好配置文件后，可以进行分布式多机训练脚本的组织，以2机16卡为例，两台机器上编写的脚本与单机多卡的运行脚本类似，区别在于指定不同的rank_id变量。

```bash
RANK_SIZE=16
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_16pcs.json
export RANK_SIZE=$RANK_SIZE

RANK_START=$1
DEVICE_START=0

for((i=0;i<=7;i++));
do
  export RANK_ID=$[i+RANK_START]
  export DEVICE_ID=$[i+DEVICE_START]
  rm -rf ./device_$RANK_ID
  mkdir ./device_$RANK_ID
  cp ./net.py ./device_$RANK_ID
  cd ./device_$RANK_ID
  env > env$i.log
  python ./net.py >train$RANK_ID.log 2>&1 &
done
```

执行时，两台机器分别执行如下命令：

```bash
# server0
bash run_rank_table_cluster.sh 0
# server1
bash run_rank_table_cluster.sh 8
```

其中，rank_table.json按照本章节展示的16卡的分布式json文件参考配置。

运行结束后，日志文件保存在`device_0`、 `device_1`等目录下，`env*.log`中记录了环境变量的相关信息，输出结果保存在`train*.log`中。

#### 跨集群

对于如今的大模型而言，使用计算集群进行训练已经成为一种常态。然而，随着模型规模的不断提升，单一集群的资源难以满足模型训练所需的显存要求。因此，支持跨集群通信成为了训练超大规模模型的前提。

目前，昇腾硬件的HCCL通信库暂不支持跨集群通信。因此，MindSpore框架提供了一套跨集群通信库，使得不同集群的NPU之间能够实现高效通信。借助这一通信库，用户可以突破单一集群的显存限制，实现超大规模模型的跨集群并行训练。

目前，MindSpore框架仅需在多机多卡的json配置文件中添加跨集群的`cluster_list`配置项即可开启这一功能，本文档同样以2机16卡（假设两个机器不在同一集群）配置文件为例，介绍跨集群相关配置项的编写方法，详细的配置文件说明可以参照本文档单机多卡部分的介绍。

```json
{
  "version": "1.0",
  "server_count": "2",
  "server_list": [
    {
      "server_id": "server_0_10.*.*.*",
      "server_ip": "10.*.*.*",
      "device": [
        {"device_id": "0", "device_ip": "192.1.*.6", "rank_id": "0", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "1", "device_ip": "192.2.*.6", "rank_id": "1", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "2", "device_ip": "192.3.*.6", "rank_id": "2", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "3", "device_ip": "192.4.*.6", "rank_id": "3", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "4", "device_ip": "192.1.*.7", "rank_id": "4", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "5", "device_ip": "192.2.*.7", "rank_id": "5", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "6", "device_ip": "192.3.*.7", "rank_id": "6", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "7", "device_ip": "192.4.*.7", "rank_id": "7", "dpu_ip": "8.2.17.60", "numa_id": ""}],
      "host_nic_ip": "reserve",
      "pod_ip": "127.0.0.1"
    },
    {
      "server_id": "server_1_10.*.*.*",
      "server_ip": "10.*.*.*",
      "device": [
        {"device_id": "0", "device_ip": "192.1.*.8", "rank_id": "8", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "1", "device_ip": "192.2.*.8", "rank_id": "9", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "2", "device_ip": "192.3.*.8", "rank_id": "10", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "3", "device_ip": "192.4.*.8", "rank_id": "11", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "4", "device_ip": "192.1.*.9", "rank_id": "12", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "5", "device_ip": "192.2.*.9", "rank_id": "13", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "6", "device_ip": "192.3.*.9", "rank_id": "14", "dpu_ip": "8.2.17.60", "numa_id": ""},
        {"device_id": "7", "device_ip": "192.4.*.9", "rank_id": "15", "dpu_ip": "8.2.17.60", "numa_id": ""}],
      "host_nic_ip": "reserve",
      "pod_ip": "127.0.0.1"
    }
  ],
  "cluster_list": [
    {
      "cluster_id": "cluster_0",
      "network_type": "ROCE",
      "az_id": "az_0",
      "region_id": "region_0",
      "server_list": [
        {
          "server_id": "server_0_10.*.*.*"
        }
      ]
    },
    {
      "cluster_id": "cluster_1",
      "network_type": "ROCE",
      "az_id": "az_1",
      "region_id": "region_1",
      "server_list": [
        {
          "server_id": "server_1_10.*.*.*"
        }
      ]
    }
  ],
  "status": "completed"
}
```

其中，跨集群需要根据实际训练环境添加和修改的参数项有：

- `server_id`表示当前机器的全局唯一标识。
- `server_ip`表示当前机器的IP地址。
- `dpu_ip`表示卡在租户VPC内的虚拟IP地址，用于跨集群通信。
- `numa_id`表示卡在当前机器上NUMA亲和的CPU核序号。
- `cluster_id`表示集群的全局唯一标识。
- `network_type`表示集群内的机器间的网络类型，目前都是"ROCE"。
- `az_id`表示集群所在的AZ id。
- `server_list`表示当前集群包含的机器列表。

准备好配置文件后，跨集群的分布式训练脚本与本文档多机多卡的分布式训练脚本保持一致，以2集群16卡为例，两个集群的两台机器上编写的脚本与多机多卡的运行脚本相同，区别在于指定不同的rank_id变量。

```bash
RANK_SIZE=16
EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2_cluster_16pcs.json
export RANK_SIZE=$RANK_SIZE

RANK_START=$1
DEVICE_START=0

for((i=0;i<=7;i++));
do
  export RANK_ID=$[i+RANK_START]
  export DEVICE_ID=$[i+DEVICE_START]
  rm -rf ./device_$RANK_ID
  mkdir ./device_$RANK_ID
  cp ./net.py ./device_$RANK_ID
  cd ./device_$RANK_ID
  env > env$i.log
  python ./net.py >train$RANK_ID.log 2>&1 &
done
```

执行时，两个集群中的两台机器分别执行如下命令：

```bash
# server0
bash run_rank_table_cross_cluster.sh 0
# server1
bash run_rank_table_cross_cluster.sh 8
```

其中，`rank_table_cross_cluster_16pcs.json`按照本章节展示的2集群16卡的跨集群分布式json文件参考配置，每个集群的每台机器上使用的`rank_table_cross_cluster_16pcs.json`配置需要保持一致。

运行结束后，日志文件保存在各个集群中每台机器的`device_0`、 `device_1`等目录下，`env*.log`中记录了环境变量的相关信息，输出结果保存在`train*.log`中。
