# rank table启动

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/rank_table.md)

## 概述

`rank table`启动是Ascend硬件平台独有的启动方式。该方式不依赖第三方库，采用单卡单进程运行方式，需要用户在脚本中创建与使用的卡的数量一致的进程。该方法在多机下各节点的脚本一致，方便快速批量部署。

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

其中需要根据实际训练环境修改的参数项有：

- `server_count`表示参与训练的机器数量。
- `server_id`表示当前机器的IP地址。
- `device_id`表示卡物理序号，即卡所在机器中的实际序号。
- `device_ip`表示集成网卡的IP地址，可以在当前机器执行指令`cat /etc/hccn.conf`，`address_x`的键值就是网卡IP地址。
- `rank_id`表示卡逻辑序号，固定从0开始编号。

## 操作实践

> 您可以在这里下载完整的样例代码：[startup_method](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/startup_method)。

目录结构如下：

```text
└─ sample_code
    ├─ startup_method
       ├── allgather_test.py
       ├── rank_table_8pcs.json
       ├── rank_table_16pcs.json
       ├── run_ran_table.sh
       ├── run_ran_table_cluster.sh
    ...
```

其中，`allgather_test.py`是定义网络结构，`run_ran_table.sh`、`run_ran_table_cluster.sh`是执行脚本。

### 1. 准备Python训练脚本

这里以数据并行为例，训练一个MNIST数据集的识别网络，网络结构和训练过程与数据并行网络一致。

首先指定运行模式、设备ID、硬件设备等，与单卡脚本不同，并行脚本还需指定并行模式等配置项，并通过init初始化HCCL通信。此处不设置`device_target`会自动指定为MindSpore包对应的后端硬件设备。

```python
import os
import mindspore as ms
from mindspore.communication import init

device_id = int(os.getenv('DEVICE_ID'))
ms.set_context(device_id=device_id)
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
bash run_ran_table.sh
```

运行结束后，日志文件保存`device0`、 `device1`等目录下，`env*.log`中记录了环境变量的相关信息，输出结果保存在`train*.log`中，示例如下：

```text
epoch: 0, step: 0, loss is 2.3391366
epoch: 0, step: 10, loss is 1.8047495
epoch: 0, step: 20, loss is 1.2186875
epoch: 0, step: 30, loss is 1.3065228
epoch: 0, step: 40, loss is 1.0825632
epoch: 0, step: 50, loss is 1.0281029
epoch: 0, step: 60, loss is 0.8405618
epoch: 0, step: 70, loss is 0.7346531
epoch: 0, step: 80, loss is 0.688364
epoch: 0, step: 90, loss is 0.51331174
epoch: 0, step: 100, loss is 0.53782797
...
```

#### 多机多卡

在Ascend环境下，跨机器的NPU单元的通信与单机内各个NPU单元的通信一样，依旧是通过HCCL进行通信，区别在于，单机内的NPU单元天然的是互通的，而跨机器的则需要保证两台机器的网络是互通的。确认的方法如下：

在1号服务器执行下述命令，会为每个设备配置2号服务器对应设备的`device ip`。例如将1号服务器卡0的目标IP配置为2号服务器的卡0的IP。配置命令需要使用`hccn_tool`工具。[`hccn_tool`](https://support.huawei.com/enterprise/zh/ascend-computing/a300t-9000-pid-250702906?category=developer-documents)是一个HCCL的工具，由CANN包自带。

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

`-i 0`指定设备ID。`-netdetect`指定网络检测对象IP属性。`-s address`表示设置属性为IP地址。`192.*.92.131`表示2号服务器的设备0的IP地址。接口命令可以[参考此处](https://support.huawei.com/enterprise/zh/doc/EDOC1100251947/8eff627f)。

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

在确认了机器之间的NPU单元的网络是通畅后，配置多机的json配置文件，本教程以16卡的配置文件为例，详细的配置文件说明可以参照本教程单机多卡部分的介绍。需要注意的是，在多机的json文件配置中，要求rank_id的排序，与server_id的字典序一致。

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
  cp ./allgather_test.py ./device_$RANK_ID
  cd ./device_$RANK_ID
  env > env$i.log
  python ./allgather_test.py >train$RANK_ID.log 2>&1 &
done
```

执行时，两台机器分别执行如下命令，其中rank_table.json按照本章节展示的16卡的分布式json文件参考配置。

```bash
# server0
bash run_ran_table_cluster.sh 0
# server1
bash run_ran_table_cluster.sh 8
```

运行结束后，日志文件保存`device_0`、 `device_1`等目录下，`env*.log`中记录了环境变量的相关信息，输出结果保存在`train*.log`中。
