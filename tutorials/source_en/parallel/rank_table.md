# rank table Startup

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/rank_table.md)

## Overview

`rank table` startup is a startup method unique to the Ascend hardware platform. This method does not rely on third-party libraries and runs as a single process on a single card, requiring the user to create a process in the script that matches the number of cards in use. This method is consistent across nodes in multiple machines and facilitates rapid batch deployment.

Related Configurations:

`rank table` mainly need to configure the rank_table file, taking the 2-card environment configuration file `rank_table_2pcs.json` as an example:

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

The parameter items that need to be modified according to the actual training environment are:

- `server_count` represents the number of machines involved in training.
- `server_id` represents the IP address of the current machine.
- `device_id` represents the physical serial number of the card, i.e., the actual serial number in the machine where the card is located.
- `device_ip` represents the IP address of the integrated NIC. You can execute the command `cat /etc/hccn.conf` on the current machine, and the key value of `address_x` is the IP address of the NIC.
- `rank_id` represents the card logical serial number, fixed numbering from 0.

## Operation Practice

> You can download the full sample code here: [startup_method](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/startup_method).

The directory structure is as follows:

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

`net.py` defines the network structure and the training process, `run_rank_table.sh`, `run_rank_table_cluster.sh` and `run_rank_table_cross_cluster.sh` are executing the scripts. `rank_table_8pcs.json`, `rank_table_16pcs.json` and `rank_table_cross_cluster_16pcs.json` are 8 cards, 16 cards and cross cluster 16 cards rank_table config file.

### 1. Preparing Python Training Scripts

Here, as an example of data parallel, a recognition network is trained for the MNIST dataset.

First specify the operation mode, device ID, hardware device, etc. Unlike single card scripts, parallel scripts also need to specify configuration items such as parallel mode and initialize HCCL or NCCL communication via init. If you don't set `device_target` here, it will be automatically specified as the backend hardware device corresponding to the MindSpore package.

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

Then build the following network:

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

Finally, the dataset is processed and the training process is defined:

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

### 2. Preparing the Startup Script

#### Single-Machine Multi-Card

The `rank table` method uses a single-card single-process operation, i.e., 1 process runs on each card, with the same number of processes as the number of cards in use. Each process creates a directory to store log information and operator compilation information. Below is an example of how to run a distributed training script using 8 cards:

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

Distributed-related environment variables are:

- `RANK_TABLE_FILE`: Path to the networking information file.
- `DEVICE_ID`: The actual serial number of the current card on the machine.
- `RANK_ID`: The logical serial number of the current card.

After configuring `rank_table_8pcs.json` in the current path, execute the following command:

```bash
bash run_rank_table.sh
```

After running, the log files are saved in `device0`, `device1` and other directories, `env*.log` records information about environment variables, and the output is saved in `train*.log`, as shown in the example below:

```text
epoch: 0, step: 0, loss is 2.3391366
epoch: 0, step: 10, loss is 1.8047495
epoch: 0, step: 20, loss is 1.2186875
...
```

#### Multi-Machine Multi-Card

In the Ascend environment, the communication of NPU units across machines is the same as the communication of individual NPU units within a single machine, still through the HCCL. The difference is that the NPU units within a single machine are naturally interoperable, while the cross-machine ones need to ensure that the networks of the two machines are interoperable. The method of confirmation is as follows:

Executing the following command on server 1 will configure each device with the `device ip` of the corresponding device on server 2. For example, configure the destination IP of card 0 on server 1 as the ip of card 0 on server 2. Configuration commands require the `hccn_tool` tool. The [`hccn_tool`](https://support.huawei.com/enterprise/zh/ascend-computing/a300t-9000-pid-250702906?category=developer-documents) is an HCCL tool that comes with the CANN package.

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

`-i 0` specifies the device ID. `-netdetect` specifies the network detection object IP attribute. `-s address` indicates that the attribute is set to an IP address. `192.*.92.131` indicates the ip address of device 0 on server 2. The interface command can be referenced [here](https://support.huawei.com/enterprise/zh/doc/EDOC1100251947/8eff627f).

After executing the above command on server 1, start checking the network link status with the following command. Another function of `hccn_tool` is used here, the meaning of which can be found [here](https://support.huawei.com/enterprise/zh/doc/EDOC1100251947/7d059b59).

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

If the connection is successful, the corresponding output is as follows:

```bash
net health status: Success
```

If the connection fails, the corresponding output is as follows:

```bash
net health status: Fault
```

After confirming that the network of the NPU units between the machines is smooth, configure the json configuration file of the multi-machine, this document takes the configuration file of the 16 cards as an example. The detailed description of the configuration file can be referred to the introduction of single-machine multi-card part in this document.

It should be noted that in the configuration of the multi-machine json file, it is required that the order of rank_id is consistent with the dictionary order of server_id.

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

After preparing the configuration file, you can carry out the organization of distributed multi-machine training scripts. In the case of 2-machine 16-card, the scripts on the two machines are similar to the scripts run on single-machine multi-card, the difference being the specification of different rank_id variables.

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

During execution, the following commands are executed on the two machines:

```bash
# server0
bash run_rank_table_cluster.sh 0
# server1
bash run_rank_table_cluster.sh 8
```

Where rank_table.json follows the 16-card distributed json file reference configuration shown in this section.

After running, the log files are saved in the directories `device_0`, `device_1`. The information about the environment variables is recorded in `env*.log`, and the output is saved in `train*.log`.

#### Cross Cluster

For today's large-scale models, using compute clusters for training has become the norm. However, as model sizes continue to grow, the resources of a single cluster can no longer meet the memory requirements for model training. Therefore, support for cross-cluster communication has become a prerequisite for training ultra-large-scale models.

Currently, the HCCL communication library of Ascend hardware does not support cross-cluster communication. To address this issue, MindSpore provides a cross-cluster communication library that enables efficient communication between NPUs in different clusters. With this library, users can overcome the memory limitations of a single cluster and achieve cross-cluster parallel training for ultra-large-scale models.

Currently, the MindSpore framework enables this feature simply by adding the `cluster_list` configuration item for cross-cluster communication in the multi-node, multi-card JSON configuration file. This document uses a 2-node, 16-card setup (assuming the two machines are not in the same cluster) as an example to illustrate how to write the relevant configuration items for cross-cluster scenarios. For detailed information about the configuration file, please refer to the single-node, multi-card section in this document.

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

For cross-cluster scenarios, the parameters that need to be added or modified based on the actual training environment are as follows:

- `server_id` represents the globally unique identifier of the current machine.
- `server_ip` represents the IP address of the current machine.
- `dpu_ip` represents the virtual IP address of the card within the tenant VPC, used for cross-cluster communication.
- `numa_id` represents the NUMA-affined CPU core ID of the card on the current machine.
- `cluster_id` represents the globally unique identifier of the cluster.
- `network_type` represents the type of network between machines within the cluster, currently set to "ROCE."
- `az_id` represents the AZ (Availability Zone) ID where the cluster is located.
- `server_list` represents the list of machines included in the current cluster.

Once the configuration file is prepared, the distributed training script for cross-cluster scenarios remains consistent with the distributed training script for multi-node, multi-card setups described in this document. Using a 2-cluster, 16-card setup as an example, the scripts on the two machines in the two clusters are the same as those used in multi-node, multi-card scenarios. The only difference lies in specifying different rank_id variables.

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

During execution, the two machines in the two clusters run the following commands, respectively:

```bash
# server0
bash run_rank_table_cross_cluster.sh 0
# server1
bash run_rank_table_cross_cluster.sh 8
```

The `rank_table_cross_cluster_16pcs.json` file is configured based on the 2-cluster, 16-card cross-cluster distributed JSON file example shown in this section. The `rank_table_cross_cluster_16pcs.json` configuration used on each machine in both clusters must remain consistent.

After execution, log files are saved in the `device_0`, `device_1`, and other corresponding directories on each machine in the clusters. The `env*.log` files record information about the environment variables, while the output results are stored in the `train*.log` files.
