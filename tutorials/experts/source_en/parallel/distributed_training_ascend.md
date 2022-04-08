# Parallel Distributed Training Example (Ascend)

`Ascend` `Distributed Parallel` `Whole Process`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/distributed_training_ascend.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial describes how to train the ResNet-50 network in data parallel and automatic parallel modes on MindSpore based on the Ascend 910 AI processor.
> Download address of the complete sample code: <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>

The directory structure is as follow:

```text
└─sample_code
    ├─distributed_training
    │      rank_table_16pcs.json
    │      rank_table_8pcs.json
    │      rank_table_2pcs.json
    │      cell_wrapper.py
    │      model_accu.py
    │      resnet.py
    │      resnet50_distributed_training.py
    │      resnet50_distributed_training_gpu.py
    │      resnet50_distributed_training_grad_accu.py
    │      run.sh
    │      run_gpu.sh
    │      run_grad_accu.sh
    │      run_cluster.sh
```

`rank_table_16pcs.json`, `rank_table_8pcs.json` and `rank_table_2pcs.json` are the networking information files. `resnet.py`,`resnet50_distributed_training.py` , `resnet50_distributed_training_gpu.py` and `resnet50_distributed_training_grad_accu.py` are the network structure files. `run.sh` , `run_gpu.sh`, `run_grad_accu.sh` and `run_cluster.sh` are the execute scripts.

Besides, we describe the usages of hybrid parallel and semi-auto parallel modes in the sections [Defining the Network](https://www.mindspore.cn/tutorials/experts/en/master/parallel/distributed_training_ascend.html#defining-the-network) and [Distributed Training Model Parameters Saving and Loading](https://www.mindspore.cn/tutorials/experts/en/master/parallel/distributed_training_ascend.html#distributed-training-model-parameters-saving-and-loading).

## Preparations

### Downloading the Dataset

This sample uses the `CIFAR-10` dataset, which consists of color images of 32 x 32 pixels in 10 classes, with 6000 images per class. There are 50,000 images in the training set and 10,000 images in the test set.

> `CIFAR-10` dataset download address: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

Download the dataset and decompress it to a local path. The folder generated after the decompression is `cifar-10-batches-bin`.

### Configuring Distributed Environment Variables

When distributed training is performed in the bare-metal environment (compared with the cloud environment where the Ascend 910 AI processor is deployed on the local host), you need to configure the networking information file for the current multi-device environment. If the HUAWEI CLOUD environment is used, skip this section because the cloud service has been configured.

The following uses the Ascend 910 AI processor as an example. The JSON configuration file for an environment with eight devices is as follows. In this example, the configuration file is named as `rank_table_8pcs.json`. For details about how to configure the 2-device environment, see the `rank_table_2pcs.json` file in the sample code.

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

The following parameters need to be modified based on the actual training environment:

- `server_count`: number of hosts.
- `server_id`: IP address of the local host.
- `device_id`: physical sequence number of a device, that is, the actual sequence number of the device on the corresponding host.
- `device_ip`: IP address of the integrated NIC. You can run the `cat /etc/hccn.conf` command on the current host. The key value of `address_x` is the IP address of the NIC.
- `rank_id`: logical sequence number of a device, which starts from 0.

### Calling the Collective Communication Library

The Huawei Collective Communication Library (HCCL) is used for the communication of MindSpore parallel distributed training and can be found in the Ascend 310 AI processor software package. In addition, `mindspore.communication.management` encapsulates the collective communication API provided by the HCCL to help users configure distributed information.
> HCCL implements multi-device multi-node communication based on the Ascend AI processor. The common restrictions on using the distributed service are as follows. For details, see the HCCL documentation.
>
> - In a single-node system, a cluster of 1, 2, 4, or 8 devices is supported. In a multi-node system, a cluster of 8 x N devices is supported.
> - Each host has four devices numbered 0 to 3 and four devices numbered 4 to 7 deployed on two different networks. During training of 2 or 4 devices, the devices must be connected and clusters cannot be created across networks.
> - When we create a multi-node system, all nodes should use one same switch.
> - The server hardware architecture and operating system require the symmetrical multi-processing (SMP) mode.
> - Currently only supports global single group communication in PyNative mode.

The sample code for calling the HCCL is as follows:

```python
import os
from mindspore import context
from mindspore.communication import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

In the preceding code:

- `mode=context.GRAPH_MODE`: sets the running mode to graph mode for distributed training. (The PyNative mode only support data parallel running.)
- `device_id`: physical sequence number of a device, that is, the actual sequence number of the device on the corresponding host.
- `init`: enables HCCL communication and completes the distributed training initialization.

## Loading the Dataset in Data Parallel Mode

During distributed training, data is imported in data parallel mode. The following takes the CIFAR-10 dataset as an example to describe how to import the CIFAR-10 dataset in data parallel mode. `data_path` indicates the dataset path, which is also the path of the `cifar-10-batches-bin` folder.

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

Different from the single-node system, the multi-node system needs to transfer the `num_shards` and `shard_id` parameters to the dataset API. The two parameters correspond to the number of devices and logical sequence numbers of devices, respectively. You are advised to obtain the parameters through the HCCL API.

- `get_rank`: obtains the ID of the current device in the cluster.
- `get_group_size`: obtains the number of devices.

> Under data parallel mode, it is recommended to load the same dataset file for each device, or it may cause accuracy problems.

## Defining the Network

In data parallel and automatic parallel modes, the network definition method is the same as that in a single-node system. The reference code of ResNet is as follows: <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/resnet/resnet.py>

In this section we focus on how to define a network in hybrid parallel or semi-auto parallel mode.

### Hybrid Parallel Mode

Hybrid parallel mode adds the setting `layerwise_parallel` for `parameter` based on the data parallel mode. The `parameter` with the setting would be saved and computed in slice tensor and would not apply gradients aggregation. In this mode, MindSpore would not infer computation and communication for parallel operators automatically. To ensure the consistency of calculation logic, users are required to manually infer extra operations and insert them to networks. Therefore, this parallel mode is suitable for the users with deep understanding of parallel theory.

In the following example, specify the `self.weight` as the `layerwise_parallel`, that is, the `self.weight` and the output of `MatMul` are sliced on the second dimension. At this time, perform ReduceSum on the second dimension would only get one sliced result. `AllReduce.Sum` is required here to accumulate the results among all devices. More information about the parallel theory please refer to the [design document](https://www.mindspore.cn/tutorials/experts/en/master/design/distributed_training_design.html).

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

### Semi Auto Parallel Mode

Compared with the auto parallel mode, semi auto parallel mode supports manual configuration on shard strategies for network tuning. The definition of shard strategies could be referred by this [design document](https://www.mindspore.cn/tutorials/experts/en/master/design/distributed_training_design.html).

In the above example `HybridParallelNet`, the script in semi auto parallel mode is as follows. The shard stratege of `MatMul` is `((1, 1), (1, 2))`, which means `self.weight` is sliced at the second dimension.

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

> - In the semi auto parallel mode, the operators that are not assigned with any shard strategies would be executed in data parallel.
> - The auto parallel mode not only supports the parallel strategy that can automatically acquire efficient operators by strategy searching algorithms, this mode also enables users to manually assign specific parallel strategies.
> - If a parameter is used by multiple operators, each operator's shard strategy for this parameter needs to be consistent, otherwise an error will be reported.

## Defining the Loss Function and Optimizer

### Defining the Loss Function

Automatic parallelism splits models using the operator granularity and obtains the optimal parallel strategy through algorithm search. Therefore, to achieve a better parallel training effect, you are advised to use small operators to implement the loss function.

In the loss function, the `SoftmaxCrossEntropyWithLogits` is expanded into multiple small operators for implementation according to a mathematical formula. The sample code is as follows:

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
        self.div = ops.Div()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()

    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
```

### Defining the Optimizer

The `Momentum` optimizer is used as the parameter update tool. The definition is the same as that in the single-node system. For details, see the implementation in the sample code.

## Training the Network

`context.set_auto_parallel_context` is an API for users to set parallel training parameters and must be called before the initialization of networks. The related parameters are as follows:

- `parallel_mode`: parallel distributed mode. The default value is `ParallelMode.STAND_ALONE`. The other options are `ParallelMode.DATA_PARALLEL` and `ParallelMode.AUTO_PARALLEL`.
- `parameter_broadcast`: the data parallel weights on the first device would be broadcast to other devices. The default value is `False`,
- `gradients_mean`: During backward computation, the framework collects gradients of parameters in data parallel mode across multiple hosts, obtains the global gradient value, and transfers the global gradient value to the optimizer for update. The default value is `False`, which indicates that the `AllReduce.Sum` operation is applied. The value `True` indicates that the `AllReduce.Mean` operation is applied.
- You are advised to set `device_num` and `global_rank` to their default values. The framework calls the HCCL API to obtain the values.

> More about the distributed training configurations please refer to the [programming guide](https://www.mindspore.cn/tutorials/experts/en/master/parallel/auto_parallel.html).

If multiple network cases exist in the script, call `context.reset_auto_parallel_context` to restore all parameters to default values before executing the next case.

In the following sample code, the automatic parallel mode is specified. To switch to the data parallel mode, you only need to change `parallel_mode` to `DATA_PARALLEL` and do not need to specify the strategy search algorithm `auto_parallel_search_mode`. In the sample code, the recursive programming strategy search algorithm is specified for automatic parallel.

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

In the preceding code:

- `dataset_sink_mode=True`: uses the dataset sink mode. That is, the training computing is sunk to the hardware platform for execution.
- `LossMonitor`: returns the loss value through the callback function to monitor the loss function.

## Running the Script

### Single-host Training

After the script required for training is edited, run the corresponding command to call the script.

Currently, MindSpore distributed execution uses the single-device single-process running mode. That is, one process runs on each device, and the number of total processes is the same as the number of devices that are being used. For device 0, the corresponding process is executed in the foreground. For other devices, the corresponding processes are executed in the background. You need to create a directory for each process to store log information and operator compilation information. The following takes the distributed training script for eight devices as an example to describe how to run the script:

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

The variables `DATA_PATH` and `RANK_SIZE` need to be transferred to the script, which indicate the absolute path of the dataset and the number of devices, respectively.

The distributed related environment variables are as follows:

- `RANK_TABLE_FILE`: path for storing the network information file.
- `DEVICE_ID`: actual sequence number of the current device on the corresponding host.
- `RANK_ID`: logical sequence number of the current device.

For details about other environment variables, see configuration items in the installation guide.

The running time is about 5 minutes, which is mainly occupied by operator compilation. The actual training time is within 20 seconds. You can use `ps -ef | grep pytest` to monitor task processes.

Log files are saved in the `device0`,`device1`... directory. The `env.log` file records environment variable information. The `train.log` file records the loss function information. The following is an example:

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

### Multi-host Training

The previous chapters introduced the distributed training of MindSpore, which is based on the Ascend environment of a single host with multiple devices. Using multiple hosts for distributed training can greatly improve the training speed.
In the Ascend environment, the communication between NPU units across hosts is the same as the communication between each NPU unit in a single host. It is still communicated through HCCL. The difference is that the NPU units in a single host are naturally interoperable, while cross-host communication needs to be guaranteed that the networks of the two hosts are interoperable.

Execute the following command on server 1 to configure the target connect IP as the `device ip` on the server 2. For example, configure the target IP of device 0 of server 1 as the IP of device 0 of server 2. Configuration command requires the `hccn_tool` tool.
[HCCL tool](https://support.huawei.com/enterprise/en/ascend-computing/a300t-9000-pid-250702906?category=developer-documents) comes with the CANN package.

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

`-i 0` specifies the device ID. `-netdetect` specifies the IP attribute of the network detection. `-s address` means to set the property to an IP address. `192.98.92.131` represents the ip address of device 0 on the server 2. Interface commands can be found [here](https://support.huawei.com/enterprise/en/doc/EDOC1100207443/efde9769/sets-the-ip-address-of-the-network-detection-object).

After executing the above command on server 1, run the following command to start the detection of the network link status. The corresponding command can be found [here](https://support.huawei.com/enterprise/en/doc/EDOC1100207443/f6c5a628/obtains-the-status-of-a-link).

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

If the connection is normal, the corresponding output is as follows:

```bash
net health status: Success
```

If the connection fails, the corresponding output is as follows:

```bash
net health status: Fault
```

After confirming that the network of the NPU unit between the hosts is connected, configure the json configuration file of multiple hosts. This tutorial takes the configuration file of 16 devices as an example. The detailed configuration file description can refer to the introduction of the single-host multi-device part of this tutorial. It should be noted that in the json file configuration of multiple hosts, the order of rank_id is required to be consistent with the lexicographic order of server_id.

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

After preparing the configuration file, you can organize distributed multi-host training scripts. Taking 2 hosts with 16 devices as an example, the scripts written on the two hosts are similar to the running scripts of a single host with multiple devices. The difference is that different rank_id variables are specified.

```bash
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_cluster.sh DATA_PATH RANK_TABLE_FILE RANK_SIZE RANK_START"
echo "For example: bash run_cluster.sh /path/dataset /path/rank_table.json 16 0"
echo "It is better to use the absolute path."
echo "The time interval between multiple hosts to execute the script should not exceed 120s"
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

When executing, the two hosts execute the following commands respectively, among which rank_table.json is configured according to the 16-device distributed json file reference configuration shown in this chapter.

```bash
# server0
bash run_cluster.sh /path/dataset /path/rank_table.json 16 0
# server1
bash run_cluster.sh /path/dataset /path/rank_table.json 16 8
```

### Non-sink Mode Training

In graph mode, you can specify to train the model in a non-sink mode by setting the environment variable [GRAPH_OP_RUN](https://www.mindspore.cn/docs/note/en/master/env_var_list.html)=1. In this case, you need to set environment variable `HCCL_WHITELIST_DISABLE=1` and train model with OpenMPI `mpirun`. The startup script is consistent with the [GPU's distributed training](https://www.mindspore.cn/tutorials/experts/en/master/parallel/distributed_training_gpu.html#running-the-script) script.

## Distributed Training Model Parameters Saving and Loading

The below content introduced how to save and load models under the four distributed parallel training modes respectively. Before saving model parameters for distributed training, it is necessary to configure distributed environment variables and collective communication library in accordance with this tutorial.

### Auto Parallel Mode

It is convenient to save and load the model parameters in auto parallel mode. Just add configuration `CheckpointConfig` and `ModelCheckpoint` to `test_train_cifar` method in the training network steps of this tutorial, and the model parameters can be saved. It should be noted that in parallel mode, you need to specify a different checkpoint save path for the scripts running on each device to prevent conflicts when reading and writing files, The code is as follows:

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

def test_train_cifar(epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    loss_cb = LossMonitor()
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

After saving the checkpoint file, users can easily load model parameters for reasoning or retraining. For example, the following code can be used for retraining:

```python
from mindspore import load_checkpoint, load_param_into_net

net = resnet50(batch_size=32, num_classes=10)
# The parameter for load_checkpoint is a .ckpt file which has been successfully saved
param_dict = load_checkpoint(pretrain_ckpt_path)
load_param_into_net(net, param_dict)
```

For checkpoint configuration policy and saving method, please refer to [Saving and Loading Model Parameters](https://www.mindspore.cn/tutorials/experts/en/master/parallel/save_model.html#checkpoint-configuration-policies).

By default, sliced parameters would be merged before saving automatocally. However, considering large-scaled networks, a large size checkpoint file will be difficult to be transferred and loaded. So every device can save sliced parameters separately by setting `integrated_save` as `False` in `CheckpointConfig`. If the shard strategies of retraining or inference are different with that of training, the special loading way is needed.

In retraining with multiple devices scenarios, users can infer shard strategy of retraining with `model.infer_train_layout` (only dataset sink mode is supported). The shard strategy will be used as `predict_strategy` for `load_distributed_checkpoint` function, which restores sliced parameters from `strategy_ckpt_load_file` (training strategy) to `predict_strategy` (retraining strategy) and load them into `model.train_network`. If there is only one device in retraining, `predict_strategy` could be `None`. The code is as follows:

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

> Distributed inference could be referred to [Distributed inference](https://www.mindspore.cn/tutorials/experts/en/master/parallel/multi_platform_inference_ascend_910.html#id1).

### Data Parallel Mode

In data parallel mode, checkpoint is used in the same way as in auto parallel mode. You just need to change:

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
```

to:

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
```

> Under data parallel mode, we recommend to load the same checkpoint for each device to avoid accuracy problems. `parameter_broadcast` could also be used for sharing the values of parameters among devices.

### Semi Auto Parallel Mode

In semi auto parallel mode, checkpoint is used in the same way as in auto parallel mode and data parallel mode. The difference is in the definition of a network and the definition of network model, you can refer to defining the network [Semi Auto Parallel Mode](https://www.mindspore.cn/tutorials/experts/en/master/parallel/distributed_training_ascend.html#semi-auto-parallel-mode) in this tutorial.

To save the model, you can use the following code:

```python
...
net = SemiAutoParallelNet()
...
ckpt_config = CheckpointConfig()
ckpt_callback = ModelCheckpoint(prefix='semi_auto_parallel', config=ckpt_config)
```

To load the model, you can use the following code:

```python
net = SemiAutoParallelNet()
# The parameter for load_checkpoint is a .ckpt file which has been successfully saved
param_dict = load_checkpoint(pretrain_ckpt_path)
load_param_into_net(net, param_dict)
```

For the three parallel training modes described above, the checkpoint file is saved in a complete way on each device. Users also can save only the checkpoint file of this device on each device, take Semi Auto parallel Mode as an example for explanation.

Only by changing the code that sets the checkpoint saving policy, the checkpoint file of each device can be saved by itself. The specific changes are as follows:

Change the checkpoint configuration policy from:

```python
# config checkpoint
ckpt_config = CheckpointConfig(keep_checkpoint_max=1)
```

to:

```python
# config checkpoint
ckpt_config = CheckpointConfig(keep_checkpoint_max=1, integrated_save=False)
```

It should be noted that if users choose this checkpoint saving policy, users need to save and load the segmented checkpoint for subsequent reasoning or retraining. Specific usage can refer to [Integrating the Saved Checkpoint Files](https://www.mindspore.cn/tutorials/experts/en/master/parallel/save_load_model_hybrid_parallel.html#integrating-the-saved-checkpoint-files).

### Hybrid Parallel Mode

For model parameter saving and loading in Hybrid Parallel Mode, please refer to [Saving and Loading Model Parameters in the Hybrid Parallel Scenario](https://www.mindspore.cn/tutorials/experts/en/master/parallel/save_load_model_hybrid_parallel.html).
