# Distributed Training Communication Fusion

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/comm_fusion.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

In distributed parallel training scenarios to train large-scale parameter models (e.g., GPT-3, Pangu-$\alpha$), data transmission of cross-device or even cross-node is a bottleneck that limits scalability as well as operator power utilization [1]. Communication fusion is an important method to improve network resource utilization and accelerate data transmission efficiency by encapsulating the communication operator of the same source and destination nodes for simultaneous execution to avoid the extra overhead caused by multiple single operator executions.

MindSpore supports the fusion of three common communication operators (`AllReduce`, `AllGather` and `ReduceScatter`) in distributed training, and provides a simple and easy-to-use interface for user configuration. The communication fusion plays an important role in the long and steady training mission support.

## Basic Principle

This section firstly introduces the relationship between computation and communication in distributed training with the example of data parallelism, and secondly introduces the necessity of communication fusion in distributed training scenarios.

### Computation and Communication in Distributed Training

The whole process of distributed training can be roughly divided into two processes: local model computation and cross-device network data interaction.

The following is an example of data parallelism [2] to introduce the overall training process. For other parallel approaches, such as model parallelism [3], pipeline parallelism [4], please refer to related papers.

As shown in the figure below, each node backs up the complete neural network model and uses the local dataset partition to train a mini-batch for forward and backward computation. The gradient obtained from the backward computation is synchronized across the nodes, and the training of the next mini-batch continues after synchronization, and so on, until the accuracy/loss reaches a threshold, or a certain number of epochs are trained. It can be seen that computation and communication alternate in the distributed training process. Work has been done on how to do pipelining of interdependent computation and transmission to reduce the percentage of cross-node data synchronization in the overall training duration [5-6], which will not be repeated here.

![image](https://gitee.com/mindspore/docs/raw/master/tutorials/experts/source_zh_cn/parallel/images/data_parallel.png)

### The Necessity of Communication Fusion

The time overhead of network communication can be measured by the following equation, where $m$ is the size of the data transmission, $\alpha$ is the network transmission rate, and $\beta$ is the inherent overhead of network startup. As can be seen, when the number of transmitted messages becomes larger, the inherent overhead share of network shartup rises, transmitting small messages does not make efficient use of network bandwidth resources. Even communication primitives in the HPC domain, such as `AllReduce` and `AllGather`, follow this principle. Therefore, communication fusion technology can effectively improve network resource utilization and reduce network synchronization delay.

$$t = \alpha m+\beta$$

### Communication Fusion Implementation

Currently, fusion is supported for each of the three communication operators `AllReduce`, `AllGather` and `ReduceScatter`, with the configuration item being a dict type, e.g.

comm_fusion={"allreduce": {"mode": "auto", "config": None}}, where "mode" has three options:

"auto": Automatic operator fusion according to the data volume threshold of 64MB, with the configuration parameter "config" as None.

"size": Communication operator fusion is performed by manually setting the data volume threshold, with the configuration parameter "config" of type int, in MB.

"index": Only "allreduce" supports the configuration of index, which indicates the way of fusion according to the sequence number of communication operator, and the configuration parameter "config" is of type list. For example, [20, 35], means the first 20 AllReduce are fused into 1, the 20th to 35th AllReduce are fused into 1, and the remaining AllReduce are fused into 1.

### Communication Fusion Usage

MindSpore provides two interfaces to enable communication fusion, each of which is described below.

#### Configuration in the Automatic Parallel Scenario

In automatic parallel or semi-automatic parallel scenarios, users can use the `comm_fusion` parameter provided by this interface to set the parallel strategy when configuring the parallel strategy via `set_auto_parallel_context`. Users can specify whether to use the index method or the fusion buffer method.

#### Using the Interfaces Provided by `Cell`

Regardless of the parallel mode scenario, users can set the index for the parameters of a layer in the model through the `Cell.set_comm_fusion` interface, and MindSpore will fuse the parameters with the same index. In auto-parallel and semi-auto-parallel scenarios, it is recommended that the `comm_fusion` parameter be used in preference for configuration.

## Operation Practice

### Sample Code Description

> You can download the full sample code here:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_comm_fusion>.

The directory structure is as follows:

```text
└─sample_code
    ├─distributed_comm_fusion
        ├── fusion_example_cell.py
        ├── rank_table_2pcs.json
        ├── rank_table_8pcs.json
        └── run_fusion_example.sh
```

The function of each file is as follows:

- fusion_example_cell.py: Example of communication fusion by using the interface provided by `Cell`.
- rank_table_2pcs.json: 2-card configuration file of RANK_TABLE_FILE.
- rank_table_8pcs.json: 8-card configuration file of RANK_TABLE_FILE.
- run_fusion_example.sh: Startup script for communication fusion.

### Configuring the Communication Fusion

The following introduces the configuration of two usage methods through the practical sample.

#### `comm_fusion` Parameter

As shown in the following code, the `comm_fusion` parameter of the `set_auto_parallel_context` interface is used to configure the fusion mode for the `AllReduce` operator to be `auto`, implying that the fusion buffer size is set to 64MB by default.

```python
from mindspore.communication import init
from mindspore import nn
import mindspore as ms
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(comm_fusion={"allreduce": {"mode": "auto", "config": None}})
init()
```

If all similar communication operators are fused into one operator, in the current training iteration, the transmission needs to wait until the computation is completely finished before it can be executed, which will cause the device to wait.

In order to avoid the above problem, the network parameters can be fused in groups: while the next group of parameters is computed, the communication of the previous group of parameters is carried out, so that the computation and communication can be hidden from each other, to perform group fusion either by limiting the size of the fusion buffer, or by index partitioning.

For more usage, you can refer to MindSpore [test cases](https://gitee.com/mindspore/mindspore/blob/master/tests/ut/python/parallel/test_comm_fusion.py).

> Users can try the size and index modes of `comm_fusion` on their own, which are essentially methods of the fusion buffer class.

#### `Cell.set_comm_fusion` Interface

As shown in the following code, the `set_comm_fusion` method is called for the instantiated DenseLayer to set the fusion value for each layer.

```python
"""Cell Fusion Example"""
import os
from mindspore.communication import init
from mindspore import nn
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()

class DenseLayer(nn.Cell):
    """A base layer with two dense layer"""
    def __init__(self):
        super().__init__()
        self.input_mapping = nn.Dense(10, 10)
        self.output_mapping = nn.Dense(10, 10)
    def construct(self, x):
        x = self.input_mapping(x)
        return self.output_mapping(x)

class Net(nn.Cell):
    """An network with many dense layers"""
    def __init__(self):
        super().__init__()
        self.layer1 = DenseLayer()
        self.layer2 = DenseLayer()
        self.layer3 = DenseLayer()
        self.layer1.set_comm_fusion(0)
        self.layer2.set_comm_fusion(1)
        self.layer3.set_comm_fusion(2)
    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

net = Net()
for item in net.trainable_params():
    print(f"The parameter {item.name}'s fusion id is {item.comm_fusion}")
```

The corresponding output, representing the fusion index value for each layer of a particular dense, is as follows:

```text
The parameter layer1.input_mapping.weight's fusion id is 0
The parameter layer1.input_mapping.bias's fusion id is 0
The parameter layer1.output_mapping.weight's fusion id is 0
The parameter layer1.output_mapping.bias's fusion id is 0
The parameter layer2.input_mapping.weight's fusion id is 1
The parameter layer2.input_mapping.bias's fusion id is 1
The parameter layer2.output_mapping.weight's fusion id is 1
The parameter layer2.output_mapping.bias's fusion id is 1
The parameter layer3.input_mapping.weight's fusion id is 2
The parameter layer3.input_mapping.bias's fusion id is 2
The parameter layer3.output_mapping.weight's fusion id is 2
The parameter layer3.output_mapping.bias's fusion id is 2
```

### Running the Code

The above code needs to be configured with distributed variables before it can run. The Ascend environment needs to be configured with RANK_TABLE_FILE, RANK_ID and DEVICE_ID. For the configuration process, refer to [here](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#configuring-distributed-environment-variables). The GPU environment needs to be configured with [OpenMPI](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#configuring-distributed-environment), NCCL and [HOST_FILE](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#multi-host-training). For the configuration process, refer to [here](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#configuring-distributed-environment).

Environment variables related to Ascend distributed are:

- RANK_TABLE_FILE: the path of networking information file. The rank_table_file file can be generated by using hccl_tools.py in the models code repository, which can be obtained from [here](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).
- DEVICE_ID: The actual serial number of the current card on the machine.
- RANK_ID: The logical serial number of the current card.

Environment variables related to GPU distributed are:

- HOST_FILE: describes the IP and number of devices for multi-card training. Each line of the file has the format [hostname] slots=[slotnum], and hostname can be an ip or hostname. Note that the username needs to be the same on different machines, but the hostname cannot be the same.

The user can access the above script in this document via [here](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_optimizer_parallel). Execute the following `bash` script to run the program and output the log in the device0/train.log0 file.

```bash
#!/bin/bash
set -e
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_fusion_example.sh DATA_PATH RANK_SIZE"
echo "For example: bash run_fusion_example.sh 8"
echo "It is better to use the absolute path."
echo "This example is expected to run on the Ascend environment."
echo "=============================================================================================================="
RANK_SIZE=$1

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

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./fusion_example_cell.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ./fusion_example_cell.py > train.log$i 2>&1 &
    cd ../
done
echo "The program launch succeed, the log is under device0/train.log0."
```

After configuring RANK_TABLE_FILE in the current directory, the following command requires the user to have 8 Ascend 910 devices. Run the command as follows:

```bash
bash run_fusion_example.sh 8
```

## References

[1] Xu Y, Lee H J, Chen D, et al. GSPMD: general and scalable parallelization for ML computation graphs[J]. arXiv preprint arXiv:2105.04663, 2021.

[2] Li M, Zhou L, Yang Z, et al. Parameter server for distributed machine learning[C]//Big learning NIPS workshop. 2013, 6: 2.

[3] Dean J, Corrado G, Monga R, et al. Large scale distributed deep networks[J]. Advances in neural information processing systems, 2012, 25.

[4] Narayanan D, Harlap A, Phanishayee A, et al. PipeDream: generalized pipeline parallelism for DNN training[C]//Proceedings of the 27th ACM Symposium on Operating Systems Principles. 2019: 1-15.

[5] Zhang H, Zheng Z, Xu S, et al. Poseidon: An efficient communication architecture for distributed deep learning on {GPU} clusters[C]//2017 USENIX Annual Technical Conference (USENIX ATC 17). 2017: 181-193.

[6] Peng Y, Zhu Y, Chen Y, et al. A generic communication scheduler for distributed dnn training acceleration[C]//Proceedings of the 27th ACM Symposium on Operating Systems Principles. 2019: 16-29.

