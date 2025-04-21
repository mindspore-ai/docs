# mpirun Startup

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/mpirun.md)

## Overview

Open Message Passing Interface (OpenMPI) is an open source, high-performance message-passing programming library for parallel computing and distributed memory computing, which realizes parallel computing by passing messages between different processes for many scientific computing and machine learning tasks. Parallel training with OpenMPI is a generalized approach to accelerate the training process by utilizing parallel computing resources on computing clusters or multi-core machines. OpenMPI serves the function of synchronizing data on the Host side as well as inter-process networking in distributed training scenarios.

Unlike rank table startup, the user does not need to configure the `RANK_TABLE_FILE` environment variable to run the script via OpenMPI `mpirun` command on the Ascend hardware platform.

> The `mpirun` startup supports Ascend and GPU, in addition to both PyNative mode and Graph mode.

Related commands:

1. The `mpirun` startup command is as follows, where `DEVICE_NUM` is the number of GPUs on the machine:

    ```bash
    mpirun -n DEVICE_NUM python net.py
    ```

2. `mpirun` can also be configured with the following parameters. For more configuration, see [mpirun documentation](https://www.open-mpi.org/doc/current/man1/mpirun.1.php):

    - `--output-filename log_output`: Save the log information of all processes to the `log_output` directory, and the logs on different cards will be saved in the corresponding files under the `log_output/1/` path by `rank_id`.
    - `--merge-stderr-to-stdout`: Merge stderr to the output message of stdout.
    - `--allow-run-as-root`: This parameter is required if the script is executed through the root user.
    - `-mca orte_abort_on_non_zero_status 0`: When a child process exits abnormally, OpenMPI will abort all child processes by default. If you don't want to abort child processes automatically, you can add this parameter.
    - `-bind-to none`: OpenMPI will specify the number of available CPU cores for the child process to be pulled up by default. If you don't want to limit the number of cores used by the process, you can add this parameter.

> OpenMPI starts up with a number of `OPMI_*` environment variables, and users should avoid manually modifying these environment variables in scripts.

## Operation Practice

The `mpirun` startup script is consistent across Ascend and GPU hardware platforms. Below is a demonstration of how to write a startup script using Ascend as an example:

> You can download the full sample code here: [startup_method](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/startup_method).

The directory structure is as follows:

```text
└─ sample_code
    ├─ startup_method
       ├── net.py
       ├── hostfile
       ├── run_mpirun_1.sh
       ├── run_mpirun_2.sh
    ...
```

`net.py` is to define the network structure and training process. `run_mpirun_1.sh` and `run_mpirun_2.sh` are the execution scripts, and `hostfile` is the file to configure the multi-machine and multi-card files.

### 1. Installing OpenMPI

Download the OpenMPI-4.1.4 source code [openmpi-4.1.4.tar.gz] (https://www.open-mpi.org/software/ompi/v4.1/). Refer to [OpenMPI official website tutorial](https://www.open-mpi.org/faq/?category=building#easy-build) for installation.

### 2. Preparing Python Training Scripts

Here, as an example of data parallel, a recognition network is trained for the MNIST dataset.

First specify the operation mode, hardware device, etc. Unlike single card scripts, parallel scripts also need to specify configuration items such as parallel mode and initialize HCCL or NCCL communication via init. If you don't set `device_target` here, it will be automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

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

### 3. Preparing the Startup Script

#### Single-Machine Multi-Card

First download the [MNIST](http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip) dataset and extract it to the current folder.

Then execute the single-machine multi-card boot script, using the single-machine 8-card example:

```bash
export DATA_PATH=./MNIST_Data/train/
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout python net.py
```

The log file will be saved to the `log_output` directory and the result will be saved in the `log_output/1/rank.*/stdout` and is as follows:

```text
epoch: 0, step: 0, loss is 2.3413472
epoch: 0, step: 10, loss is 1.6298866
epoch: 0, step: 20, loss is 1.3729795
epoch: 0, step: 30, loss is 1.2199347
epoch: 0, step: 40, loss is 0.85778403
...
```

#### Multi-Machine Multi-Card

Before running multi-machine multi-card training, you first need to follow the following configuration:

1. Ensure that the same versions of OpenMPI, NCCL, Python, and MindSpore are available on each node.

2. To configure host-to-host password-free login, you can refer to the following steps to configure it:
    - Identify the same user as the login user for each host (root is not recommended);
    - Execute `ssh-keygen -t rsa -P ""` to generate the key;
    - Execute `ssh-copy-id DEVICE-IP` to set the IP of the machine that needs password-free login;
    - Execute `ssh DEVICE-IP`. If you can log in without entering a password, the above configuration is successful;
    - Execute the above command on all machines to ensure two-by-two interoperability.

After the configuration is successful, you can start the multi-machine task with the `mpirun` command, and there are currently two ways to start a multi-machine training task:

- By means of `mpirun -H`. The startup script is as follows:

    ```bash
    export DATA_PATH=./MNIST_Data/train/
    mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 --output-filename log_output --merge-stderr-to-stdout python net.py
    ```

    indicates that 8 processes are started to run the program on the machines with ip DEVICE1_IP and DEVICE2_IP respectively. Execute on one of the nodes:

    ```bash
    bash run_mpirun_1.sh
    ```

- By means of the `mpirun --hostfile` method. For debugging purposes, this method is recommended for executing multi-machine multi-card scripts. First you need to construct the hostfile file as follows:

    ```text
    DEVICE1 slots=8
    192.168.0.1 slots=8
    ```

    The format of each line is `[hostname] slots=[slotnum]`, and hostname can be either ip or hostname. The above example indicates that there are 8 cards on DEVICE1, and there are also 8 cards on the machine with ip 192.168.0.1.

    The execution script for the 2-machine 16-card is as follows, and you need to pass in the variable `HOSTFILE`, which indicates the path to the hostfile file:

    ```bash
    export DATA_PATH=./MNIST_Data/train/
    HOSTFILE=$1
    mpirun -n 16 --hostfile $HOSTFILE --output-filename log_output --merge-stderr-to-stdout python net.py
    ```

    Execute on one of the nodes:

    ```bash
    bash run_mpirun_2.sh ./hostfile
    ```

After execution, the log file is saved to the log_output directory and the result is saved in log_output/1/rank.*/stdout.
