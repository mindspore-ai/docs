# msrun Launching

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/parallel/msrun_launcher.md)

## Overview

`msrun` is an encapsulation of the [Dynamic Cluster](https://www.mindspore.cn/tutorials/en/master/parallel/dynamic_cluster.html) startup method. Users can use `msrun` to pull multi-process distributed tasks across nodes with a single command line instruction. Users can use `msrun` to pull up multi-process distributed tasks on each node with a single command line command, and there is no need to manually set [dynamic networking environment variables](https://www.mindspore.cn/tutorials/en/master/parallel/dynamic_cluster.html). `msrun` supports both `Ascend`, `GPU` and `CPU` backends. As with the `Dynamic Cluster` startup, `msrun` has no dependencies on third-party libraries and configuration files.

> - `msrun` is available after the user installs MindSpore, and the command `msrun --help` can be used to view the supported parameters.
> - `msrun` supports `graph mode` as well as `PyNative mode`.

A parameters list of command line:

<table align="center">
    <tr>
        <th align="left">Parameters</th>
        <th align="left">Functions</th>
        <th align="left">Types</th>
        <th align="left">Values</th>
        <th align="left">Instructions</th>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--worker_num</td>
        <td align="left">The total number of Worker processes participating in the distributed task.</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">An integer greater than 0. The default value is 8.</td>
        <td align="left">The total number of Workers started on all nodes should be equal to this parameter:<br> if the total number is greater than this parameter, the extra Worker processes will fail to register; <br>if the total number is less than this parameter, the cluster will wait for a certain period of timeout before prompting the task to pull up the failed task and exit, <br>and the size of the timeout window can be configured by the parameter <code>cluster_time_out</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--local_worker_num</td>
        <td align="left">The number of Worker processes pulled up on the current node.</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">An integer greater than 0. The default value is 8.</td>
        <td align="left">When this parameter is consistent with <code>worker_num</code>, it means that all Worker processes are executed locally. <br>The <code>node_rank</code> value is ignored in this scenario.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--master_addr</td>
        <td align="left">Specifies the IP address or hostname of the Scheduler.</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">Legal IP address or hostname. The default is the IP address 127.0.0.1.</td>
        <td align="left">msrun will automatically detect on which node to pull up the Scheduler process, and users do not need to care. <br>If the corresponding IP address cannot be found or the hostname cannot be resolved by DNS, the training task will pull up and fail.<br>IPv6 addresses are not supported in the current version.<br>If a hostname is input as a parameter, msrun will automatically resolve it to an IP address, which requires the user's environment to support DNS service.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--master_port</td>
        <td align="left">Specifies the Scheduler binding port number.</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">Port number in the range 1024 to 65535. The default is 8118.</td>
        <td align="left"></td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--node_rank</td>
        <td align="left">The index of the current node.</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">An integer greater than 0. The default value is -1.</td>
        <td align="left">This parameter is ignored in single-machine multi-card scenario.<br>In multi-machine and multi-card scenarios, if this parameter is not set, the rank_id of the Worker process will be assigned automatically; <br>if it is set, the rank_id will be assigned to the Worker process on each node according to the index.<br>If the number of Worker processes per node is different, it is recommended that this parameter not be configured to automatically assign the rank_id.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--log_dir</td>
        <td align="left">Worker, and Scheduler log output paths.</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">Folder path. Defaults to the current directory.</td>
        <td align="left">If the path does not exist, msrun creates the folder recursively.<br>The log format is as follows: for the Scheduler process, the log is named <code>scheduler.log</code>; <br>For Worker process, log name is <code>worker_[rank].log</code>, where <code>rank</code> suffix is the same as the <code>rank_id</code> assigned to the Worker, <br>but they may be inconsistent in multiple-machine and multiple-card scenarios where <code>node_rank</code> is not set. <br>It is recommended that <code>grep -rn "Global rank id"</code> is executed to view <code>rank_id</code> of each Worker.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--join</td>
        <td align="left">Whether msrun waits for the Worker as well as the Scheduler to exit.</td>
        <td align="left" style="white-space:nowrap">Bool</td>
        <td align="left">True or False. Default: False.</td>
        <td align="left">If set to False, msrun will exit immediately after pulling up the process and check the logs to confirm that the distributed task is executing properly.<br>If set to True, msrun waits for all processes to exit, collects the exception log and exits.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--cluster_time_out</td>
        <td align="left">Cluster networking timeout in seconds.</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">Default: 600 seconds.</td>
        <td align="left">This parameter represents the waiting time in cluster networking. <br>If no <code>worker_num</code> number of Workers register successfully beyond this time window, the task pull-up fails.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--bind_core</td>
        <td align="left">Enable processes binding CPU cores.</td>
        <td align="left" style="white-space:nowrap">Bool/Dict</td>
        <td align="left">True/False or a device-to-CPU-range dict. Default: False.</td>
        <td align="left">If set to True, msrun will automatically allocates CPU ranges based on device affinity; when manually passing a dict, e.g., <code>{"device0":["0-10"],"device1":["11-20"]}</code>, it assigns CPU range 0-10 to process 0 (device0) and 11-20 to process 1 (device1).</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--sim_level</td>
        <td align="left">Set simulated compilation level.</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">Default: -1. Disable simulated compilation.</td>
        <td align="left">If this parameter is set, msrun starts only the processes for simulated compilation and does not execute operators. This feature is commonly used to debug large-scale distributed training parallel strategies, and to detect memory and strategy issues in advance. <br> The settings for the simulated compilation level can be found in the document: <a href="https://www.mindspore.cn/tutorials/en/master/debug/dryrun.html">DryRun</a>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--sim_rank_id</td>
        <td align="left">rank_id of the simulated process.</td>
        <td align="left" style="white-space:nowrap">Integer</td>
        <td align="left">Default: -1. Disable simulated compilation for a single process.</td>
        <td align="left">Set rank id of the simulated process.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--rank_table_file</td>
        <td align="left">rank_table configuration. Only valid on Ascend platform.</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">File path of rank_table configuration. Default: empty string.</td>
        <td align="left">This parameter represents the rank_table configuration file on Ascend platform, describing current distributed cluster. <br>Since the rank_table configuration file reflects distributed cluster information at the physical level, when using this configuration, make sure that the Devices visible to the current process are consistent with the rank_table configuration. <br>The Device visible to the current process can be set via the environment variable <code>ASCEND_RT_VISIBLE_DEVICES</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--worker_log_name</td>
        <td align="left">Specifies the worker log name.</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">File name of worker log. Default: <code>worker_[rank].log</code>.</td>
        <td align="left">This parameter represents support users configure worker log name, and support configure <code>ip</code> and <code>hostname</code> to worker log name by <code>{ip}</code> and <code>{hostname}</code> separately. <br>The suffix of worker log name is <code>rank</code> by default.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">--tail_worker_log</td>
        <td align="left">Enable output worker log to console.</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">One or multiple integers associated with the worker process rank_id. Default: -1.</td>
        <td align="left">This parameter represents output all worker logs of the current node to console by default, and supports users specify one or more worker logs output to console when <code>--join=True</code>. <br>This parameter should be in [0, local_worker_num].</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">task_script</td>
        <td align="left">User Python scripts.</td>
        <td align="left" style="white-space:nowrap">String</td>
        <td align="left">Legal script path.</td>
        <td align="left">Normally, this parameter is the python script path, and msrun will pull up the process as <code>python task_script task_script_args</code> by default.<br>msrun also supports this parameter as pytest. <br>In this scenario the task script and task parameters are passed in the parameter <code>task_script_args</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">task_script_args</td>
        <td align="left">Parameters for the user Python script.</td>
        <td align="left"></td>
        <td align="left">Parameter list.</td>
        <td align="left">For example, <code>msrun --worker_num=8 --local_worker_num=8 train.py <b>--device_target=Ascend --dataset_path=/path/to/dataset</b></code></td>
    </tr>
</table>

## Environment Variables

The following table shows the environment variables can be used in user scripts, which are set by `msrun`:

<table align="center">
    <tr>
        <th align="left">Environment Variables</th>
        <th align="left">Functions</th>
        <th align="left">Values</th>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_ROLE</td>
        <td align="left">This process role.</td>
        <td align="left">
            The current version of <code>msrun</code> exports the following two values:
            <ul>
                <li>MS_SCHED: Represents the Scheduler process.</li>
                <li>MS_WORKER: Represents the Worker process.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_SCHED_HOST</td>
        <td align="left">The IP address of the user-specified Scheduler.</td>
        <td align="left">Same as parameter <code>--master_addr</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_SCHED_PORT</td>
        <td align="left">User-specified Scheduler binding port number.</td>
        <td align="left">Same as parameter <code>--master_port</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_WORKER_NUM</td>
        <td align="left">The total number of Worker processes specified by the user.</td>
        <td align="left">Same as parameter <code>--worker_num</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">MS_TOPO_TIMEOUT</td>
        <td align="left">Cluster Timeout Time.</td>
        <td align="left">Same as parameter <code>--cluster_time_out</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">RANK_SIZE</td>
        <td align="left">The total number of Worker processes specified by the user.</td>
        <td align="left">Same as parameter <code>--worker_num</code>.</td>
    </tr>
    <tr>
        <td align="left" style="white-space:nowrap">RANK_ID</td>
        <td align="left">The rank_id assigned to the Worker process.</td>
        <td align="left">In a multi-machine multi-card scenario, if the parameter <code>--node_rank</code> is not set, <code>RANK_ID</code> will only be exported after the cluster is initialized.<br> So to use this environment variable, it is recommended to set the <code>--node_rank</code> parameter correctly.</td>
    </tr>
</table>

msrun is used as an encapsulation of the Dynamic Cluster startup method, and all user-configurable environment variables can be found in [dynamic networking environment variables](https://www.mindspore.cn/tutorials/en/master/parallel/dynamic_cluster.html).

## Launching Distributed Tasks

The startup script is consistent across hardware platforms. The following is an example of how to write a startup script for Ascend:

> You can download the full sample code here: [startup_method](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/startup_method).

The directory structure is as follows:

```text
└─ sample_code
    ├─ startup_method
       ├── msrun_1.sh
       ├── msrun_2.sh
       ├── msrun_single.sh
       ├── net.py
    ...
```

`net.py` defines the network structure and the training process. `msrun_single.sh` is a single-machine multi-card execution script that starts with `msrun`. `msrun_1.sh` and `msrun_2.sh` are multi-machine, multi-card execution scripts started with `msrun` and executed on separate nodes.

### 1. Preparing Python Training Scripts

Here is an example of data parallelism to train a recognition network for the MNIST dataset.

First specify the operation mode, hardware device, etc. Unlike single card scripts, parallel scripts also need to specify configuration items such as parallel mode and initialize the HCCL, NCCL or MCCL communication domain with `init()`. If `device_target` is not set here, it will be automatically specified as the backend hardware device corresponding to the MindSpore package.

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

### 2. Preparing the Startup Script

> For msrun, single-machine multi-card and multi-machine multi-card execution commands are similar, single-machine multi-card only needs to keep the parameters `worker_num` and `local_worker_num` the same, and single-machine multi-card scenarios do not need to set the `master_addr`, which defaults to `127.0.0.1`.

#### Single-machine Multi-card

The following is an example of performing a single-machine 8-card training session:

The script [msrun_single.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_single.sh) uses the msrun command to pull up 1 `Scheduler` process as well as 8 `Worker` processes on the current node (no need to set `master_addr`, defaults to `127.0.0.1`; no need to set `node_rank` for single-machine):

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

Execute the command:

```bash
bash msrun_single.sh
```

The single-machine 8-card distributed training task can be executed. The log file is saved to the `. /msrun_log` directory and the results are saved in `. /msrun_log/worker_*.log`. The Loss results are as follows:

```text
epoch: 0, step: 0, loss is 2.3499548
epoch: 0, step: 10, loss is 1.6682479
epoch: 0, step: 20, loss is 1.4237018
epoch: 0, step: 30, loss is 1.0437132
...
```

#### Multi-machine Multi-card

The following is an example of executing 2-machine, 8-card training, with each machine executing the startup of 4 Workers:

The script [msrun_1.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_1.sh) is executed on node 1 and uses the msrun command to pull up 1 `Scheduler` process and 4 `Worker` processes, configures `master_addr` as the IP address of node 1 (msrun automatically detects that the current node ip matches the `master_addr` and pulls up the `Scheduler` process). Set the current node to node 0 with `node_rank`:

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

The script [msrun_2.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_2.sh) is executed on node 2 and uses the msrun command to pull up 4 `Worker` processes, configures `master_addr` as the IP address of node 1. Set the current node to node 0 with `node_rank`:

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

> The difference between the instructions for node 2 and node 1 is that `node_rank` is different.

Executed at node 1:

```bash
bash msrun_1.sh
```

Executed at node 2:

```bash
bash msrun_2.sh
```

The 2-machine, 8-card distributed training task can be executed, and the log files are saved to the `. /msrun_log` directory and the results are saved in `. /msrun_log/worker_*.log`. The Loss results are as follows:

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

## Multi-Card Parallel Debugging

You can use Python built-in debugger (pdb) for parallel multi-card debugging in distributed environments, by breaking and synchronizing operations on all or a particular rank. After the `msrun` parameter is set to `--join=True` to pull up the worker processes, the standard input of all worker processes is inherited from the `msrun` master process and the standard output is output to the shell window via the `msrun` log redirection feature. Details of how to use pdb in a distributed environment are given below:

### 1. Launching the pdb Debugger

Users can start the pdb debugger in a variety of ways, such as inserting `import pdb; pdb.set_trace()` or `breakpoint()` into Python training scripts to perform breakpoint operations.

#### Python Training Script

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

#### Launching the Script

In the launching script, the `msrun` parameter needs to be set to `--join=True` to ensure that pdb commands are passed through standard input and debugging is displayed through standard output.

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 net.py
```

### 2. Debugging for rank

In a distributed environment, the user may need to debug for a particular rank, which can be accomplished by performing breakpoints on specific ranks in the training script. For example, in the stand-alone eight-card task, debugging is performed only for rank 7 with breakpoints:

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

> The `mindspore.communication.get_rank()` interface needs to be called after the `mindspore.communication.init()` interface has completed its distributed initialization to get the rank information properly, otherwise `get_rank()` returns 0 by default.

After a breakpoint operation on a rank, it will cause the execution of that rank process to stop at the breakpoint and wait for subsequent interactions, while other unbroken rank processes will continue to run, which may lead to inconsistent running speed, so you can use the `mindspore.communication.comm_func.barrier()` operator and the `mindspore.communication.api._pynative_executor.sync()` to synchronize the running of all ranks, ensuring that other ranks block and wait, and that the stops of other ranks are released once the debugging rank continues to run. For example, in a standalone 8-card task, only rank 7 is broken and all other ranks are blocked:

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

### 3. Standard Input and Standard Output for shell Terminals

`msrun` supports outputting specific worker logs to the shell's standard output via `--tail_worker_log`. To make the standard output more observable, it is recommended to use this parameter to specify the rank for which the output needs to be breakpoint debugged, e.g., in the stand-alone eight-card task, breakpoint debugging is performed only for rank 7:

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 --tail_worker_log=7 net.py
```

> - The default behavior of `msrun` without the `--tail_worker_log` parameter will output the logs of all workers on this node to the shell's standard output.
> - When debugging multiple ranks at the same time, a pdb command is passed to a rank in turn via standard input.

### 4. Common pdb Debugging Commands

- `n` (next): Execute the current line of code and jump to the next line of code.
- `s` (step): Go to the function called by the current line of code and debug it step by step.
- `c` (continue): Continue executing the program until the next breakpoint.
- `q` (quit): Exit the debugger and terminate program execution.
- `p` (print): Prints the value of a variable. For example, `p variable` displays the current value of the variable `variable`.
- `l` (list): Display the context of the current code.
- `b` (break): Set a breakpoint, either by specifying a line number or a function name.
- `h` (help): Display a help message listing all available commands.