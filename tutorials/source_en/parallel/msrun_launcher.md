# msrun Launching

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_en/parallel/msrun_launcher.md)

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
        <td align="left">If set to True, msrun will automatically allocate CPU ranges based on device affinity; if a dictionary is manually passed, CPU binding will be performed according to the CPU ranges allocated in the dictionary. For specific configurations, please refer to the **Process-Level CPU Binding** section.</td>
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

> You can download the full sample code here: [startup_method](https://atomgit.com/mindspore/docs/tree/master/docs/sample_code/startup_method).

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

The script [msrun_single.sh](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_single.sh) uses the msrun command to pull up 1 `Scheduler` process as well as 8 `Worker` processes on the current node (no need to set `master_addr`, defaults to `127.0.0.1`; no need to set `node_rank` for single-machine):

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

The single-machine 8-card distributed training task can be executed. The log file is saved to the `./msrun_log` directory and the results are saved in `./msrun_log/worker_*.log`. The Loss results are as follows:

```text
epoch: 0, step: 0, loss is 2.3499548
epoch: 0, step: 10, loss is 1.6682479
epoch: 0, step: 20, loss is 1.4237018
epoch: 0, step: 30, loss is 1.0437132
...
```

#### Multi-machine Multi-card

The following is an example of executing 2-machine, 8-card training, with each machine executing the startup of 4 Workers:

The script [msrun_1.sh](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_1.sh) is executed on node 1 and uses the msrun command to pull up 1 `Scheduler` process and 4 `Worker` processes, configures `master_addr` as the IP address of node 1 (msrun automatically detects that the current node ip matches the `master_addr` and pulls up the `Scheduler` process). Set the current node to node 0 with `node_rank`:

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

The script [msrun_2.sh](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/startup_method/msrun_2.sh) is executed on node 2 and uses the msrun command to pull up 4 `Worker` processes, configures `master_addr` as the IP address of node 1. Set the current node to node 0 with `node_rank`:

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

The 2-machine, 8-card distributed training task can be executed, and the log files are saved to the `./msrun_log` directory and the results are saved in `./msrun_log/worker_*.log`. The Loss results are as follows:

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

init()
if get_rank() == 7:
    pdb.set_trace()
ms.set_seed(1)
```

> The [mindspore.communication.get_rank()](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.get_rank.html) interface needs to be called after the [mindspore.communication.init()](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.init.html) interface has completed its distributed initialization to get the rank information properly, otherwise `get_rank()` returns 0 by default.

After a breakpoint operation on a rank, it will cause the execution of that rank process to stop at the breakpoint and wait for subsequent interactions, while other unbroken rank processes will continue to run, which may lead to inconsistent running speed, so you can use the [mindspore.ops.communication.barrier()](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.communication.barrier.html) operator and the [mindspore.runtime.synchronize()](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.synchronize.html) to synchronize the running of all ranks, ensuring that other ranks block and wait, and that the stops of other ranks are released once the debugging rank continues to run. For example, in a standalone 8-card task, only rank 7 is broken and all other ranks are blocked:

```python
import pdb
import mindspore as ms
from mindspore.communication import init, get_rank
from mindspore.ops.communication import barrier
from mindspore.runtime import synchronize

init()
if get_rank() == 7:
    pdb.set_trace()
barrier()
synchronize()
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

## Process-Level CPU/NUMA Affinity Configuration

`msrun` provides the `--bind_core` and `--bind_numa` parameters, which invoke the `taskset` and `numactl` system commands respectively to restrict the CPU core running range and NUMA node binding relationship of a process at startup. Both parameters support automatic allocation policy and user-defined policy.

### --bind_core (CPU Affinity Configuration)

Key invocation command: `taskset -c CPUA-CPUB python XXX.py`, which restricts the Python process to run on CPU cores ranging from `CPUA` to `PUB`.

#### 1. Automatic Core Binding (`--bind_core=True`)

- **Core Allocation Logic**: No need to manually specify core numbers; CPU cores are automatically allocated based on environmental information (CPU resources, NUMA nodes, device affinity):

    - Priority is given to CPU cores in the affinity pool; if the cores in the affinity pool are insufficient, cores in the non-affinity pool will be used.
    - It relies on commands such as `lscpu` and `npu-smi` to obtain hardware information. If command execution fails, allocation will be performed based only on available CPU resources.
    - The method for obtaining the affinity relationship between CPUs and NPUs is consistent with the MindSpore interface `mindspore.runtime.set_cpu_affinity`, which can be referred to [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html).

#### 2. Custom Core Binding

- **Format Requirement**: Pass a dictionary in JSON format, which needs to be wrapped with `''` around `{}` in the shell environment.

- **Parameter Specifications**:

    - The `key` of the dictionary supports `scheduler` (scheduling process) or `deviceX` (device process, where `X` is the device number).
    - The `value` of the dictionary is a list of CPU core range segments (e.g., `["0-9", "20-29"]`). An empty list means skipping core binding for the process.

- **Example Explanation**:

    ```bash
    --bind_core='{"scheduler":["0-9"], "device0":["10-19"], "device1":["20-29", "40-49"]}'
    ```

    Meaning:

    - Allocate CPU cores 0-9 to the `scheduler` process.
    - Allocate CPU cores 10-19 to the worker process 0 (corresponding to `device0`).
    - Allocate CPU cores 20-29 and 40-49 to the worker process 1 (corresponding to `device1`).

- **Notes**:

    1. The process number must match the device number. For example, if `ASCEND_RT_VISIBLE_DEVICES=6,7` is configured so that process 0 corresponds to `device6` and process 1 corresponds to `device7`, the `key` in the configuration must use `device6` and `device7` to ensure effective core binding:

        ```bash
        --bind_core='{"scheduler":["0-9"], "device6":["10-19"], "device7":["20-29", "40-49"]}'
        ```

        The scheduler process does not occupy device resources, so it does not participate in device sorting. The order of keys does not affect their effectiveness (for example, the order of `scheduler` and `device6` in the above example can be interchanged).
    2. If the list of CPU range segments is empty, the affinity setting for that process is skipped. For example:

        ```bash
        --bind_core='{"scheduler":[], "device0":[], "device1":["20-29", "40-49"]}'
        ```

        An empty list for `scheduler` or `device0` means core binding is not performed for those processes.
    3. It is recommended that the number of worker processes be consistent with the number of key-value pairs in `--bind_core`. For example, in a single-machine two-devices task, if only core binding for worker process 1 is required, all processes (including those not needing core binding) must be explicitly configured:

        ```bash
        # correct example
        --bind_core='{"scheduler":[], "device0":[], "device1":["20-29", "40-49"]}'

        # wrong example
        --bind_core='{"device1":["20-29", "40-49"]}'
        ```

        In the wrong example, worker process 0 may be mistakenly identified as corresponding to `device1` and thus have core binding skipped. The `scheduler` and worker process 1 will also be skipped because they are not included in the configuration.

#### 3. Disabling Core Binding (`--bind_core=False`)

Do not enable process-level CPU affinity setting; this is the default configuration.

### --bind_numa (NUMA Affinity Configuration)

Key invocation command: `numactl --membind NUMAX --cpunodebind NUMAX python XXX.py`, which binds the memory area of the Python process to NUMA node X and restricts the process to run on the CPU cores corresponding to NUMA node X.

#### 1. Automatic NUMA Binding (`--bind_numa=True`)

- **Core Allocation Logic**: No need to manually specify node numbers; NUMA nodes are automatically allocated based on environmental information:

    - Requires the number of NUMA nodes ≥ the number of started processes (to ensure each process exclusively occupies one node); otherwise, the NUMA binding function cannot be enabled.
    - Priority is given to NUMA nodes with device affinity; if multiple processes are affine to the same NUMA node, non-affine nodes will be used.
    - It relies on commands such as `lscpu` and `npu-smi` to obtain hardware information. If command execution fails, allocation will be performed based only on available NUMA resources.
    - The method for acquiring the affinity relationship between NUMAs and NPUs is consistent with that of `--bind_core` and the MindSpore API `mindspore.runtime.set_cpu_affinity`.

#### 2. Custom NUMA Binding

- **Format Requirement**: Pass a dictionary in JSON format, which needs to be wrapped with `''` around `{}` in the shell environment.

- **Parameter Specifications**:

    - The `key` of the dictionary supports `scheduler` (scheduling process) or `deviceX` (device process, where `X` is the device number).
    - The `value` of the dictionary is a list of NUMA nodes, which can be a single positive integer, multiple positive integers separated by commas, or a range segment (e.g., `["0", "1,2","3-4"]`). An empty list means skipping NUMA binding for the process.

- **Example Explanation**:

    ```bash
    --bind_numa='{"scheduler":["0"], "device0":["1,2"], "device1":["3-4"]}'
    ```

    Meaning:

    - Allocate NUMA node 0 to the scheduler process;
    - Allocate NUMA nodes 1 and 2 to the worker process 0;
    - Allocate NUMA nodes 3 and 4 to the worker process 1.

- **Notes**:

    The format specifications of the custom configuration dictionary passed to `--bind_numa` are consistent with those of `--bind_core`.

#### 3. Disable NUMA Binding (`--bind_numa=False`)

Do not enable process-level NUMA affinity setting; this is the default configuration.

#### 4. JSON File Configuration (`--bind_numa=PATH_TO_JSON.json`)

- **Format Requirement**: Provide the absolute path to a JSON file that describes CPU/NUMA binding.

- **Example**:

    Launch command:

    ```bash
    msrun --bind_numa=<json>
    ```

    `<json>` file example:

    ```json
    {
      "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa"},
      "bind_cpu": {"scheduler": {"main": "20-29"}, "device0": {"main": "0-9"}, "device1": {"main": "10-19"}},
      "bind_memory": {"scheduler": 2, "device0": 0, "device1": 1}
    }
    ```

    Meaning:

    - Bind CPU 20-29 for the `scheduler` process, CPU 0-9 for worker 0 (`device0`), and CPU 10-19 for worker 1 (`device1`).
    - Bind NUMA node 2 for the `scheduler` process, NUMA node 0 for worker 0 (`device0`), and NUMA node 1 for worker 1 (`device1`).

- **Notes**:

    For detailed JSON configuration guidance, see the section **"Using JSON to unify CPU/NUMA affinity"** below.

### Using --bind_numa and --bind_core Together

When `--bind_numa` and `--bind_core` are used simultaneously, the process startup command will be `numactl --membind NUMAX --physcpubind CPUA-CPUB`. In other words, `--bind_numa` controls memory area binding based on the NUMA architecture, while `--bind_core` sets the CPU affinity of the process at the granularity of CPU cores.

### Using JSON to Unify CPU/NUMA Affinity (`--bind_numa` / `mindspore.runtime.set_cpu_affinity`)

`msrun --bind_numa` and `mindspore.runtime.set_cpu_affinity` accept a unified JSON file for CPU/memory binding.

#### 1. Overview

The unified bind JSON describes both **process-level** and **thread-level** policies:

- **Process-level binding**: `msrun --bind_numa=<json>` launches scheduler/worker processes with `taskset` or `numactl` based on the JSON.
- **Thread-level binding**: `mindspore.runtime.set_cpu_affinity(True, bind_file=<json>)` binds `main/runtime/minddata/pynative threads` based on the JSON.

Together:

- Process-level: bind process/main threads to CPUs and memory NUMA nodes.
- Thread-level: bind specific module threads to CPU ranges.

#### 2. JSON File Structure

The unified JSON file is an object containing the following fields:

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

- 2.1 bind_config

    - `bind_cpu_mode`: CPU binding mode:
        - `"cpu"`: bind by CPU lists.
        - `"numa"`: bind by NUMA nodes.
        - `"none"`: no CPU binding.
    - `bind_memory_mode`: memory binding mode:
        - `"numa"`: bind by NUMA nodes.
        - `"none"`: no memory binding.
    - `actor_thread_fix_bind`: optional bool.
        - `true`: fixed binding for runtime threads. For example, if runtime range is `"5-9"` for `device0`, each runtime actor thread binds to one CPU in order (actor_thread0->CPU5, actor_thread1->CPU6, etc.).
        - `false`: non-fixed binding. For runtime range `"5-9"`, each runtime actor thread binds to the whole range `"5-9"`.

- 2.2 bind_cpu

    - When `bind_cpu_mode="cpu"`:
        - Keys: `deviceX` or `scheduler`.
        - Values: object mapping module -> CPU range.
        - Modules: `main` / `runtime` / `pynative` / `minddata`.
        - CPU ranges as strings, e.g. `"0-4"`, `"0,2,4"`, `"0-3,8-11"`.

    - When `bind_cpu_mode="numa"`:
        - Keys: `deviceX` or `scheduler`.
        - Values: NUMA node id (int or string range), e.g. `0`, `"0"`, `"0-1,3"`.

- 2.3 bind_memory

    - Effective only when `bind_memory_mode="numa"`.
    - Keys: `deviceX` or `scheduler`.
    - Values: NUMA node id (int or string range), e.g. `0`, `"0"`, `"0-1,3"`.

> `deviceX` refers to the **physical device id**. If `ASCEND_RT_VISIBLE_DEVICES` is set, use the physical ids from that list. Example: `ASCEND_RT_VISIBLE_DEVICES=3,5` → use `device3` and `device5`.

#### 3. Behavior of `msrun --bind_numa`

- `bind_cpu_mode="cpu"`:

    - Process main uses `taskset -c <main>`;
    - If `bind_memory_mode="numa"`, uses `numactl --membind <node> --physcpubind <main>` to bind CPU and memory.

- `bind_cpu_mode="numa"`:

    - Uses `numactl --cpunodebind <node>`;
    - If `bind_memory_mode="numa"`, uses `numactl --membind <node> --cpunodebind <node>` to bind CPU and memory.

- `bind_cpu_mode="none"`:

    - No CPU binding.

- `bind_memory_mode="numa"`:

    - Memory binding via `numactl --membind <node>`.

#### 4. Behavior of `set_cpu_affinity`

Usage:

```python
mindspore.runtime.set_cpu_affinity(True, bind_file="/path/to/bind.json")
```

See the API reference: [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/en/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html).

Rules:

- Only active when `bind_cpu_mode="cpu"`.
- Uses `deviceX` fields (`main/runtime/minddata/pynative`) to bind threads.
- If msrun is used, `main` is already bound at process level and is usually not re-bound.

Consistency checks:

- msrun exports `MSRUN_BIND_FILE` and a hash of the file.
- `set_cpu_affinity` requires the same file content; otherwise it errors.

#### 5. Auto-generate JSON

Use the script [gen_bind_json.py](https://atomgit.com/mindspore/docs/blob/master/docs/sample_code/set_affinity/gen_bind_json.py) to generate a unified JSON file:

```bash
python gen_bind_json.py -o bind.json
```

Features:

- Detects device count, CPU/NUMA, and NPU-NUMA affinity.
- Falls back to equal distribution when affinity detection fails.
- Defaults: `bind_cpu_mode=cpu`, `bind_memory_mode=numa`, `actor_thread_fix_bind=True`.
- Scheduler binds CPU only (no memory binding).

Common options:

- `--device-ids`: specify device ids (e.g. `0,1,2`).
- `--device-count`: device count when auto-detection fails.
- `--runtime-range`: runtime relative CPU range (default `4-8`).
- `--minddata-range`: minddata relative CPU range (default `9-12`).
- `--main-range`: main relative CPU range (default `13-19`).
- `--pynative-range`: optional pynative relative CPU range (default unset).
- `--scheduler-range`: scheduler relative CPU range (default `20-23`).
- `--scheduler-base`: scheduler base CPU list:

    - `free`: use CPUs not assigned to devices; fallback to global if insufficient.
    - `global`: use the full available CPU list.
    - `device0`: use device0's CPU list.

For further customization (additional modules or custom NUMA strategies), extend the JSON or adjust the generator ranges.

#### 6. Examples

##### Example 1: CPU binding + NUMA memory binding (process + threads)

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

Binding details:

- Process-level (cpu): `scheduler`, `device0`, `device1` `main` use `numactl --physcpubind`.
- Thread-level (cpu): `runtime` / `pynative` / `minddata` are bound by `set_cpu_affinity`.
- Memory (numa): `device0`/`device1`/`scheduler` bind to the specified NUMA nodes.

##### Example 2: NUMA CPU + memory binding (process-only)

```json
{
  "bind_config": {"bind_cpu_mode": "numa", "bind_memory_mode": "numa"},
  "bind_cpu": {"device0": 0, "device1": 1},
  "bind_memory": {"device0": 0, "device1": 1, "scheduler": "2-3,4"}
}
```

Binding details:

- Process-level (numa): `device0`/`device1` use `numactl --cpunodebind`.
- Thread-level: not applied (`bind_cpu_mode=numa`).
- Memory (numa): `device0`/`device1`/`scheduler` bind via `--membind`.

##### Example 3: CPU binding without memory binding

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

Binding details:

- Process-level (cpu): `scheduler`/`device0`/`device1` `main` use `taskset -c`.
- Thread-level (cpu): `runtime`/`pynative`/`minddata` are bound by `set_cpu_affinity`.
- Memory: not bound (`bind_memory_mode=none`).

##### Example 4: Main-only CPU binding (no module binding)

```json
{
  "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa"},
  "bind_cpu": {
    "device0": {"main": "0-4"},
    "device1": {"main": "20-24"}
  }
}
```

Binding details:

- Process-level (cpu): only `device0`/`device1` `main` use `taskset -c`.
- Thread-level: not bound (no module ranges).
- Memory: not bound (no `bind_memory` entries).

##### Example 5: NUMA CPU binding only (no memory binding)

```json
{
  "bind_config": {"bind_cpu_mode": "numa", "bind_memory_mode": "none"},
  "bind_cpu": {"device0": 0, "device1": 1}
}
```

Binding details:

- Process-level (numa): `device0`/`device1` use `numactl --cpunodebind`.
- Thread-level: not bound (`bind_cpu_mode=numa`).
- Memory: not bound (`bind_memory_mode=none`).

##### Example 6: Memory-only binding (no CPU binding)

```json
{
  "bind_config": {"bind_cpu_mode": "none", "bind_memory_mode": "numa"},
  "bind_memory": {"device0": 0, "device1": 1}
}
```

Binding details:

- Process-level (numa): memory only via `numactl --membind`.
- CPU: not bound (`bind_cpu_mode=none`).
- Thread-level: not bound.

##### Example 7: Module-only CPU binding (no main)

```json
{
  "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "none"},
  "bind_cpu": {
    "device0": {"runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
    "device1": {"runtime": "25-29", "pynative": "30-34", "minddata": "35-39"}
  }
}
```

Binding details:

- Process-level: not bound (no `main`, so no `taskset/numactl` prefix).
- Thread-level (cpu): `runtime`/`pynative`/`minddata` are bound by `set_cpu_affinity`.
- Memory: not bound (`bind_memory_mode=none`).
