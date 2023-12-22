# Functional Operator Shading

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/pynative_shard_function_parallel.md)

## Overview

Dynamic graphs support richer syntax and more flexible use, but currently MindSpore dynamic graph mode does not support the various features of automatic parallel. We have designed the shard function to support specify a part of the graph mode execution, and perform various parallel operations in the dynamic graph mode.

> Currently, functional operator slicing is only supported in parallel mode "auto_parallel" and strategy search algorithm "sharding_propagation".

Related interfaces:

```python
def shard(fn, in_strategy, out_strategy=None, parameter_plan=None, device="Ascend", level=0):
    return shard_fn(fn, in_strategy, out_strategy, device, level)
```

`in_strategy(tuple)`: Specify the sharding strategy of the input `Tensor`, each element is a tuple indicating the sharding strategy of the corresponding input `Tensor`, the length of each tuple should be equal to the dimension of the corresponding `Tensor`, indicating how each dimension is sliced. You can pass in `None`, the corresponding sharding strategy will be automatically deduced and generated.

`out_strategy(None, tuple)`: Specify the sharding strategy for the output `Tensor`, used in the same way as `in_strategy`, with a default value of None, which is not yet enabled and will be opened later. In deep learning models, the output strategy is replaced with data parallel (False) and repeated computation (True) depending on the value of full_batch.

`parameter_plan(None, dict)`: Specify the sharding strategy of each parameter, when passed into the dictionary, the key is the parameter name of type str, the value is a one-dimensional integer tuple indicating the corresponding sharding strategy. If the parameter name is wrong or the corresponding parameter has already set the sharding strategy, the setting of this parameter will be skipped. Default: None, means not set.

`device(string)`: Specify the device for execution, optional range `Ascend`, `GPU` and `CPU`, default is `Ascend`, which is not enabled yet, and will be opened later.

`level(int)`: Specify the search strategy for all operators, the cut strategy for input and output `Tensor` is specified by the user, and the cut strategy for the rest of operators will be obtained by the framework search, this parameter specifies the objective function for searching, with the optional range of 0, 1, 2, which represent maximizing the computational communication ratio, minimizing memory consumption, and maximizing the operation speed. The default is 0, which is currently not yet enabled, and will be opened later.

## Basic Principle

in the MindSpore dynamic graph mode, you can use the `@jit` decorator to specify a certain section to be compiled and executed in graph mode. In the forward execution at the same time, execute while the operators, subgraphs will be recorded, the forward execution is complete, will be automatically differentiated to get the whole graph to get the inverse graph. The specific process is shown in the following figure:

*Figure 1: Schematic diagram of the execution of the @jit decorator*

The Shard function follows this pattern, with the difference that operator-level model parallel can be performed at the link where the graph pattern is compiled and executed.

## Operation Practice

### Example Code Description

> You can download the complete the example code:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/pynative_shard_function_parallel>.

The directory structure is as follows:

```text
└─sample_code
    ├─shard_function_parallel
        ├── rank_table_8pcs.json
        ├── run_shard_function_example.sh
        ├── run_mpirun_shard_function_example.sh
        └── shard_function_example.py
```

The function of each file is as follows:

- `shard_function_example.py`: Sample code for the shard function, which describes how to use the shard function to specify part of the code to be executed in parallel.
- `rank_table_8pcs.json`: 8-card profile for RANK_TABLE_FILE.
- `run_shard_function_example.sh`: Startup script for shard function example.
- `run_mpirun_shard_function_example.sh`: shard function example startup script launched with mpirun.

### Importing the Relevant Packages and Setting the Execution Mode

As mentioned earlier, the shard function will execute a certain part of the operator-level model in graph mode in parallel with the dynamic graph mode, so you need to set the mode to PyNative when using the shard function:

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL,
                             search_mode="sharding_propagation", device_num=8)
init()
ms.set_seed(1)
```

### Specifying Output Arrangement

Specifying the output arrangement for data parallel and repeated calculations is currently supported, and can be controlled by the `dataset_strategy` or `full_batch` attributes in auto_parallel_context, which are set as follows:

```python
# Set via dataset_strategy, recommended
ms.set_auto_parallel_context(dataset_strategy="full_batch")  # The dataset is not sliced and the output tensor of the shard is not sliced; (default configuration)
ms.set_auto_parallel_context(dataset_strategy="data_parallel")  # The dataset is sliced in a data-parallel fashion, and the output tensor of the shard is also sliced in a data-parallel fashion

# This attribute is about to be deprecated through the full_batch setting
ms.set_auto_parallel_context(full_batch=True)   # The dataset is not sliced and the output tensor of the shard is not sliced; (default configuration)
ms.set_auto_parallel_context(full_batch=False)  # The dataset is sliced in a data-parallel fashion, and the output tensor of the shard is also sliced in a data-parallel fashion
```

### Cell Uses Functional Sharding

There are currently two ways to use the shard function, using the following network as an example to introduce the use of the shard function.

```python
import mindspore.nn as nn
class BasicBlock(nn.Cell):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.dense1 = nn.Dense(128, 256)
        self.gelu = nn.GELU()
        self.dense2 = nn.Dense(256, 128)
    def construct(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BasicBlock()
        self.block2 = BasicBlock()
        self.block3 = BasicBlock()
    def construct(self, x):
        # Here all blocks are executed in PyNative mode
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
```

- Self-call via Cell member method `shard`

    ```python
    class Net1(Net):
        def __init__(self):
            super(Net1, self).__init__()
            self.flatten = nn.Flatten()
            self.layer1 = nn.Dense(28*28, 128)
            self.layer2 = nn.Dense(128, 10)

        def construct(self, x):
            x = self.flatten(x)
            x = self.layer1(x)
            # block1 is executed in graph mode
            x = self.block1(x)
            # block2 and block3 are executed in PyNative mode
            x = self.block2(x)
            x = self.block3(x)
            x = self.layer2(x)
            return x

    net = Net1()
    # Slicing along the second dimension of the input makes the output into data parallel arrangement
    net.block1.shard(in_strategy=((1, 8),), parameter_plan={'self.block1.dense2.weight': (8, 1)})
    ```

- Using the functional interface `mindspore.shard`. Since the return value of the `shard` function is a function, you can't assign an instance of a class that has already been instantiated to the return value of `shard` when using the functional interface, because MindSpore doesn't support assigning class instances to other types.

    ```python
    class NetError(Net):
        def __init__(self):
            self.block1 = ms.shard(self.block1, in_strategy=((8, 1),),
                                    parameter_plan={'self.block1.dense2.weight': (8, 1)})

        def construct(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x
    ```

    Such an execution will encounter an error:

    ```text
    TypeError: For 'Cell', the type of block1 should be cell, but got function.
    ```

    The right usage is as follows:

    ```python
    class Net2(Net):
        def __init__(self):
            # Set the return value of the Cell instance after passing it through ms.shard to a different name
            self.block1_graph = ms.shard(self.block1, in_strategy=((8, 1),),
                                          parameter_plan={'self.block1.dense2.weight': (8, 1)})
            self.block2.shard(in_strategy=((1, 8),))

        def construct(self, x):
            # block1 is executed in graph mode and along the first dimensional slice
            x = self.block1_graph(x)
            # block2 is also executed in graph mode
            x = self.block2(x)
            # block3 is executed in PyNative mode
            x = self.block3(x)
            return x
    ```

### Function Uses Functional Sharding

function can be functionally sliced using `mindspore.shard`. Using the matmul+bias_add+relu function as an example:

```python
import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

ms.set_auto_parallel_context(dataset_strategy="full_batch") # This is an example where the dataset is not sliced and the output tensor of the shard is not sliced.

def dense_relu(x, weight, bias):
    x = ops.matmul(x, weight)
    x = ops.bias_add(x, bias)
    x = ops.relu(x)
    return x

x = Tensor(np.random.uniform(0, 1, (32, 128)), ms.float32)
weight = Tensor(np.random.uniform(0, 1, (128, 10)), ms.float32)
bias = Tensor(np.random.uniform(0, 1, (10,)), ms.float32)

# Specify the sharding strategy for x to be (4, 2), and the weight and bias sharding strategies to be set to None via in_strategy to indicate automatic derivation generation.
result = ms.shard(dense_relu, in_strategy=((4, 2), None, None))(x, weight, bias)
print('result.shape:', result.shape)
```

> Note that the initialization of parameters depends on Cell parameter management, and when the fn type passed in shard is function, its definition should not contain parameters (e.g., operations such as Conv2D, Dense, etc.).

### Running the Code

Currently MindSpore can pull up distributed parallel tasks through both multi-process startup and mpirun.

#### Multi-process Startup

Distributed parallelism can be initiated by multi-process when executing on Ascend and there is no subGroup communication.

> Model parallelism generates sub-Group communication when there are dimensions of an object that are not sliced full or for which at least two dimensions are sliced.
>
> That is, when started in this way, the communication generated by model parallelism within `shard` can only occur within `world group`, so the specified sharding strategy can currently only support slicing one dimension.

The above code needs to be configured with distributed variables before it can run. The Ascend environment needs to be configured with RANK_TABLE_FILE, RANK_ID, and DEVICE_ID. The configuration procedure can be found [here](https://www.mindspore.cn/tutorials/experts/en/master/parallel/rank_table.html).

Ascend Distributed related environment variables are:

- RANK_TABLE_FILE: Path to the grouping information file. rank_table_file files can be generated using hccl_tools.py in the models code repository, which can be obtained from [here](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).
- DEVICE_ID: the actual serial number of the current card on the machine.
- RANK_ID: logical serial number of the current card.

```bash
#!/bin/bash
set -e
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_shard_function_example.sh RANK_SIZE RANK_TABLE_FILE"
echo "For example: bash run_fusion_example.sh 8 ../rank_table_8pcs.json"
echo "It is better to use the absolute path."
echo "This example is expected to run on the Ascend environment."
echo "=============================================================================================================="
if [$# != 2]
then
    echo "Usage: bash run_shard_function_example.sh RANK_SIZE RANK_TABLE_FILE"
exit 1
fi
EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/
RANK_SIZE=$1
RANK_TABLE_FILE=$2
test_dist_8pcs()
{
    export RANK_TABLE_FILE=${RANK_TABLE_FILE}
    export RANK_SIZE=8
}
test_dist_${RANK_SIZE}pcs

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./shard_function_example.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./shard_function_example.py > train.log$i 2>&1 &
    cd ../
done
echo "The program launch succeed, the log is under device0/train.log0."
```

After configuring RANK_TABLE_FILE in the current directory, the following command requires the user to have 8 Atlas training series devices. Run the command as follows:

```bash
bash run_shard_function_example.sh 8 ../rank_table_8pcs.json
```

During the execution process, the framework automatically performs operator-level model parallelization for the input function of `shard`, and the parallelization strategy for each operator is obtained by the framework search, and the whole process is user-agnostic. The graph can be saved as follows:

```python
ms.set_context(save_graphs=2)
```

The parallelization strategy for each specific operator can be seen in `step_parallel_end.ir`.

#### Startup via mpirun

On Ascend and GPU, distributed parallel can be started by means of mpirun, and **this startup method supports the creation of subgroup communication**. The run command is as follows:

```bash
mpirun -n ${DEVICE_NUM} --output-filename log_output --allow-run-as-root python ${PYTHON_SCRIPT_PATH}
```

Take the sample code as an example, and start 8 cards, the corresponding command is:

```bash
bash run_mpirun_shard_function_example.sh
```

> Note that when startup from mpirun on Ascend with a large number of subgroups, you may run into an error that creating a communication domain fails. The error message is "Ascend collective Error: "HcclCommInitRootInfo failed. | Error Number 2". You can reduce the `max_device_memory` in `context` to reserve enough memory for hccl to create the communication domain.

### Running Results

After the running is completed, the following is the part of Loss results:

```text
epoch: 0, step: 0, loss is 2.3093076
epoch: 0, step: 100, loss is 2.299726
epoch: 0, step: 200, loss is 2.3076267
epoch: 0, step: 300, loss is 2.288056
epoch: 0, step: 400, loss is 2.2772775
epoch: 0, step: 500, loss is 2.1903486
epoch: 0, step: 600, loss is 1.2501067
epoch: 0, step: 700, loss is 0.7540306
...
```

## Usage Limitations

- The execution mode needs to be set to `PYNATIVE_MODE`, the parallel configuration to `AUTO_PARALLEL`, and the `search_mode` to `sharding_propagation`.
- Nested `vmap` use is supported, and must be used with `shard` outside and `vmap` inside.
- Nested use of `shard` is not supported.