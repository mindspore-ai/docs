# Distributed Parallel Supports Dynamic Shape

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/support_dynamic_shape_in_parallel.md)

## Overview

In sequence to sequence training tasks, the unequal length of training corpus is a typical characteristic of this task.
Especially in large-scale language model training scenarios based on the Transformer architecture, if the corpus is filled to the maximum length for training, there will be a lot of redundant calculations, wasting computing resources.
At the same time, there will also be batches during training and reasoning Scenes with dynamic changes in size. The
training of large models usually adopts distributed training based on static graphs, so distributed parallel components
under static graphs need to provide dynamic Shape capabilities.

> The parallel dynamic Shape function only supports execution under the Kernel By Kernel backend.

Related interface：

1. `class mindspore.Symbol(self, max=0, min=1, divisor=1, remainder=0, unique=False, **kawgs)`
   ：Symbol is a data structure to indicate the symbolic info of shape. For dynamic shape networks, compared with only
   setting the unknown dimensions(None) in Tensor, providing more symbolic shape info can help the framework better
   optimize the computation graph, to improve the performance of network
   execution. [Symbol API manual](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.Symbol.html).

## Basic Principles

The parallel capability in static graphs is essentially the ability to modify a single card computational graph based on
different parallel strategies before the automatic differential PASS is compiled in the front-end. Based on the existing
distributed parallel training capability in static graphs, MindSpore has built the ability to support dynamic shapes.
The dynamic Shape model in a static graph can represent the segmentation information of the dynamic Shape axis through
the `Symbol(...)` object during the graph compilation phase, and through the `set_inputs(...)` function The interface
brings information into the diagram. Compared to `None`, the `Symbol` class can represent richer dimensional
information, such as constraints, minimum/maximum Shape values, residues, etc. Based on the dynamic axis information
configured by the user, distributed parallel components will derive the reference relationships of input dynamic axes at
each layer, achieving the expression of dynamic Shape calculation graphs in static graphs.

## Practical Operation

Now, take a typical feedforward neural network model as an example, we will introduce the use of dynamic shape in static
graphs.

> You can download the complete sample code here.
>
> <https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/parallel_support_dynamic_shape>

The directory structure is as follows:

```text
└─ sample_code
    ├─ distributed_parallel_with_dynamic_shape
       ├── main.py
       └── run.sh
```

> This tutorial does not involve starting across physical nodes, as all processes are on the same node. This use case
> uses MPI to train the processes.

### Dataset Loading

Here, we use the MNIST handwriting recognition dataset and execute the `run.sh` script to automatically download,
decompress, and configure the dataset path. Please refer to the source code file for the detailed dataset loading code,
which will not be elaborated here.

### Building A Feedforward Neural Network

The feedforward neural network structure used here is a MatMul+ReLU+MatMul+ReLU+MatMul structure, and the model is
segmented except the last MatMul operator.

At the same time, use the distributed parallel interface provided by MindSpore to complete the initialization of
distributed components.

```python
import mindspore as ms
from mindspore import nn
from mindspore import Parameter

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_context(max_device_memory="28GB")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()


class Network(nn.Cell):
    """Network"""

    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = Parameter(initializer("normal", [28 * 28, 512], ms.float32))
        self.fc2_weight = Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul().shard(((2, 4), (4, 1)))
        self.relu1 = ops.ReLU().shard(((4, 1),))
        self.matmul2 = ops.MatMul().shard(((1, 8), (8, 1)))
        self.relu2 = ops.ReLU().shard(((8, 1),))
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = ops.reshape(x, (-1, 784))
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        return self.matmul3(x, self.fc3_weight)
```

### Defining Optimizers and Loss Functions

We use `SoftmaxCrossEntropyWithLogits` as the loss function, and the optimizer uses a `SGD` (Stochastic Gradient
Descent) optimizer.

```python
from mindspore import nn

optimizer = nn.SGD(net.trainable_params(), 1e-3)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(True)
```

### Building a Neural Network Training Framework Based on Dynamic Shape

The training code entry is main.py, and the dynamic Shape axis is defined through Symbol. The meaning
of `Symbol(divisor=8)` is that the value of the axis can be divided by 8. Through `set_inputs(...)` interface
of `nn.Cell` configures input information for dynamic Shapes.

Finally, the model structure, loss function, and optimizer are combined through the `Model(...)` interface, and
the `model.train(...)` interface is called to complete model training.

```python
import mindspore as ms
from mindspore import Symbol, Tensor
from mindspore.train import Accuracy, LossMonitor

s0 = Symbol(divisor=8)
input_dyn = Tensor(shape=[s0, 1, 28, 28], dtype=ms.float32)
label_dyn = Tensor(shape=[s0, ], dtype=ms.int32)
net.set_inputs(input_dyn)
loss_fn.set_inputs(input_dyn, label_dyn)

model = Model(net, loss_fn, optimizer)
model.train(5, data_set, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

### Training Shell Script Preparation

#### Start the Training Process

Execute the run.sh script to start the training process. The run.sh code is as follows:

```bash
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "=============================================================================================================="

EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout python main.py
```

> Note: If the current user is root, the mpirun command needs to add the `--allow-run-as-root` parameter.

### Viewing Execution Results

After successful execution, a training log will be generated in the current directory, such
as `log_output/1/rank.0/stdout`. Observe the loss changes in the log to see if the model converges.
