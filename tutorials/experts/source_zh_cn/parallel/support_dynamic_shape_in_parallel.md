# 分布式并行支持动态Shape

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_zh_cn/parallel/support_dynamic_shape_in_parallel.md)

## 概述

在序列到序列训练任务中，训练语料长度不等长是该任务的典型特点。特别在基于Transformer架构下的大规模语言模型训练场景下，若将语料填充到最大长度进行训练，会存在大量的冗余计算，浪费算力资源。同时，训练与推理时，也会存在Batch Size动态变化的场景。大模型的训练通常采用基于静态图的分布式训练，因此静态图下分布式并行组件需要提供动态Shape能力。

> 并行动态Shape功能仅支持在Kernel By Kernel后端下执行。

相关接口：

- `class mindspore.Symbol(self, max=0, min=1, divisor=1, remainder=0, unique=False, **kawgs)`
   ：符号，用来传递张量形状的符号信息（symbolic shape）的数据结构。 对于动态shape网络，相比只设置 shape
   的未知维度（None），提供未知维度的数学符号信息能帮助框架更好地优化计算图，提高网络执行性能。该接口详细介绍请参考[Symbol API文档](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/mindspore.Symbol.html)。

## 基本原理

静态图下的并行能力本质上是在前端编译的自动微分PASS前，根据不同的并行策略对单卡计算图进行修改，基于现有静态图下的分布式并行训练能力基础上，MindSpore构建了支持动态Shape的能力。
静态图下的动态Shape模型，在图编译阶段可以通过`Symbol(...)`对象表示动态Shape轴的切分信息，通过`set_inputs(...)`
接口将信息带入图中。相比于`None`，`Symbol`类可以表示更加丰富的维度信息，如该维度的约束、最小/最大Shape值、余数等。
基于用户配置的动态轴信息，分布式并行组件会推导各层输入动态轴的引用关系，实现静态图下的动态Shape计算图表达。

## 操作实践

接下来，以典型的前馈神经网络模型为例介绍静态图下动态shape使用。

> 您可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/parallel_support_dynamic_shape>

目录结构如下：

```text
└─ sample_code
    ├─ distributed_parallel_with_dynamic_shape
       ├── main.py
       └── run.sh
```

> 此教程不涉及跨物理节点启动，所有进程都在同一节点，本用例使用MPI进行训练进程。

### 数据集加载

这里我们使用MNIST手写体识别数据集，执行`run.sh`脚本即可自动下载、解压、配置数据集路径。 详细数据集加载代码请见源码文件，这里不进行赘述。

### 构建前馈神经网络网络

这里使用的前馈神经网络结构为MatMul+ReLU+MatMul+ReLU+MatMul的结构，并对除最后一个MatMul算子进行模型并行的切分。

同时，使用MindSpore提供的分布式并行接口，完成分布式组件的初始化。

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

### 定义优化器和损失函数

损失函数我们使用SoftmaxCrossEntropyWithLogits，优化器使用随机梯度下降优化器。

```python
from mindspore import nn

optimizer = nn.SGD(net.trainable_params(), 1e-3)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(True)
```

### 构建基于动态Shape的神经网络训练框架

训练代码入口为main.py，通过Symbol定义动态Shape轴，`Symbol(divisor=8)`
的含义为该轴的Shape可以被8整除（可以按8切分）。通过`nn.Cell`的`set_inputs(...)`接口配置动态Shape的输入信息。

最终，通过`Model(...)`接口将模型结构、损失函数、优化器组合在一起，调用`model.train(...)`接口完成模型训练。

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

### 训练Shell脚本准备

#### 启动训练进程

执行run.sh脚本，即可启动训练进程，run.sh代码如下：

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

> 注意：如果当前用户为root用户，mpirun命令需要增加`--allow-run-as-root`参数。

### 查看执行结果

执行成功后，会在当前目录下生成训练日志，如：`log_output/1/rank.0/stdout`，观测日志中Loss变化，可以查看模型是否收敛。
