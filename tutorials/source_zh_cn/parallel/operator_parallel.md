# 算子级并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/parallel/operator_parallel.md)

## 简介

随着深度学习的发展，网络模型正变得越来越大，如NLP领域已出现万亿级参数量的模型，模型容量远超单个设备的内存容量，导致单卡或数据并行均无法进行训练。算子级并行将网络模型中每个算子涉及到的张量进行切分，并分配到多个设备上，降低单个设备的内存消耗，从而使大模型的训练成为可能。

MindSpore提供两种粒度的算子级并行能力：算子级并行和高阶算子级并行。算子级并行通过简单切分策略描述张量维度分布，满足大多数场景需求。高阶算子级并行通过开放设备排布描述，支持复杂切分场景。两种粒度的算子级并行能力均同时支持ops和mint算子，本章将分别介绍基于ops和mint算子的算子级并行和高阶算子级并行实践。

## 算子级并行实践

### ops算子并行实践

以Ascend单机8卡为例，进行ops算子并行操作说明。

#### 样例代码说明

> 下载完整的样例代码：[distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/distributed_operator_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── distributed_operator_parallel.py
       ├── run.sh
       └── ...
    ...
```

其中，`distributed_operator_parallel.py`是算子级并行定义网络结构和训练过程的脚本，`run.sh`是算子级并行执行脚本。

#### 配置分布式环境

与单卡脚本不同，并行脚本还需通过`init`接口初始化通信域。此外，通过`set_memory`接口的`max_size`限制模型最大可用的设备内存，可以在Ascend硬件平台上给通信留下足够的设备内存。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="28GB")
init()
ms.set_seed(1)
```

#### 数据集加载

在算子级并行场景下，数据集加载方式与单卡加载方式一致，代码如下：

```python
import os
import mindspore.dataset as ds

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
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
```

#### 定义网络

在当前算子级并行模式下，需要用ops算子(Primitive)定义网络。用户可以在单卡网络的基础上手动配置一些算子的切分策略，例如配置策略后的网络结构为：

```python
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul().shard(((2, 4), (4, 1)))
        self.relu1 = ops.ReLU().shard(((4, 1),))
        self.matmul2 = ops.MatMul().shard(((1, 8), (8, 1)))
        self.relu2 = ops.ReLU().shard(((8, 1),))
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

```

以上网络的`ops.MatMul()`和`ops.ReLU()`算子都配置了切分策略，以`ops.MatMul().shard(((2, 4), (4, 1)))`为例，它的切分策略为：第一个输入的行切分2份，列切分4份；第二个输入的行切分4份；对于`ops.ReLU().shard(((8, 1),))`，它的切分策略为：第一个输入的行切分8份。需要注意的是，此处的两个`ops.ReLU()`的切分策略不同，一个是`ops.ReLU().shard(((4, 1),))`，一个是`ops.ReLU().shard(((8, 1),))`，所以要定义两次。

#### 训练网络定义

在这一步，我们需要定义损失函数、优化器以及训练过程。需要注意的是，由于大模型的参数量巨大，在单卡上定义网络时如果进行参数初始化，显存将远远不够。因此在定义网络时需要配合`no_init_parameters`接口进行延迟初始化，将参数初始化延迟到并行多卡阶段。这里包括网络和优化器的定义都需要延后初始化。

```python
from mindspore.nn.utils import no_init_parameters

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)

loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

```

#### 并行配置

需要进一步设置并行有关的配置，指定并行模式`semi_auto`为半自动并行模式。

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_step, parallel_mode="semi_auto")
```

#### 训练循环

这一步进行训练循环，外层循环是训练的epoch数，内层循环遍历数据集，调用`parallel_net`进行训练并获得损失值。

```python
for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = parallel_net(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

#### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，日志文件保存到`log_output`目录下，其中部分文件目录结构如下：

```text
└─ log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

结果保存在`log_output/worker_*.log`中，示例如下：

```text
epoch: 0 step: 0, loss is 2.3016002
epoch: 0 step: 10, loss is 2.2889402
epoch: 0 step: 20, loss is 2.2843816
epoch: 0 step: 30, loss is 2.248126
epoch: 0 step: 40, loss is 2.1581488
epoch: 0 step: 50, loss is 1.8051043
...
```

其他启动方式如`mpirun`、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/startup_method.html)。

### mint算子并行实践

以Ascend单机8卡为例，进行mint算子并行操作说明。

#### 样例代码说明

> 下载完整的样例代码：[distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/distributed_operator_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── distributed_mint_operator_parallel.py
       ├── run_mint.sh
       └── ...
    ...
```

其中，`distributed_mint_operator_parallel.py`是mint算子并行定义网络结构和训练过程的脚本，`run_mint.sh`是mint算子并行执行脚本。

#### 配置分布式环境

与单卡脚本不同，并行脚本还需通过`init`接口初始化通信域。此外，通过`set_memory`接口的`max_size`限制模型最大可用的设备内存，可以在Ascend硬件平台上给通信留下足够的设备内存。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="28GB")
init()
ms.set_seed(1)
```

#### 数据集加载

在mint算子并行场景下，数据集加载方式与ops算子并行的加载方式一致，代码如下：

```python
import os
import mindspore.dataset as ds

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
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
```

#### 定义网络

在当前mint算子并行模式下，需要用mint算子定义网络。由于mint算子作为函数式接口，并不直接对外暴露其算子类型原语(Primitive)，因此无法直接为算子配置并行策略，而需要用户在单卡网络的基础上使用`mindspore.parallel.shard`接口手动配置mint算子的切分策略，例如配置策略后的网络结构为：

```python
import mindspore as ms
from mindspore import nn, mint

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = mint.flatten
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=((2, 4), (4, 1)))
        self.relu1 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((4, 1),))
        self.matmul2 = ms.parallel.shard(mint.matmul, in_strategy=((1, 8), (8, 1)))
        self.relu2 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((8, 1),))
        self.matmul3 = mint.matmul

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x, dim=0, keepdims=True)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x, dim=0, keepdims=True)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
```

以上网络的`mint.matmul`和`mint.nn.functional.relu`算子都配置了切分策略，以`ms.parallel.shard(mint.matmul, in_strategy=((2, 4), (4, 1)))`为例，它的切分策略为：第一个输入的行切分2份，列切分4份；第二个输入的行切分4份；对于`ms.parallel.shard(mint.mean, in_strategy=((8, 1),))`，它的切分策略为：第一个输入的行切分8份。需要注意的是，此处的两个`mint.nn.functional.relu`的切分策略不同，一个是`ms.parallel.shard(mint.nn.functional.relu, in_strategy=((4, 1),))`，一个是`ms.parallel.shard(mint.nn.functional.relu, in_strategy=((8, 1),))`，所以要定义两次。

#### 并行配置

需要进一步设置并行有关的配置，指定并行模式`semi_auto`为半自动并行模式。

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(net, parallel_mode="semi_auto")
```

#### 执行网络

这一步循环执行网络的正向计算，外层循环是执行的epoch数，内层循环遍历数据集，调用`parallel_net`执行分布式计算并获得正向输出。

```python
for epoch in range(10):
    i = 0
    for image, _ in data_set:
        forward_logits = parallel_net(image)
        if i % 10 == 0:
            forward_sum = mint.sum(forward_logits).asnumpy()
            print("epoch: %s, step: %s, forward_sum is %s" % (epoch, i, forward_sum))
        i += 1
```

#### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run_mint.sh
```

训练完后，日志文件保存到`mint_log_output`目录下，其中部分文件目录结构如下：

```text
└─ mint_log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

结果保存在`mint_log_output/worker_*.log`中，示例如下：

```text
epoch: 0 step: 0, forward_sum is 0.90023
epoch: 0 step: 10, forward_sum is 1.07679
epoch: 0 step: 20, forward_sum is 1.02521
epoch: 0 step: 30, forward_sum is 0.96682
epoch: 0 step: 40, forward_sum is 0.93158
epoch: 0 step: 50, forward_sum is 0.96655
...
```

其他启动方式如`mpirun`、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/startup_method.html)。

## 高阶算子级并行实践

### 高阶ops算子并行实践

以Ascend单机8卡为例，进行高阶ops算子并行操作说明。

#### 样例代码说明

> 下载完整的样例代码：[distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/distributed_operator_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── advanced_distributed_operator_parallel.py
       ├── run_advanced.sh
       └── ...
    ...
```

其中，`advanced_distributed_operator_parallel.py`是高阶算子级并行定义网络结构和进行训练的脚本。`run_advanced.sh`是执行脚本。

#### 环境配置

进行高阶算子级并行前，首先进行环境配置，其流程与算子级并行一致，可以参考[配置分布式环境](#配置分布式环境)和[数据集加载](#数据集加载)。

#### 定义网络

高阶算子级并行对`shard`接口进行功能扩展，`shard`接口的`in_strategy`/`out_strategy`两个入参，都额外接收新的数量类型`tuple(Layout)`类型。

其中Layout使用设备矩阵进行初始化，同时要求给设备矩阵的每个轴取一个别名，如"layout = Layout((2, 2, 2), name = ("dp", "sp", "mp"))"，该设备矩阵即描述的是共有8张卡，按照(2, 2, 2)的形状进行排列，而每个轴分别取了别名"dp"、"sp"、"mp"。

对Layout进行调用传入的则是这几个轴，每个张量按照其shape选取每个维度期望映射到设备的哪个轴，同时也确定了切分的份数，如这里"dp"就表示在设备排布的最高维度的2个设备内切分2份，而"sp"表示在设备排布的中间维度的2个设备内切分2份，"mp"表示在设备排布的最低维度的2个设备内切分为2份。特别地，张量的一个维度可以映射到设备的多个维度，以表达在一个维度进行多次切分。

```python

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer
from mindspore.parallel import Layout

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
        layout2 = Layout((8,), ("tp",))
        self.matmul1 = ops.MatMul().shard((layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))
        self.relu1 = ops.ReLU().shard(((4, 1),))
        self.matmul2 = ops.MatMul().shard((layout2("None", "tp"), layout2("tp", "None")))
        self.relu2 = ops.ReLU().shard(((8, 1),))
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

```

上述定义的网络中，`self.matmul1 = ops.MatMul().shard((layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))`对输入张量x切分的layout是`layout("mp", ("sp", "dp"))`，即第一个维度按mp切成2份，第二个维度合并sp和dp，共2*2=4份。

对权重self.fc1_weight切分的layout是`layout(("sp", "dp"), "None")`，即第一个维度合并sp和dp，切分4份，第二个维度不切分。

同理，`self.matmul2 = ops.MatMul().shard((layout2("None", "tp"), layout2("tp", "None")))`对输入张量x第一个维度按行不切分，列按tp切成8份，对权重self.fc2_weight进行切分时，行按tp切分8份，列不切分。

以`self.matmul1 = ops.MatMul().shard((layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))`为例，切分后将生成如下设备与数据切片映射表：

| 设备坐标 (dp, sp, mp) | 输入 x 切片         | 权重 fc1_weight 切片     |
|-----------------------|----------------------|---------------------------|
| (0, 0, 0)             | `x[0:16, 0:196]`     | `fc1_weight[0:196, 0:512]` |
| (0, 0, 1)             | `x[16:32, 0:196]`    | `fc1_weight[0:196, 0:512]` |
| (0, 1, 0)             | `x[0:16, 196:392]`   | `fc1_weight[196:392, 0:512]` |
| (0, 1, 1)             | `x[16:32, 196:392]`  | `fc1_weight[196:392, 0:512]` |
| (1, 0, 0)             | `x[0:16, 392:588]`   | `fc1_weight[392:588, 0:512]` |
| (1, 0, 1)             | `x[16:32, 392:588]`  | `fc1_weight[392:588, 0:512]` |
| (1, 1, 0)             | `x[0:16, 588:784]`   | `fc1_weight[588:784, 0:512]` |
| (1, 1, 1)             | `x[16:32, 588:784]`  | `fc1_weight[588:784, 0:512]` |

#### 训练流程

高阶算子级并行的训练流程与算子级并行完全一致，可以参考[训练网络定义](#训练网络定义)、[并行配置](#并行配置)和[训练循环](#训练循环)。

#### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run_advanced.sh
```

训练完后，日志文件保存到`advanced_log_output`目录下，其中部分文件目录结构如下：

```text
└─ advanced_log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

结果保存在`advanced_log_output/worker_*.log`中，示例如下：

```text
epoch: 0 step: 0, loss is 2.3016002
epoch: 0 step: 10, loss is 2.2889402
epoch: 0 step: 20, loss is 2.2843816
epoch: 0 step: 30, loss is 2.248126
epoch: 0 step: 40, loss is 2.1581488
epoch: 0 step: 50, loss is 1.8051043
...
```

其他启动方式如`mpirun`、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/startup_method.html)。

### 高阶mint算子并行实践

以Ascend单机8卡为例，进行高阶mint算子并行操作说明。

#### 样例代码说明

> 下载完整的样例代码：[distributed_operator_parallel](https://gitee.com/mindspore/docs/tree/r2.6.0rc1/docs/sample_code/distributed_operator_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_operator_parallel
       ├── advanced_distributed_mint_operator_parallel.py
       ├── run_advanced_mint.sh
       └── ...
    ...
```

其中，`advanced_distributed_mint_operator_parallel.py`是高阶算子级并行定义网络结构和进行训练的脚本。`run_advanced_mint.sh`是执行脚本。

#### 环境配置

进行高阶mint算子并行前，首先进行环境配置，其流程与算子级并行一致，可以参考[配置分布式环境](#配置分布式环境)和[数据集加载](#数据集加载)。

#### 定义网络

高阶mint算子并行的切分策略配置方法与mint算子并行类似，只需要为`mindspore.parallel.shard`接口中的参数`in_strategy`传入`tuple(Layout)`类型的输入即可。

其中Layout使用设备矩阵进行初始化，同时要求给设备矩阵的每个轴取一个别名，如"layout = Layout((2, 2, 2), name = ("dp", "sp", "mp"))"，该设备矩阵即描述的是共有8张卡，按照(2, 2, 2)的形状进行排列，而每个轴分别取了别名"dp"、"sp"、"mp"。

对Layout进行调用传入的则是这几个轴，每个张量按照其shape选取每个维度期望映射到设备的哪个轴，同时也确定了切分的份数，如这里"dp"就表示在设备排布的最高维度的2个设备内切分2份，而"sp"表示在设备排布的中间维度的2个设备内切分2份，"mp"表示在设备排布的最低维度的2个设备内切分为2份。特别地，张量的一个维度可以映射到设备的多个维度，以表达在一个维度进行多次切分。

```python

import mindspore as ms
from mindspore import nn, mint

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = mint.flatten
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
        layout2 = Layout((8,), ("tp",))
        self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=(layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))
        self.relu1 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((4, 1),))
        self.matmul2 = ms.parallel.shard(mint.matmul, in_strategy=(layout2("None", "tp"), layout2("tp", "None")))
        self.relu2 = ms.parallel.shard(mint.nn.functional.relu, in_strategy=((8, 1),))
        self.matmul3 = mint.matmul

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x, dim=0, keepdims=True)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x, dim=0, keepdims=True)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
```

上述定义的网络中，`self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=(layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))`对输入张量x切分的layout是`layout("mp", ("sp", "dp"))`，即第一个维度按mp切成2份，第二个维度合并sp和dp，共2*2=4份。

对权重self.fc1_weight切分的layout是`layout(("sp", "dp"), "None")`，即第一个维度合并sp和dp，切分4份，第二个维度不切分。

同理，`self.matmul2 = ms.parallel.shard(mint.matmul, in_strategy=(layout2("None", "tp"), layout2("tp", "None")))`对输入张量x第一个维度按行不切分，列按tp切成8份，对权重self.fc2_weight进行切分时，行按tp切分8份，列不切分。

以`self.matmul1 = ms.parallel.shard(mint.matmul, in_strategy=(layout("mp", ("sp", "dp")), layout(("sp", "dp"), "None")))`为例，切分后将生成如下设备与数据切片映射表：

| 设备坐标 (dp, sp, mp) | 输入 x 切片         | 权重 fc1_weight 切片     |
|-----------------------|----------------------|---------------------------|
| (0, 0, 0)             | `x[0:16, 0:196]`     | `fc1_weight[0:196, 0:512]` |
| (0, 0, 1)             | `x[16:32, 0:196]`    | `fc1_weight[0:196, 0:512]` |
| (0, 1, 0)             | `x[0:16, 196:392]`   | `fc1_weight[196:392, 0:512]` |
| (0, 1, 1)             | `x[16:32, 196:392]`  | `fc1_weight[196:392, 0:512]` |
| (1, 0, 0)             | `x[0:16, 392:588]`   | `fc1_weight[392:588, 0:512]` |
| (1, 0, 1)             | `x[16:32, 392:588]`  | `fc1_weight[392:588, 0:512]` |
| (1, 1, 0)             | `x[0:16, 588:784]`   | `fc1_weight[588:784, 0:512]` |
| (1, 1, 1)             | `x[16:32, 588:784]`  | `fc1_weight[588:784, 0:512]` |

#### 并行配置

需要进一步设置并行有关的配置，指定并行模式`semi_auto`为半自动并行模式。

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(net, parallel_mode="semi_auto")
```

#### 执行网络

这一步循环执行网络的正向计算，外层循环是执行的epoch数，内层循环遍历数据集，调用`parallel_net`执行分布式计算并获得正向输出。

```python
for epoch in range(10):
    i = 0
    for image, _ in data_set:
        forward_logits = parallel_net(image)
        if i % 10 == 0:
            forward_sum = mint.sum(forward_logits).asnumpy()
            print("epoch: %s, step: %s, forward_sum is %s" % (epoch, i, forward_sum))
        i += 1
```

#### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run_advanced_mint.sh
```

训练完后，日志文件保存到`advanced_log_output`目录下，其中部分文件目录结构如下：

```text
└─ advanced_mint_log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

结果保存在`advanced_mint_log_output/worker_*.log`中，示例如下：

```text
epoch: 0 step: 0, forward_sum is 0.90023
epoch: 0 step: 10, forward_sum is 1.07679
epoch: 0 step: 20, forward_sum is 1.02521
epoch: 0 step: 30, forward_sum is 0.96682
epoch: 0 step: 40, forward_sum is 0.93158
epoch: 0 step: 50, forward_sum is 0.96655
...
```

其他启动方式如`mpirun`、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/startup_method.html)。
