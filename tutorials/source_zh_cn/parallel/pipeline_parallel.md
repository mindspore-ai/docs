# 流水线并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/parallel/pipeline_parallel.md)

## 简介

近年来，神经网络的规模几乎是呈指数型增长。受单卡内存的限制，训练这些大模型用到的设备数量也在不断增加。受server间通信带宽低的影响，传统数据并行叠加模型并行的这种混合并行模式的性能表现欠佳，需要引入流水线并行。流水线并行能够将模型在空间上按阶段（Stage）进行切分，每个Stage只需执行网络的一部分，大大节省了内存开销，同时缩小了通信域，缩短了通信时间。MindSpore能够根据用户的配置，将单机模型自动地转换成流水线并行模式去执行。

## 训练操作实践

下面以Ascend单机8卡为例，进行流水线并行操作说明：

### 样例代码说明

> 下载完整的样例代码：[distributed_pipeline_parallel](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_pipeline_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_pipeline_parallel
       ├── distributed_pipeline_parallel.py
       └── run.sh
    ...
```

其中，`distributed_pipeline_parallel.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

### 配置分布式环境

通过context接口指定运行模式，与单卡脚本不同，并行脚本还需通过init接口初始化通信。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
ms.set_seed(1)
```

### 数据集加载

在流水线并行场景下，数据集加载方式与单卡加载方式一致，代码如下：

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

### 定义网络

流水线并行网络结构与单卡网络结构基本一致。需要注意的是：

> - 在pipeline并行下，使能Print/Summary/TensorDump相关算子时，需要把该算子放到有pipeline_stage属性的Cell中使用，否则有概率由pipeline并行切分导致算子不生效。
> - 在pipeline并行下，网络的输出不支持动态shape。
> - 在pipeline并行下，推荐使用lazy_inline装饰器来缩短编译时间，并且仅支持将lazy_inline装饰器配置在最外层的Cell上。

```python
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, HeUniform

import math

class MatMulCell(nn.Cell):
    """
    MatMulCell definition.
    """
    def __init__(self, param=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [28 * 28, 512]
        weight_init = HeUniform(math.sqrt(5))
        self.param = Parameter(initializer(weight_init, shape), name="param")
        if param is not None:
            self.param = param
        self.print = ops.Print()
        self.matmul = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        self.print("out is:", out)
        return out


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = MatMulCell()
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

```

### 训练网络定义

在这一步，我们需要定义损失函数、优化器以及训练过程。需要注意的是，这里对网络和优化器的定义都需要延后初始化。除此之外，还需要增加 `PipelineGradReducer` 接口，用于处理流水线并行下的梯度，该接口的第一个参数为需要更新的网络参数，第二个为是否使用优化器并行。

与单卡模型不同，在这部分需要调用两个接口来配置流水线并行：

- 首先需要定义LossCell，本例中调用了`nn.WithLossCell`接口封装网络和损失函数。
- 然后需要在LossCell外包一层`Pipeline`，并指定MicroBatch的size，并通过`stage_config`配置每个包含训练参数的`Cell`的`pipeline_stage`。

```python
import mindspore as ms
from mindspore import nn, ops
from mindspore.parallel.nn import Pipeline, PipelineGradReducer
from mindspore.nn.utils import no_init_parameters

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
    pp_grad_reducer = PipelineGradReducer(optimizer.parameters, opt_shard=False)

loss_fn = nn.CrossEntropyLoss()
net_with_loss = Pipeline(nn.WithLossCell(net, loss_fn), 4, stage_config={"_backbone.flatten":0,
                            "_backbone.layer1": 0, "_backbone.relu1": 0, "_backbone.layer2": 1, "_backbone.relu2": 1, "_backbone.layer3": 1})
net_with_loss.set_train()

def forward_fn(inputs, target):
    loss = net_with_loss(inputs, target)
    return loss

grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

@ms.jit
def train_one_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = pp_grad_reducer(grads)
    optimizer(grads)
    return loss, grads

```

使能interleaved pipeline调度，`Pipeline`中的`stage_config`需要对非连续模型层进行交错式配置，配置如下：

```python
net_with_loss = Pipeline(nn.WithLossCell(net, loss_fn), 4, stage_config={"_backbone.flatten":0,
                            "_backbone.layer1": 1, "_backbone.relu1": 0, "_backbone.layer2": 1, "_backbone.relu2": 0, "_backbone.layer3": 1})
```

## 并行配置

我们需要进一步设置并行有关的配置，指定并行模式`semi_auto`为半自动并行模式，此外，还需开启流水线并行，配置`pipeline`，并通过配置`stages`数来指定stage的总数。

```python
import mindspore as ms
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_one_step, parallel_mode="semi_auto")
parallel_net.pipeline(stages=2)

```

如果需要跑interleaved pipeline调度，还需要配置：`parallel_net.pipeline(stages=2, interleave=True)`，需要注意的是，MindSpore的interleaved pipeline调度还在完善阶段，目前在O0或者O1模式下表现会更好。

```python
import mindspore as ms
import mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_one_step, parallel_mode="semi_auto")
parallel_net.pipeline(stages=2, interleave=True)
```

## 训练循环

这一步进行训练循环，外层循环是训练的epoch数，内层循环遍历数据集，调用parallel_net进行训练并获得损失值。

```python
for epoch in range(10):
    i = 0
    for data, label in data_set:
        loss, grads = parallel_net(data, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss))
        i += 1
```

> 目前流水线并行不支持自动混合精度特性。
>
> 流水线并行训练更适合用`model.train`的方式，这是因为流水线并行下的TrainOneStep逻辑复杂，而`model.train`内部封装了针对流水线并行的TrainOneStepCell，易用性更好。

### 运行单机8卡脚本

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
epoch: 0 step: 0, loss is 9.137518
epoch: 0 step: 10, loss is 8.826559
epoch: 0 step: 20, loss is 8.675843
epoch: 0 step: 30, loss is 8.307994
epoch: 0 step: 40, loss is 7.856993
epoch: 0 step: 50, loss is 7.0662785
...
```

`Print` 算子的结果为:

```text
out is:
Tensor(shape=[8, 512], dtype=Float32, value=
[[ 4.61914062e-01 5.78613281e-01 1.34995094e-01 ... 8.54492188e-02 7.91992188e-01 2.13378906e-01]
...
[  4.89746094e-01 3.56689453e-01 -4.90966797e-01 ... -3.30078125e-e01 -2.38525391e-01 7.33398438e-01]])
```

其他启动方式如`mpirun`、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/startup_method.html)。

## 推理操作实践

下面以Ascend单机8卡为例，进行流水线并行操作说明：

### 样例代码说明

> 下载完整的样例代码：[distributed_pipeline_parallel](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_pipeline_parallel)。

目录结构如下：

```text

└─ sample_code
    ├─ distributed_pipeline_parallel
       ├── distributed_pipeline_parallel_inference.py
       └── run_inference.sh
    ...

```

其中，`distributed_pipeline_parallel_inference.py`是定义网络结构和推理过程的脚本。`run_inference.sh`是执行脚本。

### 配置分布式环境

通过context接口指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需通过init初始化HCCL或NCCL通信。

```python

import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
ms.set_seed(1)

```

### 定义网络

流水线并行需要用户去定义并行的策略，通过调用`pipeline_stage`接口来指定每个layer要在哪个stage上去执行。`pipeline_stage`接口的粒度为`Cell`。所有包含训练参数的`Cell`都需要配置`pipeline_stage`，并且`pipeline_stage`要按照网络执行的先后顺序，从小到大进行配置。在单卡模型基础上，增加`pipeline_stage`配置后如下：

```python

import numpy as np
from mindspore import lazy_inline, nn, ops, Tensor, Parameter, sync_pipeline_shared_parameters

class VocabEmbedding(nn.Cell):
    """Vocab Embedding"""
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding_table = Parameter(Tensor(np.ones([vocab_size, embedding_size]), ms.float32),
                                         name='embedding_table')
        self.gather = ops.Gather()

    def construct(self, x):
        output = self.gather(self.embedding_table, x, 0)
        output = output.squeeze(1)
        return output, self.embedding_table.value()


class Head(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, state, embed):
        return self.matmul(state, embed)


class Network(nn.Cell):
    """Network"""
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.word_embedding = VocabEmbedding(vocab_size=32, embedding_size=32)
        self.layer1 = nn.Dense(32, 32)
        self.layer2 = nn.Dense(32, 32)
        self.head = Head()

    def construct(self, x):
        x, embed = self.word_embedding(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x, embed)
        return x

# Define network and set pipeline stage
net = Network()
net.word_embedding.pipeline_stage = 0
net.layer1.pipeline_stage = 1
net.layer2.pipeline_stage = 2
net.head.pipeline_stage = 3

```

### 推理网络

在network外包一层`PipelineCellInference`，并指定MicroBatch的size。`PipelineCellInference`中将输入切分为若干个micro batch，执行推理网络，最后将若干个micro batch推理结果通过`ops.Concat`算子沿batch轴拼接后返回。

在上一步中，`embed`被`self.word_embedding`和`self.head`两层共享，并且这两层被切分到了不同的stage上。

我们需要进一步设置并行有关的配置，用`AutoParallel`再包裹一次network，指定并行模式`semi_auto`为半自动并行模式，此外，还需开启流水线并行，配置`pipeline`，并通过配置`stages`数来指定stage的总数。此处不设置`device_target`会自动指定为MindSpore包对应的后端硬件设备（默认为Ascend）。`output_broadcast=True`表示流水线并行推理时，将最后一个stage的结果广播给其余stage，可以用于自回归推理场景。

在执行推理前，先编译计算图`parallel_net.compile()`，再调用`sync_pipeline_shared_parameters(parallel_net)`接口，框架自动同步stage间的共享权重。

```python

from mindspore import nn, ops

class PipelineCellInference(nn.Cell):
    """Pipeline Cell Inference wrapper"""
    def __init__(self, network, micro_batch_num):
        super().__init__()
        self.network = network
        self.micro_batch_num = micro_batch_num
        self.concat = ops.Concat()

    def construct(self, x):
        """Apply the pipeline inference"""
        ret = ()
        for i in range(self.micro_batch_num):
            micro_batch_size = x.shape[0] // self.micro_batch_num
            start = micro_batch_size * i
            end = micro_batch_size * (i + 1)

            micro_input = x[start:end]
            micro_output = self.network(micro_input)
            ret = ret + (micro_output,)

        ret = self.concat(ret)
        return ret

inference_network = PipelineCellInference(network=net, micro_batch_num=4)
inference_network.set_train(False)

parallel_net = AutoParallel(inference_network, parallel_mode="semi_auto")
parallel_net.dataset_strategy("full_batch")
parallel_net.pipeline(stages=4, output_broadcast=True)

# Compile and synchronize shared parameter.
input_ids = Tensor(np.random.randint(low=0, high=32, size=(8, 1)), ms.int32)
parallel_net.compile(input_ids)
sync_pipeline_shared_parameters(parallel_net)

# Execute the inference network
logits = parallel_net(input_ids)
print(logits.asnumpy())

```

### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式推理脚本为例，进行分布式训练：

```bash

bash run_inference.sh

```

训练完后，日志文件保存到`pipeline_inference_logs`目录下，其中部分文件目录结构如下：

```text

└─ pipeline_inference_logs
   ├── scheduler.log
   ├── worker_0.log
   ├── worker_1.log
   ├── worker_2.log
...

```

结果保存在`pipeline_inference_logs/worker_*.log`中，示例如下：

```text

[[0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556 0.01181556
  0.01181556 0.01181556]
  ...]

```