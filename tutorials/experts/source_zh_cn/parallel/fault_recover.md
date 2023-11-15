# 基于冗余信息的故障恢复

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_zh_cn/parallel/fault_recover.md)

## 概述

在进行分布式训练时，遇到故障是非常普遍的，类似于单卡训练，可以通过加载训练过程中保存的权重信息继续进行训练。区别于纯数据并行训练，当应用了模型并行后，权重是进行了切分的，卡与卡之间保存的权重信息可能不一致。

为了解决这个问题，一个方案是在保存权重checkpoint文件前，就将权重通过[AllGather](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/samples/ops/communicate_ops.html#allgather) 算子进行汇聚，每张卡均存储一个完整的权重信息，这一个功能即`mindspore.train.CheckpointConfig(integrated_save=True)`接口中的合并保存。

但是，对于大模型来说，使用汇聚保存对各种资源的开销都过于巨大，因此，本文档介绍的是每张卡仅仅保存自身的权重信息的恢复方案。对于大模型来说，往往会同时应用上数据并行与模型并行，而数据并行的维度所划分的设备，它们持有的权重信息是完全一致的，这也为大模型提供了冗余的备份，本文档也将指出如何去获取这个冗余信息。

关于并行策略与权重的切片划分的关系，可以进行如下映射。关于数据并行，模型并行的概念，请参考[算子级并行](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/operator_parallel.html) ；关于优化器并行，请参考[优化器并行](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/optimizer_parallel.html)。

- 数据并行 + 不开启优化器并行：并行通信域内的rank持有相同权重切片。
- 模型并行：并行通信域内的rank持有不同权重切片。
- 数据并行 + 开启优化器并行 + 优化器并行切满所有数据并行维度：并行通信域内的rank持有不同权重切片。
- 数据并行 + 开启优化器并行 + 优化器并行不切满所有数据并行维度：并行通信域内，优化器切分的通信域内的rank持有不同的权重切片，每个优化器切分的通信域之间持有相同的权重切片。

另外，需要注意的是，本文档介绍分布式故障恢复方案，需要在[下沉模式](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/optimize/execution_opt.html) 下使用。

相关环境变量：

`GROUP_INFO_FILE=./group_info.pb`：保存切片的权重信息，该文件解析出来后将得到一个列表，该列表中的值为rank_id，代表这些rank_id中的权重是相同的。

## 操作实践

下面以单机8卡为例，进行分布式训练下故障恢复的操作说明：

### 样例代码说明

>下载完整的样例代码：[fault_recover](https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/fault_recover)

目录结构如下：

```text
└─ sample_code
    ├─ fault_recover
        ├── train.py
        ├── run.sh
        └── recover.sh
```

其中，`train.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本，`recover.sh`是节点故障后的恢复脚本。

### 配置分布式环境

通过context接口指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`，并通过init初始化HCCL或NCCL通信。`device_target`会自动指定为MindSpore包对应的后端硬件设备。

```python
import mindspore as ms
from mindspore.communication import init, get_rank

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
os.environ['GROUP_INFO_FILE'] = "./checkpoints/rank_{}/group_info.pb".format(get_rank())
ms.set_seed(1)
```

> 此处配置环境变量GROUP_INFO_FILE存储权重的冗余信息。

### 数据集加载

在当前样例中，数据集加载方式与单卡加载方式一致，代码如下：

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

此处对算子配置一些切分策略，配置策略后的网络结构为：

```python
import mindspore as ms
from mindspore import nn, ops

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul()
        self.relu1 = ops.ReLU()
        self.matmul2 = ops.MatMul()
        self.relu2 = ops.ReLU()
        self.matmul3 = ops.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
net.matmul1.shard(((2, 4), (4, 1)))
net.relu1.shard(((4, 1),))
```

### 训练网络

在这一步，我们需要定义损失函数、优化器以及训练过程：

```python
import mindspore as ms
from mindspore import nn, train
from mindspore.communication import get_rank

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor()
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=4, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint", directory="./checkpoints/rank_{}".format(get_rank()), config=ckpt_config)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(2, data_set, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)
```

> 训练时通过指定dataset_sink_mode为True以配置为下沉模式，CheckpointConfig中需配置`integrated_save`为`False`。

### 故障恢复

分布式的故障恢复，需要事先获取切分的信息，因而，需要先调用`model.infer_train_layout`得到切分策略信息，继而再执行训练。

```python
import mindspore as ms
from mindspore.communication import get_rank

# model create
# checkpoint load
if bool(args_opt.is_recover):
    param_dict = ms.load_checkpoint("./checkpoints/rank_{}/checkpoint-2_1875.ckpt".format(get_rank()))
    model.infer_train_layout(data_set)
    ms.load_param_into_net(net, param_dict)
model.train(2, data_set, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)
```

### 运行单机八卡脚本

接下来通过命令调用对应的脚本，以`mpirun`启动方式，8卡的分布式脚本为例，通过下命令运行8卡的并行训练脚本：

```bash
bash run.sh
```

训练完成后，可以看到以下文件：

```text
├─ group_info.pb
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ checkpoints
|   ├─ rank_0
|   |   ├─ checkpoint-1_1875.ckpt
|   |   ├─ checkpoint-2_1875.ckpt
|   |   ├─ checkpoint-graph.meta
|   |   └─ group_info.pb
|   ├─ rank_1
|   |   ├─ checkpoint-1_1875.ckpt
|   |   ...
|   ...
...
```

在`log_output/1/rank.*/stdout`中，可以看到当前训练后的loss值，类似如下：

```text
epoch: 1 step: 1875, loss is 0.71328689217567444
epoch: 2 step: 1875, loss is 0.32782320742607117
```

读取group_info.pb，可以获取到权重的冗余信息，该文件解析出来后将得到一个列表，该列表中的值为rank_id，表示这些列表中的rank_id对应的权重切片都是相同的，可以相互替换。
如下面的例子，0卡的group_info.pb解析出来后，发现0卡和4卡的权重切分是完全一致的，当0卡的checkpoint丢失时，可以直接复制4卡checkpoint作为0卡的checkpoint，进行恢复。

```python
import mindspore as ms
rank_list = ms.restore_group_info_list("./checkpoints/rank_0/group_info.pb")
print(rank_list) // [0, 4]
```

而后，执行故障恢复训练脚本。

```bash
bash recover.sh
```

恢复训练结束后，查看loss如下，说明加载成功了。

```text
epoch: 1 step: 1875, loss is 0.598689079284668
epoch: 2 step: 1875, loss is 0.266701698332226
```
