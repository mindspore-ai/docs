# 参数服务器

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/parallel/parameter_server_training.md)

## 概述

Parameter Server（参数服务器）是一种在分布式训练中广泛使用的架构。该架构一共包含三个独立的组件，分别是Server、Worker和Scheduler，其作用分别是：

- Server：保存模型的权重和反向计算的梯度值，并使用优化器通过Worker上传的梯度值对模型进行更新。

- Worker：执行网络的正反向计算，通过Push接口将反向计算的梯度值上传至Server中，通过Pull接口将Server更新好的模型下载到Worker本地。

- Scheduler：用于建立Server和Worker的通信关系。

相较于同步的AllReduce训练方法，参数服务器具有更好的灵活性、可扩展性。原因有三点：

1. 参数服务器既支持同步SGD(Stochastic Gradient Descent，随机梯度下降)，也支持异步SGD的训练算法。
2. 在扩展性上，参数服务器将模型的计算与更新分别部署在Worker和Server两类进程中，使得两者的资源可以独立地横向扩缩(新增或者删除Worker和Server资源)。
3. 在大规模数据中心的环境下，计算设备、网络以及存储经常会出现各种故障，导致部分节点异常。而在参数服务器的架构下，处理此类故障更容易，且不会对训练中的任务产生影响。

MindSpore的参数服务器采用了自研的通信框架作为基础架构。基于该框架提供了远程通信能力以及抽象的Send/Broadcast等原语，实现了同步SGD的分布式训练算法。另外，结合Ascend和GPU中的高性能集合通信库(HCCL 和 NCCL)，MindSpore还提供了参数服务器和AllReduce的混合训练模式，支持将部分权重通过参数服务器进行存储和更新，其余权重仍然通过AllReduce算法进行训练。

> 参数服务器支持Ascend、GPU硬件平台，不支持`PyNative`模式。

相关接口：

1. `mindspore.set_ps_context(enable_ps=True)`开启参数服务器训练模式。

    - 此接口需在`mindspore.communication.init()`之前调用。
    - 若没有调用此接口，下面的环境变量设置不会生效。
    - 调用`mindspore.reset_ps_context()`可以关闭参数服务器训练模式。

2. 在该训练模式下，有以下两种调用接口方式，可以控制训练参数是否通过参数服务器进行更新，以及控制参数初始化位置：

    - 通过`mindspore.nn.Cell.set_param_ps()`对`nn.Cell`中所有权重进行递归设置。
    - 通过`mindspore.Parameter.set_param_ps()`对`mindspore.Parameter`权重进行设置。

    注意：

    - 对于通过参数服务器更新的训练参数，其单个权重大小不得超过INT_MAX(2^31 - 1)字节。
    - 接口`set_param_ps`可接收一个`bool`型参数：`init_in_server`，表示该训练参数是否在Server端初始化，其默认值为`False`，表示在Worker上初始化该训练参数。
    - 当前仅支持`EmbeddingLookup`算子的训练参数`embedding_table`在Server端初始化，以解决超大shape的`embedding_table`在Worker上初始化导致内存不足的问题。该算子的`target`属性需要设置为'CPU'。在Server端初始化的训练参数将不再同步到Worker上，如果涉及到多Server训练并保存CheckPoint，则训练结束后每个Server均会保存一个 CheckPoint。上述`embedding_table`表示一个二维表，用于储存和管理学习模型中使用到的嵌入向量。

3. （可选配置）针对超大shape的`embedding_table`，由于设备上无法全量存放，可以配置[EmbeddingLookup 算子](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.EmbeddingLookup.html)的`vocab_cache_size`参数，开启参数服务器训练模式下的**分布式特征缓存功能**。该功能将在设备上使用一块`vocab_cache_size`大小的独占空间作为缓存 (Embedding Cache)，供部分`embedding_table`在设备上训练，以达到提升训练性能的目的。而全量`embedding_table`仍旧存储在Server上。在训练过程中，将下批次训练用到的`embedding_table`提前放入Embedding Cache，当Embedding Cache已满，过期的`embedding_table`将会被放回至Server。训练结束后，可在Server上导出CheckPoint，保存训练后的全量`embedding_table`。Embedding Cache支持sparse模式。针对配置了`vocab_cache_size`的`EmbeddingLookup`算子，通过将其`sparse`参数都设为True，sparse模式会对该算子输入的特征id去重，以降低计算与通信量。

相关环境变量配置：

MindSpore通过读取环境变量，控制参数服务器训练。环境变量包括以下选项（所有脚本中的`MS_SCHED_HOST`及`MS_SCHED_PORT`值需保持一致）：

```text
export MS_SERVER_NUM=1                # Server number
export MS_WORKER_NUM=1                # Worker number
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED               # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

更多详细说明请查看[动态组网环境变量](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/dynamic_cluster.html)。

## 操作实践

参数服务器支持GPU和Ascend，下面以Ascend为例进行操作说明：

### 样例代码说明

> 下载完整的样例代码：[parameter_server](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/parameter_server)。

目录结构如下：

```text
└─ sample_code
    ├─ parameter_server
       ├── train.py
       └── run.sh
    ...
```

其中，`train.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

### 配置分布式环境

通过context接口指定运行模式、运行设备、运行卡号等。与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`，使能`enable_ps`开启参数服务器训练模式，并通过init初始化HCCL或NCCL通信。此处未设置`device_target`，会自动指定为MindSpore包对应的后端硬件设备。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(full_batch=True, parallel_mode=ms.ParallelMode.AUTO_PARALLEL)
ms.set_ps_context(enable_ps=True)
init()
ms.set_seed(1)
```

- `full_batch`：是否全量导入数据集。为`True`时表示全量导入，每卡的数据相同，在多Worker场景中必须设置为`True`。
- `parallel_mode`：并行模式。多Worker场景下，需要开启自动并行模式，通过设置`parallel_mode=ParallelMode.AUTO_PARALLEL`实现。

### 网络定义

参数服务器模式的网络定义是在单卡模式的基础上配置 net.set_param_ps()：

```python
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(28*28, 10, weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(10, 1, weight_init="normal", bias_init="zeros")

    def construct(self, x):
        x = self.flatten(x)
        logits = self.fc2(self.relu(self.fc1(x)))
        return logits

net = Network()
net.set_param_ps()
```

### 数据集加载

数据集加载方式与单卡模型一致，代码如下：

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

### 训练网络

在这一部分，定义优化器，损失函数和训练网络。此处采用函数式写法来定义网络，代码与单卡模式一致：

```python
import mindspore as ms
from mindspore import nn

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.MSELoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

### 运行单机 8 卡脚本

接下来通过命令调用对应的脚本，以8卡的分布式训练脚本为例，进行分布式训练。Scheduler、Server和Worker三个角色分别启动对应数量的进程。命令如下：

```bash
EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf output
mkdir output

# run Scheduler process
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_SCHED
python train.py > output/scheduler.log 2>&1 &

# run Server processes
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_PSERVER
for((server_id=0;server_id<${MS_SERVER_NUM};server_id++))
do
    python train.py > output/server_${server_id}.log 2>&1 &
done

# run Wroker processes
export MS_SERVER_NUM=8
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_WORKER
for((worker_id=0;worker_id<${MS_WORKER_NUM};worker_id++))
do
    python train.py > output/worker_${worker_id}.log 2>&1 &
done
```

或者直接执行：

```bash
bash run.sh
```

每个进程的输出结果保存在`output`文件夹中，可以在`output/scheduler.log`中查看Server与Worker通信日志：

```text
...
Assign rank id of node id: 2fa9d1ab-10b8-4a61-9acf-217a04439287, role: MS_WORKER, with host ip: 127.0.0.1, old rank id: 6, new rank id: 0
...
Assign rank id of node id: 02fb1169-edc3-465e-b307-ccaf62d1f0b3, role: MS_PSERVER, with host ip: 127.0.0.1, old rank id: 4, new rank id: 0
...
Cluster is successfully initialized.
```

训练结果保存在`output/worker_0.log`中，示例如下：

```text
epoch: 0, step: 0, loss is 26.743706
epoch: 0, step: 10, loss is 17.507723
epoch: 0, step: 20, loss is 9.616591
epoch: 0, step: 30, loss is 8.589715
epoch: 0, step: 40, loss is 8.23479
epoch: 0, step: 50, loss is 10.431321
epoch: 0, step: 60, loss is 7.7080607
epoch: 0, step: 70, loss is 8.599786
epoch: 0, step: 80, loss is 7.669814
epoch: 0, step: 90, loss is 8.584343
epoch: 0, step: 100, loss is 8.803712
```
