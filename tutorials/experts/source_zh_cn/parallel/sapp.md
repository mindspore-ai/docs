# 双递归策略搜索算法

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/sapp.md)

## 概述

双递归策略搜索算法基于符号化自动策略生成(Symbolic Automatic Parallel Planner 缩写SAPP)。SAPP算法能够对于巨大网络以及大规模切分瞬间生成最优策略。SAPP基于并行原理建模，通过建立抽象机来描述硬件集群拓扑，通过符号化简优化代价模型。其代价模型比较的不是预估的绝对时延，而是不同并行策略的相对代价，因此能够大大压缩搜索空间，对于百卡集群能够保证分钟级的搜索时间。

> 双递归策略搜索算法支持的硬件平台包括Ascend、GPU，此外还同时支持PyNative模式和Graph模式。

相关接口：

`mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, search_mode="recursive_programming")`：设置并行模式为自动并行，且搜索模式为双递归策略搜索算法。

除了以上context，双递归策略搜索算法无需额外配置。

## 基本原理

双递归策略搜索算法是一种全自动的算子级策略搜索方案，用户无需对模型进行任何配置，算法可以自动搜索出通信代价最小的并行策略。自动算子级策略搜索的核心难点在于，指数级的切分可能带来庞大的搜索空间以及构造代价模型需要的profiling信息的获取代价。

对于第一个问题，双递归策略搜索算法通过对AI训练集群进行抽象，总结出其对称多阶的特点，因此可以等价地进行递归二分，来压缩设备数带来的搜索空间；另一方面，双递归策略搜索算法将算子的通信代价进行分类，比较算子内的通信代价以及算子间的重排布代价，通过对算子的权重进行排序的方法，将指数级的搜索复杂度压缩到线性。

对于第二个问题，双递归策略搜索算法建立符号化的代价模型，传统方法的cost model着眼于如何准确地预估不同策略的绝对时延，而双递归策略搜索算法的代价模型比较的是不同策略的相对代价，因此可以大大节省profiling的成本。

因此双递归策略搜索算法对于巨大网络以及大规模集群切分能够快速生成最优策略。总而言之，双递归策略搜索算法基于并行原理建模，通过建立抽象机来描述硬件集群拓扑，通过符号化简化代价模型。其代价模型比较的不是预估的绝对时延，而是不同并行策略的相对代价，因此能够大大压缩搜索空间，对于百卡集群能够保证分钟级的搜索时间。

## 操作实践

下面以Ascend或者GPU单机8卡为例，进行双递归策略搜索算法操作说明：

### 样例代码说明

> 下载完整的样例代码：[sapp](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/sapp)。

目录结构如下：

```text
└─ sample_code
    ├─ sapp
       ├── train.py
       └── run.sh
    ...
```

其中，`train.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

### 配置分布式环境

通过context接口指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`为自动并行模式，搜索模式`search_mode`为双递归策略，并通过init初始化HCCL或NCCL通信。`device_target`会自动指定为MindSpore包对应的后端硬件设备。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE, save_graphs=2)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="recursive_programming")
init()
ms.set_seed(1)
```

### 数据集加载、定义网络、训练网络

数据集加载、定义网络和训练网络方式与单卡模型一致，代码如下：

```python
import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops

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

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.layer2 = nn.Dense(512, 512)
        self.layer3 = nn.Dense(512, 10)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        logits = self.layer3(x)
        return logits

net = Network()

optimizer = nn.Momentum(net.trainable_params(), 1e-3, 0.1)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        if i % 100 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

### 运行单机八卡脚本

接下来通过命令调用对应的脚本，以`mpirun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，日志文件保存到`log_output`目录下，通过在`train.py`中设置context: `save_graphs=2`，可以打印出编译过程中的IR图，其中部分文件目录结构如下：

```text
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ rank_0
|   ├─ step_parallel_begin_xxxx.ir
|   ├─ xx_validate_xxx.ir
|   ...
├─ rank_1
|   ├─ step_parallel_begin_xxxx.ir
|   ├─ xx_validate_xxx.ir
|   ...
...
...
```

关于Loss部分结果保存在`log_output/1/rank.*/stdout`中，示例如下：

```text
epoch: 0, step: 0, loss is 2.0924754
epoch: 0, step: 100, loss is 1.1485384
epoch: 0, step: 200, loss is 0.5181626
epoch: 0, step: 300, loss is 0.3386282
epoch: 0, step: 400, loss is 0.26848084
epoch: 0, step: 500, loss is 0.21515897
epoch: 0, step: 600, loss is 0.20717612
epoch: 0, step: 700, loss is 0.16945913
epoch: 0, step: 800, loss is 0.1535154
epoch: 0, step: 900, loss is 0.11993752
epoch: 0, step: 1000, loss is 0.13421981
...
```

在`step_parallel_begin_xxxx.ir`中，可以看到每个计算算子都被配置了切分策略：

```text
...
  %2(logits) = Flatten(%1) primitive_attrs: {BatchParallel: Bool(1)} {in_strategy: ((8, 1, 1, 1))}
      : (<Tensor[Float32], (256, 1, 28, 28)>) -> (<Tensor[Float32], (256, 784)>)
      # Scope: (Default)
  %3([CNode]2161) = Load($(@1_train_step.1797:para3_layer1.weight), %para20_u)
      : (<Ref[Tensor[Float32]], (512, 784), ref_key=:layer1.weight>, <UMonad, NoShape>) -> (<Tensor[Float32], (512, 784)>)
      # Scope: (Default)
  %4(logits) = MatMul(%2, %3) {instance name: matmul} primitive_attrs: {output_names: [output], transpose_a: Bool(0), input_names: [x1, x2], transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} {in_strategy: ((4, 2), (1, 2))}
      : (<Tensor[Float32], (256, 784)>, <Tensor[Float32], (512, 784)>) -> (<Tensor[Float32], (256, 512)>)
      # Scope: (Default)
  %5([CNode]2162) = Load($(@1_train_step.1797:para4_layer1.bias), %para20_u)
      : (<Ref[Tensor[Float32]], (512), ref_key=:layer1.bias>, <UMonad, NoShape>) -> (<Tensor[Float32], (512)>)
      # Scope: (Default)
  %6(logits) = BiasAdd(%4, %5) {instance name: bias_add} primitive_attrs: {output_names: [output], format: "NCHW", input_names: [x, b], data_format: "NCHW"} {in_strategy: ((4, 1), (1))}
      : (<Tensor[Float32], (256, 512)>, <Tensor[Float32], (512)>) -> (<Tensor[Float32], (256, 512)>)
      # Scope: (Default)
  %7(logits) = ReLU(%6) {instance name: relu} primitive_attrs: {output_names: [output], input_names: [x]} {in_strategy: ((4, 1))}
      : (<Tensor[Float32], (256, 512)>) -> (<Tensor[Float32], (256, 512)>)
      # Scope: (Default)
...
```

例如对于第一MatMul算子，输入的策略in_strategy已被配置为((4, 2), (1, 2))。

```text
input_names: [x1, x2], transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)
```

代表MatMul算子的第二个输入存在转置。

```text
(<Tensor[Float32], (256, 784)>, <Tensor[Float32], (512, 784)>) -> (<Tensor[Float32], (256, 512)>)
```

代表第一、第二个输入的shape分别为(256, 784)、(512, 784)，第二个输入存在转置，输出的shape为(256, 512)。

在`xx_validate_xxx.ir`中，可以看到各个算子的输入输出张量是已经被切分后的，在网络原有算子之间还插入了一些通信算子，如`AllReduce`：

```text
...
  %14(equiv[CNode]4) = MatMul(%12, %13) {instance name: matmul} primitive_attrs: {output_names: [output], transpose_a: Bool(0), input_names: [x1, x2], transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} cnode_attrs: {related_comm_node_id: "37501"} cnode_primal_attrs: {unique_id: "37896", related_fusion_key: "all_reduce_4-5226697808808137312_1", related_node_id: "34001"} {in_strategy: ((4, 2), (1, 2))}
      : (<Tensor[Float32], (64, 392)>, <Tensor[Float32], (512, 392)>) -> (<Tensor[Float32], (64, 512)>)
      # Scope: (Default)
      # In file /home/liguanxin/anaconda3/envs/py38/lib/python3.8/site-packages/mindspore/nn/layer/basic.py:625/        x = self.matmul(x, self.weight)/
  %15(equiv[CNode]2229) = AllReduce(%14) {instance name: forward_op_15773666391001111732} primitive_attrs: {comm_reuse: Bool(1), group: "2-5004544844489628105", fusion: I64(0), op: "sum", rank_list: (0, 1), group_ranks: "0-1", index: I64(0), group_rank_ids: (0, 1), no_eliminate: Bool(1)} cnode_primal_attrs: {unique_id: "38092", forward_comm_node_unique_id: "37499"}
      : (<Tensor[Float32], (64, 512)>) -> (<Tensor[Float32], (64, 512)>)
      # Scope: (Default)
  %16(equiv[CNode]2162) = Load(%para4_layer1.bias, U) cnode_primal_attrs: {unique_id: "37918"}
      : (<Ref[Tensor[Float32]], (512), ref_key=:layer1.bias>, <UMonad, NoShape>) -> (<Tensor[Float32], (512)>)
      # Scope: (Default)
  %17(equiv[CNode]4) = BiasAdd(%15, %16) {instance name: bias_add} primitive_attrs: {output_names: [output], format: "NCHW", input_names: [x, b], data_format: "NCHW"} cnode_attrs: {related_comm_node_id: "37503"} cnode_primal_attrs: {unique_id: "37916", related_fusion_key: "all_reduce_nccl_world_group_1", related_node_id: "33999"} {in_strategy: ((4, 1), (1))}
      : (<Tensor[Float32], (64, 512)>, <Tensor[Float32], (512)>) -> (<Tensor[Float32], (64, 512)>)
      # Scope: (Default)
      # In file /home/liguanxin/anaconda3/envs/py38/lib/python3.8/site-packages/mindspore/nn/layer/basic.py:627/            x = self.bias_add(x, self.bias)/
  %18(equiv[CNode]4) = ReLU(%17) {instance name: relu} primitive_attrs: {output_names: [output], input_names: [x]} cnode_primal_attrs: {unique_id: "37878"} {in_strategy: ((4, 1))}
      : (<Tensor[Float32], (64, 512)>) -> (<Tensor[Float32], (64, 512)>)
...
```

对于第一个MatMul算子，其两个输入从原来的(256, 784)、(512, 784)被切分为(64, 392)、(512, 392)，第二个输入转置后，算子的输出为(64, 512)。

其他启动方式如动态组网、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/startup_method.html)。
