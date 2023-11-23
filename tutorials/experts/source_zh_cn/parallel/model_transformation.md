# 模型转换

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/model_transformation.md)

## 概述

### 背景

在使用MindSpore进行分布式训练时，常常需要对训练得到的分布式Checkpoint进行转换，以进行下一步工作，如推理、微调、多阶段训练等。本教程将介绍如何将分布式训练得到的Checkpoint进行转换，以开展分布式策略与集群卡数改变的弹性训练与推理。

> 本功能仅支持半自动并行（SEMI_AUTO_PARALLEL）和自动并行（AUTO_PARALLEL）模式。

### 使用场景

如果您遇到如下场景，需要参考本教程操作，进行弹性训练与推理：

- 场景1：M卡训练，N卡微调训练，M与N可以没有倍数关系。
- 场景2：训练分为多阶段，每个阶段的集群大小不一样。
- 场景3：M卡训练，N卡推理，M与N可以没有倍数关系。
- 场景4：需要对网络的切分策略进行变更。

相关接口：

1. `mindspore.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, src_strategy_file, dst_strategy_file)`：将一个分布式网络的Checkpoint由源切分策略转换到目标切分策略。其中`src_checkpoints_dir`为源Checkpoint文件所在的目录，其子目录要求按照`rank_x/checkpoint_x.ckpt`格式进行存储，x为对应的rank id。`dst_checkpoints_dir`为目标Checkpoint文件存储的目录，`src_strategy_file`为源切分策略文件名，`dst_strategy_file`为目标切分策略文件名。

2. `mindspore.rank_list_for_transform(rank_id, src_strategy_file, dst_strategy_file)`：在对分布式Checkpoint转换的过程中，获取目标rank的Checkpoint文件所需的源Checkpoint文件rank列表。

3. `mindspore.transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name, src_strategy_file, dst_strategy_file)`：将一个分布式网络的Checkpoint由源切分策略转换到目标切分策略，对特定一个rank进行转换。其中`rank_id`为待转换得到的Checkpoint的rank号。`checkpoint_files_map`为源Checkpoint字典，其key为rank号，值为该rank号对应的Checkpoint文件路径。`save_checkpoint_file_name`为当前rank的目标Checkpoint路径以及名字。

## 操作实践

以在Ascend 8卡上训练，并在4卡上微调为例，整体操作流程如下：

1. 执行训练，配置模型参数切分策略文件存储位置，自动生成Checkpoint文件和模型参数切分策略文件。

2. 编译微调网络，配置分布式策略文件存储位置，自动生成模型参数切分策略文件。

3. 用户对依据训练与推理涉及到的策略文件对保存的Checkpoint文件进行转换。

4. 编译微调网络后，加载转换得到的分布式Checkpoint文件。

5. 执行微调网络。

需要注意，加载分布式的Checkpoint，要求[对网络进行编译](#对目标网络执行编译)后才可以加载。

### 样例代码说明

> 下载完整的样例代码：[model_saving_loading](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/model_saving_loading)。

目录结构如下：

```text
└─ sample_code
    ├─ model_saving_loading
       ├── model_transformation_infer.py
       ├── model_transformation_retrain.py
       ├── pipeline_train.py
       ├── pipeline_transformation_retrain.py
       ├── run_infer_convert.sh
       ├── run_infer.sh
       ├── run_retrain_convert.sh
       ├── run_retrain.sh
       ├── run_pipeline_train.sh
       ├── run_retrain_pipeline_convert.sh
       ├── run_retrain_pipeline.sh
       ...
    ...
```

其中每个文件的作用如下：

- `model_transformation_infer.py`：模型转换后进行推理的脚本。
- `model_transformation_retrain.py`：模型转换后进行二阶段训练的脚本。
- `pipeline_transformation_retrain.py`：流水线并行模型转换后二阶段训练的脚本。
- `pipeline_train.py`：流水线并行训练网络的脚本。
- `run_infer_convert.sh`：执行模型转换的脚本。
- `run_retrain_convert.sh`：执行模型转换的脚本。
- `run_retrain_pipeline_convert.sh`：执行流水线并行模型转换的脚本。
- `run_infer.sh`：执行模型推理的脚本。
- `run_retrain.sh`：执行二阶段训练的脚本。
- `run_retrain_pipeline.sh`：执行二阶段训练流水线并行模型的脚本。

### 分布式模型保存

首先，按照[模型保存](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_saving.html)教程执行8卡分布式训练，并行模式为`SEMI_AUTO_PARALLEL`或者`AUTO_PARALLEL`，同时通过调用`set_auto_parallel_context`接口自定义`strategy_ckpt_config`参数配置模型切分策略文件存储路径，训练一段时间后，调用存储Checkpoint的`train.ModelCheckpoint`函数，将分布式的Checkpoint存储下来。

训练结束后，将会在当前路径生成源Checkpoint文件目录以及源切分策略文件：

```text
src_checkpoints/
src_strategy.ckpt
```

> src_checkpoints内的子目录要求按照`rank_x/checkpoint_x.ckpt`格式进行存储。

即src_checkpoints的目录结构改成如下：

```text
src_checkpoints
 ├─ rank_0
 |   └─ checkpoint_0.ckpt
 ├─ rank_1
 |   └─ checkpoint_1.ckpt
 ├─ rank_2
 |   └─ checkpoint_2.ckpt
 ├─ rank_3
 |   └─ checkpoint_3.ckpt
 ├─ rank_4
 |   └─ checkpoint_4.ckpt
 ├─ rank_5
 |   └─ checkpoint_5.ckpt
 ├─ rank_6
 |   └─ checkpoint_6.ckpt
 └─ rank_7
     └─ checkpoint_7.ckpt
...
```

### 生成目标策略文件

然后需要编译新的卡数或者切分策略下的网络，生成目标网络的模型切分策略文件，本示例中，原始策略以8卡进行训练，layer1的`ops.MatMul()`算子并行策略为((2, 1), (1, 2))，不开启优化器并行，策略文件命名为src_strategy.ckpt；目标策略以4卡进行训练，layer1的`ops.MatMul()`算子并行策略为((2, 2), (2, 1))，且开启优化器并行，策略文件命名为dst_strategy.ckpt。

#### 配置分布式环境

通过context接口指定运行模式、并行模式，通过`strategy_ckpt_config`配置需要保存的分布式策略文件路径，开启优化器并行，并通过init初始化HCCL或NCCL通信。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": args_opt.dst_strategy_file})
ms.set_auto_parallel_context(enable_parallel_optimizer=True)
init()
ms.set_seed(1)
```

#### 网络定义以及加载数据集

网络定义修改了原始网络中layer1的`ops.MatMul()`算子并行策略：

```python
import os
from mindspore import nn, ops
import mindspore.dataset as ds
from mindspore.common.initializer import initializer

class Dense(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        self.matmul = ops.MatMul()
        self.add = ops.Add()

    def construct(self, x):
        x = self.matmul(x, self.weight)
        x = self.add(x, self.bias)
        return x

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.layer1 = Dense(28*28, 512)
        self.relu1 = ops.ReLU()
        self.layer2 = Dense(512, 512)
        self.relu2 = ops.ReLU()
        self.layer3 = Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.matmul.shard(((1, 4), (4, 1)))
net.layer3.matmul.shard(((2, 2), (2, 1)))

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

#### 对目标网络执行编译

进行分布式的Checkpoint的转换，依赖于原始的分布式策略文件与目标的分布式策略文件，执行原始的策略下的网络训练时，已经将分布式策略文件存储下来了，因此需要另外获取到目标策略下的分布式策略文件。通过对目标策略的网络执行编译，即可获取到目标策略网络的分布式策略文件。通过`model.infer_train_layout`接口即可以单独对网络执行编译。

```python
import mindspore as ms
from mindspore import nn, ops

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.infer_train_layout(data_set)
```

当目标网络是进行推理时，则将`model.infer_train_layout`更换为`model.infer_predict_layout`以执行编译：

```python
import numpy as np
import mindspore as ms

predict_data = ms.Tensor(np.random.randn(1, 28, 28).astype(np.float32))
model = ms.Model(net)
model.infer_predict_layout(predict_data)
```

编译完成后，即可得到目标切分策略文件`dst_strategy.ckpt`。

### 执行分布式Checkpoint转换

在这一步，需要调用分布式Checkpoint转换的接口进行分布式Checkpoint的转换，分布式Checkpoint提供两个接口对Checkpoint进行转换。

第一个接口`transform_checkpoints`，要求用户将所有的Checkpoint放置于一个目录，并且子目录必须以”rank_0、rank_1、rank_2、…“格式进行命名。用户调用该接口直接对整个目录进行转换。该方式使用较为方便，但是转换需要的内存开销会略高一些。

第二个接口`transform_checkpoint_by_rank`，用以获取到特定的rank的Checkpoint，有更大的灵活性与更低的内存开销，需要配合`rank_list_for_transform`接口使用，以获取本rank的目标Checkpoint需要哪些原始Checkpoint。

1. 使用接口`transform_checkpoints`。

    ```python
    import mindspore as ms

    ms.transform_checkpoints(args_opt.src_checkpoints_dir, args_opt.dst_checkpoints_dir, "checkpoint_", args_opt.src_strategy_file, args_opt.dst_strategy_file)
    ```

    示例代码[`model_transformation_retrain.py`](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_saving_loading/model_transformation_retrain.py)中采用此方法。

2. 调用`transform_checkpoint_by_rank`接口对当前rank对应的原始Checkpoint进行参数合并。

    > 需要保证dst_checkpoints_dir内存在子目录"rank_x"。

    ```python
    import os
    import mindspore as ms
    from mindspore.communication import get_rank

    rank_list = ms.rank_list_for_transform(get_rank(), args_opt.src_strategy_file, args_opt.dst_strategy_file)
    checkpoint_file_map = {}
    for rank_id in rank_list:
        checkpoint_file_map[rank_id] = os.path.join(args_opt.src_checkpoints_dir, "rank_{}".format(rank_id), "checkpoint_{}.ckpt".format(rank_id))
    save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
    ms.transform_checkpoint_by_rank(get_rank(), checkpoint_file_map, save_checkpoint_path, args_opt.src_strategy_file, args_opt.dst_strategy_file)
    ```

    示例代码[`model_transformation_infer.py`](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_saving_loading/model_transformation_infer.py)中采用此方法。

执行完后，将会生成转换后的目标Checkpoint文件目录：

```text
dst_checkpoints/
```

### 加载转换得到的Checkpoint文件

对目标策略的网络进行编译，调用`load_checkpoint`接口，从转换后的Checkpoint文件中加载模型参数数据。

使用`model.infer_train_layout`(训练时)或者`model.infer_predict_layout`(推理时)接口对网络进行编译，此时权重Shape在编译流程进行了切分；调用`load_checkpoint`接口，从Checkpoint文件中加载各卡的模型参数数据。

目标网络是训练场景：

```python
import os
import mindspore as ms
from mindspore import nn, train

save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
loss_cb = train.LossMonitor(20)
model.infer_train_layout(data_set)
param_dict = ms.load_checkpoint(save_checkpoint_path)
ms.load_param_into_net(net, param_dict)
model.train(2, data_set, callbacks=[loss_cb])
```

- `save_checkpoint_path`：需要加载的当前rank对应的Checkpoint模型参数文件名称。

目标网络是推理场景：

```python
import os
import mindspore as ms

save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
param_dict = ms.load_checkpoint(save_checkpoint_path)
model.infer_predict_layout(predict_data)
ms.load_param_into_net(net, param_dict)
predict_result = model.predict(predict_data)
print(predict_result)
```

- `predict_data`：用于推理的Tensor数据。

### 运行单机四卡脚本

接下来通过命令调用对应的脚本，以`mpirun`启动方式，4卡的分布式脚本为例，进行模型转换后的二阶段微调训练：

```bash
bash run_retrain_convert.sh
bash run_retrain.sh
```

或者模型转换后的推理：

```bash
bash run_infer_convert.sh
bash run_infer.sh
```

执行完成后，日志文件保存到`log_output`目录下，目标Checkpoint文件保存在`dst_checkpoints`文件夹下，目标策略文件保存在`dst_strategy.ckpt`，文件目录结构如下：

```text
├─ src_strategy.ckpt
├─ dst_strategy.ckpt
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ dst_checkpoints
|   ├─ rank_0
|   |   └─ checkpoint_0.ckpt
|   ├─ rank_1
|   |   └─ checkpoint_1.ckpt
|   |   ...
|   ...
...
```

二阶段微调训练后的Loss部分结果保存在`log_output/1/rank.*/stdout`中，示例如下：

```text
epoch: 1, step: 20, loss is 0.10617774
epoch: 1, step: 40, loss is 0.06953259
epoch: 1, step: 60, loss is 0.08409108
epoch: 1, step: 80, loss is 0.08699021
epoch: 1, step: 100, loss is 0.07113413
...
```

如果是推理任务，则结果保存在`log_output/1/rank.*/stdout`中，示例如下：

```text
[[ 0.05044775 -0.94413316  0.84689134 -0.2881832   0.66444755  1.0564336
  -0.04191193  0.25590348 -0.690101   -0.6532427 ]]
```

### 流水线并行模型转换

[流水线并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pipeline_parallel.html) 是对线性的网络进行切分，得到多个子网络，子网络之间在多卡间进行流水，因此每个子图存储下来的切分策略文件是不一致的，所有切分策略汇聚在一起才能得到完整的网络的切分信息。
因此针对流水线并行的维度，相比于其它维度的转换，需要事先执行一次汇聚切分策略文件的操作，得到汇聚后的切分策略文件，以这一份文件作为分布式Checkpoint转换依赖的策略文件。此外，与前面的[执行分布式Checkpoint转换](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_transformation.html#执行分布式checkpoint转换)没有差异。

相关接口：

`mindspore.merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file)`：流水线并行模式下，汇聚所有流水线并行子图的切分策略文件。`src_strategy_dirs`为包含所有流水线并行的子图的切分策略文件的目录，切分策略文件由`mindspore.set_auto_parallel_context(strategy_ckpt_config)`接口存储得到。`dst_strategy_file`为保存汇聚后的切分策略的文件路径。

首先，执行8卡的流水线并行训练，其中pipeline并行维度为2，且开启优化器并行。

训练代码在[pipeline_train.py](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_saving_loading/pipeline_train.py)中，网络结构在[模型保存](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_saving.html)这一章的基础上增加了流水线并行的配置，并行维度为2。

核心代码为：

```python
import mindspore as ms
from mindspore import nn, train
from mindspore.communication import init, get_rank
...
ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(pipeline_stages=2, enable_parallel_optimizer=True)
init()
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": "./src_pipeline_strategys/src_strategy_{}.ckpt".format(get_rank())})
ms.set_seed(1)
...
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=1, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./src_checkpoints_pipeline/rank_{}".format(get_rank()),
                                   config=ckpt_config)
net_with_grads = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(3, data_set, callbacks=[loss_cb, ckpoint_cb])
```

> 每张卡得到的切分策略文件是不一致的，因此需要分别保存，按照"src_pipeline_strategys/src_strategy_x.ckpt"格式进行存储。

执行8卡训练脚本执行命令为：

```bash
bash run_pipeline_train.sh
```

执行后，将会生成源Checkpoint文件目录以及源切分策略文件，文件目录结构为：

```text
├─ src_checkpoints_pipeline
|   ├─ rank_0
|   |   ├─ checkpoint-3_1875.ckpt
|   |   └─ checkpoint-graph.meta
|   ├─ rank_1
|   |   ├─ checkpoint-3_1875.ckpt
|   |   ...
|   ...
├─ src_pipeline_strategys
|   ├─ src_strategy_0.ckpt
|   ├─ src_strategy_1.ckpt
|   ├─ src_strategy_2.ckpt
|   ├─ src_strategy_3.ckpt
|   ├─ src_strategy_4.ckpt
|   ├─ src_strategy_5.ckpt
|   ├─ src_strategy_6.ckpt
|   └─ src_strategy_7.ckpt
...
```

参考[对目标网络执行编译](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_transformation.html#%E5%AF%B9%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C%E6%89%A7%E8%A1%8C%E7%BC%96%E8%AF%91)章节，同样编译目标网络以得到目标网络的切分策略文件。

下一步展开包含pipeline并行维度的分布式Checkpoint维度转换，首先使用接口`merge_pipeline_strategys`对pipline训练得到的切分策略文件进行合并，而后使用接口`transform_checkpoints`或者`transform_checkpoint_by_rank`进行分布式Checkpoint转换。

示例给出使用`transform_checkpoints`的接口，使用`transform_checkpoint_by_rank`接口请参考[执行分布式Checkpoint转换](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_transformation.html#执行分布式checkpoint转换) 章节的介绍。

```python
import mindspore as ms

ms.merge_pipeline_strategys(args_opt.src_strategy_dir, args_opt.src_strategy_file)
ms.transform_checkpoints(args_opt.src_checkpoints_dir, args_opt.dst_checkpoints_dir, "checkpoint_", args_opt.src_strategy_file, args_opt.dst_strategy_file)
```

> src_checkpoints_dir内的子目录要求按照"rank_x/checkpoint_x.ckpt"格式进行存储。

示例中，对整个Checkpoint目录进行转换的脚本执行命令为：

```bash
bash run_retrain_pipeline_convert.sh
```

转换完成后，参照[加载转换得到的Checkpoint文件](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_transformation.html#%E5%8A%A0%E8%BD%BD%E8%BD%AC%E6%8D%A2%E5%BE%97%E5%88%B0%E7%9A%84checkpoint%E6%96%87%E4%BB%B6)章节，执行没有pipeline维度的分布式网络。

示例中，加载转换后的Checkpoint进行二阶段微调训练的脚本执行命令为：

```bash
bash run_retrain_pipeline.sh
```

执行完成后，可以看到loss从0.15开始下降：

```text
epoch: 1, step: 20, loss is 0.15090162
epoch: 1, step: 40, loss is 0.13296325
epoch: 1, step: 60, loss is 0.14676111
epoch: 1, step: 80, loss is 0.11930083
epoch: 1, step: 100, loss is 0.0784434
epoch: 1, step: 120, loss is 0.10741685
```
