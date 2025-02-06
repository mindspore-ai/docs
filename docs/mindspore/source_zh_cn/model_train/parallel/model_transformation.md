# 模型转换

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/source_zh_cn/model_train/parallel/model_transformation.md)

## 概述

### 背景

在使用MindSpore进行分布式训练时，常常需要对训练得到的分布式Checkpoint进行转换（即模型转换），以进行下一步工作，如推理、微调、多阶段训练等。本教程将介绍如何将分布式训练得到的Checkpoint进行转换，以开展分布式策略与集群卡数改变的弹性训练与推理。

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

4. `mindspore.load_segmented_checkpoints(ckpt_file_dir)`：读取指定`ckpt_file_dir`路径下所有`.ckpt`权重文件并合并，返回合并后的参数字典。

5. `mindspore.load_distributed_checkpoint(network, checkpoint_filenames=None, predict_strategy=None, train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM', format='ckpt', unified_safetensors_dir=None, dst_safetensors_dir=None, rank_id=None, output_format='safetensors', name_map=None, max_process_num=64)`：将分布式权重safetensors按照目标策略直接加载到网络中或者存储到本地磁盘上。

## 操作实践

以在Ascend 8卡上训练，并在4卡上微调为例，整体操作流程如下：

1. 执行训练，配置模型参数切分策略文件存储位置，自动生成Checkpoint文件和模型参数切分策略文件。

2. 编译微调网络，配置分布式策略文件存储位置，自动生成模型参数切分策略文件。

3. 用户根据训练与推理涉及到的策略文件对保存的Checkpoint文件进行转换。

4. 编译微调网络后，加载转换得到的分布式Checkpoint文件。

5. 执行微调网络。

需要注意，加载分布式的Checkpoint，要求[对网络进行编译](#对目标网络执行编译)后才可以加载。

### 样例代码说明

> 下载完整的样例代码：[model_saving_loading](https://gitee.com/mindspore/docs/tree/r2.5.0/docs/sample_code/model_saving_loading)。

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
- `run_infer_convert.sh`：推理场景执行模型转换的脚本。
- `run_retrain_convert.sh`：二阶段训练场景执行模型转换的脚本。
- `run_retrain_pipeline_convert.sh`：执行流水线并行模型转换的脚本。
- `run_infer.sh`：执行模型推理的脚本。
- `run_retrain.sh`：执行二阶段训练的脚本。
- `run_pipeline_train.sh`：执行流水线训练的脚本。
- `run_retrain_pipeline.sh`：执行二阶段训练流水线并行模型的脚本。

### 分布式模型保存

首先，按照[模型保存](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_saving.html)教程执行8卡分布式训练，并行模式为`SEMI_AUTO_PARALLEL`或者`AUTO_PARALLEL`，同时通过调用`set_auto_parallel_context`接口自定义`strategy_ckpt_config`参数配置模型切分策略文件存储路径，训练一段时间后，调用存储Checkpoint的`train.ModelCheckpoint`函数，将分布式的Checkpoint存储下来。

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

    示例代码[`model_transformation_retrain.py`](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/sample_code/model_saving_loading/model_transformation_retrain.py)中采用此方法。

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

    示例代码[`model_transformation_infer.py`](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/sample_code/model_saving_loading/model_transformation_infer.py)中采用此方法。

执行完后，将会生成转换后的目标Checkpoint文件目录：

```text
dst_checkpoints/
```

### 多进程加速转换CheckPoint

在编写转换CheckPoint文件的过程中，用户通常会采用下述串行的逻辑，导致转换耗时长达几个小时。例如下述的写法会导致并行度为1，是**不推荐的写法**。

```python
import mindspore as ms
dst_device_num = 8
for tgt_rank in range(dst_device_num):
    rank_list = ms.rank_list_for_transform(tgt_rank, "./src_strategy.ckpt", "./dst_strategy.ckpt")
    checkpoint_files_map = {}
    for rank_id in rank_list:
        checkpoint_file_map[rank_id] = os.path.join(args_opt.src_checkpoints_dir, "rank_{}".format(rank_id), "checkpoint_{}.ckpt".format(rank_id))
    save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
    ms.transform_checkpoint_by_rank(tgt_rank, checkpoint_file_map, save_checkpoint_path, args_opt.src_strategy_file, args_opt.dst_strategy_file)
```

我们可以采用**多进程**的方法进行ckpt转换加速。
具体修改如下，每个进程根据当前的rank_id来决定自己要转换哪个目标rank权重文件。即每个进程转换一个权重文件

> 请根据转换单个权重所消耗的内存和当前host内的总内存，来设置总进程数，否则就引起Host内存不足报错。

```diff
import sys
import mindspore as ms

rank_list = ms.rank_list_for_transform(get_rank(), "./src_strategy.ckpt", "./dst_strategy.ckpt")
checkpoint_files_map = {}
for rank_id in rank_list:
    checkpoint_file_map[rank_id] = os.path.join(args_opt.src_checkpoints_dir, "rank_{}".format(rank_id), "checkpoint_{}.ckpt".format(rank_id))
save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "rank_{}".format(get_rank()), "checkpoint_{}.ckpt".format(get_rank()))
ms.transform_checkpoint_by_rank(get_rank(), checkpoint_file_map, save_checkpoint_path, args_opt.src_strategy_file, args_opt.dst_strategy_file)
```

此外，需要配置下启动方式。将单进程启动改为多进程启动的方式。如下展示了一个4进程转换的方式样例。

```shell
mpirun -n 4 --output-filename log_output --merge-stderr-to-stdout python model_transformation_infer.py --only_compile=1
```

> 本教程中使用到的代码，已经按多进程启动的方式设置。本节的目的是为了强调如何解决用户日常使用场景中转换慢的场景。

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

[流水线并行](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/pipeline_parallel.html) 是对线性的网络进行切分，得到多个子网络，子网络之间在多卡间进行流水，因此每个子图存储下来的切分策略文件是不一致的，所有切分策略汇聚在一起才能得到完整的网络的切分信息。
因此针对流水线并行的维度，相比于其他维度的转换，需要事先执行一次汇聚切分策略文件的操作，得到汇聚后的切分策略文件，以这一份文件作为分布式Checkpoint转换依赖的策略文件。此外，与前面的[执行分布式Checkpoint转换](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_transformation.html#执行分布式checkpoint转换)没有差异。

相关接口：

`mindspore.merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file)`：流水线并行模式下，汇聚所有流水线并行子图的切分策略文件。`src_strategy_dirs`为包含所有流水线并行的子图的切分策略文件的目录，切分策略文件由`mindspore.set_auto_parallel_context(strategy_ckpt_config)`接口存储得到。`dst_strategy_file`为保存汇聚后的切分策略的文件路径。

首先，执行8卡的流水线并行训练，其中pipeline并行维度为2，且开启优化器并行。

训练代码在[pipeline_train.py](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/sample_code/model_saving_loading/pipeline_train.py)中，网络结构在[模型保存](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_saving.html)这一章的基础上增加了流水线并行的配置，并行维度为2。

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

参考[对目标网络执行编译](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_transformation.html#%E5%AF%B9%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C%E6%89%A7%E8%A1%8C%E7%BC%96%E8%AF%91)章节，同样编译目标网络以得到目标网络的切分策略文件。

下一步展开包含pipeline并行维度的分布式Checkpoint维度转换，首先使用接口`merge_pipeline_strategys`对pipline训练得到的切分策略文件进行合并，而后使用接口`transform_checkpoints`或者`transform_checkpoint_by_rank`进行分布式Checkpoint转换。

示例给出使用`transform_checkpoints`的接口，使用`transform_checkpoint_by_rank`接口请参考[执行分布式Checkpoint转换](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_transformation.html#执行分布式checkpoint转换) 章节的介绍。

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

转换完成后，参照[加载转换得到的Checkpoint文件](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_transformation.html#%E5%8A%A0%E8%BD%BD%E8%BD%AC%E6%8D%A2%E5%BE%97%E5%88%B0%E7%9A%84checkpoint%E6%96%87%E4%BB%B6)章节，执行没有pipeline维度的分布式网络。

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

**流水线并行单个子网络模型转换**

`transform_checkpoints`接口同时支持流水线并行下以单个子网络为单位进行模型权重转换。不同于上文提到的流水线模型转换，该场景下无需事先执行一次汇聚切分策略文件的操作，而是将单个流水线子网络的策略文件作为分布式Checkpoint转换依赖的策略文件。采用单个子网络模型转换的方式，每个进程将转换得到该子网络中包含参数在所有目标rank下的权重文件，并以`_part*`后缀标识权重文件由流水线下的哪一个子网络转换得到。

以[流水线并行模型转换](#流水线并行模型转换)第一部分使用的训练网络网络为例，训练脚本及使用方式不再赘述。执行后，将会生成源Checkpoint文件目录以及源切分策略文件，文件目录结构为：

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

参考[对目标网络执行编译](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_transformation.html#%E5%AF%B9%E7%9B%AE%E6%A0%87%E7%BD%91%E7%BB%9C%E6%89%A7%E8%A1%8C%E7%BC%96%E8%AF%91)章节，同样编译目标网络以得到目标网络的切分策略文件。

网络训练并行策略中流水线并行维度为2，网络将被切分为两个子网络进行训练，分别取两个子网络的策略文件`src_strategy_0.ckpt`和`src_strategy_4.ckpt`使用`transform_checkpoints`接口进行单个子网络的权重转换。

```python
import mindspore as ms

stage_strategy_list = ['src_strategy_0.ckpt', 'src_strategy_4.ckpt']
for src_strategy_file in stage_strategy_list:
  ms.transform_checkpoints(args_opt.src_checkpoints_dir, args_opt.dst_checkpoints_dir, "checkpoint_", src_strategy_file, args_opt.dst_strategy_file)
```

> 当前仅支持多卡到多卡之间的权重转换，即不支持`src_strategy_file=None`或`dst_strategy_file=None`的场景，且不支持目标并行策略为流水线并行的转换。

转换后得到权重的目录结构为：

```text
├─ dst_checkpoints_dir
|   ├─ rank_0
|   |   ├─ checkpoint_0_part0.ckpt
|   |   └─ checkpoint_0_part1.ckpt
|   ├─ rank_1
|   |   ├─ checkpoint_1_part0.ckpt
|   |   └─ checkpoint_1_part1.ckpt
|   ...
```

由于以子网络为单位进行权重转换，每个子网络都会转出一个目标rank对应的权重文件。目录`rank_*`下保存对应目标rank下由所有子网络转换得到的权重文件。`checkpoint_0_part0.ckpt`为由子网络0转换得到的目标rank号为0的对应权重。

采用单个子网络模型转换方式得到的权重文件，在`rank_*`路径下所有的权重文件组成了目标策略该rank下完整的权重，加载时需要对路径下所有的文件进行读取。使用`load_segmented_checkpoints`接口能够读取指定路径下所有权重文件并进行汇聚，样例代码如下：

```python
import mindspore as ms

net = Network()

param_dict = ms.load_segmented_checkpoints(checkpoint_file_dir)
param_not_load, _ = ms.load_param_into_net(net, param_dict)
```

### 对接safetensors的权重转换流程

safetensors是 Hugging Face 社区推出的一种新的模型存储格式，相比传统文件格式，具有安全，性能好，通用性较强等特点。

MindSpore在当前版本新增了对safetensors格式的支持，包括权重保存和加载，ckpt与safetensors格式互转，对safetensors格式进行离线切分，离线聚合，在线加载等功能。

如上文流程讲解，在集群规模改变时，都需要对权重进行重新切分转换的操作，对safetensors流程来说，当前主要有两种方式：离线转换、在线转换，具体如下：

#### 离线转换

离线转换流程使用 `transform_checkpoint` 方法，针对safetensors格式，使用流程与前文使用方式一模一样，接口会自动识别传入的权重文件格式为ckpt还是safetensors并自动的进行后续的处理。

在识别到传入权重文件格式为safetensors格式时，会自动触发代码的优化逻辑，转换性能相比ckpt会有极大的性能提升，同时接口会新增参数 `process_num` 和 `output_format`，可以手动控制转换并行数以及输出文件的格式，具体参考[transform_checkpoints](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.transform_checkpoints.html#mindspore.transform_checkpoints)。

```python
# src_checkpoints_dir目录下存储的权重文件需要为safetensors格式
import mindspore as ms
ms.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, "dst_checkpoint",
                      "./src_strategy.ckpt", "./dst_strategy.ckpt", process_num=8, output_format='safetensors')

```

#### 在线转换

1. **权重文件准备**

    在进行在线加载之前，需要先把通过MindSpore训练得到的分布式权重进行聚合，聚合成一份完整的权重（按照权重个数进行自动分片，以多个文件的形式进行存储），聚合过程会抹除原始权重的切分策略信息，仅保留权重的名字和值。

    聚合完成后，后续进行网络加载时，会按照目标集群的切分策略，按需读取完全权重的一部分，将其加载到网络中去。

    具体流程如下：

    1. 调用 [mindspore.unified_safetensors](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.unified_safetensors.html)方法，传入源权重目录，源权重训练策略（需要被 `mindspore.merge_pipeline_stratrge` 接口合并过后的策略），目标权重生成目录。
    2. 在指定目录下会生成多个分片的文件，主要包括三类：
        1. `param_name_map.json`：name映射表，表示每个param与存储文件名的映射关系。
        2. `xxx_partxx.safetensors`：具体权重存储文件。
        3. `hyper_param.safetensors`：超参存储文件。

    注：在线加载流程也需要执行离线合并操作，但该操作只需要执行一次，后续加载推理都可以基于这份权重执行在线加载，该行为与Hugging Face保持一致。

    ```python
    import mindspore as ms
    src_dir = "/usr/safetensors/llama31B/4p_safetensors/"
    src_strategy_file = "/usr/safetensors/llama31B/strategy_4p.ckpt"
    dst_dir = "/usr/safetensors/llama31B/merge_llama31B_4p/"
    ms.unified_safetensors(src_dir, src_strategy_file, dst_dir)
    ```

2. **执行在线加载或离线切分**

    在线加载是通过扩展 [mindspore.load_distributed_checkpoint](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore/mindspore.load_distributed_checkpoint.html) 方法实现的。

    想执行在线加载或者离线切分时，需要配置参数如下：

    1. `network`：权重切分完需要加载到的目标分布式预测网络，该参数配置为 `None` 时，会执行离线切分流程，将切分完的网络以落盘的形式存储到本地磁盘上，否则会将权重直接加载到网络中，相比落盘场景，能节省一次保存和加载的时间。
    2. `predict_strategy`：目标切分策略，配置为 `None` 时则为权重合并场景，会将权重合并为一个完整的权重文件。
    3. `format`：输入权重格式，使用当前流程必须配置为 `safetensors`。
    4. `unified_safetensors_dir`：第一步执行 `mindspore.unified_safetensors` 聚合完的权重。
    5. `dst_safetensors_dir`：离线切分保存模式场景下，safetensors的保存目录。
    6. `rank_id`：卡的逻辑序号。非保存模式下，通过初始化网络全局自动获取；保存模式下，按传入序号保存文件，若未传入，则全量保存。

    ```python
    import mindspore as ms
    mindspore.load_distributed_checkpoint(network=None, predict_strategy="./strategy_8p.ckpt", format='safetensors', unified_safetensors_dir="./merge_llama31B_4p/", dst_safetensors_dir="./dst_dir")
    ```
