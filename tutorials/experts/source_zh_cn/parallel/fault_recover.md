# 分布式故障恢复

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/fault_recover.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

在进行分布式训练时，遇到故障是非常普遍的，类似于单卡训练，可以通过加载训练过程中保存的权重信息继续进行训练。区别于纯数据并行训练，当应用了模型并行后，权重是进行了切分的，卡与卡之间保存的权重信息可能不一致。
为了解决这个问题，一个方案是在保存权重checkpoint文件前，就将权重通过[AllGather](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#allgather) 算子进行汇聚，每张卡均存储一个完整的权重信息，这一个功能在[分布式训练模型参数保存和加载](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html#分布式训练模型参数保存和加载) 中已经介绍了。
但是，对于大模型来说，使用汇聚保存对各种资源的开销都过于巨大，因此，本文档介绍的是每张卡仅仅保存自身的权重信息的恢复方案。对于大模型来说，往往会同时应用上数据并行与模型并行，而数据并行的维度所划分的设备，它们持有的权重信息是完全一致的，这也为大模型提供了冗余的备份，本文档也将指出如何去获取这个冗余信息。
关于并行策略与权重的切片划分的关系，可以进行如下映射。关于数据并行，模型并行的概念，请参考[分布式训练](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html) 、关于优化器并行，请参考[优化器并行](https://www.mindspore.cn/docs/zh-CN/master/design/optimizer_parallel.html) 。

- 数据并行 + 不开启优化器并行：并行通信域内的rank持有相同权重切片。
- 模型并行：并行通信域内的rank持有不同权重切片。
- 数据并行 + 开启优化器并行 + 优化器并行切满所有数据并行维度：并行通信域内的rank持有不同权重切片。
- 数据并行 + 开启优化器并行 + 优化器并行不切满所有数据并行维度：并行通信域内，优化器切分的通信域内的rank持有不同的权重切片，每个优化器切分的通信域之间持有相同的权重切片。

另外，需要注意的是，本文档介绍分布式故障恢复方案，需要在[下沉模式](https://www.mindspore.cn/docs/zh-CN/master/design/on_device.html) 下使用。本文档将以[分布式并行训练Transformer模型](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/transformer.html) 为例介绍该方案，关于transformer的详细了解，请参考该教程。

>你可以在这里下载完整的样例代码：
>
><https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training_transformer>

目录结构如下：

```text
└─sample_code
    ├─distribute_training_transformer
        ├── dataset.py
        ├── model.py
        ├── rank_table_8pcs.json
        ├── run_parallel_save_ckpt.sh
        ├── run_parallel_recover_ckpt.sh
        ├── parallel_save_ckpt_train.py
        └── parallel_recover_train.py
```

## 切片保存权重

保存切片的权重信息，仅仅需要在CheckpointConfig中配置integrated_save为False。同时，配置环境变量GROUP_INFO_FILE存储权重的冗余信息。

```bash
export GROUP_INFO_FILE=./group_info.pb
```

权重存储的代码部分如下，需要注意，训练时通过指定dataset_sink_mode为True以配置为下沉模式。

```python
from mindspore import Model
from mindspore.context import ParallelMode
from mindspore.nn import PipelineCell
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
def train():
    # model create
    # checkpoint save
    ckpt_config = CheckpointConfig(save_ckpt_steps=callback_size, keep_ckpt_max=4,
                                   integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="test", config=ckpt_config)
    callback = [ckpoint_cb]
    model.train(4, dataset, callbacks=callback, dataset_sink_mode=True)
```

## 加载权重继续训练

在上一步保存了权重切片后，在训练得到的目录下，以0卡目录为例，可以看到以下文件。

```text
└─ckpt_dir0
    ├── group_info.pb
    ├── test-1_77.ckpt
    └── train.log0
```

在train.log0中，可以看到当前训练后的loss值，类似如下。

```text
epoch: 1 step: 77, loss is 7.187697
epoch: 1 step: 77, loss is 6.612632
epoch: 1 step: 77, loss is 6.393444
epoch: 1 step: 77, loss is 6.271424
```

读取group_info.pb，可以获取到权重的冗余信息，该文件解析出来后将得到一个列表，该列表中的值为rank_id，表示这些列表中的rank_id对应的权重切片都是相同的，可以相互替换。
如下面的例子，0卡的group_info.pb解析出来后，发现0卡和4卡的权重切分是完全一致的，当0卡的checkpoint丢失时，可以直接复制4卡checkpoint作为0卡的checkpoint，进行恢复。

```python

from mindspore import restore_group_info_list
rank_list = restore_group_info_list("./ckpt_dir0/group_info.pb")
print(rank_list) // [0, 4]
```

分布式的故障恢复，需要事先获取切分的信息，因而，需要先调用[model.build](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Model.html#mindspore.Model.build) 进行编译， 继而再执行训练。

```python
import os
from mindspore.context import ParallelMode
from mindspore import context
def recover_train():
    # model create
    # checkpoint load
    if args_opt.ckpt_file:
        param_dict = load_checkpoint(args_opt.ckpt_file)
        model.build(train_dataset=dataset, epoch=4)
        load_param_into_net(net, param_dict)
    model.train(2, dataset, callbacks=callback, dataset_sink_mode=True)
```

## 运行代码

首先，请参考分布式并行训练Transformer模型教程中的[准备环节](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/transformer.html#准备环节) 准备好数据集。
进入代码目录后，执行保存切片权重的训练脚本。

```bash
bash run_parallel_save_ckpt.sh DATASET_PATH
```

而后，执行故障恢复训练脚本。

```bash
bash run_parallel_recover_ckpt.sh DATASET_PATH
```

恢复训练结束后，查看loss如下，可以看到loss直接从6点多开始下降，说明加载成功了。

```text
epoch: 1 step: 77, loss is 6.465892
epoch: 1 step: 77, loss is 6.239279
```
