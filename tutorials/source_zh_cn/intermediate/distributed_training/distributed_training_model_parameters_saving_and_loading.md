# 分布式训练模型参数保存和加载

`Ascend` `GPU` `进阶` `分布式并行` `模型导出`

<!-- TOC -->

- [分布式训练模型参数保存和加载](#分布式训练模型参数保存和加载)
    - [自动并行模式](#自动并行模式)
    - [数据并行模式](#数据并行模式)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_zh_cn/intermediate/distributed_training/distributed_training_model_parameters_saving_and_loading.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

本章将会讲解在Ascend与GPU环境中进行分布式训练时，如何进行参数的保存与加载。涵盖的分布式训练模式包括自动并行（Auto Parallel）与数据并行（Data Parallel）。分布式训练进行模型参数的保存之前，需要先按照[Ascend分布式训练](https://www.mindspore.cn/tutorials/zh-CN/r1.5/intermediate/distributed_training/distributed_training_ascend.html)、[GPU分布式训练](https://www.mindspore.cn/tutorials/zh-CN/r1.5/intermediate/distributed_training/distributed_training_gpu.html)教程配置分布式环境变量和集合通信库。

## 自动并行模式

自动并行模式（Auto Parallel）下模型参数的保存和加载与非分布式训练的模型参数保存和加载用法相同，以[Ascend分布式训练](https://www.mindspore.cn/tutorials/zh-CN/r1.5/intermediate/distributed_training/distributed_training_ascend.html)为例，只需在Ascend训练网络步骤中的`test_train_cifar`方法中添加配置`CheckpointConfig`和`ModelCheckpoint`，即可实现模型参数的保存。需要注意的是，并行模式下需要对每张卡上运行的脚本指定不同的checkpoint保存路径，防止读写文件时发生冲突，具体代码如下：

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

def test_train_cifar(epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    loss_cb = LossMonitor()
    data_path = os.getenv('DATA_PATH')
    dataset = create_dataset(data_path)
    batch_size = 32
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    ckpt_config = CheckpointConfig()
    ckpt_callback = ModelCheckpoint(prefix='auto_parallel', directory="./ckpt_" + str(get_rank()) + "/", config=ckpt_config)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb, ckpt_callback], dataset_sink_mode=True)
```

保存好checkpoint文件后，用户可以很容易加载模型参数进行推理或再训练场景，如用于再训练场景可使用如下代码加载模型：

```python
from mindspore import load_checkpoint, load_param_into_net

net = resnet50(batch_size=32, num_classes=10)
# The parameter for load_checkpoint is a .ckpt file which has been successfully saved
param_dict = load_checkpoint('path/to/ckpt_file.ckpt')
load_param_into_net(net, param_dict)
```

checkpoint配置策略和保存方法可以参考[保存及加载模型](https://www.mindspore.cn/tutorials/zh-CN/r1.5/save_load_model.html)。

## 数据并行模式

数据并行模式（Data Parallel）下checkpoint的使用方法和自动并行模式（Auto Parallel）一样，只需要将`test_train_cifar`中

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
```

修改为:

```python
context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
```

> 数据并行场景下加载模型参数时建议每卡加载相同的checkpoint文件，避免造成计算误差，或者可以打开`context.set_context()`中的`parameter_broadcast`开关将0号卡的参数广播到其他卡上。
