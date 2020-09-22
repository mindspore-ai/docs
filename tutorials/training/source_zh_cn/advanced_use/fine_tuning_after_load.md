# 加载后微调模型

`Linux` `Ascend` `GPU` `模型加载` `模型微调` `初级` `中级` `高级`

<!-- TOC -->

- [加载后微调模型](#加载后微调模型)
    - [概述](#概述)
    - [模型微调](#模型微调)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_zh_cn/advanced_use/fine_tuning_after_load.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

模型微调（Fine-tuning）是指(TODO)。
本教程以GoogleNet为例，对想要使用MindSpore Hub模型进行微调的应用开发者描述了具体操作流程，帮助用户快速实现模型微调。

## 模型微调 

通过`mindspore_hub.load`完成模型加载后，可以增加一个额外的参数项只加载神经网络的特征提取部分，这样我们就能很容易地在之后增加一些新的层进行迁移学习。*当模型开发者将额外的参数（例如 `include_top`）添加到模型构造中时，可以在模型的详情页中找到这个功能。`include_top`取值为True或者False，表示是否保留顶层的全连接网络。* 

下面我们以GoogleNet为例，说明如何加载一个基于ImageNet的预训练模型，并在特定的子任务数据集上进行迁移学习（重训练）。主要的步骤如下：

1. 在[MindSpore Hub官网](https://www.mindspore.cn/resources/hub/)上搜索感兴趣的模型，并从网站上获取特定的`url`。

2. 使用`url`进行MindSpore Hub模型的加载，*注意：`include_top`参数需要模型开发者提供*。

   ```python
   import mindspore
   from mindspore import nn, context, Tensor
   from mindpsore.train.serialization import save_checkpoint
   from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
   from mindspore.ops import operations as P
   from mindspore.nn import Momentum

   import math
   import numpy as np

   import mindspore_hub as mshub
   from src.dataset import create_dataset

   context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        save_graphs=False)
   model_url = "mindspore/ascend/0.7/googlenet_v1_cifar10"
   network = mshub.load(model_url, include_top=False, num_classes=1000)
   network.set_train(False)
   ```

3. 在现有模型结构基础上，增加一个与新任务相关的分类层。

   ```python
   class ReduceMeanFlatten(nn.Cell):
         def __init__(self):
            super(ReduceMeanFlatten, self).__init__()
            self.mean = P.ReduceMean(keep_dims=True)
            self.flatten = nn.Flatten()
         
         def construct(self, x):
            x = self.mean(x, (2, 3))
            x = self.flatten(x)
            return x

   # Check MindSpore Hub website to conclude that the last output shape is 1024.
   last_channel = 1024

   # The number of classes in target task is 26.
   num_classes = 26

   reducemean_flatten = ReduceMeanFlatten()

   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(True)

   train_network = nn.SequentialCell([network, reducemean_flatten, classification_layer])
   ```

4. 为模型训练选择损失函数和优化器。

   ```python
   epoch_size = 60

   # Wrap the backbone network with loss.
   loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
   loss_net = nn.WithLossCell(train_network, loss_fn)

   lr = get_lr(global_step=0,
               lr_init=0,
               lr_max=0.05,
               lr_end=0.001,
               warmup_epochs=5,
               total_epochs=epoch_size)

   # Create an optimizer.
   optim = Momentum(filter(lambda x: x.requires_grad, loss_net.get_parameters()), Tensor(lr), 0.9, 4e-5)
   train_net = nn.TrainOneStepCell(loss_net, optim)
   ```
   
5. 构建数据集，开始重训练。

   如下所示，进行微调任务的数据集为垃圾分类数据集，存储位置为`/ssd/data/garbage/train`。 

   ```python
   dataset = create_dataset("/ssd/data/garbage/train",
                              do_train=True,
                              batch_size=32,
                              platform="Ascend",
                              repeat_num=1)

   for epoch in range(epoch_size):
         for i, items in enumerate(dataset):
            data, label = items
            data = mindspore.Tensor(data)
            label = mindspore.Tensor(label)
            
            loss = train_net(data, label)
            print(f"epoch: {epoch}/{epoch_size}, loss: {loss}")
         # Save the ckpt file for each epoch.
         ckpt_path = f"./ckpt/garbage_finetune_epoch{epoch}.ckpt"
         save_checkpoint(train_network, ckpt_path)
   ```

6. 在测试集上测试模型精度。

   ```python
   from mindspore.train.serialization import load_checkpoint, load_param_into_net

   network = mshub.load('mindspore/ascend/0.7/googlenet_v1_cifar10', pretrained=False,
                        include_top=False, num_classes=1000)

   reducemean_flatten = ReduceMeanFlatten()

   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(False)
   softmax = nn.Softmax()
   network = nn.SequentialCell([network, reducemean_flatten, 
                                 classification_layer, softmax])

   # Load a pre-trained ckpt file.
   ckpt_path = "./ckpt/garbage_finetune_epoch59.ckpt"
   trained_ckpt = load_checkpoint(ckpt_path)
   load_param_into_net(network, trained_ckpt)

   # Define loss and create model.
   model = Model(network, metrics={'acc'}, eval_network=network)

   eval_dataset = create_dataset("/ssd/data/garbage/test",
                              do_train=True,
                              batch_size=32,
                              platform="Ascend",
                              repeat_num=1)

   res = model.eval(eval_dataset)
   print("result:", res, "ckpt=", ckpt_path)
   ```