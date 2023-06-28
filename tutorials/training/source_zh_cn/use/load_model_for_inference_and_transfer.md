# 加载模型用于推理或迁移学习

`Linux` `Ascend` `GPU` `CPU` `模型加载` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_zh_cn/use/load_model_for_inference_and_transfer.md" target="_blank"><img src="../_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.1/mindspore_load_model_for_inference_and_transfer.ipynb"><img src="../_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9sb2FkX21vZGVsX2Zvcl9pbmZlcmVuY2VfYW5kX3RyYW5zZmVyLmlweW5i&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="../_static/logo_modelarts.png"></a>

## 概述

在模型训练过程中保存在本地的CheckPoint文件，或从[MindSpore Hub](https://www.mindspore.cn/resources/hub/)下载的CheckPoint文件，都可以帮助用户进行推理或迁移学习使用。

以下通过示例来介绍如何通过本地加载或Hub加载模型，用于推理验证和迁移学习。

## 本地加载模型

### 用于推理验证

针对仅推理场景可以使用`load_checkpoint`把参数直接加载到网络中，以便进行后续的推理验证。

示例代码如下：

```python
resnet = ResNet50()
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
dateset_eval = create_dataset(os.path.join(mnist_path, "test"), 32, 1) # define the test dataset
loss = CrossEntropyLoss()
model = Model(resnet, loss, metrics={"accuracy"})
acc = model.eval(dataset_eval)
```

- `load_checkpoint`方法会把参数文件中的网络参数加载到模型中。加载后，网络中的参数就是CheckPoint保存的。
- `eval`方法会验证训练后模型的精度。

### 用于迁移学习

针对任务中断再训练及微调（Fine Tune）场景，可以加载网络参数和优化器参数到模型中。

示例代码如下：

```python
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
resnet = ResNet50()
opt = Momentum()
# load the parameter into net
load_param_into_net(resnet, param_dict)
# load the parameter into optimizer
load_param_into_net(opt, param_dict)
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
model.train(epoch, dataset)
```

- `load_checkpoint`方法会返回一个参数字典。
- `load_param_into_net`会把参数字典中相应的参数加载到网络或优化器中。

## 从Hub加载模型

### 用于推理验证

`mindspore_hub.load` API用于加载预训练模型，可以实现一行代码完成模型的加载。主要的模型加载流程如下：

1. 在[MindSpore Hub官网](https://www.mindspore.cn/resources/hub)上搜索感兴趣的模型。

   例如，想使用GoogleNet对CIFAR-10数据集进行分类，可以在MindSpore Hub官网上使用关键词`GoogleNet`进行搜索。页面将会返回与GoogleNet相关的所有模型。进入相关模型页面之后，获得详情页`url`。

2. 使用`url`完成模型的加载，示例代码如下：

   ```python
   import mindspore_hub as mshub
   import mindspore
   from mindspore import context, Tensor, nn, Model
   from mindspore import dtype as mstype
   import mindspore.dataset.vision.py_transforms as py_transforms

   context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=0)

   model = "mindspore/ascend/0.7/googlenet_v1_cifar10"

   # Initialize the number of classes based on the pre-trained model.
   network = mshub.load(model, num_classes=10)
   network.set_train(False)

   # ...

   ```

3. 完成模型加载后，可以使用MindSpore进行推理，参考[推理模型总览](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.1/multi_platform_inference.html)。

### 用于迁移学习

通过`mindspore_hub.load`完成模型加载后，可以增加一个额外的参数项只加载神经网络的特征提取部分，这样我们就能很容易地在之后增加一些新的层进行迁移学习。*当模型开发者将额外的参数（例如 `include_top`）添加到模型构造中时，可以在模型的详情页中找到这个功能。`include_top`取值为True或者False，表示是否保留顶层的全连接网络。*

下面我们以GoogleNet为例，说明如何加载一个基于ImageNet的预训练模型，并在特定的子任务数据集上进行迁移学习（重训练）。主要的步骤如下：

1. 在[MindSpore Hub官网](https://www.mindspore.cn/resources/hub/)上搜索感兴趣的模型，并从网站上获取特定的`url`。

2. 使用`url`进行MindSpore Hub模型的加载，注意：`include_top`参数需要模型开发者提供，以下代码中的`src.dataset`位于[GoogleNet目录](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/googlenet/src/dataset.py)。

   ```python
   import mindspore
   from mindspore import nn, context, Tensor
   from mindspore import save_checkpoint
   from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
   import mindspore.ops as ops
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
            self.mean = ops.ReduceMean(keep_dims=True)
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
   from mindspore import load_checkpoint, load_param_into_net

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
