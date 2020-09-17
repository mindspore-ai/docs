# 使用MindSpore Hub提交、加载和微调模型

`Linux` `Ascend` `GPU` `MindSpore Hub` `模型上传` `模型加载` `模型微调` `初级` `中级` `高级`

<!-- TOC -->

- [使用MindSpore Hub提交、加载和微调模型](#使用mindspore-hub提交加载和微调模型)
    - [概述](#概述)
    - [模型上传](#模型上传)
        - [步骤](#步骤)
    - [模型加载](#模型加载)
    - [模型微调](#模型微调)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced_use/hub_tutorial.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore Hub是MindSpore生态的预训练模型应用工具，作为模型开发者和应用开发者的管道，它不仅向模型开发者提供了方便快捷的模型发布通道，而且向应用开发者提供了简单易用的模型加载和微调API。本教程以GoogleNet为例，对想要将模型发布到MindSpore Hub的模型开发者介绍了模型上传步骤，也对想要使用MindSpore Hub模型进行推理或者微调的应用开发者描述了具体操作流程。总之，本教程可以帮助模型开发者有效地提交模型，并使得应用开发者利用MindSpore Hub的接口快速实现模型推理或微调。

## 模型上传

我们接收用户通过向 [hub](https://gitee.com/mindspore/hub) 仓提交PR的方式向MindSpore Hub发布模型。这里我们以GoogleNet为例，列出模型提交到MindSpore Hub的步骤。

### 步骤

1. 将你的预训练模型托管在可以访问的存储位置。

2. 按照 [模板](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/googlenet/mindspore_hub_conf.py) 在你自己的代码仓中添加模型生成文件 `mindspore_hub_conf.py`，文件放置的位置如下： 

   ```shell script
   googlenet
   ├── src
   │   ├── googlenet.py
   ├── script
   │   ├── run_train.sh
   ├── train.py
   ├── test.py
   ├── mindspore_hub_conf.py
   ```

3. 按照 [模板](https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md) 在 `hub/mshub_res/assets/mindspore/ascend/0.7` 文件夹下创建`{model_name}_{model_version}_{dataset}.md` 文件，其中 `ascend` 为模型运行的硬件平台，`0.7` 为MindSpore的版本号，`hub/mshub_res`的目录结构为：

   ```shell script
   hub
   ├── mshub_res
   │   ├── assets
   │       ├── mindspore
   |           ├── gpu
   |               ├── 0.7
   |           ├── ascend
   |               ├── 0.7 
   |                   ├── googlenet_v1_cifar10.md
   │   ├── tools
   |       ├── md_validator.py
   |       └── md_validator.py 
   ```
   注意，`{model_name}_{model_version}_{dataset}.md` 文件中需要补充如下所示的 `file-format`、`asset-link` 和 `asset-sha256` 信息，它们分别表示模型文件格式、模型存储位置（步骤1所得）和模型哈希值，其中MindSpore Hub支持的模型文件格式有 [MindSpore CKPT](https://www.mindspore.cn/tutorial/zh-CN/master/use/saving_and_loading_model_parameters.html#checkpoint-configuration-policies)，[AIR](https://www.mindspore.cn/tutorial/zh-CN/master/use/multi_platform_inference.html)，[MindIR](https://www.mindspore.cn/tutorial/zh-CN/master/use/saving_and_loading_model_parameters.html#export-mindir-model)，[ONNX](https://www.mindspore.cn/tutorial/zh-CN/master/use/multi_platform_inference.html) 和 [MSLite](https://www.mindspore.cn/lite/tutorial/zh-CN/master/use/converter_tool.html)。

    ```shell script
   file-format: ckpt  
   asset-link: https://download.mindspore.cn/model_zoo/official/cv/googlenet/goolenet_ascend_0.2.0_cifar10_official_classification_20200713/googlenet.ckpt  
   asset-sha256: 114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7
   ```

   对于每个预训练模型，执行以下命令，用来获得`.md` 文件 `asset-sha256` 处所需的哈希值，其中 `googlenet.ckpt` 是从步骤1的存储位置处下载并保存到 `tools` 文件夹的预训练模型，运行后输出的哈希值为 `114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7`。

   ```python
   cd /hub/mshub_res/tools
   python get_sha256.py ../googlenet.ckpt
   ```

4. 使用 `hub/mshub_res/tools/md_validator.py` 在本地核对`.md`文件的格式，执行以下命令，输出结果为 `All Passed`，表示 `.md` 文件的格式和内容均符合要求。

   ```python
   python md_validator.py ../assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md
   ```

5. 在 `mindspore/hub` 仓创建PR，详细创建方式可以参考[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

一旦你的PR合并到 `mindspore/hub` 的master分支，你的模型将于24小时内在 [MindSpore Hub 网站](https://hub.mindspore.com/mindspore) 上显示。有关模型上传的更多详细信息，请参考 [README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md) 。

## 模型加载 

`mindspore_hub.load` API用于加载预训练模型，可以实现一行代码加载模型。主要的模型加载流程如下：

- 在MindSpore Hub官网上搜索感兴趣的模型。

  例如，想使用GoogleNet对CIFAR-10数据集进行分类，可以在MindSpore Hub官网上使用关键词`GoogleNet`进行搜索。页面将会返回与GoogleNet相关的所有模型。进入相关模型页面之后，获得详情页 `url`。

- 使用`url`完成模型的加载，示例代码如下：

  ```python
  import mindspore_hub as mshub
  import mindspore
  from mindspore import context, Tensor, nn
  from mindspore.train.model import Model
  from mindspore.common import dtype as mstype
  from mindspore.dataset.transforms.py_transforms import Compose
  from PIL import Image
  import cv2
  import mindspore.dataset.vision.py_transforms as py_transforms
  
  context.set_context(mode=context.GRAPH_MODE,
                      device_target="Ascend",
                      device_id=0)
  
  model = "mindspore/ascend/0.7/googlenet_v1_cifar10"
  
  # Test an image from CIFAR-10 dataset
  image = Image.open('cifar10/a.jpg')
  transforms = Compose([py_transforms.ToTensor()])
  
  # Initialize the number of classes based on the pre-trained model.
  network = mshub.load(model, num_classes=10)
  network.set_train(False)
  out = network(transforms(image))
  ```

## 模型微调 

在使用 `mindspore_hub.load` 进行模型加载时，可以增加一个额外的参数项只加载神经网络的特征提取部分。这样我们就能很容易地在之后增加一些新的层进行迁移学习。*当模型开发者将额外的参数（例如 include_top）添加到模型构造中时，可以在模型的详情页中找到这个功能。`include_top` 取值为True或者False，表示是否保留顶层的全连接网络。* 

下面我们以GoogleNet为例，说明如何加载一个基于ImageNet的预训练模型，并在特定的子任务数据集上进行迁移学习（重训练）。主要的步骤如下：

1. 在MindSpore Hub的官网上搜索感兴趣的模型，并从网站上获取特定的 `url`。

2. 使用 `url`进行MindSpore Hub模型的加载，*注意：`include_top` 参数需要模型开发者提供*。

   ```python
   import mindspore
   from mindspore import context
   import mindspore_hub as mshub
   
   context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                       save_graphs=False)
   
   network = mshub.load('mindspore/ascend/0.7/googlenet_v1_cifar10', include_top=False)
   network.set_train(False)
   ```

3. 在现有模型结构基础上增加一个与新任务相关的分类层。

   ```python
   from mindspore import nn

   # Check MindSpore Hub website to conclude that the last output shape is 1024.
   last_channel = 1024
   
   # The number of classes in target task is 26.
   num_classes = 26
   classification_layer = nn.Dense(last_channel, num_classes)
   classification_layer.set_train(True)
   
   train_network = nn.SequentialCell([network, classification_layer])
   ```

4. 为模型训练选择损失函数和优化器。

   ```python
   from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
   
   # Wrap the backbone network with loss.
   loss_fn = SoftmaxCrossEntropyWithLogits()
   loss_net = nn.WithLossCell(train_network, loss_fn)
   
   # Create an optimizer.
   optim = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), config.momentum, config.weight_decay)
   train_net = nn.TrainOneStepCell(loss_net, optim)
   ```
   
5. 构建数据集，开始重训练。如下所示，进行微调任务的数据集为垃圾分类数据集，存储位置为 `/ssd/data/garbage/train`。 

   ```python
   from src.dataset import create_dataset
   from mindspore.train.serialization import save_checkpoint
   
   dataset = create_dataset("/ssd/data/garbage/train", do_train=True, batch_size=32)
   
   epoch_size = 15
   for epoch in range(epoch_size):
       for i, items in enumerate(dataset):
           data, label = items
           data = mindspore.Tensor(data)
           label = mindspore.Tensor(label)
           
           loss = train_net(data, label)
           print(f"epoch: {epoch}, loss: {loss}")
       # Save the ckpt file for each epoch.
       ckpt_path = f"./ckpt/garbage_finetune_epoch{epoch}.ckpt"
       save_checkpoint(train_network, ckpt_path)
   ```

6. 在测试集上测试模型精度。

   ```python
   from mindspore.train.serialization import load_checkpoint, load_param_into_net
   
   network = mshub.load('mindspore/ascend/0.7/googlenet_v1_cifar10', include_top=False)
   train_network = nn.SequentialCell([network, nn.Dense(last_channel, num_classes)])
   
   # Load a pre-trained ckpt file.
   ckpt_path = "./ckpt/garbage_finetune_epoch15.ckpt"
   trained_ckpt = load_checkpoint(ckpt_path)
   load_param_into_net(train_network, trained_ckpt)
   
   # Define loss and create model.
   loss_fn = SoftmaxCrossEntropyWithLogits()
   model = Model(network, loss_fn=loss, metrics={'acc'})
   
   eval_dataset = create_dataset("/ssd/data/garbage/train", do_train=False, 
                                 batch_size=32)
   
   res = model.eval(eval_dataset)
   print("result:", res, "ckpt=", ckpt_path)
   ```