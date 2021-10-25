# 使用ResNet-50网络实现图像分类

`Ascend` `GPU` `全流程` `计算机视觉`

<!-- TOC -->

- [使用ResNet-50网络实现图像分类](#使用resnet-50网络实现图像分类)
    - [概述](#概述)
    - [图像分类](#图像分类)
    - [任务描述及准备](#任务描述及准备)
    - [下载CIFAR-10数据集](#下载cifar-10数据集)
    - [数据预加载和预处理](#数据预加载和预处理)
    - [定义卷积神经网络](#定义卷积神经网络)
    - [定义损失函数和优化器](#定义损失函数和优化器)
    - [调用`Model`高阶API进行训练和保存模型文件](#调用model高阶api进行训练和保存模型文件)
    - [加载保存的模型，并进行验证](#加载保存的模型并进行验证)
    - [参考文献](#参考文献)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/cv_resnet50.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/notebook/mindspore_computer_vision_application.ipynb"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_notebook.png"></a>&nbsp;&nbsp;
<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9jb21wdXRlcl92aXNpb25fYXBwbGljYXRpb24uaXB5bmI=&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_modelarts.png"></a>

## 概述

计算机视觉是当前深度学习研究最广泛、落地最成熟的技术领域，在手机拍照、智能安防、自动驾驶等场景有广泛应用。从2012年AlexNet在ImageNet比赛夺冠以来，深度学习深刻推动了计算机视觉领域的发展，当前最先进的计算机视觉算法几乎都是深度学习相关的。深度神经网络可以逐层提取图像特征，并保持局部不变性，被广泛应用于分类、检测、分割、检索、识别、提升、重建等视觉任务中。

本章结合图像分类任务，介绍MindSpore如何应用于计算机视觉场景。

## 图像分类

图像分类是最基础的计算机视觉应用，属于有监督学习类别。给定一张数字图像，判断图像所属的类别，如猫、狗、飞机、汽车等等。用函数来表示这个过程如下：

```python
def classify(image):
   label = model(image)
   return label
```

选择合适的model是关键。这里的model一般指的是深度卷积神经网络，如AlexNet、VGG、GoogLeNet、ResNet等等。

MindSpore实现了典型的卷积神经网络，开发者可以参考[ModelZoo](https://gitee.com/mindspore/models/tree/master/official)。

MindSpore当前支持的图像分类网络包括：典型网络LeNet、AlexNet、ResNet。

## 任务描述及准备

![cifar10](images/cifar10.png)

图1：CIFAR-10数据集[1]

如图1所示，CIFAR-10数据集共包含10类、共60000张图片。其中，每类图片6000张，50000张是训练集，10000张是测试集。每张图片大小为32*32。

图像分类的训练指标通常是精度（Accuracy），即正确预测的样本数占总预测样本数的比值。

接下来我们介绍利用MindSpore解决图片分类任务，整体流程如下：

1. 下载CIFAR-10数据集
2. 数据加载和预处理
3. 定义卷积神经网络，本例采用ResNet-50网络
4. 定义损失函数和优化器
5. 调用`Model`高阶API进行训练和保存模型文件
6. 加载保存的模型进行推理

> 本例面向Ascend 910 AI处理器硬件平台，你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/docs/tree/master/docs/sample_code/resnet>

下面对任务流程中各个环节及代码关键片段进行解释说明。

## 下载CIFAR-10数据集

先从[CIFAR-10数据集官网](https://www.cs.toronto.edu/~kriz/cifar.html)上下载CIFAR-10数据集。本例中采用binary格式的数据，Linux环境可以通过下面的命令下载：

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz --no-check-certificate
```

接下来需要解压数据集，解压命令如下：

```bash
tar -zvxf cifar-10-binary.tar.gz
```

## 数据预加载和预处理

1. 加载数据集

    数据加载可以通过内置数据集格式`Cifar10Dataset`接口完成。
    > `Cifar10Dataset`，读取类型为随机读取，内置CIFAR-10数据集，包含图像和标签，图像格式默认为uint8，标签数据格式默认为uint32。更多说明请查看API中`Cifar10Dataset`接口说明。

    数据加载代码如下，其中`data_home`为数据存储位置：

    ```python
    cifar_ds = ds.Cifar10Dataset(data_home)
    ```

2. 数据增强

    数据增强主要是对数据进行归一化和丰富数据样本数量。常见的数据增强方式包括裁剪、翻转、色彩变化等等。MindSpore通过调用`map`方法在图片上执行增强操作：

    ```python
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = C.RandomHorizontalFlip()
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")
    ```

3. 数据混洗和批处理

    最后通过数据混洗（`shuffle`）随机打乱数据的顺序，并按`batch`读取数据，进行模型训练：

    ```python
    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=args_opt.batch_size, drop_remainder=True)

    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)
    ```

## 定义卷积神经网络

卷积神经网络已经是图像分类任务的标准算法了。卷积神经网络采用分层的结构对图片进行特征提取，由一系列的网络层堆叠而成，比如卷积层、池化层、激活层等等。

ResNet通常是较好的选择。首先，它足够深，常见的有34层，50层，101层。通常层次越深，表征能力越强，分类准确率越高。其次，可学习，采用了残差结构，通过shortcut连接把低层直接跟高层相连，解决了反向传播过程中因为网络太深造成的梯度消失问题。此外，ResNet网络的性能很好，既表现为识别的准确率，也包括它本身模型的大小和参数量。

MindSpore Model Zoo中已经实现了ResNet模型，可以采用[ResNet-50](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/src/resnet.py)。调用方法如下：

```python
network = resnet50(class_num=10)
```

更多ResNet的介绍请参考：[ResNet论文](https://arxiv.org/abs/1512.03385)

## 定义损失函数和优化器

接下来需要定义损失函数（Loss）和优化器（Optimizer）。损失函数是深度学习的训练目标，也叫目标函数，可以理解为神经网络的输出（Logits）和标签(Labels)之间的距离，是一个标量数据。

常见的损失函数包括均方误差、L2损失、Hinge损失、交叉熵等等。图像分类应用通常采用交叉熵损失（`CrossEntropy`）。

优化器用于神经网络求解（训练）。由于神经网络参数规模庞大，无法直接求解，因而深度学习中采用随机梯度下降算法（SGD）及其改进算法进行求解。MindSpore封装了常见的优化器，如`SGD`、`ADAM`、`Momemtum`等等。本例采用`Momentum`优化器，通常需要设定两个参数，动量（`moment`）和权重衰减项（`weight decay`）。

MindSpore中定义损失函数和优化器的代码样例如下：

```python
# loss function definition
ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# optimization definition
opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
```

## 调用`Model`高阶API进行训练和保存模型文件

完成数据预处理、网络定义、损失函数和优化器定义之后，就可以进行模型训练了。模型训练包含两层迭代，数据集的多轮迭代（`epoch`）和一轮数据集内按分组（`batch`）大小进行的单步迭代。其中，单步迭代指的是按分组从数据集中抽取数据，输入到网络中计算得到损失函数，然后通过优化器计算和更新训练参数的梯度。

为了简化训练过程，MindSpore封装了`Model`高阶接口。用户输入网络、损失函数和优化器完成`Model`的初始化，然后调用`train`接口进行训练，`train`接口参数包括迭代次数（`epoch`）和数据集（`dataset`）。

模型保存是对训练参数进行持久化的过程。`Model`类中通过回调函数（`callback`）的方式进行模型保存，如下面代码所示。用户通过`CheckpointConfig`设置回调函数的参数，其中，`save_checkpoint_steps`指每经过固定的单步迭代次数保存一次模型，`keep_checkpoint_max`指最多保存的模型个数。

```python
'''
network, loss, optimizer are defined before.
batch_num, epoch_size are training parameters.
'''
model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

# CheckPoint CallBack definition
config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=35)
ckpoint_cb = ModelCheckpoint(prefix="train_resnet_cifar10", directory="./", config=config_ck)

# LossMonitor is used to print loss value on screen
loss_cb = LossMonitor()
model.train(epoch_size, dataset, callbacks=[ckpoint_cb, loss_cb])
```

## 加载保存的模型，并进行验证

训练得到的模型文件（如`resnet.ckpt`）可以用来预测新图像的类别。首先通过`load_checkpoint`加载模型文件。然后调用`Model`的`eval`接口预测新图像类别。

```python
param_dict = load_checkpoint(args_opt.checkpoint_path)
load_param_into_net(net, param_dict)
eval_dataset = create_dataset(training=False)
res = model.eval(eval_dataset)
print("result: ", res)
```

## 参考文献

[1] <https://www.cs.toronto.edu/~kriz/cifar.html>
