# 数据准备异构加速

`Ascend` `GPU` `CPU` `数据准备`

<!-- TOC -->

- [使用数据准备异构加速](#使用数据准备异构加速)
    - [概述](#概述)
    - [流程](#流程)
    - [如何使用数据准备异构加速](#如何使用数据准备异构加速)
    - [约束条件](#约束条件)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/enable_dataset_offload.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore提供了一种运算负载均衡的技术，可以将MindSpore的算子计算分配到不同的异构硬件上，一方面均衡不同硬件之间的运算开销，另一方面利用异构硬件的优势对算子的运算进行加速。

目前该异构硬件加速技术仅支持将数据算子均衡到网络侧，均衡数据处理管道与网络运算的计算开销。具体来说，目前数据处理管道的算子均在CPU侧运算，该功能将部分数据操作从CPU侧“移动”到网络端，利用昇腾Ascend或GPU的计算资源对数据数据处理的算子进行加速。

为了保证精度，该功能仅支持将数据处理末端的数据增强操作移至异构侧进行加速，数据处理末端指的是数据处理管道最后一个map算子所持有的、位于末端且连续的数据增强操作。

当前支持异构加速功能的数据增强算子有：

|  算子名 |  算子位置 |  算子功能 |
|---------- |--------------|-------------|
|  HWC2CHW |  mindspore.dataset.vision.c_transforms.py | 将图像的维度从(H,W,C) 转换为 (C,H,W) |
|  Normalize |  mindspore.dataset.vision.c_transforms.py |  对图像进行标准化 |
| RandomColorAdjust |  mindspore.dataset.vision.c_transforms.py |  对图像进行随机颜色调整 |
| RandomHorizontalFlip |  mindspore.dataset.vision.c_transforms.py |  对图像进行随机水平翻转  |
| RandomSharpness |   mindspore.dataset.vision.c_transforms.py |  对图像进行随机锐化 |
| RandomVerticalFlip |  mindspore.dataset.vision.c_transforms.py |  对图像进行随机垂直翻转 |
| Rescale |   mindspore.dataset.vision.c_transforms.py |  对图像的像素值进行缩放 |

## 流程

下图显示了给定数据处理管道使用异构加速的典型计算过程。

![offload](../source_zh_cn/images/offload_process.PNG)

异构加速功能对两个API进行了相关更新以允许用户启用此功能：

1. map数据算子新增offload输入参数，

2. 数据集全局配置mindspore.dataset.config中新增set_auto_offload接口。

如需检查数据增强算子是否移动至加速器，用户可以保存并检查计算图IR文件。在异构加速功能被启动后，相关计算算子会被写入IR文件中。异构加速功能同时适用于数据集下沉模式（dataset_sink_mode=True）和数据集非下沉模式（dataset_sink_mode=False）。

## 如何使用数据准备异构加速

MindSpore提供两种方式供用户启用数据准备异构加速功能。

### 方法 1

使用全局配置设置自动异构加速。在这种情况下，所有map数据处理算子的offload参数将设置为True（默认为None）。值得注意的是，如果用户指定特定map操作算子的offload为False，该map算子将直接应用该配置而不是全局配置。

  ```python
  import mindspore.dataset as ds
  ds.config.set_auto_offload(True)
  ```

### 方法 2

在map数据处理算子中将参数offload设置为True（offload默认值为None）。

```python
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C

dataset = ds.ImageFolder(dir)​
image_ops = [C.RandomCropDecodeResize(train_image_size), C.RandomHorizontalFlip(prob=0.5), C.Normalize(mean=mean, std=std), C.HWC2CHW()]​
dataset = dataset.map(operations=type_cast_op, input_columns= "label")
dataset = dataset.map(operations=image_ops , input_columns="image", offload=True)​
```

## 约束条件

异构加速器功能目前仍处于开发阶段。当前的功能使用受到以下条件限制：

1. 该功能仅支持包含 "image" 和 "label" 两个数据列的数据集。

2. 该功能目前不支持经过数据管道算子concat和zip处理后的数据集。

3. 异构加速算子必须是数据处理管道中最后一个或最后多个且连续的数据增强操作，且所对应的map算子必须定义在最后。如

  ```dataset = dataset.map(operations=type_cast_op, input_columns= "label")```

  必须在

  ```dataset = dataset.map(operations=image_ops , input_columns="image", offload=True)```

  之前，即处理"image"列的map算子必须定义在数据处理管道所有map的最后。

4. 该功能目前不支持用户在map数据算子中指定输出列。
