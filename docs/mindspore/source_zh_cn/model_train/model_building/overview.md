# 模型构建概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/source_zh_cn/model_train/model_building/overview.md)

## 模型构建

在MindSpore框架中，神经网络模型是通过组合神经网络层和Tensor操作构建的，其中mindspore.nn模块提供了丰富的常见神经网络层实现。框架的核心是Cell类，它既是构建所有网络的基础类，也是网络的基本单元。一个神经网络模型被抽象为一个Cell对象，该对象由多个子Cell有序构成，形成了层次化的嵌套结构。这种设计允许用户利用面向对象编程的思想，高效地构建和管理复杂的神经网络架构。

## 定义模型类

用户自定义的神经网络通常继承自 `mindspore.nn.Cell` 类。在继承的子类中，`__init__` 方法用于实例化子Cell（如卷积层、池化层等），并进行相关的状态管理，比如参数初始化。而 `construct` 方法中定义具体的计算逻辑。详细用法请参考[Functional与Cell](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/model_building/functional_and_cell.html)。

MindSpore搭建LeNet5模型，如下所示：

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        '''
        Forward network.
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 神经网络层

MindSpore封装了多种常见的神经网络层。用户可以在[mindspore.nn](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/mindspore.nn.html)中查找需要的神经网络层。例如，在图像处理领域，`nn.Conv2d` 层提供了便捷的卷积操作支持；而 `nn.ReLU` 作为非线性激活层，能够有效增加网络的非线性表达能力。这些预定义的神经网络层，极大地简化了网络构建的复杂性，使得用户能够更加专注于模型的设计与优化。

## 模型参数

神经网络模型的核心在于其内部的神经网络层，这些层如nn.Dense等不仅定义了数据的前向传播路径，还包含了可训练的权重参数和偏置参数。这些参数是模型学习的基石，通过反向传播算法在训练过程中不断进行优化，以最小化损失函数，提升模型性能。

MindSpore提供了便捷的接口来管理这些参数。用户可以通过调用模型实例的 `parameters_dict` 、 `get_parameters` 、`trainable_params` 等方法，获取模型的参数名称及其对应的具体值。详细用法请参考[Tensor与Parameter](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/model_building/tensor_and_parameter.html)。
