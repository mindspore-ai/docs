## LeNet描述

LeNet是1998年提出的一种典型的卷积神经网络，由2个卷积层和3个全连接层组成。

[Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791)： Y.Lecun, L.Bottou, Y.Bengio, P.Haffner.*Proceedings of the IEEE*.1998.

## MNIST数据集

使用的数据集：[MNIST](<http://yann.lecun.com/exdb/mnist/>)

- 数据集大小：52.4M，共10个类，6万张 28*28图像
    - 训练集：6万张图像
    - 测试集：5万张图像
- 数据格式：二进制文件

## 教程文件目录

此教程提供了LeNet5网络的定义文件、以及MNIST数据集的加载文件，可以直接导入使用。

```bash
.
├── lenet.py
├── mnist.py
└── README_CN.md
```

## 快速使用教程文件

```python
from lenet import LeNet5
from mnist import create_dataset

# Construct a LeNet5 instance
network = LeNet5()

# Construct a dataloader for MNIST
dataloader = create_dataset()
```