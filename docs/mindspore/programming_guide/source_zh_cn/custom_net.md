# 构建自定义网络

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_zh_cn/custom_net.md)

无论是网络结构，还是前文提到的模型层、损失函数和优化器等，本质上都是一个Cell，因此都可以自定义实现。

首先构造一个继承`Cell`的子类，然后在`__init__`方法里面定义算子和模型层等，在`construct`方法里面构造网络结构。

以LeNet网络为例，在`__init__`方法中定义了卷积层，池化层和全连接层等结构单元，然后在`construct`方法将定义的内容连接在一起，形成一个完整LeNet的网络结构。

LeNet网络实现方式如下所示：

```python
import mindspore.nn as nn

class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
