# TrainOneStepCell

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_zh_cn/trainonestepcell.md)

`TrainOneStepCell`功能是执行网络的单步训练，返回每次训练结果后的loss结果。

下面构造一个使用`TrainOneStepCell`接口进行网络训练的实例，其中`LeNet5`和包名的导入代码和上个用例共用。

```python
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.nn import Momentum, WithLossCell, TrainOneStepCell
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """
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

if __name__=="__main__":
    data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = LeNet5()
    learning_rate = 0.01
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    for i in range(5):
        train_network.set_train()
        res = train_network(data, label)
        print(f"+++++++++result:{i}++++++++++++")
        print(res)
```

```text
+++++++++result:0++++++++++++
2.302585
+++++++++result:1++++++++++++
2.2935712
+++++++++result:2++++++++++++
2.2764661
+++++++++result:3++++++++++++
2.2521412
+++++++++result:4++++++++++++
2.2214084
```

用例中构造了优化器和一个`WithLossCell`的实例，然后传入`TrainOneStepCell`中初始化一个训练网络，用例循环五次，相当于网络训练了五次，并输出每次的loss结果，由结果可以看出每次训练后loss值在逐渐减小。

后续内容会介绍MindSpore使用更加高级封装的接口，即Model类中的train方法训练模型，在其内部实现中会用到 `TrainOneStepCell`和`WithLossCell`等许多网络组件，感兴趣的读者可以查看其内部实现。
